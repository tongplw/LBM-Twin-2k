"""Independently check saved development scores and actual training coverage.

Reads complete prediction lines from a snapshot without modifying an active run.
Never selects a checkpoint or fits a model. Reserved-test checks require an
explicit flag and a completed, sealed evaluation.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .benchmark_data import balanced_schedule, digest, make_examples
from .data import ROOT, participants, question_records, stimulus


def snapshot_records(path):
    raw = Path(path).read_bytes()
    return [json.loads(line) for line in raw[:raw.rfind(b"\n")+1].splitlines()]


def check_score(atom, row):
    """Recompute directly from source bounds, independently of the scorer."""
    assert row["truth"] == atom["value"] and row["kind"] == atom["kind"]
    value = row["prediction"]
    valid = value is not None and math.isfinite(value)
    if valid and atom["lower"] is not None:
        valid = atom["lower"] <= value <= atom["upper"]
    if valid and atom["levels"]:
        valid = float(value).is_integer()
    expected = {"invalid": float(not valid), "exact": None, "agreement": None,
                "nll": None, "brier": None, "absolute_error": None}
    if atom["kind"] != "unbounded_numeric":
        expected["exact"] = float(valid and value == atom["value"])
        if not valid:
            expected["agreement"] = 0.
        elif atom["kind"] in {"binary", "nominal"}:
            expected["agreement"] = expected["exact"]
        else:
            expected["agreement"] = 1-abs(value-atom["value"])/(atom["upper"]-atom["lower"])
    if valid and not atom["levels"]:
        expected["absolute_error"] = abs(value-atom["value"])
    if atom["levels"]:
        p = row["probabilities"]
        assert len(p) == atom["levels"] and all(math.isfinite(x) and x >= 0 for x in p)
        assert math.isclose(sum(p), 1, abs_tol=1e-6)
        y = int(atom["value"])-1
        assert value == int(np.argmax(p))+1
        expected["nll"] = -math.log(max(p[y], 1e-12))
        expected["brier"] = sum((x-(i == y))**2 for i, x in enumerate(p))
    for name, value in expected.items():
        assert (value is None and row[name] is None) or (
            value is not None and row[name] is not None and
            math.isclose(value, row[name], rel_tol=1e-10, abs_tol=1e-10)), (atom["key"], name)


def check_history_targets(person, examples):
    """Check IDs and exact target content even if copied under a different ID."""
    def content(question):
        return digest({k: v for k, v in question.items() if k != "QuestionID"})
    targets = {content(e["question"]) for e in examples}
    ids = {e["question"]["QuestionID"] for e in examples}
    for _, _, _, q in question_records(person["wave1_3_persona_json"]):
        if q["QuestionID"] in ids or content(stimulus(q)) in targets:
            raise ValueError("A target ID or duplicate target question occurs in safe history")


def audit(root, include_test=False):
    root = Path(root)
    if include_test:
        run = json.loads((root / "report/experiment.json").read_text())
        if run["status"] != "complete" or not run["final_test_opened"]:
            raise ValueError("Test audit requires a completed reserved-test evaluation")
        sealed = json.loads((root / "selection.json").read_text())
        if sealed != run["selection"]:
            raise ValueError("Final selection differs from the sealed development decision")
    manifest = json.loads((root / "manifest.json").read_text())
    spec = manifest["spec"]
    if spec["scope"] != "full" or spec.get("train_people") is not None or spec.get("dev_people") is not None:
        raise ValueError("This audit requires the full training/development cohorts")
    splits = json.loads((ROOT / "splits/participants.json").read_text())
    audited_splits = ("train", "dev", "test") if include_test else ("train", "dev")
    membership = {pid: split for split in audited_splits for pid in splits[split]}
    examples = {split: [] for split in audited_splits}
    for person in participants():
        split = membership.get(person["pid"])
        if split:
            rows = make_examples(person, split, manifest["spec"]["scope"])
            check_history_targets(person, rows)
            examples[split].extend(rows)
    for split in examples:
        order = {pid: i for i, pid in enumerate(splits[split])}
        examples[split].sort(key=lambda e: (order[e["pid"]], e["id"]))
    training = examples["train"]
    accum = manifest["plan"]["training"]["gradient_accumulation"]
    schedule = balanced_schedule(training, manifest["spec"]["max_updates"]*accum,
                                 manifest["plan"]["training"]["seed"])
    atoms = {(e["pid"], a["key"]): a for split in audited_splits if split != "train"
             for e in examples[split] for a in e["atoms"]}
    result = {"audited_at": datetime.now(timezone.utc).isoformat(), "test_outcomes_opened": include_test,
              "training": {}, "development_predictions": {}, "test_predictions": {},
              "safe_history_participants_checked": len(membership)}
    for path in sorted(root.glob("sft_*/training.json")):
        report = json.loads(path.read_text())
        assert report["schedule_sha256"] == digest(schedule)
        consumed = schedule[:report["updates"]*accum]
        curve = report["loss_curve"]
        assert len(curve) == report["updates"]
        assert all(math.isfinite(p[f]) for p in curve for f in ("answer_loss", "gradient_norm"))
        result["training"][path.parent.name] = {
            "updates": report["updates"], "processed_draws": len(consumed),
            "unique_questions_processed": len(set(consumed)),
            "participants_processed": len({training[i]["pid"] for i in consumed}),
            "response_rows_processed": sum(len(training[i]["atoms"]) for i in consumed),
            "processed_task_counts": dict(Counter(training[i]["group"] for i in consumed)),
            "available_questions": len(training), "planned_draws": len(schedule),
            "planned_fraction_of_full_pass": len(schedule)/len(training),
            "planned_response_rows": sum(len(training[i]["atoms"]) for i in schedule),
            "planned_participants": len({training[i]["pid"] for i in schedule}),
            "schedule_hash_verified": True, "finite_losses_and_gradients": True}
    paths = list(root.glob("sft_*/update-*-dev/predictions.jsonl"))
    paths += list(root.glob("frozen_*/predictions.jsonl"))
    paths += list(root.glob("sft_shuffled_*/predictions.jsonl"))
    paths += list(root.glob("sft_hybrid/predictions.jsonl"))
    if include_test:
        paths.append(root / "final-test/predictions.jsonl")
    for path in sorted(paths):
        split = "test" if path.parent.name == "final-test" else "dev"
        expected_keys = [(e["pid"], e["id"]) for e in examples[split]]
        records = snapshot_records(path)
        assert [(r["pid"], r["id"]) for r in records] == expected_keys[:len(records)]
        for record, example in zip(records, examples[split]):
            assert [r["key"] for r in record["scored"]] == [a["key"] for a in example["atoms"]]
            assert len(record["values"]) == len(example["atoms"])
            for value, p, row in zip(record["values"], record["probabilities"], record["scored"]):
                assert row["invalid"] or value == row["prediction"]
                assert p == row["probabilities"]
        rows = [r for record in records for r in record["scored"]]
        kinds = defaultdict(Counter)
        pooled = defaultdict(list)
        for row in rows:
            check_score(atoms[row["pid"], row["key"]], row)
            counts = kinds[row["kind"]]
            counts["responses"] += 1
            counts["invalid"] += int(row["invalid"])
            counts["partial_differs_from_exact"] += row["agreement"] != row["exact"]
            for field in ("exact", "agreement", "nll", "brier"):
                if row[field] is not None:
                    pooled[field, row["task_group"], row["pid"]].append(row[field])
        means = defaultdict(lambda: defaultdict(list))
        for (field, group, pid), values in pooled.items():
            means[field][group].append(sum(values)/len(values))
        macro = {f: sum(sum(v)/len(v) for v in groups.values())/len(groups)
                 for f, groups in means.items()}
        complete = len(records) == len(expected_keys)
        summary_path = path.parent / "summary.json"
        if complete and summary_path.exists():
            summary = json.loads(summary_path.read_text())
            for field, value in macro.items():
                assert math.isclose(value, summary[field]["mean"], abs_tol=1e-10)
        result["test_predictions" if split == "test" else "development_predictions"][str(path.parent.relative_to(root))] = {
            "questions": len(records), "expected_questions": len(expected_keys),
            "complete": complete, "source_scores_verified": len(rows),
            "by_response_type": dict(kinds),
            "task_macro_snapshot_not_final": macro,
            "estimated_full_inference_hours": sum(r["inference_seconds"] for r in records)
                /max(1, len(records))*len(expected_keys)/3600}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--include-test", action="store_true", help="Audit test predictions only after sealed evaluation completes")
    args = parser.parse_args()
    result = audit(args.run_directory, args.include_test)
    output = args.run_directory / "report/audit.json"
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
