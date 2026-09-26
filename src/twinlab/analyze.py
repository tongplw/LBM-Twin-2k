"""Run cohort EDA and development baselines, never score the reserved test labels."""
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import json
import platform

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from .data import ROOT, participants, question_records, target_atoms, make_splits, task_group
from .metrics import agreement_score, bootstrap_mean
from .download import TOKENIZER_FILE, MODEL
from .audit import audit_full_personas


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return {k: float(v) for k, v in zip(["min", "p25", "median", "p75", "p95", "max"],
                                       np.quantile(a, [0, .25, .5, .75, .95, 1]))}


def grouped_scores(rows, field, pids, families, group_by="task_group"):
    """Pool response scores within person/group, never average subgroup means."""
    grouped = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            grouped[(row["pid"], row[group_by])].append(row[field])
    matrix = np.full((len(pids), len(families)), np.nan)
    for i, pid in enumerate(pids):
        for j, fam in enumerate(families):
            if (pid, fam) in grouped: matrix[i, j] = np.mean(grouped[(pid, fam)])
    return matrix


def macro_interval(matrix, draws=2000):
    # Participant rows carry every family together through each bootstrap draw.
    rng = np.random.default_rng(42)
    boot = [np.nanmean(np.nanmean(matrix[rng.integers(0, len(matrix), len(matrix))], axis=0))
            for _ in range(draws)]
    return {"mean": float(np.nanmean(np.nanmean(matrix, axis=0))),
            "lo": float(np.quantile(boot, .025)), "hi": float(np.quantile(boot, .975)),
            "participants": len(matrix), "families": matrix.shape[1]}


def retest_agreement_comparison(rows, pids, groups, draws=2000):
    """Compare scoring rules on identical people, bounded responses and weights."""
    allowed = set(pids)
    if any(row["pid"] not in allowed for row in rows):
        raise ValueError("Retest rows include participants outside the requested set")
    selected = [row for row in rows if row["kind"] != "unbounded_numeric"]
    if any(row.get("exact") is None or row.get("human") is None for row in selected):
        raise ValueError("Exact and partial-credit scores must have identical coverage")
    exact = grouped_scores(selected, "exact", pids, groups)
    partial = grouped_scores(selected, "human", pids, groups)
    if not np.array_equal(np.isfinite(exact), np.isfinite(partial)):
        raise ValueError("Exact and partial-credit scores must have identical coverage")
    if not np.isfinite(partial).all():
        raise ValueError("Missing participant/task cells in retest comparison")

    def complement(interval):
        return {**interval, "mean": 1 - interval["mean"],
                "lo": 1 - interval["hi"], "hi": 1 - interval["lo"]}

    exact_interval = macro_interval(exact, draws)
    partial_interval = macro_interval(partial, draws)
    return {
        "participants": len(pids), "paired_responses": len(selected),
        "excluded_unbounded_responses": sum(row["kind"] == "unbounded_numeric" for row in rows),
        "main_task_groups": len(groups),
        "aggregation": "response mean within person/main group, then people, then equal main groups",
        "exact_agreement": exact_interval,
        "partial_credit_agreement": partial_interval,
        "weighted_exact_disagreement": complement(exact_interval),
        "normalized_disagreement": complement(partial_interval),
        "note": "Exact disagreement uses task weights; normalized disagreement is not a changed-answer rate.",
    }


def retest_by_answer_type(rows, pids, draws=2000):
    """Use the same people and scoring, with equal task weights within each type."""
    result = {}
    for kind in ("binary", "nominal", "ordinal", "bounded_numeric"):
        selected = [r for r in rows if r["kind"] == kind]
        if selected:
            groups = sorted({r["task_group"] for r in selected})
            result[kind] = retest_agreement_comparison(selected, pids, groups, draws)
    return result


def export_retest_comparison(summary):
    """Publish aggregate EDA scores, keeping participant records private."""
    output = ROOT / "results/human_consistency.csv"
    output.parent.mkdir(exist_ok=True)
    comparisons = {"overall_bounded": summary["train_dev_retest_comparison"],
                   **summary["train_dev_retest_by_answer_type"]}
    rows = []
    for kind, score in comparisons.items():
        rows.append({"answer_type": kind,
                     **{k: score[k] for k in ("participants", "paired_responses", "main_task_groups")},
                     **{f"{metric}_{part}": score[metric][part]
                        for metric in ("exact_agreement", "partial_credit_agreement")
                        for part in ("mean", "lo", "hi")}})
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    raw = ROOT / "data/raw"
    out = ROOT / "data/analysis"
    out.mkdir(parents=True, exist_ok=True)
    ids = []
    for path in sorted(raw.glob("wave_persona_chunk_*.parquet")):
        ids += list(map(str, pq.read_table(path, columns=["pid"])["pid"].to_pylist()))
    if len(ids) != len(set(ids)): raise ValueError("Duplicate participant rows")
    splits = make_splits(ids)
    split_path = ROOT / "splits/participants.json"
    split_path.parent.mkdir(exist_ok=True)
    if split_path.exists() and json.loads(split_path.read_text()) != splits:
        raise ValueError("Participant split changed")
    if not split_path.exists():
        split_path.write_text(json.dumps(splits, indent=2) + "\n")
    membership = {pid: split for split, group in splits.items() for pid in group}
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    demographic = defaultdict(Counter)
    demo_by_pid = {}
    lengths, schema_counts, duplicate_examples = [], Counter(), {}
    persona_types, persona_blocks = Counter(), Counter()
    catalog = json.loads((raw / "question_catalog.json").read_text())
    catalog_counts = Counter(q["QuestionID"] for q in catalog)
    catalog_columns = Counter(col for q in catalog for col in q.get("csv_columns", []))
    train_values = defaultdict(list)
    train_price_values = defaultdict(list)
    pair_rows, dev_rows = [], []
    expected_target_atoms = Counter()
    target_types = Counter()
    target_item_metadata = {}
    outcome_pids = []
    audit_references = {}
    safe_keys = set()
    for index, person in enumerate(participants()):
        pid = person["pid"]
        persona = list(question_records(person["wave1_3_persona_json"]))
        counts = Counter(q["QuestionID"] for _, _, _, q in persona)
        schema_counts["persona_question_instances"] += len(persona)
        if max(counts.values()) > 1: schema_counts["participants_with_reused_qids"] += 1
        for bi, qi, block, q in persona:
            persona_types[q["QuestionType"]] += 1
            persona_blocks[block] += 1
            safe_keys.add((bi, qi, q["QuestionID"]))
            if counts[q["QuestionID"]] > 1:
                duplicate_examples.setdefault(q["QuestionID"], set()).add((bi, block, q["QuestionType"]))
            if block == "Demographics":
                ans = q.get("Answers", {}).get("SelectedText", "Missing")
                if isinstance(ans, str): demographic[q["QuestionID"]][ans] += 1
        demos = {q["QuestionID"]: q.get("Answers", {}).get("SelectedText", "Missing")
                 for _, _, block, q in persona if block == "Demographics"}
        demo_by_pid[pid] = demos
        lengths.append({"pid": pid, "characters": len(person["wave1_3_persona_text"]),
                        "tokens": len(tokenizer.encode(person["wave1_3_persona_text"], add_special_tokens=False).ids)})
        # Deliberately do not deserialize outcome fields for reserved test people.
        if membership[pid] == "test": continue
        outcome_pids.append(pid)
        earlier = {a.key: a for a in target_atoms(person["wave4_Q_wave1_3_A"])}
        later = {a.key: a for a in target_atoms(person["wave4_Q_wave4_A"])}
        if earlier.keys() != later.keys():
            raise ValueError(f"Unmatched repeat items for {pid}")
        audit_references[pid] = {key: (earlier[key], late) for key, late in later.items()}
        target_qids = {a.qid for a in later.values()}
        schema_counts["persona_target_qid_overlap"] += len(set(counts) & target_qids)
        expected_target_atoms[len(later)] += 1
        for key, late in later.items():
            early = earlier[key]
            target_types[late.kind] += 1
            target_item_metadata[key] = {k: v for k, v in asdict(late).items()
                                         if k not in {"value", "invalid", "stimulus_hash", "price"}}
            target_item_metadata[key]["task_group"] = task_group(late.family)
            schema_counts["target_atom_pairs"] += 1
            schema_counts["early_invalid"] += early.invalid
            schema_counts["late_invalid"] += late.invalid
            schema_counts["early_missing_or_invalid"] += early.value is None
            schema_counts["late_missing_or_invalid"] += late.value is None
            if early.stimulus_hash != late.stimulus_hash:
                schema_counts["changed_stimulus_pairs"] += 1
                continue
            if membership[pid] == "train" and early.value is not None:
                train_values[key].append(early.value)
                if early.price is not None:
                    train_price_values[key].append((early.price, early.value))
            if early.value is None or late.value is None: continue
            score = agreement_score(late.value, early.value, late.lower, late.upper, late.kind)
            row = {"pid": pid, "split": membership[pid], "key": key, "qid": late.qid,
                   "family": late.family, "task_group": task_group(late.family),
                   "kind": late.kind, "levels": late.levels,
                   "lower": late.lower, "upper": late.upper,
                   "earlier": early.value, "later": late.value, "human": score,
                   "exact": float(early.value == late.value),
                   "absolute_error": abs(early.value-late.value)}
            row["price"] = late.price
            pair_rows.append(row)
            if membership[pid] == "dev": dev_rows.append(row)
        if index % 300 == 0: print(f"Parsed {index + 1}/2058 participants", flush=True)
    # Both baselines use training participants' earlier answers only. The second
    # replaces product marginals with smoothed, product-specific price-bin counts.
    for row in dev_rows:
        values = np.asarray(train_values[row["key"]])
        if not len(values): raise ValueError(f"No baseline support for {row['key']}")
        if row["levels"]:
            k = row["levels"]
            counts = np.bincount(values.astype(int), minlength=k+1)[1:k+1]
            p = (counts + 1) / (counts.sum() + k)
            mode = int(np.argmax(counts)) + 1
            point = mode if row["kind"] == "binary" else float(np.median(values))
            row["baseline_nll"] = float(-np.log(p[int(row["later"])-1]))
            row["uniform_nll"] = float(np.log(k))
            onehot = np.eye(k)[int(row["later"])-1]
            row["baseline_brier"] = float(np.sum((p-onehot)**2))
            row["baseline_exact"] = float(mode == row["later"])
            row["uniform"] = float(np.mean([
                agreement_score(row["later"], j, row["lower"], row["upper"], row["kind"])
                for j in range(1, k+1)]))
        else:
            point = float(np.median(values))
            row["baseline_nll"] = row["uniform_nll"] = row["baseline_brier"] = None
            row["baseline_exact"] = None
            if row["lower"] is not None:
                lo, hi, y = row["lower"], row["upper"], row["later"]
                row["uniform"] = 1 - ((y-lo)**2+(hi-y)**2)/(2*(hi-lo)**2)
            else: row["uniform"] = None
        row["baseline"] = agreement_score(row["later"], point, row["lower"], row["upper"], row["kind"])
        row["baseline_absolute_error"] = abs(point-row["later"])
        row["price_aware"] = row["baseline"]
        row["price_aware_nll"] = row["baseline_nll"]
        if row["price"] is not None:
            price_data = np.asarray(train_price_values[row["key"]])
            edges = np.unique(np.quantile(price_data[:,0], [.2, .4, .6, .8]))
            bin_id = int(np.searchsorted(edges, row["price"], side="right"))
            in_bin = np.searchsorted(edges, price_data[:,0], side="right") == bin_id
            local = np.bincount(price_data[in_bin,1].astype(int), minlength=3)[1:3]
            global_counts = np.bincount(price_data[:,1].astype(int), minlength=3)[1:3]
            prior = (global_counts + 1)/(global_counts.sum()+2)
            probs = (local + 5*prior)/(local.sum()+5)
            row["price_aware"] = float(np.argmax(probs)+1 == row["later"])
            row["price_aware_nll"] = float(-np.log(probs[int(row["later"])-1]))
            row["training_price_bin"] = bin_id
            row["predicted_purchase_probability"] = float(probs[0])
    families = sorted({r["family"] for r in pair_rows})
    groups = sorted({r["task_group"] for r in pair_rows})
    if len(groups) != 17 or len(families) != 19:
        raise ValueError("Expected 17 main groups and 19 diagnostic families")
    development_ids = splits["dev"]
    matrices = {name: grouped_scores(dev_rows, name, development_ids, groups)
                for name in ("human", "baseline", "price_aware", "uniform", "baseline_nll", "price_aware_nll", "uniform_nll")}
    diagnostic_matrices = {name: grouped_scores(dev_rows, name, development_ids, families, "family")
                           for name in ("human", "baseline", "price_aware", "uniform")}
    family_results = []
    for j, fam in enumerate(families):
        selected = [r for r in dev_rows if r["family"] == fam]
        item_groups = defaultdict(list)
        for r in selected: item_groups[r["key"]].append((r["earlier"], r["later"]))
        correlations = []
        for values in item_groups.values():
            a = np.asarray(values)
            if len(a) > 2 and a[:, 0].std() > 0 and a[:, 1].std() > 0:
                correlations.append(float(np.corrcoef(a.T)[0, 1]))
        family_results.append({"family": fam, "task_group": task_group(fam), "matched_pairs": len(selected),
            "participants": len({r["pid"] for r in selected}),
            "scored_bounded_pairs": sum(r["human"] is not None for r in selected),
            **{name: bootstrap_mean(diagnostic_matrices[name][:, j]) for name in ("human", "baseline", "price_aware", "uniform")},
            "mean_item_correlation": float(np.mean(correlations)) if correlations else None})
    group_results = []
    for j, group in enumerate(groups):
        selected = [r for r in dev_rows if r["task_group"] == group]
        group_results.append({"task_group": group,
            "families": sorted({r["family"] for r in selected}),
            "matched_pairs": len(selected),
            "participants": len({r["pid"] for r in selected}),
            "scored_bounded_pairs": sum(r["human"] is not None for r in selected),
            **{name: bootstrap_mean(matrices[name][:, j])
               for name in ("human", "baseline", "price_aware", "uniform")}})
    cohort_human = grouped_scores(pair_rows, "human", outcome_pids, groups)
    def full_persona_rows():
        for path in sorted(raw.glob("persona_chunk_*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["pid", "persona_json"]):
                yield from batch.to_pylist()

    full_audit = audit_full_personas(full_persona_rows(), audit_references, membership)
    if full_audit["missing_participants"]:
        raise ValueError("Incomplete full-persona coverage; run twinlab.download first")
    summaries = {
        "scope": {"cohort": len(ids), "outcome_exploration": len(outcome_pids),
                  "development_baselines": len(development_ids), "reserved_test": len(splits["test"]),
                  "test_outcome_fields_deserialized": False, "training_run": False},
        "grouping": {"main_task_groups": 17, "diagnostic_families": 19,
                     "family_to_task_group": {fam: task_group(fam) for fam in families},
                     "aggregation": "response mean within person/main group, then people, then 17 equal groups",
                     "unbounded_numeric_in_composite": False,
                     "paper_score_replication": False},
        "schema": dict(schema_counts), "target_atoms_per_person": dict(expected_target_atoms),
        "target_types": dict(target_types), "persona_question_types": dict(persona_types),
        "catalog": {"entries": len(catalog), "unique_qids": len(catalog_counts),
                    "reused_qids": {k: v for k, v in catalog_counts.items() if v>1},
                    "colliding_csv_columns": {k: v for k,v in catalog_columns.items() if v>1}},
        "persona_reused_qid_examples": {k: sorted(v) for k,v in duplicate_examples.items()},
        "context_tokenizer": MODEL,
        "context_tokens": quantiles([v["tokens"] for v in lengths]),
        "context_characters": quantiles([v["characters"] for v in lengths]),
        "context_over_8192_fraction": float(np.mean([v["tokens"] > 8192 for v in lengths])),
        "demographics": {qid: dict(count) for qid, count in demographic.items()},
        "full_persona_audit": full_audit,
        "human_reference_exploratory": macro_interval(cohort_human),
        "train_dev_retest_comparison": retest_agreement_comparison(pair_rows, outcome_pids, groups),
        "train_dev_retest_by_answer_type": retest_by_answer_type(pair_rows, outcome_pids),
        "development_scores": {name: macro_interval(matrices[name]) for name in ("human", "baseline", "price_aware", "uniform")},
        "development_retest_comparison": retest_agreement_comparison(dev_rows, development_ids, groups),
        "human_minus_baseline": macro_interval(matrices["human"] - matrices["baseline"]),
        "human_minus_price_aware": macro_interval(matrices["human"] - matrices["price_aware"]),
        "development_nll": {name: macro_interval(matrices[name][:, np.isfinite(matrices[name]).any(axis=0)])
                            for name in ("baseline_nll", "price_aware_nll", "uniform_nll")},
        "families": family_results,
        "task_groups": group_results,
        "native_numeric_errors": {},
        "subgroup_retest": {},
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
    }
    for fam in ("Anchoring: countries", "Anchoring: height"):
        selected = [r for r in dev_rows if r["family"] == fam and r["kind"] == "unbounded_numeric"]
        summaries["native_numeric_errors"][fam] = {
            "n": len(selected), "retest_absolute_error": quantiles([r["absolute_error"] for r in selected]),
            "median_baseline_absolute_error": quantiles([r["baseline_absolute_error"] for r in selected]),
            "retest_mae": float(np.mean([r["absolute_error"] for r in selected])),
            "baseline_mae": float(np.mean([r["baseline_absolute_error"] for r in selected]))}
    person_means = np.nanmean(matrices["human"], axis=1)
    for qid in ("QID13", "QID12"):
        summaries["subgroup_retest"][qid] = {
            group: bootstrap_mean([person_means[i] for i, pid in enumerate(development_ids)
                                  if demo_by_pid[pid].get(qid) == group])
            for group in demographic[qid]}
    # Transition matrices for two fixed, all-participant questions, dev only.
    transitions = {}
    for qid in ("QID196", "QID291"):
        chosen = [r for r in dev_rows if r["qid"] == qid]
        k = chosen[0]["levels"]
        mat = np.zeros((k,k), dtype=int)
        for row in chosen: mat[int(row["earlier"])-1, int(row["later"])-1] += 1
        transitions[qid] = mat.tolist()
    summaries["transitions"] = transitions
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    export_retest_comparison(summaries)
    (out / "item_metadata.json").write_text(json.dumps(target_item_metadata, indent=2) + "\n")
    # Private intermediate values support exact reruns and figure construction.
    private = ROOT / "data/processed"
    private.mkdir(exist_ok=True)
    (private / "context_lengths.json").write_text(json.dumps(lengths))
    (private / "development_pairs.json").write_text(json.dumps(dev_rows))
    print(json.dumps({"scope": summaries["scope"], "development": summaries["development_scores"],
                      "tokens": summaries["context_tokens"], "schema": summaries["schema"],
                      "full_persona_audit": full_audit,
                      "train_dev_retest_comparison": summaries["train_dev_retest_comparison"]}, indent=2))


if __name__ == "__main__": main()
