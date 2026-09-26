"""Run the fixed-budget study locally; resume with the identical command.

Example: python -m twinlab.run_experiment --name full-study --scope full
Small pilot runs always leave the final test unopened.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from .data import ROOT
from .benchmark_data import load_cohort, messages_for, answer_parts
from .benchmark_metrics import (Baselines, score_prediction, summarize, paired_difference,
                                average_assignments, fit_temperature, temperature_scale)
from .experiment_engine import Engine, atomic_json, read_predictions
from .retrieval import prepare_rankings
from .selection import AGREEMENT_RULE, NLL_RULE, selection_metric, select_best, selection_description


def now():
    return datetime.now(timezone.utc).isoformat()


def saved_rows(destination):
    return [r for record in read_predictions(Path(destination)/"predictions.jsonl") for r in record["scored"]]


def select_saved_checkpoints(plan, runtime_profile, directory, people, examples, report,
                             max_updates, eval_every, on_progress):
    """Evaluate saved candidates after training, retaining the same selection rule."""
    required = sorted(set(range(eval_every, max_updates+1, eval_every)) | {max_updates})
    previous_best = report.get("selected_update")
    completed = {c["update"]: c for c in report["checkpoints"]}
    for update in required:
        if update in completed:
            continue
        adapter = directory / f"update-{update}-adapter"
        if not adapter.exists():
            raise ValueError(f"Missing saved checkpoint for development selection: {adapter}")
        report["status"] = "evaluating_checkpoints"
        on_progress(report, update)
        engine = Engine(plan, adapter, runtime_profile=runtime_profile)
        _, summary = engine.evaluate(people, examples, report["condition"], directory / f"update-{update}-dev")
        engine.close()
        completed[update] = {"update": update, "development": summary}
        report["checkpoints"] = [completed[u] for u in sorted(completed)]
        atomic_json(directory / "training.json", report)
        on_progress(report, update)
    metric = selection_metric(plan, checkpoint=True)
    chosen = select_best({u: completed[u]["development"] for u in required}, metric)
    adapter = directory / f"update-{chosen}-adapter"
    if adapter.exists():
        shutil.copytree(adapter, directory / "best_adapter", dirs_exist_ok=True)
    elif chosen != previous_best or not (directory / "best_adapter").exists():
        raise ValueError("Selected adapter is missing")
    report.update(status="complete", selected_update=chosen, selected_by=selection_description(metric))
    atomic_json(directory / "training.json", report)
    on_progress(report, chosen)
    return report


def run(args):
    from transformers import AutoConfig, AutoTokenizer
    root = ROOT / "checkpoints" / args.name
    root.mkdir(parents=True, exist_ok=True)
    public = root / "report"
    public.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    runtime_path = root / "runtime_profile.json"
    if args.runtime_profile:
        runtime_profile = json.loads(args.runtime_profile.read_text())
        if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime_profile:
            raise ValueError("Run already has a different runtime profile")
        atomic_json(runtime_path, runtime_profile)
    runtime_profile = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    sessions_path = root / "execution_sessions.json"
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
    sessions.append({"started": now(), "runtime_profile": runtime_profile,
                     "defer_checkpoint_evaluation": args.defer_checkpoint_evaluation,
                     "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((ROOT / "src/twinlab").glob("*.py"))}})
    atomic_json(sessions_path, sessions)
    spec = {"scope": args.scope, "train_people": args.train_people, "dev_people": args.dev_people,
            "max_updates": args.max_updates, "eval_every": args.eval_every, "seeds": [42, 43, 44]}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["spec"] != spec:
            raise ValueError("Run name already has different settings")
        plan = manifest["plan"]
        requested_metric = getattr(args, 'selection_metric', None)
        if requested_metric and any(selection_metric(plan, checkpoint=c) != requested_metric for c in (False, True)):
            raise ValueError('Requested selection metric differs from the saved run')
    else:
        plan = json.loads((ROOT / "configs/model_plan.json").read_text())
        requested_metric = getattr(args, 'selection_metric', None)
        if requested_metric:
            plan['selection_metric'] = AGREEMENT_RULE if requested_metric == 'agreement' else NLL_RULE
            plan['execution_budget']['checkpoint_selection'] = plan['selection_metric']
        plan["model_revision"] = AutoConfig.from_pretrained(plan["model"], revision=plan.get("model_revision"))._commit_hash
        plan["embedding_revision"] = AutoConfig.from_pretrained(plan["retrieval"]["embedding_model"], revision=plan.get("embedding_revision"))._commit_hash
        sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((ROOT / "src/twinlab").glob("*.py"))}
        manifest = {"created": now(), "spec": spec, "plan": plan, "source_sha256": sources,
                    "split_sha256": hashlib.sha256((ROOT / "splits/participants.json").read_bytes()).hexdigest(),
                    "dataset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted((ROOT / "data/raw").glob("wave_persona_chunk_*.parquet"))}}
        atomic_json(manifest_path, manifest)
    status = {"status": "running", "started": manifest["created"], "updated": now(), "spec": spec,
              "final_test_opened": False, "stages": {}, "comparisons": {}, "manifest": manifest}
    status_path = public / "experiment.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status["status"] in {"complete", "pilot_complete_final_test_unopened"}:
            if status["status"] == "complete":
                from .finalize_submission import finalize
                finalize(root)
            print(f"Run already complete: {public / 'REPORT.md'}", flush=True)
            return
        status["status"] = "running"
        status.pop("error", None)

    def publish(stage=None, summary=None):
        if stage:
            status["stages"][stage] = summary
        status["updated"] = now()
        status["runtime_profile"] = runtime_profile
        atomic_json(status_path, status)
        from .report_experiment import write_report
        write_report(public)

    def evaluate(engine, stage, people, examples, mode, seed=42, directory=None):
        status["active_stage"] = stage
        publish()
        directory = directory or root / stage
        rows, summary = engine.evaluate(people, examples, mode, directory,
                                       root / "retrieval-dev", seed)
        publish(stage, summary)
        return rows

    publish()
    try:
        print("Loading training/development cohorts; final test remains closed", flush=True)
        train_people, training = load_cohort("train", args.scope, args.train_people)
        dev_people, development = load_cohort("dev", args.scope, args.dev_people)
        example = training[0]
        readable = "# Private complete training example\n\nDo not include this participant history in the submission.\n\n"
        for message in messages_for(example, train_people[example["pid"]]):
            readable += f"## {message['role']}\n\n{message['content']}\n\n"
        readable += "## Supervised answer\n\n"+",".join(answer_parts(example))+"\n"
        (root / "example.md").write_text(readable)
        status["coverage"] = {"train_participants": len(train_people), "dev_participants": len(dev_people),
            "training_questions": len(training), "development_questions": len(development),
            "training_responses": sum(len(e["atoms"]) for e in training),
            "development_responses": sum(len(e["atoms"]) for e in development),
            "task_groups": len({e["group"] for e in training})}
        baseline = Baselines(training)
        outcomes = {}
        for mode in ("common_answer", "price_aware", "human_retest"):
            rows = [r for e in development for r in score_prediction(e, baseline.predict(e, mode))]
            outcomes[mode] = rows
            publish(mode, summarize(rows))

        for mode in ("question_only", "full_history"):
            status["active_stage"] = "train_"+mode
            publish()
            directory = root / ("sft_"+mode)
            report_path = directory / "training.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            if report.get("status") not in {"complete", "trained", "evaluating_checkpoints"}:
                engine = Engine(plan, training=True, runtime_profile=runtime_profile)
                def progress(report):
                    status.setdefault("training", {})[mode] = report
                    publish()
                report = engine.train(train_people, training, dev_people, development, mode, directory,
                                      args.max_updates, args.eval_every, progress, runtime_profile,
                                      defer_validation=args.defer_checkpoint_evaluation)
                engine.close()
            status.setdefault("training", {})[mode] = report
            if report["status"] != "complete":
                publish()
                continue
            epoch_path = directory / f"update-{report['selected_update']}-dev"
            outcomes["sft_"+mode] = saved_rows(epoch_path)
            publish("sft_"+mode, json.loads((epoch_path/"summary.json").read_text()))

        for mode in ("question_only", "full_history"):
            report = status["training"][mode]
            if report["status"] == "complete":
                continue
            directory = root / ("sft_"+mode)
            def selection_progress(report, update):
                status["training"][mode] = report
                status["active_stage"] = f"select_{mode}_update_{update}"
                publish()
            report = select_saved_checkpoints(plan, runtime_profile, directory, dev_people, development,
                                             report, args.max_updates, args.eval_every, selection_progress)
            selected_path = directory / f"update-{report['selected_update']}-dev"
            outcomes["sft_"+mode] = saved_rows(selected_path)
            publish("sft_"+mode, json.loads((selected_path / "summary.json").read_text()))

        # The frozen encoder is released before loading the prediction model.
        print("Preparing fixed, label-blind semantic retrieval", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(plan["model"], revision=plan["model_revision"])
        prepare_rankings(dev_people, development, tokenizer, root / "retrieval-dev", plan["embedding_revision"])
        engine = Engine(plan, runtime_profile=runtime_profile)
        for mode in ("question_only", "demographics", "full_history", "bm25", "semantic", "hybrid"):
            stage = "frozen_"+mode
            outcomes[stage] = evaluate(engine, stage, dev_people, development, mode)
        for mode in ("shuffled", "random"):
            repeated = []
            for seed in spec["seeds"]:
                stage = f"frozen_{mode}_{seed}"
                repeated.append(evaluate(engine, stage, dev_people, development, mode, seed))
            outcomes["frozen_"+mode] = average_assignments(repeated)
            summary = summarize(outcomes["frozen_"+mode])
            summary["assignment_macro_nll"] = [status["stages"][f"frozen_{mode}_{s}"]["nll"]["mean"] for s in spec["seeds"]]
            publish("frozen_"+mode, summary)
        engine.close()

        engine = Engine(plan, root / "sft_full_history/best_adapter", runtime_profile=runtime_profile)
        outcomes["sft_hybrid"] = evaluate(engine, "sft_hybrid", dev_people, development, "hybrid")
        repeated = []
        for seed in spec["seeds"]:
            repeated.append(evaluate(engine, f"sft_shuffled_{seed}", dev_people, development, "shuffled", seed))
        outcomes["sft_shuffled"] = average_assignments(repeated)
        summary = summarize(outcomes["sft_shuffled"])
        summary["assignment_macro_nll"] = [status["stages"][f"sft_shuffled_{s}"]["nll"]["mean"] for s in spec["seeds"]]
        publish("sft_shuffled", summary)
        engine.close()

        comparisons = [("frozen_full_history", "frozen_question_only"),
            ("frozen_full_history", "frozen_demographics"), ("frozen_full_history", "frozen_shuffled"),
            ("sft_question_only", "frozen_question_only"), ("sft_full_history", "frozen_full_history"),
            ("sft_full_history", "sft_question_only"), ("sft_full_history", "sft_shuffled"),
            ("frozen_hybrid", "frozen_full_history"), ("frozen_hybrid", "frozen_random"),
            ("frozen_hybrid", "frozen_bm25"), ("frozen_hybrid", "frozen_semantic"),
            ("sft_hybrid", "sft_full_history"), ("sft_full_history", "price_aware")]
        for a, b in comparisons:
            status["comparisons"][a+" minus "+b] = {field: paired_difference(outcomes[a], outcomes[b], field)
                                                  for field in ("exact", "agreement", "nll")}
        candidates = ["frozen_question_only", "frozen_full_history", "frozen_hybrid",
                      "sft_question_only", "sft_full_history", "sft_hybrid"]
        metric = selection_metric(plan)
        selected = select_best({name: status["stages"][name] for name in candidates}, metric)
        calibration = fit_temperature(outcomes[selected])
        selection_path = root / "selection.json"
        if selection_path.exists():
            sealed = json.loads(selection_path.read_text())
            if sealed["method"] != selected or sealed["calibration"] != calibration:
                raise ValueError("Development selection changed after being sealed")
            status["selection"] = sealed
        else:
            status["selection"] = {"method": selected, "rule": selection_description(metric),
                                   "calibration": calibration, "frozen_at": now()}
            atomic_json(selection_path, status["selection"])
        publish("selected_calibrated_development", summarize(temperature_scale(outcomes[selected], calibration["temperature"])))

        expected_dev = plan["execution_budget"].get("development_participants", 309)
        full_cohort = args.scope == "full" and len(train_people) == 1440 and len(dev_people) == expected_dev
        if full_cohort:
            # Seal selection BEFORE the first read of reserved test outcomes.
            status["final_test_opened"] = True
            publish()
            test_people, testing = load_cohort("test", "full")
            for mode in ("common_answer", "price_aware", "human_retest"):
                rows = [r for e in testing for r in score_prediction(e, baseline.predict(e, mode))]
                publish("test_"+mode, summarize(rows))
            mode = selected.removeprefix("sft_").removeprefix("frozen_")
            adapter = ("sft_question_only" if mode == "question_only" else "sft_full_history") if selected.startswith("sft_") else None
            if mode == "hybrid":
                prepare_rankings(test_people, testing, tokenizer, root / "retrieval-test", plan["embedding_revision"])
            engine = Engine(plan, root / adapter / "best_adapter" if adapter else None, runtime_profile=runtime_profile)
            rows, summary = engine.evaluate(test_people, testing, mode, root / "final-test",
                                            root / "retrieval-test")
            publish("test_selected_uncalibrated", summary)
            publish("test_selected_calibrated", summarize(temperature_scale(rows, calibration["temperature"])))
            engine.close()
            status["status"] = "complete"
        else:
            status["status"] = "pilot_complete_final_test_unopened"
        publish()
        if status["status"] == "complete":
            from .finalize_submission import finalize
            finalize(root)
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        publish()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scope", choices=["full", "pilot"], default="full")
    parser.add_argument("--train-people", type=int)
    parser.add_argument("--dev-people", type=int)
    parser.add_argument("--max-updates", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument('--selection-metric', choices=['agreement', 'nll'],
                        help='Override checkpoint and configuration selection (NLL reproduces the original study)')
    parser.add_argument("--runtime-profile", type=Path,
                        help="Reproducible update-indexed microbatch/checkpointing settings")
    parser.add_argument("--defer-checkpoint-evaluation", action="store_true",
                        help="Finish both fixed-budget training runs before scoring remaining saved checkpoints")
    args = parser.parse_args()
    if args.max_updates < 1 or args.eval_every < 1:
        parser.error("Update and evaluation budgets must be positive")
    if not re_safe_name(args.name):
        parser.error("Run name must contain only letters, digits, underscores and hyphens")
    if any(n is not None and n < 2 for n in (args.train_people, args.dev_people)):
        parser.error("At least two participants are required for shuffled controls")
    import fcntl
    directory = ROOT / "checkpoints" / args.name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This experiment already has a running process")
        run(args)


def re_safe_name(value):
    import re
    return bool(re.fullmatch(r"[a-zA-Z0-9_-]+", value))


if __name__ == "__main__":
    main()
