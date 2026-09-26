"""Compare inference CPU thread counts on development data, checking exact outputs."""
import argparse
import json
from pathlib import Path
import time

from .benchmark_data import load_cohort
from .experiment_engine import Engine, atomic_json


PREDICTION_FIELDS = ("values", "probabilities", "raw_output", "input_tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--adapter", default="sft_full_history/update-64-adapter")
    parser.add_argument("--threads", nargs="+", type=int, default=[8])
    parser.add_argument("--participants", type=int, default=2)
    parser.add_argument("--reference-predictions", type=Path,
                        help="Saved JSONL predictions from the same adapter and full-history condition")
    args = parser.parse_args()
    if any(n < 1 for n in args.threads) or args.participants < 1:
        parser.error("Thread and participant counts must be positive")
    root = args.run_directory
    plan = json.loads((root / "manifest.json").read_text())["plan"]
    people, examples = load_cohort("dev", limit=args.participants)
    engine = Engine(plan, root / args.adapter)
    report = {"status": "running", "adapter": args.adapter, "execution": engine.metadata,
              "participants": len(people), "questions": len(examples),
              "task_groups": sorted({e["group"] for e in examples}), "measurements": [],
              "policy": "Development only; identical question order and empty prefix cache after warmup; no training or final test."}
    destination = root / "report/inference_runtime_benchmark.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    reference = None
    if args.reference_predictions:
        # Read a snapshot; never truncate a live evaluation writer's file.
        raw = args.reference_predictions.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raw = raw[:raw.rfind(b"\n")+1]
        saved = {(r["pid"], r["id"]): r for r in map(json.loads, raw.splitlines())}
        reference = [saved[(e["pid"], e["id"])] for e in examples]
        report["reference_predictions"] = str(args.reference_predictions)
    try:
        for threads in args.threads:
            engine.torch.set_num_threads(threads)
            engine.reset_prefix_cache()
            engine.predict(examples[0], people[examples[0]["pid"]])
            engine.reset_prefix_cache()
            engine.torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            records = [engine.predict(e, people[e["pid"]]) for e in examples]
            elapsed = time.perf_counter()-start
            if reference is None:
                reference = records
            identical = all(all(a[k] == b[k] for k in PREDICTION_FIELDS)
                            for a, b in zip(reference, records))
            error = max((abs(a-b) for ra, rb in zip(reference, records)
                         for pa, pb in zip(ra["probabilities"], rb["probabilities"]) if pa is not None
                         for a, b in zip(pa, pb)), default=0.)
            row = {"cpu_threads": threads, "seconds": elapsed,
                   "seconds_per_question": elapsed/len(records),
                   "identical_outputs": identical, "max_probability_error": error,
                   "peak_allocated_gib": engine.torch.cuda.max_memory_allocated()/2**30}
            report["measurements"].append(row)
            atomic_json(destination, report)
            print(json.dumps(row), flush=True)
            if not identical:
                raise ValueError("Inference outputs differ from the reference")
        report["status"] = "passed"
        atomic_json(destination, report)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
