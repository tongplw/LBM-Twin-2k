"""GPU integration checks for every answer format; never open final-test data."""
import json
import time

from .data import ROOT
from .benchmark_data import load_cohort, tokenized_example
from .benchmark_metrics import score_prediction
from .experiment_engine import Engine, atomic_json


def main():
    plan = json.loads((ROOT / "configs/model_plan.json").read_text())
    people, examples = load_cohort("dev", limit=2)
    selected = {}
    for e in examples:
        key = e["question"]["QuestionType"]+":"+e["atoms"][0]["kind"]+":"+str(len(e["atoms"]))
        selected.setdefault(key, e)
    report = {"status": "running", "final_test_opened": False, "inference_formats": {},
              "training_formats": {}}
    engine = Engine(plan)
    for kind, e in selected.items():
        prediction = engine.predict(e, people[e["pid"]], use_prefix_cache=False)
        cached = engine.predict(e, people[e["pid"]], use_prefix_cache=True)
        replay = engine.predict(e, people[e["pid"]], use_prefix_cache=True)
        import numpy as np
        if prediction["probabilities"][0] is not None:
            error = float(np.max(np.abs(np.array(prediction["probabilities"])-np.array(cached["probabilities"]))))
            assert error < 1e-5, f"Cached probabilities differ from fresh split-prefill: {error}"
            assert cached["values"] == replay["values"]
            assert np.allclose(cached["probabilities"], replay["probabilities"], atol=1e-5)
        else:
            error = None
            assert prediction["values"] == cached["values"] == replay["values"], "Cache changed numeric output"
        scored = score_prediction(e, prediction)
        assert len(scored) == len(e["atoms"])
        report["inference_formats"][kind] = {"rows": len(scored), "input_tokens": prediction["input_tokens"],
            "seconds": prediction["inference_seconds"], "invalid_rows": sum(r["invalid"] for r in scored),
            "probability_rows": sum(r["nll"] is not None for r in scored),
            "cached_seconds": replay["inference_seconds"], "cache_probability_max_error": error,
            "cache_point_predictions_identical": prediction["values"] == cached["values"]}
        print("Validated inference", kind, report["inference_formats"][kind], flush=True)
    engine.close()
    # Runtime-only optimizer checks use TRAIN participants' EARLIER answers.
    train_people, training = load_cohort("train", limit=2)
    engine = Engine(plan, training=True)
    torch = engine.torch
    optimizer = torch.optim.AdamW([p for p in engine.model.parameters() if p.requires_grad], lr=1e-4)
    for kind in ("Matrix", "Slider", "TE"):
        e = next(row for row in training if row["question"]["QuestionType"] == kind)
        prepared = tokenized_example(engine.tokenizer, e, train_people[e["pid"]], plan["max_sequence_tokens"])
        inputs = torch.tensor([prepared["ids"]], device="cuda")
        positions = torch.tensor([i-1 for start, end in prepared["spans"] for i in range(start, end)], device="cuda")
        engine.model.train()
        before = {k: p.detach().cpu().clone() for k, p in engine.model.named_parameters() if p.requires_grad}
        t0 = time.perf_counter()
        logits = engine.model(input_ids=inputs, logits_to_keep=positions, use_cache=False).logits[0].float()
        loss = torch.nn.functional.cross_entropy(logits, inputs[0, positions+1])
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(engine.model.parameters(), 1.)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        changed = sum(not torch.equal(before[k], p.detach().cpu()) for k, p in engine.model.named_parameters() if p.requires_grad)
        assert changed > 0
        torch.cuda.synchronize()
        report["training_formats"][kind] = {"loss": float(loss.detach()), "changed_adapter_tensors": changed,
            "seconds": time.perf_counter()-t0, "tokens": len(prepared["ids"])}
        print("Validated optimizer update", kind, report["training_formats"][kind], flush=True)
    engine.close()
    report["status"] = "passed"
    report["note"] = "Diagnostic updates only; these weights are discarded and are not study results."
    (ROOT / "data/validation").mkdir(parents=True, exist_ok=True)
    atomic_json(ROOT / "data/validation/full_format_gpu_validation.json", report)


if __name__ == "__main__":
    main()
