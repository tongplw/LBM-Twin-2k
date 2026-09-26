"""Measure local training microbatches without modifying study weights or data."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .batching import answer_losses
from .experiment_engine import Engine, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--full-history", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.run_directory
    plan = json.loads((root / "manifest.json").read_text())["plan"]
    paths = sorted((root / "sft_question_only/tokens").glob("*.npz"))[:32]
    prepared = []
    for path in paths:
        with np.load(path) as p:
            prepared.append({"ids": p["ids"].tolist(), "spans": p["spans"].tolist()})
    assert len(prepared) == 32
    engine = Engine(plan, training=True)
    torch = engine.torch
    from peft import set_peft_model_state_dict
    state = torch.load(root / "sft_question_only/resume.pt", map_location="cpu", weights_only=False)
    set_peft_model_state_dict(engine.model, state["adapter"])
    del state
    if args.full_history:
        from .benchmark_data import load_cohort, tokenized_example
        people, examples = load_cohort("train", limit=2)
        selected = [next(e for e in examples if e["pid"] == pid and e["question"]["QuestionType"] == "Matrix")
                    for pid in people]
        prepared = [tokenized_example(engine.tokenizer, e, people[e["pid"]], plan["max_sequence_tokens"])
                    for e in selected]*2
    report = {"status": "running", "execution": engine.metadata, "measurements": [],
              "examples": len(prepared), "min_tokens": min(len(p["ids"]) for p in prepared),
              "max_tokens": max(len(p["ids"]) for p in prepared),
              "note": "Disposable profiling; no optimizer updates or final-test outcomes."}
    destination = root / ("report/full_history_runtime_benchmark.json" if args.full_history else "report/runtime_benchmark.json")
    def measure(batch, threads, checkpointing, items):
        torch.set_num_threads(threads)
        if checkpointing:
            engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            engine.model.gradient_checkpointing_disable()
        engine.model.train()
        engine.model.zero_grad(set_to_none=True)
        answer_losses(engine.model, items[:batch], torch, engine.tokenizer.pad_token_id).mean().backward()
        engine.model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for begin in range(0, len(items), batch):
            loss = answer_losses(engine.model, items[begin:begin+batch], torch, engine.tokenizer.pad_token_id).sum()/len(items)
            loss.backward()
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        row = {"microbatch": batch, "cpu_threads": threads, "checkpointing": checkpointing,
               "examples": len(items), "seconds": seconds, "seconds_per_example": seconds/len(items),
               "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}
        report["measurements"].append(row)
        atomic_json(destination, report)
        print(json.dumps(row), flush=True)
        engine.model.zero_grad(set_to_none=True)
        return row
    if args.full_history:
        for batch in (1, 2):
            try:
                measure(batch, 8, True, prepared)
            except torch.cuda.OutOfMemoryError:
                engine.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                report["measurements"].append({"microbatch": batch, "out_of_memory": True})
        report["status"] = "passed"
        atomic_json(destination, report)
        engine.close()
        return
    if args.validate_only:
        report = json.loads(destination.read_text())
        report["status"] = "validating_full_batch"
        atomic_json(destination, report)
    else:
        # Match execution overhead before spending time sweeping batch sizes.
        for threads in (8, 1, 2, 4):
            measure(1, threads, True, prepared[:4])
        best_threads = min(report["measurements"], key=lambda r: r["seconds_per_example"])["cpu_threads"]
        for batch in (1, 4, 8, 16, 32):
            for checkpointing in (True, False):
                try:
                    measure(batch, best_threads, checkpointing, prepared)
                except torch.cuda.OutOfMemoryError:
                    engine.model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    report["measurements"].append({"microbatch": batch, "cpu_threads": best_threads,
                                                  "checkpointing": checkpointing, "out_of_memory": True})
                    atomic_json(destination, report)
    # Dropout off: compare batched and independent per-example losses/gradients.
    engine.model.train()
    engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for module in engine.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    engine.model.zero_grad(set_to_none=True)
    singles = []
    for p in prepared:
        loss = answer_losses(engine.model, [p], torch, engine.tokenizer.pad_token_id)[0]
        singles.append(float(loss.detach()))
        (loss/len(prepared)).backward()
    reference = torch.cat([p.grad.flatten().float() for p in engine.model.parameters() if p.requires_grad]).cpu()
    engine.model.zero_grad(set_to_none=True)
    batched = answer_losses(engine.model, prepared, torch, engine.tokenizer.pad_token_id)
    batched.mean().backward()
    candidate = torch.cat([p.grad.flatten().float() for p in engine.model.parameters() if p.requires_grad]).cpu()
    reference, candidate = reference.double(), candidate.double()
    relative = float((reference-candidate).norm()/reference.norm())
    cosine = float(torch.nn.functional.cosine_similarity(reference, candidate, dim=0))
    error = float(np.max(np.abs(np.array(singles)-batched.detach().cpu().numpy())))
    report["batch_validation"] = {"single_losses": singles, "batched_losses": batched.detach().cpu().tolist(),
        "max_loss_difference": error, "gradient_relative_l2_difference": relative, "gradient_cosine": cosine,
        "note": "BF16 padding and batched kernels change rounding; dropout disabled for this comparison."}
    # Isolate padding from batching: independent padded examples should reproduce
    # the batched objective more closely than unpadded BF16 computations.
    del batched, reference
    engine.model.zero_grad(set_to_none=True)
    padded_singles = []
    width = max(len(p["ids"]) for p in prepared)
    for p in prepared:
        loss = answer_losses(engine.model, [p], torch, engine.tokenizer.pad_token_id, pad_to_width=width)[0]
        padded_singles.append(float(loss.detach()))
        (loss/len(prepared)).backward()
    padded_gradient = torch.cat([p.grad.flatten().float() for p in engine.model.parameters() if p.requires_grad]).cpu().double()
    padded_relative = float((padded_gradient-candidate).norm()/padded_gradient.norm())
    padded_cosine = float(torch.nn.functional.cosine_similarity(padded_gradient, candidate, dim=0))
    padded_error = float(np.max(np.abs(np.array(padded_singles)-np.array(report["batch_validation"]["batched_losses"]))))
    report["same_padding_validation"] = {"max_loss_difference": padded_error,
        "gradient_relative_l2_difference": padded_relative, "gradient_cosine": padded_cosine}
    report["status"] = "validation_pending"
    atomic_json(destination, report)
    assert padded_error < .05 and padded_relative < .1 and padded_cosine > .995, report["same_padding_validation"]
    assert relative < .1 and cosine > .995, report["batch_validation"]
    report["status"] = "passed"
    atomic_json(destination, report)
    print(json.dumps(report["batch_validation"]), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
