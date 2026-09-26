"""Resumable LoRA training and label-blind full-benchmark inference."""
import copy
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np

from .benchmark_data import digest, messages_for, tokenized_example
from .benchmark_metrics import score_prediction, summarize
from .retrieval import donor_assignment, select_items
from .selection import selection_metric, selection_loss, selection_description


def language_lora_targets(model, linear_class):
    names = [name for name, module in model.named_modules()
             if name.startswith("model.language_model.layers.") and isinstance(module, linear_class)]
    if not names:
        raise ValueError("No Qwen language-layer linear modules found")
    return names


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def read_predictions(path):
    records = []
    path = Path(path)
    if not path.exists():
        return records
    # Recover only an interrupted final write. Invalid complete lines are errors.
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[:raw.rfind(b"\n")+1]
        path.write_bytes(raw)
    for line in raw.splitlines():
        records.append(json.loads(line))
    return records


class Engine:
    def __init__(self, plan, adapter=None, training=False, runtime_profile=None):
        import torch
        import transformers
        import peft
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, set_seed
        from peft import LoraConfig, PeftModel, get_peft_model
        self.torch, self.plan = torch, plan
        self.question_prediction_reuse = bool((runtime_profile or {}).get("question_only_prediction_reuse", False))
        self._prefix_cache = {}
        self._token_prefix_cache = {}
        self._prefix_owner = None
        torch.set_num_threads(8)
        set_seed(plan["training"]["seed"])
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This experiment requires the specified bfloat16 CUDA GPU")
        self.tokenizer = AutoTokenizer.from_pretrained(plan["model"], revision=plan.get("model_revision"))
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            plan["model"], revision=plan.get("model_revision"), dtype=torch.bfloat16,
            attn_implementation="sdpa").cuda()
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=training)
        elif training:
            cfg = plan["lora"]
            targets = language_lora_targets(self.model, torch.nn.Linear)
            self.model = get_peft_model(self.model, LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"],
                lora_dropout=cfg["dropout"], target_modules=targets, task_type="CAUSAL_LM"))
        if training:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.model.config.use_cache = self.model.config.text_config.use_cache = False
        self.label_ids = {chr(65+i): self.tokenizer.encode(chr(65+i), add_special_tokens=False) for i in range(26)}
        if any(len(v) != 1 for v in self.label_ids.values()):
            raise ValueError("Scoring implementation requires single-token option labels")
        self.metadata = {"torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16",
            "model_revision": self.model.config._commit_hash,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "attention": "sdpa", "batch_size": 1, "cache_reuse_between_questions": "immutable history prefix; copied before each independent question"}
        import importlib.metadata
        self.metadata["flash_linear_attention"] = importlib.metadata.version("flash-linear-attention")
        self.metadata["cuda_runtime"] = torch.version.cuda
        from .runtime_environment import snapshot
        self.metadata["host_runtime"] = snapshot()
        for warning in self.metadata["host_runtime"]["warnings"]:
            print("Runtime warning:", warning, flush=True)

    def close(self):
        self.reset_prefix_cache()
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()

    def reset_prefix_cache(self):
        self._prefix_cache.clear()
        self._token_prefix_cache.clear()
        self._prefix_owner = None

    def predict(self, example, items, use_prefix_cache=True):
        torch = self.torch
        if self.model.training:
            self.model.eval()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if self._prefix_owner != example["pid"]:
            self.reset_prefix_cache()
            self._prefix_owner = example["pid"]
        messages = messages_for(example, items)
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        before, after = rendered.rsplit("\nQUESTION\n", 1)
        prefix_text, suffix_text = before+"\n", "QUESTION\n"+after
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)
        if prefix_text not in self._token_prefix_cache:
            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            # Verify the literal newline boundary against complete chat tokenization.
            if prefix_ids+suffix_ids != self.tokenizer.encode(rendered, add_special_tokens=False):
                raise ValueError("Cached tokenization would change the exact chat-template input")
            self._token_prefix_cache[prefix_text] = prefix_ids
        prefix_ids = self._token_prefix_cache[prefix_text]
        ids = prefix_ids+suffix_ids
        if len(ids)+128 > self.plan["max_sequence_tokens"]:
            raise ValueError("Inference would exceed native context; no truncation")
        x = torch.tensor([ids], device="cuda")
        values, probabilities = [], []
        cached, prefix_length, cache_hit = None, 0, False
        with torch.inference_mode():
            if items:
                key = digest(prefix_ids)
                prefix_length = len(prefix_ids)
                cache_hit = use_prefix_cache and key in self._prefix_cache
                if not cache_hit:
                    if len(self._prefix_cache) >= 8:
                        self._prefix_cache.clear()
                    prefix_output = self.model(input_ids=x[:, :prefix_length], logits_to_keep=1, use_cache=True)
                    if use_prefix_cache:
                        self._prefix_cache[key] = prefix_output.past_key_values
                    else:
                        cached = prefix_output.past_key_values
                    del prefix_output
                # DeltaNet states mutate in place. Never let one target modify the saved prefix.
                if use_prefix_cache:
                    cached = copy.deepcopy(self._prefix_cache[key])
            if example["atoms"][0]["levels"]:
                output = self.model(input_ids=x[:, prefix_length:], past_key_values=cached,
                                    logits_to_keep=1, use_cache=True)
                for i, atom in enumerate(example["atoms"]):
                    choices = [self.label_ids[chr(65+j)][0] for j in range(atom["levels"])]
                    p = output.logits[0, -1, choices].float().softmax(-1)
                    index = int(p.argmax())
                    values.append(index+1)
                    probabilities.append(p.cpu().tolist())
                    if i+1 < len(example["atoms"]):
                        # A single joint matrix prediction; previous MODEL choices only.
                        continuation = [choices[index]]+self.tokenizer.encode(",", add_special_tokens=False)
                        output = self.model(input_ids=torch.tensor([continuation], device="cuda"),
                            past_key_values=output.past_key_values, use_cache=True, logits_to_keep=1)
                raw = ",".join(chr(64+v) for v in values)
            else:
                output = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x),
                    past_key_values=cached, max_new_tokens=128, do_sample=False, use_cache=True,
                    logits_to_keep=1, pad_token_id=self.tokenizer.eos_token_id)
                generated = output[0, len(ids):].cpu().tolist()
                raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
                parts = raw.split(",")
                terminated = self.tokenizer.eos_token_id in generated
                number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
                if terminated and len(parts) == len(example["atoms"]) and all(re.fullmatch(number, p.strip()) for p in parts):
                    values = [float(part) for part in parts]
                else:
                    values = [None]*len(example["atoms"])
                probabilities = [None]*len(values)
        torch.cuda.synchronize()
        return {"values": values, "probabilities": probabilities, "raw_output": raw,
                "input_tokens": len(ids), "inference_seconds": time.perf_counter()-start,
                "prefix_cache_hit": cache_hit, "cached_prefix_tokens": prefix_length if cache_hit else 0,
                "processed_input_tokens": len(ids)-(prefix_length if cache_hit else 0)}

    def evaluate(self, people, examples, mode, destination, retrieval_cache=None, seed=42):
        self.reset_prefix_cache()
        self.torch.cuda.reset_peak_memory_stats()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        identity = digest({"examples": examples, "mode": mode, "seed": seed,
                           "people": people, "plan": self.plan})
        provenance = destination / "identity.json"
        if provenance.exists() and json.loads(provenance.read_text())["hash"] != identity:
            raise ValueError("Evaluation inputs changed; refusing cached predictions")
        atomic_json(provenance, {"hash": identity, "mode": mode, "seed": seed})
        path = destination / "predictions.jsonl"
        records = read_predictions(path)
        keys = [(e["pid"], e["id"]) for e in examples]
        if [(r["pid"], r["id"]) for r in records] != keys[:len(records)]:
            raise ValueError("Cached prediction order/coverage differs")
        summary_path = destination / "summary.json"
        if len(records) == len(examples) and summary_path.exists():
            return [r for record in records for r in record["scored"]], json.loads(summary_path.read_text())
        sessions_path = destination / "runtime_sessions.json"
        sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
        if sessions:
            sessions[-1]["end_prediction"] = len(records)
        elif records:
            sessions.append({"first_prediction": 0, "end_prediction": len(records),
                             "execution": "not recorded before runtime instrumentation"})
        sessions.append({"started": datetime.now(timezone.utc).isoformat(),
                         "first_prediction": len(records), "execution": self.metadata})
        atomic_json(sessions_path, sessions)
        donors = donor_assignment(people, seed) if mode == "shuffled" else {}
        # Identical answer-free prompts under the same fixed model have identical
        # deterministic predictions. Never cache labels or reuse across checkpoints.
        reuse = self.question_prediction_reuse and mode == "question_only"
        prediction_cache = {}
        prediction_fields = ("values", "probabilities", "raw_output", "input_tokens")
        if reuse:
            for e, record in zip(examples, records):
                key = digest(messages_for(e, []))
                reusable = {field: record[field] for field in prediction_fields}
                if key in prediction_cache and prediction_cache[key] != reusable:
                    raise ValueError("Identical question-only prompts have inconsistent saved outputs")
                prediction_cache[key] = reusable
        warmup_path = destination / "warmup.json"
        if len(records) < len(examples):
            e = examples[len(records)]
            items = people[donors.get(e["pid"], e["pid"])]
            selected, _ = select_items(items, e, mode, retrieval_cache, seed, tokenizer=self.tokenizer)
            warmup = self.predict(e, selected)
            self.reset_prefix_cache()
            atomic_json(warmup_path, {"seconds": warmup["inference_seconds"],
                "policy": "one unscored same-condition prediction before each invocation; cache then reset"})
        with path.open("a") as stream:
            for index in range(len(records), len(examples)):
                e = examples[index]
                items = people[donors.get(e["pid"], e["pid"])]
                selected, cost = select_items(items, e, mode, retrieval_cache, seed, tokenizer=self.tokenizer)
                lookup_start = time.perf_counter()
                key = digest(messages_for(e, [])) if reuse else None
                if reuse and key in prediction_cache:
                    result = {**prediction_cache[key], "inference_seconds": time.perf_counter()-lookup_start,
                              "prefix_cache_hit": False, "cached_prefix_tokens": 0,
                              "processed_input_tokens": 0, "prediction_reused": True}
                else:
                    result = self.predict(e, selected)
                    result["prediction_reused"] = False
                    if reuse:
                        prediction_cache[key] = {field: result[field] for field in prediction_fields}
                record = {"pid": e["pid"], "id": e["id"], **result, **cost,
                          "scored": score_prediction(e, result)}
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
                records.append(record)
                if (index+1) % 50 == 0:
                    print(f"Evaluate {destination.name}: {index+1}/{len(examples)}", flush=True)
        rows = [r for record in records for r in record["scored"]]
        summary = summarize(rows)
        summary["timing"] = {"input_tokens": int(sum(r["input_tokens"] for r in records)),
            "mean_input_tokens": float(np.mean([r["input_tokens"] for r in records])),
            "inference_seconds": sum(r["inference_seconds"] for r in records),
            "retrieval_seconds": sum(r["retrieval_seconds"] for r in records),
            "query_encoding_seconds": sum(r["query_seconds"] for r in records),
            "processed_input_tokens": sum(r.get("processed_input_tokens", r["input_tokens"]) for r in records),
            "prefix_cache_hits": sum(r.get("prefix_cache_hit", False) for r in records),
            "identical_prompt_predictions_reused": sum(r.get("prediction_reused", False) for r in records),
            "question_only_reuse_enabled": reuse,
            "warmup_policy": "one unscored same-condition prediction; cache reset before scoring"}
        summary["timing"]["total_prediction_seconds"] = sum(summary["timing"][field]
            for field in ("inference_seconds", "retrieval_seconds", "query_encoding_seconds"))
        sessions[-1]["end_prediction"] = len(records)
        atomic_json(sessions_path, sessions)
        summary.update(status="complete", input_policy=mode, seed=seed, execution=self.metadata,
                       runtime_sessions=sessions)
        summary["peak_allocated_gib"] = self.torch.cuda.max_memory_allocated()/2**30
        atomic_json(destination / "summary.json", summary)
        self.reset_prefix_cache()
        return rows, summary

    def train(self, train_people, training, dev_people, development, mode, destination,
              max_updates=128, eval_every=32, on_progress=None, runtime_profile=None,
              defer_validation=False):
        """Fixed matched update budget; sample the task-balanced objective directly."""
        torch = self.torch
        from transformers import get_linear_schedule_with_warmup
        from .benchmark_data import balanced_schedule, sampled_coverage
        from .batching import answer_losses, training_runtime
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        token_dir = destination / "tokens"
        token_dir.mkdir(exist_ok=True)
        accum = self.plan["training"]["gradient_accumulation"]
        schedule = balanced_schedule(training, max_updates*accum, self.plan["training"]["seed"])
        schedule_hash = digest(schedule)
        identity = digest({"plan": self.plan, "training": training, "development": development,
                           "history": train_people, "mode": mode, "max_updates": max_updates,
                           "eval_every": eval_every, "schedule": schedule_hash})
        identity_path = destination / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text())["hash"] != identity:
            raise ValueError("Training provenance changed; refusing resume")
        atomic_json(identity_path, {"hash": identity})
        optim = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
            lr=self.plan["optimizer"]["learning_rate"], weight_decay=self.plan["optimizer"]["weight_decay"])
        scheduler = get_linear_schedule_with_warmup(optim,
            max(1, math.ceil(max_updates*self.plan["training"]["warmup_ratio"])), max_updates)
        metric = selection_metric(self.plan, checkpoint=True)
        report = {"status": "training", "condition": mode, "checkpoints": [], "selected_update": None,
                  "plan": self.plan, "execution": self.metadata, "updates": 0, "processed_tokens": 0,
                  "training_seconds": 0., "peak_allocated_gib": 0., "available_training_questions": len(training),
                  "planned_updates": max_updates, "planned_draws": len(schedule),
                  **sampled_coverage(training, schedule, 0),
                  "sampling": "cycle all participants; randomized task order per person; response-count-weighted question sample",
                  "schedule_sha256": schedule_hash, "loss_curve": [], "selected_by": selection_description(metric)}
        best_selection_score = float("inf")
        resume = destination / "resume.pt"
        if resume.exists():
            state = torch.load(resume, map_location="cpu", weights_only=False)
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.model, state["adapter"])
            optim.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            report = state["report"]
            best_selection_score = min((selection_loss(c["development"], metric)
                                        for c in report["checkpoints"]), default=float("inf"))
            del state
        for point in report["loss_curve"]:
            expected = training_runtime(runtime_profile, mode, point["update"]-1)
            actual = point.get("runtime", training_runtime(None, mode, 0))
            if expected != actual:
                raise ValueError("Runtime profile would rewrite already-completed update settings")
        report["runtime_profile"] = runtime_profile

        def checkpoint():
            from peft import get_peft_model_state_dict
            report.update(sampled_coverage(training, schedule, report["updates"]*accum))
            state = {"adapter": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(self.model).items()},
                     "optimizer": optim.state_dict(), "scheduler": scheduler.state_dict(),
                     "best_selection_score": best_selection_score, "report": report,
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                     "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
            temp = resume.with_suffix(".tmp")
            torch.save(state, temp)
            temp.replace(resume)
            atomic_json(destination / "training.json", report)
            if on_progress:
                on_progress(report)

        def validate():
            nonlocal best_selection_score
            update = report["updates"]
            if any(c["update"] == update for c in report["checkpoints"]):
                return
            # Retain every evaluated checkpoint, including candidates that lose
            # under the current metric; later audits must not substitute weights.
            save_candidate()
            self.reset_prefix_cache()
            _, validation = self.evaluate(dev_people, development, mode, destination / f"update-{update}-dev")
            report["checkpoints"].append({"update": update, "development": validation})
            score = selection_loss(validation, metric)
            if score < best_selection_score:
                best_selection_score = score
                report["selected_update"] = update
                self.model.save_pretrained(destination / "best_adapter")
                self.tokenizer.save_pretrained(destination / "best_adapter")
            checkpoint()
            self.reset_prefix_cache()
            self.model.train()

        def save_candidate():
            update = report["updates"]
            self.model.save_pretrained(destination / f"update-{update}-adapter")
            self.tokenizer.save_pretrained(destination / f"update-{update}-adapter")
            report["saved_checkpoint_updates"] = sorted(set(report.get("saved_checkpoint_updates", [])) | {update})
            checkpoint()

        if report["updates"] and (report["updates"] % eval_every == 0 or report["updates"] == max_updates):
            if defer_validation:
                save_candidate()
            else:
                validate()
        self.reset_prefix_cache()
        self.model.train()
        for update in range(report["updates"], max_updates):
            runtime = training_runtime(runtime_profile, mode, update)
            torch.set_num_threads(runtime["cpu_threads"])
            if runtime["checkpointing"]:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                self.model.gradient_checkpointing_disable()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            indices = schedule[update*accum:(update+1)*accum]
            optim.zero_grad(set_to_none=True)
            group_loss = 0.
            batch_inputs = []
            for index in indices:
                e = training[index]
                path = token_dir / f"{index}.npz"
                if path.exists():
                    with np.load(path) as saved:
                        ids, spans = saved["ids"].tolist(), saved["spans"].tolist()
                else:
                    items = train_people[e["pid"]] if mode == "full_history" else []
                    prepared = tokenized_example(self.tokenizer, e, items, self.plan["max_sequence_tokens"])
                    ids, spans = prepared["ids"], prepared["spans"]
                    temp = path.with_suffix(".tmp")
                    with temp.open("wb") as stream:
                        np.savez_compressed(stream, ids=np.asarray(ids, dtype=np.int32), spans=np.asarray(spans))
                    temp.replace(path)
                batch_inputs.append({"ids": ids, "spans": spans})
                report["processed_tokens"] += len(ids)
            for begin in range(0, len(batch_inputs), runtime["microbatch"]):
                chunk = batch_inputs[begin:begin+runtime["microbatch"]]
                if len(chunk) == 1:
                    ids, spans = chunk[0]["ids"], chunk[0]["spans"]
                    positions = torch.tensor([j-1 for start, end in spans for j in range(start, end)], device="cuda")
                    inputs = torch.tensor([ids], device="cuda")
                    logits = self.model(input_ids=inputs, logits_to_keep=positions, use_cache=False).logits[0].float()
                    losses = torch.nn.functional.cross_entropy(logits, inputs[0, positions+1], reduction="none")
                    atom_losses, cursor = [], 0
                    for start, end in spans:
                        atom_losses.append(losses[cursor:cursor+end-start].mean())
                        cursor += end-start
                    loss = torch.stack(atom_losses).mean()/len(indices)
                else:
                    loss = answer_losses(self.model, chunk, torch, self.tokenizer.pad_token_id).sum()/len(indices)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss; do not skip examples")
                loss.backward()
                group_loss += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.plan["optimizer"]["clip_grad_norm"])
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite gradients")
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            report["updates"] = update+1
            report["training_seconds"] += time.perf_counter()-t0
            report["peak_allocated_gib"] = max(report["peak_allocated_gib"], torch.cuda.max_memory_allocated()/2**30)
            report["loss_curve"].append({"update": update+1, "answer_loss": group_loss,
                                         "gradient_norm": float(norm), "learning_rate": scheduler.get_last_lr()[0],
                                         "runtime": runtime})
            checkpoint()
            print(json.dumps({"condition": mode, "update": update+1, "total_updates": max_updates,
                "loss": group_loss, "seconds_per_update": report["training_seconds"]/(update+1)}), flush=True)
            if (update+1) % eval_every == 0 or update+1 == max_updates:
                if defer_validation:
                    save_candidate()
                else:
                    validate()
        report["status"] = "trained" if defer_validation else "complete"
        atomic_json(destination / "training.json", report)
        return report
