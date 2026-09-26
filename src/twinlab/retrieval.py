"""Within-person BM25, frozen Qwen embeddings, RRF and matched random controls."""
import json
import random
import time
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .benchmark_data import digest, target_text, pack_evidence


def bm25_scores(query, documents, k1=1.5, b=.75):
    tokens = [re.findall(r"[a-z0-9]+", doc.lower()) for doc in documents]
    query = set(re.findall(r"[a-z0-9]+", query.lower()))
    avgdl = sum(map(len, tokens))/max(len(tokens), 1)
    df = Counter(term for doc in tokens for term in set(doc))
    result = []
    for doc in tokens:
        tf = Counter(doc)
        result.append(sum(math.log(1+(len(tokens)-df[t]+.5)/(df[t]+.5))*tf[t]*(k1+1)
            /(tf[t]+k1*(1-b+b*len(doc)/max(avgdl, 1))) for t in query))
    return result


def donor_assignment(ids, seed):
    """Seeded cyclic derangement: no self matches, donors stay within the split."""
    ids = sorted(ids)
    if len(ids) < 2:
        raise ValueError("Donor evaluation requires at least two participants")
    random.Random(seed).shuffle(ids)
    return {pid: ids[(i+1) % len(ids)] for i, pid in enumerate(ids)}


def rrf_scores(keyword, semantic, constant=60):
    if len(keyword) != len(semantic):
        raise ValueError("Retrieval branches must rank the same candidate items")
    out = np.zeros(len(keyword))
    for scores, positive_only in ((keyword, True), (semantic, False)):
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        for rank, i in enumerate(order, 1):
            if not positive_only or scores[i] > 0:
                out[i] += 1 / (constant+rank)
    return out


def packed_length(indices, lengths, parents=None, shared=None):
    value = sum(lengths[i] for i in indices)
    if parents is not None:
        seen = set()
        for i in indices:
            if parents[i] in seen:
                value -= shared[i]
            seen.add(parents[i])
    return int(value)


def matched_random(lengths, target, count, seed, parents=None, shared=None):
    """Match item count exactly and token sum approximately without reading labels."""
    if count >= len(lengths):
        return list(range(len(lengths)))
    rng = random.Random(seed)
    best, error = None, float("inf")
    # Fixed effort independent of observed outcomes. Include the residual in logs.
    for _ in range(256):
        selected = rng.sample(range(len(lengths)), count)
        delta = abs(packed_length(selected, lengths, parents, shared)-target)
        if delta < error:
            best, error = selected, delta
        if delta == 0:
            break
    # Conditional random sampling must also reach short, clustered matrix inputs.
    # Uniform subsets alone rarely do. Random swaps optimize length only, never relevance.
    selected = list(best)
    for _ in range(2048):
        slot = rng.randrange(count)
        replacement = rng.randrange(len(lengths))
        if replacement in selected:
            continue
        trial = selected.copy()
        trial[slot] = replacement
        delta = abs(packed_length(trial, lengths, parents, shared)-target)
        if delta <= error:
            selected, error = trial, delta
            best = selected.copy()
        if delta == 0:
            break
    return best


class Encoder:
    def __init__(self, model="Qwen/Qwen3-Embedding-0.6B", revision=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, padding_side="left")
        self.model = AutoModel.from_pretrained(model, revision=revision, dtype=torch.bfloat16,
                                              attn_implementation="sdpa").cuda().eval()
        self.revision = self.model.config._commit_hash

    def encode(self, texts, query=False):
        torch = self.torch
        if query:
            texts = ["Instruct: Retrieve past survey responses relevant to predicting this person's answer.\nQuery: "+s
                     for s in texts]
        result = []
        with torch.inference_mode():
            for start in range(0, len(texts), 16):
                inputs = self.tokenizer(texts[start:start+16], padding=True, truncation=False, return_tensors="pt")
                if inputs.input_ids.shape[1] > self.model.config.max_position_embeddings:
                    raise ValueError("Evidence item exceeds embedding context; no silent truncation")
                hidden = self.model(**inputs.to("cuda")).last_hidden_state[:, -1].float()
                result.append(torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy())
        return np.concatenate(result)


def packing_metadata(items, tokenizer):
    parents, shared = [], []
    for item in items:
        parents.append(item["source"].rsplit("/row", 1)[0])
        common = json.loads(json.dumps(item))
        common["source"] = parents[-1]
        if "/row" in item["source"]:
            for key in ("Rows", "RowsID", "Statements", "StatementsID"):
                common["question"].pop(key, None)
            common["answer"] = {}
            shared.append(len(tokenizer.encode(json.dumps(common, ensure_ascii=False, separators=(",", ":")),
                                                add_special_tokens=False)))
        else:
            shared.append(0)
    return np.array(parents), np.array(shared)


def prepare_rankings(people, examples, tokenizer, destination, revision=None):
    """Cache semantic similarities and lengths; encoder never sees target answers."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    by_person = {}
    for e in examples:
        by_person.setdefault(e["pid"], []).append(e)
    encoder = None
    start = time.perf_counter()
    metadata = {"encoder": "Qwen/Qwen3-Embedding-0.6B", "query_encoding_included": True,
                "history_embeddings_cached": True, "people": {}}
    for number, (pid, rows) in enumerate(by_person.items(), 1):
        items = people[pid]
        docs = [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in items
                if item["block"] != "Demographics"]
        queries = [target_text(e) for e in rows]
        identity = digest({"docs": docs, "queries": queries, "encoder": metadata["encoder"], "revision": revision})
        path = destination / f"{pid}.npz"
        if path.exists():
            with np.load(path) as data:
                if str(data["identity"]) == identity:
                    if "parents" not in data or "shared" not in data:
                        raise ValueError("Old retrieval cache lacks packed-length metadata")
                    metadata["encoder_revision"] = str(data["encoder_revision"])
                    metadata["people"][pid] = {"history_seconds": float(data["history_seconds"]),
                        "query_seconds": float(data["query_seconds"])*len(rows),
                        "items": len(docs), "queries": len(rows)}
                    continue
            raise ValueError("Retrieval cache provenance changed; use a new run directory")
        if encoder is None:
            encoder = Encoder(revision=revision)
            metadata["encoder_revision"] = encoder.revision
        t0 = time.perf_counter()
        history = encoder.encode(docs)
        history_seconds = time.perf_counter()-t0
        t0 = time.perf_counter()
        query_vectors = encoder.encode(queries, query=True)
        query_seconds = time.perf_counter()-t0
        lengths = np.array([len(tokenizer.encode(doc, add_special_tokens=False)) for doc in docs])
        candidates = [item for item in items if item["block"] != "Demographics"]
        parents, shared = packing_metadata(candidates, tokenizer)
        np.savez(path, identity=identity, similarity=query_vectors @ history.T, lengths=lengths,
                 query_ids=np.array([e["id"] for e in rows]), query_seconds=query_seconds/len(rows),
                 history_seconds=history_seconds, encoder_revision=encoder.revision,
                 parents=np.array(parents), shared=np.array(shared))
        metadata["people"][pid] = {"history_seconds": history_seconds, "query_seconds": query_seconds,
                                   "items": len(docs), "queries": len(rows)}
        if number % 10 == 0:
            print(f"Embedded histories: {number}/{len(by_person)}", flush=True)
    metadata["wall_seconds_this_invocation"] = time.perf_counter()-start
    metadata["total_history_encoding_seconds"] = sum(p["history_seconds"] for p in metadata["people"].values())
    metadata["total_query_encoding_seconds"] = sum(p["query_seconds"] for p in metadata["people"].values())
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    del encoder
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def select_items(items, example, mode, cache_dir=None, seed=42, k=16, tokenizer=None):
    start = time.perf_counter()
    if mode in {"full_history", "shuffled"}:
        return items, {"retrieval_seconds": 0., "query_seconds": 0.}
    if mode == "question_only":
        return [], {"retrieval_seconds": 0., "query_seconds": 0.}
    demo = [i for i, item in enumerate(items) if item["block"] == "Demographics"]
    if mode == "demographics":
        return [items[i] for i in demo], {"retrieval_seconds": 0., "query_seconds": 0.}
    candidates = [i for i in range(len(items)) if i not in demo]
    docs = [json.dumps(items[i], ensure_ascii=False, separators=(",", ":")) for i in candidates]
    keyword = bm25_scores(target_text(example), docs)
    with np.load(Path(cache_dir) / f"{example['pid']}.npz") as data:
        matches = np.flatnonzero(data["query_ids"] == example["id"])
        if len(matches) != 1:
            raise ValueError("Missing/duplicate cached retrieval query")
        semantic = data["similarity"][int(matches[0])]
        lengths = data["lengths"]
        parents, shared = data["parents"], data["shared"]
        query_seconds = float(data["query_seconds"])
    hybrid = rrf_scores(keyword, semantic)
    scores = {"bm25": keyword, "semantic": semantic, "hybrid": hybrid, "random": hybrid}[mode]
    chosen = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))[:k]
    reference_length = packed_length(chosen, lengths, parents, shared)
    reference_chosen = chosen.copy()
    match_error = None
    if mode == "random":
        if tokenizer is None:
            raise ValueError("Random controls require the prediction tokenizer for real length matching")
        def actual_length(indices):
            selected = [items[i] for i in sorted(demo+[candidates[j] for j in indices])]
            return len(tokenizer.encode(json.dumps(pack_evidence(selected), ensure_ascii=False, separators=(",", ":")),
                                        add_special_tokens=False))
        target_actual = actual_length(reference_chosen)
        random_seed = seed+int(digest([example["pid"], example["id"]])[:8], 16)
        chosen = matched_random(lengths, reference_length, len(chosen),
                                random_seed, parents, shared)
        best_actual_error = abs(actual_length(chosen)-target_actual)
        adjusted_target = reference_length
        for attempt in range(8):
            current_length = actual_length(chosen)
            if best_actual_error/max(1, target_actual) <= .02:
                break
            adjusted_target += target_actual-current_length
            trial = matched_random(lengths, adjusted_target, len(chosen), random_seed+attempt+1, parents, shared)
            error = abs(actual_length(trial)-target_actual)
            if error < best_actual_error:
                chosen, best_actual_error = trial, error
        match_error = best_actual_error/max(1, target_actual)
    selected = [items[i] for i in sorted(demo+[candidates[j] for j in chosen])]
    return selected, {"retrieval_seconds": time.perf_counter()-start,
                      "query_seconds": query_seconds if mode in {"semantic", "hybrid", "random"} else 0.,
                      "selected_non_demo": len(chosen), "selected_item_tokens": packed_length(chosen, lengths, parents, shared),
                      "reference_item_tokens": reference_length, "actual_history_length_relative_error": match_error}
