"""Lossless question traversal and explicit normalization of repeated outcomes."""
from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

# Paper-aligned main groups; retain finer families for diagnostic reporting.
TASK_GROUPS = {
    "Base-rate neglect": "Base-rate neglect",
    "Outcome bias": "Outcome bias",
    "Sunk cost": "Sunk cost",
    "Allais choices": "Allais choices",
    "Disease framing": "Disease framing",
    "Conjunction": "Conjunction",
    "Anchoring: countries": "Anchoring",
    "Anchoring: height": "Anchoring",
    "Absolute vs relative": "Absolute vs relative",
    "Myside bias": "Myside bias",
    "Less is more": "Less is More (combined)",
    "Proportion dominance": "Less is More (combined)",
    "WTA / WTP": "WTA / WTP",
    "False consensus": "False consensus",
    "Affect heuristic": "Affect heuristic",
    "Omission bias": "Omission bias",
    "Probability matching": "Probability matching",
    "Ratio bias": "Ratio bias",
    "Pricing": "Pricing",
}


def task_group(family_name):
    """Fail closed if an outcome has no reviewed main-group mapping."""
    return TASK_GROUPS[family_name]

# Reviewed ordered scales in this pinned questionnaire, not a universal rule
# that every multiple-choice response has meaningful numeric distances.
ORDERED_QIDS = {f"QID{i}" for i in (
    157, 158, 159, 160, 161, 162, 171, 172, 173, 174, 175, 176, 177, 178,
    179, 189, 190, 191, 194, 195, 287, 288, 289, 291)}


def participants():
    for path in sorted((ROOT / "data/raw").glob("wave_persona_chunk_*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                row["pid"] = str(row["pid"])
                yield row


def question_records(serialized):
    """Preserve block occurrence and question position; QID alone is not a key."""
    blocks = json.loads(serialized) if isinstance(serialized, str) else serialized
    for bi, block in enumerate(blocks):
        for qi, question in enumerate(block.get("Questions", [])):
            yield (bi, qi, block.get("BlockName", "").strip(), question)


def make_splits(pids):
    ids = sorted(set(map(str, pids)), key=int)
    if len(ids) != 2058:
        raise ValueError(f"Expected 2,058 unique participants, got {len(ids)}")
    random.Random(42).shuffle(ids)
    # These records were opened in the preliminary investigation; never test.
    for replacement, pid in enumerate(("574", "2001")):
        index = ids.index(pid)
        ids[replacement], ids[index] = ids[index], ids[replacement]
    return {"train": sorted(ids[:1440], key=int),
            "dev": sorted(ids[1440:1749], key=int),
            "test": sorted(ids[1749:], key=int)}


def family(qid, block):
    if qid in {"QID287", "QID290"}: return "False consensus"
    if qid in {"QID288", "QID289"}: return "Affect heuristic"
    if qid == "QID291": return "Omission bias"
    if qid == "QID196": return "Ratio bias"
    names = [("Base-rate", "Base-rate neglect"), ("Disease", "Disease framing"),
             ("Linda", "Conjunction"), ("Outcome", "Outcome bias"),
             ("Anchoring - African", "Anchoring: countries"),
             ("Anchoring - redwood", "Anchoring: height"),
             ("Less is More", "Less is more"), ("Proportion", "Proportion dominance"),
             ("Sunk cost", "Sunk cost"), ("Absolute", "Absolute vs relative"),
             ("WTA/WTP", "WTA / WTP"), ("Allais", "Allais choices"),
             ("Myside", "Myside bias"), ("Probability", "Probability matching"),
             ("Product Preferences", "Pricing")]
    for prefix, value in names:
        if block.startswith(prefix): return value
    raise ValueError(f"Unmapped target family: {qid}, {block}")


@dataclass(frozen=True)
class Atom:
    key: str
    qid: str
    subid: str
    family: str
    condition: str
    kind: str
    value: float | None
    lower: float | None
    upper: float | None
    levels: int | None
    invalid: bool
    stimulus_hash: str
    price: float | None = None


def number(value):
    if value is None or str(value).strip() == "": return None
    try:
        result = float(str(value).replace(",", "").strip())
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def stimulus(q):
    """Allowlist, not blacklist: answers/metadata cannot enter model prompts."""
    names = ("QuestionID", "QuestionText", "QuestionType", "Options", "Rows",
             "RowsID", "Columns", "Statements", "StatementsID", "Range")
    return {name: q[name] for name in names if name in q}


def target_atoms(serialized):
    atoms = []
    seen = set()
    for bi, qi, block, q in question_records(serialized):
        qid, qt = q["QuestionID"], q["QuestionType"]
        if qt == "DB" or q.get("is_descriptive"): continue
        answer = q.get("Answers", {})
        fam = family(qid, block)
        digest = hashlib.sha256(json.dumps(stimulus(q), sort_keys=True).encode()).hexdigest()
        price = None
        if fam == "Pricing":
            match = re.search(r"priced at:\s*\$([0-9]+(?:\.[0-9]+)?)", q.get("QuestionText", ""))
            if not match: raise ValueError(f"Unparsed product price for {qid}")
            price = float(match[1])
        levels = None
        lo = hi = None
        if qt == "MC":
            if q.get("Settings", {}).get("Selector") in {"MAVR", "MAHR"}:
                raise ValueError("Target multi-select requires an explicit scorer")
            levels = len(q["Options"])
            kind = "binary" if levels == 2 else "ordinal"
            values, ids = [answer.get("SelectedByPosition")], ["0"]
            lo, hi = 1, levels
        elif qt == "Matrix":
            rows = q.get("Rows", q.get("Statements", []))
            ids = q.get("RowsID", q.get("StatementsID", [str(i + 1) for i in range(len(rows))]))
            values = answer.get("SelectedByPosition", [None] * len(ids))
            levels = len(q["Columns"])
            kind = "binary" if levels == 2 else "ordinal"
            lo, hi = 1, levels
        elif qt == "Slider":
            rows = q.get("Statements", [""])
            ids = q.get("StatementsID", [str(i + 1) for i in range(len(rows))])
            values = answer.get("Values", [None] * len(ids))
            kind = "bounded_numeric"
            lo, hi = q["Range"]["Min"], q["Range"]["Max"]
        elif qt == "TE":
            values, ids = [answer.get("Text")], ["0"]
            if qid in {"QID181", "QID182"}:
                kind, lo, hi = "bounded_numeric", 0, 20
            elif qid in {"QID164", "QID166", "QID168", "QID170"}:
                kind = "unbounded_numeric"
            else:
                raise ValueError(f"Unmapped target text item {qid}")
        else:
            raise ValueError(f"Unsupported target type {qt}")
        if kind == "ordinal" and qid not in ORDERED_QIDS:
            raise ValueError(f"Unreviewed categorical scale: {qid}; specify nominal versus ordered")
        if len(values) != len(ids):
            raise ValueError(f"Length mismatch: {qid}")
        for subid, raw in zip(ids, values):
            # Retest blocks have stable names; input persona records retain paths.
            key = f"{block}|{qid}|{subid}"
            if key in seen: raise ValueError(f"Duplicate target key {key}")
            seen.add(key)
            val = number(raw)
            invalid = raw is not None and str(raw).strip() != "" and val is None
            if val is not None and lo is not None:
                invalid = not lo <= val <= hi or (levels is not None and not val.is_integer())
            if invalid: val = None
            atoms.append(Atom(key, qid, str(subid), fam, block, kind, val, lo, hi,
                              levels, invalid, digest, price))
    return atoms
