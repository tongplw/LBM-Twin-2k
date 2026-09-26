"""Task-balanced scoring, paired participant intervals and train-only references."""
from collections import defaultdict

import numpy as np

from .analyze import grouped_scores, macro_interval
from .metrics import agreement_score


def score_prediction(example, prediction):
    rows = []
    if len(prediction["values"]) != len(example["atoms"]):
        raise ValueError("Prediction row count changed")
    for i, atom in enumerate(example["atoms"]):
        value = prediction["values"][i]
        valid = value is not None and np.isfinite(value)
        if valid and atom["lower"] is not None:
            valid = atom["lower"] <= value <= atom["upper"]
        if valid and atom["levels"]:
            valid = float(value).is_integer()
        bounded = atom["kind"] != "unbounded_numeric"
        p = prediction.get("probabilities", [None]*len(example["atoms"]))[i]
        row = {"pid": example["pid"], "key": atom["key"], "task_group": example["group"],
               "family": atom["family"], "kind": atom["kind"], "truth": atom["value"],
               "prediction": value if valid else None, "invalid": float(not valid),
               "exact": float(valid and value == atom["value"]) if bounded else None,
               "agreement": (agreement_score(atom["value"], value, atom["lower"], atom["upper"], atom["kind"])
                             if valid else 0.) if bounded else None,
               "absolute_error": abs(atom["value"]-value) if valid and not atom["levels"] else None,
               "nll": None, "brier": None, "probabilities": p}
        if atom["levels"] and p is not None:
            p = np.asarray(p, dtype=float)
            if len(p) != atom["levels"] or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
                raise ValueError("Invalid option distribution")
            y = int(atom["value"])-1
            row["nll"] = float(-np.log(max(p[y], 1e-12)))
            row["brier"] = float(np.sum((p-np.eye(len(p))[y])**2))
        rows.append(row)
    return rows


def summarize(rows, draws=2000):
    if not rows:
        raise ValueError("No scored responses")
    pids = sorted({r["pid"] for r in rows})
    groups = sorted({r["task_group"] for r in rows})
    result = {"participants": len(pids), "responses": len(rows), "task_groups": len(groups),
              "aggregation": "within participant/task, across participants, then equal task groups",
              "per_task": {}, "numeric_errors": {}}
    result["invalid_response_rate"] = float(np.mean([r["invalid"] for r in rows]))
    for field in ("exact", "agreement", "nll", "brier", "invalid"):
        eligible = sorted({r["task_group"] for r in rows if r.get(field) is not None})
        matrix = grouped_scores(rows, field, pids, eligible)
        result[field] = macro_interval(matrix, draws) if eligible else None
        result[field+"_responses"] = sum(r.get(field) is not None for r in rows)
        for group in eligible:
            result["per_task"].setdefault(group, {})[field] = float(np.nanmean(matrix[:, eligible.index(group)]))
    categorical = [r for r in rows if r["kind"] in {"binary", "ordinal", "nominal"}]
    result["probability_coverage"] = sum(r["nll"] is not None for r in categorical)/max(1, len(categorical))
    for key in sorted({r["key"] for r in rows if r["kind"].endswith("numeric")}):
        selected = [r for r in rows if r["key"] == key]
        values = [r["absolute_error"] for r in selected if r["absolute_error"] is not None]
        result["numeric_errors"][key] = {"valid": len(values), "attempted": len(selected),
            "mae": float(np.mean(values)) if values else None,
            "median_absolute_error": float(np.median(values)) if values else None}
    return result


def paired_difference(left, right, field, draws=2000):
    """Left minus right; negative NLL and positive agreement favor the left."""
    a = {(r["pid"], r["key"]): r for r in left}
    b = {(r["pid"], r["key"]): r for r in right}
    if a.keys() != b.keys() or len(a) != len(left) or len(b) != len(right):
        raise ValueError("Paired comparisons require identical unique response keys")
    differences = []
    for key, row in a.items():
        if (row[field] is None) != (b[key][field] is None):
            raise ValueError("Paired metric coverage differs")
        if row[field] is not None:
            differences.append({**row, "delta": row[field]-b[key][field]})
    pids = sorted({r["pid"] for r in differences})
    groups = sorted({r["task_group"] for r in differences})
    return macro_interval(grouped_scores(differences, "delta", pids, groups), draws)


def average_assignments(runs):
    """Average response metrics within person before bootstrapping repeated controls."""
    keyed = [{(r["pid"], r["key"]): r for r in run} for run in runs]
    if any(run.keys() != keyed[0].keys() for run in keyed):
        raise ValueError("Repeated controls differ in participant/response coverage")
    result = []
    for key, row in keyed[0].items():
        out = dict(row)
        for field in ("exact", "agreement", "nll", "brier", "invalid", "absolute_error"):
            values = [run[key][field] for run in keyed]
            out[field] = float(np.mean(values)) if all(v is not None for v in values) else None
        result.append(out)
    return result


class Baselines:
    def __init__(self, training):
        self.values, self.prices = defaultdict(list), defaultdict(list)
        for example in training:
            for atom in example["atoms"]:
                self.values[atom["key"]].append(atom["value"])
                if atom["price"] is not None:
                    self.prices[atom["key"]].append((atom["price"], atom["value"]))

    def predict(self, example, mode="price_aware"):
        predictions, probs = [], []
        for atom in example["atoms"]:
            values = np.asarray(self.values[atom["key"]])
            if not len(values):
                raise ValueError("Baseline has no training support")
            if mode == "human_retest":
                predictions.append(atom["earlier"])
                probs.append(None)
                continue
            k = atom["levels"]
            p = None
            if k:
                counts = np.bincount(values.astype(int), minlength=k+1)[1:k+1]
                p = (counts+1)/(len(values)+k)
                if mode == "price_aware" and atom["price"] is not None:
                    prices = np.asarray(self.prices[atom["key"]])
                    edges = np.unique(np.quantile(prices[:, 0], [.2, .4, .6, .8]))
                    bucket = np.searchsorted(edges, atom["price"], side="right")
                    selected = prices[np.searchsorted(edges, prices[:, 0], side="right") == bucket, 1]
                    local = np.bincount(selected.astype(int), minlength=k+1)[1:k+1]
                    p = (local+5*p)/(len(selected)+5)
                value = int(np.argmax(p))+1
            else:
                value = float(np.median(values))
            predictions.append(value)
            probs.append(p.tolist() if p is not None else None)
        return {"values": predictions, "probabilities": probs}


def temperature_scale(rows, temperature):
    result = []
    for row in rows:
        out = dict(row)
        if row["probabilities"] is not None:
            logits = np.log(np.maximum(row["probabilities"], 1e-12))/temperature
            p = np.exp(logits-logits.max())
            p /= p.sum()
            y = int(row["truth"])-1
            out.update(probabilities=p.tolist(), nll=float(-np.log(max(p[y], 1e-12))),
                       brier=float(np.sum((p-np.eye(len(p))[y])**2)))
        result.append(out)
    return result


def fit_temperature(rows):
    pids = sorted({r["pid"] for r in rows})
    groups = sorted({r["task_group"] for r in rows if r["nll"] is not None})
    best = (float("inf"), 1.)
    for t in np.exp(np.linspace(np.log(.1), np.log(10), 161)):
        scaled = temperature_scale(rows, t)
        score = float(np.nanmean(np.nanmean(grouped_scores(scaled, "nll", pids, groups), axis=0)))
        if score < best[0]:
            best = (score, float(t))
    return {"temperature": best[1], "development_macro_nll": best[0],
            "at_search_boundary": bool(np.isclose(best[1], .1) or np.isclose(best[1], 10.)),
            "search": "161 log-spaced values on [0.1, 10]; development only"}
