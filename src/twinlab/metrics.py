"""Transparent scoring; uncertainty is clustered at participant level."""
import numpy as np


def agreement_score(truth, prediction, lower, upper, kind):
    if kind in {"binary", "nominal"}:
        return float(truth == prediction)
    if lower is None or upper is None:
        return None
    return 1.0 - abs(truth - prediction) / (upper - lower)


def bootstrap_mean(values, draws=2000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values): return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for start in range(0, draws, 100):
        n = min(100, draws - start)
        indices = rng.integers(0, len(values), size=(n, len(values)))
        boot[start:start+n] = values[indices].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi), "n": len(values)}


def categorical_metrics(probabilities, targets):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(targets, dtype=int)
    if p.ndim != 2 or len(p) != len(y) or not len(y):
        raise ValueError("Expected a nonempty N x K probability matrix")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Invalid probability distribution")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("Target index outside valid options")
    onehot = np.eye(p.shape[1])[y]
    return {"nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
            "brier": float(((p-onehot)**2).sum(axis=1).mean()),
            "accuracy": float((p.argmax(axis=1) == y).mean())}
