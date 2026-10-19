"""Network bootstrap intervals and descriptive split statistics."""

import random

import numpy as np


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def network_interval(
    values: dict[str, float], seed: int = 20260924, repetitions: int = 2000
) -> dict[str, float]:
    names = sorted(values)
    observed = mean([values[name] for name in names])
    rng = random.Random(seed)
    draws = [mean([values[rng.choice(names)] for _ in names]) for _ in range(repetitions)]
    return {"mean": observed, "low": percentile(draws, 0.025), "high": percentile(draws, 0.975)}


def split_statistics(x):
    a = np.array(x, float)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
        "min": float(a.min()),
        "max": float(a.max()),
        "values": a.tolist(),
    }
