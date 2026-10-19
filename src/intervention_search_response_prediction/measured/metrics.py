"""Continuous response scores and episode measurements."""

import numpy as np


def continuous_scores(prediction, truth):
    p = np.asarray(prediction, float)
    y = np.asarray(truth, float)
    if (
        p.shape != y.shape
        or p.ndim != 2
        or len(p) < 2
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("invalid score arrays")
    return {
        "mse": float(np.mean((p - y) ** 2)),
        "discrimination": float(2 * np.mean((p - p.mean(axis=0)) * (y - y.mean(axis=0)))),
    }


def result(t, agent, mode, before, after, history, final, timings, elapsed, **extra):
    truth = t.values(t.probes)
    b = continuous_scores(before, truth)
    a = continuous_scores(after, truth)
    row = {
        "task_id": t.id,
        "dataset": t.dataset,
        "screen": t.id.split(":")[0],
        "seed": t.seed,
        "agent": agent,
        "mode": mode,
        "units": t.units,
        "before": b,
        "after": a,
        "mse_gain": b["mse"] - a["mse"],
        "discrimination_gain": a["discrimination"] - b["discrimination"],
        "history": history,
        "probes": list(t.probes),
        "probe_before": before,
        "probe_after": after,
        "probe_truth": truth.tolist(),
        "final": final,
        "decision_seconds": timings,
        "episode_seconds": elapsed,
        **extra,
    }
    if t.search:
        values = t.values(t.candidates)[:, 0]
        effect = t.records[final]["value"][0]
        row.update(
            final_effect=effect,
            regret=float(max(values) - effect),
            best_observed_curve=np.maximum.accumulate([r["value"][0] for r in history]).tolist(),
        )
    else:
        additive = t.priors(t.probes)
        row["additive_mse"] = float(np.mean((additive - truth) ** 2))
        row["interaction_residual"] = continuous_scores(
            np.array(after) - additive, truth - additive
        )
    return row
