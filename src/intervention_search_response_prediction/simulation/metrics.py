"""Scores on a fixed set of intervention-response probes."""

from __future__ import annotations

from collections.abc import Sequence


def probe_scores(
    predictions: tuple[tuple[float, ...], ...],
    truths: tuple[tuple[int, ...], ...],
) -> dict[str, float]:
    if not predictions or len(predictions) != len(truths):
        raise ValueError("one prediction vector per nonempty probe set is required")
    width = len(truths[0])
    if not width or any(len(p) != width or len(y) != width for p, y in zip(predictions, truths)):
        raise ValueError("probe vectors must have equal nonzero width")
    if any(not 0 <= p <= 1 for row in predictions for p in row):
        raise ValueError("probabilities must be in [0, 1]")
    if any(y not in (0, 1) for row in truths for y in row):
        raise ValueError("truths must be binary")
    count = len(truths)
    brier = sum((p - y) ** 2 for ps, ys in zip(predictions, truths) for p, y in zip(ps, ys)) / (
        count * width
    )
    constant = 0.0
    discrimination = 0.0
    variable = 0
    for r in range(width):
        mean_p = sum(row[r] for row in predictions) / count
        mean_y = sum(row[r] for row in truths) / count
        mean_py = sum(p[r] * y[r] for p, y in zip(predictions, truths)) / count
        constant += mean_y * (1 - mean_y)
        discrimination += 2 * (mean_py - mean_p * mean_y)
        variable += 0 < mean_y < 1
    constant /= width
    discrimination /= width
    return {
        "brier": brier,
        "constant_brier": constant,
        "excess_brier": brier - constant,
        "discrimination": discrimination,
        "variable_readout_fraction": variable / width,
    }


def unclamped_discrimination(
    predictions: Sequence[Sequence[float]],
    truths: Sequence[Sequence[int]],
    probes: Sequence[Sequence[Sequence[str | int]]],
    readouts: Sequence[str],
) -> tuple[float, float]:
    """Return D after omitting probes that directly clamp the scored readout.

    The second value is the fraction of original probe-readout pairs omitted.
    Readouts with fewer than two remaining probes contribute zero.
    """
    if len(predictions) != len(truths) or len(probes) != len(truths) or not readouts:
        raise ValueError("probe arrays must align")
    discrimination = 0.0
    omitted = 0
    for r, name in enumerate(readouts):
        keep = [i for i, action in enumerate(probes) if name not in {pair[0] for pair in action}]
        omitted += len(probes) - len(keep)
        if len(keep) >= 2:
            ps = [predictions[i][r] for i in keep]
            ys = [truths[i][r] for i in keep]
            count = len(keep)
            discrimination += 2 * (
                sum(p * y for p, y in zip(ps, ys)) / count - sum(ps) * sum(ys) / count**2
            )
    return discrimination / len(readouts), omitted / (len(probes) * len(readouts))
