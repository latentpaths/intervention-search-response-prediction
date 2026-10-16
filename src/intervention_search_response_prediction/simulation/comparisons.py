"""Matched-task and association checks for the paper, from saved episodes."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from ..io import read_jsonl
from ..paths import EPISODES, NOTES, Layout, repository_root, writable_output
from ..statistics import mean, network_interval, percentile


def _normalized(row: dict) -> float | None:
    reference = row["constant_brier"]
    if reference == 0:
        if abs(row["discrimination_after"]) > 1e-12:
            raise ValueError("nonzero discrimination with constant probe outcomes")
        return None
    value = row["discrimination_after"] / (2 * reference)
    return 0.0 if abs(value) < 1e-12 else value


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and math.isclose(
            values[order[position]], values[order[end]], abs_tol=1e-12
        ):
            end += 1
        rank = (position + end - 1) / 2 + 1
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _spearman(rows: list[dict]) -> float | None:
    xs = _ranks([row["discrimination_after"] for row in rows])
    ys = _ranks([row["goal_success"] for row in rows])
    xbar, ybar = mean(xs), mean(ys)
    xx = sum((x - xbar) ** 2 for x in xs)
    yy = sum((y - ybar) ** 2 for y in ys)
    if xx == 0 or yy == 0:
        return None
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / math.sqrt(xx * yy)


def _cluster_spearman(groups: dict[str, list[dict]], seed: int = 20260924) -> dict | None:
    names = sorted(groups)
    observed = _spearman([row for name in names for row in groups[name]])
    if observed is None:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        sample = [row for _ in names for row in groups[rng.choice(names)]]
        coefficient = _spearman(sample)
        if coefficient is not None:
            draws.append(coefficient)
    return {
        "mean": observed,
        "low": percentile(draws, 0.025),
        "high": percentile(draws, 0.975),
        "valid_draws": len(draws),
        "episodes": sum(map(len, groups.values())),
    }


def _mean_cluster_spearman(
    conditions: dict[str, dict[str, list[dict]]], seed: int = 20260924
) -> dict:
    names = sorted(set.intersection(*(set(groups) for groups in conditions.values())))

    def estimate(selected: list[str]) -> float:
        return mean(
            [
                _spearman([row for name in selected for row in groups[name]])
                for groups in conditions.values()
            ]
        )  # type: ignore[list-item]

    observed = estimate(names)
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        selected = [rng.choice(names) for _ in names]
        try:
            draws.append(estimate(selected))
        except TypeError:
            # A degenerate binary-success bootstrap has no rank correlation.
            continue
    return {
        "mean": observed,
        "low": percentile(draws, 0.025),
        "high": percentile(draws, 0.975),
        "valid_draws": len(draws),
        "conditions": list(conditions),
    }


def _normalized_summary(rows: list[dict]) -> dict:
    by_model: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _normalized(row)
        if value is not None:
            by_model[row["model"]].append(value)
    return {
        "estimate": network_interval({name: mean(values) for name, values in by_model.items()}),
        "defined_tasks": sum(len(values) for values in by_model.values()),
        "total_tasks": len(rows),
    }


def analyze(classical: list[dict], language: list[dict]) -> dict:
    language = [row for row in language if not row.get("smoke")]
    named = {
        (row["task_id"], row["condition"]): row
        for row in language
        if row["label_condition"] == "named" and row["condition"] in ("history", "predictive")
    }
    anonymous = {
        (row["task_id"], row["condition"]): row
        for row in language
        if row["label_condition"] == "anonymous" and row["condition"] in ("history", "predictive")
    }
    named_ids = {
        task
        for task, condition in named
        if condition == "history" and (task, "predictive") in named
    }
    matched_ids = {
        task
        for task in named_ids
        if all((task, condition) in anonymous for condition in ("history", "predictive"))
    }
    if len(named_ids) != len({named[(task, "history")]["model"] for task in named_ids}):
        raise ValueError("named selection must contain one task per network")
    if len(matched_ids) != len({named[(task, "history")]["model"] for task in matched_ids}):
        raise ValueError("matched selection must contain one task per network")
    result: dict = {
        "task_counts": {"named": len(named_ids), "matched": len(matched_ids)},
        "classical": {},
        "language": {},
        "association": {},
    }
    for subset, ids in (
        ("all", {row["task_id"] for row in classical}),
        ("named", named_ids),
        ("matched", matched_ids),
    ):
        result["classical"][subset] = {}
        for strategy in sorted({row["strategy"] for row in classical}):
            selected = [
                row for row in classical if row["task_id"] in ids and row["strategy"] == strategy
            ]
            by_network: dict[str, list[dict]] = defaultdict(list)
            for row in selected:
                by_network[row["model"]].append(row)
            result["classical"][subset][strategy] = {
                field: network_interval(
                    {
                        name: mean([row[field] for row in group])
                        for name, group in by_network.items()
                    }
                )
                for field in ("brier_after", "discrimination_after", "goal_success")
            } | {"normalized_discrimination": _normalized_summary(selected)}
    classical_index = {(row["task_id"], row["strategy"]): row for row in classical}
    for task in named_ids:
        reference = named[(task, "history")]["constant_brier"]
        for strategy in result["classical"]["named"]:
            if not math.isclose(
                classical_index[(task, strategy)]["constant_brier"], reference, abs_tol=1e-12
            ):
                raise ValueError(f"probe reference differs for {task}")
    result["language_minus_active_ridge"] = {}
    for condition in ("history", "predictive"):
        result["language_minus_active_ridge"][condition] = {
            field: network_interval(
                {
                    named[(task, condition)]["model"]: named[(task, condition)][field]
                    - classical_index[(task, "active-ridge")][field]
                    for task in named_ids
                }
            )
            for field in ("brier_after", "discrimination_after", "goal_success")
        }
    for subset, ids in (("named", named_ids), ("matched", matched_ids)):
        result["language"][subset] = {}
        for condition in ("history", "predictive"):
            selected = [named[(task, condition)] for task in sorted(ids)]
            result["language"][subset][condition] = {
                "normalized_discrimination": _normalized_summary(selected)
            }
    result["language"]["anonymous"] = {
        condition: {
            "normalized_discrimination": _normalized_summary(
                [anonymous[(task, condition)] for task in sorted(matched_ids)]
            )
        }
        for condition in ("history", "predictive")
    }
    posthoc_rows = [
        row
        for row in language
        if row["condition"] == "summary_note" and row["label_condition"] == "named"
    ]
    if posthoc_rows:
        result["language"]["posthoc"] = {
            "normalized_discrimination": _normalized_summary(posthoc_rows)
        }
    interaction_by_model = {
        named[(task, "history")]["model"]: (
            (
                anonymous[(task, "predictive")]["discrimination_after"]
                - named[(task, "predictive")]["discrimination_after"]
            )
            - (
                anonymous[(task, "history")]["discrimination_after"]
                - named[(task, "history")]["discrimination_after"]
            )
        )
        for task in matched_ids
    }
    result["label_prompt_interaction"] = network_interval(interaction_by_model)
    variable_groups = {}
    for strategy in sorted({row["strategy"] for row in classical}):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in classical:
            if row["strategy"] == strategy:
                groups[row["model"]].append(row)
        result["association"][strategy] = _cluster_spearman(groups)
        if result["association"][strategy] is not None:
            variable_groups[strategy] = groups
    result["association"]["classical_variable_mean"] = _mean_cluster_spearman(variable_groups)
    language_groups: dict[str, dict[str, list[dict]]] = {
        condition: defaultdict(list) for condition in ("history", "predictive")
    }
    for task in named_ids:
        for condition in language_groups:
            row = named[(task, condition)]
            language_groups[condition][row["model"]].append(row)
    for condition, groups in language_groups.items():
        result["association"][f"named_{condition}"] = _cluster_spearman(groups)
    result["association"]["named_condition_mean"] = _mean_cluster_spearman(language_groups)
    pooled: dict[str, list[dict]] = defaultdict(list)
    for groups in language_groups.values():
        for model, rows in groups.items():
            pooled[model].extend(rows)
    result["association"]["named_pooled_78"] = _cluster_spearman(pooled)
    return result


def main() -> None:
    import argparse

    from ..io import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = repository_root()
    args.input = args.input or Layout(root).notes
    args.output = args.output or root / f"outputs/analysis/{NOTES}"
    output = writable_output(args.output)
    result = analyze(
        read_jsonl(Layout(root).baselines / EPISODES),
        read_jsonl(args.input / EPISODES),
    )
    write_json(output / "comparisons.json", result)
    print(json.dumps({"task_counts": result["task_counts"], "output": str(output)}))


if __name__ == "__main__":
    main()
