"""Network-balanced summaries of complete paired language-agent episodes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..paths import EPISODES, NOTES, Layout
from ..statistics import mean, network_interval

FIELDS = (
    "brier_before",
    "brier_after",
    "brier_gain",
    "constant_brier",
    "excess_brier_before",
    "excess_brier_after",
    "discrimination_before",
    "discrimination_after",
    "discrimination_gain",
    "variable_readout_fraction",
    "goal_success",
    "observed_success",
    "tokens_input",
    "tokens_output",
    "cost",
)


def summarize(rows: list[dict]) -> dict:
    posthoc = [row for row in rows if not row.get("smoke") and row["condition"] == "summary_note"]
    rows = [
        dict(
            row,
            observed_success=int(row["first_success"] is not None),
            discrimination_gain=row["discrimination_after"] - row["discrimination_before"],
        )
        for row in rows
        if not row.get("smoke") and row["condition"] in ("history", "predictive")
    ]
    indexed = {(row["task_id"], row["label_condition"], row["condition"]): row for row in rows}
    result = {"episodes": len(rows), "named": {}, "anonymous": {}, "paired": {}}
    for labels in ("named", "anonymous"):
        tasks = sorted(
            {
                row["task_id"]
                for row in rows
                if row["label_condition"] == labels
                and all(
                    (row["task_id"], labels, cond) in indexed for cond in ("history", "predictive")
                )
            }
        )
        result[labels]["tasks"] = len(tasks)
        result[labels]["networks"] = len(
            {indexed[(task, labels, "history")]["model"] for task in tasks}
        )
        for condition in ("history", "predictive"):
            subset = [indexed[(task, labels, condition)] for task in tasks]
            by_network: dict[str, list[dict]] = defaultdict(list)
            for row in subset:
                by_network[row["model"]].append(row)
            result[labels][condition] = (
                {
                    field: network_interval(
                        {
                            name: mean([row[field] for row in group])
                            for name, group in by_network.items()
                        }
                    )
                    for field in FIELDS
                }
                if by_network
                else {}
            )
            result[labels][condition]["failures"] = {
                key: sum(row["failures"][key] for row in subset)
                for key in (
                    "invalid_action",
                    "invalid_probability",
                    "invalid_probe",
                    "invalid_final",
                )
            }
            result[labels][condition]["first_success_counts"] = {
                str(step): sum(row["first_success"] == step for row in subset)
                for step in range(1, 9)
            } | {"censored": sum(row["first_success"] is None for row in subset)}
            result[labels][condition]["first_success_cdf"] = (
                [
                    network_interval(
                        {
                            name: mean(
                                [
                                    int(
                                        row["first_success"] is not None
                                        and row["first_success"] <= step
                                    )
                                    for row in group
                                ]
                            )
                            for name, group in by_network.items()
                        },
                        seed=20261000 + step,
                    )
                    for step in range(1, 9)
                ]
                if by_network
                else []
            )
        paired: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for task in tasks:
            a = indexed[(task, labels, "predictive")]
            b = indexed[(task, labels, "history")]
            paired[a["model"]].append((a, b))
        result["paired"][labels] = (
            {
                field: network_interval(
                    {
                        name: mean([a[field] - b[field] for a, b in group])
                        for name, group in paired.items()
                    }
                )
                for field in FIELDS
            }
            if paired
            else {}
        )
        clean = {
            name: [
                (a, b)
                for a, b in group
                if a["failures"]["invalid_action"] == 0 and b["failures"]["invalid_action"] == 0
            ]
            for name, group in paired.items()
        }
        clean = {name: group for name, group in clean.items() if group}
        result["paired"][f"{labels}_no_assay_fallback"] = {
            "tasks": sum(map(len, clean.values())),
            "networks": len(clean),
            "metrics": {
                field: network_interval(
                    {
                        name: mean([a[field] - b[field] for a, b in group])
                        for name, group in clean.items()
                    }
                )
                for field in (
                    "brier_after",
                    "discrimination_after",
                    "discrimination_gain",
                    "goal_success",
                )
            }
            if clean
            else {},
        }
    common = sorted(
        {
            row["task_id"]
            for row in rows
            if all(
                (row["task_id"], label, cond) in indexed
                for label in ("named", "anonymous")
                for cond in ("history", "predictive")
            )
        }
    )
    result["paired"]["anonymization_tasks"] = len(common)
    result["matched"] = {"tasks": len(common)}
    for labels in ("named", "anonymous"):
        result["matched"][labels] = {}
        for condition in ("history", "predictive"):
            subset = [indexed[(task, labels, condition)] for task in common]
            by_network = defaultdict(list)
            for row in subset:
                by_network[row["model"]].append(row)
            result["matched"][labels][condition] = (
                {
                    field: network_interval(
                        {
                            name: mean([row[field] for row in group])
                            for name, group in by_network.items()
                        }
                    )
                    for field in FIELDS
                }
                if by_network
                else {}
            )
    for condition in ("history", "predictive"):
        paired = defaultdict(list)
        for task in common:
            a = indexed[(task, "anonymous", condition)]
            b = indexed[(task, "named", condition)]
            paired[a["model"]].append((a, b))
        result["paired"][f"anonymous_minus_named_{condition}"] = (
            {
                field: network_interval(
                    {
                        name: mean([a[field] - b[field] for a, b in group])
                        for name, group in paired.items()
                    }
                )
                for field in FIELDS
            }
            if paired
            else {}
        )
    if posthoc:
        posthoc = [
            dict(
                row,
                observed_success=int(row["first_success"] is not None),
                discrimination_gain=row["discrimination_after"] - row["discrimination_before"],
            )
            for row in posthoc
        ]
        by_network: dict[str, list[dict]] = defaultdict(list)
        for row in posthoc:
            by_network[row["model"]].append(row)
        result["posthoc"] = {
            "condition": "summary_note",
            "tasks": len(posthoc),
            "summary": {
                field: network_interval(
                    {
                        name: mean([row[field] for row in group])
                        for name, group in by_network.items()
                    }
                )
                for field in FIELDS
            },
            "failures": {
                key: sum(row["failures"][key] for row in posthoc)
                for key in (
                    "invalid_action",
                    "invalid_probability",
                    "invalid_probe",
                    "invalid_final",
                )
            },
        }
        for comparator in ("history", "predictive"):
            pairs: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
            for row in posthoc:
                original = indexed.get((row["task_id"], "named", comparator))
                if original is not None:
                    pairs[row["model"]].append((row, original))
            result["posthoc"][f"minus_{comparator}"] = (
                {
                    field: network_interval(
                        {
                            name: mean([a[field] - b[field] for a, b in group])
                            for name, group in pairs.items()
                        }
                    )
                    for field in FIELDS
                }
                if pairs
                else {}
            )
    return result


def classical_reference(rows: list[dict], task_ids: set[str]) -> dict:
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["task_id"] in task_ids:
            by_strategy[row["strategy"]].append(
                dict(
                    row,
                    observed_success=int(row["first_success"] is not None),
                    discrimination_gain=row["discrimination_after"] - row["discrimination_before"],
                )
            )
    result = {}
    for strategy, subset in by_strategy.items():
        by_network: dict[str, list[dict]] = defaultdict(list)
        for row in subset:
            by_network[row["model"]].append(row)
        result[strategy] = {
            field: network_interval(
                {name: mean([row[field] for row in group]) for name, group in by_network.items()}
            )
            for field in FIELDS
            if field in subset[0]
        }
    return result


def write_figures(summary: dict, output_dir: Path, root: Path | None = None) -> None:
    """Draw paired language-agent comparisons from saved scores."""
    from ..paths import writable_output

    output_dir = writable_output(output_dir, root)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..plotting import (
        BRIER,
        BRIER_LIGHT,
        DISCRIMINATION,
        DISCRIMINATION_BG,
        DISCRIMINATION_LIGHT,
        GRID,
        MUTED,
        TARGET,
        configure,
        save,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    configure()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.08))
    for ax, fields, title, ylim, metric_color, light_color in (
        (
            axes[0],
            ("brier_before", "brier_after"),
            "Held-out Brier (lower better)",
            (0.085, 0.35),
            BRIER,
            BRIER_LIGHT,
        ),
        (
            axes[1],
            ("discrimination_before", "discrimination_after"),
            "Permutation gap",
            (0.02, 0.19),
            DISCRIMINATION,
            DISCRIMINATION_LIGHT,
        ),
    ):
        for condition, label in (("history", "No note"), ("predictive", "Hypothesis note")):
            estimates = [summary["named"][condition][field] for field in fields]
            means = [estimate["mean"] for estimate in estimates]
            errors = [
                [mean - estimate["low"] for mean, estimate in zip(means, estimates)],
                [estimate["high"] - mean for mean, estimate in zip(means, estimates)],
            ]
            predictive = condition == "predictive"
            ax.errorbar(
                (0, 1),
                means,
                yerr=errors,
                color=metric_color if predictive else light_color,
                marker="o" if predictive else "s",
                linestyle="-" if predictive else "--",
                markersize=4.4,
                capsize=2.5,
                linewidth=1.5,
            )
            ax.text(1.08, means[-1], label, fontsize=7, color=metric_color, va="center")
        ax.set(
            xticks=(0, 1),
            xticklabels=("Before", "After"),
            xlim=(-0.12, 1.95),
            ylim=ylim,
            title=title,
        )
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.title.set_fontsize(8.5)
        ax.title.set_color(metric_color)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.87, bottom=0.21, wspace=0.28)
    save(fig, output_dir / "probe-scores.pdf")
    plt.close(fig)

    paired = summary["paired"]["named"]
    no_fallback = summary["paired"]["named_no_assay_fallback"]["metrics"]
    fig, axes = plt.subplots(
        1, 3, figsize=(5.5, 2.05), gridspec_kw={"width_ratios": [1.0, 1.15, 0.8]}
    )
    panels = (
        (
            axes[0],
            "Brier difference",
            (("brier_after", r"$B_8$", 2), ("brier_gain", r"$\Delta B$", 1)),
            (-0.09, 0.11),
            (-0.08, 0, 0.08),
            1,
            BRIER,
        ),
        (
            axes[1],
            "Permutation-gap difference",
            (
                ("discrimination_after", r"$D_8$", 2),
                ("discrimination_gain", r"$\Delta D$", 1),
                ("discrimination_after", "No fallback", 0),
            ),
            (-0.035, 0.077),
            (-0.02, 0, 0.04),
            1,
            DISCRIMINATION,
        ),
        (
            axes[2],
            "Goal success (pp)",
            (("goal_success", "Final", 2),),
            (-6, 24),
            (0, 10, 20),
            100,
            TARGET,
        ),
    )
    for ax, title, rows, limits, ticks, scale, metric_color in panels:
        for field, label, y in rows:
            effect = no_fallback[field] if label == "No fallback" else paired[field]
            mean, low, high = [scale * effect[k] for k in ("mean", "low", "high")]
            color = DISCRIMINATION_LIGHT if label == "No fallback" else metric_color
            ax.plot((low, high), (y, y), color=color, linewidth=1.8, solid_capstyle="round")
            ax.plot(
                mean,
                y,
                marker="s" if label == "No fallback" else "o",
                color=color,
                markerfacecolor="white" if label == "No fallback" else color,
                markersize=4.4,
                zorder=3,
            )
        ax.axvline(0, color=MUTED, linewidth=0.75, linestyle=(0, (2, 2)))
        ax.set(
            xlim=limits,
            ylim=(-0.5, 2.5),
            xticks=ticks,
            yticks=[row[2] for row in rows],
            yticklabels=[row[1] for row in rows],
            title=title,
        )
        ax.set_xticklabels([f"{tick:g}" for tick in ticks])
        ax.grid(axis="x", color=GRID, linewidth=0.55)
        ax.tick_params(axis="both", labelsize=6.7)
        ax.title.set_fontsize(8)
        ax.title.set_color(metric_color)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.85, bottom=0.20, wspace=0.52)
    save(fig, output_dir / "note-effects.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 1.90))
    fields = ("discrimination_before", "discrimination_after", "discrimination_gain")
    labels = (r"$D_0$", r"$D_8$", r"$\Delta D$")
    axis.axhspan(0.52, 1.48, color=DISCRIMINATION_BG, zorder=0)
    for condition, offset, label, color, marker in (
        ("history", 0.12, "No note", DISCRIMINATION_LIGHT, "s"),
        ("predictive", -0.12, "Hypothesis note", DISCRIMINATION, "o"),
    ):
        effects = summary["paired"][f"anonymous_minus_named_{condition}"]
        y = [2 - i + offset for i in range(3)]
        means = [effects[field]["mean"] for field in fields]
        errors = [
            [value - effects[field]["low"] for value, field in zip(means, fields)],
            [effects[field]["high"] - value for value, field in zip(means, fields)],
        ]
        axis.errorbar(
            means,
            y,
            xerr=errors,
            fmt=marker,
            capsize=2.5,
            markersize=4.4,
            color=color,
            linewidth=1.5,
            label=label,
        )
    axis.axvline(0, color=MUTED, linewidth=0.75, linestyle=(0, (2, 2)))
    axis.set(
        xlim=(-0.105, 0.065),
        ylim=(-0.48, 2.50),
        yticks=(2, 1, 0),
        yticklabels=labels,
        xlabel="Opaque IDs minus biological names: permutation gap",
    )
    axis.grid(axis="x", color=GRID, linewidth=0.55)
    axis.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=2)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.78, bottom=0.24)
    save(fig, output_dir / "node-label-effects.pdf")
    plt.close(fig)


def analyze(episodes_path: Path, classical_path: Path) -> dict:
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    summary = summarize(rows)
    paired_ids = {
        row["task_id"]
        for row in rows
        if not row.get("smoke")
        and row["label_condition"] == "named"
        and all(
            any(
                other["task_id"] == row["task_id"]
                and other["label_condition"] == "named"
                and other["condition"] == condition
                for other in rows
            )
            for condition in ("history", "predictive")
        )
    }
    classical_rows = [json.loads(line) for line in classical_path.read_text().splitlines()]
    summary["classical_on_named_tasks"] = classical_reference(classical_rows, paired_ids)
    return summary


def main() -> None:
    import argparse

    from ..io import write_json
    from ..paths import repository_root, writable_output

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = repository_root()
    args.input = args.input or Layout(root).notes
    args.output = args.output or root / f"outputs/analysis/{NOTES}"
    output = writable_output(args.output)
    summary = analyze(args.input / EPISODES, Layout(root).baselines / EPISODES)
    write_json(output / "summary.json", summary)
    write_figures(summary, output / "figures")
    print(json.dumps({"episodes": summary["episodes"], "output": str(output)}))


if __name__ == "__main__":
    main()
