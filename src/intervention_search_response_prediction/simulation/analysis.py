"""Network-level summaries and publication figures for episode JSONL files."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..io import read_jsonl
from ..statistics import mean, network_interval


def summarize(records: list[dict]) -> dict:
    strategies = sorted({record["strategy"] for record in records})
    networks = sorted({record["model"] for record in records})
    tasks = sorted({record["task_id"] for record in records})
    result: dict = {
        "networks": len(networks),
        "tasks": len(tasks),
        "episodes": len(records),
        "strategies": {},
    }
    for strategy in strategies:
        subset = [record for record in records if record["strategy"] == strategy]
        by_network: dict[str, list[dict]] = defaultdict(list)
        for record in subset:
            by_network[record["model"]].append(record)
        metrics = {}
        for field in (
            "brier_before",
            "brier_after",
            "brier_gain",
            "constant_brier",
            "excess_brier_before",
            "excess_brier_after",
            "discrimination_before",
            "discrimination_after",
            "variable_readout_fraction",
            "prospective_brier",
            "goal_success",
        ):
            network_values = {
                name: mean([row[field] for row in rows]) for name, rows in by_network.items()
            }
            metrics[field] = network_interval(network_values)
        curve = []
        for step in range(9):
            network_values = {
                name: mean([row["learning_curve"][step] for row in rows])
                for name, rows in by_network.items()
            }
            curve.append(network_interval(network_values, seed=20260924 + step))
        metrics["learning_curve"] = curve
        first_success_curve = []
        for step in range(1, 9):
            network_values = {
                name: mean(
                    [
                        int(row["first_success"] is not None and row["first_success"] <= step)
                        for row in rows
                    ]
                )
                for name, rows in by_network.items()
            }
            first_success_curve.append(network_interval(network_values, seed=20261000 + step))
        metrics["first_success_cdf"] = first_success_curve
        metrics["observed_success_rate"] = first_success_curve[-1]
        metrics["n_episodes"] = len(subset)
        result["strategies"][strategy] = metrics
    indexed = {(row["task_id"], row["strategy"]): row for row in records}
    for left, right, label in (
        ("graph-ridge-observed", "graph-mean", "ridge_observed_minus_mean"),
        ("graph-ridge", "graph-ridge-observed", "predicted_minus_observed_final"),
        ("active-ridge", "graph-ridge", "active_minus_graph_ridge"),
    ):
        paired = defaultdict(list)
        for task_id in tasks:
            a, b = indexed[(task_id, left)], indexed[(task_id, right)]
            paired[a["model"]].append((a, b))
        result[label] = {}
        for field in (
            "brier_after",
            "brier_gain",
            "excess_brier_after",
            "discrimination_after",
            "goal_success",
        ):
            network_values = {
                name: mean([a[field] - b[field] for a, b in pairs])
                for name, pairs in paired.items()
            }
            result[label][field] = network_interval(network_values)
    return result


def write_figures(summary: dict, output_dir: Path, root: Path | None = None) -> None:
    from ..paths import writable_output

    output_dir = writable_output(output_dir, root)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..plotting import (
        BRIER,
        BRIER_LIGHT,
        DISCRIMINATION,
        GRID,
        MUTED,
        TARGET,
        TARGET_LIGHT,
        configure,
        save,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    configure()

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(5.5, 2.28), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    left_series = (
        ("random-half", "Fixed $1/2$", MUTED, "--"),
        ("random-mean", "Random + mean", BRIER_LIGHT, "-"),
        ("graph-mean", "Graph-ranked + mean", "#527795", "-."),
        ("graph-ridge-observed", "Graph-ranked + ridge", BRIER, "--"),
        ("active-ridge", "Uncertainty + ridge", "#1E6F99", "-"),
    )
    for name, label, color, style in left_series:
        curve = summary["strategies"][name]["learning_curve"]
        x = list(range(9))
        mean = [point["mean"] for point in curve]
        left.plot(x, mean, color=color, linewidth=1.55, linestyle=style)
        if name != "random-half":
            left.fill_between(
                x,
                [point["low"] for point in curve],
                [point["high"] for point in curve],
                color=color,
                alpha=0.075,
                linewidth=0,
            )
        left.text(
            8.22,
            mean[-1],
            label,
            fontsize=6.55,
            color=MUTED if name == "random-half" else BRIER,
            va="center",
        )
    left.set(
        xlim=(0, 13.0),
        ylim=(0.14, 0.325),
        xticks=(0, 2, 4, 6, 8),
        xlabel="Assays",
        ylabel="Held-out Brier (lower better)",
    )
    left.set_title("Response prediction", fontsize=8.7, color=BRIER, pad=5)
    left.grid(axis="y", color=GRID, linewidth=0.6)

    right_series = (
        ("random-mean", "Random assays", TARGET_LIGHT, -0.017, "-"),
        ("graph-mean", "Graph-ranked", "#C07857", -0.006, "--"),
        ("active-ridge", "Uncertainty", TARGET, 0.013, "-"),
    )
    for name, label, color, offset, style in right_series:
        curve = summary["strategies"][name]["first_success_cdf"]
        x = list(range(1, 9))
        mean = [100 * point["mean"] for point in curve]
        right.plot(x, mean, color=color, linewidth=1.65, linestyle=style)
        right.fill_between(
            x,
            [100 * point["low"] for point in curve],
            [100 * point["high"] for point in curve],
            color=color,
            alpha=0.075,
            linewidth=0,
        )
        right.text(8.19, mean[-1] + 100 * offset, label, fontsize=6.7, color=TARGET, va="center")
    right.set(
        xlim=(1, 13.0),
        ylim=(0, 82),
        xticks=(1, 2, 4, 6, 8),
        xlabel="Assays",
        ylabel="Target seen by assay $t$ (%)",
    )
    right.set_title("Target observed", fontsize=8.7, color=TARGET, pad=5)
    right.grid(axis="y", color=GRID, linewidth=0.6)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.23, wspace=0.42)
    save(fig, output_dir / "learning-curves.pdf")
    plt.close(fig)

    contrasts = (
        ("brier_after", "Held-out Brier", 1, (-0.074, 0.014), (-0.06, -0.03, 0), BRIER),
        (
            "discrimination_after",
            "Permutation gap $D_8$",
            1,
            (-0.009, 0.035),
            (0, 0.02),
            DISCRIMINATION,
        ),
        ("goal_success", "Goal success (pp)", 100, (-12, 3.5), (-10, -5, 0), TARGET),
    )
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75))
    for ax, (field, title, scale, limits, ticks, metric_color) in zip(axes, contrasts):
        for y, key in ((1, "ridge_observed_minus_mean"), (0, "predicted_minus_observed_final")):
            effect = summary[key][field]
            mean, low, high = [scale * effect[k] for k in ("mean", "low", "high")]
            color = MUTED if low == high else metric_color
            ax.plot((low, high), (y, y), color=color, linewidth=1.8, solid_capstyle="round")
            ax.plot(mean, y, marker="o", markersize=4.2, color=color, zorder=3)
            if low == high:
                ax.plot(
                    mean,
                    y,
                    marker="o",
                    markersize=5.8,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    zorder=4,
                )
        ax.axvline(0, color=MUTED, linewidth=0.75, linestyle=(0, (2, 2)))
        ax.set(xlim=limits, ylim=(-0.45, 1.45), xticks=ticks, yticks=(1, 0), title=title)
        ax.grid(axis="x", color=GRID, linewidth=0.55)
        ax.tick_params(axis="both", labelsize=6.5)
        ax.title.set_fontsize(8)
        ax.title.set_color(metric_color)
    axes[0].set_yticklabels(("Change predictor", "Change final rule"), fontsize=7)
    for ax in axes[1:]:
        ax.set_yticklabels(())
    fig.subplots_adjust(left=0.25, right=0.99, top=0.82, bottom=0.25, wspace=0.35)
    save(fig, output_dir / "component-effects.pdf")
    plt.close(fig)


def write_analysis(
    episodes_path: Path, output_dir: Path, summary_path: Path, root: Path | None = None
) -> dict:
    from ..paths import writable_output

    output_dir = writable_output(output_dir, root)
    summary_path = writable_output(summary_path, root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(episodes_path)
    summary = summarize(records)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_figures(summary, output_dir, root)
    return summary
