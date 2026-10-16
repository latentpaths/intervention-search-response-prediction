"""Summaries and result plots for retained measured-response experiments."""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from pathlib import Path

import matplotlib

from ..paths import BASELINE_EPISODES, INFERENCE_LEDGER, MEASURED, PAIRED_BLOCKS, Layout

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..io import write_json
from ..paths import repository_root, writable_output
from ..statistics import split_statistics
from .ledger import reconcile
from .tasks import tasks

LABELS = {
    "random_mean": "Random assays + mean",
    "ridge_random": "Random assays + ridge",
    "ridge_active": "Uncertainty + ridge",
    "additive": "Sum of singles",
    "summary": "LLM: Observation note",
    "hypothesis": "LLM: Hypothesis note",
}


MODES = {
    "classical": "Reference",
    "fixed": "Same assays",
    "adaptive": "LLM-selected",
    "shuffled": "Shuffled feedback",
}


COLORS = {
    "random_mean": "#7D8791",
    "ridge_random": "#27465F",
    "ridge_active": "#147D78",
    "additive": "#8A6E46",
    "summary": "#7D6093",
    "hypothesis": "#B45C3C",
}


def label(screen):
    if screen == "norman":
        return "Norman K562"
    return (
        screen.replace("schmidt_", "")
        .replace("Prim.", "Primary ")
        .replace("Pri.", "Primary ")
        .replace("Supp.", "Supplementary ")
        .replace("CRa.", "CRISPRa ")
        .replace("CRi.", "CRISPRi ")
        .replace(".", " ")
    )


def collect(input_dir=None, data_dir=None):
    input_dir = Path(input_dir) if input_dir is not None else Layout(repository_root()).measured
    rows = json.loads((input_dir / BASELINE_EPISODES).read_text())
    blocks = (
        [json.loads(x) for x in (input_dir / PAIRED_BLOCKS).read_text().splitlines()]
        if (input_dir / PAIRED_BLOCKS).exists()
        else []
    )
    for b in blocks:
        rows.extend(b["episodes"])
    groups = defaultdict(list)
    for r in rows:
        groups[(r["screen"], r["mode"], r["agent"])].append(r)
    summary = []
    for (screen, mode, agent), rs in sorted(groups.items()):
        s = {"screen": screen, "mode": mode, "agent": agent, "n": len(rs)}
        for key, fn in {
            "mse_before": lambda r: r["before"]["mse"],
            "mse_after": lambda r: r["after"]["mse"],
            "mse_gain": lambda r: r["mse_gain"],
            "discrimination_before": lambda r: r["before"]["discrimination"],
            "discrimination_after": lambda r: r["after"]["discrimination"],
            "discrimination_gain": lambda r: r["discrimination_gain"],
        }.items():
            s[key] = split_statistics([fn(r) for r in rs])
        for key in ("final_effect", "regret", "additive_mse"):
            if key in rs[0]:
                s[key] = split_statistics([r[key] for r in rs])
        if "interaction_residual" in rs[0]:
            s["residual_discrimination"] = split_statistics(
                [r["interaction_residual"]["discrimination"] for r in rs]
            )
        if "charged_usd" in rs[0]:
            s["episode_api_cost_usd"] = split_statistics([r["charged_usd"] for r in rs])
            s["original_api_seconds"] = split_statistics(
                [v for r in rs for v in r["original_call_seconds"]]
            )
        s["episode_wall_seconds"] = split_statistics([r["episode_seconds"] for r in rs])
        s["decision_latency"] = split_statistics([v for r in rs for v in r["decision_seconds"]])
        s["decision_latency"]["p95"] = float(
            np.percentile([v for r in rs for v in r["decision_seconds"]], 95)
        )
        s["decision_latency"]["median"] = float(
            np.median([v for r in rs for v in r["decision_seconds"]])
        )
        summary.append(s)
    paired = []
    for screen in sorted({r["screen"] for r in rows}):
        for mode in ("fixed", "adaptive", "shuffled"):
            a = {
                r["task_id"]: r
                for r in rows
                if r["screen"] == screen and r["mode"] == mode and r["agent"] == "summary"
            }
            b = {
                r["task_id"]: r
                for r in rows
                if r["screen"] == screen and r["mode"] == mode and r["agent"] == "hypothesis"
            }
            ids = sorted(a.keys() & b.keys())
            if not ids:
                continue
            p = {
                "screen": screen,
                "mode": mode,
                "n": len(ids),
                "direction": "hypothesis minus summary",
            }
            for key, fn in {
                "mse_after": lambda r: r["after"]["mse"],
                "mse_gain": lambda r: r["mse_gain"],
                "discrimination_after": lambda r: r["after"]["discrimination"],
            }.items():
                p[key] = split_statistics([fn(b[i]) - fn(a[i]) for i in ids])
            if "regret" in a[ids[0]]:
                p["regret"] = split_statistics([b[i]["regret"] - a[i]["regret"] for i in ids])
            paired.append(p)
    events = [json.loads(x) for x in (input_dir / INFERENCE_LEDGER).read_text().splitlines()]
    calls = [x for x in events if x["event"] == "response"]
    study = [x for x in calls if x["phase"] == "study"]
    accounted, pending = reconcile(events)
    latency = [x["latency_seconds"] for x in study]
    classical = [v for r in rows if r["mode"] == "classical" for v in r["decision_seconds"]]
    failures = sum(len(r.get("failures", [])) for r in rows)
    total = {
        "groups": summary,
        "paired": paired,
        "completed_blocks": len(blocks),
        "expected_blocks": 105,
        "episodes": len(rows),
        "llm_study_responses": len(study),
        "schema_failures": failures,
        "cost_usd": {
            phase: sum(x["charged_usd"] for x in calls if x["phase"] == phase)
            for phase in ("smoke", "study")
        },
        "budget_accounted_usd": accounted,
        "unresolved_attempts": sum(len(attempts) for attempts in pending.values()),
        "unresolved_reserved_usd": sum(sum(attempts) for attempts in pending.values()),
        "request_errors": sum(x["event"] == "error" for x in events),
        "api_latency_seconds": {
            "median": float(np.median(latency)),
            "p95": float(np.percentile(latency, 95)),
        }
        if latency
        else {},
        "classical_decision_seconds": {
            "median": float(np.median(classical)),
            "p95": float(np.percentile(classical, 95)),
        },
        "tokens": {
            k: sum(x["usage"].get(k, 0) for x in study)
            for k in ("prompt_tokens", "completion_tokens")
        },
        "units_note": "MSE and discrimination have squared assay units; screens are not pooled.",
        "uncertainty_note": "Five overlapping seeded splits; range and SD are protocol variability, not biological confidence intervals.",
        "hardware": platform.platform(),
        "decision_latency_note": "Per-episode decision wall time includes repairs; original API-call time excludes cache replay.",
    }
    llm = [r for r in rows if r["mode"] != "classical"]
    total["hardware_details"] = (
        json.loads((input_dir / "hardware.json").read_text())
        if (input_dir / "hardware.json").exists()
        else None
    )
    total["primary_probe_fallbacks"] = sum(
        f["attempt"] == 1 and f["step"] in ("before", "after")
        for r in llm
        if r["mode"] == "fixed"
        for f in r.get("failures", [])
    )
    total["fallback_steps"] = sum(f["attempt"] == 1 for r in llm for f in r.get("failures", []))
    total["episodes_with_schema_failures"] = sum(bool(r.get("failures")) for r in llm)
    total["schema_failures_by_condition"] = {
        agent: {
            "malformed_responses": sum(
                len(r.get("failures", [])) for r in llm if r["agent"] == agent
            ),
            "fallback_steps": sum(
                f["attempt"] == 1 for r in llm if r["agent"] == agent for f in r.get("failures", [])
            ),
        }
        for agent in ("summary", "hypothesis")
    }
    fresh = [r for r in llm if r.get("cache_hits", 0) == 0]
    total["llm_decision_seconds_uncached"] = (
        split_statistics([v for r in fresh for v in r["decision_seconds"]]) if fresh else None
    )
    if fresh:
        ds = [v for r in fresh for v in r["decision_seconds"]]
        total["llm_decision_seconds_uncached"].update(
            median=float(np.median(ds)), p95=float(np.percentile(ds, 95))
        )
    total["full_episode_seconds_uncached"] = (
        split_statistics([r["episode_seconds"] for r in fresh]) if fresh else None
    )
    if fresh:
        es = [r["episode_seconds"] for r in fresh]
        total["full_episode_seconds_uncached"].update(
            median=float(np.median(es)), p95=float(np.percentile(es, 95))
        )
    total["cache_replay_episodes"] = len(llm) - len(fresh)
    total["condition_usage"] = {}
    for condition in ("summary", "hypothesis"):
        cc = [x for x in study if ":" + condition + ":" in x["call_id"]]
        total["condition_usage"][condition] = {
            "calls": len(cc),
            "input_tokens": sum(x["usage"].get("prompt_tokens", 0) for x in cc),
            "output_tokens": sum(x["usage"].get("completion_tokens", 0) for x in cc),
            "cost_usd": sum(x["charged_usd"] for x in cc),
        }
    total["latency_targets"] = {
        "classical_all_decisions_under_one_second": bool(max(classical) < 1),
        "llm_median_decision_under_30_seconds": bool(np.median(ds) < 30) if fresh else None,
    }
    total["feedback_controls"] = []
    total["search_comparisons"] = []
    for screen in sorted({r["screen"] for r in rows if "regret" in r}):
        for agent in ("summary", "hypothesis"):
            adaptive = {
                r["task_id"]: r
                for r in rows
                if r["screen"] == screen and r["mode"] == "adaptive" and r["agent"] == agent
            }
            for baseline in ("random_mean", "ridge_random", "ridge_active"):
                classical_rows = {
                    r["task_id"]: r
                    for r in rows
                    if r["screen"] == screen and r["mode"] == "classical" and r["agent"] == baseline
                }
                ids = sorted(adaptive.keys() & classical_rows.keys())
                if ids:
                    total["search_comparisons"].append(
                        {
                            "screen": screen,
                            "agent": agent,
                            "baseline": baseline,
                            "direction": "LLM with chosen assays minus mean/ridge reference",
                            "regret_difference": split_statistics(
                                [adaptive[i]["regret"] - classical_rows[i]["regret"] for i in ids]
                            ),
                            "final_effect_difference": split_statistics(
                                [
                                    adaptive[i]["final_effect"] - classical_rows[i]["final_effect"]
                                    for i in ids
                                ]
                            ),
                        }
                    )
    for screen in sorted({r["screen"] for r in rows}):
        for agent in ("summary", "hypothesis"):
            real = {
                r["task_id"]: r
                for r in rows
                if r["screen"] == screen and r["mode"] == "fixed" and r["agent"] == agent
            }
            shuffled = {
                r["task_id"]: r
                for r in rows
                if r["screen"] == screen and r["mode"] == "shuffled" and r["agent"] == agent
            }
            ids = sorted(real.keys() & shuffled.keys())
            if ids:
                total["feedback_controls"].append(
                    {
                        "screen": screen,
                        "agent": agent,
                        "direction": "shuffled minus real final MSE",
                        "difference": split_statistics(
                            [shuffled[i]["after"]["mse"] - real[i]["after"]["mse"] for i in ids]
                        ),
                        "initial_difference": split_statistics(
                            [shuffled[i]["before"]["mse"] - real[i]["before"]["mse"] for i in ids]
                        ),
                        "gain_difference": split_statistics(
                            [shuffled[i]["mse_gain"] - real[i]["mse_gain"] for i in ids]
                        ),
                    }
                )
    # Exploratory reference added after the primary contrast was inspected.
    # A fixed published averaging rule, not a coefficient tuned on probe outcomes.
    from .metrics import continuous_scores

    controls = []
    for task in tasks(data_dir):
        if task.dataset == "norman":
            controls.append(
                {
                    "task_id": task.id,
                    **continuous_scores(0.5 * task.priors(task.probes), task.values(task.probes)),
                }
            )
    total["posthoc_matching_mean"] = {
        "status": "post hoc; added after primary comparisons, before interpreting superiority",
        "definition": "half the sum of constituent single response deltas; equivalent to averaging their expression centroids on the evaluation scale",
        "source": "https://www.nature.com/articles/s41587-025-02777-8",
        "mse": split_statistics([c["mse"] for c in controls]),
        "splits": controls,
    }
    return rows, total


def figures(rows, total, dest, root=None):
    dest = writable_output(dest, root)
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    # Each panel keeps one biological screen and its own vertical scale.
    screens = sorted({r["screen"] for r in rows if r["dataset"] == "schmidt"})
    fig, axs = plt.subplots(2, 3, figsize=(7, 5.3), layout="constrained")
    agents = ["random_mean", "ridge_random", "ridge_active", "summary", "hypothesis"]
    for ax, screen in zip(axs.flat, screens):
        for j, agent in enumerate(agents):
            rr = [
                r
                for r in rows
                if r["screen"] == screen
                and r["agent"] == agent
                and r["mode"] in ("classical", "fixed")
            ]
            if not rr:
                continue
            y = [r["mse_gain"] for r in rr]
            ax.scatter(
                np.full(len(y), j) + np.linspace(-0.12, 0.12, len(y)),
                y,
                s=12,
                color=COLORS[agent],
                alpha=0.65,
            )
            ax.plot([j - 0.23, j + 0.23], [np.mean(y)] * 2, color=COLORS[agent], lw=2)
        ax.set_title(label(screen).replace(" CD", "\nCD"), fontsize=8)
        ax.set_xticks(
            range(5),
            [
                "Random + mean",
                "Random + ridge",
                "Uncertainty + ridge",
                "Observation note",
                "Hypothesis note",
            ],
            rotation=35,
        )
        ax.set_ylabel("MSE before − after")
        ax.axhline(0, color="#777777", lw=0.7, ls="--")
    fig.savefig(dest / "prediction-gains.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(7, 5.3), layout="constrained")
    for ax, screen in zip(axs.flat, screens):
        for agent in agents:
            rr = [
                r
                for r in rows
                if r["screen"] == screen
                and r["agent"] == agent
                and r["mode"] in ("classical", "adaptive")
            ]
            if not rr:
                continue
            a = np.array([r["best_observed_curve"] for r in rr])
            ax.plot(range(1, 9), a.mean(axis=0), label=LABELS[agent], color=COLORS[agent])
        ax.set_title(label(screen).replace(" CD", "\nCD"), fontsize=8)
        ax.set_xlabel("Assays revealed")
        ax.set_ylabel("Best log2 enrichment")
    axs[0, 0].legend(fontsize=5.2, loc="upper left")
    fig.savefig(dest / "search-progress.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.4), layout="constrained")
    for agent in [
        "random_mean",
        "ridge_random",
        "ridge_active",
        "additive",
        "summary",
        "hypothesis",
    ]:
        rr = [
            r
            for r in rows
            if r["screen"] == "norman"
            and r["agent"] == agent
            and r["mode"] in ("classical", "fixed")
        ]
        if not rr:
            continue
        x = np.mean([r["charged_usd"] for r in rr]) if "charged_usd" in rr[0] else 0.0
        y = np.mean([r["after"]["mse"] for r in rr])
        axs[0].scatter(x, y, color=COLORS[agent], s=35, label=LABELS[agent])
        latency = np.median(
            [s for r in rr for s in r.get("original_call_seconds", r["decision_seconds"])]
        )
        axs[1].scatter(latency, y, color=COLORS[agent], s=35)
    axs[0].set_xlabel("API cost per episode (USD; CPU excluded)")
    axs[1].set_xlabel("Median time (s; API call or CPU decision)")
    axs[1].set_xscale("log")
    for ax in axs:
        ax.set_ylabel("Norman final probe MSE")
        ax.set_ylim(bottom=0)
    axs[0].legend(fontsize=7)
    fig.savefig(dest / "cost-and-latency.pdf")
    plt.close(fig)


def write_analysis(input_dir, output, data_dir=None):
    output = writable_output(output)
    rows, total = collect(input_dir, data_dir)
    write_json(output / "summary.json", total)
    figures(rows, total, output / "figures")
    return total


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = repository_root()
    total = write_analysis(
        args.input or Layout(root).measured,
        args.output or root / f"outputs/analysis/{MEASURED}",
        args.data,
    )
    print(
        json.dumps({"completed_blocks": total["completed_blocks"], "episodes": total["episodes"]})
    )


if __name__ == "__main__":
    main()
