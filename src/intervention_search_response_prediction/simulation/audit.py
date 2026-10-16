"""Recompute study outcomes and check the frozen language-agent protocol."""

from __future__ import annotations

import json
from math import isclose
from pathlib import Path

from ..io import read_jsonl
from ..paths import (
    EPISODES,
    NOTE_CALLS,
    REFERENCE_COMPARISONS,
    REFERENCE_SUMMARY,
    SELECTION,
    Layout,
    repository_root,
)
from .analysis import (
    summarize as summarize_classical,
)
from .comparisons import (
    analyze as analyze_extended,
)
from .environment import Episode
from .inference import (
    MODEL_ID,
    _json_content,
    agent_view,
    named_action,
    node_labels,
    selected_tasks,
)
from .metrics import probe_scores
from .note_analysis import (
    summarize as summarize_language,
)
from .tasks import load_models, make_tasks


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit_classical(root: Path) -> dict:
    tasks = {
        task.id: task
        for model in load_models(Layout(root).models)
        for seed in (0, 1, 2)
        for task in make_tasks(model, seed)
    }
    rows = read_jsonl(Layout(root).baselines / EPISODES)
    strategies = {row["strategy"] for row in rows}
    indexed = {(row["task_id"], row["strategy"]): row for row in rows}
    assert len(tasks) == 232 and len(strategies) == 6
    assert len(rows) == len(indexed) == len(tasks) * len(strategies) == 1392
    assert all(row["oracle_success"] for row in rows)
    for row in rows:
        task = tasks[row["task_id"]]
        labels = task.model.nodes
        lookup = {name: index for index, name in enumerate(labels)}
        truth = Episode(task).probe_truths()
        assert row["probe_truths"] == [list(values) for values in truth]
        assert row["probes"] == [named_action(action, labels) for action in task.probes]
        assert row["readouts"] == [labels[index] for index in task.readouts]
        for phase, suffix in (("before", "before"), ("after", "after")):
            forecasts = tuple(tuple(values) for values in row[f"probe_predictions_{phase}"])
            scores = probe_scores(forecasts, truth)
            for field, source in (
                (f"brier_{suffix}", "brier"),
                (f"excess_brier_{suffix}", "excess_brier"),
                (f"discrimination_{suffix}", "discrimination"),
            ):
                assert isclose(row[field], scores[source], abs_tol=1e-12)
            assert isclose(row["constant_brier"], scores["constant_brier"], abs_tol=1e-12)
        assayed = []
        hits = []
        for i, entry in enumerate(row["trajectory"], 1):
            action = tuple(sorted((lookup[node], bit) for node, bit in entry["action"]))
            assert action in task.actions and action not in task.probes
            assayed.append(action)
            state = task.response(action)
            assert entry["observation"] == [(state >> node) & 1 for node in task.readouts]
            if task.succeeds(action):
                hits.append(i)
        assert len(assayed) == len(set(assayed)) == 8
        assert row["first_success"] == (hits[0] if hits else None)
        final = tuple(sorted((lookup[node], bit) for node, bit in row["final_action"]))
        assert row["goal_success"] == int(task.succeeds(final))
    for task_id in tasks:
        mean = indexed[(task_id, "graph-mean")]
        observed = indexed[(task_id, "graph-ridge-observed")]
        predicted = indexed[(task_id, "graph-ridge")]
        assert (
            [entry["action"] for entry in mean["trajectory"]]
            == [entry["action"] for entry in observed["trajectory"]]
            == [entry["action"] for entry in predicted["trajectory"]]
        )
        assert mean["final_action"] == observed["final_action"]
        assert observed["probe_predictions_after"] == predicted["probe_predictions_after"]
    assert summarize_classical(rows) == json.loads(
        (Layout(root).baselines / REFERENCE_SUMMARY).read_text()
    )
    return {
        "models": len({task.model.name for task in tasks.values()}),
        "tasks": len(tasks),
        "episodes": len(rows),
    }


def audit(root: Path | None = None) -> dict:
    root = repository_root(root)
    selected, anonymous, smoke = selected_tasks(Layout(root).models)
    tasks = {task.id: task for task in (*selected, smoke)}
    selection = json.loads((Layout(root).notes / SELECTION).read_text())
    assert selection["named"] == [task.id for task in selected]
    assert selection["anonymous"] == [task.id for task in anonymous]
    assert selection["smoke"] == smoke.id

    all_calls = _rows(Layout(root).notes / NOTE_CALLS)
    calls = {row["call_id"]: row for row in all_calls}
    assert len(calls) == len(all_calls)
    cost = sum(float(row["usage"].get("cost") or 0) for row in all_calls)
    assert cost <= selection["spending_cap_usd"]
    all_study = [row for row in _rows(Layout(root).notes / EPISODES) if not row.get("smoke")]
    study = [row for row in all_study if row["condition"] in ("history", "predictive")]
    posthoc = [row for row in all_study if row["condition"] == "summary_note"]
    expected = {
        (task.id, "named", condition)
        for task in selected
        for condition in ("history", "predictive")
    }
    expected |= {
        (task.id, "anonymous", condition)
        for task in anonymous
        for condition in ("history", "predictive")
    }
    keys = [(row["task_id"], row["label_condition"], row["condition"]) for row in study]
    assert len(keys) == len(set(keys)) == len(expected) == 118
    assert set(keys) == expected
    indexed = {(row["task_id"], row["label_condition"], row["condition"]): row for row in study}
    for task in anonymous:
        for condition in ("history", "predictive"):
            assert (
                indexed[(task.id, "named", condition)]["probe_truths"]
                == indexed[(task.id, "anonymous", condition)]["probe_truths"]
            )

    assert all(row["label_condition"] == "named" for row in posthoc)
    assert len({row["task_id"] for row in posthoc}) == len(posthoc)
    for row in (*study, *posthoc):
        task = tasks[row["task_id"]]
        anonymous_labels = row["label_condition"] == "anonymous"
        labels = node_labels(task, anonymous_labels)
        label_to_node = {label: index for index, label in enumerate(labels)}
        episode = Episode(task)
        view = episode.inspect_network()
        menu = view.menu
        expected_view = agent_view(view, task, labels)
        truth = episode.probe_truths()
        assert row["model_id"] == MODEL_ID
        assert row["probe_truths"] == [list(values) for values in truth]
        assert row["probes"] == [named_action(action, labels) for action in task.probes]
        assert len(row["trajectory"]) == 8
        assayed = []
        for entry in row["trajectory"]:
            action = tuple(sorted((label_to_node[node], bit) for node, bit in entry["action"]))
            assert action in menu
            assayed.append(action)
            state = task.response(action)
            assert entry["observation"] == [(state >> node) & 1 for node in task.readouts]
            assert len(entry["prediction"]) == len(task.readouts)
            assert all(0 <= p <= 1 for p in entry["prediction"])
        assert len(set(assayed)) == 8
        assert not set(assayed) & set(task.probes)
        hits = [
            i
            for i, entry in enumerate(row["trajectory"], 1)
            if all(
                entry["observation"][task.readouts.index(node)] == bit for node, bit in task.target
            )
        ]
        assert row["first_success"] == (hits[0] if hits else None)
        final = tuple(sorted((label_to_node[node], bit) for node, bit in row["final_action"]))
        assert final in task.actions
        assert row["goal_success"] == int(task.succeeds(final))
        for phase, name in (("before", "before"), ("after", "after")):
            forecast = row[f"probe_predictions_{phase}"]
            score = probe_scores(tuple(tuple(values) for values in forecast), truth)
            suffix = "before" if phase == "before" else "after"
            for field, source in (
                (f"brier_{suffix}", "brier"),
                (f"excess_brier_{suffix}", "excess_brier"),
                (f"discrimination_{suffix}", "discrimination"),
            ):
                assert isclose(row[field], score[source], abs_tol=1e-12)
            assert isclose(row["constant_brier"], score["constant_brier"], abs_tol=1e-12)
        assert isclose(row["brier_gain"], row["brier_before"] - row["brier_after"], abs_tol=1e-12)
        prefix = f"study:{task.id}:{row['label_condition']}:{row['condition']}"
        ids = [
            f"{prefix}:before",
            *(f"{prefix}:assay:{i}" for i in range(8)),
            f"{prefix}:after",
            f"{prefix}:final",
        ]
        assert all(call_id in calls for call_id in ids)
        assert isclose(
            row["cost"],
            sum(float(calls[call_id]["usage"].get("cost") or 0) for call_id in ids),
            abs_tol=1e-12,
        )
        assert row["tokens_input"] == sum(
            int(calls[call_id]["usage"].get("prompt_tokens") or 0) for call_id in ids
        )
        assert row["tokens_output"] == sum(
            int(calls[call_id]["usage"].get("completion_tokens") or 0) for call_id in ids
        )
        note = ""
        for call_id in ids:
            call = calls[call_id]
            assert call["model"] == MODEL_ID
            assert call.get("sdk_version")
            assert (call["usage"].get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ) == 0
            prompt = json.loads(call["message"])
            assert prompt["task"] == expected_view
            if row["condition"] == "predictive":
                assert prompt["previous_hypothesis"] == note
            else:
                assert "previous_hypothesis" not in prompt
            if row["condition"] == "summary_note":
                assert prompt["previous_note"] == note
                assert (
                    "Use the note field to record a brief summary of what you have observed so far."
                    in call["system"]
                )
                assert "response hypothesis" not in call["system"]
            else:
                assert "previous_note" not in prompt
            assert "rules" not in prompt["task"] and "probe_truths" not in prompt
            assert "model" not in prompt["task"]
            if anonymous_labels:
                assert all(node.startswith("node_") for node in prompt["task"]["nodes"])
                assert all(
                    node in labels for entry in prompt["history"] for node, _ in entry["action"]
                )
                assert all(
                    node in labels for action in prompt.get("probes", []) for node, _ in action
                )
            if call_id.endswith(":before"):
                assert not prompt["history"]
            if call_id.endswith(":after") or call_id.endswith(":final"):
                assert len(prompt["history"]) == 8
            if call_id.endswith(":before") or call_id.endswith(":after"):
                assert prompt["phase"] == "probe" and len(prompt["probes"]) == 10
                assert "outcomes" not in prompt
            if ":assay:" in call_id and row["condition"] in ("predictive", "summary_note"):
                parsed = _json_content(call["content"])
                if parsed and isinstance(parsed.get("note"), str):
                    note = parsed["note"][:800]

    stored = json.loads((Layout(root).notes / REFERENCE_SUMMARY).read_text())
    recomputed = summarize_language(all_study)
    for key in ("episodes", "named", "anonymous", "paired", "matched"):
        assert stored[key] == recomputed[key]
    if posthoc:
        assert stored["posthoc"] == recomputed["posthoc"]
    extended_path = Layout(root).notes / REFERENCE_COMPARISONS
    if extended_path.exists():
        assert json.loads(extended_path.read_text()) == analyze_extended(
            read_jsonl(Layout(root).baselines / EPISODES), all_study
        )
    named = [task.id for task in selected]
    paired_gain = []
    clean_final = []
    for task_id in named:
        history = indexed[(task_id, "named", "history")]
        predictive = indexed[(task_id, "named", "predictive")]
        paired_gain.append(
            (predictive["discrimination_after"] - predictive["discrimination_before"])
            - (history["discrimination_after"] - history["discrimination_before"])
        )
        if (
            not history["failures"]["invalid_action"]
            and not predictive["failures"]["invalid_action"]
        ):
            clean_final.append(predictive["discrimination_after"] - history["discrimination_after"])
    assert isclose(
        sum(paired_gain) / len(paired_gain),
        stored["paired"]["named"]["discrimination_gain"]["mean"],
        abs_tol=1e-12,
    )
    assert len(clean_final) == stored["paired"]["named_no_assay_fallback"]["tasks"] == 20
    assert isclose(
        sum(clean_final) / len(clean_final),
        stored["paired"]["named_no_assay_fallback"]["metrics"]["discrimination_after"]["mean"],
        abs_tol=1e-12,
    )
    return {
        "classical": audit_classical(root),
        "study_episodes": len(study),
        "study_calls": 11 * len(study),
        "posthoc_episodes": len(posthoc),
        "posthoc_calls": 11 * len(posthoc),
        "named_tasks": len(selected),
        "anonymous_tasks": len(anonymous),
        "total_calls_including_smoke": len(all_calls),
        "total_cost_usd": cost,
        "failures": {
            key: sum(row["failures"][key] for row in study)
            for key in ("invalid_action", "invalid_probability", "invalid_probe", "invalid_final")
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))


if __name__ == "__main__":
    main()
