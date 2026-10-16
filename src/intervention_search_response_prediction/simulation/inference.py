"""Paired, tool-free Prime Inference agent study with an append-only call cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from openai import OpenAI

from ..paths import EPISODES, NOTE_CALLS, NOTES, SELECTION, Layout, repository_root, writable_output
from .environment import (
    Episode,
    Observation,
    View,
)
from .metrics import probe_scores
from .model import Action
from .procedures import _best_observed
from .run import _verify_models
from .tasks import Task, load_models, make_tasks

MODEL_ID = "openai/gpt-6-sol"
MAX_TOKENS = 1024
COST_CAP = 50.0
SELECTION_SALT = "cellworld-agenticls-2026-frozen-v1"
CONDITIONS = ("history", "predictive")
POSTHOC_CONDITION = "summary_note"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def selected_tasks(models_dir: Path) -> tuple[tuple[Task, ...], tuple[Task, ...], Task]:
    _verify_models(models_dir)
    by_model: dict[str, list[Task]] = {}
    for model in load_models(models_dir):
        by_model[model.name] = [task for seed in (0, 1, 2) for task in make_tasks(model, seed)]
    selected = tuple(
        min(tasks, key=lambda task: _hash(f"{SELECTION_SALT}:main:{task.id}"))
        for _, tasks in sorted(by_model.items())
    )
    selected = tuple(
        sorted(selected, key=lambda task: _hash(f"{SELECTION_SALT}:order:{task.model.name}"))
    )
    anonymous = tuple(
        sorted(selected, key=lambda task: _hash(f"{SELECTION_SALT}:anonymous:{task.model.name}"))[
            :20
        ]
    )
    first = selected[0]
    smoke = min(
        (task for task in by_model[first.model.name] if task.id != first.id),
        key=lambda task: _hash(f"{SELECTION_SALT}:smoke:{task.id}"),
    )
    return selected, anonymous, smoke


def node_labels(task: Task, anonymous: bool) -> tuple[str, ...]:
    if not anonymous:
        return task.model.nodes
    import random

    ids = list(range(len(task.model.nodes)))
    random.Random(int(_hash(f"{SELECTION_SALT}:rename:{task.model.sha256}")[:16], 16)).shuffle(ids)
    return tuple(f"node_{ids[i]:02d}" for i in range(len(ids)))


def named_action(action: Action, labels: tuple[str, ...]) -> list[list[str | int]]:
    return [[labels[node], value] for node, value in action]


def agent_view(view: View, task: Task, labels: tuple[str, ...]) -> dict:
    forbidden = [named_action(action, labels) for action in task.probes]
    return {
        "nodes": list(labels),
        "initial": {labels[i]: (view.initial >> i) & 1 for i in range(len(labels))},
        "horizon": view.horizon,
        "readouts": [labels[i] for i in view.readouts],
        "target": {labels[i]: bit for i, bit in view.target},
        "signed_edges": [[labels[src], labels[dst], sign] for src, dst, sign in view.signed_edges],
        "eligible_clamp_nodes": [
            labels[i] for i in range(len(labels)) if i not in dict(view.target)
        ],
        "forbidden_assays": forbidden,
        "assay_limit": 8,
        "action_rule": "Clamp one or two distinct eligible nodes to binary values; forbidden assays cannot be executed. A final action may be any legal clamp, including a forbidden assay.",
    }


def _history(entries: list[Observation], labels: tuple[str, ...]) -> list[dict]:
    return [
        {
            "action": named_action(entry.action, labels),
            "prediction": list(entry.probabilities),
            "observation": list(entry.values),
        }
        for entry in entries
    ]


def prompt_for(
    phase: str,
    view_data: dict,
    history: list[dict],
    condition: str,
    state_note: str,
    probes: list[list] | None = None,
) -> tuple[str, str]:
    system = (
        "You are an experimental agent studying a hidden deterministic Boolean system. "
        "Each assay resets to the disclosed initial state. The signed graph may be incomplete. "
        "You do not know the update rules. Return one valid JSON object only, with no Markdown. "
        "Probability arrays follow the stated readout order. Be concise."
    )
    if condition == "predictive":
        system += " Maintain a short, revisable response hypothesis in the note field. Revise it when observations disagree with predictions."
    elif condition == POSTHOC_CONDITION:
        system += " Use the note field to record a brief summary of what you have observed so far."
    data: dict = {"phase": phase, "task": view_data, "history": history}
    if condition == "predictive":
        data["previous_hypothesis"] = state_note
    elif condition == POSTHOC_CONDITION:
        data["previous_note"] = state_note
    if phase == "assay":
        data["request"] = (
            "Choose one unused legal assay and predict every readout BEFORE seeing its response. "
            'Return {"action":[["node",0]],"p":[0.5,0.5,0.5,0.5]}'
            + (
                ', plus a short "note" string describing or revising your response hypothesis.'
                if condition == "predictive"
                else ', plus a short "note" string.'
                if condition == POSTHOC_CONDITION
                else "."
            )
        )
    elif phase == "probe":
        data["probes"] = probes
        data["request"] = (
            'For each probe in the given order, return one probability per readout: {"predictions":[[0.5,0.5,0.5,0.5],...]} . No outcomes are provided.'
        )
    elif phase == "final":
        data["request"] = (
            'Submit one legal final intervention for the target. Return {"action":[["node",0]]}.'
        )
    else:
        raise ValueError(phase)
    return system, json.dumps(data, sort_keys=True, separators=(",", ":"))


def _json_content(content: str) -> dict | None:
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _action(value: object, labels: tuple[str, ...], legal: tuple[Action, ...]) -> Action | None:
    if not isinstance(value, list) or not 1 <= len(value) <= 2:
        return None
    lookup = {name: i for i, name in enumerate(labels)}
    pairs = []
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or pair[0] not in lookup
            or type(pair[1]) is not int
            or pair[1] not in (0, 1)
        ):
            return None
        pairs.append((lookup[pair[0]], pair[1]))
    if len({i for i, _ in pairs}) != len(pairs):
        return None
    parsed = tuple(sorted(pairs))
    return parsed if parsed in legal else None


def _probabilities(value: object, width: int) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != width:
        return None
    if any(type(p) not in (int, float) or not 0 <= p <= 1 for p in value):
        return None
    return tuple(float(p) for p in value)


class PrimeCalls:
    def __init__(self, path: Path, cap: float = COST_CAP):
        self.path = writable_output(path)
        path = self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cached = (
            {
                row["call_id"]: row
                for row in (json.loads(line) for line in path.read_text().splitlines())
            }
            if path.exists()
            else {}
        )
        self.cost = sum(float(row["usage"].get("cost") or 0) for row in self.cached.values())
        self.cap = cap
        key = os.environ.get("PRIME_API_KEY")
        if not key:
            config = Path.home() / ".prime" / "config.json"
            if config.exists():
                key = json.loads(config.read_text()).get("api_key")
        if not key:
            raise RuntimeError("set PRIME_API_KEY or authenticate the Prime CLI")
        self.client = OpenAI(
            api_key=key, base_url="https://api.pinference.ai/api/v1", max_retries=0, timeout=120
        )
        self.sdk_version = version("openai")

    def ask(self, call_id: str, system: str, message: str) -> dict:
        request_hash = _hash(system + "\n" + message)
        if call_id in self.cached:
            record = self.cached[call_id]
            if record["request_hash"] != request_hash:
                raise ValueError(f"cached call differs: {call_id}")
            return record
        # Reserve maximum output cost and a conservative input-token estimate.
        reserve = 10.0 * MAX_TOKENS / 1_000_000 + 2.0 * (len(system) + len(message)) / 3 / 1_000_000
        if self.cost + 1.5 * reserve > self.cap:
            raise RuntimeError(f"Prime spending cap reached before {call_id}: ${self.cost:.4f}")
        response = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
            temperature=0,
            max_tokens=MAX_TOKENS,
            reasoning_effort="none",
            response_format={"type": "json_object"},
            extra_body={"usage": {"include": True}},
        ).model_dump()
        content = response["choices"][0]["message"].get("content") or ""
        record = {
            "call_id": call_id,
            "request_hash": request_hash,
            "model": response.get("model"),
            "sdk_version": self.sdk_version,
            "system": system,
            "message": message,
            "response_id": response.get("id"),
            "content": content,
            "finish_reason": response["choices"][0].get("finish_reason"),
            "usage": response.get("usage") or {},
        }
        with self.path.open("a") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        self.cached[call_id] = record
        self.cost += float(record["usage"].get("cost") or 0)
        return record


def run_episode(
    task: Task, condition: str, anonymous: bool, calls: PrimeCalls, smoke: bool = False
) -> dict:
    episode = Episode(task)
    view = episode.inspect_network()
    labels = node_labels(task, anonymous)
    view_data = agent_view(view, task, labels)
    identifier = f"{'smoke-sdk' if smoke else 'study'}:{task.id}:{'anonymous' if anonymous else 'named'}:{condition}"
    truth = episode.probe_truths()
    probes = [named_action(action, labels) for action in task.probes]
    state_note = ""
    failures = {
        "invalid_action": 0,
        "invalid_probability": 0,
        "invalid_probe": 0,
        "invalid_final": 0,
    }

    def get_probe(when: str) -> tuple[tuple[float, ...], ...]:
        system, message = prompt_for(
            "probe", view_data, _history(episode.history, labels), condition, state_note, probes
        )
        record = calls.ask(f"{identifier}:{when}", system, message)
        parsed = _json_content(record["content"])
        rows = parsed.get("predictions") if parsed else None
        if not isinstance(rows, list) or len(rows) != len(probes):
            failures["invalid_probe"] += 1
            return ((0.5,) * len(view.readouts),) * len(probes)
        forecasts = tuple(_probabilities(row, len(view.readouts)) for row in rows)
        if any(row is None for row in forecasts):
            failures["invalid_probe"] += 1
            return ((0.5,) * len(view.readouts),) * len(probes)
        return forecasts  # type: ignore[return-value]

    before = get_probe("before")
    for round_index in range(8):
        system, message = prompt_for(
            "assay", view_data, _history(episode.history, labels), condition, state_note
        )
        record = calls.ask(f"{identifier}:assay:{round_index}", system, message)
        parsed = _json_content(record["content"])
        available = tuple(
            a for a in view.menu if a not in {entry.action for entry in episode.history}
        )
        action = _action(parsed.get("action"), labels, available) if parsed else None
        if action is None:
            action = available[0]
            probabilities = None
            failures["invalid_action"] += 1
        else:
            probabilities = _probabilities(parsed.get("p"), len(view.readouts))
        if probabilities is None:
            probabilities = (0.5,) * len(view.readouts)
            failures["invalid_probability"] += 1
        episode.predict(action, probabilities)
        episode.experiment(action)
        if (
            condition in ("predictive", POSTHOC_CONDITION)
            and parsed
            and isinstance(parsed.get("note"), str)
        ):
            state_note = parsed["note"][:800]
    after = get_probe("after")
    system, message = prompt_for(
        "final", view_data, _history(episode.history, labels), condition, state_note
    )
    record = calls.ask(f"{identifier}:final", system, message)
    parsed = _json_content(record["content"])
    final_action = _action(parsed.get("action"), labels, view.final_actions) if parsed else None
    if final_action is None:
        final_action = _best_observed(view, episode.history)
        failures["invalid_final"] += 1
    success = episode.submit(final_action)
    first_success = next(
        (
            i
            for i, entry in enumerate(episode.history, 1)
            if all(entry.values[view.readouts.index(node)] == bit for node, bit in view.target)
        ),
        None,
    )
    before_scores = probe_scores(before, truth)
    after_scores = probe_scores(after, truth)
    call_records = [
        calls.cached[f"{identifier}:{name}"]
        for name in ["before", *(f"assay:{i}" for i in range(8)), "after", "final"]
    ]
    return {
        "task_id": task.id,
        "model": task.model.name,
        "condition": condition,
        "label_condition": "anonymous" if anonymous else "named",
        "smoke": smoke,
        "transport": "openai-sdk",
        "model_id": MODEL_ID,
        "goal_success": int(success),
        "first_success": first_success,
        "brier_before": before_scores["brier"],
        "brier_after": after_scores["brier"],
        "brier_gain": before_scores["brier"] - after_scores["brier"],
        "constant_brier": after_scores["constant_brier"],
        "excess_brier_before": before_scores["excess_brier"],
        "excess_brier_after": after_scores["excess_brier"],
        "discrimination_before": before_scores["discrimination"],
        "discrimination_after": after_scores["discrimination"],
        "variable_readout_fraction": after_scores["variable_readout_fraction"],
        "probe_truths": [list(row) for row in truth],
        "probe_predictions_before": [list(row) for row in before],
        "probe_predictions_after": [list(row) for row in after],
        "probes": probes,
        "final_action": named_action(final_action, labels),
        "trajectory": _history(episode.history, labels),
        "failures": failures,
        "tokens_input": sum(int(row["usage"].get("prompt_tokens") or 0) for row in call_records),
        "tokens_output": sum(
            int(row["usage"].get("completion_tokens") or 0) for row in call_records
        ),
        "cost": sum(float(row["usage"].get("cost") or 0) for row in call_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "select", "posthoc"))
    parser.add_argument("--models", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = repository_root()
    args.models = args.models or Layout(root).models
    args.output = args.output or root / f"outputs/{NOTES}"
    args.output = writable_output(args.output)
    selected, anonymous, smoke = selected_tasks(args.models)
    args.output.mkdir(parents=True, exist_ok=True)
    selection = {
        "salt": SELECTION_SALT,
        "model_id": MODEL_ID,
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": MAX_TOKENS,
        "spending_cap_usd": COST_CAP,
        "named": [task.id for task in selected],
        "anonymous": [task.id for task in anonymous],
        "smoke": smoke.id,
    }
    (args.output / SELECTION).write_text(json.dumps(selection, indent=2) + "\n")
    if args.command == "select":
        print(json.dumps(selection, indent=2))
        return
    calls = PrimeCalls(args.output / NOTE_CALLS)
    episodes_path = args.output / EPISODES
    completed = (
        {
            (row["task_id"], row["label_condition"], row["condition"])
            for row in (json.loads(line) for line in episodes_path.read_text().splitlines())
            if row.get("smoke") == (args.command == "smoke")
            and row.get("transport") == "openai-sdk"
        }
        if episodes_path.exists()
        else set()
    )
    if args.command == "smoke":
        queue = [(smoke, "named", condition) for condition in CONDITIONS]
    elif args.command == "posthoc":
        queue = [(task, "named", POSTHOC_CONDITION) for task in selected]
    else:
        queue = [(task, "named", condition) for task in selected for condition in CONDITIONS] + [
            (task, "anonymous", condition) for task in anonymous for condition in CONDITIONS
        ]
    for task, label_condition, condition in queue:
        key = (task.id, label_condition, condition)
        if key in completed and args.command != "smoke":
            continue
        if args.command == "smoke" and key in completed:
            continue
        row = run_episode(
            task, condition, label_condition == "anonymous", calls, args.command == "smoke"
        )
        with episodes_path.open("a") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        completed.add(key)
        print(
            json.dumps(
                {
                    "task": task.id,
                    "condition": condition,
                    "labels": label_condition,
                    "completed": len(completed),
                    "cost_total": round(calls.cost, 4),
                    "brier": round(row["brier_after"], 4),
                    "goal": row["goal_success"],
                    "failures": row["failures"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
