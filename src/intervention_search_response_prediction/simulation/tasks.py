"""Deterministic construction of finite-horizon intervention tasks."""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from pathlib import Path

from .model import (
    Action,
    BooleanModel,
    legal_actions,
)


@dataclass(frozen=True)
class Task:
    model: BooleanModel
    seed: int
    initial: int
    horizon: int
    readouts: tuple[int, ...]
    target: tuple[tuple[int, int], ...]
    actions: tuple[Action, ...]
    probes: tuple[Action, ...]
    success_rate: float

    @property
    def id(self) -> str:
        bits = "-".join(f"{self.model.nodes[node]}={value}" for node, value in self.target)
        return f"{self.model.name}|s{self.seed}|{bits}"

    def response(self, action: Action) -> int:
        return self.model.simulate(self.initial, action, self.horizon)

    def succeeds(self, action: Action) -> bool:
        state = self.response(action)
        return all(((state >> node) & 1) == value for node, value in self.target)


def make_tasks(
    model: BooleanModel, seed: int, max_tasks: int = 2, horizon: int = 3
) -> tuple[Task, ...]:
    """Select reachable, nontrivial goals from a pinned model and seed.

    The evaluator uses hidden rules for eligibility. Agent inputs never include
    generated outcomes or selection scores.
    """
    model_seed = int(hashlib.sha256(f"{model.sha256}:{seed}".encode()).hexdigest()[:16], 16)
    rng = random.Random(model_seed)
    initial = rng.getrandbits(len(model.nodes))
    all_actions = legal_actions(len(model.nodes))
    outcomes = {action: model.simulate(initial, action, horizon) for action in all_actions}
    baseline = model.simulate(initial, (), horizon)
    variation = []
    for node in range(len(model.nodes)):
        ones = sum((state >> node) & 1 for state in outcomes.values())
        p = ones / len(outcomes)
        if 0 < p < 1:
            variation.append((p * (1 - p), node))
    variation.sort(key=lambda pair: (-pair[0], pair[1]))
    if len(variation) < 3:
        return ()
    readouts = tuple(sorted(node for _, node in variation[: min(4, len(variation))]))
    candidates = []
    for width in (1, 2):
        for nodes in itertools.combinations(readouts, width):
            target = tuple((node, 1 - ((baseline >> node) & 1)) for node in nodes)
            actions = legal_actions(len(model.nodes), frozenset(nodes))
            if len(actions) < 18:
                continue
            success = sum(
                all(((outcomes[a] >> node) & 1) == value for node, value in target) for a in actions
            )
            fraction = success / len(actions)
            if success and 0.01 <= fraction <= 0.35:
                candidates.append((fraction, target, actions))
    if not candidates:
        return ()
    # Prefer two-readout targets when feasible; ties are stable and independent of agent outcomes.
    candidates.sort(key=lambda row: (-len(row[1]), abs(row[0] - 0.10), row[1]))
    tasks = []
    used_targets: set[int] = set()
    for fraction, target, actions in candidates:
        if len(tasks) >= max_tasks:
            break
        if set(node for node, _ in target) <= used_targets:
            continue
        probe_rng = random.Random(
            int(hashlib.sha256(f"{model_seed}:{target}".encode()).hexdigest()[:16], 16)
        )
        probes = tuple(probe_rng.sample(actions, 10))
        tasks.append(
            Task(model, seed, initial, horizon, readouts, target, actions, probes, fraction)
        )
        used_targets.update(node for node, _ in target)
    return tuple(tasks)


def load_models(directory: Path) -> tuple[BooleanModel, ...]:
    return tuple(BooleanModel.from_bnet(path) for path in sorted(directory.glob("*.bnet")))
