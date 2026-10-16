"""Explicit mean and ridge procedures for Boolean intervention tasks."""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass
from functools import cache

from .environment import (
    Episode,
    Observation,
    View,
)
from .metrics import probe_scores
from .model import Action
from .tasks import Task


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        if abs(scale) < 1e-12:
            raise ValueError("singular ridge system")
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[col])]
    return [row[-1] for row in augmented]


class Forecaster:
    def __init__(self, readouts: int):
        self.readouts = readouts

    def update(self, observation: Observation) -> None:
        pass

    def forecast(self, action: Action) -> tuple[float, ...]:
        raise NotImplementedError

    def variance(self, action: Action) -> float:
        return 0.0


class HalfForecaster(Forecaster):
    def forecast(self, action: Action) -> tuple[float, ...]:
        return (0.5,) * self.readouts


class MeanForecaster(Forecaster):
    def __init__(self, readouts: int):
        super().__init__(readouts)
        self.positives = [0] * readouts
        self.count = 0

    def update(self, observation: Observation) -> None:
        self.count += 1
        for i, value in enumerate(observation.values):
            self.positives[i] += value

    def forecast(self, action: Action) -> tuple[float, ...]:
        return tuple((1 + count) / (2 + self.count) for count in self.positives)


def _kernel(a: Action, b: Action) -> float:
    return 1.0 + len(set(a) & set(b))


class RidgeForecaster(Forecaster):
    """Ridge fit on an intercept and node-value indicators; prior mean is 1/2."""

    def __init__(self, readouts: int, penalty: float = 2.0):
        super().__init__(readouts)
        self.penalty = penalty
        self.history: list[Observation] = []
        self.alphas: list[list[float]] = []
        self.matrix: list[list[float]] = []

    def update(self, observation: Observation) -> None:
        self.history.append(observation)
        actions = [entry.action for entry in self.history]
        self.matrix = [
            [_kernel(a, b) + (self.penalty if i == j else 0.0) for j, b in enumerate(actions)]
            for i, a in enumerate(actions)
        ]
        self.alphas = [
            _solve(self.matrix, [entry.values[r] - 0.5 for entry in self.history])
            for r in range(self.readouts)
        ]

    def forecast(self, action: Action) -> tuple[float, ...]:
        if not self.history:
            return (0.5,) * self.readouts
        kernel = [_kernel(action, entry.action) for entry in self.history]
        return tuple(
            min(1.0, max(0.0, 0.5 + sum(x * y for x, y in zip(kernel, alpha))))
            for alpha in self.alphas
        )

    def variance(self, action: Action) -> float:
        if not self.history:
            return _kernel(action, action)
        k = [_kernel(action, entry.action) for entry in self.history]
        solution = _solve(self.matrix, k)
        return max(0.0, _kernel(action, action) - sum(x * y for x, y in zip(k, solution)))


@cache
def _signed_distance(
    edges: tuple[tuple[int, int, int], ...], source: int, target: int, desired_sign: int
) -> int | None:
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for start, end, sign in edges:
        adjacency.setdefault(start, []).append((end, sign))
    queue = deque([(source, 1, 0)])
    seen = {(source, 1)}
    while queue:
        node, sign, length = queue.popleft()
        if node == target and length and sign == desired_sign:
            return length
        for next_node, edge_sign in adjacency.get(node, ()):
            state = (next_node, sign * edge_sign)
            if state not in seen:
                seen.add(state)
                queue.append((next_node, state[1], length + 1))
    return None


def graph_score(view: View, action: Action) -> float:
    score = 0.0
    for target, desired_value in view.target:
        desired_change = desired_value - ((view.initial >> target) & 1)
        best = 0.0
        for source, value in action:
            source_change = value - ((view.initial >> source) & 1)
            if not source_change or not desired_change:
                continue
            path_sign = 1 if source_change == desired_change else -1
            distance = _signed_distance(view.signed_edges, source, target, path_sign)
            if distance is not None:
                best = max(best, 1 / (1 + distance))
        score += best
    return score - 0.01 * (len(action) - 1)


def _best_observed(view: View, history: list[Observation]) -> Action:
    target_positions = [(view.readouts.index(node), value) for node, value in view.target]
    return min(
        history,
        key=lambda entry: (
            sum(entry.values[position] != desired for position, desired in target_positions),
            len(entry.action),
            entry.action,
        ),
    ).action


def _best_predicted(view: View, forecaster: Forecaster) -> Action:
    def score(action: Action) -> float:
        forecast = forecaster.forecast(action)
        joint = 1.0
        for node, desired in view.target:
            p = forecast[view.readouts.index(node)]
            joint *= p if desired else 1 - p
        return joint

    return min(view.final_actions, key=lambda action: (-score(action), len(action), action))


def _brier(
    predictions: tuple[tuple[float, ...], ...], truths: tuple[tuple[int, ...], ...]
) -> float:
    return sum((p - y) ** 2 for ps, ys in zip(predictions, truths) for p, y in zip(ps, ys)) / sum(
        len(ys) for ys in truths
    )


@dataclass(frozen=True)
class Strategy:
    name: str
    policy: str
    forecaster: str
    final: str


STRATEGIES = (
    Strategy("random-half", "random", "half", "observed"),
    Strategy("random-mean", "random", "mean", "observed"),
    Strategy("graph-mean", "graph", "mean", "observed"),
    Strategy("graph-ridge-observed", "graph", "ridge", "observed"),
    Strategy("graph-ridge", "graph", "ridge", "predicted"),
    Strategy("active-ridge", "active", "ridge", "predicted"),
)


def run_episode(task: Task, strategy: Strategy) -> dict:
    episode = Episode(task)
    view = episode.inspect_network()
    forecaster: Forecaster = {
        "half": HalfForecaster,
        "mean": MeanForecaster,
        "ridge": RidgeForecaster,
    }[strategy.forecaster](len(view.readouts))
    truth = episode.probe_truths()
    before = tuple(forecaster.forecast(action) for action in task.probes)
    before_scores = probe_scores(before, truth)
    curve = [before_scores["brier"]]
    seed = int(hashlib.sha256(f"{task.id}:{strategy.policy}".encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rank = sorted(view.menu, key=lambda action: (-graph_score(view, action), len(action), action))
    used: set[Action] = set()
    for _ in range(episode.budget):
        remaining = [action for action in view.menu if action not in used]
        if strategy.policy == "random":
            action = rng.choice(remaining)
        elif strategy.policy == "graph":
            action = next(action for action in rank if action not in used)
        else:
            candidates = [action for action in rank if action not in used][:40]
            action = min(
                candidates,
                key=lambda a: (-forecaster.variance(a), -graph_score(view, a), len(a), a),
            )
        probabilities = forecaster.forecast(action)
        episode.predict(action, probabilities)
        observation = episode.experiment(action)
        forecaster.update(observation)
        used.add(action)
        curve.append(_brier(tuple(forecaster.forecast(probe) for probe in task.probes), truth))
    after = tuple(forecaster.forecast(action) for action in task.probes)
    after_scores = probe_scores(after, truth)
    final_action = (
        _best_observed(view, episode.history)
        if strategy.final == "observed"
        else _best_predicted(view, forecaster)
    )
    goal_success = episode.submit(final_action)
    first_success = next(
        (
            i
            for i, entry in enumerate(episode.history, 1)
            if all(entry.values[view.readouts.index(node)] == value for node, value in view.target)
        ),
        None,
    )
    prospective = sum(
        (p - y) ** 2 for entry in episode.history for p, y in zip(entry.probabilities, entry.values)
    ) / (len(episode.history) * len(view.readouts))
    return {
        "task_id": task.id,
        "model": task.model.name,
        "model_sha256": task.model.sha256,
        "seed": task.seed,
        "strategy": strategy.name,
        "nodes": len(view.nodes),
        "initial": task.initial,
        "horizon": task.horizon,
        "readouts": [view.nodes[node] for node in view.readouts],
        "target": {view.nodes[node]: value for node, value in view.target},
        "probes": [[(view.nodes[node], value) for node, value in action] for action in task.probes],
        "goal_success": int(goal_success),
        "first_success": first_success,
        "brier_before": _brier(before, truth),
        "brier_after": _brier(after, truth),
        "brier_gain": _brier(before, truth) - _brier(after, truth),
        "constant_brier": after_scores["constant_brier"],
        "excess_brier_before": before_scores["excess_brier"],
        "excess_brier_after": after_scores["excess_brier"],
        "discrimination_before": before_scores["discrimination"],
        "discrimination_after": after_scores["discrimination"],
        "variable_readout_fraction": after_scores["variable_readout_fraction"],
        "probe_truths": [list(row) for row in truth],
        "probe_predictions_before": [list(row) for row in before],
        "probe_predictions_after": [list(row) for row in after],
        "learning_curve": curve,
        "prospective_brier": prospective,
        "oracle_success": 1,
        "reachable_fraction": task.success_rate,
        "final_action": [(view.nodes[node], value) for node, value in final_action],
        "trajectory": [
            {
                "action": [(view.nodes[node], value) for node, value in entry.action],
                "probabilities": list(entry.probabilities),
                "observation": list(entry.values),
            }
            for entry in episode.history
        ],
    }
