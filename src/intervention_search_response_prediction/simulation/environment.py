"""Agent-visible episode interface and evaluator-only scoring."""

from __future__ import annotations

from dataclasses import dataclass

from .model import Action
from .tasks import Task


@dataclass(frozen=True)
class View:
    nodes: tuple[str, ...]
    initial: int
    horizon: int
    readouts: tuple[int, ...]
    target: tuple[tuple[int, int], ...]
    signed_edges: tuple[tuple[int, int, int], ...]
    menu: tuple[Action, ...]
    final_actions: tuple[Action, ...]


@dataclass(frozen=True)
class Observation:
    action: Action
    probabilities: tuple[float, ...]
    values: tuple[int, ...]


class Episode:
    def __init__(self, task: Task, budget: int = 8):
        self._task = task
        self.budget = budget
        self.menu = tuple(action for action in task.actions if action not in frozenset(task.probes))
        if len(self.menu) < budget:
            raise ValueError("too few non-probe interventions")
        self.history: list[Observation] = []
        self._pending: tuple[Action, tuple[float, ...]] | None = None
        self._submitted: Action | None = None

    def inspect_network(self) -> View:
        task = self._task
        return View(
            task.model.nodes,
            task.initial,
            task.horizon,
            task.readouts,
            task.target,
            task.model.signed_edges(),
            self.menu,
            task.actions,
        )

    def predict(self, action: Action, probabilities: tuple[float, ...]) -> None:
        if self._submitted is not None:
            raise ValueError("episode ended")
        if self._pending is not None:
            raise ValueError("the pending prediction cannot be revised")
        if len(self.history) >= self.budget:
            raise ValueError("assay budget exhausted")
        if action not in self.menu or action in {entry.action for entry in self.history}:
            raise ValueError("invalid, reserved, or repeated assay")
        if len(probabilities) != len(self._task.readouts):
            raise ValueError("one probability is required for each readout")
        if not all(0 <= value <= 1 for value in probabilities):
            raise ValueError("probabilities must lie in [0, 1]")
        self._pending = (action, tuple(float(x) for x in probabilities))

    def experiment(self, action: Action) -> Observation:
        if self._pending is None or self._pending[0] != action:
            raise ValueError("commit a prediction for this action first")
        _, probabilities = self._pending
        state = self._task.response(action)
        values = tuple((state >> node) & 1 for node in self._task.readouts)
        entry = Observation(action, probabilities, values)
        self.history.append(entry)
        self._pending = None
        return entry

    def submit(self, action: Action) -> bool:
        if self._submitted is not None:
            raise ValueError("episode ended")
        if action not in self._task.actions:
            raise ValueError("invalid final intervention")
        self._submitted = action
        return self._task.succeeds(action)

    def probe_truths(self) -> tuple[tuple[int, ...], ...]:
        """Evaluator only. Never expose this on an agent's View."""
        return tuple(
            tuple((self._task.response(action) >> node) & 1 for node in self._task.readouts)
            for action in self._task.probes
        )
