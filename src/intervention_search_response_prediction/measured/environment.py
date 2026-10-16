"""Forecast commitment, assay feedback, and final submission for measured tasks."""

import numpy as np


class MeasuredEpisode:
    def __init__(self, task):
        self.task = task
        self.history = []
        self.pending = None
        self.ended = False

    def predict(self, action, p):
        if self.ended or len(self.history) >= 8 or self.pending is not None:
            raise ValueError("episode closed or pending forecast")
        if action not in self.task.candidates or action in [r["action"] for r in self.history]:
            raise ValueError("illegal/repeated/probe action")
        a = np.asarray(p, dtype=float)
        if a.shape != (len(self.task.readouts),) or not np.isfinite(a).all():
            raise ValueError("invalid forecast")
        self.pending = (action, a.tolist())

    def experiment(self, action):
        if self.pending is None or action != self.pending[0]:
            raise ValueError("forecast must be committed first")
        r = {
            "action": action,
            "prediction": self.pending[1],
            "value": list(self.task.records[action]["value"]),
            "replicate_ids": self.task.records[action].get("replicate_ids", []),
        }
        self.history.append(r)
        self.pending = None
        return dict(r)

    def submit(self, action):
        if (
            self.ended
            or self.pending is not None
            or len(self.history) != 8
            or action not in self.task.candidates
        ):
            raise ValueError("invalid submission")
        self.ended = True
        return list(self.task.records[action]["value"])
