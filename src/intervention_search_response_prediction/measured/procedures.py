"""Mean, additive, and ridge procedures for measured response replay."""

import hashlib
import time

import numpy as np

from .environment import MeasuredEpisode
from .metrics import result


class Ridge:
    def __init__(self, t):
        self.t = t

    def predict(self, ids, history):
        q = self.t.matrix(ids)
        base = self.t.priors(ids)
        if not history:
            return base, np.sum(q * q, axis=1) / 2
        acts = [r["action"] for r in history]
        x = self.t.matrix(acts)
        y = np.array([r["value"] for r in history]) - self.t.priors(acts)
        kernel = x @ x.T + 2 * np.eye(len(x))
        cross = q @ x.T
        pred = base + cross @ np.linalg.solve(kernel, y)
        variance = np.maximum(
            0,
            (np.sum(q * q, axis=1) - np.sum(cross * np.linalg.solve(kernel, cross.T).T, axis=1))
            / 2,
        )
        return pred, variance


def run_classical(t, agent):
    start = time.perf_counter()
    e = MeasuredEpisode(t)
    ridge = Ridge(t)
    timings = []

    def predict(ids):
        if agent == "additive":
            return t.priors(ids)
        if agent == "random_mean":
            mean = (
                np.mean([r["value"] for r in e.history], axis=0)
                if e.history
                else np.zeros(len(t.readouts))
            )
            return np.tile(mean, (len(ids), 1))
        return ridge.predict(ids, e.history)[0]

    before = predict(t.probes).tolist()
    for a in t.fixed_actions():
        tick = time.perf_counter()
        if agent == "ridge_active":
            unused = [x for x in t.candidates if x not in [r["action"] for r in e.history]]
            _, v = ridge.predict(unused, e.history)
            a = max(
                zip(unused, v),
                key=lambda x: (x[1], hashlib.sha256(f"{t.seed}:{x[0]}".encode()).hexdigest()),
            )[0]
        e.predict(a, predict([a])[0])
        timings.append(time.perf_counter() - tick)
        e.experiment(a)
    after = predict(t.probes).tolist()
    final = (
        max(e.history, key=lambda r: r["value"][0])["action"]
        if t.search
        else e.history[-1]["action"]
    )
    e.submit(final)
    return result(
        t, agent, "classical", before, after, e.history, final, timings, time.perf_counter() - start
    )
