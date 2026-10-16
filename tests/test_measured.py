import copy
import itertools
import json
import tempfile
import unittest

import numpy as np

from intervention_search_response_prediction.measured.environment import MeasuredEpisode
from intervention_search_response_prediction.measured.inference import BudgetExceeded, Calls, valid
from intervention_search_response_prediction.measured.metrics import continuous_scores
from intervention_search_response_prediction.measured.procedures import Ridge
from intervention_search_response_prediction.measured.tasks import MeasuredTask, tasks
from intervention_search_response_prediction.paths import PreparedData


def toy():
    ids = [f"g{i}" for i in range(12)]
    return MeasuredTask(
        "test",
        "test",
        "cells",
        "CRISPRa",
        "units",
        ["r"],
        {a: {"genes": [a], "value": [float(i)], "replicate_ids": []} for i, a in enumerate(ids)},
        tuple(ids[:10]),
        tuple(ids[10:]),
        {a: [1.0, i % 2] for i, a in enumerate(ids)},
        {a: [0.0] for a in ids},
        {},
        0,
    )


class MeasuredTests(unittest.TestCase):
    def test_prediction_commitment_and_legality(self):
        e = MeasuredEpisode(toy())
        with self.assertRaises(ValueError):
            e.experiment("g0")
        with self.assertRaises(ValueError):
            e.predict("g10", [0.0])
        with self.assertRaises(ValueError):
            e.predict("g0", [float("nan")])
        e.predict("g0", [0.1])
        with self.assertRaises(ValueError):
            e.predict("g0", [0.2])
        e.experiment("g0")
        with self.assertRaises(ValueError):
            e.predict("g0", [0.0])
        for a in e.task.candidates[1:8]:
            e.predict(a, [0.0])
            e.experiment(a)
        with self.assertRaises(ValueError):
            e.predict("g8", [0.0])
        e.submit("g9")
        with self.assertRaises(ValueError):
            e.submit("g9")

    def test_view_does_not_contain_truth(self):
        t = toy()
        before = t.view()
        for r in t.records.values():
            r["value"] = [999999.0]
        self.assertEqual(before, t.view())
        self.assertNotIn("999999", json.dumps(t.view()))

    def test_permutation_identity(self):
        p = np.array([[1.0, 3.0], [2.0, -1.0], [4.0, 5.0]])
        y = np.array([[2.0, 0.0], [1.0, 3.0], [6.0, -2.0]])
        score = continuous_scores(p, y)
        shuffled = np.mean(
            [continuous_scores(p[list(ix)], y)["mse"] for ix in itertools.permutations(range(3))]
        )
        self.assertAlmostEqual(score["discrimination"], shuffled - score["mse"])
        self.assertAlmostEqual(continuous_scores(np.ones_like(p), y)["discrimination"], 0.0)

    def test_ridge_matches_primal_posterior(self):
        t = toy()
        history = [{"action": "g1", "value": [2.0]}, {"action": "g2", "value": [1.0]}]
        pred, var = Ridge(t).predict(t.probes, history)
        x = t.matrix(["g1", "g2"])
        q = t.matrix(t.probes)
        inv = np.linalg.inv(x.T @ x + 2 * np.eye(2))
        np.testing.assert_allclose(pred, q @ inv @ x.T @ np.array([[2.0], [1.0]]))
        np.testing.assert_allclose(var, np.sum((q @ inv) * q, axis=1))

    def test_validator(self):
        self.assertFalse(
            valid({"action": "g0", "prediction": [True], "note": ""}, "assay", 1, ["g0"], 1)
        )
        self.assertFalse(valid({"predictions": [[float("inf")]]}, "probe", 1, [], 1))
        self.assertFalse(valid({"action": "g99"}, "final", 1, ["g0"], 1))

    def test_budget_reserves_and_recovers_unknown_request(self):
        with tempfile.TemporaryDirectory() as d:
            c = Calls(d, client=object())
            c.reserve_block("a", "study", 44.0)
            with self.assertRaises(BudgetExceeded):
                c.reserve_block("b", "study", 2.0)
            c.release_block("a")
            c.reserve_block("s", "smoke", 5.0)
            with self.assertRaises(BudgetExceeded):
                c.reserve_block("s2", "smoke", 0.01)
            c.append({"event": "reserve", "call_id": "lost", "phase": "study", "reserve_usd": 0.25})
            c.process_lock.close()
            resumed = Calls(d, client=object())
            self.assertEqual(resumed.spent["study"], 0.25)
            resumed.process_lock.close()

    def test_combination_outcomes_do_not_fit_readouts_or_features(self):
        from unittest.mock import patch

        from intervention_search_response_prediction.paths import repository_root

        DATA = repository_root() / "data"

        try:
            original = tasks()
            with np.load(PreparedData(DATA).counts) as z:
                counts = z["counts"].copy()
                cells = z["cell_counts"].copy()
            meta = json.loads((PreparedData(DATA).metadata).read_text())
        except FileNotFoundError:
            self.skipTest("source data not prepared")
        for i, k in enumerate(meta["perturbations"]):
            if len(k) == 2:
                counts[i] = 0.0
                counts[i, 0] = 1e9
        with patch(
            "intervention_search_response_prediction.measured.tasks.np.load",
            return_value={"counts": counts, "cell_counts": cells},
        ):
            changed = tasks()
        for a, b in zip(original, changed):
            if a.dataset == "norman":
                self.assertEqual(a.readouts, b.readouts)
                self.assertEqual(a.features, b.features)
                self.assertEqual(a.prior, b.prior)
                self.assertEqual(a.disclosed, b.disclosed)
                self.assertEqual(a.candidates, b.candidates)

    def test_cached_calls_do_not_spend_twice_and_hash_is_checked(self):
        from types import SimpleNamespace

        response = {
            "model": "fake",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 2},
        }
        fake = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(model_dump=lambda: response)
                )
            )
        )
        with tempfile.TemporaryDirectory() as d:
            c = Calls(d, client=fake)
            c.reserve_block("b", "study", 1.0)
            first = c.ask("x", "system", "message", "study", "b")
            second = c.ask("x", "system", "message", "study", "b")
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertAlmostEqual(c.spent["study"], 0.001)
            with self.assertRaises(ValueError):
                c.ask("x", "system", "changed", "study", "b")
            c.process_lock.close()
            resumed = Calls(d, client=fake)
            self.assertAlmostEqual(resumed.spent["study"], 0.001)
            resumed.process_lock.close()

    def test_fixed_histories_are_matched_and_probes_do_not_update_notes(self):
        from intervention_search_response_prediction.measured.inference import run_llm

        class FakeCalls:
            def __init__(self):
                self.requests = []

            def ask(self, call_id, system, message, phase, block):
                msg = json.loads(message)
                self.requests.append((call_id, msg))
                width = len(msg["view"]["readouts"])
                if msg["phase"] == "probe":
                    value = {
                        "predictions": [[0.0] * width for _ in msg["actions"]],
                        "note": "PROBE NOTE MUST NOT BE CARRIED",
                    }
                elif msg["phase"] == "final":
                    value = {"action": msg["legal_actions"][0]}
                else:
                    value = {
                        "action": msg["legal_actions"][0],
                        "prediction": [0.2] * width,
                        "note": "observations only",
                    }
                return {
                    "call_id": call_id,
                    "content": json.dumps(value),
                    "finish_reason": "stop",
                    "charged_usd": 0.0,
                    "cache_hit": False,
                    "latency_seconds": 0.0,
                }

        a = FakeCalls()
        b = FakeCalls()
        t = toy()
        x = run_llm(t, "summary", "fixed", a, "smoke", "a")
        y = run_llm(t, "hypothesis", "fixed", b, "smoke", "b")
        self.assertEqual(x["visible_history"], y["visible_history"])
        for (_, r), (_, s) in zip(a.requests, b.requests):
            self.assertEqual(r["history"], s["history"])
            self.assertNotIn("PROBE NOTE", r["previous_note"])
            self.assertFalse(any(v in r["history"] for v in t.probes))
        z = run_llm(t, "summary", "shuffled", FakeCalls(), "smoke", "c")
        self.assertEqual(
            sorted(v["value"] for v in x["visible_history"]),
            sorted(v["value"] for v in z["visible_history"]),
        )
        self.assertEqual(
            [v["action"] for v in x["visible_history"]], [v["action"] for v in z["visible_history"]]
        )
        self.assertEqual(x["probe_truth"], z["probe_truth"])

    def test_random_predictor_ablation_fixes_assays_and_final_choice(self):
        from intervention_search_response_prediction.measured.procedures import run_classical

        a = run_classical(toy(), "random_mean")
        b = run_classical(toy(), "ridge_random")
        self.assertEqual([h["action"] for h in a["history"]], [h["action"] for h in b["history"]])
        self.assertEqual(a["final"], b["final"])
        self.assertEqual(a["regret"], b["regret"])

    def test_retry_of_uncertain_attempt_preserves_its_reservation(self):
        from types import SimpleNamespace

        response = {
            "model": "fake",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 2},
        }
        fake = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(model_dump=lambda: response)
                )
            )
        )
        with tempfile.TemporaryDirectory() as d:
            c = Calls(d, client=fake)
            c.append({"event": "reserve", "call_id": "lost", "phase": "study", "reserve_usd": 0.25})
            c.process_lock.close()
            resumed = Calls(d, client=fake)
            resumed.reserve_block("b", "study", 1.0)
            resumed.ask("lost", "system", "message", "study", "b")
            self.assertAlmostEqual(resumed.spent["study"], 0.251)
            resumed.process_lock.close()
            again = Calls(d, client=fake)
            self.assertAlmostEqual(again.spent["study"], 0.251)
            again.process_lock.close()

    def test_data_splits_and_normalization_isolation(self):
        try:
            ts = tasks()
        except FileNotFoundError:
            self.skipTest("source data not prepared")
        self.assertEqual(len(ts), 35)
        for t in ts:
            self.assertFalse(set(t.candidates) & set(t.probes))
            self.assertTrue(set(t.fixed_actions()) <= set(t.candidates))
            self.assertEqual(len(set(t.fixed_actions())), 8)
            self.assertTrue(np.isfinite(t.values(t.candidates + t.probes)).all())
            if t.dataset == "norman":
                old = t.view()
                prior = copy.deepcopy(t.prior)
                for a in t.probes:
                    t.records[a]["value"] = [999.0] * len(t.readouts)
                self.assertEqual(t.view(), old)
                self.assertEqual(t.prior, prior)
                for a in t.candidates + t.probes:
                    u, v = t.records[a]["genes"]
                    np.testing.assert_allclose(
                        t.prior[a], np.array(t.disclosed[u]) + t.disclosed[v]
                    )


if __name__ == "__main__":
    unittest.main()
