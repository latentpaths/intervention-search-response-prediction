import itertools
import json
import tempfile
import unittest
from pathlib import Path

from intervention_search_response_prediction.paths import Layout
from intervention_search_response_prediction.simulation.comparisons import _normalized, _spearman
from intervention_search_response_prediction.simulation.environment import Episode
from intervention_search_response_prediction.simulation.inference import (
    _action,
    _json_content,
    _probabilities,
    agent_view,
    named_action,
    node_labels,
    prompt_for,
)
from intervention_search_response_prediction.simulation.metrics import (
    probe_scores,
    unclamped_discrimination,
)
from intervention_search_response_prediction.simulation.model import BooleanModel, legal_actions
from intervention_search_response_prediction.simulation.procedures import STRATEGIES, run_episode
from intervention_search_response_prediction.simulation.tasks import Task, make_tasks

MODEL = Layout(Path(__file__).resolve().parents[1]).models / "Lac Operon.bnet"


def model_from_text(text: str) -> BooleanModel:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "toy.bnet"
        path.write_text(text)
        return BooleanModel.from_bnet(path)


class ModelTests(unittest.TestCase):
    def test_synchronous_updates_and_clamp(self) -> None:
        model = model_from_text("targets, factors\nA, B\nB, A\nY, A & !B\n")
        initial = 1  # A=1, B=Y=0.
        self.assertEqual(model.simulate(initial, (), 1), 6)
        self.assertEqual(model.simulate(initial, (), 2), 1)
        self.assertEqual(model.simulate(initial, ((0, 1),), 1), 7)
        self.assertEqual(model.simulate(initial, ((0, 1),), 2), 3)

    def test_operator_precedence_and_sign(self) -> None:
        model = model_from_text("targets, factors\nA, A\nB, B\nC, C\nY, !A & B | C\n")
        self.assertEqual((model.simulate(0b0010, (), 1) >> 3) & 1, 1)
        self.assertEqual((model.simulate(0b0011, (), 1) >> 3) & 1, 0)
        self.assertEqual((model.simulate(0b0100, (), 1) >> 3) & 1, 1)
        self.assertIn((0, 3, -1), model.signed_edges())
        self.assertIn((2, 3, 1), model.signed_edges())

    def test_rejects_non_boolean_python(self) -> None:
        with self.assertRaises((ValueError, SyntaxError)):
            model_from_text("targets, factors\nA, __import__('os')\n")


class EpisodeTests(unittest.TestCase):
    def test_prediction_precedes_assay_and_probes_are_withheld(self) -> None:
        model = model_from_text("targets, factors\nA, A\nB, B\nY, A & !B\n")
        actions = legal_actions(3, frozenset({2}))
        task = Task(model, 0, 0, 1, (0, 1, 2), ((2, 1),), actions, (actions[0],), 0.1)
        episode = Episode(task, budget=1)
        self.assertNotIn(actions[0], episode.inspect_network().menu)
        with self.assertRaises(ValueError):
            episode.experiment(actions[1])
        with self.assertRaises(ValueError):
            episode.predict(actions[0], (0.5, 0.5, 0.5))
        episode.predict(actions[1], (0.5, 0.5, 0.5))
        with self.assertRaises(ValueError):
            episode.predict(actions[2], (0.5, 0.5, 0.5))
        observation = episode.experiment(actions[1])
        self.assertEqual(len(observation.values), 3)
        with self.assertRaises(ValueError):
            episode.predict(actions[2], (0.5, 0.5, 0.5))
        self.assertIsInstance(episode.submit(actions[0]), bool)

    def test_task_generation_is_deterministic(self) -> None:
        model = BooleanModel.from_bnet(MODEL)
        first = make_tasks(model, 0)
        second = make_tasks(model, 0)
        self.assertEqual(first, second)
        for task in first:
            self.assertEqual(len(task.probes), 10)
            self.assertEqual(len(set(task.probes)), 10)
            self.assertTrue(set(task.probes).issubset(set(task.actions)))
            self.assertTrue(
                all(
                    not ({node for node, _ in action} & {node for node, _ in task.target})
                    for action in task.actions
                )
            )

    def test_graph_prediction_ablation_fixes_assays_and_final_action(self) -> None:
        model = BooleanModel.from_bnet(MODEL)
        task = make_tasks(model, 0)[0]
        strategies = {strategy.name: strategy for strategy in STRATEGIES}
        mean = run_episode(task, strategies["graph-mean"])
        ridge = run_episode(task, strategies["graph-ridge-observed"])
        self.assertEqual(
            [entry["action"] for entry in mean["trajectory"]],
            [entry["action"] for entry in ridge["trajectory"]],
        )
        self.assertEqual(mean["final_action"], ridge["final_action"])
        self.assertEqual(mean["goal_success"], ridge["goal_success"])


class MetricTests(unittest.TestCase):
    def test_action_independent_and_oracle_constant(self) -> None:
        truth = ((0,), (0,), (1,), (1,))
        fixed = probe_scores(((0.5,),) * 4, truth)
        self.assertAlmostEqual(fixed["brier"], 0.25)
        self.assertAlmostEqual(fixed["constant_brier"], 0.25)
        self.assertAlmostEqual(fixed["excess_brier"], 0.0)
        self.assertAlmostEqual(fixed["discrimination"], 0.0)

    def test_perfect_and_reversed_predictions(self) -> None:
        truth = ((0,), (0,), (1,), (1,))
        perfect = probe_scores(tuple(tuple(float(y) for y in row) for row in truth), truth)
        reversed_score = probe_scores(tuple((float(1 - row[0]),) for row in truth), truth)
        self.assertAlmostEqual(perfect["brier"], 0.0)
        self.assertAlmostEqual(perfect["discrimination"], 0.5)
        self.assertAlmostEqual(reversed_score["brier"], 1.0)
        self.assertAlmostEqual(reversed_score["discrimination"], -0.5)

    def test_constant_outcomes_have_no_discrimination(self) -> None:
        score = probe_scores(((0.0,), (0.2,), (0.8,), (1.0,)), ((1,),) * 4)
        self.assertEqual(score["constant_brier"], 0.0)
        self.assertAlmostEqual(score["discrimination"], 0.0)
        self.assertEqual(score["variable_readout_fraction"], 0.0)

    def test_discrimination_equals_permutation_loss_difference(self) -> None:
        truth = ((0,), (1,), (0,), (1,))
        forecasts = ((0.1,), (0.8,), (0.3,), (0.7,))
        observed = probe_scores(forecasts, truth)
        permuted = [
            probe_scores(tuple(forecasts[i] for i in order), truth)["brier"]
            for order in itertools.permutations(range(4))
        ]
        self.assertAlmostEqual(
            sum(permuted) / len(permuted) - observed["brier"], observed["discrimination"]
        )

    def test_unclamped_discrimination_omits_direct_readout_clamps(self) -> None:
        forecasts = ((0.0,), (1.0,), (0.0,), (1.0,))
        truths = ((0,), (1,), (0,), (1,))
        probes = ((("R", 0),), (("R", 1),), (("A", 0),), (("A", 1),))
        discrimination, excluded = unclamped_discrimination(forecasts, truths, probes, ("R",))
        self.assertAlmostEqual(discrimination, 0.5)
        self.assertAlmostEqual(excluded, 0.5)


class LanguageViewTests(unittest.TestCase):
    def test_strict_response_parsing(self) -> None:
        self.assertIsNone(_json_content("```json\n{}\n```"))
        self.assertIsNone(_json_content("[]"))
        self.assertEqual(_json_content('{"action":[]}'), {"action": []})
        self.assertIsNone(_probabilities([True, 0.5], 2))
        self.assertIsNone(_probabilities([0.5, 1.1], 2))

    def test_anonymization_preserves_actions_and_hides_names(self) -> None:
        task = make_tasks(
            BooleanModel.from_bnet(MODEL),
            0,
        )[0]
        view = Episode(task).inspect_network()
        labels = node_labels(task, True)
        self.assertEqual(len(set(labels)), len(labels))
        self.assertTrue(all(label.startswith("node_") for label in labels))
        original_action = task.actions[0]
        self.assertEqual(
            _action(named_action(original_action, labels), labels, task.actions), original_action
        )
        serialized = json.dumps(agent_view(view, task, labels))
        for name in task.model.nodes:
            self.assertNotIn(f'"{name}"', serialized)
        for action in task.probes:
            self.assertEqual(
                task.response(action),
                task.response(_action(named_action(action, labels), labels, task.actions)),
            )

    def test_agent_prompt_contains_no_rules_or_probe_outcomes(self) -> None:
        task = make_tasks(
            BooleanModel.from_bnet(MODEL),
            0,
        )[0]
        view = Episode(task).inspect_network()
        data = agent_view(view, task, node_labels(task, True))
        system, prompt = prompt_for("probe", data, [], "history", "", [])
        parsed = json.loads(prompt)
        self.assertNotIn("model", parsed["task"])
        self.assertNotIn("probe_truths", prompt)
        self.assertNotIn("rules", parsed["task"])
        self.assertNotIn(task.model.name, prompt)
        self.assertIn("hidden", system)

    def test_posthoc_prompt_asks_for_observations_only(self) -> None:
        system, message = prompt_for("assay", {"nodes": ["x"]}, [], "summary_note", "prior")
        request = json.loads(message)
        self.assertIn("brief summary of what you have observed so far", system)
        self.assertEqual(request["previous_note"], "prior")
        self.assertIn('"note"', request["request"])
        self.assertNotIn("hypothesis", system + message)


class ExtendedAnalysisTests(unittest.TestCase):
    def test_normalized_discrimination_excludes_zero_denominator(self) -> None:
        self.assertIsNone(_normalized({"constant_brier": 0, "discrimination_after": 0}))
        self.assertAlmostEqual(
            _normalized({"constant_brier": 0.2, "discrimination_after": 0.1}), 0.25
        )

    def test_spearman_handles_tied_binary_goal_and_constant_forecast(self) -> None:
        rows = [
            {"discrimination_after": value, "goal_success": success}
            for value, success in ((0, 0), (1, 0), (2, 1), (3, 1))
        ]
        self.assertAlmostEqual(_spearman(rows), 2 / 5**0.5)
        for row in rows:
            row["discrimination_after"] = 0
        self.assertIsNone(_spearman(rows))


if __name__ == "__main__":
    unittest.main()
