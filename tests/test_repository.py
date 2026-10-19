"""Regression checks for consolidation, immutable evidence, and fresh runs."""

import contextlib
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intervention_search_response_prediction.integrity import (
    sha256,
    verify_evidence,
)
from intervention_search_response_prediction.measured.inference import Calls
from intervention_search_response_prediction.measured.protocol import freeze
from intervention_search_response_prediction.paths import (
    Layout,
    PreparedData,
    writable_output,
)
from intervention_search_response_prediction.simulation.inference import SELECTION_SALT, PrimeCalls

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_package_imports_do_not_discover_a_checkout(self):
        script = """
import importlib
import pkgutil
from unittest.mock import patch
import intervention_search_response_prediction as package
with patch('intervention_search_response_prediction.paths.repository_root',
           side_effect=AssertionError('Repository lookup during import')):
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        importlib.import_module(module.name)
"""
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run([sys.executable, "-c", script], cwd=temp, check=True)

    def test_cli_help_requires_no_repository_or_author_tools(self):
        script = """
import contextlib
import io
import runpy
import sys
from unittest.mock import patch
modules = ['cli', 'simulation.inference', 'simulation.note_analysis',
           'simulation.comparisons', 'simulation.sensitivity', 'simulation.audit',
           'measured.run', 'measured.data', 'measured.analysis', 'measured.audit']
with patch('intervention_search_response_prediction.paths.repository_root',
           side_effect=AssertionError('Repository lookup during help')):
    for module in modules:
        name = 'intervention_search_response_prediction.' + module
        sys.argv = [name, '--help']
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                runpy.run_module(name, run_name='__main__')
            except SystemExit as error:
                assert error.code == 0, (name, error.code)
            else:
                raise AssertionError('Missing help exit: ' + name)
"""
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run([sys.executable, "-c", script], cwd=temp, check=True)

    def test_modified_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = root / "experiments/sample.json"
            sample.parent.mkdir(parents=True)
            sample.write_text("{}")
            manifest = {"experiments/sample.json": sha256(sample)}
            (root / "experiments").mkdir(exist_ok=True)
            (root / "experiments/checksums.json").write_text(json.dumps(manifest))
            sample.write_text('{"changed":true}')
            from intervention_search_response_prediction.reproduction import validate

            with (
                patch(
                    "intervention_search_response_prediction.simulation.audit.audit"
                ) as simulation,
                patch("intervention_search_response_prediction.measured.audit.audit") as measured,
                self.assertRaisesRegex(ValueError, "Retained evidence checksum mismatch"),
            ):
                validate(root)
            simulation.assert_not_called()
            measured.assert_not_called()

    def test_missing_public_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "experiments").mkdir(exist_ok=True)
            (root / "experiments/checksums.json").write_text(
                json.dumps({"data/missing.json": "absent"})
            )
            with self.assertRaisesRegex(ValueError, "Retained evidence checksum mismatch"):
                verify_evidence(root)

    def test_checkout_discovery_requires_matching_project_metadata(self):
        script = """
from unittest.mock import patch
from pathlib import Path
from intervention_search_response_prediction.paths import repository_root
with patch('intervention_search_response_prediction.paths.__file__', str(Path.cwd() / 'src/package/paths.py')):
    try:
        repository_root()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Discovered an unrelated project')
"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "experiments").mkdir(exist_ok=True)
            (root / "experiments/checksums.json").write_text("{}")
            (root / "pyproject.toml").write_text('[project]\nname = "unrelated"\n')
            environment = {key: value for key, value in os.environ.items() if key != "ISRP_ROOT"}
            subprocess.run([sys.executable, "-c", script], cwd=root, env=environment, check=True)

    def test_outputs_cannot_overlap_retained_inputs_or_paper(self):
        for path in (
            ROOT,
            ROOT / "experiments",
            ROOT / "experiments/simulation-notes/new",
            ROOT / "src",
            ROOT / "data",
            ROOT / "experiments/checksums.json",
            ROOT / "pyproject.toml",
            ROOT / "paper.pdf",
            ROOT / "paper/figures",
            ROOT / "docs",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                writable_output(path)
        self.assertEqual(writable_output(ROOT / "outputs/test"), ROOT / "outputs/test")

    def test_output_symlinks_cannot_bypass_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            alias = Path(temp) / "evidence"
            alias.symlink_to(ROOT / "experiments", target_is_directory=True)
            with self.assertRaises(ValueError):
                writable_output(alias / "llm/new")

    def test_inference_caches_cannot_write_to_retained_results(self):
        with self.assertRaises(ValueError):
            PrimeCalls(ROOT / "experiments/simulation-notes/inference-calls.jsonl")
        with self.assertRaises(ValueError):
            Calls(ROOT / "experiments/measured-replay", client=object())

    def test_historical_selection_salt_is_preserved(self):
        selection = json.loads(
            (ROOT / "experiments/simulation-notes/task-selection.json").read_text()
        )
        self.assertEqual(SELECTION_SALT, selection["salt"])

    def test_fresh_classical_run_can_be_audited_without_inference(self):
        from intervention_search_response_prediction.io import write_json as save
        from intervention_search_response_prediction.measured.audit import audit
        from intervention_search_response_prediction.measured.procedures import run_classical

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp).resolve()
            task_list = freeze(output)
            save(output / "baseline-episodes.json", [run_classical(task_list[0], "random_mean")])
            result = audit(output)
            self.assertTrue(result["passed"])
            self.assertEqual(result["episodes"], 1)
            self.assertEqual(result["completed_blocks"], 0)

    def test_fresh_freeze_does_not_reuse_historical_execution_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp).resolve()
            task_list = freeze(output)
            self.assertEqual(len(task_list), 35)
            manifest = json.loads((output / "protocol.json").read_text())
            self.assertIn("runtime_sha256", manifest)
            self.assertNotIn("code_sha256", manifest)
            self.assertIn("measured/protocol.py", manifest["runtime_sha256"])
            self.assertEqual(len(freeze(output)), 35)
            manifest["runtime_sha256"]["measured/protocol.py"] = "changed"
            (output / "protocol.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Frozen run differs"):
                freeze(output)

    def test_installed_cli_discovers_relocated_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy2(ROOT / "pyproject.toml", root / "pyproject.toml")
            (root / "experiments").mkdir(exist_ok=True)
            (root / "experiments/checksums.json").write_text("{}")
            environment = {key: value for key, value in os.environ.items() if key != "ISRP_ROOT"}
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from intervention_search_response_prediction.paths import repository_root; print(repository_root())",
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(Path(result.stdout.strip()).resolve(), root.resolve())

    def test_compact_inputs_keep_original_logical_hashes_after_relocation(self):
        from intervention_search_response_prediction.io import logical_sha256

        prepared = PreparedData(Layout(ROOT).data)
        frozen = json.loads((Layout(ROOT).measured / "protocol.json").read_text())
        for original, path in prepared.historical_sources.items():
            self.assertEqual(logical_sha256(path), frozen["sources"][original])
        with gzip.open(str(prepared.responses) + ".gz", "rb") as stream:
            self.assertEqual(
                hashlib.sha256(stream.read()).hexdigest(), frozen["sources"]["schmidt.json"]
            )

    def test_prepared_input_root_can_be_relocated(self):
        from intervention_search_response_prediction.measured.protocol import protocol_payload
        from intervention_search_response_prediction.measured.tasks import tasks

        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            for source in PreparedData(Layout(ROOT).data).historical_sources.values():
                actual = source if source.exists() else source.with_name(source.name + ".gz")
                target = data / actual.relative_to(Layout(ROOT).data)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(actual, target)
            frozen = json.loads((Layout(ROOT).measured / "protocol.json").read_text())
            expected = {key: value for key, value in frozen.items() if key != "code_sha256"}
            actual = json.loads(json.dumps(protocol_payload(tasks(data), data)))
            self.assertEqual(actual, expected)

    def test_figures_uses_explicit_study_input_and_output(self):
        from intervention_search_response_prediction.cli import main

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp).resolve()
            inputs = Layout(ROOT).baselines
            with (
                patch.object(
                    sys,
                    "argv",
                    ["isrp", "figures", "--input", str(inputs), "--output", str(output)],
                ),
                patch(
                    "intervention_search_response_prediction.simulation.analysis.write_analysis",
                    return_value={"networks": 39, "tasks": 232, "episodes": 1392},
                ) as analyze,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main()
            analyze.assert_called_once_with(
                inputs / "episodes.jsonl", output / "figures", output / "summary.json", ROOT
            )

    def test_ambiguous_results_flag_is_rejected(self):
        from intervention_search_response_prediction.cli import main

        with (
            patch.object(sys, "argv", ["isrp", "figures", "--results", str(ROOT)]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 2)

    def test_offline_validation_never_changes_evidence_or_calls_provider(self):
        from intervention_search_response_prediction.reproduction import validate

        manifest = json.loads((ROOT / "experiments/checksums.json").read_text())
        paper_hash = sha256(ROOT / "paper.pdf")
        original_open = Path.open

        def public_open(path, *args, **kwargs):
            if path.resolve().is_relative_to(ROOT / "paper"):
                raise AssertionError(f"Public validation read the local workspace: {path}")
            return original_open(path, *args, **kwargs)

        with (
            patch.object(Path, "open", public_open),
            patch(
                "socket.socket.connect", side_effect=AssertionError("Unexpected network request")
            ),
            patch.object(PrimeCalls, "__init__", side_effect=AssertionError("Inference invoked")),
            patch.object(Calls, "__init__", side_effect=AssertionError("Inference invoked")),
        ):
            result = validate()
        self.assertTrue(result["passed"])
        self.assertEqual(result["scope"], "public file integrity and scientific checks")
        self.assertEqual(result["integrity"]["scope"], "public file integrity")
        self.assertNotIn("provenance", result)
        self.assertEqual(
            result["simulation"]["classical"], {"models": 39, "tasks": 232, "episodes": 1392}
        )
        self.assertEqual(result["measured"]["completed_blocks"], 105)
        unavailable = result["measured"]["raw_source_files_unavailable"]
        self.assertEqual(len(unavailable), 5)
        prepared = PreparedData(Layout(ROOT).data)
        for path in unavailable:
            self.assertIn(
                Path(path).parent,
                (prepared.responses.parent / "raw", prepared.metadata.parent / "raw"),
            )
        self.assertEqual(sha256(ROOT / "paper.pdf"), paper_hash)
        for relative, expected in manifest.items():
            self.assertEqual(sha256(ROOT / relative), expected, relative)


if __name__ == "__main__":
    unittest.main()
