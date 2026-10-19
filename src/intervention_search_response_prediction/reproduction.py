"""Offline verification and regeneration of the three recorded studies."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import (
    BASELINES,
    EPISODES,
    MEASURED,
    NOTES,
    REFERENCE_COMPARISONS,
    REFERENCE_SENSITIVITY,
    REFERENCE_SUMMARY,
    Layout,
    repository_root,
    writable_output,
)


def validate(root: Path | None = None) -> dict:
    root = repository_root(root)
    from .integrity import verify_evidence
    from .measured.audit import audit as measured_audit
    from .simulation.audit import audit

    layout = Layout(root)
    integrity = verify_evidence(root)
    return {
        "passed": True,
        "scope": "public file integrity and scientific checks",
        "integrity": integrity,
        "simulation": audit(root),
        "measured": measured_audit(layout.measured, layout.data),
    }


def _require_equal(actual: dict, reference: Path) -> None:
    expected = json.loads(reference.read_text())
    if actual != expected:
        raise ValueError(f"Reproduced values differ from retained summary: {reference}")


def reproduce(output: Path, root: Path | None = None) -> dict:
    root = repository_root(root)
    from .io import read_jsonl, write_json
    from .measured.analysis import collect, figures
    from .simulation.analysis import summarize, write_figures
    from .simulation.comparisons import analyze as comparisons
    from .simulation.note_analysis import analyze as notes
    from .simulation.note_analysis import write_figures as note_figures
    from .simulation.sensitivity import analyze as sensitivity

    output = writable_output(output, root)
    checks = validate(root)
    layout = Layout(root)
    baseline_rows = read_jsonl(layout.baselines / EPISODES)
    note_rows = read_jsonl(layout.notes / EPISODES)
    summary = summarize(baseline_rows)
    note_summary = notes(layout.notes / EPISODES, layout.baselines / EPISODES)
    comparison_summary = comparisons(baseline_rows, note_rows)
    sensitivity_summary = sensitivity(root)
    _require_equal(summary, layout.baselines / REFERENCE_SUMMARY)
    _require_equal(note_summary, layout.notes / REFERENCE_SUMMARY)
    _require_equal(comparison_summary, layout.notes / REFERENCE_COMPARISONS)
    _require_equal(sensitivity_summary, layout.notes / REFERENCE_SENSITIVITY)
    rows, measured = collect(layout.measured, layout.data)
    # This field describes the historical reporting host, not a scientific result.
    reference = json.loads((layout.measured / REFERENCE_SUMMARY).read_text())
    measured["hardware"] = reference["hardware"]
    _require_equal(measured, layout.measured / REFERENCE_SUMMARY)

    write_json(output / BASELINES / "summary.json", summary)
    write_json(output / NOTES / "summary.json", note_summary)
    write_json(output / NOTES / "comparisons.json", comparison_summary)
    write_json(output / NOTES / "readout-sensitivity.json", sensitivity_summary)
    write_json(output / MEASURED / "summary.json", measured)
    write_figures(summary, output / BASELINES / "figures", root)
    note_figures(note_summary, output / NOTES / "figures", root)
    figures(rows, measured, output / MEASURED / "figures", root)
    write_json(output / "validation.json", checks)
    return {
        "passed": True,
        "output": str(output),
        **{key: summary[key] for key in ("networks", "tasks", "episodes")},
    }
