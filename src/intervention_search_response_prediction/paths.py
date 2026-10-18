"""Canonical input locations and protection for retained scientific records.

Constructing paths does not discover a checkout. Only command defaults resolve
one, so installed scientific objects and command help work from any directory.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

BASELINES = "simulation-baselines"
NOTES = "simulation-notes"
MEASURED = "measured-replay"
EPISODES = "episodes.jsonl"
TASKS = "tasks.json"
SELECTION = "task-selection.json"
NOTE_CALLS = "inference-calls.jsonl"
BASELINE_EPISODES = "baseline-episodes.json"
PAIRED_BLOCKS = "paired-blocks.jsonl"
INFERENCE_LEDGER = "inference-ledger.jsonl"
PROTOCOL = "protocol.json"
FROZEN_AT = "frozen-at.json"
REFERENCE_SUMMARY = "reference-summary.json"
REFERENCE_COMPARISONS = "reference-comparisons.json"
REFERENCE_SENSITIVITY = "reference-readout-sensitivity.json"


@dataclass(frozen=True)
class PreparedData:
    """Prepared inputs, with historical logical names preserved for auditing."""

    directory: Path

    @property
    def responses(self) -> Path:
        # I/O resolves the lossless .gz archive while hashing decompressed JSON.
        return self.directory / "tcell-cytokine-screens/responses.json"

    @property
    def counts(self) -> Path:
        return self.directory / "k562-combinations/pooled-counts.npz"

    @property
    def metadata(self) -> Path:
        return self.directory / "k562-combinations/metadata.json"

    @property
    def historical_sources(self) -> dict[str, Path]:
        return {
            "schmidt.json": self.responses,
            "norman_metadata.json": self.metadata,
            "norman_counts.npz": self.counts,
        }

    def raw_source(self, metadata: Path, recorded_name: str) -> Path:
        """Translate an embedded historical raw path without changing evidence."""
        return metadata.parent / "raw" / Path(recorded_name).name


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def models(self) -> Path:
        return self.data / "boolean-models"

    def study(self, name: str) -> Path:
        return self.root / "experiments" / name

    @property
    def baselines(self) -> Path:
        return self.study(BASELINES)

    @property
    def notes(self) -> Path:
        return self.study(NOTES)

    @property
    def measured(self) -> Path:
        return self.study(MEASURED)

    @property
    def checksums(self) -> Path:
        return self.root / "experiments" / "checksums.json"


def repository_root(path: Path | None = None) -> Path:
    """Support editable installs, installed wheels, and relocated checkouts."""
    if path is not None:
        return Path(path).expanduser().resolve()
    if override := os.environ.get("ISRP_ROOT"):
        return Path(override).expanduser().resolve()
    candidates = (Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2])
    for directory in candidates:
        metadata = directory / "pyproject.toml"
        if not metadata.is_file() or not Layout(directory).checksums.is_file():
            continue
        try:
            project = tomllib.loads(metadata.read_text()).get("project", {})
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if project.get("name") == "intervention-search-response-prediction":
            return directory
    raise RuntimeError("Run from an ISRP checkout or set ISRP_ROOT to its directory")


def writable_output(path: Path, root: Path | None = None) -> Path:
    """Reject outputs that could modify source files, inputs, or retained evidence."""
    output = Path(path).expanduser().resolve()
    root = repository_root(root)
    for protected in (
        root / name
        for name in (
            "paper.pdf",
            "paper",
            "experiments",
            "docs",
            "pyproject.toml",
            "data",
            "src",
            "tests",
        )
    ):
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError(f"Output overlaps protected repository content: {output}")
    return output
