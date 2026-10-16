"""Frozen scientific inputs and current-source manifests for measured studies."""

import hashlib
import json
import time
from pathlib import Path

from ..io import logical_sha256, read_json, write_json
from ..paths import FROZEN_AT, MEASURED, Layout, PreparedData, repository_root, writable_output
from .tasks import VERSION, tasks

PROTOCOL = {
    "version": VERSION,
    "seeds": list(range(5)),
    "assay_budget": 8,
    "schmidt_candidates": 128,
    "schmidt_probes": 32,
    "norman_probe_fraction": 0.2,
    "norman_min_cells": 30,
    "norman_readouts": 8,
    "norman_readout_selection": "largest variance across disclosed single perturbation deltas; Ensembl ID tie break",
    "normalization": "Norman log2(1+1e6*pooled UMI/pooled library UMI), subtract control; Schmidt source LFC unchanged",
    "primary": "hypothesis minus summary on fixed random histories, final probe MSE; each screen separately",
    "secondary": [
        "discrimination",
        "adaptive search",
        "shuffled fixed feedback",
        "Norman additive residual error",
    ],
    "conditions": ["summary", "hypothesis"],
    "modes": ["fixed", "adaptive", "shuffled"],
    "ridge": "Gaussian linear posterior mean, unit observation variance and prior precision 2; additive prior for Norman, zero for Schmidt",
    "repair": "one schema-only retry; then observed-data-only fallback; failures included",
    "note_characters": 600,
    "max_output_tokens": 1536,
    "model": "openai/gpt-6-sol",
    "temperature": 0,
    "reasoning_effort": "none",
    "smoke_cap_usd": 5,
    "study_cap_usd": 45,
    "combined_cap_usd": 50,
    "uncertainty": "split mean, SD and full range; overlapping splits are not biological replicates",
    "source_warning": "Public historical outcomes may occur in LLM training data; no contamination-free claim.",
}


def protocol_payload(task_list, data_dir=None):
    """Scientific inputs and disclosed task views, independent of source layout."""
    data_dir = Path(data_dir) if data_dir is not None else Layout(repository_root()).data
    return {
        "protocol": PROTOCOL,
        "sources": {
            name: logical_sha256(path)
            for name, path in PreparedData(data_dir).historical_sources.items()
        },
        "tasks": [
            {
                "id": t.id,
                "candidates": t.candidates,
                "probes": t.probes,
                "fixed_actions": t.fixed_actions(),
                "readouts": t.readouts,
                "view_sha256": hashlib.sha256(
                    json.dumps(t.view(), sort_keys=True).encode()
                ).hexdigest(),
            }
            for t in task_list
        ],
    }


def verify_historical_protocol(data_dir=None, input_dir=None):
    """Recompute frozen scientific inputs without certifying historical execution."""

    input_dir = Path(input_dir) if input_dir is not None else Layout(repository_root()).measured
    task_list = tasks(data_dir)
    frozen = read_json(input_dir / "protocol.json")
    expected = {key: value for key, value in frozen.items() if key != "code_sha256"}
    actual = json.loads(json.dumps(protocol_payload(task_list, data_dir)))
    if actual != expected:
        raise ValueError("Scientific inputs differ from the historical frozen protocol")
    return task_list


def freeze(output_dir=None, data_dir=None):
    """Freeze current sources for a fresh run; never amend the historical study."""

    output_dir = writable_output(
        output_dir if output_dir is not None else repository_root() / f"outputs/{MEASURED}"
    )
    task_list = tasks(data_dir)
    payload = protocol_payload(task_list, data_dir)
    package = Path(__file__).resolve().parents[1]
    payload["runtime_sha256"] = {
        str(path.relative_to(package)): logical_sha256(path)
        for path in sorted(package.rglob("*.py"))
    }
    payload = json.loads(json.dumps(payload))
    path = output_dir / "protocol.json"
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(
                "Frozen run differs; preserve prior results and use a new output directory"
            )
    else:
        write_json(path, payload)
        write_json(
            output_dir / FROZEN_AT,
            {"unix_time": time.time(), "protocol_sha256": logical_sha256(path)},
        )
    return task_list
