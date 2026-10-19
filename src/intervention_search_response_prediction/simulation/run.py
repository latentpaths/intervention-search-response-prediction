"""Deterministic Boolean baseline execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..paths import EPISODES, TASKS, writable_output
from .analysis import write_analysis
from .procedures import STRATEGIES, run_episode
from .tasks import load_models, make_tasks


def _verify_models(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {entry["file"]: entry["sha256"] for entry in manifest["models"]}
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.glob("*.bnet")
    }
    if actual != expected:
        raise ValueError("model files do not match the pinned source manifest")
    return manifest


def run(
    models_dir: Path,
    output_dir: Path,
    figures_dir: Path,
    seeds: tuple[int, ...],
    root: Path | None = None,
) -> dict:
    output_dir = writable_output(output_dir, root)
    figures_dir = writable_output(figures_dir, root)
    source = _verify_models(models_dir)
    tasks = tuple(
        task
        for model in load_models(models_dir)
        for seed in seeds
        for task in make_tasks(model, seed)
    )
    if not tasks:
        raise ValueError("no eligible tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_commit": source["source_commit"],
        "seeds": list(seeds),
        "horizon": 3,
        "source_selection": source["selection"],
        "assay_budget": 8,
        "probes_per_task": 10,
        "tasks": [
            {
                "id": task.id,
                "model_sha256": task.model.sha256,
                "initial": task.initial,
                "readouts": [task.model.nodes[i] for i in task.readouts],
                "target": {task.model.nodes[i]: value for i, value in task.target},
                "probes": [
                    [(task.model.nodes[i], value) for i, value in action] for action in task.probes
                ],
                "reachable_fraction": task.success_rate,
            }
            for task in tasks
        ],
    }
    (output_dir / TASKS).write_text(json.dumps(manifest, indent=2) + "\n")
    episodes_path = output_dir / EPISODES
    with episodes_path.open("w") as output:
        for task in tasks:
            for strategy in STRATEGIES:
                output.write(json.dumps(run_episode(task, strategy), sort_keys=True) + "\n")
    return write_analysis(episodes_path, figures_dir, output_dir / "summary.json", root)
