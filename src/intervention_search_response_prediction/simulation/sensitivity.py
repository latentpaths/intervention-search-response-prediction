"""Check whether direct readout clamps explain intervention discrimination."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..io import read_jsonl
from ..paths import EPISODES, NOTES, Layout, repository_root
from ..statistics import mean, network_interval
from .inference import node_labels, selected_tasks
from .metrics import unclamped_discrimination


def _summarize(rows: list[dict], group_field: str) -> dict:
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        score = unclamped_discrimination(
            row["probe_predictions_after"], row["probe_truths"], row["probes"], row["readouts"]
        )
        grouped[row[group_field]][row["model"]].append(score)
    return {
        condition: {
            "discrimination_unclamped": network_interval(
                {model: mean([score[0] for score in scores]) for model, scores in models.items()}
            ),
            "direct_clamp_fraction": network_interval(
                {model: mean([score[1] for score in scores]) for model, scores in models.items()}
            ),
        }
        for condition, models in grouped.items()
    }


def analyze(
    root: Path | None = None,
    include_posthoc: bool = False,
    input_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict:
    root = repository_root(root)
    selected, _, _ = selected_tasks(models_dir or Layout(root).models)
    tasks = {task.id: task for task in selected}
    classical = read_jsonl(Layout(root).baselines / EPISODES)
    llm = [
        row
        for row in read_jsonl((input_dir or Layout(root).notes) / EPISODES)
        if not row.get("smoke") and (include_posthoc or row["condition"] != "summary_note")
    ]
    for row in llm:
        task = tasks[row["task_id"]]
        labels = node_labels(task, row["label_condition"] == "anonymous")
        row["readouts"] = [labels[index] for index in task.readouts]
        row["group"] = f"{row['label_condition']}/{row['condition']}"
    result = {"classical": _summarize(classical, "strategy"), "language": _summarize(llm, "group")}
    return result


def main() -> None:
    import argparse

    from ..io import write_json
    from ..paths import writable_output

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-posthoc", action="store_true")
    args = parser.parse_args()
    output = writable_output(
        args.output or repository_root(args.root) / f"outputs/analysis/{NOTES}"
    )
    write_json(
        output / "readout-sensitivity.json",
        analyze(args.root, args.include_posthoc, args.input, args.models),
    )
    print(output)


if __name__ == "__main__":
    main()
