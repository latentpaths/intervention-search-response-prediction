"""Validate evidence, reproduce results offline, or run Boolean baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .paths import BASELINES, EPISODES, Layout, repository_root, writable_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validation = subparsers.add_parser("validate", help="Check evidence and both study audits")
    validation.add_argument("--root", type=Path)
    reproduction = subparsers.add_parser("reproduce", help="Regenerate analyses without inference")
    reproduction.add_argument("--root", type=Path)
    reproduction.add_argument("--output", type=Path)
    run_parser = subparsers.add_parser("run", help="Run deterministic Boolean procedures")
    run_parser.add_argument("--root", type=Path)
    run_parser.add_argument("--models", type=Path)
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    figures_parser = subparsers.add_parser("figures", help="Analyze recorded Boolean episodes")
    figures_parser.add_argument("--root", type=Path)
    figures_parser.add_argument("--input", type=Path)
    figures_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = repository_root(args.root)
    layout = Layout(root)
    if args.command == "validate":
        from .reproduction import validate

        print(json.dumps(validate(root), indent=2))
        return
    if args.command == "reproduce":
        from .reproduction import reproduce

        print(json.dumps(reproduce(args.output or root / "outputs/reproduction", root), indent=2))
        return
    from .simulation.analysis import write_analysis
    from .simulation.run import run

    default = f"outputs/{BASELINES}" if args.command == "run" else f"outputs/analysis/{BASELINES}"
    output = writable_output(args.output or root / default, root)
    if args.command == "run":
        summary = run(
            args.models or layout.models, output, output / "figures", tuple(args.seeds), root
        )
    else:
        summary = write_analysis(
            (args.input or layout.baselines) / EPISODES,
            output / "figures",
            output / "summary.json",
            root,
        )
    print(json.dumps({key: summary[key] for key in ("networks", "tasks", "episodes")}))


if __name__ == "__main__":
    main()
