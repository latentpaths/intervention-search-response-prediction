"""Run measured-response reference procedures or resume paired LLM experiments."""

import argparse
import concurrent.futures
import hashlib
import json
import platform
import threading
import time
from pathlib import Path

from ..io import write_json
from ..paths import BASELINE_EPISODES, MEASURED, PAIRED_BLOCKS, repository_root, writable_output
from .inference import Calls, block_bound, run_llm
from .procedures import run_classical
from .protocol import freeze
from .tasks import tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["smoke", "freeze", "classical", "llm"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    if a.workers < 1:
        p.error("--workers must be positive")
    output_dir = writable_output(
        a.output if a.output is not None else repository_root() / f"outputs/{MEASURED}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if a.command == "freeze":
        ts = freeze(output_dir, a.data)
        print("Frozen", len(ts), "tasks")
        return
    if a.command == "smoke":
        ts = tasks(a.data)
        calls = Calls(output_dir)
        # Dedicated seed outside the scored seed range, with a new disjoint shuffled candidate/probe split.
        import random

        for source in (ts[0], ts[-1]):
            import copy

            t = copy.deepcopy(source)
            ids = list(t.candidates + t.probes)
            random.Random(71931).shuffle(ids)
            n = len(t.candidates)
            t.candidates = tuple(ids[:n])
            t.probes = tuple(ids[n:])
            t.seed = 71931
            t.id += ":smoke-v2"
            block = t.id
            calls.reserve_block(block, "smoke", block_bound(t))
            try:
                row = run_llm(t, "summary", "fixed", calls, "smoke", block)
                write_json(output_dir / ("smoke_" + t.dataset + ".json"), row)
                print(
                    "smoke",
                    t.dataset,
                    "failures",
                    len(row["failures"]),
                    "cost",
                    row["charged_usd"],
                    flush=True,
                )
            finally:
                calls.release_block(block)
        return
    ts = freeze(output_dir, a.data)
    if a.command == "classical":
        rows = [
            run_classical(t, agent)
            for t in ts
            for agent in (
                ["random_mean", "ridge_random", "ridge_active"]
                + (["additive"] if t.dataset == "norman" else [])
            )
        ]
        write_json(output_dir / BASELINE_EPISODES, rows)
        print("classical", len(rows), "episodes")
        return
    calls = Calls(output_dir)
    output = output_dir / PAIRED_BLOCKS
    lock = threading.Lock()
    completed = (
        {json.loads(x)["block"] for x in output.read_text().splitlines()}
        if output.exists()
        else set()
    )
    # Primary fixed comparisons first; then adaptive and shuffled diagnostics.
    # Within mode, round-robin across screens ensures coverage before later seeds.
    queue = [
        (mode, t)
        for mode in ("fixed", "adaptive", "shuffled")
        for t in sorted(ts, key=lambda t: (t.seed, t.id))
    ]
    errors = []

    def worker(mode, t):
        block = f"{mode}:{t.id}"
        if block in completed:
            return
        calls.reserve_block(block, "study", block_bound(t))
        try:
            conditions = ["summary", "hypothesis"]
            if int(hashlib.sha256(block.encode()).hexdigest(), 16) % 2:
                conditions.reverse()
            rows = [run_llm(t, c, mode, calls, "study", block) for c in conditions]
            with lock:
                with output.open("a") as f:
                    f.write(json.dumps({"block": block, "episodes": rows}, allow_nan=False) + "\n")
                    f.flush()
                print("complete", block, "spent", round(calls.spent["study"], 4), flush=True)
        finally:
            calls.release_block(block)

    # Launch bounded waves; never discard failed or partial blocks from the ledger.
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for start in range(0, len(queue), a.workers):
            wave = queue[start : start + a.workers]
            futures = [pool.submit(worker, *item) for item in wave]
            for item, f in zip(wave, futures):
                try:
                    f.result()
                except Exception as ex:
                    errors.append(
                        {
                            "block": item[0] + ":" + item[1].id,
                            "error": str(ex),
                            "type": type(ex).__name__,
                        }
                    )
                    print("STOP", errors[-1], flush=True)
            if errors:
                break
    write_json(
        output_dir / "run_status.json",
        {
            "errors": errors,
            "spent": calls.spent,
            "expected_blocks": len(queue),
            "completed_blocks": len(output.read_text().splitlines()) if output.exists() else 0,
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "workers": a.workers,
            },
            "ended_at": time.time(),
        },
    )


if __name__ == "__main__":
    main()
