"""Independent re-scoring and information-flow checks for the measured pilot."""

import json
from pathlib import Path

import numpy as np

from ..io import logical_sha256, read_json
from ..paths import (
    BASELINE_EPISODES,
    FROZEN_AT,
    INFERENCE_LEDGER,
    PAIRED_BLOCKS,
    Layout,
    PreparedData,
    repository_root,
)
from .inference import compact
from .ledger import reconcile
from .metrics import continuous_scores
from .protocol import (
    freeze,
    verify_historical_protocol,
)


def audit(input_dir=None, data_dir=None):
    input_dir = Path(input_dir) if input_dir is not None else Layout(repository_root()).measured
    data_dir = Path(data_dir) if data_dir is not None else Layout(repository_root()).data
    protocol = read_json(input_dir / "protocol.json")
    if "code_sha256" in protocol:
        task_list = verify_historical_protocol(data_dir, input_dir)
    else:
        task_list = freeze(input_dir, data_dir)
    ts = {t.id: t for t in task_list}
    checks = 0
    unavailable = []
    prepared = PreparedData(data_dir)
    for metadata in (prepared.responses, prepared.metadata):
        for source in read_json(metadata)["sources"]:
            if "file" in source:
                path = prepared.raw_source(metadata, source["file"])
                if path.exists() or path.with_name(path.name + ".gz").exists():
                    assert logical_sha256(path) == source["sha256"]
                    checks += 1
                else:
                    unavailable.append(str(path))
    calls_path = input_dir / INFERENCE_LEDGER
    events = (
        [json.loads(x) for x in calls_path.read_text().splitlines()] if calls_path.exists() else []
    )
    calls = {r["call_id"]: r for r in events if r["event"] == "response"}
    assert len(calls) == sum(r["event"] == "response" for r in events), (
        "duplicate completed call IDs"
    )
    freeze_stamp = json.loads((input_dir / FROZEN_AT).read_text())
    assert freeze_stamp["protocol_sha256"] == logical_sha256(input_dir / "protocol.json")
    assert all(r["timestamp"] >= freeze_stamp["unix_time"] for r in events if r["phase"] == "study")
    checks += 3
    blocks = (
        [json.loads(x) for x in (input_dir / PAIRED_BLOCKS).read_text().splitlines()]
        if (input_dir / PAIRED_BLOCKS).exists()
        else []
    )
    assert len({b["block"] for b in blocks}) == len(blocks)
    rows = json.loads((input_dir / BASELINE_EPISODES).read_text())
    for block in blocks:
        assert {r["agent"] for r in block["episodes"]} == {"summary", "hypothesis"}
        assert len(block["episodes"]) == 2
        a, b = block["episodes"]
        assert a["task_id"] == b["task_id"] and a["mode"] == b["mode"]
        if a["mode"] != "adaptive":
            assert a["visible_history"] == b["visible_history"]
        rows.extend(block["episodes"])
        checks += 1
    for r in rows:
        t = ts[r["task_id"]]
        history = r["history"]
        assert r["probes"] == list(t.probes)
        np.testing.assert_allclose(r["probe_truth"], t.values(t.probes), rtol=0, atol=0)
        assert len(history) == 8 and len({h["action"] for h in history}) == 8
        for h in history:
            assert h["action"] in t.candidates
            np.testing.assert_allclose(h["value"], t.records[h["action"]]["value"], rtol=0, atol=0)
        assert r["final"] in t.candidates
        for z in ("before", "after"):
            score = continuous_scores(r["probe_" + z], t.values(t.probes))
            for k, v in score.items():
                assert abs(v - r[z][k]) < 1e-10
        assert abs(r["mse_gain"] - (r["before"]["mse"] - r["after"]["mse"])) < 1e-10
        assert (
            abs(
                r["discrimination_gain"]
                - (r["after"]["discrimination"] - r["before"]["discrimination"])
            )
            < 1e-10
        )
        if not t.search:
            additive = t.priors(t.probes)
            assert (
                abs(r["additive_mse"] - float(np.mean((additive - t.values(t.probes)) ** 2)))
                < 1e-10
            )
            residual = continuous_scores(
                np.array(r["probe_after"]) - additive, t.values(t.probes) - additive
            )
            for key, value in residual.items():
                assert abs(value - r["interaction_residual"][key]) < 1e-10
        if t.search:
            assert abs(r["final_effect"] - t.records[r["final"]]["value"][0]) < 1e-10
            assert (
                abs(r["regret"] - (max(t.values(t.candidates)[:, 0]) - r["final_effect"])) < 1e-10
            )
            np.testing.assert_allclose(
                r["best_observed_curve"], np.maximum.accumulate([h["value"][0] for h in history])
            )
        if r["mode"] == "classical":
            checks += 1
            continue
        if r["mode"] != "adaptive":
            assert [h["action"] for h in history] == list(t.fixed_actions())
        for call_id in r["call_ids"]:
            c = calls[call_id]
            msg = json.loads(c["request"]["message"].split("\nYour preceding")[0])
            assert msg["view"] == compact(t.view())
            step = call_id.rsplit(":", 2)[1]
            n = int(step[5:]) if step.startswith("assay") else (0 if step == "before" else 8)
            assert msg["history"] == compact(r["visible_history"][:n])
            assert "probe_truth" not in msg and "values" not in msg["view"]
            if msg["phase"] == "probe":
                assert msg["actions"] == list(t.probes)
            checks += 1
        assert len(r["call_ids"]) == (11 if t.search else 10) + len(
            [x for x in r["failures"] if x["attempt"] == 0]
        )
        if r["mode"] == "shuffled":
            assert sorted(tuple(h["value"]) for h in history) == sorted(
                tuple(h["value"]) for h in r["visible_history"]
            )
        else:
            assert [h["value"] for h in history] == [h["value"] for h in r["visible_history"]]
    spent, _ = reconcile(events)
    assert spent["smoke"] <= 5 and spent["study"] <= 45 and sum(spent.values()) <= 50
    summary = {
        "checks": checks,
        "episodes": len(rows),
        "completed_blocks": len(blocks),
        "spent_usd": spent,
        "passed": True,
        "raw_source_files_unavailable": unavailable,
        "scope": "trusted in-process agents; no OS sandbox; no independent biological replicate inference",
    }
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.data), indent=2))


if __name__ == "__main__":
    main()
