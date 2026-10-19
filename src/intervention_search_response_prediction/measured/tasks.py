"""Measured task definitions and deterministic candidate/probe construction."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import logical_sha256, read_json
from ..paths import Layout, PreparedData, repository_root

VERSION = "measured-pilot-v1"


@dataclass
class MeasuredTask:
    id: str
    dataset: str
    context: str
    modality: str
    units: str
    readouts: list[str]
    records: dict
    candidates: tuple[str, ...]
    probes: tuple[str, ...]
    features: dict
    prior: dict
    disclosed: dict
    seed: int
    search: bool = True

    def view(self):
        ids = self.candidates + self.probes
        return {
            "task_id": self.id,
            "dataset": self.dataset,
            "context": self.context,
            "modality": self.modality,
            "units": self.units,
            "readouts": self.readouts,
            "budget": 8,
            "goal": "maximize measured enrichment"
            if self.search
            else "predict combined-perturbation responses",
            "candidates": list(self.candidates),
            "annotations": {
                a: {
                    "genes": self.records[a]["genes"],
                    "gene_groups": self.records[a].get("groups", []),
                }
                for a in ids
            },
            "disclosed_single_responses": self.disclosed,
            "definition": "Change from non-targeting control in log2(1+CPM), pooled counts, selected gene panel."
            if not self.search
            else "Positive log2 guide enrichment in cytokine-high versus cytokine-low sorting bins; not a cytokine concentration.",
        }

    def values(self, ids):
        return np.array([self.records[a]["value"] for a in ids], dtype=float)

    def priors(self, ids):
        return np.array([self.prior[a] for a in ids], dtype=float)

    def matrix(self, ids):
        return np.array([self.features[a] for a in ids], dtype=float)

    def fixed_actions(self):
        return tuple(random.Random(self.seed + 90421).sample(list(self.candidates), 8))


def tasks(data_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else Layout(repository_root()).data
    out = []
    schmidt = read_json(PreparedData(data_dir).responses)
    for screen in schmidt["screens"]:
        records = {r["id"]: r for r in screen["records"]}
        for seed in range(5):
            rng = random.Random(f"{VERSION}:{screen['id']}:{seed}")
            selected = rng.sample(sorted(records), 160)
            candidates = tuple(selected[:128])
            probes = tuple(selected[128:])
            families = sorted({g for a in selected for g in records[a]["groups"]})
            features = {}
            for a in selected:
                bits = np.array([float(g in records[a]["groups"]) for g in families])
                features[a] = [1.0] + (bits / max(1, float(np.linalg.norm(bits)))).tolist()
            out.append(
                MeasuredTask(
                    f"{screen['id']}:s{seed}",
                    "schmidt",
                    screen["context"],
                    screen["modality"],
                    screen["units"],
                    screen["readouts"],
                    records,
                    candidates,
                    probes,
                    features,
                    {a: [0.0] for a in selected},
                    {},
                    seed,
                )
            )
    meta = json.loads((PreparedData(data_dir).metadata).read_text())
    if logical_sha256(PreparedData(data_dir).counts) != meta["counts_sha256"]:
        raise ValueError("Norman counts checksum mismatch")
    z = np.load(PreparedData(data_dir).counts)
    counts = z["counts"]
    cells = z["cell_counts"]
    keys = [tuple(x) for x in meta["perturbations"]]
    keyidx = {k: i for i, k in enumerate(keys)}
    control = keyidx[()]
    norm = np.log2(1 + counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 1e6)
    delta = norm - norm[control]
    singles = [i for i, k in enumerate(keys) if len(k) == 1 and cells[i] >= 30]
    expressed = counts[[control] + singles].sum(axis=0) > 0
    variance = delta[singles].var(axis=0)
    rank = sorted(np.flatnonzero(expressed), key=lambda j: (-variance[j], meta["genes"][j][0]))[:8]
    readouts = [meta["genes"][j][1] + " (" + meta["genes"][j][0] + ")" for j in rank]
    disclosed = {keys[i][0]: delta[i, rank].tolist() for i in singles}
    eligible = [
        i
        for i, k in enumerate(keys)
        if len(k) == 2 and cells[i] >= 30 and all(g in disclosed for g in k)
    ]
    records = {}
    prior = {}
    features = {}
    scale = np.maximum(np.std([disclosed[g] for g in sorted(disclosed)], axis=0), 1e-6)
    for i in eligible:
        k = keys[i]
        a = "+".join(k)
        u, v = (np.array(disclosed[g]) for g in k)
        records[a] = {
            "id": a,
            "genes": list(k),
            "value": delta[i, rank].tolist(),
            "cell_count": int(cells[i]),
            "replicate_ids": sorted(meta["gemgroup_cell_counts"][i]),
            "groups": [],
        }
        prior[a] = (u + v).tolist()
        features[a] = [1.0] + ((u + v) / scale).tolist() + (np.abs(u - v) / scale).tolist()
    for seed in range(5):
        ids = sorted(records)
        random.Random(f"{VERSION}:norman:{seed}").shuffle(ids)
        n = math.ceil(len(ids) * 0.2)
        probes = tuple(ids[:n])
        candidates = tuple(ids[n:])
        if len(candidates) < 8:
            raise ValueError("insufficient Norman combinations")
        out.append(
            MeasuredTask(
                f"norman:s{seed}",
                "norman",
                "K562 cultured human cells",
                "CRISPRa",
                "delta log2(1+pooled CPM)",
                readouts,
                records,
                candidates,
                probes,
                features,
                prior,
                disclosed,
                seed,
                False,
            )
        )
    return out
