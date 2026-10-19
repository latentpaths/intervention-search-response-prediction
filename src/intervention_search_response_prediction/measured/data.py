"""Download source measurements and prepare compact perturbation tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import math
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..io import logical_sha256, write_json
from ..paths import PreparedData, repository_root, writable_output

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/"


SCHMIDT = "https://orcs.thebiogrid.org/uploads/processed/6807b7659f6c0/NIHMS1814143-supplement-Supplementary_table_1.xlsx"


HGNC = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"


def fetch(name, url, raw_dir):
    root = repository_root()
    raw_dir.mkdir(parents=True, exist_ok=True)
    p = raw_dir / name
    if not p.exists() and Path(str(p) + ".gz").exists():
        with gzip.open(str(p) + ".gz", "rb") as source, p.open("wb") as target:
            while b := source.read(1024 * 1024):
                target.write(b)
    if not p.exists():
        with (
            urllib.request.urlopen(url, timeout=90) as r,
            p.with_suffix(p.suffix + ".part").open("wb") as f,
        ):
            while b := r.read(1024 * 1024):
                f.write(b)
        p.with_suffix(p.suffix + ".part").replace(p)
    return p, {
        "url": url,
        "file": str(p.relative_to(root)) if p.is_relative_to(root) else str(p),
        "sha256": logical_sha256(p),
        "bytes": p.stat().st_size,
    }


def prepare_schmidt(data_dir):
    prepared = PreparedData(Path(data_dir))
    data_dir = prepared.responses.parent
    from openpyxl import load_workbook

    p, source = fetch("schmidt_table1.xlsx", SCHMIDT, data_dir / "raw")
    hp, hs = fetch("hgnc_complete_set.txt", HGNC, data_dir / "raw")
    groups = {}
    with hp.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            groups[r["symbol"]] = r.get("gene_group", "").split("|") if r.get("gene_group") else []
    w = load_workbook(p, read_only=True, data_only=True)
    screens = []
    for s in w:
        if not any("." + x + "." in s.title for x in ("IL2", "IFNG")):
            continue
        rows = iter(s.values)
        header = next(rows)
        records = []
        excluded = Counter()
        for row in rows:
            d = dict(zip(header, row))
            gene = d["id"]
            value = d["pos|lfc"]
            if not isinstance(gene, str) or gene.startswith(
                ("Non-Target", "non-target", "NO-TARGET")
            ):
                excluded["control_or_missing_id"] += 1
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or d["num"] < 2:
                excluded["missing_effect_or_fewer_than_two_guides"] += 1
                continue
            if abs(value - d["neg|lfc"]) > 1e-8:
                raise ValueError("directional LFC differs")
            records.append(
                {
                    "id": gene,
                    "genes": [gene],
                    "value": [float(value)],
                    "groups": groups.get(gene, []),
                    "replicate_ids": [],
                    "guide_count": int(d["num"]),
                }
            )
        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate perturbation")
        parts = s.title.split(".")
        screens.append(
            {
                "id": "schmidt_" + ".".join(parts[:-1]),
                "context": f"stimulated primary human {parts[2]}+ T cells",
                "modality": "CRISPRa" if parts[1].lower() == "cra" else "CRISPRi",
                "readouts": [parts[3]],
                "units": "log2 guide enrichment, cytokine-high / cytokine-low",
                "source_sheet": s.title,
                "replicate_note": "Published pooled gene summary; donor-level effects unavailable in this table. Guide count is not biological n.",
                "records": sorted(records, key=lambda r: r["id"]),
                "exclusions": dict(excluded),
            }
        )
    write_json(
        prepared.responses,
        {
            "sources": [source, hs],
            "study_doi": "10.1126/science.abj4008",
            "screens": screens,
            "excluded_sheets": ["Supp.Cra.CD4.TNF.gene_summary"],
            "selection": "All finite gene effects with >=2 guides; no significance or effect-size filtering.",
        },
    )
    print("Prepared", [(s["id"], len(s["records"])) for s in screens], flush=True)


class HashReader:
    def __init__(self, raw):
        self.raw = raw
        self.hash = hashlib.sha256()
        self.size = 0

    def read(self, n=-1):
        b = self.raw.read(n)
        self.hash.update(b)
        self.size += len(b)
        return b


def prepare_norman(data_dir):
    prepared = PreparedData(Path(data_dir))
    data_dir = prepared.metadata.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    paths = {}
    for suffix in ("barcodes.tsv", "genes.tsv", "cell_identities.csv"):
        name = "GSE133344_filtered_" + suffix + ".gz"
        paths[suffix], meta = fetch(name, GEO + name, data_dir / "raw")
        sources.append(meta)
    with gzip.open(paths["barcodes.tsv"], "rt") as f:
        barcodes = [x.strip() for x in f]
    with gzip.open(paths["genes.tsv"], "rt") as f:
        genes = [x.strip().split("\t") for x in f]
    bybarcode = {}
    excludes = Counter()
    with gzip.open(paths["cell_identities.csv"], "rt") as f:
        for row in csv.DictReader(f):
            if row["good_coverage"] != "True" or row["number_of_cells"] != "1":
                excludes["poor_coverage_or_multiplet"] += 1
                continue
            halves = row["guide_identity"].split("__")
            if len(halves) != 2 or halves[0] != halves[1]:
                excludes["discordant_guide_identity"] += 1
                continue
            g = tuple(sorted(x for x in halves[0].split("_") if not x.startswith("NegCtrl")))
            if len(g) > 2 or len(set(g)) != len(g):
                excludes["unsupported_guide_pair"] += 1
                continue
            if row["cell_barcode"] in bybarcode:
                raise ValueError("duplicate barcode")
            bybarcode[row["cell_barcode"]] = (g, row["gemgroup"])
    keys = sorted(set(x[0] for x in bybarcode.values()))
    index = {g: i for i, g in enumerate(keys)}
    cellmap = np.array([index[bybarcode[b][0]] if b in bybarcode else -1 for b in barcodes])
    counts = np.bincount(cellmap[cellmap >= 0], minlength=len(keys))
    batches = defaultdict(Counter)
    for b in barcodes:
        if b in bybarcode:
            batches[bybarcode[b][0]][bybarcode[b][1]] += 1
    sums = np.zeros((len(keys), len(genes)), dtype=np.float64)
    url = GEO + "GSE133344_filtered_matrix.mtx.gz"
    start = time.perf_counter()
    entries = 0
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = HashReader(response)
        with gzip.GzipFile(fileobj=raw) as z:
            line = z.readline()
            if not line.startswith(b"%%MatrixMarket matrix coordinate integer general"):
                raise ValueError(line)
            line = z.readline()
            while line.startswith(b"%"):
                line = z.readline()
            ng, nc, nnz = map(int, line.split())
            assert (ng, nc) == (len(genes), len(barcodes))
            while True:
                block = z.read(8 * 1024 * 1024)
                if not block:
                    break
                block += z.readline()
                a = np.fromstring(block.decode("ascii"), dtype=np.int64, sep=" ").reshape(-1, 3)
                rows = cellmap[a[:, 1] - 1]
                valid = rows >= 0
                np.add.at(sums, (rows[valid], a[valid, 0] - 1), a[valid, 2])
                entries += len(a)
                if entries // 20000000 != (entries - len(a)) // 20000000:
                    print(
                        f"Norman: {entries}/{nnz} entries, {time.perf_counter() - start:.0f}s",
                        flush=True,
                    )
            if entries != nnz:
                raise ValueError("truncated matrix")
        sources.append(
            {
                "url": url,
                "sha256": raw.hash.hexdigest(),
                "bytes": raw.size,
                "storage": "streamed; compact pooled counts retained",
            }
        )
    np.savez_compressed(prepared.counts, counts=sums, cell_counts=counts)
    write_json(
        prepared.metadata,
        {
            "sources": sources,
            "study_doi": "10.1126/science.aax4438",
            "genes": genes,
            "perturbations": [list(k) for k in keys],
            "gemgroup_cell_counts": [dict(batches[k]) for k in keys],
            "exclusions": dict(excludes),
            "replicate_note": "GEM groups are technical processing groups, not independent biological donors.",
            "matrix_entries": entries,
            "preprocessing_seconds": time.perf_counter() - start,
            "counts_sha256": logical_sha256(prepared.counts),
        },
    )
    print("Prepared Norman", sums.shape, int(counts.sum()), "cells", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["schmidt", "norman", "all"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = writable_output(args.output or repository_root() / "outputs/prepared-data")
    if args.dataset in ("schmidt", "all"):
        prepare_schmidt(output)
    if args.dataset in ("norman", "all"):
        prepare_norman(output)


if __name__ == "__main__":
    main()
