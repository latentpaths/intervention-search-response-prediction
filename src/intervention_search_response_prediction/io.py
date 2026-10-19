"""Read study records and preserve checksums of losslessly compressed inputs."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


def logical_sha256(path):
    h = hashlib.sha256()
    p = Path(path)
    # Losslessly compressed local archives retain the original logical checksum.
    stream = p.open("rb") if p.exists() else gzip.open(str(p) + ".gz", "rb")
    with stream as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    with gzip.open(str(p) + ".gz", "rt") as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]
