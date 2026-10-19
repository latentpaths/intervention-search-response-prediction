"""Verify public input and study-record bytes before scientific validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .paths import Layout, repository_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_evidence(root: Path | None = None) -> dict:
    """Check retained public files; this does not verify historical execution."""
    root = repository_root(root)
    manifest = json.loads(Layout(root).checksums.read_text())
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Retained evidence checksum mismatch: {relative}")
    return {"retained_files": len(manifest), "scope": "public file integrity"}
