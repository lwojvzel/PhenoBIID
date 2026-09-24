from __future__ import annotations
from pathlib import Path
import hashlib
ROOT = Path(__import__("os").environ["PHENOBIID_WORKSPACE"]).resolve()

CROPS = ("maize", "rice", "soybean", "wheat")

SPLITS = ("train", "validation", "test")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
