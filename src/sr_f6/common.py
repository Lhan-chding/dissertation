"""Small, content-addressed records for the separately registered SR-F6 run."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    """Idempotent exact content only; never replace changed or partial evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if path.is_symlink():
        raise PermissionError("Evidence path is a symlink")
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"Preserve differing existing record: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def safe_evidence(root, relative):
    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(p in ("..", ".") for p in relative.parts)
    ):
        raise PermissionError("Evidence must be a relative descendant")
    path = root / relative
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise PermissionError("Symlinked evidence is not permitted")
    if not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise PermissionError("Evidence is missing or escapes the run")
    return path
