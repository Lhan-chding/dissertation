"""Fail-closed JSON/JSONL and provenance helpers for Study C2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing JSONL: {path}")
    rows = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL must contain mappings: {path}")
    return rows


def write_json_new(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            serialized = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write(serialized + "\n")


__all__ = ["read_json", "read_jsonl", "sha256_file", "write_json_new", "write_jsonl_new"]
