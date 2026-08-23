"""Fail-closed JSON and hashing helpers for Study C3 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Study C3 source is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Study C3 JSON is missing or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Study C3 JSON root must be an object: {path}")
    return value


def read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Study C3 JSONL is missing or unsafe: {path}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Study C3 JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return tuple(rows)


def write_json_new(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"Study C3 output overwrite is forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_jsonl_new(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"Study C3 output overwrite is forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False))
            stream.write("\n")


def append_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    if path.is_symlink():
        raise ValueError(f"Study C3 append target must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


__all__ = [
    "append_jsonl",
    "canonical_sha256",
    "read_json",
    "read_jsonl",
    "sha256_file",
    "write_json_new",
    "write_jsonl_new",
]
