"""Atomic JSONL writing and strict mapping-only reading."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from .manifests import canonical_json


class JsonlDecodeError(ValueError):
    """A malformed or non-object JSONL row with path and line provenance."""

    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = path
        self.line_number = line_number
        super().__init__(f"{path}:{line_number}: {message}")


def _path(path: str | os.PathLike[str]) -> Path:
    result = Path(path)
    if not result.name:
        raise ValueError("JSONL path must name a file")
    return result


def read_jsonl(path: str | os.PathLike[str]) -> tuple[dict[str, object], ...]:
    """Read every line as a JSON object, never silently dropping bad rows."""

    source = _path(path)
    records: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise JsonlDecodeError(source, line_number, "blank JSONL row")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlDecodeError(source, line_number, exc.msg) from exc
            if not isinstance(value, dict):
                raise JsonlDecodeError(source, line_number, "JSONL row must be an object")
            records.append(value)
    return tuple(records)


def write_jsonl(path: str | os.PathLike[str], records: Iterable[Mapping[str, object]]) -> Path:
    """Atomically replace ``path`` with canonical mapping-valued JSONL rows."""

    destination = _path(path)
    encoded: list[str] = []
    for line_number, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"JSONL row {line_number} must be a mapping")
        encoded.append(canonical_json(record))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for row in encoded:
                stream.write(row)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def append_jsonl(path: str | os.PathLike[str], records: Iterable[Mapping[str, object]]) -> Path:
    """Atomically append a batch by replacing the complete visible file."""

    destination = _path(path)
    existing: tuple[Mapping[str, object], ...] = ()
    if destination.exists():
        existing = read_jsonl(destination)
    return write_jsonl(destination, (*existing, *tuple(records)))
