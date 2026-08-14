"""Bounded, duplicate-free JSON input and no-clobber JSON publication."""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

_MAX_DEPTH = 64
_MAX_NODES = 100_000


def load_strict_json_mapping(
    path: Path,
    *,
    label: str,
    max_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    """Load one regular UTF-8 JSON object with bounded structural complexity."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"cannot inspect {label}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if metadata.st_size > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if len(raw) > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} contains a non-finite number")
        return number

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-standard number {value}")

    try:
        loaded = json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError, RecursionError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(loaded, dict) or any(not isinstance(key, str) for key in loaded):
        raise ValueError(f"{label} must contain a string-keyed JSON object")
    pending: list[tuple[object, int]] = [(loaded, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if depth > _MAX_DEPTH or nodes > _MAX_NODES:
            raise ValueError(f"{label} exceeds the permitted depth or complexity")
        if isinstance(current, dict):
            pending.extend((value, depth + 1) for value in current.values())
        elif isinstance(current, list):
            pending.extend((value, depth + 1) for value in current)
    return loaded


def write_new_json(path: Path, value: object) -> None:
    """Atomically create one JSON file without replacing an existing target."""

    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
