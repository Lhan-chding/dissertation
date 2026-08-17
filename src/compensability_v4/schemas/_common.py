"""Strict validation and immutable JSON helpers for v4 records."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from types import MappingProxyType

_HASH = re.compile(r"[0-9a-f]{64}\Z")


def require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def require_closed_keys(
    mapping: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] | None = None,
    name: str,
) -> None:
    allowed_optional = optional or set()
    missing = required - set(mapping)
    unknown = set(mapping) - required - allowed_optional
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {name} fields: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {name} fields: {', '.join(sorted(unknown))}")
        raise ValueError("; ".join(details))


def require_identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def require_integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def require_relative_path(value: object, name: str) -> str:
    result = require_identifier(value, name)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or "\\" in result:
        raise ValueError(f"{name} must be a safe repository-relative POSIX path")
    return result


def freeze_json(value: object, name: str = "value", *, depth: int = 0) -> object:
    if depth > 64:
        raise ValueError(f"{name} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} mapping keys must be strings")
        return MappingProxyType(
            {
                key: freeze_json(item, f"{name}.{key}", depth=depth + 1)
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze_json(item, name, depth=depth + 1) for item in value)
    raise TypeError(f"{name} must be JSON-compatible")


def thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return copy.deepcopy(value)
