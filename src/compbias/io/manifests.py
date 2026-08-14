"""Canonical hashes and immutable dataset manifests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


def _canonicalize(value: object) -> object:
    to_mapping = getattr(value, "to_mapping", None)
    if callable(to_mapping):
        return _canonicalize(to_mapping())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonicalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Mapping):
        canonical: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool, Enum)):
                raise TypeError("manifest mapping keys must be scalar JSON keys")
            normalized_key = str(key.value if isinstance(key, Enum) else key)
            if normalized_key in canonical:
                raise ValueError("manifest mapping keys collide after canonicalization")
            canonical[normalized_key] = _canonicalize(item)
        return canonical
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=canonical_json)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest values must be finite")
        return value
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return _canonicalize(item_method())
    raise TypeError(f"unsupported manifest value type: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Serialize supported values with stable mapping order and no NaN tokens."""

    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def manifest_sha256(payload: object) -> str:
    """Return the lowercase SHA-256 digest of a canonical JSON payload."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_name: str
    schema_version: str
    sample_count: int
    sample_ids: tuple[str, ...]
    content_sha256: str
    config_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "sample_ids": list(self.sample_ids),
            "content_sha256": self.content_sha256,
            "config_sha256": self.config_sha256,
        }


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sample_payload(sample: object) -> tuple[str, object]:
    to_mapping = getattr(sample, "to_mapping", None)
    payload = to_mapping() if callable(to_mapping) else sample
    if not isinstance(payload, Mapping):
        raise TypeError("each dataset sample must serialize to a mapping")
    sample_id = _nonempty_string(payload.get("sample_id"), "sample_id")
    if sample_id in {".", ".."} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", sample_id) is None:
        raise ValueError("sample_id must be a safe basename")
    return sample_id, _canonicalize(payload)


def build_dataset_manifest(
    samples: Iterable[object],
    *,
    config: object,
    dataset_name: str,
    schema_version: str,
) -> DatasetManifest:
    """Build a deterministic manifest independent of sample iteration order."""

    serialized = tuple(_sample_payload(sample) for sample in samples)
    identifiers = tuple(sample_id for sample_id, _payload in serialized)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("dataset samples contain duplicate sample_id values")
    ordered = tuple(sorted(serialized, key=lambda item: item[0]))
    return DatasetManifest(
        dataset_name=_nonempty_string(dataset_name, "dataset_name"),
        schema_version=_nonempty_string(schema_version, "schema_version"),
        sample_count=len(ordered),
        sample_ids=tuple(sample_id for sample_id, _payload in ordered),
        content_sha256=manifest_sha256([payload for _sample_id, payload in ordered]),
        config_sha256=manifest_sha256(config),
    )
