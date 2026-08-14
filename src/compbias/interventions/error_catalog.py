"""Validation and lookup helpers for executable error catalogs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from compbias.envs.cva_world.corruptions import apply_error, validate_error_spec


def validate_error_catalog(error_catalog: Iterable[object]) -> tuple[object, ...]:
    """Validate a complete catalog and return an immutable ordered snapshot."""

    errors = tuple(error_catalog)
    if not errors:
        raise ValueError("error_catalog must not be empty")
    identifiers: list[str] = []
    for error in errors:
        validate_error_spec(error)
        error_id = getattr(error, "error_id", None)
        if not isinstance(error_id, str) or not error_id:
            raise ValueError("every error specification must have a non-empty error_id")
        identifiers.append(error_id)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("error_catalog contains duplicate error_id values")
    if "truth" not in identifiers:
        raise ValueError("error_catalog must contain a truth intervention")
    return errors


def index_error_catalog(error_catalog: Iterable[object]) -> Mapping[str, object]:
    """Return a read-only ``error_id`` index after full catalog validation."""

    errors = validate_error_catalog(error_catalog)
    return MappingProxyType({error.error_id: error for error in errors})


def apply_catalog_error(
    scene: Mapping[str, object], error_catalog: Iterable[object], error_id: str
) -> Mapping[str, object]:
    """Apply one catalogued error without mutating the caller-owned scene."""

    if not isinstance(error_id, str) or not error_id:
        raise ValueError("error_id must be a non-empty string")
    index = index_error_catalog(error_catalog)
    try:
        error = index[error_id]
    except KeyError as exc:
        raise KeyError(f"unknown error_id: {error_id}") from exc
    return apply_error(scene, error)
