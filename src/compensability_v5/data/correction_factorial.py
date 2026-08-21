"""Validation contracts for the frozen v5 correction-factorial manifest."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation

_PRIMARY_SPLITS = frozenset({"v5_support_train", "v5_support_dev"})
_STRESS_SPLIT = "v5_stress"
_SPLITS = _PRIMARY_SPLITS | {_STRESS_SPLIT}
_FIELDS = frozenset(
    {
        "scene_id",
        "split",
        "truth",
        "natural_observation",
        "error_count",
        "error_magnitudes",
        "error_domain",
        "constraint_matrix",
        "graph_signature",
        "answer_operation",
        "fiber_size",
        "orbit_parent",
        "transformation",
        "image_path",
        "prompt_path",
        "image_hash",
        "prompt_hash",
    }
)
_TRANSFORMATION_FIELDS = frozenset({"variable_permutation", "fact_permutation"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RESERVED_MARKERS = ("phase8", "phase-8", "confirm")


class FactorialManifestError(ValueError):
    """A row violates the frozen correction-factorial manifest contract."""


class SplitIsolationError(ValueError):
    """Development data overlap reserved confirmation data or another split."""


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _closed_fields(
    payload: Mapping[object, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = frozenset(payload.keys())
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            detail.append(f"unknown {', '.join(sorted(map(str, unknown)))}")
        raise FactorialManifestError(f"{label} has invalid closed schema: {'; '.join(detail)}")


def _nonempty_string(row: Mapping[str, object], field: str) -> str:
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise FactorialManifestError(f"{field} must be a non-empty string")
    return value


def _integer_list(value: object, *, field: str, length: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        suffix = f" with length {length}" if length is not None else ""
        raise FactorialManifestError(f"{field} must be an integer list{suffix}")
    if any(not _is_integer(item) for item in value):
        raise FactorialManifestError(f"{field} must contain integers and never bool")
    return tuple(value)


def _validate_permutation(value: object, *, size: int, field: str) -> None:
    permutation = _integer_list(value, field=field, length=size)
    if set(permutation) != set(range(size)):
        raise FactorialManifestError(f"{field} must be a permutation of range({size})")


def _validate_row(row: object, index: int) -> Mapping[str, object]:
    if not isinstance(row, Mapping):
        raise FactorialManifestError(f"row {index} must be a mapping")
    _closed_fields(row, _FIELDS, label=f"row {index}")

    _nonempty_string(row, "scene_id")
    split = _nonempty_string(row, "split")
    if split not in _SPLITS:
        raise FactorialManifestError(f"split must be one of {', '.join(sorted(_SPLITS))}")

    truth = _integer_list(row["truth"], field="truth", length=4)
    observation = _integer_list(row["natural_observation"], field="natural_observation", length=4)
    error_count = row["error_count"]
    if not _is_integer(error_count) or error_count <= 0:
        raise FactorialManifestError("error_count must be a positive integer")
    magnitudes = _integer_list(row["error_magnitudes"], field="error_magnitudes")
    observed_magnitudes = tuple(
        observed - expected
        for observed, expected in zip(observation, truth, strict=True)
        if observed != expected
    )
    if error_count != len(magnitudes) or magnitudes != observed_magnitudes:
        raise FactorialManifestError(
            "error_count and error_magnitudes must exactly describe the natural observation error"
        )

    error_domain = _nonempty_string(row, "error_domain")
    if error_domain not in {"in_domain", "out_of_domain"}:
        raise FactorialManifestError("error_domain must be in_domain or out_of_domain")
    if split in _PRIMARY_SPLITS:
        if error_count != 1 or error_domain != "in_domain":
            raise FactorialManifestError(
                "primary train/dev rows require exactly one in-domain natural error"
            )
        if any(value < 2 or value > 18 for value in observation):
            raise FactorialManifestError(
                "primary natural observations must remain in domain [2, 18]"
            )

    matrix = row["constraint_matrix"]
    if not isinstance(matrix, list) or not matrix:
        raise FactorialManifestError("constraint_matrix must be a non-empty list")
    for matrix_row in matrix:
        _integer_list(matrix_row, field="constraint_matrix", length=4)
    _nonempty_string(row, "graph_signature")

    operation = row["answer_operation"]
    if not isinstance(operation, Mapping):
        raise FactorialManifestError("answer_operation must be a mapping")
    try:
        apply_answer_operation(WorldAction(truth), operation)
    except (TypeError, ValueError) as error:
        raise FactorialManifestError(f"answer_operation is invalid: {error}") from error

    fiber_size = row["fiber_size"]
    if not _is_integer(fiber_size) or fiber_size <= 0:
        raise FactorialManifestError("fiber_size must be a positive integer")
    _nonempty_string(row, "orbit_parent")

    transformation = row["transformation"]
    if not isinstance(transformation, Mapping):
        raise FactorialManifestError("transformation must be a mapping")
    _closed_fields(transformation, _TRANSFORMATION_FIELDS, label="transformation")
    _validate_permutation(
        transformation["variable_permutation"], size=4, field="variable_permutation"
    )
    _validate_permutation(
        transformation["fact_permutation"], size=len(matrix), field="fact_permutation"
    )

    for field in ("image_path", "prompt_path"):
        path = _nonempty_string(row, field)
        if path.startswith("/") or ".." in path.split("/"):
            raise FactorialManifestError(f"{field} must be a safe relative artifact path")
    for field in ("image_hash", "prompt_hash"):
        digest = _nonempty_string(row, field)
        if _SHA256.fullmatch(digest) is None:
            raise FactorialManifestError(f"{field} must be a lowercase SHA-256 digest")
    return row


def validate_factorial_manifest(rows: Iterable[Mapping[str, object]]) -> None:
    """Validate completeness and internal consistency of every frozen row."""

    if isinstance(rows, (str, bytes)):
        raise FactorialManifestError("manifest rows must be an iterable of mappings")
    try:
        row_tuple = tuple(rows)
    except TypeError as error:
        raise FactorialManifestError("manifest rows must be iterable") from error
    if not row_tuple:
        raise FactorialManifestError("factorial manifest must contain at least one row")
    validated = tuple(_validate_row(row, index) for index, row in enumerate(row_tuple))
    scene_ids = tuple(row["scene_id"] for row in validated)
    if len(set(scene_ids)) != len(scene_ids):
        raise FactorialManifestError("scene_id values must be globally unique")


def _is_reserved(value: str, reserved_values: frozenset[str]) -> bool:
    normalized = value.casefold().replace("_", "-")
    return value.casefold() in reserved_values or any(
        marker in normalized for marker in _RESERVED_MARKERS
    )


def validate_factorial_isolation(
    rows: Iterable[Mapping[str, object]],
    *,
    reserved_scene_ids: Iterable[str],
    reserved_path_fragments: Iterable[str],
) -> None:
    """Reject any development/confirm overlap, including implicit Phase-8 provenance."""

    try:
        row_tuple = tuple(rows)
        reserved_ids_tuple = tuple(reserved_scene_ids)
        fragments = tuple(reserved_path_fragments)
    except TypeError as error:
        raise SplitIsolationError("isolation inputs must be iterable") from error
    if not row_tuple:
        raise SplitIsolationError("isolation audit requires at least one manifest row")
    try:
        validate_factorial_manifest(row_tuple)
    except FactorialManifestError as error:
        raise SplitIsolationError(f"manifest is invalid for isolation: {error}") from error
    if not reserved_ids_tuple or not fragments:
        raise SplitIsolationError(
            "reserved confirm/phase8 scene ids and path fragments are required for isolation"
        )
    if any(not isinstance(value, str) or not value for value in reserved_ids_tuple):
        raise SplitIsolationError("reserved_scene_ids must contain only non-empty strings")
    if any(not isinstance(value, str) or not value for value in fragments):
        raise SplitIsolationError("reserved_path_fragments must contain only non-empty strings")

    reserved_ids = frozenset(value.casefold() for value in reserved_ids_tuple)
    normalized_fragments = tuple(fragment.casefold() for fragment in fragments)
    seen: dict[str, dict[object, str]] = {
        field: {}
        for field in (
            "scene_id",
            "orbit_parent",
            "image_path",
            "prompt_path",
            "image_hash",
            "prompt_hash",
        )
    }
    for index, raw_row in enumerate(row_tuple):
        if not isinstance(raw_row, Mapping):
            raise SplitIsolationError(f"isolation row {index} must be a mapping")
        required = {
            "scene_id",
            "split",
            "orbit_parent",
            "image_path",
            "prompt_path",
            "image_hash",
            "prompt_hash",
        }
        missing = required - set(raw_row.keys())
        if missing:
            raise SplitIsolationError(
                f"isolation row {index} is missing {', '.join(sorted(missing))}"
            )
        values: dict[str, str] = {}
        for field in required:
            value = raw_row[field]
            if not isinstance(value, str) or not value:
                raise SplitIsolationError(f"{field} must be a non-empty string for isolation")
            values[field] = value

        if _is_reserved(values["split"], frozenset()):
            raise SplitIsolationError("confirm/phase8 split is reserved from v5 development")
        for field in ("scene_id", "orbit_parent"):
            if _is_reserved(values[field], reserved_ids):
                raise SplitIsolationError(f"{field} overlaps reserved confirm/phase8 data")
        for field in ("image_path", "prompt_path"):
            normalized_path = values[field].casefold()
            if _is_reserved(values[field], frozenset()) or any(
                fragment in normalized_path for fragment in normalized_fragments
            ):
                raise SplitIsolationError(f"{field} overlaps reserved confirm/phase8 paths")

        split = values["split"]
        for field, value in values.items():
            if field == "split":
                continue
            prior_split = seen[field].get(value)
            if prior_split is not None and prior_split != split:
                raise SplitIsolationError(f"{field} leaks across {prior_split} and {split}")
            seen[field][value] = split


__all__ = [
    "FactorialManifestError",
    "SplitIsolationError",
    "validate_factorial_isolation",
    "validate_factorial_manifest",
]
