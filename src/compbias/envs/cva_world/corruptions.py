"""Validated, reversible perceptual corruptions for CVA-World scenes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from types import MappingProxyType

from .schema import ErrorSpec

_SUPPORTED_FAMILIES = {
    "truth",
    "numeric_offset",
    "omission",
    "duplication",
    "local_offset",
    "local_to_global_inconsistency",
    "relation_flip",
}


def _numeric_parameter(parameters: Mapping[str, object], key: str) -> Real:
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{key} must be a finite number")
    return value


def _field(parameters: Mapping[str, object]) -> str:
    value = parameters.get("field")
    if not isinstance(value, str) or not value:
        raise ValueError("field must be a non-empty string")
    return value


def validate_error_spec(error: ErrorSpec) -> None:
    """Validate parameters and severity against the corruption's semantics."""

    if not isinstance(error, ErrorSpec):
        raise TypeError("error must be an ErrorSpec")
    if error.family not in _SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported error family: {error.family}")
    parameters = error.parameters

    expected_parameter_keys = {
        "truth": set(),
        "numeric_offset": {"field", "delta"},
        "local_offset": {"field", "index", "delta"},
        "local_to_global_inconsistency": {"field", "indices"},
        "omission": {"field", "amount"},
        "duplication": {"field", "amount"},
        "relation_flip": {"field", "pairs"},
    }
    if set(parameters) != expected_parameter_keys[error.family]:
        raise ValueError(f"{error.family} parameters fields do not match the closed schema")

    if error.family == "truth":
        if error.error_id != "truth" or error.severity != 0 or parameters:
            raise ValueError("truth must be the zero-severity identity corruption")
        return

    _field(parameters)
    if error.family in {"numeric_offset", "local_offset"}:
        delta = _numeric_parameter(parameters, "delta")
        if delta == 0 or not math.isclose(error.severity, abs(float(delta))):
            raise ValueError("severity must equal the absolute nonzero delta")
        if error.family == "local_offset":
            index = parameters.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("index must be a non-negative integer")
        return

    if error.family == "local_to_global_inconsistency":
        indices = parameters.get("indices")
        if (
            not isinstance(indices, Sequence)
            or isinstance(indices, (str, bytes))
            or len(indices) != 2
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
            or indices[0] == indices[1]
        ):
            raise ValueError(
                "local_to_global_inconsistency indices must be two distinct non-negative integers"
            )
        if error.severity != 1:
            raise ValueError("local_to_global_inconsistency severity must be 1")
        return

    if error.family in {"omission", "duplication"}:
        amount = _numeric_parameter(parameters, "amount")
        if amount <= 0 or not math.isclose(error.severity, float(amount)):
            raise ValueError("severity must equal the positive amount")
        return

    pairs = parameters.get("pairs")
    if not isinstance(pairs, Mapping) or not pairs:
        raise ValueError("relation_flip pairs must be a non-empty mapping")
    if error.severity != 1:
        raise ValueError("relation_flip severity must be 1")
    for source, target in pairs.items():
        if (
            not isinstance(source, str)
            or not isinstance(target, str)
            or pairs.get(target) != source
        ):
            raise ValueError("relation_flip pairs must define an involution")


def _clone(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value


def _numeric_scene_value(scene: Mapping[str, object], field: str) -> Real:
    if field not in scene:
        raise KeyError(f"scene has no field {field!r}")
    value = scene[field]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"scene field {field!r} must be numeric")
    return value


def _apply(scene: Mapping[str, object], error: ErrorSpec, *, reverse: bool) -> Mapping[str, object]:
    if not isinstance(scene, Mapping):
        raise TypeError("scene must be a mapping")
    validate_error_spec(error)
    result = dict(_clone(scene))
    if error.family == "truth":
        return MappingProxyType(result)

    field = _field(error.parameters)
    direction = -1 if reverse else 1
    if error.family == "numeric_offset":
        result[field] = _numeric_scene_value(scene, field) + direction * _numeric_parameter(
            error.parameters, "delta"
        )
    elif error.family in {"omission", "duplication"}:
        sign = -1 if error.family == "omission" else 1
        value = _numeric_scene_value(scene, field)
        result[field] = value + direction * sign * _numeric_parameter(error.parameters, "amount")
        if result[field] < 0:  # type: ignore[operator]
            raise ValueError(f"corruption would make {field!r} negative")
    elif error.family == "local_offset":
        values = scene.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"scene field {field!r} must be a sequence")
        index = error.parameters["index"]
        if not isinstance(index, int) or index >= len(values):
            raise ValueError("local_offset index out of range")
        current = values[index]
        if isinstance(current, bool) or not isinstance(current, Real):
            raise TypeError("local_offset target must be numeric")
        changed = list(_clone(values))
        changed[index] = current + direction * _numeric_parameter(error.parameters, "delta")
        result[field] = tuple(changed) if isinstance(values, tuple) else changed
    elif error.family == "local_to_global_inconsistency":
        values = scene.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"scene field {field!r} must be a sequence")
        indices = error.parameters["indices"]
        assert isinstance(indices, Sequence)
        first, second = indices
        assert isinstance(first, int) and isinstance(second, int)
        if max(first, second) >= len(values):
            raise ValueError("local_to_global_inconsistency index out of range")
        changed = list(_clone(values))
        changed[first], changed[second] = changed[second], changed[first]
        result[field] = tuple(changed) if isinstance(values, tuple) else changed
    else:
        pairs = error.parameters["pairs"]
        assert isinstance(pairs, Mapping)
        relation = scene.get(field)
        if not isinstance(relation, str) or relation not in pairs:
            raise ValueError(f"scene relation {relation!r} is absent from flip pairs")
        result[field] = pairs[relation]
    return MappingProxyType(result)


def apply_error(scene: Mapping[str, object], error: ErrorSpec) -> Mapping[str, object]:
    """Return an immutable corrupted copy of ``scene``."""

    return _apply(scene, error, reverse=False)


def reverse_error(scene: Mapping[str, object], error: ErrorSpec) -> Mapping[str, object]:
    """Invert a previously applied registered corruption."""

    return _apply(scene, error, reverse=True)
