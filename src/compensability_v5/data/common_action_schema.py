"""Closed schema for the common four-integer v5 action space."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_WORLD_SIZE = 4
_ACTION_FIELDS = frozenset({"world"})
_OPERATION_FIELDS = frozenset({"operator", "indices"})
_OPERATORS = frozenset({"sum", "difference"})


def _is_integer(value: object) -> bool:
    """Return whether *value* is an integer, excluding booleans."""

    return isinstance(value, int) and not isinstance(value, bool)


def _require_exact_fields(
    payload: Mapping[object, object], expected: frozenset[str], *, schema_name: str
) -> None:
    actual = frozenset(payload.keys())
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown fields: {', '.join(sorted(map(str, unknown)))}")
        raise ValueError(f"{schema_name} must use the closed schema ({'; '.join(details)})")


@dataclass(frozen=True, slots=True)
class WorldAction:
    """An immutable action in the shared four-integer world space."""

    world: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.world, tuple) or len(self.world) != _WORLD_SIZE:
            raise TypeError("world must be a tuple of exactly four integers")
        if any(not _is_integer(value) for value in self.world):
            raise TypeError("world values must be integers (bool is not an integer here)")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> WorldAction:
        """Parse a JSON-shaped mapping without retaining mutable input state."""

        if not isinstance(payload, Mapping):
            raise TypeError("world action must be a mapping")
        _require_exact_fields(payload, _ACTION_FIELDS, schema_name="world action")
        values = payload["world"]
        if not isinstance(values, list) or len(values) != _WORLD_SIZE:
            raise TypeError("world must be a list of exactly four integers")
        if any(not _is_integer(value) for value in values):
            raise TypeError("world values must be integers (bool is not an integer here)")
        return cls((values[0], values[1], values[2], values[3]))

    def to_mapping(self) -> dict[str, list[int]]:
        """Return a fresh JSON-compatible representation of this action."""

        return {"world": list(self.world)}


def _parse_answer_operation(operation: Mapping[str, object]) -> tuple[str, tuple[int, int]]:
    if not isinstance(operation, Mapping):
        raise TypeError("answer operation must be a mapping")
    _require_exact_fields(operation, _OPERATION_FIELDS, schema_name="answer operation")

    operator = operation["operator"]
    if not isinstance(operator, str) or operator not in _OPERATORS:
        raise ValueError("answer operation operator must be 'sum' or 'difference'")

    raw_indices = operation["indices"]
    if not isinstance(raw_indices, list) or len(raw_indices) != 2:
        raise TypeError("answer operation indices must be a list of exactly two integers")
    if any(not _is_integer(index) for index in raw_indices):
        raise TypeError("answer operation indices must be integers (bool is not allowed)")
    first, second = raw_indices
    if first == second or not all(0 <= index < _WORLD_SIZE for index in (first, second)):
        raise ValueError("answer operation indices must be distinct world indices in [0, 3]")
    return operator, (first, second)


def apply_answer_operation(action: WorldAction, operation: Mapping[str, Any]) -> int:
    """Project a common-space world action to its metadata-defined answer."""

    if not isinstance(action, WorldAction):
        raise TypeError("action must be a WorldAction")
    operator, (first, second) = _parse_answer_operation(operation)
    if operator == "sum":
        return action.world[first] + action.world[second]
    return action.world[first] - action.world[second]


__all__ = ["WorldAction", "apply_answer_operation"]
