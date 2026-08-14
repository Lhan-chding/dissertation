"""Closed chart-operation semantics shared by recoverability audits."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum


class Operation(str, Enum):
    """The three operations frozen by the chart pilot."""

    SUM = "sum"
    DIFFERENCE = "difference"
    MAX_MINUS_MIN = "max_minus_min"


def _operation(value: Operation | str) -> Operation:
    try:
        return Operation(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported recoverability operation: {value!r}") from error


def _values(value: Sequence[int], *, label: str) -> tuple[int, int, int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{label} must contain exactly four integers")
    if any(type(item) is not int for item in value):
        raise TypeError(f"{label} must contain exact integers")
    return value[0], value[1], value[2], value[3]


def apply_operation(values: Sequence[int], operation: Operation | str) -> int:
    """Apply the registered question semantics without modifying ``values``."""

    registered = _operation(operation)
    frozen = _values(values, label="values")
    if registered is Operation.SUM:
        return frozen[0] + frozen[1]
    if registered is Operation.DIFFERENCE:
        return frozen[0] - frozen[1]
    return max(frozen) - min(frozen)


def is_operator_null_error(
    truth: Sequence[int],
    perceived: Sequence[int],
    operation: Operation | str,
) -> bool:
    """Return true only for a real value error invisible to the operation."""

    true_values = _values(truth, label="truth")
    perceived_values = _values(perceived, label="perceived")
    return true_values != perceived_values and apply_operation(
        true_values, operation
    ) == apply_operation(perceived_values, operation)
