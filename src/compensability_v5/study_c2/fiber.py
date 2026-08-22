"""Exact reward fibers over the registered four-integer action domain."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from itertools import product

from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation

VALUE_DOMAIN = tuple(range(2, 19))


def _operation_key(operation: Mapping[str, object]) -> tuple[str, tuple[int, ...]]:
    operator = operation.get("operator")
    indices = operation.get("indices")
    if not isinstance(operator, str) or not isinstance(indices, list):
        raise ValueError("operation must contain operator and index list")
    return operator, tuple(int(index) for index in indices)


@cache
def _full_size(operator: str, indices: tuple[int, ...], answer: int) -> int:
    operation = {"operator": operator, "indices": list(indices)}
    return sum(
        apply_answer_operation(WorldAction(world), operation) == answer
        for world in product(VALUE_DOMAIN, repeat=4)
    )


def full_reward_fiber_size(operation: Mapping[str, object], answer: int) -> int:
    """Enumerate the exact full-domain answer fiber (17^4 worlds)."""

    operator, indices = _operation_key(operation)
    return _full_size(operator, indices, int(answer))


def one_edit_fiber_size(truth: tuple[int, int, int, int], operation: Mapping[str, object]) -> int:
    """Count non-truth one-coordinate edits with the same answer."""

    target = apply_answer_operation(WorldAction(truth), operation)
    count = 0
    for index in range(4):
        for value in VALUE_DOMAIN:
            if value == truth[index]:
                continue
            candidate = list(truth)
            candidate[index] = value
            count += apply_answer_operation(WorldAction(tuple(candidate)), operation) == target
    return count


__all__ = ["VALUE_DOMAIN", "full_reward_fiber_size", "one_edit_fiber_size"]
