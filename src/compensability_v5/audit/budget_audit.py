"""Fail-closed compute-budget matching for v5 support-training arms."""

from __future__ import annotations

import math
from collections.abc import Mapping

_ARMS = ("B0", "B1", "B2", "B3")
_ARM_FIELDS = frozenset(
    {
        "unique_source_scenes",
        "rows",
        "target_tokens",
        "steps",
        "optimizer",
        "lora_rank",
        "lora_targets",
        "gradient_accumulation",
        "approximate_flops",
    }
)
_INTEGER_FIELDS = (
    "unique_source_scenes",
    "rows",
    "target_tokens",
    "steps",
    "lora_rank",
    "gradient_accumulation",
)
_EXACT_FIELDS = (
    "unique_source_scenes",
    "rows",
    "steps",
    "optimizer",
    "lora_rank",
    "lora_targets",
    "gradient_accumulation",
    "approximate_flops",
)
_OPTIMIZER_FIELDS = frozenset({"name", "learning_rate", "weight_decay"})


class BudgetMismatchError(ValueError):
    """A registered support arm is missing, malformed, or compute-mismatched."""


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_closed_fields(
    payload: Mapping[object, object], expected: frozenset[str], *, label: str
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
        raise BudgetMismatchError(f"{label} has invalid closed schema: {'; '.join(detail)}")


def _validate_arm(name: str, arm: object) -> Mapping[str, object]:
    if not isinstance(arm, Mapping):
        raise BudgetMismatchError(f"{name} budget must be a mapping")
    _validate_closed_fields(arm, _ARM_FIELDS, label=f"{name} budget")

    for field in _INTEGER_FIELDS:
        value = arm[field]
        if not _is_integer(value) or value <= 0:
            raise BudgetMismatchError(f"{name}.{field} must be a positive integer")

    approximate_flops = arm["approximate_flops"]
    if not _is_finite_number(approximate_flops) or approximate_flops <= 0:
        raise BudgetMismatchError(f"{name}.approximate_flops must be finite and positive")

    targets = arm["lora_targets"]
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(target, str) or not target for target in targets)
        or len(set(targets)) != len(targets)
    ):
        raise BudgetMismatchError(f"{name}.lora_targets must be a non-empty unique string list")

    optimizer = arm["optimizer"]
    if not isinstance(optimizer, Mapping):
        raise BudgetMismatchError(f"{name}.optimizer must be a mapping")
    _validate_closed_fields(optimizer, _OPTIMIZER_FIELDS, label=f"{name}.optimizer")
    if not isinstance(optimizer["name"], str) or not optimizer["name"]:
        raise BudgetMismatchError(f"{name}.optimizer.name must be a non-empty string")
    learning_rate = optimizer["learning_rate"]
    weight_decay = optimizer["weight_decay"]
    if not _is_finite_number(learning_rate) or learning_rate <= 0:
        raise BudgetMismatchError(f"{name}.optimizer.learning_rate must be finite and positive")
    if not _is_finite_number(weight_decay) or weight_decay < 0:
        raise BudgetMismatchError(f"{name}.optimizer.weight_decay must be finite and nonnegative")
    return arm


def assert_budget_matched(
    budgets: Mapping[str, Mapping[str, object]],
    *,
    target_token_relative_tolerance: float = 0.01,
) -> None:
    """Assert that B0--B3 differ only within the target-token tolerance."""

    if not isinstance(budgets, Mapping):
        raise BudgetMismatchError("budgets must be a mapping containing arms B0, B1, B2, B3")
    actual_arms = frozenset(budgets.keys())
    expected_arms = frozenset(_ARMS)
    if actual_arms != expected_arms:
        raise BudgetMismatchError("budget arms must be exactly B0, B1, B2, B3")
    if (
        not _is_finite_number(target_token_relative_tolerance)
        or target_token_relative_tolerance < 0
        or target_token_relative_tolerance > 1
    ):
        raise BudgetMismatchError("target_tokens tolerance must be finite and in [0, 1]")

    arms = {name: _validate_arm(name, budgets[name]) for name in _ARMS}
    reference = arms["B0"]
    for field in _EXACT_FIELDS:
        for name in _ARMS[1:]:
            if arms[name][field] != reference[field]:
                raise BudgetMismatchError(f"{field} differs between B0 and {name}")

    token_counts = [arms[name]["target_tokens"] for name in _ARMS]
    minimum = min(token_counts)
    maximum = max(token_counts)
    if maximum > minimum * (1.0 + float(target_token_relative_tolerance)):
        raise BudgetMismatchError(
            "target_tokens differ beyond the preregistered relative tolerance"
        )


__all__ = ["BudgetMismatchError", "assert_budget_matched"]
