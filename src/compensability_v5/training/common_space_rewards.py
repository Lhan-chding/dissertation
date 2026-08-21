"""Reward functions over the single v5 four-integer action protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation


def exact_state_reward(candidate: WorldAction, truth: WorldAction) -> float:
    """Return one exactly when the generated world equals the true world."""

    if not isinstance(candidate, WorldAction) or not isinstance(truth, WorldAction):
        raise TypeError("candidate and truth must both be WorldAction instances")
    return float(candidate.world == truth.world)


def answer_reward(
    candidate: WorldAction,
    truth: WorldAction,
    operation: Mapping[str, Any],
) -> float:
    """Return one when candidate and truth occupy the same answer fiber."""

    if not isinstance(candidate, WorldAction) or not isinstance(truth, WorldAction):
        raise TypeError("candidate and truth must both be WorldAction instances")
    return float(
        apply_answer_operation(candidate, operation) == apply_answer_operation(truth, operation)
    )


__all__ = ["answer_reward", "exact_state_reward"]
