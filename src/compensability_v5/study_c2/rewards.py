"""Dual reward labels and the X/S/F/U rollout partition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation


class RolloutKind(str, Enum):
    X = "X"
    S = "S"
    F = "F"
    U = "U"


@dataclass(frozen=True, slots=True)
class RewardClassification:
    kind: RolloutKind
    parsed_world: tuple[int, int, int, int] | None
    answer_reward: int
    state_reward: int


def classify_world(
    candidate: tuple[int, int, int, int] | None,
    *,
    truth: tuple[int, int, int, int],
    operation: Mapping[str, object],
) -> RewardClassification:
    if candidate is None:
        return RewardClassification(RolloutKind.U, None, 0, 0)
    candidate_action = WorldAction(candidate)
    truth_action = WorldAction(truth)
    exact = candidate == truth
    answer = apply_answer_operation(candidate_action, operation) == apply_answer_operation(
        truth_action, operation
    )
    kind = RolloutKind.X if exact else RolloutKind.S if answer else RolloutKind.F
    return RewardClassification(kind, candidate, int(answer), int(exact))


def classify_completion(
    completion: str,
    *,
    truth: tuple[int, int, int, int],
    operation: Mapping[str, object],
) -> RewardClassification:
    from .action_protocol import parse_first_world_action

    return classify_world(parse_first_world_action(completion), truth=truth, operation=operation)


__all__ = ["RewardClassification", "RolloutKind", "classify_completion", "classify_world"]
