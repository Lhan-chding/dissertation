"""Mutually exclusive Study C3 action and parse-failure classifications."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation

from .semantic_action_parser import World, parse_complete_action_any_domain, parse_p0, parse_p1


class ActionClass(str, Enum):
    X = "X"
    S = "S"
    W = "W"
    I = "I"  # noqa: E741 - registered Study C3 invalid-action symbol


class FailureCategory(str, Enum):
    CANONICAL_VALID_IN_DOMAIN = "canonical_valid_in_domain"
    SEMANTIC_VALID_NONCANONICAL = "semantic_valid_noncanonical"
    COMPLETE_OUT_OF_DOMAIN = "complete_out_of_domain"
    LABELED_INCOMPLETE_OR_TRUNCATED = "labeled_incomplete_or_truncated"
    FREE_PROSE_WITHOUT_COMPLETE_ACTION = "free_prose_without_complete_action"
    OTHER_INVALID = "other_invalid"


@dataclass(frozen=True, slots=True)
class ActionClassification:
    action_class: ActionClass
    parsed_world: World | None
    exact: int
    answer_correct: int
    valid: int


def classify_action(
    candidate: World | None,
    *,
    truth: World,
    operation: Mapping[str, object],
) -> ActionClassification:
    if candidate is None:
        return ActionClassification(ActionClass.I, None, 0, 0, 0)
    exact = candidate == truth
    answer_correct = apply_answer_operation(
        WorldAction(candidate), operation
    ) == apply_answer_operation(WorldAction(truth), operation)
    action_class = ActionClass.X if exact else ActionClass.S if answer_correct else ActionClass.W
    return ActionClassification(action_class, candidate, int(exact), int(answer_correct), 1)


def classify_failure(completion: str) -> FailureCategory:
    if parse_p0(completion) is not None:
        return FailureCategory.CANONICAL_VALID_IN_DOMAIN
    if parse_p1(completion) is not None:
        return FailureCategory.SEMANTIC_VALID_NONCANONICAL
    if parse_complete_action_any_domain(completion) is not None:
        return FailureCategory.COMPLETE_OUT_OF_DOMAIN
    if re.search(r"\bv[1-4]\s*[:=]", completion, flags=re.IGNORECASE):
        return FailureCategory.LABELED_INCOMPLETE_OR_TRUNCATED
    if re.search(r"[A-Za-z]", completion):
        return FailureCategory.FREE_PROSE_WITHOUT_COMPLETE_ACTION
    return FailureCategory.OTHER_INVALID


__all__ = [
    "ActionClass",
    "ActionClassification",
    "FailureCategory",
    "classify_action",
    "classify_failure",
]
