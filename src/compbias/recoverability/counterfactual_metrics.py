"""Paired coherent-world counterfactual compliance metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class CounterfactualForkPair:
    scene_id: str
    fork_id: str
    valid_answer: int
    counterfactual_answer: int
    original_answer: int
    counterfactual_gold_answer: int
    valid_faithful: bool
    counterfactual_faithful: bool

    def __post_init__(self) -> None:
        for value, label in ((self.scene_id, "scene_id"), (self.fork_id, "fork_id")):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"{label} must be a bounded safe identifier")
        for label in (
            "valid_answer",
            "counterfactual_answer",
            "original_answer",
            "counterfactual_gold_answer",
        ):
            if type(getattr(self, label)) is not int:
                raise TypeError(f"{label} must be an exact integer")
        if type(self.valid_faithful) is not bool or type(self.counterfactual_faithful) is not bool:
            raise TypeError("faithfulness indicators must be boolean")


@dataclass(frozen=True, slots=True)
class CounterfactualSummary:
    paired_forks: int
    target_shift_effect: float
    original_retention_effect: float
    strict_dual_world_faithful_success: float
    answer_direction_compliance: float
    counterfactual_consistency: float


def summarize_counterfactual_consistency(
    pairs: tuple[CounterfactualForkPair, ...],
) -> CounterfactualSummary:
    if not isinstance(pairs, tuple) or not pairs:
        raise ValueError("pairs must be a non-empty tuple")
    if any(not isinstance(item, CounterfactualForkPair) for item in pairs):
        raise TypeError("pairs must contain CounterfactualForkPair instances")
    keys = [(item.scene_id, item.fork_id) for item in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("counterfactual pair keys must be unique")
    if any(item.original_answer == item.counterfactual_gold_answer for item in pairs):
        raise ValueError("original and counterfactual gold answers must differ")
    target: list[int] = []
    original: list[int] = []
    dual: list[int] = []
    direction: list[int] = []
    consistency: list[int] = []
    for item in pairs:
        valid_correct = item.valid_answer == item.original_answer
        counterfactual_correct = item.counterfactual_answer == item.counterfactual_gold_answer
        target.append(
            int(counterfactual_correct) - int(item.valid_answer == item.counterfactual_gold_answer)
        )
        original.append(
            int(valid_correct) - int(item.counterfactual_answer == item.original_answer)
        )
        both_correct = valid_correct and counterfactual_correct
        consistency.append(int(both_correct))
        dual.append(int(both_correct and item.valid_faithful and item.counterfactual_faithful))
        direction.append(
            int(
                item.counterfactual_answer - item.valid_answer
                == item.counterfactual_gold_answer - item.original_answer
            )
        )
    denominator = len(pairs)
    return CounterfactualSummary(
        paired_forks=denominator,
        target_shift_effect=sum(target) / denominator,
        original_retention_effect=sum(original) / denominator,
        strict_dual_world_faithful_success=sum(dual) / denominator,
        answer_direction_compliance=sum(direction) / denominator,
        counterfactual_consistency=sum(consistency) / denominator,
    )
