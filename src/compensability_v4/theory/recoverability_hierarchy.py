"""Named gaps and ordered recoverability levels from v4 Sections 2 and 3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class RecoverabilityLevel(str, Enum):
    DESIGN_IDENTIFIABILITY = "R0"
    INTERFACE_CONDITIONAL_RECOVERABILITY = "R1"
    ALGORITHMIC_DECODABILITY = "R2"
    POLICY_ACCESSIBILITY = "R3"
    RL_LEARNABILITY = "R4"


def _rate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def search_gap(candidate_selection_accuracy: float, free_recovery_accuracy: float) -> float:
    return _rate(candidate_selection_accuracy, "candidate_selection_accuracy") - _rate(
        free_recovery_accuracy, "free_recovery_accuracy"
    )


def localization_gap(given_index_accuracy: float, inferred_index_accuracy: float) -> float:
    return _rate(given_index_accuracy, "given_index_accuracy") - _rate(
        inferred_index_accuracy, "inferred_index_accuracy"
    )


def interface_gap(cache_accuracy: float, hard_text_accuracy: float) -> float:
    return _rate(cache_accuracy, "cache_accuracy") - _rate(
        hard_text_accuracy, "hard_text_accuracy"
    )


@dataclass(frozen=True, slots=True)
class RecoverabilityGaps:
    search: float
    localization: float
    interface: float

    @classmethod
    def from_accuracies(
        cls,
        *,
        t3: float,
        t4_given_index: float,
        t5: float,
        t6: float,
        cache: float,
        hard_text: float,
    ) -> "RecoverabilityGaps":
        return cls(
            search=search_gap(t5, t6),
            localization=localization_gap(t4_given_index, t3),
            interface=interface_gap(cache, hard_text),
        )
