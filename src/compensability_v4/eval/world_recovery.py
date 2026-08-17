"""Mutually exclusive and complete v4 world-recovery taxonomy."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from compensability_v4.theory.constraint_system import validate_world


class RecoveryClassification(str, Enum):
    COPY = "copy"
    SINGLE_EDIT = "single_edit"
    OVEREDIT = "overedit"
    TRUE_RECOVERY = "true_recovery"


def hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    canonical_left = validate_world(left, "left")
    canonical_right = validate_world(right, "right")
    return sum(a != b for a, b in zip(canonical_left, canonical_right, strict=True))


def classify_world_recovery(
    *, truth: Sequence[int], observed: Sequence[int], prediction: Sequence[int]
) -> RecoveryClassification:
    canonical_truth = validate_world(truth, "truth")
    canonical_observed = validate_world(observed, "observed")
    canonical_prediction = validate_world(prediction, "prediction")
    if canonical_prediction == canonical_truth:
        return RecoveryClassification.TRUE_RECOVERY
    if canonical_prediction == canonical_observed:
        return RecoveryClassification.COPY
    if hamming_distance(canonical_prediction, canonical_observed) == 1:
        return RecoveryClassification.SINGLE_EDIT
    return RecoveryClassification.OVEREDIT


@dataclass(frozen=True, slots=True)
class RecoveryRates:
    number_of_scenes: int
    copy_rate: float
    single_edit_rate: float
    overedit_rate: float
    true_recovery_rate: float


def summarize_world_recovery(
    rows: Iterable[tuple[Sequence[int], Sequence[int], Sequence[int]]],
) -> RecoveryRates:
    classifications = tuple(
        classify_world_recovery(truth=truth, observed=observed, prediction=prediction)
        for truth, observed, prediction in rows
    )
    if not classifications:
        raise ValueError("rows must not be empty")
    count = len(classifications)

    def rate(label: RecoveryClassification) -> float:
        return sum(value is label for value in classifications) / count

    return RecoveryRates(
        number_of_scenes=count,
        copy_rate=rate(RecoveryClassification.COPY),
        single_edit_rate=rate(RecoveryClassification.SINGLE_EDIT),
        overedit_rate=rate(RecoveryClassification.OVEREDIT),
        true_recovery_rate=rate(RecoveryClassification.TRUE_RECOVERY),
    )
