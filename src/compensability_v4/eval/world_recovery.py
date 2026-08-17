"""World-level classification helpers."""

from __future__ import annotations

from enum import Enum


class RecoveryClassification(str, Enum):
    COPY = "copy"
    SINGLE_EDIT = "single_edit"
    OVEREDIT = "overedit"
    TRUE_RECOVERY = "true_recovery"


def classify_world_recovery(
    *,
    truth: tuple[int, int, int, int],
    observed: tuple[int, int, int, int],
    prediction: tuple[int, int, int, int],
) -> RecoveryClassification:
    if prediction == truth:
        return RecoveryClassification.TRUE_RECOVERY
    if prediction == observed:
        return RecoveryClassification.COPY
    if sum(left != right for left, right in zip(prediction, observed, strict=True)) == 1:
        return RecoveryClassification.SINGLE_EDIT
    return RecoveryClassification.OVEREDIT
