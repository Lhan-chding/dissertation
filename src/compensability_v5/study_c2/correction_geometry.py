"""Correction-vector support, direction, and answer-null decomposition."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _answer_vector(operation: Mapping[str, object]) -> tuple[float, float, float, float]:
    operator = operation.get("operator")
    indices = operation.get("indices")
    if not isinstance(indices, list):
        raise ValueError("operation indices are required")
    vector = [0.0] * 4
    if operator == "sum":
        vector[int(indices[0])] = 1.0
        vector[int(indices[1])] = 1.0
    elif operator == "difference":
        vector[int(indices[0])] = 1.0
        vector[int(indices[1])] = -1.0
    else:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(vector)  # type: ignore[return-value]


def correction_vector_metrics(
    *,
    truth: tuple[int, int, int, int],
    observation: tuple[int, int, int, int],
    candidate: tuple[int, int, int, int],
    operation: Mapping[str, object],
) -> dict[str, float | int]:
    error = tuple(observed - actual for observed, actual in zip(observation, truth, strict=True))
    correction = tuple(
        observed - predicted for observed, predicted in zip(observation, candidate, strict=True)
    )
    error_support = {index for index, value in enumerate(error) if value}
    correction_support = {index for index, value in enumerate(correction) if value}
    union = error_support | correction_support
    overlap = 1.0 if not union else len(error_support & correction_support) / len(union)
    error_norm = math.sqrt(sum(value * value for value in error))
    correction_norm = math.sqrt(sum(value * value for value in correction))
    cosine = (
        0.0
        if error_norm == 0.0 or correction_norm == 0.0
        else sum(left * right for left, right in zip(error, correction, strict=True))
        / (error_norm * correction_norm)
    )
    answer_vector = _answer_vector(operation)
    vector_norm_sq = sum(value * value for value in answer_vector)
    projection_scale = (
        0.0
        if vector_norm_sq == 0.0
        else sum(left * right for left, right in zip(correction, answer_vector, strict=True))
        / vector_norm_sq
    )
    null_component = tuple(
        value - projection_scale * direction
        for value, direction in zip(correction, answer_vector, strict=True)
    )
    return {
        "edit_count": len(correction_support),
        "support_overlap": overlap,
        "direction_cosine": cosine,
        "extra_edit_count": len(correction_support - error_support),
        "answer_null_component_l1": sum(abs(value) for value in null_component),
        "exact_recovery": int(candidate == truth),
        "copy_observation": int(candidate == observation),
    }


__all__ = ["correction_vector_metrics"]
