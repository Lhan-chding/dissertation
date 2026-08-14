"""Shared natural-error taxonomy for collection and fail-closed replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from compbias.models.structured_parser import ParseStatus

from .structured_generation import numeric_answer_matches

PERCEPTION_ERROR_TYPES = frozenset(
    {
        "visual_error",
        "compensated_visual_error",
        "operator_invariant_visual_error",
    }
)


def _integer_values(value: object, *, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) < 2
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{label} must contain at least two integers")
    return tuple(value)


def pilot_operation_result(values: object, operation: object) -> int:
    """Apply the registered ground-truth pilot operation to validated values."""

    integers = _integer_values(values, label="pilot values")
    if operation == "sum":
        return integers[0] + integers[1]
    if operation == "difference":
        return integers[0] - integers[1]
    if operation == "max_minus_min":
        return max(integers) - min(integers)
    raise ValueError("pilot operation must be sum, difference, or max_minus_min")


def natural_error_type(record: Mapping[str, object], parsed: object) -> str:
    """Classify one first response using only canonical data and strict parsing."""

    status = getattr(parsed, "status", None)
    if status is not ParseStatus.OK:
        return "parse_failure"
    expected_values = _integer_values(record.get("values"), label="dataset values")
    perceived = getattr(parsed, "perceived_scene", None)
    perceived_raw = perceived.get("values") if isinstance(perceived, Mapping) else None
    perceived_values = _integer_values(perceived_raw, label="perceived values")
    perception_correct = perceived_values == expected_values
    answer_correct = numeric_answer_matches(getattr(parsed, "answer", None), record.get("answer"))
    if perception_correct:
        return "none" if answer_correct else "reasoning_error"
    if not answer_correct:
        return "visual_error"
    perceived_result = pilot_operation_result(perceived_values, record.get("operation"))
    if numeric_answer_matches(perceived_result, record.get("answer")):
        return "operator_invariant_visual_error"
    return "compensated_visual_error"


def is_perception_error(error_type: str) -> bool:
    return error_type in PERCEPTION_ERROR_TYPES
