"""Bounded format-repair orchestration for structured VLM evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from compbias.models.structured_parser import ParseResult, ParseStatus, parse_trajectory

_ALLOWED_OPERATIONS = frozenset({"difference", "sum", "max_minus_min"})
_MAX_QUESTION_BYTES = 4_096
_MAX_SAMPLE_ID_BYTES = 256
_MAX_FORMAT_RETRIES = 2
_MAX_ABSOLUTE_NUMBER = 1_000_000


@dataclass(frozen=True)
class StructuredGeneration:
    """Final strict parse plus every bounded formatting attempt."""

    raw_text: str
    parsed: ParseResult
    attempts: tuple[Mapping[str, object], ...]


def _bounded_text(value: str, *, label: str, maximum_bytes: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError(f"{label} must be valid UTF-8") from error
    if size > maximum_bytes:
        raise ValueError(f"{label} exceeds {maximum_bytes} bytes")
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL")
    return value


def _validate_expected_value_count(expected_value_count: int) -> int:
    if (
        not isinstance(expected_value_count, int)
        or isinstance(expected_value_count, bool)
        or not 2 <= expected_value_count <= 32
    ):
        raise ValueError("expected_value_count must be an integer between 2 and 32")
    return expected_value_count


def build_structured_instruction(*, operation: str, expected_value_count: int) -> str:
    """Return one closed grammar shared by smoke, collection, and training."""

    if operation not in _ALLOWED_OPERATIONS:
        raise ValueError(f"unsupported operation: {operation!r}")
    expected_value_count = _validate_expected_value_count(expected_value_count)
    example_values = (2, 8, *([5] * (expected_value_count - 2)))
    example_answer = {
        "sum": example_values[0] + example_values[1],
        "difference": example_values[0] - example_values[1],
        "max_minus_min": max(example_values) - min(example_values),
    }[operation]
    example_perception = json.dumps(
        {"values": example_values},
        separators=(",", ":"),
        allow_nan=False,
    )
    labels = ", ".join(chr(ord("A") + index) for index in range(expected_value_count))
    return (
        "Required grammar: "
        '<perception>{"values":[INTEGER,...]}</perception>'
        f'<reasoning>{{"operation":"{operation}"}}</reasoning>'
        "<answer>INTEGER_OR_FINITE_NUMBER</answer> "
        f"The values array must contain exactly {expected_value_count} integers. "
        f"Transcribe all {expected_value_count} labeled values in {labels} order, even when the "
        "question uses only some of them. For sum use A+B; for difference use A-B; all other "
        "displayed values still belong in perception. "
        "Do not insert \\n or any escape sequence between the perception JSON and its closing tag. "
        "The response's final character must be >; add no trailing punctuation or prose. "
        "Example format only, not the answer to this task: "
        f"<perception>{example_perception}</perception>"
        f'<reasoning>{{"operation":"{operation}"}}</reasoning>'
        f"<answer>{example_answer}</answer>"
    )


def build_structured_messages(
    *,
    question: str,
    operation: str,
    retry_index: int,
    expected_value_count: int,
) -> tuple[dict[str, object], ...]:
    """Build a strict prompt without quoting an untrusted prior response."""

    question = _bounded_text(question, label="question", maximum_bytes=_MAX_QUESTION_BYTES)
    if not isinstance(retry_index, int) or isinstance(retry_index, bool):
        raise TypeError("retry_index must be an integer")
    if not 0 <= retry_index <= _MAX_FORMAT_RETRIES:
        raise ValueError(f"retry_index must be between 0 and {_MAX_FORMAT_RETRIES}")

    retry_instruction = ""
    if retry_index:
        retry_instruction = (
            f" A previous attempt failed format validation. This is format-repair attempt "
            f"{retry_index} of {_MAX_FORMAT_RETRIES}. Retry from the image and question; do not "
            "quote, explain, or repeat the previous response."
        )
    system = (
        "You are a strict structured-output interface. Return exactly one line containing the "
        "three required tags in order. Do not use Markdown fences or add prose."
    )
    instruction = build_structured_instruction(
        operation=operation,
        expected_value_count=expected_value_count,
    )
    user = f"{question}{retry_instruction} {instruction}"
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    )


def _schema_failure(parsed: ParseResult, error_code: str) -> ParseResult:
    return ParseResult(
        status=ParseStatus.INVALID_TYPE,
        sample_id=parsed.sample_id,
        raw_text=parsed.raw_text,
        error_code=error_code,
    )


def _is_bounded_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= _MAX_ABSOLUTE_NUMBER
    if isinstance(value, float):
        return math.isfinite(value) and abs(value) <= _MAX_ABSOLUTE_NUMBER
    return False


def validate_pilot_trajectory(
    parsed: ParseResult,
    *,
    operation: str,
    expected_value_count: int,
) -> ParseResult:
    """Enforce the closed pilot schema after generic tag/JSON parsing."""

    if operation not in _ALLOWED_OPERATIONS:
        raise ValueError(f"unsupported operation: {operation!r}")
    expected_value_count = _validate_expected_value_count(expected_value_count)
    if parsed.status is not ParseStatus.OK:
        return parsed

    perception = parsed.perceived_scene
    if perception is None or set(perception) != {"values"}:
        return _schema_failure(parsed, "pilot_perception_schema")
    values = perception["values"]
    if not isinstance(values, tuple) or len(values) != expected_value_count:
        return _schema_failure(parsed, "pilot_values_shape")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or abs(value) > _MAX_ABSOLUTE_NUMBER
        for value in values
    ):
        return _schema_failure(parsed, "pilot_values_type")

    reasoning = parsed.reasoning_action
    if reasoning is None or set(reasoning) != {"operation"}:
        return _schema_failure(parsed, "pilot_reasoning_schema")
    if reasoning["operation"] != operation:
        return _schema_failure(parsed, "pilot_operation_mismatch")

    answer = parsed.answer
    if not _is_bounded_number(answer):
        return _schema_failure(parsed, "pilot_answer_type")
    return parsed


def numeric_answer_matches(answer: object, expected: object) -> bool:
    """Compare finite JSON numbers without accepting booleans or strings."""

    return _is_bounded_number(answer) and _is_bounded_number(expected) and answer == expected


def generate_with_format_retries(
    generate_once: Callable[[tuple[dict[str, object], ...]], str],
    *,
    question: str,
    operation: str,
    sample_id: str,
    expected_value_count: int,
    max_format_retries: int = _MAX_FORMAT_RETRIES,
) -> StructuredGeneration:
    """Generate at most three times, retrying only strict-format failures."""

    sample_id = _bounded_text(
        sample_id,
        label="sample_id",
        maximum_bytes=_MAX_SAMPLE_ID_BYTES,
    )
    if not isinstance(max_format_retries, int) or isinstance(max_format_retries, bool):
        raise TypeError("max_format_retries must be an integer")
    if not 0 <= max_format_retries <= _MAX_FORMAT_RETRIES:
        raise ValueError(f"max_format_retries must be between 0 and {_MAX_FORMAT_RETRIES}")

    attempts: list[Mapping[str, object]] = []
    final_raw = ""
    final_parsed: ParseResult | None = None
    for retry_index in range(max_format_retries + 1):
        messages = build_structured_messages(
            question=question,
            operation=operation,
            retry_index=retry_index,
            expected_value_count=expected_value_count,
        )
        raw = generate_once(messages)
        if not isinstance(raw, str):
            raise TypeError("model decoder must return a string")
        parsed = validate_pilot_trajectory(
            parse_trajectory(raw, sample_id=sample_id),
            operation=operation,
            expected_value_count=expected_value_count,
        )
        attempts.append(
            {
                "attempt_index": retry_index,
                "raw_text": raw,
                "status": parsed.status.value,
                "error_code": parsed.error_code,
            }
        )
        final_raw = raw
        final_parsed = parsed
        if parsed.status is ParseStatus.OK:
            break

    assert final_parsed is not None
    return StructuredGeneration(
        raw_text=final_raw,
        parsed=final_parsed,
        attempts=tuple(attempts),
    )
