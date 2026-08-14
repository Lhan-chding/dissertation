"""Structured trajectories preserve every failure as an explicit immutable record."""

import json
from dataclasses import FrozenInstanceError

import pytest

from compbias.models.structured_parser import (
    ParseResult,
    ParseStatus,
    parse_many,
    parse_trajectory,
)

VALID = (
    '<perception>{"value": 9}</perception>'
    '<reasoning>{"operation": "add", "operand": 1}</reasoning>'
    "<answer>10</answer>"
)


def test_valid_trajectory_parses_all_three_structured_sections() -> None:
    result = parse_trajectory(VALID, sample_id="digit_001")

    assert isinstance(result, ParseResult)
    assert result.status is ParseStatus.OK
    assert result.sample_id == "digit_001"
    assert result.raw_text == VALID
    assert result.perceived_scene == {"value": 9}
    assert result.reasoning_action == {"operation": "add", "operand": 1}
    assert result.answer == 10
    assert result.error_code is None
    with pytest.raises(FrozenInstanceError):
        result.status = ParseStatus.INVALID_JSON  # type: ignore[misc]

    exported = result.to_mapping()
    assert json.loads(json.dumps(exported)) == exported
    exported["perceived_scene"]["value"] = 100
    assert result.perceived_scene == {"value": 9}


@pytest.mark.parametrize(
    ("raw", "status", "error_code"),
    [
        (
            "<perception>{oops}</perception><reasoning>{}</reasoning><answer>1</answer>",
            ParseStatus.INVALID_JSON,
            "invalid_perception_json",
        ),
        (
            '<perception>{"value": 1}</perception><answer>1</answer>',
            ParseStatus.MISSING_FIELD,
            "missing_reasoning",
        ),
        (
            '<perception>{"value": 1}</perception><reasoning>{}</reasoning>'
            "<answer>1</answer><answer>2</answer>",
            ParseStatus.MALFORMED,
            "duplicate_answer",
        ),
        ("free-form answer: 10", ParseStatus.MALFORMED, "missing_structured_tags"),
    ],
)
def test_parse_failures_are_returned_not_raised_or_silently_dropped(
    raw: str, status: ParseStatus, error_code: str
) -> None:
    result = parse_trajectory(raw, sample_id="bad_001")

    assert result.status is status
    assert result.error_code == error_code
    assert result.raw_text == raw
    assert result.answer is None


def test_parse_many_preserves_input_order_and_one_result_per_rollout() -> None:
    records = (
        {"sample_id": "ok", "raw_text": VALID},
        {"sample_id": "bad", "raw_text": "not structured"},
    )

    parsed = parse_many(records)

    assert isinstance(parsed, tuple)
    assert [result.sample_id for result in parsed] == ["ok", "bad"]
    assert [result.status for result in parsed] == [ParseStatus.OK, ParseStatus.MALFORMED]
    assert len(parsed) == len(records)


def test_parser_rejects_non_string_boundary_input_with_a_clear_result() -> None:
    result = parse_trajectory(None, sample_id="none")  # type: ignore[arg-type]

    assert result.status is ParseStatus.INVALID_TYPE
    assert result.error_code == "raw_text_not_string"


def test_non_string_failure_evidence_is_detached_and_json_serializable() -> None:
    raw = {"nested": [1]}

    result = parse_trajectory(raw, sample_id="mapping")  # type: ignore[arg-type]
    raw["nested"].append(2)
    exported = result.to_mapping()

    assert exported["raw_text"] == {"nested": [1]}
    json.dumps(exported)


@pytest.mark.parametrize(
    "fragment",
    [
        '{"value": NaN}',
        '{"value": Infinity}',
        '{"value": 1e999}',
        '{"value": 1, "value": 2}',
    ],
)
def test_parser_rejects_nonstandard_or_ambiguous_json(fragment: str) -> None:
    raw = (
        f"<perception>{fragment}</perception>"
        '<reasoning>{"operation": "identity"}</reasoning>'
        "<answer>1</answer>"
    )

    result = parse_trajectory(raw, sample_id="strict-json")

    assert result.status is ParseStatus.INVALID_JSON
    assert result.error_code == "invalid_perception_json"


def test_parser_bounds_total_bytes_and_json_nesting_depth() -> None:
    oversized = (
        '<perception>{"value":"'
        + "x" * 70_000
        + '"}</perception><reasoning>{}</reasoning><answer>1</answer>'
    )
    too_deep = (
        "<perception>"
        + "[" * 40
        + "0"
        + "]" * 40
        + "</perception><reasoning>{}</reasoning><answer>1</answer>"
    )

    size_result = parse_trajectory(oversized, sample_id="oversized")
    depth_result = parse_trajectory(too_deep, sample_id="too-deep")

    assert size_result.status is ParseStatus.MALFORMED
    assert size_result.error_code == "raw_text_too_large"
    assert depth_result.status is ParseStatus.INVALID_JSON
    assert depth_result.error_code == "invalid_perception_json"


def test_parser_returns_failure_for_invalid_unicode_instead_of_raising() -> None:
    raw = '<perception>{"value":"\ud800"}</perception><reasoning>{}</reasoning><answer>1</answer>'

    result = parse_trajectory(raw, sample_id="invalid-unicode")

    assert result.status is ParseStatus.MALFORMED
    assert result.error_code == "raw_text_invalid_unicode"
