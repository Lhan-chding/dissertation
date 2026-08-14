from __future__ import annotations

import json

import pytest

from compbias.recoverability.dsl import (
    ProgramExecutionError,
    ProgramParseError,
    TrustedBinding,
    evaluate_program,
    execute_program,
    parse_program,
)


def _raw(
    *,
    variables: dict[str, int] | None = None,
    steps: list[dict[str, object]] | None = None,
    answer: int = 0,
) -> str:
    return json.dumps(
        {
            "variables": variables or {},
            "steps": steps or [],
            "answer": answer,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


@pytest.mark.parametrize(
    ("variables", "step", "expected"),
    [
        ({"a": 8}, {"op": "read", "inputs": ["a"], "output": "result"}, 8),
        (
            {"a": 8, "b": 4},
            {"op": "add", "inputs": ["a", "b"], "output": "result"},
            12,
        ),
        (
            {"a": 8, "b": 4},
            {"op": "subtract", "inputs": ["a", "b"], "output": "result"},
            4,
        ),
        (
            {"a": 8, "b": 4, "c": 11},
            {"op": "max", "inputs": ["a", "b", "c"], "output": "result"},
            11,
        ),
        (
            {"a": 8, "b": 4, "c": 11},
            {"op": "min", "inputs": ["a", "b", "c"], "output": "result"},
            4,
        ),
        (
            {"a": 8, "b": 11, "c": 11},
            {"op": "argmax", "inputs": ["a", "b", "c"], "output": "result"},
            1,
        ),
        (
            {"a": 4, "b": 4, "c": 11},
            {"op": "argmin", "inputs": ["a", "b", "c"], "output": "result"},
            0,
        ),
        (
            {"total": 13, "b": 4},
            {
                "op": "solve_sum_constraint",
                "inputs": ["total", "b"],
                "output": "result",
            },
            9,
        ),
        (
            {"difference": 5, "b": 4},
            {
                "op": "solve_difference_constraint",
                "inputs": ["difference", "b"],
                "output": "result",
            },
            9,
        ),
        (
            {"left": 5, "right": 9},
            {
                "op": "interpolate_arithmetic_progression",
                "inputs": ["left", "right"],
                "output": "result",
            },
            7,
        ),
        (
            {"duplicate": 8},
            {"op": "lookup_duplicate", "inputs": ["duplicate"], "output": "result"},
            8,
        ),
        (
            {"a": 8, "b": 4},
            {"op": "compare", "inputs": ["a", "b"], "output": "result"},
            1,
        ),
    ],
)
def test_every_registered_dsl_operation_has_deterministic_semantics(
    variables: dict[str, int], step: dict[str, object], expected: int
) -> None:
    program = parse_program(_raw(variables=variables, steps=[step], answer=expected))

    result = execute_program(program, constraint_bindings={})

    assert result.program_execution_success is True
    assert result.executed_result == expected
    assert result.final_answer == expected
    assert result.program_answer_match is True


def test_executor_tracks_only_external_constraint_bindings_on_real_dataflow() -> None:
    raw = _raw(
        variables={"total": 13, "b": 4, "unused": 99},
        steps=[
            {
                "op": "solve_sum_constraint",
                "inputs": ["total", "b"],
                "output": "a_recovered",
            },
            {
                "op": "subtract",
                "inputs": ["a_recovered", "b"],
                "output": "result",
            },
        ],
        answer=5,
    )

    result = execute_program(
        parse_program(raw),
        constraint_bindings={
            "total": TrustedBinding("sum_ab", 13),
            "unused": TrustedBinding("sham_unused", 99),
        },
    )

    assert result.executed_result == 5
    assert result.consumed_constraint_ids == ("sum_ab",)
    assert "sham_unused" not in result.consumed_constraint_ids


def test_executor_rejects_model_declared_values_that_disagree_with_trusted_evidence() -> None:
    raw = _raw(
        variables={"total": 99, "b": 4},
        steps=[
            {
                "op": "solve_sum_constraint",
                "inputs": ["total", "b"],
                "output": "a_recovered",
            }
        ],
        answer=95,
    )

    with pytest.raises(ProgramExecutionError, match="trusted evidence"):
        execute_program(
            parse_program(raw),
            constraint_bindings={"total": TrustedBinding("sum_ab", 13)},
        )


def test_program_answer_mismatch_is_recorded_and_never_overwrites_the_answer() -> None:
    raw = _raw(
        variables={"a": 8, "b": 4},
        steps=[{"op": "subtract", "inputs": ["a", "b"], "output": "result"}],
        answer=5,
    )

    result = execute_program(parse_program(raw), constraint_bindings={})

    assert result.executed_result == 4
    assert result.final_answer == 5
    assert result.program_answer_match is False


@pytest.mark.parametrize(
    "raw",
    [
        '{"variables":{},"steps":[],"answer":0} trailing',
        '{"variables":{},"steps":[],"answer":0,"extra":1}',
        '{"variables":{"a":1,"a":2},"steps":[],"answer":0}',
        '{"variables":{"a":true},"steps":[],"answer":0}',
        '{"variables":{"a":NaN},"steps":[],"answer":0}',
        '{"variables":{"a":Infinity},"steps":[],"answer":0}',
        '{"variables":{},"steps":[],"answer":1.0}',
        "[]",
        "not-json",
    ],
)
def test_parser_rejects_noncanonical_or_nonfinite_json(raw: str) -> None:
    with pytest.raises(ProgramParseError):
        parse_program(raw)


@pytest.mark.parametrize(
    "step",
    [
        {"op": "divide", "inputs": ["a", "b"], "output": "result"},
        {"op": "add", "inputs": ["a", "b"], "output": "result", "extra": 1},
        {"op": "add", "inputs": ["a", "missing"], "output": "result"},
        {"op": "read", "inputs": ["a"], "output": "a"},
        {"op": "read", "inputs": ["a"], "output": "bad output"},
    ],
)
def test_parser_rejects_unknown_ops_keys_forward_refs_and_overwrites(
    step: dict[str, object],
) -> None:
    with pytest.raises(ProgramParseError):
        parse_program(_raw(variables={"a": 1, "b": 2}, steps=[step]))


def test_parser_rejects_duplicate_outputs_and_excess_steps_or_bytes() -> None:
    duplicate = _raw(
        variables={"a": 1},
        steps=[
            {"op": "read", "inputs": ["a"], "output": "result"},
            {"op": "read", "inputs": ["a"], "output": "result"},
        ],
    )
    with pytest.raises(ProgramParseError, match="output"):
        parse_program(duplicate)

    too_many = _raw(
        variables={"a": 1},
        steps=[{"op": "read", "inputs": ["a"], "output": f"value_{index}"} for index in range(33)],
    )
    with pytest.raises(ProgramParseError, match="steps"):
        parse_program(too_many)
    with pytest.raises(ProgramParseError, match="bytes"):
        parse_program(" " * 16_385)


@pytest.mark.parametrize(
    ("variables", "step", "message"),
    [
        (
            {"left": 4, "right": 9},
            {
                "op": "interpolate_arithmetic_progression",
                "inputs": ["left", "right"],
                "output": "result",
            },
            "integer midpoint",
        ),
        (
            {"a": 1},
            {"op": "add", "inputs": ["a"], "output": "result"},
            "arity",
        ),
    ],
)
def test_executor_fails_closed_on_invalid_operation_domain(
    variables: dict[str, int], step: dict[str, object], message: str
) -> None:
    program = parse_program(_raw(variables=variables, steps=[step]))

    with pytest.raises(ProgramExecutionError, match=message):
        execute_program(program, constraint_bindings={})


def test_evaluation_wrapper_preserves_parse_and_execution_failures() -> None:
    malformed = evaluate_program("not-json", constraint_bindings={})
    assert malformed.program_parse_success is False
    assert malformed.program_execution_success is False
    assert malformed.error_code == "program_parse_failure"

    invalid_execution = evaluate_program(
        _raw(
            variables={"left": 4, "right": 9},
            steps=[
                {
                    "op": "interpolate_arithmetic_progression",
                    "inputs": ["left", "right"],
                    "output": "result",
                }
            ],
        ),
        constraint_bindings={},
    )
    assert invalid_execution.program_parse_success is True
    assert invalid_execution.program_execution_success is False
    assert invalid_execution.error_code == "program_execution_failure"


def test_parser_and_executor_do_not_mutate_caller_inputs() -> None:
    raw = _raw(
        variables={"a": 8},
        steps=[{"op": "read", "inputs": ["a"], "output": "result"}],
        answer=8,
    )
    bindings = {"a": TrustedBinding("observed_a", 8)}
    original = dict(bindings)

    first = execute_program(parse_program(raw), constraint_bindings=bindings)
    second = execute_program(parse_program(raw), constraint_bindings=bindings)

    assert first == second
    assert bindings == original
