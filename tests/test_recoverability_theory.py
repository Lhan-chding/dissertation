from __future__ import annotations

import inspect

import pytest

from compbias.recoverability.answer_source import (
    AnswerSourceCandidate,
    TraceEvidence,
    classify_trace_candidate,
)
from compbias.recoverability.compatibility import (
    CompatibilityQuery,
    KnownValueConstraint,
    PairSumConstraint,
    analyze_compatibility,
)
from compbias.recoverability.operators import (
    Operation,
    apply_operation,
    is_operator_null_error,
)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (Operation.SUM, 13),
        (Operation.DIFFERENCE, 5),
        (Operation.MAX_MINUS_MIN, 7),
    ],
)
def test_registered_chart_operators_have_frozen_semantics(
    operation: Operation, expected: int
) -> None:
    values = (9, 4, 11, 6)

    assert apply_operation(values, operation) == expected
    assert values == (9, 4, 11, 6)


@pytest.mark.parametrize(
    ("operation", "truth", "perceived", "expected"),
    [
        (Operation.SUM, (9, 4, 6, 2), (8, 5, 6, 2), True),
        (Operation.DIFFERENCE, (9, 4, 6, 2), (8, 3, 6, 2), True),
        (Operation.MAX_MINUS_MIN, (9, 4, 6, 2), (9, 4, 7, 2), True),
        (Operation.MAX_MINUS_MIN, (9, 4, 6, 2), (10, 4, 6, 3), True),
        (Operation.MAX_MINUS_MIN, (9, 4, 6, 2), (10, 4, 6, 2), False),
        (Operation.SUM, (9, 4, 6, 2), (9, 4, 6, 2), False),
    ],
)
def test_operator_null_is_exactly_answer_invariance_under_a_real_error(
    operation: Operation,
    truth: tuple[int, ...],
    perceived: tuple[int, ...],
    expected: bool,
) -> None:
    assert is_operator_null_error(truth, perceived, operation) is expected


@pytest.mark.parametrize(
    "values",
    [(1, 2, 3), (1, 2, 3, True), (1, 2, 3, 4.0)],
)
def test_operator_inputs_are_closed_exact_integer_vectors(values: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        apply_operation(values, Operation.SUM)  # type: ignore[arg-type]


def test_recoverability_is_computed_only_from_stage2_visible_information() -> None:
    query = CompatibilityQuery(
        observed_values=(8, 4, 6, 2),
        operation=Operation.DIFFERENCE,
        constraints=(
            PairSumConstraint(constraint_id="sum_ab", left_index=0, right_index=1, total=13),
            PairSumConstraint(constraint_id="sum_bc", left_index=1, right_index=2, total=10),
        ),
        value_domain=tuple(range(0, 11)),
        max_mismatches=1,
    )

    report = analyze_compatibility(query)

    assert report.status == "ok"
    assert report.compatible_values == ((9, 4, 6, 2),)
    assert report.compatible_answers == (5,)
    assert report.exactly_recoverable is True
    assert report.bayes_ceiling == 1.0
    assert "gold" not in inspect.signature(CompatibilityQuery).parameters


def test_ablated_visible_information_remains_nonrecoverable() -> None:
    report = analyze_compatibility(
        CompatibilityQuery(
            observed_values=(8, 4, 6, 2),
            operation=Operation.DIFFERENCE,
            constraints=(),
            value_domain=tuple(range(0, 11)),
            max_mismatches=1,
        )
    )

    assert report.status == "ok"
    assert len(report.compatible_answers) > 1
    assert report.exactly_recoverable is False
    assert report.bayes_ceiling < 1.0


def test_inconsistent_visible_information_is_invalid_not_recoverable() -> None:
    report = analyze_compatibility(
        CompatibilityQuery(
            observed_values=(8, 4, 6, 2),
            operation=Operation.DIFFERENCE,
            constraints=(
                KnownValueConstraint(constraint_id="impossible", index=0, value=20),
            ),
            value_domain=tuple(range(0, 11)),
            max_mismatches=1,
        )
    )

    assert report.status == "inconsistent"
    assert report.compatible_values == ()
    assert report.compatible_answers == ()
    assert report.exactly_recoverable is False
    assert report.bayes_ceiling == 0.0


def _trace(**updates: object) -> TraceEvidence:
    payload: dict[str, object] = {
        "true_values": (9, 4, 6, 2),
        "perceived_values": (8, 4, 6, 2),
        "operation": Operation.DIFFERENCE,
        "true_answer": 5,
        "final_answer": 5,
        "perception_parse_success": True,
        "program_parse_success": True,
        "program_execution_success": True,
        "executed_result": 5,
        "recoverability_status": "recoverable",
        "consumed_constraint_ids": ("duplicate_a",),
        "required_constraint_ids": ("duplicate_a",),
    }
    payload.update(updates)
    return TraceEvidence(**payload)  # type: ignore[arg-type]


def test_trace_level_label_is_a_candidate_never_a_causal_claim() -> None:
    result = classify_trace_candidate(_trace())

    assert result is AnswerSourceCandidate.FAITHFUL_REPAIR_CANDIDATE
    assert all("genuine" not in candidate.value for candidate in AnswerSourceCandidate)
    assert all("causal" not in candidate.value for candidate in AnswerSourceCandidate)


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        (
            _trace(perception_parse_success=False),
            AnswerSourceCandidate.PARSE_FAILURE,
        ),
        (
            _trace(
                perceived_values=(9, 4, 6, 2),
                recoverability_status="not_applicable",
                consumed_constraint_ids=(),
                required_constraint_ids=(),
            ),
            AnswerSourceCandidate.FULLY_GROUNDED_CORRECT,
        ),
        (
            _trace(
                perceived_values=(8, 3, 6, 2),
                recoverability_status="not_applicable",
                consumed_constraint_ids=(),
                required_constraint_ids=(),
            ),
            AnswerSourceCandidate.OPERATOR_INVARIANT_CORRECT,
        ),
        (
            _trace(recoverability_status="nonrecoverable"),
            AnswerSourceCandidate.NONRECOVERABLE_CORRECT_CANDIDATE,
        ),
        (
            _trace(executed_result=4),
            AnswerSourceCandidate.TRACE_BYPASS_OR_UNFAITHFUL,
        ),
        (
            _trace(consumed_constraint_ids=()),
            AnswerSourceCandidate.STRICT_ERROR_CANCELLATION,
        ),
        (
            _trace(final_answer=4, executed_result=4),
            AnswerSourceCandidate.VISUAL_ERROR,
        ),
        (
            _trace(
                perceived_values=(9, 4, 6, 2),
                final_answer=4,
                executed_result=4,
                recoverability_status="not_applicable",
                consumed_constraint_ids=(),
                required_constraint_ids=(),
            ),
            AnswerSourceCandidate.REASONING_ERROR,
        ),
    ],
)
def test_trace_candidate_taxonomy_is_conservative_and_ordered(
    trace: TraceEvidence, expected: AnswerSourceCandidate
) -> None:
    assert classify_trace_candidate(trace) is expected


def test_repair_candidate_requires_every_declared_constraint_on_program_dataflow() -> None:
    trace = _trace(
        consumed_constraint_ids=("duplicate_a",),
        required_constraint_ids=("duplicate_a", "trend_aux"),
    )

    assert classify_trace_candidate(trace) is AnswerSourceCandidate.STRICT_ERROR_CANCELLATION
