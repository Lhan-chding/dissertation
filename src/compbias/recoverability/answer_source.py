"""Conservative trace-level answer-source candidates.

Population-level causal repair is deliberately absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .operators import Operation, apply_operation


class AnswerSourceCandidate(str, Enum):
    PARSE_FAILURE = "parse_failure"
    FULLY_GROUNDED_CORRECT = "fully_grounded_correct"
    REASONING_ERROR = "reasoning_error"
    OPERATOR_INVARIANT_CORRECT = "operator_invariant_correct"
    FAITHFUL_REPAIR_CANDIDATE = "faithful_repair_candidate"
    NONRECOVERABLE_CORRECT_CANDIDATE = "nonrecoverable_correct_candidate"
    STRICT_ERROR_CANCELLATION = "strict_error_cancellation"
    TRACE_BYPASS_OR_UNFAITHFUL = "trace_bypass_or_unfaithful"
    VISUAL_ERROR = "visual_error"


@dataclass(frozen=True, slots=True)
class TraceEvidence:
    true_values: tuple[int, int, int, int]
    perceived_values: tuple[int, int, int, int]
    operation: Operation
    true_answer: int
    final_answer: int
    perception_parse_success: bool
    program_parse_success: bool
    program_execution_success: bool
    executed_result: int | None
    recoverability_status: Literal["recoverable", "nonrecoverable", "not_applicable"]
    consumed_constraint_ids: tuple[str, ...]
    required_constraint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("true_values", "perceived_values"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != 4:
                raise ValueError(f"{name} must be an exact four-integer tuple")
            if any(type(value) is not int for value in values):
                raise TypeError(f"{name} must contain exact integers")
        try:
            object.__setattr__(self, "operation", Operation(self.operation))
        except (TypeError, ValueError) as error:
            raise ValueError("operation is not registered") from error
        for name in ("true_answer", "final_answer"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if self.executed_result is not None and type(self.executed_result) is not int:
            raise TypeError("executed_result must be an exact integer or None")
        for name in (
            "perception_parse_success",
            "program_parse_success",
            "program_execution_success",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if self.recoverability_status not in {
            "recoverable",
            "nonrecoverable",
            "not_applicable",
        }:
            raise ValueError("recoverability_status is not registered")
        for name in ("consumed_constraint_ids", "required_constraint_ids"):
            identifiers = getattr(self, name)
            if not isinstance(identifiers, tuple) or any(
                not isinstance(item, str) or not item for item in identifiers
            ):
                raise ValueError(f"{name} must contain non-empty string identifiers")
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"{name} must not contain duplicates")


def classify_trace_candidate(trace: TraceEvidence) -> AnswerSourceCandidate:
    """Classify observable trace facts without making an arm-level causal claim."""

    if not isinstance(trace, TraceEvidence):
        raise TypeError("trace must be TraceEvidence")
    if not trace.perception_parse_success or not trace.program_parse_success:
        return AnswerSourceCandidate.PARSE_FAILURE

    perception_correct = trace.perceived_values == trace.true_values
    answer_correct = trace.final_answer == trace.true_answer
    program_matches = (
        trace.program_execution_success and trace.executed_result == trace.final_answer
    )
    if perception_correct:
        if answer_correct and program_matches:
            return AnswerSourceCandidate.FULLY_GROUNDED_CORRECT
        return AnswerSourceCandidate.REASONING_ERROR

    if not answer_correct:
        return AnswerSourceCandidate.VISUAL_ERROR
    perceived_answer = apply_operation(trace.perceived_values, trace.operation)
    if perceived_answer == trace.true_answer:
        return AnswerSourceCandidate.OPERATOR_INVARIANT_CORRECT
    if not program_matches:
        return AnswerSourceCandidate.TRACE_BYPASS_OR_UNFAITHFUL
    if trace.recoverability_status == "nonrecoverable":
        return AnswerSourceCandidate.NONRECOVERABLE_CORRECT_CANDIDATE
    required = set(trace.required_constraint_ids)
    consumed = set(trace.consumed_constraint_ids)
    if trace.recoverability_status == "recoverable" and required and required <= consumed:
        return AnswerSourceCandidate.FAITHFUL_REPAIR_CANDIDATE
    return AnswerSourceCandidate.STRICT_ERROR_CANCELLATION
