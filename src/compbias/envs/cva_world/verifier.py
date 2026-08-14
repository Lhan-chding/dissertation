"""Strict family-aware outcome verification for CVA-World answers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from compbias.models.structured_parser import ParseResult, ParseStatus

from .canonical_solver import solve_sample
from .schema import CVASample, TaskFamily


class MalformedOutputError(ValueError):
    """Raised when a model output cannot be validly scored for its task family."""


_NUMERIC_FAMILIES = frozenset(
    {
        TaskFamily.DIGIT_OFFSET,
        TaskFamily.COUNT_TRANSFORM,
        TaskFamily.GAUGE_CALIBRATION,
        TaskFamily.BAR_CHART_AGGREGATE,
    }
)


def _validate_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_family_answer(answer: object, family: TaskFamily, *, source: str) -> object:
    if family in _NUMERIC_FAMILIES:
        if isinstance(answer, bool) or not isinstance(answer, Real):
            raise MalformedOutputError(
                f"{family.value} {source} answer must be a finite real number"
            )
        try:
            finite = math.isfinite(float(answer))
        except (OverflowError, ValueError) as error:
            raise MalformedOutputError(f"{family.value} {source} answer must be finite") from error
        if not finite:
            raise MalformedOutputError(f"{family.value} {source} answer must be finite")
        return answer
    if not isinstance(answer, str) or not answer.strip():
        raise MalformedOutputError(
            f"{TaskFamily.RELATION_RULE.value} {source} answer must be a non-empty string"
        )
    return answer


def _extract_answer(sample: CVASample, output: object) -> object:
    if not isinstance(output, ParseResult):
        return output
    if output.status is not ParseStatus.OK:
        detail = f": {output.error_code}" if output.error_code else ""
        raise MalformedOutputError(
            f"{sample.sample_id} structured output status is {output.status.value}{detail}"
        )
    if output.sample_id != sample.sample_id:
        raise MalformedOutputError(
            f"structured output sample_id {output.sample_id!r} does not match {sample.sample_id!r}"
        )
    return output.answer


def _is_correct(predicted: object, canonical: object, family: TaskFamily) -> bool:
    checked_prediction = _validate_family_answer(predicted, family, source="predicted")
    checked_canonical = _validate_family_answer(canonical, family, source="canonical")
    return bool(checked_prediction == checked_canonical)


@dataclass(frozen=True, slots=True)
class AnswerVerification:
    """One immutable binary outcome score with its canonical provenance."""

    sample_id: str
    task_family: TaskFamily
    predicted_answer: object
    canonical_answer: object
    is_correct: bool
    outcome_reward: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _validate_identifier(self.sample_id, "sample_id"))
        try:
            family = TaskFamily(self.task_family)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid task_family: {self.task_family!r}") from error
        object.__setattr__(self, "task_family", family)
        if not isinstance(self.is_correct, bool):
            raise TypeError("is_correct must be boolean")
        computed_correctness = _is_correct(
            self.predicted_answer,
            self.canonical_answer,
            family,
        )
        if self.is_correct is not computed_correctness:
            raise ValueError("is_correct must match the verified answer comparison")
        expected_reward = float(self.is_correct)
        if self.outcome_reward != expected_reward:
            raise ValueError("outcome_reward must equal the binary correctness indicator")
        object.__setattr__(self, "outcome_reward", expected_reward)

    @property
    def accuracy(self) -> float:
        """Return this example's binary accuracy contribution."""

        return self.outcome_reward


@dataclass(frozen=True, slots=True)
class VerificationBatch:
    """Ordered immutable results and aggregate outcome metrics for one batch."""

    results: tuple[AnswerVerification, ...]

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if not results:
            raise ValueError("verification batch must not be empty")
        if any(not isinstance(result, AnswerVerification) for result in results):
            raise TypeError("results must contain only AnswerVerification records")
        object.__setattr__(self, "results", results)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(result.is_correct for result in self.results)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def mean_outcome_reward(self) -> float:
        return sum(result.outcome_reward for result in self.results) / self.total


def verify_answer(sample: CVASample, output: object) -> AnswerVerification:
    """Verify one raw or successfully parsed answer against a solver-checked sample.

    Structured parse failures are raised as :class:`MalformedOutputError`; they
    are never silently converted to an incorrect-answer reward.
    """

    if not isinstance(sample, CVASample):
        raise TypeError("sample must be a CVASample")
    solved = solve_sample(sample)
    predicted = _extract_answer(sample, output)
    correct = _is_correct(predicted, solved.answer, sample.task_family)
    return AnswerVerification(
        sample_id=sample.sample_id,
        task_family=sample.task_family,
        predicted_answer=predicted,
        canonical_answer=solved.answer,
        is_correct=correct,
        outcome_reward=float(correct),
    )


def verify_many(
    samples: Iterable[CVASample],
    outputs: Iterable[object],
) -> VerificationBatch:
    """Verify aligned sample/output iterables and report binary batch metrics."""

    try:
        sample_records = tuple(samples)
        output_records = tuple(outputs)
    except TypeError as error:
        raise TypeError("samples and outputs must be iterable") from error
    if not sample_records:
        raise ValueError("verification batch must not be empty")
    if len(sample_records) != len(output_records):
        raise ValueError("samples and outputs must have the same length")
    if any(not isinstance(sample, CVASample) for sample in sample_records):
        raise TypeError("samples must contain only CVASample records")
    identifiers = tuple(sample.sample_id for sample in sample_records)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate sample_id values are not allowed")
    return VerificationBatch(
        results=tuple(
            verify_answer(sample, output)
            for sample, output in zip(sample_records, output_records, strict=True)
        )
    )


__all__ = [
    "AnswerVerification",
    "MalformedOutputError",
    "VerificationBatch",
    "verify_answer",
    "verify_many",
]
