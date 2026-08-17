"""Conservative source labels for answer-only successes."""

from __future__ import annotations

from enum import Enum


class AnswerSource(str, Enum):
    GENUINE_RECOVERY = "genuine_recovery"
    OPERATOR_INVARIANCE = "operator_invariance"
    ERROR_CANCELLATION = "error_cancellation"
    VISUAL_REREAD = "visual_reread"
    GUESS_OR_ANSWER_PRIOR = "guess_or_answer_prior"
    UNRESOLVED = "unresolved"


def classify_answer_source(
    *,
    answer_correct: bool,
    world_recovered: bool,
    operator_invariant: bool = False,
    error_cancelled: bool = False,
    visual_reread_evidence: bool = False,
    guess_or_prior_evidence: bool = False,
) -> AnswerSource:
    values = (
        answer_correct,
        world_recovered,
        operator_invariant,
        error_cancelled,
        visual_reread_evidence,
        guess_or_prior_evidence,
    )
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("answer-source evidence flags must be boolean")
    if answer_correct and world_recovered:
        return AnswerSource.GENUINE_RECOVERY
    candidates = tuple(
        source
        for present, source in (
            (operator_invariant, AnswerSource.OPERATOR_INVARIANCE),
            (error_cancelled, AnswerSource.ERROR_CANCELLATION),
            (visual_reread_evidence, AnswerSource.VISUAL_REREAD),
            (guess_or_prior_evidence, AnswerSource.GUESS_OR_ANSWER_PRIOR),
        )
        if answer_correct and present
    )
    if len(candidates) == 1:
        return candidates[0]
    return AnswerSource.UNRESOLVED
