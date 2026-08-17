"""Observation-anchor and constraint-assimilation estimands.

These helpers compute measurements; they never turn an empirical effect size into an execution
gate.  Only malformed, non-finite, or non-paired inputs are rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

AssimilationPattern = Literal[
    "no_assimilation",
    "transient_assimilation",
    "persistent_but_insufficient_assimilation",
    "successful_revision",
]


@dataclass(frozen=True, slots=True)
class CandidateLogProbabilities:
    """Teacher-forced true/observed candidate log probabilities for one cue condition."""

    condition: str
    logp_true: float
    logp_observed: float
    true_rank: int
    observed_rank: int

    def __post_init__(self) -> None:
        if self.condition not in {"no_cue", "valid_cue", "sham_cue", "counterfactual_cue"}:
            raise ValueError("condition is not registered")
        if not all(math.isfinite(value) for value in (self.logp_true, self.logp_observed)):
            raise ValueError("candidate log probabilities must be finite")
        if self.true_rank < 1 or self.observed_rank < 1:
            raise ValueError("candidate ranks are one-based positive integers")

    @property
    def margin(self) -> float:
        """True-minus-observed log-probability margin."""

        return self.logp_true - self.logp_observed


@dataclass(frozen=True, slots=True)
class ObservationAnchorMetrics:
    """The v4 facts-induced change and remaining observation-anchor margin."""

    no_cue_margin: float
    valid_cue_margin: float
    delta_f: float
    m_f: float


def observation_anchor_metrics(
    no_cue: CandidateLogProbabilities,
    valid_cue: CandidateLogProbabilities,
) -> ObservationAnchorMetrics:
    """Compute ``Delta_F`` and ``M_F`` from a paired scene.

    ``M_F`` is the valid-cue true-minus-observed margin. ``Delta_F`` is its paired change from
    no cue.  Positive ``Delta_F`` with negative ``M_F`` is therefore "facts helped but did not
    overcome the observation anchor"; this is an interpretation, not a pass threshold.
    """

    if no_cue.condition != "no_cue" or valid_cue.condition != "valid_cue":
        raise ValueError("observation-anchor metrics require a no-cue/valid-cue pair")
    no_cue_margin = no_cue.margin
    valid_cue_margin = valid_cue.margin
    return ObservationAnchorMetrics(
        no_cue_margin=no_cue_margin,
        valid_cue_margin=valid_cue_margin,
        delta_f=valid_cue_margin - no_cue_margin,
        m_f=valid_cue_margin,
    )


def validate_prompt_pair(
    no_cue_payload: dict[str, object],
    valid_cue_payload: dict[str, object],
) -> None:
    """Require paired scoring payloads to differ only in registered facts and condition."""

    expected_differences = {"condition", "facts"}
    if set(no_cue_payload) != set(valid_cue_payload):
        raise ValueError("paired payload fields differ")
    differences = {key for key in no_cue_payload if no_cue_payload[key] != valid_cue_payload[key]}
    if not differences or not differences.issubset(expected_differences):
        raise ValueError("no-cue and valid-cue payloads must differ only in facts/condition")
    if no_cue_payload.get("condition") != "no_cue":
        raise ValueError("first paired payload must be no_cue")
    if valid_cue_payload.get("condition") != "valid_cue":
        raise ValueError("second paired payload must be valid_cue")


def classify_assimilation_profile(
    no_cue_margins: tuple[float, ...],
    valid_cue_margins: tuple[float, ...],
    *,
    numerical_tolerance: float = 1e-8,
) -> AssimilationPattern:
    """Classify a layerwise profile using only a numerical equality tolerance."""

    if len(no_cue_margins) != len(valid_cue_margins) or not no_cue_margins:
        raise ValueError("paired non-empty layerwise margins are required")
    if numerical_tolerance < 0 or not math.isfinite(numerical_tolerance):
        raise ValueError("numerical_tolerance must be finite and non-negative")
    values = no_cue_margins + valid_cue_margins
    if not all(math.isfinite(value) for value in values):
        raise ValueError("layerwise margins must be finite")
    deltas = tuple(
        valid - no_cue for no_cue, valid in zip(no_cue_margins, valid_cue_margins, strict=True)
    )
    active = tuple(delta > numerical_tolerance for delta in deltas)
    if not any(active):
        return "no_assimilation"
    if not active[-1]:
        return "transient_assimilation"
    if valid_cue_margins[-1] >= -numerical_tolerance:
        return "successful_revision"
    return "persistent_but_insufficient_assimilation"


__all__ = [
    "AssimilationPattern",
    "CandidateLogProbabilities",
    "ObservationAnchorMetrics",
    "classify_assimilation_profile",
    "observation_anchor_metrics",
    "validate_prompt_pair",
]
