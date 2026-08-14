"""Exact Phase-N prevalence inference with fail-closed support accounting."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class NaturalPrevalenceObservation:
    scene_id: str
    parse_success: bool
    operator_sensitive_error: bool | None
    strict_repair_candidate: bool | None

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or _IDENTIFIER.fullmatch(self.scene_id) is None:
            raise ValueError("scene_id must be a bounded safe identifier")
        if type(self.parse_success) is not bool:
            raise TypeError("parse_success must be boolean")
        if not self.parse_success:
            if (
                self.operator_sensitive_error is not None
                or self.strict_repair_candidate is not None
            ):
                raise ValueError("parse failure labels must remain None")
            return
        if type(self.operator_sensitive_error) is not bool:
            raise TypeError("parsed operator_sensitive_error must be boolean")
        if type(self.strict_repair_candidate) is not bool:
            raise TypeError("parsed strict_repair_candidate must be boolean")
        if self.strict_repair_candidate and not self.operator_sensitive_error:
            raise ValueError("strict repair candidate requires an operator-sensitive error")


def _count(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


@dataclass(frozen=True, slots=True)
class NaturalPrevalenceCounts:
    total_scenes: int
    parsed_scenes: int
    operator_sensitive_errors: int
    strict_natural_repair_candidates: int

    def __post_init__(self) -> None:
        total = _count(self.total_scenes, "total_scenes", positive=True)
        parsed = _count(self.parsed_scenes, "parsed_scenes")
        eligible = _count(self.operator_sensitive_errors, "operator_sensitive_errors")
        candidates = _count(
            self.strict_natural_repair_candidates,
            "strict_natural_repair_candidates",
        )
        if not candidates <= eligible <= parsed <= total:
            raise ValueError(
                "prevalence counts must satisfy candidates <= eligible <= parsed <= total"
            )


@dataclass(frozen=True, slots=True)
class NaturalPrevalenceSummary:
    primary_denominator: str
    primary_rate: float
    parsed_prevalence: float
    all_attempt_prevalence: float
    parse_rate: float
    one_sided_cp_upper: float
    parse_failure_sensitivity_lower: float
    parse_failure_sensitivity_upper: float
    h1_supported: bool
    inconclusive: bool
    reason_code: str | None


def _probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 < result < 1:
        raise ValueError(f"{label} must lie strictly between zero and one")
    return result


def _clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    if trials < 1:
        return 1.0
    if successes == trials:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))


def summarize_natural_prevalence(
    counts: NaturalPrevalenceCounts,
    *,
    null_rate: float,
    alpha: float,
    minimum_eligible: int,
) -> NaturalPrevalenceSummary:
    """Test H1 on its registered denominator and expose missing-parse sensitivity."""

    if not isinstance(counts, NaturalPrevalenceCounts):
        raise TypeError("counts must be NaturalPrevalenceCounts")
    threshold = _probability(null_rate, "null_rate")
    alpha_value = _probability(alpha, "alpha")
    minimum = _count(minimum_eligible, "minimum_eligible", positive=True)
    total = counts.total_scenes
    parsed = counts.parsed_scenes
    eligible = counts.operator_sensitive_errors
    candidates = counts.strict_natural_repair_candidates
    parse_failures = total - parsed
    sensitivity_denominator = eligible + parse_failures
    if sensitivity_denominator:
        sensitivity_lower = candidates / sensitivity_denominator
        sensitivity_upper = (candidates + parse_failures) / sensitivity_denominator
    else:
        sensitivity_lower = 0.0
        sensitivity_upper = 1.0
    upper = _clopper_pearson_upper(candidates, eligible, alpha_value)
    enough_support = eligible >= minimum
    below_null = upper < threshold
    h1_supported = enough_support and below_null
    if not enough_support:
        reason = "phase_n_eligible_below_preregistered_minimum"
    elif not below_null:
        reason = "phase_n_h1_upper_not_below_threshold"
    else:
        reason = None
    return NaturalPrevalenceSummary(
        primary_denominator="parsed_operator_sensitive_errors",
        primary_rate=candidates / eligible if eligible else 0.0,
        parsed_prevalence=candidates / parsed if parsed else 0.0,
        all_attempt_prevalence=candidates / total,
        parse_rate=parsed / total,
        one_sided_cp_upper=upper,
        parse_failure_sensitivity_lower=sensitivity_lower,
        parse_failure_sensitivity_upper=sensitivity_upper,
        h1_supported=h1_supported,
        inconclusive=not h1_supported,
        reason_code=reason,
    )


def analyze_natural_prevalence(
    observations: tuple[NaturalPrevalenceObservation, ...],
    *,
    null_rate: float,
    alpha: float,
    minimum_eligible: int,
) -> NaturalPrevalenceSummary:
    """Validate raw independent scenes before reducing them to registered counts."""

    if not isinstance(observations, tuple) or not observations:
        raise ValueError("observations must be a non-empty tuple")
    if any(not isinstance(item, NaturalPrevalenceObservation) for item in observations):
        raise TypeError("observations must contain NaturalPrevalenceObservation instances")
    identifiers = [item.scene_id for item in observations]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("scene identifiers must be unique")
    parsed = sum(item.parse_success for item in observations)
    eligible = sum(item.operator_sensitive_error is True for item in observations)
    candidates = sum(item.strict_repair_candidate is True for item in observations)
    return summarize_natural_prevalence(
        NaturalPrevalenceCounts(len(observations), parsed, eligible, candidates),
        null_rate=null_rate,
        alpha=alpha,
        minimum_eligible=minimum_eligible,
    )
