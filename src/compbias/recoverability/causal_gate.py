"""Closed causal authorization gate for any future recoverability RL study."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_FAMILY_QUOTAS = {"cross_series": 400, "trend": 400}
_POWER_TARGET_SCENES = 1066


@dataclass(frozen=True, slots=True)
class ConfirmatoryFamilyEvidence:
    family: str
    eligible_scenes: int
    recoverable_effect_ci_low: float
    holm_rejects_null: bool

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or _IDENTIFIER.fullmatch(self.family) is None:
            raise ValueError("family must be a bounded safe identifier")
        if type(self.eligible_scenes) is not int or self.eligible_scenes < 0:
            raise ValueError("eligible_scenes must be a non-negative integer")
        if (
            isinstance(self.recoverable_effect_ci_low, bool)
            or not isinstance(self.recoverable_effect_ci_low, (int, float))
            or not math.isfinite(float(self.recoverable_effect_ci_low))
        ):
            raise ValueError("recoverable_effect_ci_low must be finite")
        if type(self.holm_rejects_null) is not bool:
            raise TypeError("holm_rejects_null must be boolean")


@dataclass(frozen=True, slots=True)
class CausalGateEvidence:
    parser_rate_lower: float
    program_answer_consistency_lower: float
    eligible_scenes: int
    power_target_scenes: int
    confirmatory_families: tuple[ConfirmatoryFamilyEvidence, ...]
    recoverability_interaction_ci_low: float
    nonrecoverable_equivalence_passed: bool
    sham_equivalence_passed: bool
    operator_invariant_equivalence_passed: bool
    counterfactual_target_ci_low: float
    counterfactual_original_ci_low: float
    counterfactual_control_passed: bool
    complete_matched_sets: bool
    all_traces_included: bool

    def __post_init__(self) -> None:
        numeric = (
            self.parser_rate_lower,
            self.program_answer_consistency_lower,
            self.recoverability_interaction_ci_low,
            self.counterfactual_target_ci_low,
            self.counterfactual_original_ci_low,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("causal gate numeric evidence must be finite")
        if not isinstance(self.confirmatory_families, tuple) or any(
            not isinstance(item, ConfirmatoryFamilyEvidence) for item in self.confirmatory_families
        ):
            raise TypeError("confirmatory_families must contain typed evidence")
        for value, label in (
            (self.eligible_scenes, "eligible_scenes"),
            (self.power_target_scenes, "power_target_scenes"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if self.power_target_scenes != _POWER_TARGET_SCENES:
            raise ValueError("power_target_scenes must equal the preregistered 1066")
        names = [item.family for item in self.confirmatory_families]
        if len(set(names)) != len(names):
            raise ValueError("confirmatory family names must be unique")
        boolean_fields = (
            self.nonrecoverable_equivalence_passed,
            self.sham_equivalence_passed,
            self.operator_invariant_equivalence_passed,
            self.counterfactual_control_passed,
            self.complete_matched_sets,
            self.all_traces_included,
        )
        if any(type(value) is not bool for value in boolean_fields):
            raise TypeError("causal gate clause indicators must be boolean")


@dataclass(frozen=True, slots=True)
class CausalGateResult:
    gate_passed: bool
    rl_authorized: bool
    reason_codes: tuple[str, ...]


def evaluate_causal_gate(evidence: CausalGateEvidence) -> CausalGateResult:
    if not isinstance(evidence, CausalGateEvidence):
        raise TypeError("evidence must be CausalGateEvidence")
    reasons: list[str] = []
    if evidence.parser_rate_lower < 0.98:
        reasons.append("grammar_parse_lower_below_threshold")
    if evidence.program_answer_consistency_lower < 0.95:
        reasons.append("program_consistency_lower_below_threshold")
    if evidence.eligible_scenes < evidence.power_target_scenes:
        reasons.append("eligible_scenes_below_power_target")
    family_map = {item.family: item for item in evidence.confirmatory_families}
    if set(family_map) != set(_FAMILY_QUOTAS):
        reasons.append("confirmatory_family_below_power_target")
    else:
        if any(
            family_map[family].eligible_scenes < quota for family, quota in _FAMILY_QUOTAS.items()
        ):
            reasons.append("confirmatory_family_below_power_target")
        if any(
            item.recoverable_effect_ci_low <= 0 or not item.holm_rejects_null
            for item in family_map.values()
        ):
            reasons.append("recoverable_effect_lower_not_positive")
    if evidence.recoverability_interaction_ci_low <= 0:
        reasons.append("recoverability_interaction_lower_not_positive")
    if not evidence.nonrecoverable_equivalence_passed:
        reasons.append("nonrecoverable_equivalence_failed")
    if not evidence.sham_equivalence_passed:
        reasons.append("sham_equivalence_failed")
    if not evidence.operator_invariant_equivalence_passed:
        reasons.append("operator_invariant_equivalence_failed")
    if evidence.counterfactual_target_ci_low <= 0:
        reasons.append("counterfactual_target_shift_lower_not_positive")
    if evidence.counterfactual_original_ci_low <= 0:
        reasons.append("counterfactual_original_suppression_lower_not_positive")
    if not evidence.counterfactual_control_passed:
        reasons.append("counterfactual_control_gate_failed")
    if not evidence.complete_matched_sets:
        reasons.append("incomplete_matched_sets")
    if not evidence.all_traces_included:
        reasons.append("correct_only_selection_detected")
    passed = not reasons
    return CausalGateResult(
        gate_passed=passed,
        rl_authorized=passed,
        reason_codes=tuple(reasons),
    )
