from __future__ import annotations

from dataclasses import replace

import pytest

from compbias.recoverability.causal_gate import (
    CausalGateEvidence,
    ConfirmatoryFamilyEvidence,
    evaluate_causal_gate,
)
from compbias.recoverability.counterfactual_metrics import (
    CounterfactualForkPair,
    summarize_counterfactual_consistency,
)
from compbias.recoverability.natural_inference import (
    NaturalPrevalenceCounts,
    summarize_natural_prevalence,
)
from compbias.recoverability.paired_effects import (
    ArmForks,
    SceneCrossover,
    interval_is_equivalent,
    paired_scene_effect,
)
from compbias.recoverability.power import (
    PowerSimulationConfig,
    simulate_paired_power,
)


def test_phase_n_primary_prevalence_uses_only_parsed_operator_sensitive_errors() -> None:
    summary = summarize_natural_prevalence(
        NaturalPrevalenceCounts(
            total_scenes=4000,
            parsed_scenes=3800,
            operator_sensitive_errors=800,
            strict_natural_repair_candidates=10,
        ),
        null_rate=0.05,
        alpha=0.05,
        minimum_eligible=800,
    )

    assert summary.primary_denominator == "parsed_operator_sensitive_errors"
    assert summary.primary_rate == 10 / 800
    assert summary.parsed_prevalence == 10 / 3800
    assert summary.all_attempt_prevalence == 10 / 4000
    assert summary.one_sided_cp_upper < 0.05
    assert summary.h1_supported is True
    assert summary.inconclusive is False


def test_phase_n_is_inconclusive_when_support_is_below_the_frozen_minimum() -> None:
    summary = summarize_natural_prevalence(
        NaturalPrevalenceCounts(4000, 3800, 799, 0),
        null_rate=0.05,
        alpha=0.05,
        minimum_eligible=800,
    )

    assert summary.one_sided_cp_upper < 0.05
    assert summary.h1_supported is False
    assert summary.inconclusive is True
    assert summary.reason_code == "insufficient_operator_sensitive_support"


def test_phase_n_reports_fail_closed_parse_failure_sensitivity_bounds() -> None:
    summary = summarize_natural_prevalence(
        NaturalPrevalenceCounts(4000, 3800, 800, 10),
        null_rate=0.05,
        alpha=0.05,
        minimum_eligible=800,
    )

    assert summary.parse_rate == 0.95
    assert summary.parse_failure_sensitivity_lower == 10 / 1000
    assert summary.parse_failure_sensitivity_upper == 210 / 1000


def test_counterfactual_metrics_require_both_worlds_not_direction_alone() -> None:
    pairs = (
        CounterfactualForkPair("scene_1", "fork_1", 4, 5, 4, 5, True, True),
        CounterfactualForkPair("scene_2", "fork_1", 3, 4, 4, 5, False, False),
    )

    summary = summarize_counterfactual_consistency(pairs)

    assert summary.target_shift_effect == 0.5
    assert summary.original_retention_effect == 0.0
    assert summary.strict_dual_world_faithful_success == 0.5
    assert summary.answer_direction_compliance == 1.0
    assert summary.counterfactual_consistency == 0.5


def test_counterfactual_metrics_reject_same_gold_or_duplicate_pair_keys() -> None:
    same_gold = CounterfactualForkPair("scene_1", "fork_1", 4, 4, 4, 4, True, True)
    with pytest.raises(ValueError, match="must differ"):
        summarize_counterfactual_consistency((same_gold,))

    pair = CounterfactualForkPair("scene_1", "fork_1", 4, 5, 4, 5, True, True)
    with pytest.raises(ValueError, match="unique"):
        summarize_counterfactual_consistency((pair, pair))


def _scene(
    scene_id: str,
    family: str,
    *,
    valid: tuple[bool, ...],
    ablated: tuple[bool, ...],
    sham: tuple[bool, ...] | None = None,
) -> SceneCrossover:
    return SceneCrossover(
        scene_id=scene_id,
        family=family,
        stratum="operator_sensitive_recoverable",
        arms=(
            ArmForks("ablated", ablated),
            ArmForks("valid", valid),
            ArmForks("sham", sham if sham is not None else ablated),
        ),
        forks_per_arm=8,
    )


def test_paired_effect_resamples_scenes_and_standardizes_equally_by_family() -> None:
    scenes = tuple(
        _scene(
            f"{family}_{index}",
            family,
            valid=(True,) * 8,
            ablated=(False,) * 8,
        )
        for family in ("cross_series", "trend")
        for index in range(4)
    )

    effect = paired_scene_effect(
        scenes,
        treatment_arm="valid",
        control_arm="ablated",
        confidence=0.95,
        bootstrap_resamples=500,
        seed=2026081604,
    )

    assert effect.estimate == 1.0
    assert effect.ci_low == effect.ci_high == 1.0
    assert effect.n_independent_scenes == 8
    assert effect.n_forks_observed == 8 * 3 * 8
    assert effect.resampling_unit == "semantic_scene_within_family"
    assert effect.family_weighting == "equal_preregistered_family_weight"


def test_paired_effect_rejects_unbalanced_or_duplicate_scene_records() -> None:
    balanced = _scene(
        "scene_1", "cross_series", valid=(True,) * 8, ablated=(False,) * 8
    )
    unbalanced = replace(
        balanced,
        scene_id="scene_2",
        arms=(
            ArmForks("ablated", (False,) * 8),
            ArmForks("valid", (True,) * 7),
            ArmForks("sham", (False,) * 8),
        ),
    )
    with pytest.raises(ValueError, match="forks"):
        paired_scene_effect(
            (balanced, unbalanced),
            treatment_arm="valid",
            control_arm="ablated",
            bootstrap_resamples=100,
            seed=1,
        )
    with pytest.raises(ValueError, match="unique"):
        paired_scene_effect(
            (balanced, balanced),
            treatment_arm="valid",
            control_arm="ablated",
            bootstrap_resamples=100,
            seed=1,
        )


def test_tost_equivalence_uses_the_full_90_percent_interval() -> None:
    assert interval_is_equivalent(ci_low=-0.01, ci_high=0.01, margin=0.02) is True
    assert interval_is_equivalent(ci_low=-0.01, ci_high=0.021, margin=0.02) is False


def _passing_gate() -> CausalGateEvidence:
    return CausalGateEvidence(
        parser_rate_lower=0.985,
        program_answer_consistency_lower=0.96,
        confirmatory_families=(
            ConfirmatoryFamilyEvidence("cross_series", 267, 0.02, True),
            ConfirmatoryFamilyEvidence("trend", 267, 0.01, True),
        ),
        recoverability_interaction_ci_low=0.015,
        nonrecoverable_equivalence_passed=True,
        sham_equivalence_passed=True,
        operator_invariant_equivalence_passed=True,
        counterfactual_target_ci_low=0.01,
        counterfactual_original_ci_low=0.01,
        counterfactual_control_passed=True,
        complete_matched_sets=True,
        all_traces_included=True,
    )


def test_causal_gate_passing_fixture_authorizes_only_future_rl() -> None:
    result = evaluate_causal_gate(_passing_gate())

    assert result.gate_passed is True
    assert result.rl_authorized is True
    assert result.reason_codes == ()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("parser_rate_lower", 0.979, "parser_rate_lower_below_98_percent"),
        (
            "program_answer_consistency_lower",
            0.949,
            "program_answer_consistency_lower_below_95_percent",
        ),
        ("recoverability_interaction_ci_low", 0.0, "recoverability_interaction_not_positive"),
        (
            "nonrecoverable_equivalence_passed",
            False,
            "nonrecoverable_effect_not_equivalent",
        ),
        ("sham_equivalence_passed", False, "sham_effect_not_equivalent"),
        (
            "operator_invariant_equivalence_passed",
            False,
            "operator_invariant_effect_not_equivalent",
        ),
        ("counterfactual_target_ci_low", 0.0, "counterfactual_target_effect_not_positive"),
        (
            "counterfactual_original_ci_low",
            0.0,
            "counterfactual_original_effect_not_positive",
        ),
        ("counterfactual_control_passed", False, "counterfactual_control_failed"),
        ("complete_matched_sets", False, "incomplete_matched_sets"),
        ("all_traces_included", False, "correct_only_selection_detected"),
    ],
)
def test_each_causal_gate_clause_fails_closed_with_stable_reason(
    field: str, value: object, reason: str
) -> None:
    result = evaluate_causal_gate(replace(_passing_gate(), **{field: value}))

    assert result.gate_passed is False
    assert result.rl_authorized is False
    assert reason in result.reason_codes


def test_causal_gate_requires_both_preregistered_families_and_holm_support() -> None:
    one_family = replace(
        _passing_gate(),
        confirmatory_families=(
            ConfirmatoryFamilyEvidence("cross_series", 267, 0.02, True),
        ),
    )
    weak_family = replace(
        _passing_gate(),
        confirmatory_families=(
            ConfirmatoryFamilyEvidence("cross_series", 267, 0.02, True),
            ConfirmatoryFamilyEvidence("trend", 266, 0.0, False),
        ),
    )

    assert evaluate_causal_gate(one_family).reason_codes == (
        "confirmatory_family_set_incomplete",
    )
    result = evaluate_causal_gate(weak_family)
    assert "confirmatory_family_support_below_quota" in result.reason_codes
    assert "confirmatory_family_effect_not_positive" in result.reason_codes
    assert "confirmatory_family_holm_test_failed" in result.reason_codes


def test_power_simulation_is_seeded_and_treats_forks_as_nested_repeats() -> None:
    config = PowerSimulationConfig(
        scenes=300,
        forks_per_arm=8,
        baseline_rate=0.20,
        target_effect=0.05,
        discordance=0.20,
        scene_icc=0.25,
        alpha=0.05,
        repetitions=200,
        seed=2026081605,
    )

    first = simulate_paired_power(config)
    second = simulate_paired_power(config)

    assert first == second
    assert first.independent_unit == "semantic_scene"
    assert first.scenes == 300
    assert first.forks_per_arm == 8
    assert 0 <= first.estimated_power <= 1


def test_power_simulation_common_random_numbers_are_monotone_for_large_effect_change() -> None:
    low = PowerSimulationConfig(300, 8, 0.20, 0.01, 0.20, 0.25, 0.05, 300, 9)
    high = replace(low, target_effect=0.10)

    assert simulate_paired_power(high).estimated_power >= simulate_paired_power(
        low
    ).estimated_power


@pytest.mark.parametrize(
    "updates",
    [
        {"scenes": True},
        {"forks_per_arm": 0},
        {"baseline_rate": float("nan")},
        {"target_effect": -0.01},
        {"discordance": 0.01},
        {"scene_icc": 1.0},
        {"alpha": 0.0},
        {"repetitions": 0},
    ],
)
def test_power_configuration_rejects_invalid_or_adaptive_inputs(
    updates: dict[str, object],
) -> None:
    base = PowerSimulationConfig(300, 8, 0.20, 0.05, 0.20, 0.25, 0.05, 100, 9)
    with pytest.raises((TypeError, ValueError)):
        replace(base, **updates)
