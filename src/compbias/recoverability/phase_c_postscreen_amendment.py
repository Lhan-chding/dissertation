"""Prospective post-screen Phase-C amendment made before any arm outcome."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .phase_c_screen_result import PhaseCScreenFrozenResult
from .power import PowerSimulationConfig, simulate_paired_power


@dataclass(frozen=True, slots=True)
class PhaseCPostscreenAmendment:
    schema_version: int
    status: str
    amendment_id: str
    amendment_date: str
    screen_result_path: str
    original_screen_passed: bool
    original_screen_exit: int
    arm_outcomes_observed: bool
    fixed_family_quota_gate_withdrawn: bool
    amendment_reason: str
    frozen_eligible_scenes: int
    frozen_eligible_by_family: tuple[tuple[str, int], ...]
    include_all_frozen_eligible_scenes: bool
    original_fixed_family_quotas: tuple[tuple[str, int], ...]
    original_target_power: float
    original_target_power_met: bool
    target_effect: float
    holm_adjusted_one_sided_alpha: float
    power_simulation_repetitions: int
    power_simulation_seed: int
    amended_power_by_confirmatory_family: tuple[tuple[str, float], ...]
    dataset_id: str
    output_subdirectory: str
    seed: int
    arms: tuple[str, ...]
    confirmatory_arms: tuple[str, ...]
    diagnostic_arms: tuple[str, ...]
    forks_per_arm: int
    model_call_cap: int
    format_retries: int
    allow_screen_rerun: bool
    allow_sample_extension: bool
    allow_quota_redistribution: bool
    allow_eligible_scene_exclusion: bool
    confirmatory_arm_execution_authorized: bool
    training_authorized: bool
    rl_authorized: bool


def _power(scenes: int, seed: int) -> float:
    return simulate_paired_power(
        PowerSimulationConfig(
            scenes=scenes,
            forks_per_arm=8,
            baseline_rate=0.20,
            target_effect=0.05,
            discordance=0.30,
            scene_icc=0.40,
            alpha=0.025,
            repetitions=2000,
            seed=seed,
        )
    ).estimated_power


def load_phase_c_postscreen_amendment(
    path: Path, *, screen: PhaseCScreenFrozenResult
) -> PhaseCPostscreenAmendment:
    if not isinstance(screen, PhaseCScreenFrozenResult):
        raise TypeError("screen must be a PhaseCScreenFrozenResult")
    mapping = load_yaml_mapping(path, label="Phase C v3 post-screen amendment")
    fields = set(PhaseCPostscreenAmendment.__dataclass_fields__)
    reject_unknown_fields(mapping, fields, label="Phase C v3 post-screen amendment")
    if set(mapping) != fields:
        raise ValueError("Phase C v3 post-screen amendment is incomplete")
    converted = {
        **mapping,
        "frozen_eligible_by_family": tuple(sorted(mapping["frozen_eligible_by_family"].items())),
        "original_fixed_family_quotas": tuple(
            sorted(mapping["original_fixed_family_quotas"].items())
        ),
        "amended_power_by_confirmatory_family": tuple(
            sorted(mapping["amended_power_by_confirmatory_family"].items())
        ),
        "arms": tuple(mapping["arms"]),
        "confirmatory_arms": tuple(mapping["confirmatory_arms"]),
        "diagnostic_arms": tuple(mapping["diagnostic_arms"]),
    }
    candidate = PhaseCPostscreenAmendment(**converted)
    expected_arms = (
        "ablated",
        "valid",
        "sham",
        "counterfactual",
        "oracle_perception",
        "operator_swap",
    )
    canonical = {
        "schema_version": 3,
        "status": "AMENDED_AFTER_SCREEN_BEFORE_ARM_OUTCOMES",
        "amendment_id": "recoverability-phase-c-v3-20260816",
        "amendment_date": "2026-08-16",
        "screen_result_path": "configs/recoverability/phase_c_screen_v2_frozen_result.yaml",
        "original_screen_passed": False,
        "original_screen_exit": 3,
        "arm_outcomes_observed": False,
        "fixed_family_quota_gate_withdrawn": True,
        "amendment_reason": (
            "fixed_family_quotas_were_internal_power_planning_targets_not_field_consensus_validity_criteria"
        ),
        "frozen_eligible_scenes": 580,
        "frozen_eligible_by_family": (
            ("cross_series", 208),
            ("duplicate_encoding", 182),
            ("trend", 190),
        ),
        "include_all_frozen_eligible_scenes": True,
        "original_fixed_family_quotas": (
            ("cross_series", 400),
            ("duplicate_encoding", 266),
            ("trend", 400),
        ),
        "original_target_power": 0.90,
        "original_target_power_met": False,
        "target_effect": 0.05,
        "holm_adjusted_one_sided_alpha": 0.025,
        "power_simulation_repetitions": 2000,
        "power_simulation_seed": 2026081901,
        "amended_power_by_confirmatory_family": (
            ("cross_series", 0.7375),
            ("trend", 0.6865),
        ),
        "dataset_id": "CVA-Recoverability-Causal-v3",
        "output_subdirectory": "cva_recoverability_causal_v3",
        "seed": 2026081901,
        "arms": expected_arms,
        "confirmatory_arms": expected_arms[:4],
        "diagnostic_arms": expected_arms[4:],
        "forks_per_arm": 8,
        "model_call_cap": 27840,
        "format_retries": 0,
        "allow_screen_rerun": False,
        "allow_sample_extension": False,
        "allow_quota_redistribution": False,
        "allow_eligible_scene_exclusion": False,
        "confirmatory_arm_execution_authorized": True,
        "training_authorized": False,
        "rl_authorized": False,
    }
    if any(getattr(candidate, key) != value for key, value in canonical.items()):
        raise ValueError("Phase C v3 post-screen amendment differs from the frozen decision")
    if (
        screen.screen_passed
        or screen.phase_c_screen_exit != 3
        or screen.eligible_scenes != candidate.frozen_eligible_scenes
        or screen.eligible_by_family != candidate.frozen_eligible_by_family
    ):
        raise ValueError("Phase C screen evidence does not support the post-screen amendment")
    observed_power = dict(candidate.amended_power_by_confirmatory_family)
    if _power(208, candidate.power_simulation_seed) != observed_power["cross_series"]:
        raise ValueError("cross-series power sensitivity is not reproducible")
    if _power(190, candidate.power_simulation_seed + 1) != observed_power["trend"]:
        raise ValueError("trend power sensitivity is not reproducible")
    return candidate
