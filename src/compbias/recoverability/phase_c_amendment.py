"""Prospective Phase-C v2 amendment made after Phase N and before Phase C."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .phase_n_result import PhaseNFrozenResult


@dataclass(frozen=True, slots=True)
class PhaseCAmendment:
    schema_version: int
    status: str
    amendment_id: str
    amendment_date: str
    original_protocol_path: str
    original_protocol_sha256: str
    phase_n_result_path: str
    original_continuation_threshold: float
    amended_continuation_threshold: float
    observed_phase_n_primary_rate: float
    observed_phase_n_one_sided_cp_upper: float
    original_phase_n_gate_passed: bool
    original_phase_n_inconclusive: bool
    phase_c_outcomes_observed: bool
    confirmatory_phase_c_authorized: bool
    amendment_reason: str
    dataset_id: str
    output_subdirectory: str
    seed: int
    intake_scenes: int
    selected_family_quotas: tuple[tuple[str, int], ...]
    arms: tuple[str, ...]
    confirmatory_arms: tuple[str, ...]
    diagnostic_arms: tuple[str, ...]
    forks_per_arm: int
    format_retries: int
    allow_quota_redistribution: bool
    allow_sample_extension: bool
    training_authorized: bool
    rl_authorized: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_phase_c_amendment(path: Path, *, phase_n: PhaseNFrozenResult) -> PhaseCAmendment:
    """Load the closed amendment and bind it to the unchanged Phase-N result."""

    if not isinstance(phase_n, PhaseNFrozenResult):
        raise TypeError("phase_n must be a PhaseNFrozenResult")
    mapping = load_yaml_mapping(path, label="Phase C v2 amendment")
    fields = set(PhaseCAmendment.__dataclass_fields__)
    reject_unknown_fields(mapping, fields, label="Phase C v2 amendment")
    if set(mapping) != fields:
        raise ValueError("Phase C v2 amendment is incomplete")
    quotas = mapping["selected_family_quotas"]
    if not isinstance(quotas, dict):
        raise ValueError("Phase C family quotas are invalid")
    candidate = PhaseCAmendment(
        **{
            **mapping,
            "selected_family_quotas": tuple(sorted(quotas.items())),
            "arms": tuple(mapping["arms"]),
            "confirmatory_arms": tuple(mapping["confirmatory_arms"]),
            "diagnostic_arms": tuple(mapping["diagnostic_arms"]),
        }
    )
    expected_quotas = (("cross_series", 400), ("duplicate_encoding", 266), ("trend", 400))
    expected_arms = (
        "ablated",
        "valid",
        "sham",
        "counterfactual",
        "oracle_perception",
        "operator_swap",
    )
    canonical = {
        "schema_version": 2,
        "status": "AMENDED_AFTER_PHASE_N_BEFORE_PHASE_C",
        "amendment_id": "recoverability-phase-c-v2-20260815",
        "amendment_date": "2026-08-15",
        "original_protocol_path": "configs/recoverability/recoverability_v1.yaml",
        "original_protocol_sha256": (
            "6b7c8bf8df850cb026a6494c7c41530cc7a83d65add3ebaac0cbca857f0fb40c"
        ),
        "phase_n_result_path": "configs/recoverability/phase_n_frozen_result.yaml",
        "original_continuation_threshold": 0.05,
        "amended_continuation_threshold": 0.10,
        "observed_phase_n_primary_rate": phase_n.primary_rate,
        "observed_phase_n_one_sided_cp_upper": phase_n.one_sided_cp_upper,
        "original_phase_n_gate_passed": False,
        "original_phase_n_inconclusive": True,
        "phase_c_outcomes_observed": False,
        "confirmatory_phase_c_authorized": True,
        "amendment_reason": (
            "original_five_percent_continuation_boundary_was_judged_overly_subjective_and_too_strict"
        ),
        "dataset_id": "CVA-Recoverability-Causal-v2",
        "output_subdirectory": "cva_recoverability_causal_v2",
        "seed": 2026081801,
        "intake_scenes": 8000,
        "selected_family_quotas": expected_quotas,
        "arms": expected_arms,
        "confirmatory_arms": expected_arms[:4],
        "diagnostic_arms": expected_arms[4:],
        "forks_per_arm": 8,
        "format_retries": 0,
        "allow_quota_redistribution": False,
        "allow_sample_extension": False,
        "training_authorized": False,
        "rl_authorized": False,
    }
    if any(getattr(candidate, key) != value for key, value in canonical.items()):
        raise ValueError("Phase C v2 amendment differs from the canonical amendment")
    root = path.resolve().parents[2]
    original = root / candidate.original_protocol_path
    frozen = root / candidate.phase_n_result_path
    if (
        original.is_symlink()
        or not original.is_file()
        or _sha256(original) != candidate.original_protocol_sha256
    ):
        raise ValueError("original protocol differs from the amendment anchor")
    if frozen.resolve() == path.resolve() or frozen.is_symlink() or not frozen.is_file():
        raise ValueError("Phase N frozen result path is invalid")
    if not (
        phase_n.one_sided_cp_upper < candidate.amended_continuation_threshold
        and phase_n.one_sided_cp_upper >= candidate.original_continuation_threshold
        and phase_n.h1_supported is False
        and phase_n.inconclusive is True
    ):
        raise ValueError("Phase N result does not support the recorded amendment decision")
    return candidate
