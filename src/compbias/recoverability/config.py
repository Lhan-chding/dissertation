"""Closed preregistration configuration for Recoverability v1."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_RELATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _safe(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a bounded safe identifier")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _false(value: object, label: str) -> bool:
    if value is not False:
        raise ValueError(f"{label} must be false; adaptive sample extension is forbidden")
    return False


def _probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not 0 < result < 1:
        raise ValueError(f"{label} must lie in (0, 1)")
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


@dataclass(frozen=True, slots=True)
class PhaseNConfig:
    dataset_id: str
    output_subdirectory: str
    seed: int
    scenes: int
    source_protocol: str
    max_format_retries: int
    allow_sample_extension: bool


@dataclass(frozen=True, slots=True)
class BridgeConfig:
    dataset_id: str
    output_subdirectory: str
    seed: int
    scenes: int
    protocols_per_scene: int


@dataclass(frozen=True, slots=True)
class PhaseCConfig:
    dataset_id: str
    output_subdirectory: str
    seed: int
    intake_scenes: int
    selected_family_quotas: tuple[tuple[str, int], ...]
    arms: tuple[str, ...]
    confirmatory_arms: tuple[str, ...]
    diagnostic_arms: tuple[str, ...]
    forks_per_arm: int
    allow_quota_redistribution: bool
    allow_sample_extension: bool


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    alpha: float
    confidence: float
    tost_confidence: float
    target_power: float
    target_effect: float
    equivalence_margin: float
    bootstrap_resamples: int
    bootstrap_seed: int
    power_repetitions: int
    power_seed: int
    phase_n_minimum_eligible: int
    phase_n_null_rate: float
    required_eligible_scenes: int
    confirmatory_families: tuple[str, ...]
    exploratory_families: tuple[str, ...]
    power_artifact_path: str
    power_artifact_sha256: str


@dataclass(frozen=True, slots=True)
class RecoverabilityProtocol:
    schema_version: int
    status: Literal["PREREGISTERED_NOT_RUN"]
    model_id: str
    phase_n: PhaseNConfig
    bridge: BridgeConfig
    phase_c: PhaseCConfig
    analysis: AnalysisConfig


def _phase_n(value: object) -> PhaseNConfig:
    mapping = _mapping(value, "phase_n")
    fields = {
        "dataset_id",
        "output_subdirectory",
        "seed",
        "scenes",
        "source_protocol",
        "max_format_retries",
        "allow_sample_extension",
    }
    reject_unknown_fields(mapping, fields, label="phase_n")
    if set(mapping) != fields:
        raise ValueError("phase_n must contain every registered field")
    retries = mapping["max_format_retries"]
    if retries != 0 or type(retries) is not int:
        raise ValueError("phase_n.max_format_retries must remain zero")
    scenes = _positive_int(mapping["scenes"], "phase_n.scenes")
    if scenes != 4000:
        raise ValueError("phase_n.scenes must equal the preregistered 4000")
    return PhaseNConfig(
        dataset_id=_safe(mapping["dataset_id"], "phase_n.dataset_id"),
        output_subdirectory=_safe(mapping["output_subdirectory"], "phase_n.output_subdirectory"),
        seed=_positive_int(mapping["seed"], "phase_n.seed"),
        scenes=scenes,
        source_protocol=_safe(mapping["source_protocol"], "phase_n.source_protocol"),
        max_format_retries=retries,
        allow_sample_extension=_false(
            mapping["allow_sample_extension"], "phase_n.allow_sample_extension"
        ),
    )


def _bridge(value: object) -> BridgeConfig:
    mapping = _mapping(value, "bridge")
    fields = {
        "dataset_id",
        "output_subdirectory",
        "seed",
        "scenes",
        "protocols_per_scene",
    }
    reject_unknown_fields(mapping, fields, label="bridge")
    if set(mapping) != fields:
        raise ValueError("bridge must contain every registered field")
    scenes = _positive_int(mapping["scenes"], "bridge.scenes")
    protocols = _positive_int(mapping["protocols_per_scene"], "bridge.protocols_per_scene")
    if (scenes, protocols) != (300, 2):
        raise ValueError("bridge must remain 300 scenes by two protocols")
    return BridgeConfig(
        dataset_id=_safe(mapping["dataset_id"], "bridge.dataset_id"),
        output_subdirectory=_safe(mapping["output_subdirectory"], "bridge.output_subdirectory"),
        seed=_positive_int(mapping["seed"], "bridge.seed"),
        scenes=scenes,
        protocols_per_scene=protocols,
    )


def _phase_c(value: object) -> PhaseCConfig:
    mapping = _mapping(value, "phase_c")
    fields = {
        "dataset_id",
        "output_subdirectory",
        "seed",
        "intake_scenes",
        "selected_family_quotas",
        "arms",
        "confirmatory_arms",
        "diagnostic_arms",
        "forks_per_arm",
        "allow_quota_redistribution",
        "allow_sample_extension",
    }
    reject_unknown_fields(mapping, fields, label="phase_c")
    if set(mapping) != fields:
        raise ValueError("phase_c must contain every registered field")
    quotas_value = _mapping(mapping["selected_family_quotas"], "selected_family_quotas")
    quotas = tuple(
        sorted(
            (_safe(key, "family"), _positive_int(count, key)) for key, count in quotas_value.items()
        )
    )
    if dict(quotas) != {"cross_series": 267, "duplicate_encoding": 266, "trend": 267}:
        raise ValueError("phase_c selected family quotas are not preregistered")
    arms_value = mapping["arms"]
    confirmatory_value = mapping["confirmatory_arms"]
    diagnostic_value = mapping["diagnostic_arms"]
    for raw, label in (
        (arms_value, "arms"),
        (confirmatory_value, "confirmatory_arms"),
        (diagnostic_value, "diagnostic_arms"),
    ):
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"phase_c.{label} must be a non-empty list")
    arms = tuple(_safe(item, "arm") for item in arms_value)
    confirmatory = tuple(_safe(item, "confirmatory arm") for item in confirmatory_value)
    diagnostic = tuple(_safe(item, "diagnostic arm") for item in diagnostic_value)
    registered = (
        "ablated",
        "valid",
        "sham",
        "counterfactual",
        "oracle_perception",
        "operator_swap",
    )
    if arms != registered:
        raise ValueError("phase_c arms must match the frozen order")
    if confirmatory != registered[:4] or diagnostic != registered[4:]:
        raise ValueError("phase_c confirmatory and diagnostic arms must partition arms")
    if set(confirmatory).intersection(diagnostic) or set(confirmatory).union(diagnostic) != set(
        arms
    ):
        raise ValueError("phase_c arm partitions are invalid")
    intake = _positive_int(mapping["intake_scenes"], "phase_c.intake_scenes")
    if intake != 6000:
        raise ValueError("phase_c.intake_scenes must equal the preregistered 6000")
    forks = _positive_int(mapping["forks_per_arm"], "phase_c.forks_per_arm")
    if forks != 8:
        raise ValueError("phase_c.forks_per_arm must equal eight")
    return PhaseCConfig(
        dataset_id=_safe(mapping["dataset_id"], "phase_c.dataset_id"),
        output_subdirectory=_safe(mapping["output_subdirectory"], "phase_c.output_subdirectory"),
        seed=_positive_int(mapping["seed"], "phase_c.seed"),
        intake_scenes=intake,
        selected_family_quotas=quotas,
        arms=arms,
        confirmatory_arms=confirmatory,
        diagnostic_arms=diagnostic,
        forks_per_arm=forks,
        allow_quota_redistribution=_false(
            mapping["allow_quota_redistribution"], "phase_c.allow_quota_redistribution"
        ),
        allow_sample_extension=_false(
            mapping["allow_sample_extension"], "phase_c.allow_sample_extension"
        ),
    )


def _analysis(value: object) -> AnalysisConfig:
    mapping = _mapping(value, "analysis")
    fields = {
        "alpha",
        "confidence",
        "tost_confidence",
        "target_power",
        "target_effect",
        "equivalence_margin",
        "bootstrap_resamples",
        "bootstrap_seed",
        "power_repetitions",
        "power_seed",
        "phase_n_minimum_eligible",
        "phase_n_null_rate",
        "required_eligible_scenes",
        "confirmatory_families",
        "exploratory_families",
        "power_artifact_path",
        "power_artifact_sha256",
    }
    reject_unknown_fields(mapping, fields, label="analysis")
    if set(mapping) != fields:
        raise ValueError("analysis must contain every registered field")
    fixed_probabilities = {
        "alpha": 0.05,
        "confidence": 0.95,
        "tost_confidence": 0.90,
        "target_power": 0.90,
        "target_effect": 0.05,
        "equivalence_margin": 0.02,
        "phase_n_null_rate": 0.05,
    }
    parsed_probabilities = {
        key: _probability(mapping[key], f"analysis.{key}") for key in fixed_probabilities
    }
    if parsed_probabilities != fixed_probabilities:
        raise ValueError("analysis probability thresholds differ from the preregistration")
    fixed_integers = {
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 2026081604,
        "power_repetitions": 2_000,
        "power_seed": 2026081605,
        "phase_n_minimum_eligible": 800,
        "required_eligible_scenes": 800,
    }
    parsed_integers = {
        key: _positive_int(mapping[key], f"analysis.{key}") for key in fixed_integers
    }
    if parsed_integers != fixed_integers:
        raise ValueError("analysis integer settings differ from the preregistration")
    confirmatory = mapping["confirmatory_families"]
    exploratory = mapping["exploratory_families"]
    if confirmatory != ["cross_series", "trend"]:
        raise ValueError("analysis confirmatory families must remain cross_series and trend")
    if exploratory != ["duplicate_encoding"]:
        raise ValueError("analysis exploratory family must remain duplicate_encoding")
    artifact_path = mapping["power_artifact_path"]
    artifact_sha256 = mapping["power_artifact_sha256"]
    if not isinstance(artifact_path, str) or _RELATIVE.fullmatch(artifact_path) is None:
        raise ValueError("analysis power artifact path is invalid")
    if artifact_path != "configs/recoverability/power_plan_v1.json":
        raise ValueError("analysis power artifact path is not preregistered")
    if not isinstance(artifact_sha256, str) or _SHA256.fullmatch(artifact_sha256) is None:
        raise ValueError("analysis power artifact SHA-256 is invalid")
    return AnalysisConfig(
        **parsed_probabilities,
        **parsed_integers,
        confirmatory_families=("cross_series", "trend"),
        exploratory_families=("duplicate_encoding",),
        power_artifact_path=artifact_path,
        power_artifact_sha256=artifact_sha256,
    )


def load_recoverability_protocol(path: Path) -> RecoverabilityProtocol:
    """Load the closed v1 protocol and reject adaptive or cross-stage drift."""

    mapping = load_yaml_mapping(path, label="recoverability protocol")
    fields = {
        "schema_version",
        "status",
        "model_id",
        "phase_n",
        "bridge",
        "phase_c",
        "analysis",
    }
    reject_unknown_fields(mapping, fields, label="recoverability protocol")
    if set(mapping) != fields:
        raise ValueError("recoverability protocol must contain every registered field")
    if mapping["schema_version"] != 1 or type(mapping["schema_version"]) is not int:
        raise ValueError("recoverability protocol schema_version must equal one")
    if mapping["status"] != "PREREGISTERED_NOT_RUN":
        raise ValueError("recoverability protocol status must remain PREREGISTERED_NOT_RUN")
    protocol = RecoverabilityProtocol(
        schema_version=1,
        status="PREREGISTERED_NOT_RUN",
        model_id=_safe(mapping["model_id"], "model_id"),
        phase_n=_phase_n(mapping["phase_n"]),
        bridge=_bridge(mapping["bridge"]),
        phase_c=_phase_c(mapping["phase_c"]),
        analysis=_analysis(mapping["analysis"]),
    )
    identifiers = {
        protocol.phase_n.dataset_id,
        protocol.bridge.dataset_id,
        protocol.phase_c.dataset_id,
    }
    outputs = {
        protocol.phase_n.output_subdirectory,
        protocol.bridge.output_subdirectory,
        protocol.phase_c.output_subdirectory,
    }
    seeds = {protocol.phase_n.seed, protocol.bridge.seed, protocol.phase_c.seed}
    if len(identifiers) != 3 or len(outputs) != 3 or len(seeds) != 3:
        raise ValueError("recoverability phases must use disjoint IDs, outputs, and seeds")
    return protocol
