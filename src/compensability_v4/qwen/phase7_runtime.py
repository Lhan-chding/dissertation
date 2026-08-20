"""Frozen contracts and objective statistics for Phase 7 multimodal diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from compensability_v4.data.splits import CONFIRM_SPLITS, DatasetSplit
from compensability_v4.eval.statistics import holm_adjust

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FAMILIES = frozenset({"cross_series", "duplicate_encoding", "trend"})
_CHECKPOINTS = (
    "Base",
    "C0",
    "C1",
    "T",
    "Base_AnswerOnly_RL",
    "Recovery_LoRA_RecoveryOutcome_RL",
    "Recovery_LoRA_AnswerOnly_RL",
)
_OOD_AXES = ("iid", "style_ood", "constraint_graph_ood", "error_mechanism_ood")
_METRICS = (
    "stage1_visual_exact",
    "post_revision_world_exact",
    "reasoning_operator_exact",
    "final_answer_exact",
    "operator_invariant_correct",
    "genuine_recovery",
    "error_cancellation",
    "trace_mismatch",
    "error_mechanism_shift",
)
_ROW_FIELDS = frozenset(
    {
        "scene_id",
        "checkpoint",
        "checkpoint_sha256",
        "family",
        "split",
        "ood_axis",
        "seed",
        "rollout_id",
        "image_sha256",
        *_METRICS,
    }
)

PHASE7_LOCKED_PATHS = (
    "configs/recoverability/v4_phase_7.yaml",
    "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
    "docs/QWEN_V4_SERVER_HANDOFF.md",
    "pyproject.toml",
    "requirements-gpu.lock.txt",
    "scripts/v4/15_prepare_phase7_multimodal.py",
    "scripts/v4/16_evaluate_phase7_multimodal.py",
    "src/compensability_v4/qwen/phase7_runtime.py",
)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be boolean")
    return value


@dataclass(frozen=True, slots=True)
class Phase7PlanConfig:
    chain: tuple[str, ...]
    checkpoints: tuple[str, ...]
    ood_axes: tuple[str, ...]
    required_metrics: tuple[str, ...]
    bootstrap_confidence: float
    bootstrap_resamples: int
    bootstrap_seed: int
    tost_margin: float
    confirmatory_evaluation_authorized: bool
    support_dev_diagnostic_authorized: bool
    subjective_success_threshold: None


@dataclass(frozen=True, slots=True)
class Phase7ChainRow:
    scene_id: str
    checkpoint: str
    checkpoint_sha256: str
    family: str
    split: DatasetSplit
    ood_axis: str
    seed: int
    rollout_id: int
    image_sha256: str
    stage1_visual_exact: bool
    post_revision_world_exact: bool
    reasoning_operator_exact: bool
    final_answer_exact: bool
    operator_invariant_correct: bool
    genuine_recovery: bool
    error_cancellation: bool
    trace_mismatch: bool
    error_mechanism_shift: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> Phase7ChainRow:
        if not isinstance(payload, Mapping) or set(payload) != _ROW_FIELDS:
            missing = sorted(_ROW_FIELDS - set(payload)) if isinstance(payload, Mapping) else []
            extra = sorted(set(payload) - _ROW_FIELDS) if isinstance(payload, Mapping) else []
            detail = f"missing={missing}, extra={extra}"
            raise ValueError(f"Phase 7 chain row fields are incomplete: {detail}")
        checkpoint = _string(payload["checkpoint"], "checkpoint")
        if checkpoint not in _CHECKPOINTS:
            raise ValueError("checkpoint is outside the frozen Phase 7 checkpoint set")
        family = _string(payload["family"], "family")
        if family not in _FAMILIES:
            raise ValueError("family is outside the frozen Phase 7 family set")
        axis = _string(payload["ood_axis"], "ood_axis")
        if axis not in _OOD_AXES:
            raise ValueError("ood_axis is outside the frozen Phase 7 OOD set")
        try:
            split = DatasetSplit(payload["split"])
        except (TypeError, ValueError) as error:
            raise ValueError("split is not a registered DatasetSplit") from error
        metrics = {name: _boolean(payload[name], name) for name in _METRICS}
        if metrics["operator_invariant_correct"] and not metrics["final_answer_exact"]:
            raise ValueError(
                "operator-invariant correct requires an objectively correct final answer"
            )
        expected_cancellation = (
            not metrics["post_revision_world_exact"]
            and metrics["final_answer_exact"]
            and not metrics["operator_invariant_correct"]
        )
        if metrics["error_cancellation"] is not expected_cancellation:
            raise ValueError("error cancellation label contradicts the frozen objective definition")
        expected_recovery = (
            not metrics["stage1_visual_exact"] and metrics["post_revision_world_exact"]
        )
        if metrics["genuine_recovery"] is not expected_recovery:
            raise ValueError("genuine recovery label contradicts the frozen objective definition")
        return cls(
            scene_id=_string(payload["scene_id"], "scene_id"),
            checkpoint=checkpoint,
            checkpoint_sha256=_sha256(payload["checkpoint_sha256"], "checkpoint_sha256"),
            family=family,
            split=split,
            ood_axis=axis,
            seed=_integer(payload["seed"], "seed", minimum=0),
            rollout_id=_integer(payload["rollout_id"], "rollout_id", minimum=0),
            image_sha256=_sha256(payload["image_sha256"], "image_sha256"),
            **metrics,
        )

    def to_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload["split"] = self.split.value
        return payload


def load_phase7_config(path: Path) -> Phase7PlanConfig:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Phase 7 config must be a regular file")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "status",
        "chain",
        "checkpoints",
        "ood_axes",
        "required_metrics",
        "statistics",
        "authorization",
        "subjective_success_threshold",
        "integrity_gates",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("status") != "PHASE_7_MULTIMODAL_DIAGNOSTIC_AUTHORIZED"
    ):
        raise ValueError("Phase 7 config schema differs from the frozen contract")
    chain = tuple(payload.get("chain", ()))
    expected_chain = (
        "image",
        "natural_observation",
        "revision_or_recovery",
        "chart_operation",
        "final_answer",
    )
    if chain != expected_chain:
        raise ValueError("Phase 7 full-chain contract drifted")
    if tuple(payload.get("checkpoints", ())) != _CHECKPOINTS:
        raise ValueError("Phase 7 checkpoint contract drifted")
    if tuple(payload.get("ood_axes", ())) != _OOD_AXES:
        raise ValueError("Phase 7 OOD contract drifted")
    if tuple(payload.get("required_metrics", ())) != _METRICS:
        raise ValueError("Phase 7 metric contract drifted")
    statistics = payload.get("statistics")
    if statistics != {
        "bootstrap_confidence": 0.95,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 2026082101,
        "tost_margin": 0.02,
        "holm_correction": True,
        "scene_is_statistical_unit": True,
        "rollout_is_statistical_unit": False,
    }:
        raise ValueError("Phase 7 statistical contract drifted")
    authorization = payload.get("authorization")
    if authorization != {
        "support_dev_diagnostic_authorized": True,
        "confirmatory_evaluation_authorized": False,
        "training_authorized": False,
        "rl_authorized": False,
        "downloads_authorized": False,
    }:
        raise ValueError("Phase 7 authorization contract drifted")
    if payload.get("subjective_success_threshold") is not None:
        raise ValueError("Phase 7 cannot apply a subjective success threshold")
    if payload.get("integrity_gates") != {
        "require_hash_bound_sources": True,
        "require_hash_bound_checkpoints": True,
        "require_complete_trace": True,
        "forbid_artifact_overwrite": True,
    }:
        raise ValueError("Phase 7 integrity gates drifted")
    return Phase7PlanConfig(
        chain=expected_chain,
        checkpoints=_CHECKPOINTS,
        ood_axes=_OOD_AXES,
        required_metrics=_METRICS,
        bootstrap_confidence=0.95,
        bootstrap_resamples=10_000,
        bootstrap_seed=2026082101,
        tost_margin=0.02,
        confirmatory_evaluation_authorized=False,
        support_dev_diagnostic_authorized=True,
        subjective_success_threshold=None,
    )


def validate_phase7_rows(
    rows: Iterable[Phase7ChainRow], *, confirmatory_evaluation_authorized: bool
) -> tuple[Phase7ChainRow, ...]:
    if type(confirmatory_evaluation_authorized) is not bool:
        raise TypeError("confirmatory_evaluation_authorized must be boolean")
    frozen = tuple(rows)
    if not frozen:
        raise ValueError("Phase 7 rows must not be empty")
    keys: set[tuple[str, str, int, int]] = set()
    scene_metadata: dict[str, tuple[str, DatasetSplit, str, str]] = {}
    checkpoint_hashes: dict[str, str] = {}
    for row in frozen:
        if not isinstance(row, Phase7ChainRow):
            raise TypeError("Phase 7 rows must be immutable Phase7ChainRow instances")
        if row.split in CONFIRM_SPLITS and not confirmatory_evaluation_authorized:
            raise ValueError("Phase 7 confirmatory data are fail-closed and not authorized")
        if row.split not in CONFIRM_SPLITS and row.split is not DatasetSplit.SUPPORT_DEV:
            raise ValueError("Phase 7 diagnostics may use only support_dev")
        key = (row.scene_id, row.checkpoint, row.seed, row.rollout_id)
        if key in keys:
            raise ValueError("Phase 7 row identity must be unique")
        keys.add(key)
        metadata = (row.family, row.split, row.ood_axis, row.image_sha256)
        previous_metadata = scene_metadata.setdefault(row.scene_id, metadata)
        if previous_metadata != metadata:
            raise ValueError("Phase 7 scene metadata drifted across rows")
        previous_hash = checkpoint_hashes.setdefault(row.checkpoint, row.checkpoint_sha256)
        if previous_hash != row.checkpoint_sha256:
            raise ValueError("Phase 7 checkpoint hash drifted across rows")
    return tuple(
        sorted(frozen, key=lambda row: (row.scene_id, row.checkpoint, row.seed, row.rollout_id))
    )


def _scene_values(rows: Iterable[Phase7ChainRow], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.scene_id, []).append(float(getattr(row, metric)))
    return {scene: sum(values) / len(values) for scene, values in sorted(grouped.items())}


def _interval_from_values(
    values: Mapping[str, float], *, n_resamples: int, seed: int, confidence: float = 0.95
) -> dict[str, float | int]:
    if not values:
        raise ValueError("Phase 7 interval requires scene values")
    observed = tuple(values.values())
    estimate = sum(observed) / len(observed)
    rng = random.Random(seed)
    boot = sorted(
        sum(rng.choice(observed) for _ in observed) / len(observed) for _ in range(n_resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, min(n_resamples - 1, int(alpha * n_resamples)))
    high_index = max(0, min(n_resamples - 1, math.ceil((1.0 - alpha) * n_resamples) - 1))
    return {
        "estimate": estimate,
        "ci_low": boot[low_index],
        "ci_high": boot[high_index],
        "confidence": confidence,
        "number_of_scenes": len(observed),
    }


def _metric_summary(
    rows: tuple[Phase7ChainRow, ...], metric: str, *, n_resamples: int, seed: int
) -> dict[str, object]:
    global_result = _interval_from_values(
        _scene_values(rows, metric), n_resamples=n_resamples, seed=seed
    )
    global_result["number_of_rollouts"] = len(rows)
    by_family = {
        family: _interval_from_values(
            _scene_values((row for row in rows if row.family == family), metric),
            n_resamples=n_resamples,
            seed=seed,
        )
        for family in sorted({row.family for row in rows})
    }
    by_axis = {
        axis: _interval_from_values(
            _scene_values((row for row in rows if row.ood_axis == axis), metric),
            n_resamples=n_resamples,
            seed=seed,
        )
        for axis in sorted({row.ood_axis for row in rows})
    }
    return {"global": global_result, "by_family": by_family, "by_ood_axis": by_axis}


_EFFECT_SPECS = {
    "T_minus_C0": ("C0", "T", "final_answer_exact"),
    "T_minus_C1": ("C1", "T", "final_answer_exact"),
    "recovery_reward_rl_minus_answer_only_rl": (
        "Recovery_LoRA_AnswerOnly_RL",
        "Recovery_LoRA_RecoveryOutcome_RL",
        "post_revision_world_exact",
    ),
}

_SEEDED_RL_CONTRAST = (
    ("Recovery_LoRA_AnswerOnly_RL", 1.0),
    ("T", -1.0),
    ("Base_AnswerOnly_RL", -1.0),
    ("Base", 1.0),
)


def _sign_flip_p_value(differences: tuple[float, ...], *, seed: int) -> float:
    observed = abs(sum(differences) / len(differences))
    if observed == 0.0:
        return 1.0
    if len(differences) <= 16:
        count = 0
        total = 1 << len(differences)
        for mask in range(total):
            candidate = sum(
                value if mask & (1 << index) else -value for index, value in enumerate(differences)
            ) / len(differences)
            count += abs(candidate) >= observed - 1e-15
        return count / total
    rng = random.Random(seed)
    total = 10_000
    count = 0
    for _ in range(total):
        candidate = sum(value if rng.getrandbits(1) else -value for value in differences)
        count += abs(candidate / len(differences)) >= observed - 1e-15
    return (count + 1) / (total + 1)


def _paired_effect(
    rows: tuple[Phase7ChainRow, ...],
    *,
    before_checkpoint: str,
    after_checkpoint: str,
    metric: str,
    n_resamples: int,
    seed: int,
    tost_margin: float,
) -> dict[str, object]:
    before = _scene_values((row for row in rows if row.checkpoint == before_checkpoint), metric)
    after = _scene_values((row for row in rows if row.checkpoint == after_checkpoint), metric)
    shared = tuple(sorted(before.keys() & after.keys()))
    values = {scene: after[scene] - before[scene] for scene in shared}
    return _effect_from_values(values, n_resamples=n_resamples, seed=seed, tost_margin=tost_margin)


def _effect_from_values(
    values: Mapping[str, float], *, n_resamples: int, seed: int, tost_margin: float
) -> dict[str, object]:
    if not values:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "paired_scene_count": 0,
            "two_sided_sign_flip_p_value": None,
            "holm_adjusted_p_value": None,
            "tost": {"margin": tost_margin, "equivalent": None},
        }
    interval = _interval_from_values(values, n_resamples=n_resamples, seed=seed)
    tost = _interval_from_values(
        values, n_resamples=n_resamples, seed=seed + 10_000, confidence=0.90
    )
    differences = tuple(values[scene] for scene in sorted(values))
    return {
        "estimate": interval["estimate"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "confidence": interval["confidence"],
        "paired_scene_count": len(values),
        "two_sided_sign_flip_p_value": _sign_flip_p_value(differences, seed=seed),
        "holm_adjusted_p_value": None,
        "tost": {
            "method": "scene_clustered_percentile_bootstrap_ci",
            "margin": tost_margin,
            "confidence": 0.90,
            "ci_low": tost["ci_low"],
            "ci_high": tost["ci_high"],
            "equivalent": bool(
                float(tost["ci_low"]) > -tost_margin and float(tost["ci_high"]) < tost_margin
            ),
        },
    }


def _contrast_effect(
    rows: tuple[Phase7ChainRow, ...],
    *,
    checkpoint_weights: tuple[tuple[str, float], ...],
    metric: str,
    n_resamples: int,
    seed: int,
    tost_margin: float,
) -> dict[str, object]:
    checkpoint_values = {
        checkpoint: _scene_values((row for row in rows if row.checkpoint == checkpoint), metric)
        for checkpoint, _weight in checkpoint_weights
    }
    shared = set.intersection(*(set(values) for values in checkpoint_values.values()))
    values = {
        scene: sum(
            weight * checkpoint_values[checkpoint][scene]
            for checkpoint, weight in checkpoint_weights
        )
        for scene in sorted(shared)
    }
    return _effect_from_values(values, n_resamples=n_resamples, seed=seed, tost_margin=tost_margin)


def _effect_block(
    rows: tuple[Phase7ChainRow, ...], *, n_resamples: int, seed: int, tost_margin: float
) -> dict[str, dict[str, object]]:
    effects = {
        name: _paired_effect(
            rows,
            before_checkpoint=before,
            after_checkpoint=after,
            metric=metric,
            n_resamples=n_resamples,
            seed=seed + index,
            tost_margin=tost_margin,
        )
        for index, (name, (before, after, metric)) in enumerate(_EFFECT_SPECS.items())
    }
    effects["seeded_rl_minus_base_rl"] = _contrast_effect(
        rows,
        checkpoint_weights=_SEEDED_RL_CONTRAST,
        metric="final_answer_exact",
        n_resamples=n_resamples,
        seed=seed + len(_EFFECT_SPECS),
        tost_margin=tost_margin,
    )
    available = {
        name: float(result["two_sided_sign_flip_p_value"])
        for name, result in effects.items()
        if result["two_sided_sign_flip_p_value"] is not None
    }
    adjusted = holm_adjust(available)
    for name, value in adjusted.items():
        effects[name]["holm_adjusted_p_value"] = value
    return effects


def summarize_phase7(
    rows: Iterable[Phase7ChainRow],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
    tost_margin: float = 0.02,
) -> dict[str, object]:
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if type(bootstrap_seed) is not int:
        raise TypeError("bootstrap_seed must be an integer")
    if isinstance(tost_margin, bool) or not isinstance(tost_margin, (int, float)):
        raise TypeError("tost_margin must be numeric")
    if not math.isfinite(float(tost_margin)) or tost_margin <= 0.0:
        raise ValueError("tost_margin must be positive and finite")
    frozen = validate_phase7_rows(
        rows,
        confirmatory_evaluation_authorized=False,
    )
    metrics = {
        metric: _metric_summary(
            frozen, metric, n_resamples=bootstrap_resamples, seed=bootstrap_seed + index
        )
        for index, metric in enumerate(_METRICS)
    }
    checkpoints = sorted({row.checkpoint for row in frozen})
    by_checkpoint = {
        checkpoint: {
            "number_of_scenes": len(
                {row.scene_id for row in frozen if row.checkpoint == checkpoint}
            ),
            "number_of_rollouts": sum(row.checkpoint == checkpoint for row in frozen),
            "metrics": {
                metric: _metric_summary(
                    tuple(row for row in frozen if row.checkpoint == checkpoint),
                    metric,
                    n_resamples=bootstrap_resamples,
                    seed=bootstrap_seed + index,
                )["global"]
                for index, metric in enumerate(_METRICS)
            },
        }
        for checkpoint in checkpoints
    }
    effects = _effect_block(
        frozen,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed + 100,
        tost_margin=float(tost_margin),
    )
    by_family_effects = {
        family: _effect_block(
            tuple(row for row in frozen if row.family == family),
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed + 200 + index,
            tost_margin=float(tost_margin),
        )
        for index, family in enumerate(sorted({row.family for row in frozen}))
    }
    by_axis_effects = {
        axis: _effect_block(
            tuple(row for row in frozen if row.ood_axis == axis),
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed + 300 + index,
            tost_margin=float(tost_margin),
        )
        for index, axis in enumerate(sorted({row.ood_axis for row in frozen}))
    }
    seed_variability = {
        str(seed): {
            metric: _metric_summary(
                tuple(row for row in frozen if row.seed == seed),
                metric,
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed + index,
            )["global"]
            for index, metric in enumerate(_METRICS)
        }
        for seed in sorted({row.seed for row in frozen})
    }
    return {
        "schema_version": 1,
        "status": "PHASE_7_MULTIMODAL_DIAGNOSTIC_EVALUATED",
        "number_of_rows": len(frozen),
        "number_of_scenes": len({row.scene_id for row in frozen}),
        "scene_is_statistical_unit": True,
        "rollout_is_statistical_unit": False,
        "subjective_success_threshold_applied": False,
        "confirmatory_data_used": any(row.split in CONFIRM_SPLITS for row in frozen),
        "metrics": metrics,
        "by_checkpoint": by_checkpoint,
        "registered_effects": effects,
        "registered_effects_by_family": by_family_effects,
        "registered_effects_by_ood_axis": by_axis_effects,
        "seed_level_variability": seed_variability,
    }


def build_phase7_execution_manifest(
    *,
    config: Phase7PlanConfig,
    source_sha256: Mapping[str, str],
    checkpoint_sha256: Mapping[str, str],
    config_sha256: str,
    package_lock_sha256: str,
) -> dict[str, object]:
    if not isinstance(config, Phase7PlanConfig):
        raise TypeError("config must be a frozen Phase7PlanConfig")
    required_sources = {
        "dataset_records",
        "support_dev",
        "phase4_summary",
        "phase5_summary",
        "phase6_evaluation",
    }
    if not isinstance(source_sha256, Mapping) or set(source_sha256) != required_sources:
        raise ValueError("Phase 7 source hashes do not close the frozen evidence set")
    sources = {
        name: _sha256(value, f"source_sha256[{name}]") for name, value in source_sha256.items()
    }
    if not isinstance(checkpoint_sha256, Mapping) or set(checkpoint_sha256) != set(
        config.checkpoints
    ):
        raise ValueError("Phase 7 checkpoint hashes do not close the frozen checkpoint set")
    checkpoints = {
        name: _sha256(value, f"checkpoint_sha256[{name}]")
        for name, value in checkpoint_sha256.items()
    }
    return {
        "schema_version": 1,
        "artifact_type": "v4_phase7_execution_manifest",
        "status": "PHASE_7_MULTIMODAL_EXECUTION_MANIFEST_PREPARED",
        "config_sha256": _sha256(config_sha256, "config_sha256"),
        "package_lock_sha256": _sha256(package_lock_sha256, "package_lock_sha256"),
        "source_sha256": sources,
        "checkpoint_sha256": checkpoints,
        "chain": list(config.chain),
        "required_metrics": list(config.required_metrics),
        "ood_axes": list(config.ood_axes),
        "support_dev_diagnostic_authorized": config.support_dev_diagnostic_authorized,
        "confirmatory_evaluation_authorized": config.confirmatory_evaluation_authorized,
        "scene_is_statistical_unit": True,
        "rollout_is_statistical_unit": False,
        "subjective_success_threshold_applied": False,
        "training_invoked": False,
        "rl_invoked": False,
    }


def validate_phase7_execution_manifest(
    payload: Mapping[str, object],
    *,
    config: Phase7PlanConfig,
    config_sha256: str,
    package_lock_sha256: str,
) -> dict[str, object]:
    extras = {
        "execution_parameters",
        "support_dev_image_bundle_sha256",
        "stage1_prompt_config_sha256",
    }
    if not isinstance(payload, Mapping):
        raise ValueError("Phase 7 execution manifest must be one mapping")
    expected = build_phase7_execution_manifest(
        config=config,
        source_sha256=payload.get("source_sha256", {}),  # type: ignore[arg-type]
        checkpoint_sha256=payload.get("checkpoint_sha256", {}),  # type: ignore[arg-type]
        config_sha256=config_sha256,
        package_lock_sha256=package_lock_sha256,
    )
    if set(payload) != set(expected) | extras or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Phase 7 execution manifest schema or frozen contract drifted")
    if not isinstance(payload["execution_parameters"], Mapping):
        raise ValueError("Phase 7 execution manifest parameters are malformed")
    _sha256(payload["support_dev_image_bundle_sha256"], "support_dev_image_bundle_sha256")
    _sha256(payload["stage1_prompt_config_sha256"], "stage1_prompt_config_sha256")
    return dict(payload)


def write_phase7_outputs(
    *,
    output_root: Path,
    rows: Iterable[Phase7ChainRow],
    summary: Mapping[str, object],
    source_sha256: Mapping[str, str],
) -> dict[str, Path]:
    frozen = validate_phase7_rows(rows, confirmatory_evaluation_authorized=False)
    if (
        not isinstance(summary, Mapping)
        or summary.get("status") != "PHASE_7_MULTIMODAL_DIAGNOSTIC_EVALUATED"
        or summary.get("scene_is_statistical_unit") is not True
        or summary.get("rollout_is_statistical_unit") is not False
        or summary.get("subjective_success_threshold_applied") is not False
        or type(summary.get("confirmatory_data_used")) is not bool
    ):
        raise ValueError("Phase 7 summary does not satisfy the publication contract")
    if not isinstance(source_sha256, Mapping) or not source_sha256:
        raise ValueError("Phase 7 summary requires hash-bound sources")
    sources = {
        name: _sha256(value, f"source_sha256[{name}]") for name, value in source_sha256.items()
    }
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("refusing to overwrite Phase 7 outputs")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        per_scene = staging / "per_scene.jsonl"
        summary_path = staging / "summary.json"
        per_scene.write_text(
            "".join(
                json.dumps(row.to_mapping(), sort_keys=True, allow_nan=False) + "\n"
                for row in frozen
            ),
            encoding="utf-8",
        )
        payload = {**dict(summary), "source_sha256": sources}
        summary_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "per_scene": output_root / "per_scene.jsonl",
        "summary": output_root / "summary.json",
    }


def verify_phase7_package_lock(
    *, lock_path: Path, repository_root: Path, expected_paths: tuple[str, ...]
) -> str:
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("Phase 7 package lock must be a regular file")
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    rows = payload.get("files") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status") != "FROZEN_PHASE_7_MULTIMODAL_SURFACE"
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("Phase 7 package lock is malformed")
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("Phase 7 package lock row is malformed")
        relative = row["path"]
        digest = row["sha256"]
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or relative in observed
        ):
            raise ValueError("Phase 7 package lock row has invalid fields")
        candidate = repository_root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"Phase 7 package lock missing file: {relative}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Phase 7 package lock mismatch: {relative}")
        observed.add(relative)
    if observed != set(expected_paths):
        raise RuntimeError("Phase 7 package lock closure mismatch")
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


__all__ = [
    "PHASE7_LOCKED_PATHS",
    "Phase7ChainRow",
    "Phase7PlanConfig",
    "build_phase7_execution_manifest",
    "load_phase7_config",
    "summarize_phase7",
    "validate_phase7_execution_manifest",
    "validate_phase7_rows",
    "verify_phase7_package_lock",
    "write_phase7_outputs",
]
