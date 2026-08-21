"""Fail-closed contracts for the frozen Phase 8 confirmatory evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType

import yaml

from compensability_v4.data.splits import CONFIRM_SPLITS, DatasetSplit
from compensability_v4.eval.answer_source import AnswerSource, classify_answer_source
from compensability_v4.qwen.phase7_runtime import (
    Phase7ChainRow,
    _effect_block,
    _metric_summary,
)
from compensability_v4.schemas.observation import NaturalObservation
from compensability_v4.schemas.scene import RecoveryScene

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
_AXIS_SPLITS = {
    "iid": DatasetSplit.CONFIRM_IID,
    "style_ood": DatasetSplit.CONFIRM_STYLE_OOD,
    "constraint_graph_ood": DatasetSplit.CONFIRM_CONSTRAINT_OOD,
    "error_mechanism_ood": DatasetSplit.CONFIRM_ERROR_MECHANISM_OOD,
}
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
_ENDPOINTS = ("free_generation_answer_exact", "deterministic_chain_answer_exact")
_BASE_ROW_FIELDS = frozenset(
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
_ROW_FIELDS = _BASE_ROW_FIELDS | frozenset({*_ENDPOINTS, "answer_source"})
_ACK = "I_UNDERSTAND_THIS_CONSUMES_THE_FROZEN_PHASE_8_CONFIRM_SET"
_SOURCE_KEYS = frozenset(
    {
        "legacy_diagnostic",
        "symbolic_support_train",
        "natural_error_support_train",
        "support_dev",
        "phase7_evaluation",
        "confirm_scenes",
        "confirm_observations",
        "confirm_summary",
        "confirm_image_bundle",
        "prompt_config",
    }
)

PHASE8_LOCKED_PATHS = (
    "configs/recoverability/v4_phase_8.yaml",
    "configs/recoverability/v4/phase_1_3_prompts.yaml",
    "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
    "docs/QWEN_V4_SERVER_HANDOFF.md",
    "pyproject.toml",
    "requirements-gpu.lock.txt",
    "scripts/v4/18_freeze_phase8_confirm_data.py",
    "scripts/v4/19_evaluate_phase8_confirmatory.py",
    "src/compensability_v4/data/splits.py",
    "src/compensability_v4/qwen/phase8_confirm_runtime.py",
    "src/compensability_v4/qwen/phase8_execution.py",
)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class Phase8PlanConfig:
    checkpoints: tuple[str, ...]
    ood_axes: tuple[str, ...]
    fixed_scene_counts: Mapping[str, int]
    required_metrics: tuple[str, ...]
    generation_seed: int
    evaluation_seed: int
    bootstrap_seed: int
    bootstrap_confidence: float
    bootstrap_resamples: int
    tost_margin: float
    confirmatory_evaluation_authorized: bool
    require_explicit_ack: bool
    subjective_success_threshold: None


@dataclass(frozen=True, slots=True)
class Phase8NaturalErrorExample:
    scene_id: str
    ood_axis: str
    scene: RecoveryScene
    observation: NaturalObservation
    error_indices: tuple[int, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "ood_axis": self.ood_axis,
            "scene": self.scene.to_mapping(),
            "observation": self.observation.to_mapping(),
            "error_indices": list(self.error_indices),
        }


@dataclass(frozen=True, slots=True)
class FrozenPhase8NaturalErrors:
    examples: tuple[Phase8NaturalErrorExample, ...]
    candidate_scene_count: int
    natural_error_count: int
    all_natural_stage1_errors_included: bool
    selection_uses_model_outcome_threshold: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "candidate_scene_count": self.candidate_scene_count,
            "natural_error_count": self.natural_error_count,
            "all_natural_stage1_errors_included": self.all_natural_stage1_errors_included,
            "selection_uses_model_outcome_threshold": self.selection_uses_model_outcome_threshold,
            "examples": [example.to_mapping() for example in self.examples],
        }


@dataclass(frozen=True, slots=True)
class Phase8ConfirmRow:
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
    free_generation_answer_exact: bool
    deterministic_chain_answer_exact: bool
    answer_source: AnswerSource

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> Phase8ConfirmRow:
        if not isinstance(payload, Mapping) or set(payload) != _ROW_FIELDS:
            missing = sorted(_ROW_FIELDS - set(payload)) if isinstance(payload, Mapping) else []
            extra = sorted(set(payload) - _ROW_FIELDS) if isinstance(payload, Mapping) else []
            raise ValueError(
                f"Phase 8 result row fields are incomplete: missing={missing}, extra={extra}"
            )
        base = Phase7ChainRow.from_mapping({name: payload[name] for name in _BASE_ROW_FIELDS})
        expected_split = _AXIS_SPLITS[base.ood_axis]
        if base.split is not expected_split:
            raise ValueError("Phase 8 confirm split and OOD axis are inconsistent")
        endpoints: dict[str, bool] = {}
        for name in _ENDPOINTS:
            value = payload[name]
            if type(value) is not bool:
                raise TypeError(f"{name} must be boolean")
            endpoints[name] = value
        if endpoints["free_generation_answer_exact"] is not base.final_answer_exact:
            raise ValueError("free-generation endpoint must match the frozen final-answer metric")
        try:
            answer_source = AnswerSource(payload["answer_source"])
        except (TypeError, ValueError) as error:
            raise ValueError("answer source is not registered") from error
        expected_source = classify_answer_source(
            answer_correct=endpoints["free_generation_answer_exact"],
            world_recovered=base.genuine_recovery,
            operator_invariant=base.operator_invariant_correct,
            error_cancelled=base.error_cancellation,
            visual_reread_evidence=base.stage1_visual_exact,
        )
        if answer_source is not expected_source:
            raise ValueError("answer source contradicts the frozen objective evidence")
        base_fields = base.to_mapping()
        base_fields["split"] = base.split
        return cls(**base_fields, **endpoints, answer_source=answer_source)  # type: ignore[arg-type]

    def to_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload["split"] = self.split.value
        payload["answer_source"] = self.answer_source.value
        return payload


def load_phase8_config(path: Path) -> Phase8PlanConfig:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Phase 8 config must be a regular file")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "status",
        "checkpoints",
        "ood_axes",
        "fixed_scene_counts",
        "required_metrics",
        "seeds",
        "statistics",
        "authorization",
        "subjective_success_threshold",
        "integrity_gates",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("status") != "PHASE_8_CONFIRMATORY_EVALUATION_AUTHORIZED"
        or tuple(payload.get("checkpoints", ())) != _CHECKPOINTS
        or tuple(payload.get("ood_axes", ())) != _OOD_AXES
        or tuple(payload.get("required_metrics", ())) != _METRICS
    ):
        raise ValueError("Phase 8 config schema differs from the frozen contract")
    counts = payload.get("fixed_scene_counts")
    if not isinstance(counts, dict) or set(counts) != set(_OOD_AXES):
        raise ValueError("Phase 8 fixed scene counts must close all four OOD axes")
    fixed_counts = MappingProxyType(
        {axis: _positive_integer(counts[axis], f"fixed_scene_counts[{axis}]") for axis in _OOD_AXES}
    )
    seeds = payload.get("seeds")
    if not isinstance(seeds, dict) or set(seeds) != {"generation", "evaluation", "bootstrap"}:
        raise ValueError("Phase 8 seed contract is malformed")
    seed_values = tuple(_positive_integer(seeds[name], f"{name}_seed") for name in seeds)
    prior = {2026082005, 2026082006, 2026082007, 2026082101, 2026082102}
    if len(set(seed_values)) != 3 or set(seed_values) & prior:
        raise ValueError("Phase 8 seeds must be distinct and unused by prior phases")
    statistics = payload.get("statistics")
    if statistics != {
        "bootstrap_confidence": 0.95,
        "bootstrap_resamples": 10_000,
        "tost_margin": 0.02,
        "holm_correction": True,
        "scene_is_statistical_unit": True,
        "rollout_is_statistical_unit": False,
    }:
        raise ValueError("Phase 8 statistical contract drifted")
    authorization = payload.get("authorization")
    if authorization != {
        "confirmatory_evaluation_authorized": True,
        "require_explicit_ack": True,
        "training_authorized": False,
        "rl_authorized": False,
        "downloads_authorized": False,
    }:
        raise ValueError("Phase 8 authorization contract drifted")
    if payload.get("subjective_success_threshold") is not None:
        raise ValueError("Phase 8 cannot apply a subjective success threshold")
    if payload.get("integrity_gates") != {
        "require_triple_isolation": True,
        "require_all_natural_stage1_errors": True,
        "require_hash_bound_sources": True,
        "require_hash_bound_checkpoints": True,
        "forbid_artifact_overwrite": True,
    }:
        raise ValueError("Phase 8 integrity gates drifted")
    return Phase8PlanConfig(
        checkpoints=_CHECKPOINTS,
        ood_axes=_OOD_AXES,
        fixed_scene_counts=MappingProxyType(fixed_counts),
        required_metrics=_METRICS,
        generation_seed=seeds["generation"],
        evaluation_seed=seeds["evaluation"],
        bootstrap_seed=seeds["bootstrap"],
        bootstrap_confidence=0.95,
        bootstrap_resamples=10_000,
        tost_margin=0.02,
        confirmatory_evaluation_authorized=True,
        require_explicit_ack=True,
        subjective_success_threshold=None,
    )


def validate_phase8_isolation(
    confirm_scenes: Iterable[RecoveryScene], prior_scenes: Iterable[RecoveryScene]
) -> tuple[RecoveryScene, ...]:
    confirm = tuple(confirm_scenes)
    prior = tuple(prior_scenes)
    if not confirm:
        raise ValueError("Phase 8 confirm scenes must not be empty")
    if any(not isinstance(scene, RecoveryScene) for scene in (*confirm, *prior)):
        raise TypeError("Phase 8 isolation inputs must contain RecoveryScene records")
    if any(scene.split not in CONFIRM_SPLITS for scene in confirm):
        raise ValueError("Phase 8 confirm scenes must use registered confirm splits")
    if any(scene.split in CONFIRM_SPLITS for scene in prior):
        raise ValueError("Phase 8 prior regimes cannot contain confirm scenes")
    if len({scene.scene_id for scene in (*prior, *confirm)}) != len(prior) + len(confirm):
        raise ValueError("Phase 8 scene_id values must be globally isolated")
    for field in ("semantic_scene_id", "numeric_table_id", "constraint_graph_id"):
        prior_values = {getattr(scene, field) for scene in prior}
        confirm_values = [getattr(scene, field) for scene in confirm]
        if prior_values & set(confirm_values) or len(set(confirm_values)) != len(confirm_values):
            raise ValueError(f"Phase 8 confirm scenes violate {field} isolation")
    return confirm


def freeze_phase8_natural_errors(
    scenes: Iterable[RecoveryScene],
    observations: Iterable[NaturalObservation],
    *,
    fixed_scene_counts: Mapping[str, int],
) -> FrozenPhase8NaturalErrors:
    frozen_scenes = tuple(scenes)
    frozen_observations = tuple(observations)
    if not frozen_scenes:
        raise ValueError("Phase 8 candidate scenes must not be empty")
    if not isinstance(fixed_scene_counts, Mapping) or not fixed_scene_counts:
        raise ValueError("Phase 8 fixed scene counts must not be empty")
    counts = {
        axis: _positive_integer(value, f"fixed scene count {axis}")
        for axis, value in fixed_scene_counts.items()
    }
    if not set(counts) <= set(_OOD_AXES):
        raise ValueError("Phase 8 fixed scene count has an unknown OOD axis")
    observed_counts = Counter(
        axis
        for scene in frozen_scenes
        for axis, split in _AXIS_SPLITS.items()
        if scene.split is split
    )
    if dict(observed_counts) != counts:
        raise ValueError("Phase 8 fixed scene count differs from the frozen candidate budget")
    by_scene: dict[str, NaturalObservation] = {}
    for observation in frozen_observations:
        if observation.scene_id in by_scene:
            raise ValueError("Phase 8 observations contain duplicate scene IDs")
        by_scene[observation.scene_id] = observation
    if set(by_scene) != {scene.scene_id for scene in frozen_scenes}:
        raise ValueError("Phase 8 observations must cover every fixed candidate scene exactly once")
    examples: list[Phase8NaturalErrorExample] = []
    for scene in frozen_scenes:
        observation = by_scene[scene.scene_id]
        differences = tuple(
            index
            for index, (truth, observed) in enumerate(
                zip(scene.truth, observation.observed_values, strict=True)
            )
            if truth != observed
        )
        if differences and observation.error_index not in differences:
            raise ValueError("Phase 8 natural observation error_index is inconsistent")
        if differences:
            axis = next(axis for axis, split in _AXIS_SPLITS.items() if scene.split is split)
            examples.append(
                Phase8NaturalErrorExample(
                    scene_id=scene.scene_id,
                    ood_axis=axis,
                    scene=scene,
                    observation=observation,
                    error_indices=differences,
                )
            )
    return FrozenPhase8NaturalErrors(
        examples=tuple(examples),
        candidate_scene_count=len(frozen_scenes),
        natural_error_count=len(examples),
        all_natural_stage1_errors_included=True,
        selection_uses_model_outcome_threshold=False,
    )


def validate_phase8_rows(rows: Iterable[Phase8ConfirmRow]) -> tuple[Phase8ConfirmRow, ...]:
    frozen = tuple(rows)
    if not frozen:
        raise ValueError("Phase 8 result rows must not be empty")
    identities: set[tuple[str, str, int, int]] = set()
    scene_metadata: dict[str, tuple[str, DatasetSplit, str, str]] = {}
    checkpoint_hashes: dict[str, str] = {}
    for row in frozen:
        if not isinstance(row, Phase8ConfirmRow):
            raise TypeError("Phase 8 rows must be immutable Phase8ConfirmRow instances")
        if row.split not in CONFIRM_SPLITS or row.split is not _AXIS_SPLITS[row.ood_axis]:
            raise ValueError("Phase 8 row is outside the authorized confirm split")
        identity = (row.scene_id, row.checkpoint, row.seed, row.rollout_id)
        if identity in identities:
            raise ValueError("Phase 8 row identity must be unique")
        identities.add(identity)
        metadata = (row.family, row.split, row.ood_axis, row.image_sha256)
        if scene_metadata.setdefault(row.scene_id, metadata) != metadata:
            raise ValueError("Phase 8 scene metadata drifted across rows")
        if (
            checkpoint_hashes.setdefault(row.checkpoint, row.checkpoint_sha256)
            != row.checkpoint_sha256
        ):
            raise ValueError("Phase 8 checkpoint hash drifted across rows")
    return tuple(
        sorted(frozen, key=lambda row: (row.scene_id, row.checkpoint, row.seed, row.rollout_id))
    )


def summarize_phase8(
    rows: Iterable[Phase8ConfirmRow],
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
    if not math.isfinite(float(tost_margin)) or tost_margin <= 0:
        raise ValueError("tost_margin must be positive and finite")
    frozen = validate_phase8_rows(rows)
    metrics = {
        metric: _metric_summary(
            frozen, metric, n_resamples=bootstrap_resamples, seed=bootstrap_seed + index
        )
        for index, metric in enumerate(_METRICS)
    }
    endpoints = {
        endpoint: _metric_summary(
            frozen,
            endpoint,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed + len(_METRICS) + index,
        )
        for index, endpoint in enumerate(_ENDPOINTS)
    }
    checkpoints = sorted({row.checkpoint for row in frozen})
    by_checkpoint = {
        checkpoint: {
            "number_of_scenes": len(
                {row.scene_id for row in frozen if row.checkpoint == checkpoint}
            ),
            "number_of_rollouts": sum(row.checkpoint == checkpoint for row in frozen),
            "metrics": {
                name: _metric_summary(
                    tuple(row for row in frozen if row.checkpoint == checkpoint),
                    name,
                    n_resamples=bootstrap_resamples,
                    seed=bootstrap_seed + index,
                )["global"]
                for index, name in enumerate((*_METRICS, *_ENDPOINTS))
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
    deterministic_effects = _effect_block(
        tuple(
            replace(row, final_answer_exact=row.deterministic_chain_answer_exact) for row in frozen
        ),
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed + 400,
        tost_margin=float(tost_margin),
    )
    seed_variability = {
        str(seed): {
            name: _metric_summary(
                tuple(row for row in frozen if row.seed == seed),
                name,
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed + index,
            )["global"]
            for index, name in enumerate((*_METRICS, *_ENDPOINTS))
        }
        for seed in sorted({row.seed for row in frozen})
    }
    return {
        "schema_version": 1,
        "status": "PHASE_8_CONFIRMATORY_EVALUATED",
        "number_of_rows": len(frozen),
        "number_of_scenes": len({row.scene_id for row in frozen}),
        "scene_is_statistical_unit": True,
        "rollout_is_statistical_unit": False,
        "subjective_success_threshold_applied": False,
        "confirmatory_data_used": True,
        "metrics": metrics,
        "answer_endpoints": endpoints,
        "answer_source_counts": dict(
            sorted(Counter(row.answer_source.value for row in frozen).items())
        ),
        "by_checkpoint": by_checkpoint,
        "registered_effects": effects,
        "registered_effects_by_answer_endpoint": {
            "free_generation_answer_exact": effects,
            "deterministic_chain_answer_exact": deterministic_effects,
        },
        "registered_effects_by_family": {
            family: _effect_block(
                tuple(row for row in frozen if row.family == family),
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed + 200 + index,
                tost_margin=float(tost_margin),
            )
            for index, family in enumerate(sorted({row.family for row in frozen}))
        },
        "registered_effects_by_ood_axis": {
            axis: _effect_block(
                tuple(row for row in frozen if row.ood_axis == axis),
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed + 300 + index,
                tost_margin=float(tost_margin),
            )
            for index, axis in enumerate(sorted({row.ood_axis for row in frozen}))
        },
        "seed_level_variability": seed_variability,
    }


def build_phase8_execution_manifest(
    *,
    config: Phase8PlanConfig,
    source_sha256: Mapping[str, str],
    checkpoint_sha256: Mapping[str, str],
    config_sha256: str,
    package_lock_sha256: str,
    authorization_ack: str | None,
) -> dict[str, object]:
    if not isinstance(config, Phase8PlanConfig):
        raise TypeError("config must be a frozen Phase8PlanConfig")
    if not config.confirmatory_evaluation_authorized or not config.require_explicit_ack:
        raise PermissionError("Phase 8 confirmatory evaluation is not authorized")
    if authorization_ack != _ACK:
        raise PermissionError("Phase 8 requires the exact explicit confirm-set ACK")
    if not isinstance(source_sha256, Mapping) or set(source_sha256) != _SOURCE_KEYS:
        raise ValueError("Phase 8 source hashes do not close all prior and confirm evidence")
    sources = {
        name: _sha256(value, f"source_sha256[{name}]") for name, value in source_sha256.items()
    }
    if not isinstance(checkpoint_sha256, Mapping) or set(checkpoint_sha256) != set(_CHECKPOINTS):
        raise ValueError("Phase 8 checkpoint hashes do not close all seven checkpoints")
    checkpoints = {
        name: _sha256(value, f"checkpoint_sha256[{name}]")
        for name, value in checkpoint_sha256.items()
    }
    return {
        "schema_version": 1,
        "artifact_type": "v4_phase8_execution_manifest",
        "status": "PHASE_8_CONFIRMATORY_EXECUTION_MANIFEST_PREPARED",
        "config_sha256": _sha256(config_sha256, "config_sha256"),
        "package_lock_sha256": _sha256(package_lock_sha256, "package_lock_sha256"),
        "source_sha256": sources,
        "checkpoint_sha256": checkpoints,
        "fixed_scene_counts": dict(config.fixed_scene_counts),
        "ood_axes": list(config.ood_axes),
        "generation_seed": config.generation_seed,
        "evaluation_seed": config.evaluation_seed,
        "bootstrap_seed": config.bootstrap_seed,
        "confirmatory_evaluation_authorized": True,
        "authorization_ack_verified": True,
        "scene_is_statistical_unit": True,
        "rollout_is_statistical_unit": False,
        "subjective_success_threshold_applied": False,
        "training_invoked": False,
        "rl_invoked": False,
    }


def write_phase8_outputs(
    *,
    output_root: Path,
    rows: Iterable[Phase8ConfirmRow],
    summary: Mapping[str, object],
    source_sha256: Mapping[str, str],
) -> dict[str, Path]:
    frozen = validate_phase8_rows(rows)
    if (
        not isinstance(summary, Mapping)
        or summary.get("status") != "PHASE_8_CONFIRMATORY_EVALUATED"
        or summary.get("scene_is_statistical_unit") is not True
        or summary.get("rollout_is_statistical_unit") is not False
        or summary.get("subjective_success_threshold_applied") is not False
        or summary.get("confirmatory_data_used") is not True
    ):
        raise ValueError("Phase 8 summary does not satisfy the publication contract")
    if not isinstance(source_sha256, Mapping) or not source_sha256:
        raise ValueError("Phase 8 summary requires hash-bound sources")
    sources = {
        name: _sha256(value, f"source_sha256[{name}]") for name, value in source_sha256.items()
    }
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("refusing to overwrite Phase 8 outputs")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        (staging / "per_scene.jsonl").write_text(
            "".join(
                json.dumps(row.to_mapping(), sort_keys=True, allow_nan=False) + "\n"
                for row in frozen
            ),
            encoding="utf-8",
        )
        payload = {**dict(summary), "source_sha256": sources}
        (staging / "summary.json").write_text(
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


def verify_phase8_package_lock(
    *, lock_path: Path, repository_root: Path, expected_paths: tuple[str, ...]
) -> str:
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("Phase 8 package lock must be a regular file")
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    rows = payload.get("files") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status") != "FROZEN_PHASE_8_CONFIRMATORY_SURFACE"
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("Phase 8 package lock is malformed")
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("Phase 8 package lock row is malformed")
        relative = row["path"]
        digest = row["sha256"]
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or relative in observed
        ):
            raise ValueError("Phase 8 package lock row has invalid fields")
        candidate = repository_root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"Phase 8 package lock missing file: {relative}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Phase 8 package lock mismatch: {relative}")
        observed.add(relative)
    if observed != set(expected_paths):
        raise RuntimeError("Phase 8 package lock closure mismatch")
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


__all__ = [
    "PHASE8_LOCKED_PATHS",
    "FrozenPhase8NaturalErrors",
    "Phase8ConfirmRow",
    "Phase8NaturalErrorExample",
    "Phase8PlanConfig",
    "build_phase8_execution_manifest",
    "freeze_phase8_natural_errors",
    "load_phase8_config",
    "summarize_phase8",
    "validate_phase8_isolation",
    "validate_phase8_rows",
    "verify_phase8_package_lock",
    "write_phase8_outputs",
]
