"""Shared fail-closed gates for v4 server-only scripts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from compensability_v4.qwen.model_loader import (
    MODEL_PATH,
    MODEL_SNAPSHOT_SHA256,
    require_server_model,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/recoverability/v4_phase_0_3.yaml"
PACKAGE_LOCK_PATH = ROOT / "configs/recoverability/v4/server_package_lock_phase_0_3.yaml"
LEGACY_SCREEN_RECORDS_SHA256 = "f964dd6c005bd7344804aca8c33de2f621cc8e171f8d0f4ccc73a08081f2414a"
CAPABILITY_PER_SCENE_SHA256 = "d01c391e136ed0e5c0ed52e50fe70f6ec128d221d218e7012c9adbcb4293929f"
CAPABILITY_SUMMARY_SHA256 = "8837a7275915f5a90c91eae8378ff6bd0466819842381996db1be8e4925705c7"
CAPABILITY_PAIRED_GAPS_SHA256 = "a2a5acb5b203719e7e5225e56643a25a4189277d13704cccd4ec86d61573e256"
CANDIDATE_LABELS_SHA256 = "a7a448f230038698c4127b220362c95d47f57cef90cc7904e71b1dacacc04dbd"
CANDIDATE_SCORES_SHA256 = "c303731438760a30fb2f78a489d20465a1c1e292b01941795f29351a0e234a62"
CANDIDATE_SUMMARY_SHA256 = "5e366dfecb2a4fd530407f896326bc99d3605385cfdadea34c0f478392280c73"
MODEL_INTROSPECTION_PATH = ROOT / "artifacts/v4/model_introspection.json"
MODULE_MANIFEST_PATH = ROOT / "artifacts/v4/module_manifest.txt"
MODEL_INTROSPECTION_SHA256 = "ed96d19a238d68497617071e29604313e0aae9a41a9e3bd24dbad451d87a0640"
MODULE_MANIFEST_SHA256 = "1c98fd8ba74fa5c30b8f585ffee5020544baf5be61f23e0c28c61a132973e8f0"
REQUIRED_RUNTIME_MODULES = (
    "model.visual.blocks.0",
    "model.visual.blocks.31",
    "model.visual.merger",
    "model.language_model.layers.0",
    "model.language_model.layers.35",
    "model.language_model.norm",
    "lm_head",
)


@dataclass(frozen=True, slots=True)
class ValidatedServerInputs:
    config_sha256: str
    package_lock_sha256: str
    model_snapshot_sha256: str
    inputs: tuple[dict[str, str], ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blocked_unless_execute(execute: bool) -> bool:
    if execute:
        return False
    print("BLOCKED: server-only v4 action requires explicit --execute.")
    return True


def _canonical_file(path: Path, expected: Path, label: str) -> None:
    if path.is_symlink() or path.resolve() != expected.resolve() or not path.is_file():
        raise RuntimeError(f"{label} must be the canonical repository file: {expected}")


def _load_config(path: Path) -> dict[str, object]:
    _canonical_file(path, CONFIG_PATH, "v4 config")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("v4 config schema is invalid")
    model = payload.get("model")
    authorization = payload.get("authorization")
    reporting = payload.get("reporting")
    integrity = payload.get("integrity_gates")
    vision = payload.get("vision_input")
    runtime = payload.get("runtime_evidence")
    sections = (model, authorization, reporting, integrity, vision, runtime)
    if not all(isinstance(item, dict) for item in sections):
        raise RuntimeError("v4 config sections are incomplete")
    assert isinstance(model, dict)
    assert isinstance(authorization, dict)
    assert isinstance(reporting, dict)
    assert isinstance(integrity, dict)
    assert isinstance(vision, dict)
    assert isinstance(runtime, dict)
    if (
        model.get("local_path") != MODEL_PATH
        or model.get("snapshot_sha256") != MODEL_SNAPSHOT_SHA256
    ):
        raise RuntimeError("v4 model pin differs from the archived plan")
    if model.get("model_loading_allowed") is not True or model.get("offline_only") is not True:
        raise RuntimeError("v4 model loading/offline contract is invalid")
    if authorization.get("training_authorized") is not False:
        raise RuntimeError("Phase 0-3 must not authorize training")
    if authorization.get("rl_authorized") is not False:
        raise RuntimeError("Phase 0-3 must not authorize RL")
    if authorization.get("downloads_authorized") is not False:
        raise RuntimeError("Phase 0-3 must remain offline")
    required_reporting = {
        "subjective_success_thresholds_forbidden": True,
        "report_scene_clustered_confidence_intervals": True,
        "report_paired_and_family_stratified_effects": True,
        "report_policy_support_and_reward_variance": True,
    }
    if any(reporting.get(key) is not value for key, value in required_reporting.items()):
        raise RuntimeError("v4 reporting contract is incomplete")
    forbidden_thresholds = {"minimum_visual_repair_rate", "minimum_recovery_accuracy"}
    if forbidden_thresholds & set(reporting):
        raise RuntimeError("subjective empirical success thresholds are forbidden")
    if vision.get("resized_height") != 280 or vision.get("resized_width") != 280:
        raise RuntimeError("v4 fixed visual budget must remain 280x280")
    if any(value is not True for value in integrity.values()):
        raise RuntimeError("all objective v4 integrity gates must remain enabled")
    expected_runtime = {
        "model_introspection_path": "artifacts/v4/model_introspection.json",
        "model_introspection_sha256": MODEL_INTROSPECTION_SHA256,
        "module_manifest_path": "artifacts/v4/module_manifest.txt",
        "module_manifest_sha256": MODULE_MANIFEST_SHA256,
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "language_layers": 36,
        "vision_layers": 32,
        "module_count": 839,
        "required_modules": list(REQUIRED_RUNTIME_MODULES),
    }
    if set(runtime) != set(expected_runtime) or any(
        runtime.get(key) != value for key, value in expected_runtime.items()
    ):
        raise RuntimeError("v4 runtime evidence contract drifted")
    phase_1 = payload.get("phase_1_capability_chain")
    expected_phase_1 = {
        "source_scenes": 580,
        "world_recoverable_scenes": 579,
        "excluded_ambiguous_scenes": 1,
        "included_family_counts": {
            "cross_series": 208,
            "duplicate_encoding": 182,
            "trend": 189,
        },
        "excluded_family_counts": {"trend": 1},
        "model_call_cap": 3474,
        "calls_per_scene": 6,
        "t1_calls_per_scene": 1,
        "t1_yes_calls": 290,
        "t1_no_calls": 289,
        "t5_candidate_count": 4,
        "t5_true_label_slot_counts": [145, 145, 145, 144],
        "max_new_tokens": 32,
        "do_sample": False,
        "seed": 2026081701,
        "bootstrap_resamples": 10000,
    }
    if (
        not isinstance(phase_1, dict)
        or set(phase_1) != set(expected_phase_1)
        or phase_1 != expected_phase_1
    ):
        raise RuntimeError("v4 Phase 1 capability execution contract drifted")
    phase_2 = payload.get("phase_2_candidate_scoring")
    expected_phase_2 = {
        "source_scenes": 580,
        "world_recoverable_scenes": 579,
        "included_family_counts": {
            "cross_series": 208,
            "duplicate_encoding": 182,
            "trend": 189,
        },
        "cue_conditions": [
            "no_cue",
            "valid_cue",
            "sham_cue",
            "counterfactual_cue",
        ],
        "candidate_count": 4,
        "model_forward_cap": 2316,
        "calls_per_scene": 4,
        "true_label_slot_counts": [145, 145, 145, 144],
        "seed": 2026081701,
        "bootstrap_resamples": 10000,
        "generation_allowed": False,
        "phase_1_revision": "0995637d488cfa822f6ccb6a2a47f1d96df333b9",
        "phase_1_config_sha256": (
            "a26feecb95dddc13549fe802b96137d4117d9cea4cb833f6156022acf4694aa5"
        ),
        "phase_1_package_lock_sha256": (
            "27859072ab266f50cbd547e319973e7068f7a0a04ae65a770d4c15df265b73b7"
        ),
        "capability_per_scene_sha256": CAPABILITY_PER_SCENE_SHA256,
        "capability_summary_sha256": CAPABILITY_SUMMARY_SHA256,
        "capability_paired_gaps_sha256": CAPABILITY_PAIRED_GAPS_SHA256,
    }
    if (
        not isinstance(phase_2, dict)
        or set(phase_2) != set(expected_phase_2)
        or phase_2 != expected_phase_2
    ):
        raise RuntimeError("v4 Phase 2 candidate-scoring execution contract drifted")
    layerwise = payload.get("phase_2_layerwise_assimilation")
    expected_layerwise = {
        "source_scenes": 580,
        "world_recoverable_scenes": 579,
        "included_family_counts": {
            "cross_series": 208,
            "duplicate_encoding": 182,
            "trend": 189,
        },
        "cue_conditions": [
            "no_cue",
            "valid_cue",
            "sham_cue",
            "counterfactual_cue",
        ],
        "model_forward_cap": 2316,
        "calls_per_scene": 4,
        "language_layers": 36,
        "final_logit_absolute_tolerance": 1e-5,
        "final_logit_relative_tolerance": 1e-5,
        "numerical_equality_tolerance": 1e-8,
        "bootstrap_resamples": 10000,
        "generation_allowed": False,
        "phase_2_revision": "fa8f9e64cf37190ffa8ba70206691fb043be3f1f",
        "phase_2_config_sha256": (
            "39ac4534cf2786f18ea26bfa84d3230edfdd205f3397502934a90b792724401f"
        ),
        "phase_2_package_lock_sha256": (
            "75fc91ef1fa1b217c07485242b3036bb3a789e4a9ebbd715f2e61639541c6c7a"
        ),
        "candidate_labels_sha256": CANDIDATE_LABELS_SHA256,
        "candidate_scores_sha256": CANDIDATE_SCORES_SHA256,
        "candidate_summary_sha256": CANDIDATE_SUMMARY_SHA256,
    }
    if (
        not isinstance(layerwise, dict)
        or set(layerwise) != set(expected_layerwise)
        or layerwise != expected_layerwise
    ):
        raise RuntimeError("v4 Phase 2 layerwise execution contract drifted")
    return payload


def validate_runtime_evidence(runtime: dict[str, object]) -> dict[str, object]:
    if not isinstance(runtime, dict):
        raise RuntimeError("runtime evidence section must be a mapping")
    if runtime.get("model_introspection_path") != "artifacts/v4/model_introspection.json":
        raise RuntimeError("runtime evidence introspection path drifted")
    if runtime.get("module_manifest_path") != "artifacts/v4/module_manifest.txt":
        raise RuntimeError("runtime evidence module path drifted")
    _canonical_file(MODEL_INTROSPECTION_PATH, MODEL_INTROSPECTION_PATH, "model introspection")
    _canonical_file(MODULE_MANIFEST_PATH, MODULE_MANIFEST_PATH, "module manifest")
    expected_intro = runtime.get("model_introspection_sha256")
    expected_manifest = runtime.get("module_manifest_sha256")
    if (
        not isinstance(expected_intro, str)
        or not isinstance(expected_manifest, str)
        or sha256(MODEL_INTROSPECTION_PATH) != expected_intro
        or sha256(MODULE_MANIFEST_PATH) != expected_manifest
    ):
        raise RuntimeError("runtime evidence SHA-256 mismatch")
    payload = json.loads(MODEL_INTROSPECTION_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("runtime introspection artifact is malformed")
    if payload.get("model_class") != runtime.get("model_class"):
        raise RuntimeError("runtime evidence model class drifted")
    if payload.get("language_layers") != runtime.get("language_layers"):
        raise RuntimeError("runtime evidence language layer count drifted")
    if payload.get("module_count") != runtime.get("module_count"):
        raise RuntimeError("runtime evidence module count drifted")
    vision = payload.get("vision_config")
    if not isinstance(vision, dict) or vision.get("depth") != runtime.get("vision_layers"):
        raise RuntimeError("runtime evidence vision depth drifted")
    modules = tuple(MODULE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines())
    required_modules = runtime.get("required_modules")
    if len(modules) != runtime.get("module_count") or len(set(modules)) != len(modules):
        raise RuntimeError("runtime evidence module manifest count or uniqueness drifted")
    if not isinstance(required_modules, list) or any(
        not isinstance(module, str) or module not in modules for module in required_modules
    ):
        raise RuntimeError("runtime evidence required modules drifted")
    return payload


def _verify_package_lock(path: Path) -> str:
    _canonical_file(path, PACKAGE_LOCK_PATH, "v4 package lock")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list) or not files:
        raise RuntimeError("v4 package lock is empty or malformed")
    seen: set[str] = set()
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise RuntimeError("v4 package lock row is malformed")
        relative, expected = row["path"], row["sha256"]
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError("v4 package lock row has invalid fields")
        if relative in seen:
            raise RuntimeError("v4 package lock contains duplicate paths")
        seen.add(relative)
        candidate = ROOT / relative
        if candidate.is_symlink() or not candidate.is_file() or sha256(candidate) != expected:
            raise RuntimeError(f"v4 package lock mismatch: {relative}")
    expected_closure = {
        "configs/recoverability/v4_phase_0_3.yaml",
        "configs/recoverability/v4/phase_1_3_prompts.yaml",
        "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
        "docs/QWEN_V4_SERVER_HANDOFF.md",
        "pyproject.toml",
        "requirements-gpu.lock.txt",
    }
    expected_closure.update(
        candidate.relative_to(ROOT).as_posix()
        for candidate in (ROOT / "scripts/v4").glob("*.py")
        if candidate.is_file() and not candidate.is_symlink()
    )
    expected_closure.update(
        candidate.relative_to(ROOT).as_posix()
        for candidate in (ROOT / "src/compensability_v4").rglob("*.py")
        if candidate.is_file() and not candidate.is_symlink()
    )
    if seen != expected_closure:
        missing = sorted(expected_closure - seen)
        extra = sorted(seen - expected_closure)
        raise RuntimeError(f"v4 package lock closure mismatch; missing={missing}, extra={extra}")
    return sha256(path)


def _validate_raw_input(path: Path, expected_sha256: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"raw evidence must be a non-empty regular file: {path}")
    observed = sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"raw evidence SHA-256 mismatch: {path}")
    if path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            rows = 0
            for line in stream:
                if line.strip():
                    json.loads(line)
                    rows += 1
            if rows == 0:
                raise RuntimeError(f"raw JSONL evidence has no records: {path}")
    return {"path": str(path.resolve()), "sha256": observed}


def validate_server_inputs(
    *,
    config: Path,
    package_lock: Path,
    model_path: Path,
    inputs: Iterable[Path],
    input_sha256: Iterable[str],
    expected_input_sha256: Iterable[str],
    require_raw_evidence: bool,
) -> ValidatedServerInputs:
    """Validate every integrity boundary before model loading or artifact creation."""

    _load_config(config)
    package_digest = _verify_package_lock(package_lock)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
    paths = tuple(inputs)
    digests = tuple(input_sha256)
    expected_digests = tuple(expected_input_sha256)
    if len(paths) != len(digests):
        raise RuntimeError("each --input requires one matching --input-sha256")
    if require_raw_evidence and not paths:
        raise RuntimeError("this phase requires hash-bound raw server evidence")
    if digests != expected_digests:
        raise RuntimeError("input digest does not match the frozen evidence SHA-256 contract")
    validated = tuple(
        _validate_raw_input(path, digest) for path, digest in zip(paths, digests, strict=True)
    )
    require_server_model(model_path, MODEL_SNAPSHOT_SHA256)
    return ValidatedServerInputs(
        config_sha256=sha256(config),
        package_lock_sha256=package_digest,
        model_snapshot_sha256=MODEL_SNAPSHOT_SHA256,
        inputs=validated,
    )


def write_execution_manifest(
    output: Path,
    *,
    phase: str,
    validation: ValidatedServerInputs,
    intended_artifacts: Iterable[str],
    integrity_gates: Iterable[str],
) -> None:
    """Write a no-overwrite proof that a server phase passed objective pre-work gates."""

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite server execution manifest: {output}")
    if output.parent.is_symlink():
        raise RuntimeError("execution manifest parent must not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "artifact_type": "v4_server_execution_manifest",
        "phase": phase,
        "status": "PREWORK_MANIFEST_ONLY_PHASE_NOT_EXECUTED",
        "config_sha256": validation.config_sha256,
        "package_lock_sha256": validation.package_lock_sha256,
        "model_snapshot_sha256": validation.model_snapshot_sha256,
        "hash_bound_inputs": list(validation.inputs),
        "intended_artifacts": list(intended_artifacts),
        "integrity_gates": list(integrity_gates),
        "phase_specific_gates_executed": False,
        "training_invoked": False,
        "rl_invoked": False,
        "subjective_success_threshold_applied": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


__all__ = [
    "CONFIG_PATH",
    "LEGACY_SCREEN_RECORDS_SHA256",
    "MODEL_INTROSPECTION_PATH",
    "MODULE_MANIFEST_PATH",
    "PACKAGE_LOCK_PATH",
    "ROOT",
    "ValidatedServerInputs",
    "blocked_unless_execute",
    "sha256",
    "validate_runtime_evidence",
    "validate_server_inputs",
    "write_execution_manifest",
]
