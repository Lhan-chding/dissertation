"""Fail-closed Phase 6 manifest preparation from frozen Phase 4/5 evidence."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARM_CHECKPOINTS = frozenset({"Base", "T"})
_REWARD_MODES = frozenset({"none", "answer_only", "recovery_outcome", "constraint_aware"})
_ADAPTER_PATHS = {
    "C0": "C0_format_only/final_adapter",
    "C1": "C1_forward_arithmetic/final_adapter",
    "T": "T_constraint_recovery/final_adapter",
}


def _load_mapping_file(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]

        payload = yaml.safe_load(text)
    except ModuleNotFoundError:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain one mapping object")
    return payload


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256 hex digest")
    return value


def _tree_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"Phase 6 adapter directory is missing or unsafe: {path}")
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    if not files:
        raise RuntimeError(f"Phase 6 adapter directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise RuntimeError("Phase 6 adapter tree must not contain symlinks")
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_tree_hashes(run_root: Path) -> dict[str, str]:
    if run_root.is_symlink() or not run_root.is_dir():
        raise RuntimeError("Phase 6 Phase-4 run root is missing or unsafe")
    hashes: dict[str, str] = {}
    for checkpoint, relative in _ADAPTER_PATHS.items():
        adapter = run_root / relative
        if not (adapter / "adapter_config.json").is_file():
            raise RuntimeError(f"Phase 6 {checkpoint} adapter config is missing")
        model_files = tuple(adapter.glob("adapter_model.*"))
        if len(model_files) != 1 or model_files[0].is_symlink():
            raise RuntimeError(f"Phase 6 {checkpoint} adapter weights are missing or ambiguous")
        hashes[checkpoint] = _tree_sha256(adapter)
    return hashes


@dataclass(frozen=True, slots=True)
class Phase6Arm:
    name: str
    initialization_checkpoint: str
    reward_mode: str
    execute_rl: bool

    @classmethod
    def from_mapping(cls, value: object) -> "Phase6Arm":
        if not isinstance(value, dict) or set(value) != {
            "name",
            "initialization_checkpoint",
            "reward_mode",
            "execute_rl",
        }:
            raise ValueError("Phase 6 arm schema differs from the frozen contract")
        name = value["name"]
        checkpoint = value["initialization_checkpoint"]
        reward_mode = value["reward_mode"]
        execute_rl = value["execute_rl"]
        if not isinstance(name, str) or not name:
            raise ValueError("Phase 6 arm name must be non-empty")
        if checkpoint not in _ARM_CHECKPOINTS:
            raise ValueError("Phase 6 initialization checkpoint is not registered")
        if reward_mode not in _REWARD_MODES:
            raise ValueError("Phase 6 reward mode is not registered")
        if not isinstance(execute_rl, bool):
            raise TypeError("Phase 6 execute_rl must be boolean")
        if execute_rl != (reward_mode != "none"):
            raise ValueError("Phase 6 execute_rl must match the reward mode contract")
        return cls(
            name=name,
            initialization_checkpoint=checkpoint,
            reward_mode=reward_mode,
            execute_rl=execute_rl,
        )


@dataclass(frozen=True, slots=True)
class Phase6PlanConfig:
    model_snapshot_sha256: str
    required_pass_at_k: tuple[int, ...]
    informative_group_size: int
    required_source_sha256_keys: tuple[str, ...]
    arms: tuple[Phase6Arm, ...]
    constraint_aware_enabled: bool
    constraint_aware_arm_name: str
    constraint_aware_reward_mode: str
    required_metrics: tuple[str, ...]
    rollout_group_size: int
    phase4_run_root: Path
    policy_support_summary: Path
    execution_manifest: Path


def load_phase6_config(path: Path) -> Phase6PlanConfig:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Phase 6 config must be a regular file")
    payload = _load_mapping_file(path)
    expected = {
        "schema_version",
        "status",
        "model",
        "authorization",
        "phase5",
        "arms",
        "optional_controls",
        "diagnostics",
        "integrity_gates",
        "artifacts",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("schema_version") != 1
        or payload.get("status") != "PHASE_6_RL_PLAN_AUTHORIZED"
    ):
        raise ValueError("Phase 6 config schema differs from the frozen contract")
    if payload.get("authorization") != {
        "manifest_preparation_authorized": True,
        "local_training_authorized": False,
        "downloads_authorized": False,
    }:
        raise ValueError("Phase 6 authorization contract drifted")
    model = payload.get("model")
    if not isinstance(model, dict) or set(model) != {"local_path", "snapshot_sha256"}:
        raise ValueError("Phase 6 model contract is malformed")
    model_snapshot_sha256 = _require_sha256(model["snapshot_sha256"], "Phase 6 model snapshot")
    phase5 = payload.get("phase5")
    if not isinstance(phase5, dict):
        raise ValueError("Phase 6 Phase 5 evidence contract is malformed")
    required_pass_at_k = phase5.get("required_pass_at_k")
    if (
        phase5.get("required_status") != "PHASE_5_POLICY_SUPPORT_EXECUTED"
        or not isinstance(required_pass_at_k, list)
        or tuple(required_pass_at_k) != (1, 2, 4, 8, 16)
        or phase5.get("informative_group_size") != 8
        or phase5.get("require_scene_statistical_unit") is not True
        or phase5.get("require_no_subjective_threshold") is not True
        or phase5.get("require_no_confirmatory_data") is not True
        or tuple(phase5.get("required_source_sha256_keys", ()))
        != ("Base", "C0", "C1", "T", "support_dev")
    ):
        raise ValueError("Phase 6 Phase 5 evidence contract drifted")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("Phase 6 diagnostics contract is malformed")
    metrics = diagnostics.get("required_metrics")
    if (
        not isinstance(metrics, list)
        or tuple(metrics)
        != (
            "group_rewards",
            "reward_variance",
            "all_zero_group_rate",
            "all_one_group_rate",
            "non_degenerate_group_rate",
            "kl",
            "entropy",
            "exact_world_recovery",
            "observation_copy_rate",
            "answer_accuracy",
        )
        or diagnostics.get("rollout_group_size") != 8
        or diagnostics.get("record_scene_level_metrics") is not True
    ):
        raise ValueError("Phase 6 diagnostics contract drifted")
    optional_controls = payload.get("optional_controls")
    if (
        not isinstance(optional_controls, dict)
        or optional_controls.get("enable_constraint_aware_rl") is not False
        or optional_controls.get("planned_arm_name") != "Recovery_LoRA_ConstraintAware_RL"
        or optional_controls.get("reward_mode") != "constraint_aware"
    ):
        raise ValueError("Phase 6 optional-control contract drifted")
    arms_payload = payload.get("arms")
    if not isinstance(arms_payload, list):
        raise ValueError("Phase 6 arms must be a list")
    arms = tuple(Phase6Arm.from_mapping(item) for item in arms_payload)
    if tuple(arm.name for arm in arms) != (
        "Base",
        "Base_AnswerOnly_RL",
        "Recovery_LoRA",
        "Recovery_LoRA_RecoveryOutcome_RL",
        "Recovery_LoRA_AnswerOnly_RL",
    ):
        raise ValueError("Phase 6 arm order drifted")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Phase 6 artifacts contract is malformed")
    gates = payload.get("integrity_gates")
    if gates != {
        "require_offline_execution": True,
        "require_hash_bound_phase5_summary": True,
        "require_phase4_adapter_hash_match": True,
        "forbid_artifact_overwrite": True,
    }:
        raise ValueError("Phase 6 integrity gates drifted")
    return Phase6PlanConfig(
        model_snapshot_sha256=model_snapshot_sha256,
        required_pass_at_k=(1, 2, 4, 8, 16),
        informative_group_size=8,
        required_source_sha256_keys=("Base", "C0", "C1", "T", "support_dev"),
        arms=arms,
        constraint_aware_enabled=False,
        constraint_aware_arm_name="Recovery_LoRA_ConstraintAware_RL",
        constraint_aware_reward_mode="constraint_aware",
        required_metrics=tuple(metrics),
        rollout_group_size=8,
        phase4_run_root=Path(str(artifacts["phase4_run_root"])),
        policy_support_summary=Path(str(artifacts["policy_support_summary"])),
        execution_manifest=Path(str(artifacts["execution_manifest"])),
    )


def load_phase5_policy_support_summary(path: Path, *, expected_sha256: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Phase 6 policy-support summary must be a regular JSON file")
    _require_sha256(expected_sha256, "Phase 6 policy-support summary SHA-256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError("Phase 6 policy-support summary SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 6 policy-support summary must contain one JSON object")
    source_hashes = payload.get("source_sha256")
    by_checkpoint = payload.get("by_checkpoint")
    if (
        payload.get("status") != "PHASE_5_POLICY_SUPPORT_EXECUTED"
        or payload.get("scene_is_statistical_unit") is not True
        or payload.get("subjective_success_threshold_applied") is not False
        or payload.get("confirmatory_data_used") is not False
        or payload.get("training_invoked") is not False
        or payload.get("rl_invoked") is not False
        or payload.get("informative_group_size") != 8
        or tuple(payload.get("pass_at_k", ())) != (1, 2, 4, 8, 16)
        or not isinstance(source_hashes, dict)
        or tuple(sorted(source_hashes)) != ("Base", "C0", "C1", "T", "support_dev")
        or not isinstance(by_checkpoint, dict)
        or tuple(sorted(by_checkpoint)) != ("Base", "C0", "C1", "T")
    ):
        raise ValueError("Phase 5 policy-support summary differs from the frozen Phase 6 contract")
    for key, value in source_hashes.items():
        _require_sha256(value, f"Phase 5 source hash {key}")
    model_snapshot_sha256 = payload.get("model_snapshot_sha256")
    if model_snapshot_sha256 is not None:
        if _require_sha256(model_snapshot_sha256, "Phase 5 model snapshot") != (
            "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
        ):
            raise ValueError("Phase 5 model snapshot differs from the frozen Phase 6 contract")
    package_lock_sha256 = payload.get("package_lock_sha256")
    if package_lock_sha256 is not None:
        _require_sha256(package_lock_sha256, "Phase 5 package lock")
    support_dev_summary_sha256 = payload.get("support_dev_summary_sha256")
    if support_dev_summary_sha256 is not None:
        _require_sha256(support_dev_summary_sha256, "Phase 5 support-dev summary")
    return payload


def build_phase6_execution_manifest(
    *,
    config: Phase6PlanConfig,
    phase5_summary: dict[str, object],
    phase5_summary_sha256: str,
    phase4_run_root: Path,
    config_sha256: str,
    package_lock_sha256: str,
) -> dict[str, object]:
    summary_sources = phase5_summary["source_sha256"]
    assert isinstance(summary_sources, dict)
    observed = _checkpoint_tree_hashes(phase4_run_root)
    for checkpoint in ("C0", "C1", "T"):
        if observed[checkpoint] != summary_sources[checkpoint]:
            raise RuntimeError(
                f"Phase 6 Phase 4 adapter hash mismatch for {checkpoint}: "
                f"{observed[checkpoint]} != {summary_sources[checkpoint]}"
            )
    arms: list[dict[str, object]] = []
    for arm in config.arms:
        initialization_sha256 = (
            config.model_snapshot_sha256
            if arm.initialization_checkpoint == "Base"
            else str(summary_sources[arm.initialization_checkpoint])
        )
        arms.append(
            {
                "name": arm.name,
                "initialization_checkpoint": arm.initialization_checkpoint,
                "initialization_sha256": initialization_sha256,
                "reward_mode": arm.reward_mode,
                "execute_rl": arm.execute_rl,
                "planned_output_root": f"artifacts/v4/phase6/{arm.name}",
            }
        )
    manifest = {
        "schema_version": 1,
        "artifact_type": "v4_phase6_execution_manifest",
        "status": "PHASE_6_RL_EXECUTION_MANIFEST_PREPARED",
        "phase5_policy_support_summary_sha256": _require_sha256(
            phase5_summary_sha256, "Phase 6 Phase 5 summary SHA-256"
        ),
        "config_sha256": _require_sha256(config_sha256, "Phase 6 config hash"),
        "package_lock_sha256": _require_sha256(
            package_lock_sha256, "Phase 6 package-lock hash"
        ),
        "model_snapshot_sha256": config.model_snapshot_sha256,
        "source_sha256": dict(sorted((key, str(value)) for key, value in summary_sources.items())),
        "phase4_adapter_sha256": observed,
        "number_of_held_out_natural_errors": phase5_summary["number_of_held_out_natural_errors"],
        "held_out_family_counts": phase5_summary["held_out_family_counts"],
        "number_of_checkpoint_scene_rows": phase5_summary["number_of_checkpoint_scene_rows"],
        "sampling_rollouts_per_scene": phase5_summary["sampling_rollouts_per_scene"],
        "sampling_temperature": phase5_summary["sampling_temperature"],
        "sampling_seed": phase5_summary["sampling_seed"],
        "pass_at_k": list(config.required_pass_at_k),
        "informative_group_size": config.informative_group_size,
        "required_phase6_metrics": list(config.required_metrics),
        "constraint_aware_enabled": config.constraint_aware_enabled,
        "constraint_aware_arm_name": config.constraint_aware_arm_name,
        "constraint_aware_reward_mode": config.constraint_aware_reward_mode,
        "arms": arms,
        "phase5_policy_support_by_checkpoint": phase5_summary["by_checkpoint"],
        "scene_is_statistical_unit": True,
        "subjective_success_threshold_applied": False,
        "confirmatory_data_used": False,
        "training_invoked": False,
        "rl_invoked": False,
    }
    return manifest


def write_phase6_execution_manifest(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite a Phase 6 execution manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def verify_phase6_package_lock(
    *, lock_path: Path, repository_root: Path, expected_paths: tuple[str, ...]
) -> str:
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("Phase 6 package lock must be a regular file")
    payload = _load_mapping_file(lock_path)
    rows = payload.get("files") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status") != "FROZEN_PHASE_6_RL_SURFACE"
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("Phase 6 package lock is malformed")
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("Phase 6 package lock row is malformed")
        relative = row["path"]
        digest = row["sha256"]
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or relative in observed
        ):
            raise ValueError("Phase 6 package lock row has invalid fields")
        candidate = repository_root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"Phase 6 package lock missing file: {relative}")
        file_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if file_digest != digest:
            raise RuntimeError(f"Phase 6 package lock mismatch: {relative}")
        observed.add(relative)
    if observed != set(expected_paths):
        raise RuntimeError("Phase 6 package lock closure mismatch")
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


__all__ = [
    "Phase6Arm",
    "Phase6PlanConfig",
    "build_phase6_execution_manifest",
    "load_phase5_policy_support_summary",
    "load_phase6_config",
    "verify_phase6_package_lock",
    "write_phase6_execution_manifest",
]
