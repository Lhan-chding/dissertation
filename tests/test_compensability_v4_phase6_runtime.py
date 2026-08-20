from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from compensability_v4.qwen.phase6_runtime import (
    build_phase6_execution_manifest,
    load_phase5_policy_support_summary,
    load_phase6_config,
    load_phase6_execution_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _adapter_tree(root: Path, name: str) -> str:
    adapter = root / name / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(name.encode("utf-8"))
    digest = hashlib.sha256()
    for item in sorted(path for path in adapter.rglob("*") if path.is_file()):
        digest.update(item.relative_to(adapter).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _phase5_summary_payload(run_root: Path) -> dict[str, object]:
    c0_hash = _adapter_tree(run_root, "C0_format_only")
    c1_hash = _adapter_tree(run_root, "C1_forward_arithmetic")
    t_hash = _adapter_tree(run_root, "T_constraint_recovery")
    return {
        "schema_version": 1,
        "status": "PHASE_5_POLICY_SUPPORT_EXECUTED",
        "number_of_held_out_natural_errors": 32,
        "held_out_family_counts": {
            "cross_series": 16,
            "duplicate_encoding": 5,
            "trend": 11,
        },
        "number_of_checkpoint_scene_rows": 128,
        "sampling_rollouts_per_scene": 16,
        "sampling_temperature": 0.7,
        "sampling_seed": 2026082005,
        "pass_at_k": [1, 2, 4, 8, 16],
        "informative_group_size": 8,
        "by_checkpoint": {
            "Base": {
                "scene_count": 32,
                "mean_p_i": 0.25,
                "mean_G_K": 0.6,
                "pass_at_k": {"1": 0.25},
            },
            "C0": {"scene_count": 32, "mean_p_i": 0.4, "mean_G_K": 0.8, "pass_at_k": {"1": 0.4}},
            "C1": {"scene_count": 32, "mean_p_i": 0.5, "mean_G_K": 0.85, "pass_at_k": {"1": 0.5}},
            "T": {"scene_count": 32, "mean_p_i": 0.7, "mean_G_K": 0.9, "pass_at_k": {"1": 0.7}},
        },
        "scene_is_statistical_unit": True,
        "subjective_success_threshold_applied": False,
        "confirmatory_data_used": False,
        "training_invoked": False,
        "rl_invoked": False,
        "config_sha256": "b" * 64,
        "model_snapshot_sha256": "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87",
        "package_lock_sha256": "c" * 64,
        "source_sha256": {
            "Base": "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87",
            "C0": c0_hash,
            "C1": c1_hash,
            "T": t_hash,
            "support_dev": "d" * 64,
        },
        "support_dev_summary_sha256": "f" * 64,
    }


def test_phase6_config_freezes_required_arms_and_metrics() -> None:
    config = load_phase6_config(ROOT / "configs/recoverability/v4_phase_6.yaml")

    assert tuple(arm.name for arm in config.arms) == (
        "Base",
        "Base_AnswerOnly_RL",
        "Recovery_LoRA",
        "Recovery_LoRA_RecoveryOutcome_RL",
        "Recovery_LoRA_AnswerOnly_RL",
    )
    assert config.constraint_aware_enabled is False
    assert config.informative_group_size == 8
    assert "reward_variance" in config.required_metrics
    assert "exact_world_recovery" in config.required_metrics


def test_phase6_policy_support_summary_requires_phase5_executed_contract(tmp_path: Path) -> None:
    payload = _phase5_summary_payload(tmp_path / "runs")
    path = tmp_path / "informative_group_rate.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = load_phase5_policy_support_summary(path, expected_sha256=digest)
    assert loaded["number_of_held_out_natural_errors"] == 32

    payload["status"] = "NOT_EXECUTED"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="Phase 5"):
        load_phase5_policy_support_summary(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_phase6_execution_manifest_binds_phase5_summary_and_phase4_hashes(tmp_path: Path) -> None:
    config = load_phase6_config(ROOT / "configs/recoverability/v4_phase_6.yaml")
    run_root = tmp_path / "phase4-r1"
    summary = _phase5_summary_payload(run_root)
    summary_path = tmp_path / "informative_group_rate.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    summary_sha256 = hashlib.sha256(summary_path.read_bytes()).hexdigest()

    manifest = build_phase6_execution_manifest(
        config=config,
        phase5_summary=summary,
        phase5_summary_sha256=summary_sha256,
        phase4_run_root=run_root,
        config_sha256="1" * 64,
        package_lock_sha256="2" * 64,
    )

    assert manifest["status"] == "PHASE_6_RL_EXECUTION_MANIFEST_PREPARED"
    assert manifest["phase5_policy_support_summary_sha256"] == summary_sha256
    assert manifest["number_of_held_out_natural_errors"] == 32
    assert manifest["training_invoked"] is False
    assert manifest["rl_invoked"] is False
    assert [arm["name"] for arm in manifest["arms"]] == [
        "Base",
        "Base_AnswerOnly_RL",
        "Recovery_LoRA",
        "Recovery_LoRA_RecoveryOutcome_RL",
        "Recovery_LoRA_AnswerOnly_RL",
    ]
    recovery_lora = next(arm for arm in manifest["arms"] if arm["name"] == "Recovery_LoRA")
    assert recovery_lora["initialization_checkpoint"] == "T"
    assert recovery_lora["initialization_sha256"] == summary["source_sha256"]["T"]

    manifest_path = tmp_path / "execution_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    loaded = load_phase6_execution_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
        expected_config_sha256="1" * 64,
        expected_package_lock_sha256="2" * 64,
    )
    assert loaded["phase5_policy_support_summary_sha256"] == summary_sha256

    manifest["training_invoked"] = True
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="execution manifest"):
        load_phase6_execution_manifest(
            manifest_path,
            expected_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            expected_config_sha256="1" * 64,
            expected_package_lock_sha256="2" * 64,
        )


def test_phase6_execution_manifest_rejects_drifted_source_hash_mapping(tmp_path: Path) -> None:
    config = load_phase6_config(ROOT / "configs/recoverability/v4_phase_6.yaml")
    run_root = tmp_path / "phase4-r1"
    summary = _phase5_summary_payload(run_root)
    summary_path = tmp_path / "informative_group_rate.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    manifest = build_phase6_execution_manifest(
        config=config,
        phase5_summary=summary,
        phase5_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        phase4_run_root=run_root,
        config_sha256="1" * 64,
        package_lock_sha256="2" * 64,
    )
    manifest["source_sha256"] = {
        "Base": "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
    }
    manifest_path = tmp_path / "execution_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="source hashes"):
        load_phase6_execution_manifest(
            manifest_path,
            expected_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            expected_config_sha256="1" * 64,
            expected_package_lock_sha256="2" * 64,
        )
