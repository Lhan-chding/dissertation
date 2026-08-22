from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from compensability_v5.study_c2.io import sha256_file
from compensability_v5.study_c2.report_runtime import preflight_report, run_report

ARMS = ("C2_answer_reward", "C2_exact_state_reward")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_evidence(root: Path) -> None:
    c2 = root / "artifacts/v5/study_c2"
    config = root / "configs/v5/study_c2_identifiable_reward.yaml"
    package_lock = root / "configs/v5/server_package_lock.yaml"
    stage24_contract = c2 / "stage24_execution_contract.json"
    stage25_contract = c2 / "stage25_execution_contract.json"
    _write_text(config, "schema_version: 2\n")
    _write_text(package_lock, "schema_version: 1\n")
    _write_json(stage24_contract, {"schema_version": 2, "stage": 24})
    _write_json(stage25_contract, {"schema_version": 2, "stage": 25})
    config_sha = sha256_file(config)
    lock_sha = sha256_file(package_lock)
    fiber_sha = "f" * 64
    b3_sha = "b" * 64

    support = c2 / "frozen_policy_support"
    _write_text(support / "raw_rows.jsonl", '{"kind":"U"}\n')
    _write_json(
        support / "summary.json",
        {
            "schema_version": 2,
            "status": "REWARD_CONTRAST_IDENTIFIED",
            "rollout_count": 6144,
            "counts": {"F": 655, "S": 635, "U": 4711, "X": 143},
            "k_selection": {"selected_k": 8, "efficiency_by_k": {"8": 0.1}},
            "gpu_invoked": True,
            "per_scene": [{"scene_id": "support-0"}],
        },
    )
    _write_json(
        support / "manifest.json",
        {
            "schema_version": 2,
            "status": "STUDY_C2_FROZEN_SUPPORT_COMPLETE",
            "prompt_count": 96,
            "rollout_count": 6144,
            "rollouts_per_prompt": 64,
            "raw_rows_sha256": sha256_file(support / "raw_rows.jsonl"),
            "summary_sha256": sha256_file(support / "summary.json"),
            "config_sha256": config_sha,
            "package_lock_sha256": lock_sha,
            "fiber_rows_sha256": fiber_sha,
            "b3_adapter_sha256": b3_sha,
            "gpu_invoked": True,
            "training_invoked": False,
            "rl_invoked": False,
        },
    )

    gradient = c2 / "shared_gradient_audit"
    _write_text(gradient / "per_group.jsonl", '{"group_index":0}\n')
    _write_json(
        gradient / "summary.json",
        {
            "schema_version": 2,
            "status": "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED",
            "continue_to_main_rl": True,
            "group_count": 768,
            "group_size": 8,
            "rollout_count": 6144,
            "reward_hamming_distance": 635,
            "reward_hamming_rate": 635 / 6144,
            "ESGR_group_count": 56,
            "RDGR_group_count": 356,
            "gradient_difference_norm_mean": 41.52615734760021,
            "gradient_difference_norm_max": 271.4385559212199,
            "gradient_cosine_mean": 0.5808503836499084,
            "gpu_invoked": True,
            "optimizer_step_invoked": False,
            "training_invoked": False,
            "rl_invoked": False,
        },
    )
    _write_json(
        gradient / "manifest.json",
        {
            "schema_version": 2,
            "status": "STUDY_C2_SHARED_GRADIENT_AUDIT_COMPLETE",
            "scientific_status": "STUDY_C2_SHARED_GRADIENT_CONTRAST_IDENTIFIED",
            "continue_to_main_rl": True,
            "group_count": 768,
            "group_size": 8,
            "rollout_count": 6144,
            "per_group_sha256": sha256_file(gradient / "per_group.jsonl"),
            "summary_sha256": sha256_file(gradient / "summary.json"),
            "support_manifest_sha256": sha256_file(support / "manifest.json"),
            "support_raw_rows_sha256": sha256_file(support / "raw_rows.jsonl"),
            "support_summary_sha256": sha256_file(support / "summary.json"),
            "execution_contract_sha256": sha256_file(stage24_contract),
            "config_sha256": config_sha,
            "package_lock_sha256": lock_sha,
            "fiber_rows_sha256": fiber_sha,
            "b3_adapter_sha256": b3_sha,
            "gpu_invoked": True,
            "optimizer_step_invoked": False,
            "training_invoked": False,
            "rl_invoked": False,
        },
    )

    training = c2 / "training"
    pair_arms: dict[str, object] = {}
    arm_manifest_hashes: dict[str, str] = {}
    for index, arm in enumerate(ARMS):
        arm_root = training / arm
        reward_id = "answer_reward_v1" if index == 0 else "exact_state_reward_v1"
        _write_json(arm_root / "arm_config.json", {"name": arm, "reward_function_id": reward_id})
        _write_text(arm_root / "raw_reward_trace.jsonl", json.dumps({"arm": arm}) + "\n")
        _write_text(arm_root / "group_diagnostics.jsonl", json.dumps({"arm": arm}) + "\n")
        _write_json(arm_root / "trainer_log_history.json", [{"step": 192}])
        _write_json(
            arm_root / "summary.json",
            {
                "schema_version": 2,
                "status": "STUDY_C2_ARM_TRAINING_SUMMARIZED",
                "arm": arm,
                "reward_function_id": reward_id,
                "training_prompt_count": 192,
                "optimizer_steps": 192,
                "rollout_count": 1536,
                "group_size": 8,
                "counts": {"F": 1, "S": 2, "U": 3, "X": 4},
            },
        )
        final_adapter_sha = str(index + 1) * 64
        manifest = {
            "schema_version": 2,
            "status": "STUDY_C2_ARM_TRAINING_COMPLETE",
            "arm": arm,
            "reward_function_id": reward_id,
            "training_prompt_count": 192,
            "expected_optimizer_steps": 192,
            "group_size": 8,
            "matched_pair_count": 96,
            "arm_config_sha256": sha256_file(arm_root / "arm_config.json"),
            "raw_reward_trace_sha256": sha256_file(arm_root / "raw_reward_trace.jsonl"),
            "group_diagnostics_sha256": sha256_file(arm_root / "group_diagnostics.jsonl"),
            "summary_sha256": sha256_file(arm_root / "summary.json"),
            "trainer_log_sha256": sha256_file(arm_root / "trainer_log_history.json"),
            "final_adapter_sha256": final_adapter_sha,
            "stage24_manifest_sha256": sha256_file(gradient / "manifest.json"),
            "stage24_per_group_sha256": sha256_file(gradient / "per_group.jsonl"),
            "stage24_summary_sha256": sha256_file(gradient / "summary.json"),
            "execution_contract_sha256": sha256_file(stage25_contract),
            "config_sha256": config_sha,
            "package_lock_sha256": lock_sha,
            "fiber_rows_sha256": fiber_sha,
            "b3_adapter_sha256": b3_sha,
            "model_snapshot_sha256": "e" * 64,
            "reward_only_pair_verified": True,
            "gpu_invoked": True,
            "optimizer_step_invoked": True,
            "training_invoked": True,
            "rl_invoked": True,
        }
        _write_json(arm_root / "manifest.json", manifest)
        manifest_sha = sha256_file(arm_root / "manifest.json")
        arm_manifest_hashes[arm] = manifest_sha
        pair_arms[arm] = {
            "manifest_sha256": manifest_sha,
            "final_adapter_sha256": final_adapter_sha,
            "raw_reward_trace_sha256": manifest["raw_reward_trace_sha256"],
        }
    _write_json(
        training / "manifest.json",
        {
            "schema_version": 2,
            "status": "STUDY_C2_TWO_ARM_TRAINING_COMPLETE",
            "arms": pair_arms,
            "training_prompt_count_per_arm": 192,
            "optimizer_steps_per_arm": 192,
            "reward_only_pair_verified": True,
            "gpu_invoked": True,
            "training_invoked": True,
            "rl_invoked": True,
        },
    )

    evaluation = c2 / "evaluation"
    _write_text(evaluation / "raw_rows.jsonl", '{"arm":"C2_answer_reward"}\n')
    _write_json(
        evaluation / "summary.json",
        {
            "schema_version": 2,
            "status": "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE",
            "evaluation_pair_count": 88,
            "evaluation_scene_count": 176,
            "sampled_rollouts": 16,
            "raw_row_count": 5632,
            "by_arm": {
                "C2_answer_reward": {
                    "answer_mean": 0.42,
                    "exact_mean": 0.11,
                    "parse_rate_mean": 0.70,
                    "scene_count": 176,
                },
                "C2_exact_state_reward": {
                    "answer_mean": 0.37,
                    "exact_mean": 0.16,
                    "parse_rate_mean": 0.56,
                    "scene_count": 176,
                },
            },
            "pair_bootstrap": {
                "estimate": -0.009,
                "pair_count": 88,
                "bootstrap_resamples": 10000,
                "bootstrap_seed": 2026082403,
                "bootstrap_95_ci": [-0.033, 0.014],
            },
            "training_pair_manifest_sha256": sha256_file(training / "manifest.json"),
            "config_sha256": config_sha,
            "package_lock_sha256": lock_sha,
            "fiber_rows_sha256": fiber_sha,
            "b3_adapter_sha256": b3_sha,
            "model_snapshot_sha256": "e" * 64,
            "reward_only_pair_verified": True,
            "gpu_invoked": True,
            "optimizer_step_invoked": False,
            "training_invoked": False,
            "rl_invoked": False,
        },
    )
    _write_json(
        evaluation / "manifest.json",
        {
            "schema_version": 2,
            "status": "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE",
            "evaluation_pair_count": 88,
            "evaluation_scene_count": 176,
            "sampled_rollouts": 16,
            "raw_row_count": 5632,
            "raw_rows_sha256": sha256_file(evaluation / "raw_rows.jsonl"),
            "summary_sha256": sha256_file(evaluation / "summary.json"),
            "training_pair_manifest_sha256": sha256_file(training / "manifest.json"),
            "arm_manifests": {
                arm: {
                    "manifest_sha256": arm_manifest_hashes[arm],
                    "final_adapter_sha256": pair_arms[arm]["final_adapter_sha256"],
                }
                for arm in ARMS
            },
            "config_sha256": config_sha,
            "package_lock_sha256": lock_sha,
            "fiber_rows_sha256": fiber_sha,
            "b3_adapter_sha256": b3_sha,
            "model_snapshot_sha256": "e" * 64,
            "reward_only_pair_verified": True,
            "gpu_invoked": True,
            "optimizer_step_invoked": False,
            "training_invoked": False,
            "rl_invoked": False,
        },
    )


def test_report_preflight_binds_stage23_through_stage26_without_gpu(tmp_path: Path) -> None:
    _build_evidence(tmp_path)

    payload = preflight_report(evidence_root=tmp_path)

    assert payload["status"] == "STUDY_C2_IDENTIFIABLE_REWARD_GRPO_REPORT_PREFLIGHT_OK"
    assert payload["source_file_count"] == 26
    assert payload["gpu_invoked"] is False
    assert payload["optimizer_step_invoked"] is False
    assert payload["training_invoked"] is False
    assert payload["rl_invoked"] is False
    assert payload["upstream_statuses"] == {
        "stage23": "STUDY_C2_FROZEN_SUPPORT_COMPLETE",
        "stage24": "STUDY_C2_SHARED_GRADIENT_AUDIT_COMPLETE",
        "stage25": "STUDY_C2_TWO_ARM_TRAINING_COMPLETE",
        "stage26": "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE",
    }


def test_report_write_is_fact_only_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _build_evidence(evidence)
    first = tmp_path / "first"
    second = tmp_path / "second"

    result_one = run_report(evidence_root=evidence, output_root=first)
    result_two = run_report(evidence_root=evidence, output_root=second)

    expected = {
        "STUDY_C2_IDENTIFIABLE_REWARD_GRPO_FACTS.md",
        "sha256_manifest.json",
        "study_c2_identifiable_reward_grpo_evidence.tar.gz",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert result_one["outputs"] == result_two["outputs"]
    markdown = (first / "STUDY_C2_IDENTIFIABLE_REWARD_GRPO_FACTS.md").read_text()
    assert "STUDY_C2_POST_TRAINING_EVALUATION_COMPLETE" in markdown
    assert '"estimate": -0.009' in markdown
    assert "recommend" not in markdown.casefold()
    with tarfile.open(first / "study_c2_identifiable_reward_grpo_evidence.tar.gz") as archive:
        members = set(archive.getnames())
    assert "artifacts/v5/study_c2/evaluation/raw_rows.jsonl" in members
    assert "artifacts/v5/study_c2/training/C2_answer_reward/raw_reward_trace.jsonl" in members
    assert "artifacts/v5/study_c2/training/C2_answer_reward/group_diagnostics.jsonl" in members
    assert "artifacts/v5/study_c2/training/C2_answer_reward/trainer_log_history.json" in members
    assert not any("adapter_model" in member or "checkpoint" in member for member in members)

    with pytest.raises(FileExistsError, match="overwrite forbidden"):
        run_report(evidence_root=evidence, output_root=first)


def test_report_fails_closed_on_upstream_hash_or_status_drift(tmp_path: Path) -> None:
    _build_evidence(tmp_path)
    raw_rows = tmp_path / "artifacts/v5/study_c2/evaluation/raw_rows.jsonl"
    raw_rows.write_text('{"drifted":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Stage 26 raw rows SHA-256 drifted"):
        preflight_report(evidence_root=tmp_path)

    _build_evidence(tmp_path)
    summary = tmp_path / "artifacts/v5/study_c2/evaluation/summary.json"
    payload = json.loads(summary.read_text())
    payload["status"] = "DRIFTED"
    _write_json(summary, payload)
    manifest = tmp_path / "artifacts/v5/study_c2/evaluation/manifest.json"
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["summary_sha256"] = sha256_file(summary)
    _write_json(manifest, manifest_payload)
    with pytest.raises(ValueError, match="Stage 26 summary drifted"):
        preflight_report(evidence_root=tmp_path)


def test_report_rejects_symlinked_evidence(tmp_path: Path) -> None:
    _build_evidence(tmp_path)
    raw_rows = tmp_path / "artifacts/v5/study_c2/evaluation/raw_rows.jsonl"
    target = tmp_path / "outside.jsonl"
    target.write_text(raw_rows.read_text(), encoding="utf-8")
    raw_rows.unlink()
    raw_rows.symlink_to(target)

    with pytest.raises(ValueError, match="unsafe evidence symlink"):
        preflight_report(evidence_root=tmp_path)


def test_report_rejects_symlinked_roots_and_sensitive_values(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    _build_evidence(evidence)
    linked_evidence = tmp_path / "linked-evidence"
    linked_evidence.symlink_to(evidence, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe symlink component in evidence root"):
        preflight_report(evidence_root=linked_evidence)

    output_target = tmp_path / "output-target"
    output_target.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(output_target, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe symlink component in Stage 27 output parent"):
        run_report(evidence_root=evidence, output_root=output_link / "report")

    raw_rows = evidence / "artifacts/v5/study_c2/evaluation/raw_rows.jsonl"
    dummy_secret = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    raw_rows.write_text(json.dumps({"api_key": dummy_secret}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sensitive-value pattern"):
        preflight_report(evidence_root=evidence)


def test_report_rejects_stage26_final_adapter_drift(tmp_path: Path) -> None:
    _build_evidence(tmp_path)
    manifest = tmp_path / "artifacts/v5/study_c2/evaluation/manifest.json"
    payload = json.loads(manifest.read_text())
    payload["arm_manifests"]["C2_answer_reward"]["final_adapter_sha256"] = "a" * 64
    _write_json(manifest, payload)

    with pytest.raises(ValueError, match="final adapter provenance drifted"):
        preflight_report(evidence_root=tmp_path)
