"""RED acceptance contracts for Phase 0 and server-only execution gates."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from compensability_v4.diagnostics.legacy_audit import build_legacy_audit

ROOT = Path(__file__).resolve().parents[1]


def test_phase_zero_audit_uses_narrow_frozen_evidence_statuses() -> None:
    audit = build_legacy_audit(ROOT)
    registry = {row["experiment_id"]: row for row in audit.registry_rows}
    claims = {row["claim_id"]: row for row in audit.claim_rows}

    assert (
        registry["phase_c_v3_dsl"]["registered_interpretation"] == "measurement_interface_failure"
    )
    assert registry["phase_c_v3_dsl"]["true_world_recoveries"] == ""
    assert registry["qwen_world_only_valid_cue_50"]["interface_family"] == "text_replay"
    assert registry["qwen_world_only_valid_cue_50"]["true_world_recoveries"] == "0"
    assert (
        registry["qwen_world_only_no_cue_100"]["evidence_status"]
        == "awaiting_hash_bound_server_evidence"
    )
    assert registry["qwen_world_only_no_cue_100"]["model_calls"] == "100"
    assert claims["qwen_world_only_copying"]["status"] == "allowed"
    assert claims["qwen_natural_visual_state_absence"]["status"] == "forbidden"


def test_phase_zero_cli_writes_exact_required_artifacts(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/v4/00_audit_legacy.py", "--artifact-root", str(tmp_path)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = tmp_path / "v4" / "audit"
    assert {path.name for path in output.iterdir()} == {
        "legacy_experiment_registry.csv",
        "claim_evidence_matrix.csv",
        "scoring_contract.md",
        "legacy_hash_manifest.json",
    }
    with (output / "legacy_experiment_registry.csv").open(newline="", encoding="utf-8") as stream:
        assert any(row["experiment_id"] == "phase_c_v3_dsl" for row in csv.DictReader(stream))
    manifest = json.loads((output / "legacy_hash_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert all(len(row["sha256"]) == 64 for row in manifest["inputs"])


@pytest.mark.parametrize(
    "script",
    [
        "01_introspect_qwen.py",
        "02_run_capability_chain.py",
        "03_score_candidates.py",
        "04_layerwise_assimilation.py",
        "05_validate_cache_runner.py",
        "06_run_interface_ladder.py",
    ],
)
def test_server_phase_scripts_are_inert_without_execute(script: str) -> None:
    result = subprocess.run(
        [sys.executable, f"scripts/v4/{script}"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "BLOCKED" in result.stdout


def test_v4_runtime_config_forbids_training_and_rl() -> None:
    text = (ROOT / "configs/recoverability/v4_phase_0_3.yaml").read_text(encoding="utf-8")
    assert "model_loading_allowed: true" in text
    assert "training_authorized: false" in text
    assert "rl_authorized: false" in text
    assert "resized_height: 280" in text
    assert "resized_width: 280" in text


def test_v4_research_config_forbids_subjective_success_thresholds() -> None:
    text = (ROOT / "configs/recoverability/v4_phase_0_3.yaml").read_text(encoding="utf-8")

    assert "subjective_success_thresholds_forbidden: true" in text
    assert "report_scene_clustered_confidence_intervals: true" in text
    assert "report_paired_and_family_stratified_effects: true" in text
    assert "report_policy_support_and_reward_variance: true" in text
    assert "minimum_visual_repair_rate" not in text
    assert "minimum_recovery_accuracy" not in text


def test_phase_four_training_surface_is_guarded() -> None:
    phase_zero_to_three = (ROOT / "configs/recoverability/v4_phase_0_3.yaml").read_text(
        encoding="utf-8"
    )
    assert "training_authorized: false" in phase_zero_to_three
    phase_four = (ROOT / "configs/recoverability/v4_phase_4.yaml").read_text(encoding="utf-8")
    assert "training_authorized: true" in phase_four
    assert "rl_authorized: false" in phase_four
    assert "require_explicit_gpu_acknowledgement: true" in phase_four
    assert (ROOT / "scripts/v4/07_build_support_data.py").is_file()
    assert (ROOT / "scripts/v4/08_train_phase4_lora.py").is_file()
