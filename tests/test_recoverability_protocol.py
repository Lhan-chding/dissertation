from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from compbias.recoverability.config import load_recoverability_protocol
from compbias.recoverability.design import build_design_report
from compbias.recoverability.evidence import load_negative_pilot_record, verify_protocol_lock

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "recoverability" / "recoverability_v1.yaml"
NEGATIVE = ROOT / "configs" / "recoverability" / "v0_3_negative_pilot.yaml"
LOCK = ROOT / "configs" / "recoverability" / "protocol_lock_v1.yaml"


def test_negative_v0_3_pilot_is_final_failed_and_training_was_not_run() -> None:
    record = load_negative_pilot_record(NEGATIVE)

    assert record.status == "final_failed_preregistered_pilot"
    assert record.server_revision_observed == "c5e4c6f"
    assert record.model_snapshot_sha256 == (
        "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
    )
    assert record.records == 200
    assert record.answer_accuracy == 0.355
    assert record.parse_rate == 0.935
    assert record.natural_perception_error_rate == 0.225
    assert dict(record.error_counts) == {
        "compensated_visual_error": 1,
        "none": 67,
        "operator_invariant_visual_error": 3,
        "parse_failure": 13,
        "reasoning_error": 75,
        "visual_error": 41,
    }
    assert sum(record.error_counts.values()) == record.records
    assert record.gate_passed is False
    assert record.calibration_exit == 3
    assert record.original_pilot_a == "terminated_not_run"
    assert record.original_pilot_b == "terminated_not_run"
    assert set(record.required_server_artifacts) >= {
        "calibration_records_v0_3.jsonl",
        "calibration_records_v0_3.summary.json",
        "base-calibration-v0.3.log",
    }


def test_protocol_physically_separates_natural_bridge_and_causal_studies() -> None:
    protocol = load_recoverability_protocol(CONFIG)

    assert protocol.status == "PREREGISTERED_NOT_RUN"
    assert protocol.model_id == "Qwen2.5-VL-3B-Instruct"
    assert protocol.phase_n.dataset_id == "CVA-Natural-Prevalence-v1"
    assert protocol.bridge.dataset_id == "CVA-Recoverability-Bridge-v1"
    assert protocol.phase_c.dataset_id == "CVA-Recoverability-Causal-v1"
    assert (
        len(
            {
                protocol.phase_n.output_subdirectory,
                protocol.bridge.output_subdirectory,
                protocol.phase_c.output_subdirectory,
            }
        )
        == 3
    )
    assert (
        len(
            {
                protocol.phase_n.seed,
                protocol.bridge.seed,
                protocol.phase_c.seed,
            }
        )
        == 3
    )
    assert protocol.phase_n.source_protocol == "CVA-Chart-Pilot-v0.3"
    assert protocol.phase_n.max_format_retries == 0
    assert protocol.phase_n.allow_sample_extension is False
    assert protocol.phase_c.allow_quota_redistribution is False
    assert protocol.phase_c.allow_sample_extension is False


def test_design_counts_are_exact_and_forks_never_become_independent_n() -> None:
    report = build_design_report(load_recoverability_protocol(CONFIG))

    assert report.phase_n_scenes == 4000
    assert report.bridge_scenes == 300
    assert report.bridge_model_calls == 600
    assert report.phase_c_intake_scenes == 6000
    assert dict(report.selected_family_quotas) == {
        "cross_series": 267,
        "duplicate_encoding": 266,
        "trend": 267,
    }
    assert report.selected_independent_scenes == 800
    assert report.arms == (
        "ablated",
        "valid",
        "sham",
        "counterfactual",
        "oracle_perception",
        "operator_swap",
    )
    assert report.forks_per_arm == 8
    assert report.total_downstream_forks == 38_400
    assert report.confirmatory_forks == 25_600
    assert report.diagnostic_forks == 12_800
    assert report.independent_analysis_unit == "semantic_scene"


def test_protocol_lock_binds_frozen_v0_3_sources_without_modifying_them() -> None:
    result = verify_protocol_lock(LOCK, repository_root=ROOT)

    assert result.verified is True
    assert len(result.files) == 5
    assert all(len(item.sha256) == 64 for item in result.files)
    assert {item.relative_path for item in result.files} == {
        "configs/data/cva_chart_pilot_v0_3.yaml",
        "src/compbias/gpu_pilot/chart_data.py",
        "src/compbias/gpu_pilot/structured_generation.py",
        "src/compbias/gpu_pilot/taxonomy.py",
        "src/compbias/models/structured_parser.py",
    }


def test_protocol_rejects_adaptive_or_ambiguous_sample_plans(tmp_path: Path) -> None:
    source = CONFIG.read_text(encoding="utf-8")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        source.replace("allow_sample_extension: false", "allow_sample_extension: true", 1)
    )

    with pytest.raises(ValueError, match="sample extension"):
        load_recoverability_protocol(invalid)


def test_local_server_plan_is_metadata_only_and_cannot_execute_gpu() -> None:
    command = [
        sys.executable,
        str(ROOT / "experiments" / "recoverability_v1" / "03_build_server_plan.py"),
        "--config",
        str(CONFIG),
        "--protocol-lock",
        str(LOCK),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)

    assert payload["status"] == "PREREGISTERED_NOT_RUN"
    assert payload["gpu_invoked"] is False
    assert payload["training_invoked"] is False
    assert payload["server_execution_permitted"] is False
    assert payload["next_stage"] == "server_measurement_bridge"
    rejected = subprocess.run([*command, "--execute"], cwd=ROOT, capture_output=True, text=True)
    assert rejected.returncode != 0
