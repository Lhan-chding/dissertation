from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from compbias.gpu_pilot.structured_generation import validate_pilot_trajectory
from compbias.gpu_pilot.taxonomy import natural_error_type
from compbias.models.structured_parser import parse_trajectory
from compbias.recoverability.evidence_capture import capture_v03_evidence

ROOT = Path(__file__).resolve().parents[1]
NEGATIVE_PILOT = ROOT / "configs" / "recoverability" / "v0_3_negative_pilot.yaml"


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    records = tmp_path / "calibration_records_v0_3.jsonl"
    categories = (
        ["none"] * 67
        + ["parse_failure"] * 13
        + ["reasoning_error"] * 75
        + ["visual_error"] * 41
        + ["compensated_visual_error"]
        + ["operator_invariant_visual_error"] * 3
    )

    def row(index: int, category: str) -> dict[str, object]:
        sample_id = f"calibration-{index:06d}"
        truth = [8, 4, 5, 9]
        perceived = {
            "none": truth,
            "reasoning_error": truth,
            "visual_error": [7, 4, 5, 9],
            "compensated_visual_error": [7, 4, 5, 9],
            "operator_invariant_visual_error": [9, 5, 5, 9],
        }.get(category)
        answer = {
            "none": 4,
            "reasoning_error": 5,
            "visual_error": 3,
            "compensated_visual_error": 4,
            "operator_invariant_visual_error": 4,
        }.get(category)
        raw = "malformed"
        if perceived is not None and answer is not None:
            raw = (
                f'<perception>{{"values":{json.dumps(perceived)}}}</perception>'
                '<reasoning>{"operation":"difference"}</reasoning>'
                f"<answer>{answer}</answer>"
            )
        parsed = validate_pilot_trajectory(
            parse_trajectory(raw, sample_id=sample_id),
            operation="difference",
            expected_value_count=4,
        )
        source = {
            "schema_version": 1,
            "dataset_id": "CVA-Chart-Pilot-v0.3",
            "sample_id": sample_id,
            "split": "calibration",
            "chart_type": "grouped_bar",
            "operation": "difference",
            "values": truth,
            "question": "What is the first value minus the second value?",
            "answer": 4,
            "image": f"images/{sample_id}.png",
            "mechanism": "iid",
        }
        derived = natural_error_type(source, parsed)
        assert derived == category
        return {
            **source,
            "rollout_id": f"calibration-rollout-{index:06d}",
            "raw_text": raw,
            "parsed": parsed.to_mapping(),
            "format_attempts": [
                {
                    "attempt_index": 0,
                    "raw_text": raw,
                    "status": parsed.status.value,
                    "error_code": parsed.error_code,
                }
            ],
            "format_retries": 0,
            "reward": int(
                category in {"none", "compensated_visual_error", "operator_invariant_visual_error"}
            ),
            "error_type": category,
        }

    records.write_text(
        "".join(
            json.dumps(row(index, category), sort_keys=True) + "\n"
            for index, category in enumerate(categories)
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "calibration_records_v0_3.summary.json"
    summary.write_text(
        json.dumps(
            {
                "answer_accuracy": 0.355,
                "error_counts": {
                    "compensated_visual_error": 1,
                    "none": 67,
                    "operator_invariant_visual_error": 3,
                    "parse_failure": 13,
                    "reasoning_error": 75,
                    "visual_error": 41,
                },
                "gate_failures": [
                    "evidence_parse_rate_below_95_percent",
                    "fewer_than_three_supported_natural_error_families",
                ],
                "gate_passed": False,
                "model_snapshot_sha256": (
                    "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"
                ),
                "natural_perception_error_rate": 0.225,
                "parse_rate": 0.935,
                "records": 200,
                "schema_version": 1,
                "split": "calibration",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pilot_log = tmp_path / "pilot-data-v0.3.log"
    pilot_log.write_text("dataset generation completed\n", encoding="utf-8")
    calibration_log = tmp_path / "base-calibration-v0.3.log"
    calibration_log.write_text(
        'Loading weights: 100%\n{"gate_passed": false, "records": 200}\n',
        encoding="utf-8",
    )
    return {
        "records_path": records,
        "summary_path": summary,
        "pilot_data_log_path": pilot_log,
        "calibration_log_path": calibration_log,
    }


def test_capture_binds_external_bytes_and_fixed_failed_metrics(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "v0_3_external_evidence.json"

    report = capture_v03_evidence(
        negative_pilot_path=NEGATIVE_PILOT,
        output_path=output,
        **inputs,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert report.verified is True
    assert report.records == 200
    assert report.calibration_exit == 3
    assert report.gate_passed is False
    assert payload["artifact_type"] == "recoverability_v1_v0_3_external_evidence"
    assert payload["status"] == "FROZEN_FAILED_NOT_TO_BE_RERUN"
    assert payload["calibration_exit_evidence"] == (
        "replayed_raw_records_and_frozen_calibration_gate"
    )
    assert payload["calibration_gate_failures"] == [
        "evidence_parse_rate_below_95_percent",
        "fewer_than_three_supported_natural_error_families",
    ]
    assert payload["source_files"] == [
        {
            "basename": item.basename,
            "bytes": item.bytes,
            "sha256": item.sha256,
        }
        for item in report.source_files
    ]
    assert {item["basename"] for item in payload["source_files"]} == {
        "base-calibration-v0.3.log",
        "calibration_records_v0_3.jsonl",
        "calibration_records_v0_3.summary.json",
        "pilot-data-v0.3.log",
    }


def test_capture_rejects_metric_tamper_and_does_not_publish(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    summary = json.loads(inputs["summary_path"].read_text(encoding="utf-8"))
    summary["parse_rate"] = 0.94
    inputs["summary_path"].write_text(json.dumps(summary) + "\n", encoding="utf-8")
    output = tmp_path / "evidence.json"

    with pytest.raises(ValueError, match="parse_rate"):
        capture_v03_evidence(
            negative_pilot_path=NEGATIVE_PILOT,
            output_path=output,
            **inputs,
        )

    assert not output.exists()


def test_capture_rejects_gate_failure_tamper(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    summary = json.loads(inputs["summary_path"].read_text(encoding="utf-8"))
    summary["gate_failures"] = []
    inputs["summary_path"].write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="gate_failures"):
        capture_v03_evidence(
            negative_pilot_path=NEGATIVE_PILOT,
            output_path=tmp_path / "evidence.json",
            **inputs,
        )


def test_capture_rejects_empty_calibration_log(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    inputs["calibration_log_path"].write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="calibration log must not be empty"):
        capture_v03_evidence(
            negative_pilot_path=NEGATIVE_PILOT,
            output_path=tmp_path / "evidence.json",
            **inputs,
        )


def test_capture_rejects_sample_id_only_records_even_with_matching_claimed_summary(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    inputs["records_path"].write_text(
        "".join(
            json.dumps({"sample_id": f"calibration-{index:06d}"}) + "\n" for index in range(200)
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="full dataset source"):
        capture_v03_evidence(
            negative_pilot_path=NEGATIVE_PILOT,
            output_path=tmp_path / "evidence.json",
            **inputs,
        )


def test_capture_rejects_bad_record_count_symlinks_and_overwrite(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    first_record = inputs["records_path"].read_text(encoding="utf-8").splitlines()[0]
    inputs["records_path"].write_text(first_record + "\n", encoding="utf-8")
    output = tmp_path / "evidence.json"
    with pytest.raises(ValueError, match="exactly 200"):
        capture_v03_evidence(
            negative_pilot_path=NEGATIVE_PILOT,
            output_path=output,
            **inputs,
        )

    inputs = _write_inputs(tmp_path)
    link_directory = tmp_path / "links"
    link_directory.mkdir()
    linked_log = link_directory / "pilot-data-v0.3.log"
    linked_log.symlink_to(inputs["pilot_data_log_path"])
    inputs["pilot_data_log_path"] = linked_log
    with pytest.raises(ValueError, match="regular file"):
        capture_v03_evidence(
            negative_pilot_path=NEGATIVE_PILOT,
            output_path=output,
            **inputs,
        )

    inputs["pilot_data_log_path"] = tmp_path / "pilot-data-v0.3.log"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        capture_v03_evidence(
            negative_pilot_path=NEGATIVE_PILOT,
            output_path=output,
            **inputs,
        )


def test_capture_cli_is_metadata_only_and_bridge_is_step_three() -> None:
    capture_script = ROOT / "experiments" / "recoverability_v1" / "02_capture_v03_evidence.py"
    bridge_script = ROOT / "experiments" / "recoverability_v1" / "03_bridge.py"
    help_result = subprocess.run(
        [sys.executable, str(capture_script), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = capture_script.read_text(encoding="utf-8")

    assert bridge_script.is_file()
    assert "--records" in help_result.stdout
    assert "--summary" in help_result.stdout
    assert "--output" in help_result.stdout
    assert "import torch" not in source
    assert "from_pretrained" not in source
