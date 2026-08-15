from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import json

from compbias.recoverability import measurement_qualification_result as result_module
from compbias.recoverability.measurement_qualification_result import (
    load_measurement_qualification_frozen_result,
    verify_measurement_qualification_result_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_RESULT = ROOT / "configs/recoverability/measurement_qualification_frozen_result.yaml"


def test_measurement_qualification_result_is_frozen_from_server_evidence() -> None:
    result = load_measurement_qualification_frozen_result(FROZEN_RESULT)

    assert result.status == "FINAL_PASSED_MEASUREMENT_QUALIFICATION_DO_NOT_RERUN"
    assert result.qualification_exit == 0
    assert result.qualification_passed is True
    assert result.scenes == 300
    assert result.model_calls == 599
    assert result.stage1_parse_successes == 299
    assert result.stage1_parse_rate == 0.9966666666666667
    assert result.stage1_parse_lower == 0.9842854451084161
    assert result.exact_transcription_rate == 0.8533333333333334
    assert result.stage2_program_parse_successes == 299
    assert result.stage2_execution_successes == 299
    assert result.executor_answer_successes == 299
    assert result.stage2_program_parse_rate == 1.0
    assert result.stage2_execution_rate == 1.0
    assert result.executor_answer_accuracy == 1.0
    assert result.stage2_program_parse_lower == 0.9900308532071007
    assert result.stage2_execution_lower == 0.9900308532071007
    assert result.executor_answer_lower == 0.9900308532071007
    assert result.gate_failures == ()
    assert result.format_retries == 0
    assert result.hypothesis_tested is False
    assert result.confirmatory_execution_authorized is False
    assert result.training_invoked is False
    assert dict(result.source_sha256) == {
        "attempt_marker": "d8957deaa71283db638c7b644c51b69f0182843dbe447d3ba04750d5cb45e190",
        "console": "2513a50896e863cd2d294f5b5cb9fde06c0934dda4290cd70551dbeb131d4311",
        "preflight": "784149ae5482f4d1b7b31a2e22ea71cb8d0c04fc7441a5f6f07004a8e2328ffe",
        "qualification_records": (
            "55d5eea889ed1733bc58d75f5783ae0a34e00e827361ac20ccfcee4351f18e4d"
        ),
        "qualification_report": (
            "11a2d7f44d7fdf115954baacbac1b6c53269a8e00368c2b78f27ba53c24c640e"
        ),
    }


def test_measurement_qualification_artifacts_are_bound_before_reuse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frozen = load_measurement_qualification_frozen_result(FROZEN_RESULT)
    preflight = tmp_path / "preflight.json"
    marker = tmp_path / "attempt.json"
    report = tmp_path / "report.json"
    records = tmp_path / "records.jsonl"
    console = tmp_path / "console.log"
    preflight.write_text("{}\n", encoding="utf-8")
    marker.write_text(
        json.dumps(
            {
                "status": "MEASUREMENT_QUALIFICATION_STARTED",
                "model_snapshot_sha256": frozen.model_snapshot_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = asdict(frozen)
    for key in ("status", "qualification_exit", "source_sha256"):
        payload.pop(key)
    payload["gate_failures"] = []
    report.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    records.write_text("{}\n" * 300, encoding="utf-8")
    console.write_text("measurement_qualification_exit=0\n", encoding="utf-8")
    digest_by_name = {
        "preflight.json": dict(frozen.source_sha256)["preflight"],
        "attempt.json": dict(frozen.source_sha256)["attempt_marker"],
        "report.json": dict(frozen.source_sha256)["qualification_report"],
        "records.jsonl": dict(frozen.source_sha256)["qualification_records"],
        "console.log": dict(frozen.source_sha256)["console"],
    }
    monkeypatch.setattr(result_module, "_sha256", lambda path: digest_by_name[path.name])

    verified = verify_measurement_qualification_result_artifacts(
        frozen,
        preflight=preflight,
        attempt_marker=marker,
        report=report,
        records=records,
        console_log=console,
    )

    assert verified == frozen
