from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from compbias.recoverability.bridge_v1_failure import (
    BridgeV1FailureRecord,
    load_bridge_v1_failure,
    replay_bridge_v1_records,
    verify_bridge_v1_failure_artifacts,
)
from compbias.recoverability.stage1_v2 import (
    validate_stage1_v2_runtime_paths,
    verify_stage1_v2_server_package_lock,
)

ROOT = Path(__file__).resolve().parents[1]
FAILURE_CONFIG = ROOT / "configs" / "recoverability" / "bridge_v1_failure.yaml"
SERVER_LOCK = ROOT / "configs" / "recoverability" / "server_package_lock_stage1_v2.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_failure_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    valid = '{"target_facts":[3,7,5,2],"redundant_facts":[],"axis_facts":["integer_ticks"]}'
    raw_outputs = (
        f"```json\n{valid}\n```",
        '{"target_facts":[3,7],"redundant_facts":[],"axis_facts":["integer_ticks"]}',
        '{"target_facts":[3,7,5,2],"redundant_ffacts":[],"axis_facts":["integer_ticks"]}',
    )
    records = tmp_path / "bridge_records.jsonl"
    with records.open("x", encoding="utf-8") as stream:
        for index, raw in enumerate(raw_outputs):
            stream.write(
                json.dumps(
                    {
                        "scene_id": f"iid_test-{index:06d}",
                        "stage1_raw": raw,
                        "stage1_parse_success": False,
                        "stage2_raw": None,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    report = tmp_path / "bridge_report.json"
    _write_json(
        report,
        {
            "artifact_type": "recoverability_v1_bridge_report",
            "dataset_id": "CVA-Recoverability-Bridge-v1",
            "model_snapshot_sha256": "a" * 64,
            "scenes": 3,
            "model_calls": 6,
            "legacy_parse_rate": 0.96,
            "legacy_answer_accuracy": 0.40,
            "legacy_perception_error_rate": 0.24,
            "stage1_parse_rate": 0.0,
            "stage2_program_parse_rate": 0.0,
            "program_answer_consistency": 0.0,
            "two_stage_answer_accuracy": 0.0,
            "protocols_mergeable": False,
            "training_invoked": False,
        },
    )
    diagnostic = tmp_path / "bridge-stage1-diagnostic.json"
    _write_json(
        diagnostic,
        {
            "artifact_type": "recoverability_v1_stage1_diagnostic",
            "records": 3,
            "source_records_sha256": _sha256(records),
            "replayed_stage1_parse_successes": 0,
            "stage2_invocations": 0,
            "parse_error_counts": {
                "ValueError: Stage-1 evidence schema is invalid": 1,
                "ValueError: Stage-1 output must be one exact JSON object": 1,
                "ValueError: target_facts must contain exactly four integers": 1,
            },
            "raw_signature_counts": {
                "json_object_keys:axis_facts,redundant_facts,target_facts": 1,
                "json_object_keys:axis_facts,redundant_ffacts,target_facts": 1,
                "markdown_fence": 1,
            },
        },
    )
    attempted = tmp_path / "attempted.json"
    _write_json(attempted, {"schema_version": 1, "status": "BRIDGE_ATTEMPT_STARTED"})
    console = tmp_path / "bridge-console.log"
    console.write_text("bridge_exit=3\n", encoding="utf-8")
    return records, report, diagnostic, attempted, console


def test_bridge_v1_failure_is_frozen_as_interface_failure_not_hypothesis_result() -> None:
    failure = load_bridge_v1_failure(FAILURE_CONFIG)

    assert failure.status == "FINAL_FAILED_STAGE1_INTERFACE"
    assert failure.bridge_exit == 3
    assert failure.scenes == 300
    assert failure.model_calls == 600
    assert failure.legacy_parse_rate == pytest.approx(0.9633333333333334)
    assert failure.legacy_answer_accuracy == pytest.approx(0.39666666666666667)
    assert failure.stage1_parse_rate == 0.0
    assert failure.stage2_invocations == 0
    assert failure.hypothesis_tested is False
    assert failure.training_invoked is False
    assert dict(failure.parse_error_counts) == {
        "schema_invalid": 8,
        "not_exact_json_object": 100,
        "target_facts_not_four_integers": 192,
    }
    assert all(len(digest) == 64 for _label, digest in failure.source_sha256)


def test_bridge_v1_failure_replay_recomputes_strict_taxonomy(tmp_path: Path) -> None:
    records, *_ = _write_failure_artifacts(tmp_path)

    replay = replay_bridge_v1_records(records)

    assert replay.records == 3
    assert replay.stage1_parse_successes == 0
    assert replay.stage2_invocations == 0
    assert dict(replay.parse_error_counts) == {
        "schema_invalid": 1,
        "not_exact_json_object": 1,
        "target_facts_not_four_integers": 1,
    }

    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    rows[0]["stage2_raw"] = "must-not-exist"
    records.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Stage 2 was invoked"):
        replay_bridge_v1_records(records)


def test_bridge_v1_failure_artifacts_are_hash_bound_and_replayed(tmp_path: Path) -> None:
    records, report, diagnostic, attempted, console = _write_failure_artifacts(tmp_path)
    expected = BridgeV1FailureRecord(
        schema_version=1,
        status="FINAL_FAILED_STAGE1_INTERFACE",
        dataset_id="CVA-Recoverability-Bridge-v1",
        model_snapshot_sha256="a" * 64,
        bridge_exit=3,
        scenes=3,
        model_calls=6,
        legacy_parse_rate=0.96,
        legacy_answer_accuracy=0.40,
        legacy_perception_error_rate=0.24,
        stage1_parse_rate=0.0,
        stage2_invocations=0,
        hypothesis_tested=False,
        training_invoked=False,
        parse_error_counts=(
            ("not_exact_json_object", 1),
            ("schema_invalid", 1),
            ("target_facts_not_four_integers", 1),
        ),
        source_sha256=tuple(
            sorted(
                {
                    "attempt_marker": _sha256(attempted),
                    "bridge_console": _sha256(console),
                    "bridge_records": _sha256(records),
                    "bridge_report": _sha256(report),
                    "stage1_diagnostic": _sha256(diagnostic),
                }.items()
            )
        ),
    )

    verification = verify_bridge_v1_failure_artifacts(
        expected,
        records_path=records,
        report_path=report,
        diagnostic_path=diagnostic,
        attempt_marker_path=attempted,
        console_log_path=console,
    )

    assert verification.verified is True
    assert verification.records == 3
    assert verification.hypothesis_tested is False

    console.write_text("bridge_exit=0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_bridge_v1_failure_artifacts(
            expected,
            records_path=records,
            report_path=report,
            diagnostic_path=diagnostic,
            attempt_marker_path=attempted,
            console_log_path=console,
        )


def test_stage1_v2_probe_cli_is_explicit_one_shot_and_development_only() -> None:
    script = ROOT / "experiments" / "recoverability_v1" / "04_stage1_v2_probe.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    for argument in (
        "--execute",
        "--runtime",
        "--probe-config",
        "--server-package-lock",
        "--preflight-report",
        "--external-evidence",
        "--v03-records",
        "--bridge-v1-records",
        "--bridge-v1-report",
        "--bridge-v1-diagnostic",
        "--bridge-v1-attempt-marker",
        "--bridge-v1-console-log",
    ):
        assert argument in completed.stdout

    blocked = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 2
    assert "BLOCKED" in blocked.stdout

    source = script.read_text(encoding="utf-8")
    assert source.index("run_metadata_preflight(") < source.index("load_local_qwen(")


def test_stage1_v2_server_lock_binds_the_complete_new_execution_surface() -> None:
    verification = verify_stage1_v2_server_package_lock(SERVER_LOCK, repository_root=ROOT)
    paths = {item.relative_path for item in verification.files}

    assert verification.verified is True
    assert {
        "configs/paths.example.yaml",
        "configs/recoverability/bridge_v1_failure.yaml",
        "configs/recoverability/stage1_v2_probe.yaml",
        "experiments/recoverability_v1/04_stage1_v2_probe.py",
        "src/compbias/recoverability/bridge_v1_failure.py",
        "src/compbias/recoverability/stage1_v2.py",
    }.issubset(paths)
    assert "experiments/recoverability_v1/03_bridge.py" in paths
    assert all(not path.startswith("tests/") for path in paths)


def test_stage1_v2_runtime_paths_must_byte_match_the_locked_example(tmp_path: Path) -> None:
    registered = tmp_path / "paths.example.yaml"
    runtime = tmp_path / "paths.yaml"
    registered.write_text("schema_version: 1\nproject_root: /fixed\n", encoding="utf-8")
    runtime.write_bytes(registered.read_bytes())

    validate_stage1_v2_runtime_paths(runtime, registered_example=registered)

    runtime.write_text("schema_version: 1\nproject_root: /changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte-match"):
        validate_stage1_v2_runtime_paths(runtime, registered_example=registered)

    runtime.unlink()
    with pytest.raises(ValueError, match="regular file"):
        validate_stage1_v2_runtime_paths(runtime, registered_example=registered)


def test_bridge_v1_replay_rejects_empty_record_file(tmp_path: Path) -> None:
    records = tmp_path / "bridge_records.jsonl"
    records.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        replay_bridge_v1_records(records)


def test_stage1_v2_preflight_is_metadata_only_and_uses_the_new_lock() -> None:
    script = ROOT / "experiments" / "recoverability_v1" / "00_stage1_v2_preflight.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    source = script.read_text(encoding="utf-8")

    assert "--server-package-lock" in completed.stdout
    assert "--runtime" in completed.stdout
    assert "--output" in completed.stdout
    assert "--execute" not in completed.stdout
    assert "import torch" not in source
    assert "from_pretrained" not in source
