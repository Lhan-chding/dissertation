from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from compbias.recoverability.stage2_v1 import Stage2V1Scene, run_stage2_v1_probe
from compbias.recoverability.stage2_v1_failure import (
    Stage2V1FailureRecord,
    load_stage2_v1_failure,
    verify_stage2_v1_diagnostic_package_lock,
    verify_stage2_v1_failure_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
FAILURE_CONFIG = ROOT / "configs" / "recoverability" / "stage2_v1_failure.yaml"
DIAGNOSTIC_LOCK = (
    ROOT / "configs" / "recoverability" / "server_package_lock_stage2_v1_diagnostic.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene(index: int) -> Stage2V1Scene:
    operations = ("difference", "sum", "max_minus_min")
    return Stage2V1Scene(
        scene_id=f"dev-{index:06d}",
        operation=operations[index % len(operations)],
        evidence=(8, 4, 5, 9),
    )


def _program(scene: Stage2V1Scene, *, answer_offset: int = 0) -> str:
    a, b, c, d = scene.evidence
    if scene.operation == "difference":
        steps = [{"op": "subtract", "inputs": ["a", "b"], "output": "result"}]
        answer = a - b
    elif scene.operation == "sum":
        steps = [{"op": "add", "inputs": ["a", "b"], "output": "result"}]
        answer = a + b
    else:
        steps = [
            {"op": "max", "inputs": ["a", "b", "c", "d"], "output": "high"},
            {"op": "min", "inputs": ["a", "b", "c", "d"], "output": "low"},
            {"op": "subtract", "inputs": ["high", "low"], "output": "result"},
        ]
        answer = max(scene.evidence) - min(scene.evidence)
    return json.dumps(
        {
            "variables": {"a": a, "b": b, "c": c, "d": d},
            "steps": steps,
            "answer": answer + answer_offset,
        },
        separators=(",", ":"),
    )


def _write_frozen_artifacts(
    tmp_path: Path,
) -> tuple[
    Stage2V1FailureRecord,
    tuple[Stage2V1Scene, ...],
    dict[str, Path],
]:
    scenes = tuple(_scene(index) for index in range(24))
    raw_by_id: dict[str, str] = {}
    for index, scene in enumerate(scenes):
        if index < 5:
            raw_by_id[scene.scene_id] = "```json\n{}\n```" if index == 0 else "not-json"
        elif index < 11:
            raw_by_id[scene.scene_id] = _program(scene, answer_offset=1)
        else:
            raw_by_id[scene.scene_id] = _program(scene)
    replay_report, replay_records = run_stage2_v1_probe(
        scenes,
        generate=lambda scene, _messages: raw_by_id[scene.scene_id],
    )
    records = tmp_path / "probe_records.jsonl"
    records.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in replay_records),
        encoding="utf-8",
    )
    report_payload = {
        **asdict(replay_report),
        "error_counts": dict(replay_report.error_counts),
        "schema_version": 1,
        "artifact_type": "recoverability_stage2_v1_development_probe",
        "dataset_id": "CVA-Recoverability-Stage2-V1-Dev-Probe",
        "source_dataset_id": "CVA-Chart-Pilot-v0.3",
        "source_split": "dev",
        "model_snapshot_sha256": "a" * 64,
        "source_stage1_records_sha256": "b" * 64,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
    }
    report = tmp_path / "probe_report.json"
    report.write_text(json.dumps(report_payload, sort_keys=True), encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "artifact_type": "recoverability_stage2_v1_metadata_preflight",
                "ready": True,
                "large_gpu_started": False,
                "model_loaded": False,
                "training_authorized": False,
                "server_package_lock_verified": True,
                "server_package_lock_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    console = tmp_path / "console.log"
    console.write_text("frozen output\nstage2_probe_exit=3\n", encoding="utf-8")
    paths = {
        "preflight": preflight,
        "console": console,
        "probe_report": report,
        "probe_records": records,
    }
    frozen = Stage2V1FailureRecord(
        schema_version=1,
        status="FINAL_FAILED_DEVELOPMENT_PROBE_DO_NOT_RERUN",
        dataset_id="CVA-Recoverability-Stage2-V1-Dev-Probe",
        source_dataset_id="CVA-Chart-Pilot-v0.3",
        source_split="dev",
        model_snapshot_sha256="a" * 64,
        source_stage1_records_sha256="b" * 64,
        server_package_lock_sha256="c" * 64,
        probe_exit=3,
        scenes=24,
        model_calls=24,
        program_parse_rate=19 / 24,
        execution_rate=19 / 24,
        program_answer_consistency=13 / 24,
        operation_result_accuracy=13 / 24,
        probe_passed=False,
        hypothesis_tested=False,
        confirmatory_execution_authorized=False,
        training_invoked=False,
        error_counts=(("program_answer_mismatch", 6), ("program_parse_failure", 5)),
        source_sha256=tuple(sorted((label, _sha256(path)) for label, path in paths.items())),
    )
    return frozen, scenes, paths


def test_stage2_v1_failure_is_externally_frozen_as_interface_evidence() -> None:
    frozen = load_stage2_v1_failure(FAILURE_CONFIG)

    assert frozen.status == "FINAL_FAILED_DEVELOPMENT_PROBE_DO_NOT_RERUN"
    assert frozen.probe_exit == 3
    assert frozen.scenes == frozen.model_calls == 24
    assert frozen.program_parse_rate == pytest.approx(19 / 24)
    assert frozen.program_answer_consistency == pytest.approx(13 / 24)
    assert frozen.operation_result_accuracy == pytest.approx(13 / 24)
    assert dict(frozen.error_counts) == {
        "program_answer_mismatch": 6,
        "program_parse_failure": 5,
    }
    assert frozen.probe_passed is False
    assert frozen.hypothesis_tested is False
    assert frozen.confirmatory_execution_authorized is False
    assert frozen.training_invoked is False
    assert dict(frozen.source_sha256) == {
        "console": "9de0fbc1e610886c99d5271f08306cbf091e0c9de02eaab6762fbe52f41d1502",
        "preflight": "0828fd63071dcb00ac97269800e565262568232064834922538c72afa6606d7e",
        "probe_records": "b8ce766b9ba0555fe780e88195a0e4d4d3294cc8813357e7a33a5cfeea19793c",
        "probe_report": "aca0671f5c4b5e4334544955733b7143931618f9e9a33c7cdd5a8bd09cc55548",
    }


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("probe_exit: 3", "probe_exit: true"),
        ("probe_passed: false", "probe_passed: 0"),
    ],
)
def test_stage2_v1_failure_config_rejects_boolean_numeric_aliases(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    tampered = tmp_path / "failure.yaml"
    tampered.write_text(
        FAILURE_CONFIG.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError)):
        load_stage2_v1_failure(tampered)


def test_stage2_v1_failure_replays_every_raw_record_without_model_calls(tmp_path: Path) -> None:
    frozen, scenes, paths = _write_frozen_artifacts(tmp_path)

    diagnostic = verify_stage2_v1_failure_artifacts(
        frozen,
        preflight_path=paths["preflight"],
        console_path=paths["console"],
        report_path=paths["probe_report"],
        records_path=paths["probe_records"],
        scenes=scenes,
    )

    assert diagnostic.verified is True
    assert diagnostic.records == 24
    assert diagnostic.replayed_program_parse_successes == 19
    assert diagnostic.replayed_program_answer_matches == 13
    assert diagnostic.model_calls == 0
    assert diagnostic.hypothesis_tested is False
    assert diagnostic.confirmatory_execution_authorized is False
    assert diagnostic.training_invoked is False
    assert sum(item.total for item in diagnostic.by_operation) == 24
    assert sum(item.parse_failures for item in diagnostic.by_operation) == 5
    assert sum(item.answer_mismatches for item in diagnostic.by_operation) == 6
    assert any("markdown_fence" in key for key, _count in diagnostic.failure_signatures)

    paths["probe_records"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_stage2_v1_failure_artifacts(
            frozen,
            preflight_path=paths["preflight"],
            console_path=paths["console"],
            report_path=paths["probe_report"],
            records_path=paths["probe_records"],
            scenes=scenes,
        )


def test_stage2_v1_failure_replay_rejects_stored_flag_tampering(tmp_path: Path) -> None:
    frozen, scenes, paths = _write_frozen_artifacts(tmp_path)
    rows = [json.loads(line) for line in paths["probe_records"].read_text().splitlines()]
    rows[-1]["program_answer_match"] = False
    paths["probe_records"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    frozen = replace(
        frozen,
        source_sha256=tuple(
            (label, _sha256(paths[label]) if label == "probe_records" else digest)
            for label, digest in frozen.source_sha256
        ),
    )

    with pytest.raises(ValueError, match="stored record does not replay"):
        verify_stage2_v1_failure_artifacts(
            frozen,
            preflight_path=paths["preflight"],
            console_path=paths["console"],
            report_path=paths["probe_report"],
            records_path=paths["probe_records"],
            scenes=scenes,
        )


def test_stage2_v1_diagnostic_cli_is_read_only_and_package_locked() -> None:
    script = ROOT / "experiments" / "recoverability_v1" / "06_diagnose_stage2_v1_failure.py"
    source = script.read_text(encoding="utf-8")
    lock_text = DIAGNOSTIC_LOCK.read_text(encoding="utf-8")

    assert "--failure-config" in source
    assert "--stage2-report" in source
    assert "--stage2-records" in source
    assert "--output" in source
    assert "load_local_qwen" not in source
    assert "decode_text_qwen_once" not in source
    assert "torch" not in source
    assert "cuda" not in source.lower()
    assert "stage2_v1_failure.py" in lock_text
    assert "06_diagnose_stage2_v1_failure.py" in lock_text
    verification = verify_stage2_v1_diagnostic_package_lock(
        DIAGNOSTIC_LOCK,
        repository_root=ROOT,
    )
    assert verification.verified is True
