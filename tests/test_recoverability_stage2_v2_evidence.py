from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from compbias.recoverability.stage2_v2 import Stage2V2Scene, run_stage2_v2_probe
from compbias.recoverability.stage2_v2_evidence import (
    Stage2V2FrozenResult,
    load_stage2_v2_frozen_result,
    verify_stage2_v2_artifacts,
    verify_stage2_v2_evidence_package_lock,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_RESULT = ROOT / "configs" / "recoverability" / "stage2_v2_frozen_result.yaml"
EVIDENCE_LOCK = (
    ROOT / "configs" / "recoverability" / "server_package_lock_stage2_v2_evidence.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene(index: int) -> Stage2V2Scene:
    return Stage2V2Scene(
        scene_id=f"dev-{index:06d}",
        operation=("difference", "max_minus_min", "sum")[index % 3],
        evidence=(8, 4, 5, 9),
    )


def _program(scene: Stage2V2Scene) -> str:
    if scene.operation == "sum":
        steps = [{"op": "add", "inputs": ["a", "b"], "output": "result"}]
    elif scene.operation == "difference":
        steps = [{"op": "subtract", "inputs": ["a", "b"], "output": "result"}]
    else:
        steps = [
            {"op": "max", "inputs": ["a", "b", "c", "d"], "output": "high"},
            {"op": "min", "inputs": ["a", "b", "c", "d"], "output": "low"},
            {"op": "subtract", "inputs": ["high", "low"], "output": "result"},
        ]
    return json.dumps(
        {
            "variables": dict(zip(("a", "b", "c", "d"), scene.evidence, strict=True)),
            "steps": steps,
            "return": "result",
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_artifacts(
    tmp_path: Path,
) -> tuple[Stage2V2FrozenResult, tuple[Stage2V2Scene, ...], dict[str, Path]]:
    scenes = tuple(_scene(index) for index in range(24))
    replay_report, replay_records = run_stage2_v2_probe(
        scenes,
        generate=lambda scene, _messages: _program(scene),
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
        "artifact_type": "recoverability_stage2_v2_development_probe",
        "dataset_id": "CVA-Recoverability-Stage2-V2-Dev-Probe",
        "source_dataset_id": "CVA-Chart-Pilot-v0.3",
        "source_split": "dev",
        "model_snapshot_sha256": "a" * 64,
        "source_stage1_records_sha256": "b" * 64,
        "source_stage2_v1_diagnostic_sha256": "c" * 64,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
    }
    report = tmp_path / "probe_report.json"
    report.write_text(json.dumps(report_payload, sort_keys=True), encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "artifact_type": "recoverability_stage2_v2_metadata_preflight",
                "ready": True,
                "large_gpu_started": False,
                "model_loaded": False,
                "training_authorized": False,
                "server_package_lock_verified": True,
                "server_package_lock_sha256": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    console = tmp_path / "console.log"
    console.write_text("stage2_v2_probe_exit=0\n", encoding="utf-8")
    attempt = tmp_path / "attempted.json"
    attempt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "STAGE2_V2_DEVELOPMENT_PROBE_STARTED",
                "hypothesis_test": False,
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "preflight": preflight,
        "console": console,
        "probe_report": report,
        "probe_records": records,
        "attempt_marker": attempt,
    }
    frozen = Stage2V2FrozenResult(
        schema_version=1,
        status="FINAL_PASSED_DEVELOPMENT_PROBE_DO_NOT_RERUN",
        dataset_id="CVA-Recoverability-Stage2-V2-Dev-Probe",
        source_dataset_id="CVA-Chart-Pilot-v0.3",
        source_split="dev",
        model_snapshot_sha256="a" * 64,
        source_stage1_records_sha256="b" * 64,
        source_stage2_v1_diagnostic_sha256="c" * 64,
        server_package_lock_sha256="d" * 64,
        probe_exit=0,
        scenes=24,
        model_calls=24,
        program_parse_rate=1.0,
        execution_rate=1.0,
        executor_answer_accuracy=1.0,
        probe_passed=True,
        hypothesis_tested=False,
        confirmatory_execution_authorized=False,
        training_invoked=False,
        error_counts=(),
        source_sha256=tuple(sorted((label, _sha256(path)) for label, path in paths.items())),
    )
    return frozen, scenes, paths


def test_stage2_v2_result_is_externally_frozen_as_interface_evidence() -> None:
    frozen = load_stage2_v2_frozen_result(FROZEN_RESULT)

    assert frozen.status == "FINAL_PASSED_DEVELOPMENT_PROBE_DO_NOT_RERUN"
    assert frozen.probe_exit == 0
    assert frozen.scenes == frozen.model_calls == 24
    assert frozen.program_parse_rate == 1.0
    assert frozen.execution_rate == 1.0
    assert frozen.executor_answer_accuracy == 1.0
    assert frozen.error_counts == ()
    assert frozen.probe_passed is True
    assert frozen.hypothesis_tested is False
    assert frozen.confirmatory_execution_authorized is False
    assert frozen.training_invoked is False
    assert dict(frozen.source_sha256) == {
        "attempt_marker": "5cf97d5cb67c5e9830ac7455b0759b47b6d1fe1e76ee5ce42f365f4059e51c7c",
        "console": "3649a50b21a5482ab0e20bfe07ed63f4d49f72c1f1b783791b852044253eed81",
        "preflight": "c3b8949f03ae7ba2947ad5632bfd68dc822f3aead33adc218e90619a0957fe0c",
        "probe_records": "6b0604a08ebbf4611c62b7fe9f1d9e03954385b1bcfaedf613f0b70b32f1d2f8",
        "probe_report": "d207cff9f6bdcb48142e3f1bb8a3d8676d7aa5abdbcd0c226666dbb58ac587b7",
    }


def test_stage2_v2_artifacts_replay_every_raw_record_without_model_calls(
    tmp_path: Path,
) -> None:
    frozen, scenes, paths = _write_artifacts(tmp_path)

    verification = verify_stage2_v2_artifacts(
        frozen,
        preflight_path=paths["preflight"],
        console_path=paths["console"],
        report_path=paths["probe_report"],
        records_path=paths["probe_records"],
        attempt_marker_path=paths["attempt_marker"],
        scenes=scenes,
    )

    assert verification.verified is True
    assert verification.records == 24
    assert verification.replayed_program_parse_successes == 24
    assert verification.replayed_program_execution_successes == 24
    assert verification.replayed_executor_answer_correct == 24
    assert verification.model_calls == 0
    assert verification.hypothesis_tested is False
    assert verification.confirmatory_execution_authorized is False
    assert verification.training_invoked is False

    paths["probe_records"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_stage2_v2_artifacts(
            frozen,
            preflight_path=paths["preflight"],
            console_path=paths["console"],
            report_path=paths["probe_report"],
            records_path=paths["probe_records"],
            attempt_marker_path=paths["attempt_marker"],
            scenes=scenes,
        )


def test_stage2_v2_replay_rejects_stored_flag_tampering(tmp_path: Path) -> None:
    frozen, scenes, paths = _write_artifacts(tmp_path)
    rows = [json.loads(line) for line in paths["probe_records"].read_text().splitlines()]
    rows[-1]["executor_answer_correct"] = False
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
        verify_stage2_v2_artifacts(
            frozen,
            preflight_path=paths["preflight"],
            console_path=paths["console"],
            report_path=paths["probe_report"],
            records_path=paths["probe_records"],
            attempt_marker_path=paths["attempt_marker"],
            scenes=scenes,
        )


def test_stage2_v2_evidence_capture_is_model_free_and_package_locked() -> None:
    script = ROOT / "experiments" / "recoverability_v1" / "08_capture_stage2_v2_evidence.py"
    source = script.read_text(encoding="utf-8")
    lock_text = EVIDENCE_LOCK.read_text(encoding="utf-8")

    assert "--frozen-result" in source
    assert "--stage2-v2-report" in source
    assert "--stage2-v2-records" in source
    assert "--attempt-marker" in source
    assert "--output" in source
    assert "load_local_qwen" not in source
    assert "decode_text_qwen_once" not in source
    assert "torch" not in source
    assert "cuda" not in source.lower()
    assert "_BOOTSTRAP_EVIDENCE_PATHS" in source
    assert "frozenset(paths) != _BOOTSTRAP_EVIDENCE_PATHS" in source
    assert "stage2_v2_evidence.py" in lock_text
    assert "08_capture_stage2_v2_evidence.py" in lock_text
    verification = verify_stage2_v2_evidence_package_lock(
        EVIDENCE_LOCK,
        repository_root=ROOT,
    )
    assert verification.verified is True
