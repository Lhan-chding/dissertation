"""Generic-gate callback contracts for the Study B runtime."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from compensability_v5.server_runtime import study_b as study_b_server

_RUNTIME_TEST = Path(__file__).with_name("test_study_b_runtime.py")
_SPEC = importlib.util.spec_from_file_location("_study_b_runtime_test_support", _RUNTIME_TEST)
assert _SPEC is not None and _SPEC.loader is not None
_SUPPORT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SUPPORT)
MODEL_SHA = _SUPPORT.MODEL_SHA
StudyBError = _SUPPORT.StudyBError
_FakeBackend = _SUPPORT._FakeBackend
_evaluation_rows = _SUPPORT._evaluation_rows
_support_package = _SUPPORT._support_package
run_study_b = _SUPPORT.run_study_b


def test_generic_study_b_callbacks_run_complete_fake_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    support_path = tmp_path / "support.json"
    evaluation_path = tmp_path / "evaluation.json"
    support_path.write_text(json.dumps(_support_package()), encoding="utf-8")
    evaluation_path.write_text(json.dumps({"rows": _evaluation_rows()}), encoding="utf-8")
    output = tmp_path / "study-b"
    backend = _FakeBackend()
    monkeypatch.setattr(study_b_server, "require_offline_environment", lambda: None)
    monkeypatch.setattr(study_b_server, "verify_runtime_package_lock", lambda _path: {})
    monkeypatch.setattr(study_b_server, "QwenStudyBBackend", lambda **_kwargs: backend)
    completed = study_b_server.run_budget_matched_lora(
        {
            "phase": "phase4_budget_matched_lora",
            "output": str(output),
            "config_sha256": "a" * 64,
            "package_lock_sha256": study_b_server.sha256_file(study_b_server.PACKAGE_LOCK),
            "input_sha256": {
                str(support_path): study_b_server.sha256_file(support_path),
                str(evaluation_path): study_b_server.sha256_file(evaluation_path),
            },
        }
    )
    orbit_output = tmp_path / "orbit.json"
    exported = study_b_server.run_orbit_support(
        {
            "phase": "phase5_structural_support_evaluation",
            "output": str(orbit_output),
            "input_sha256": {
                str(output / "completed.json"): study_b_server.sha256_file(
                    output / "completed.json"
                )
            },
        },
        {"k": 6},
    )

    assert completed["status"] == "STUDY_B_SINGLE_SEED_COMPLETE"
    assert exported["status"] == "STUDY_B_ORBIT_SUPPORT_EXPORTED"
    assert exported["k"] == 6
    assert json.loads(orbit_output.read_text()) == exported


def test_generic_study_b_callback_rejects_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(StudyBError, match="SHA-256"):
        study_b_server._input_paths({"input_sha256": {str(source): "0" * 64}})


def test_generic_study_b_callback_rejects_package_lock_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_lock = tmp_path / "package-lock.yaml"
    package_lock.write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.setattr(study_b_server, "PACKAGE_LOCK", package_lock)
    with pytest.raises(StudyBError, match="package-lock SHA-256"):
        study_b_server._verify_callback_package_lock("0" * 64)


def test_orbit_callback_recomputes_stop_signal_from_paired_ci(tmp_path: Path) -> None:
    output = tmp_path / "study-b"
    run_study_b(
        support_package=_support_package(),
        evaluation_rows=_evaluation_rows(),
        output=output,
        backend=_FakeBackend(),
        expected_model_sha256=MODEL_SHA,
    )
    completed_path = output / "completed.json"
    completed = json.loads(completed_path.read_text())
    completed["primary_contrasts"]["B3_minus_B2"]["iid_exact_world_rate"] = 0.5
    completed_path.write_text(json.dumps(completed), encoding="utf-8")
    with pytest.raises(StudyBError, match="primary contrasts drifted"):
        study_b_server.run_orbit_support(
            {
                "phase": "phase5_structural_support_evaluation",
                "output": str(tmp_path / "bad-contrast-orbit.json"),
                "input_sha256": {str(completed_path): study_b_server.sha256_file(completed_path)},
            }
        )
    completed = json.loads((output / "completed.json").read_text())
    completed["primary_contrasts"]["B3_minus_B2"]["iid_exact_world_rate"] = 0.0
    completed["primary_contrasts"]["paired_inference"]["stop_signal"]["triggered"] = False
    completed["stop_signal"]["triggered"] = False
    completed_path.write_text(json.dumps(completed), encoding="utf-8")

    with pytest.raises(StudyBError, match="stop signal drifted"):
        study_b_server.run_orbit_support(
            {
                "phase": "phase5_structural_support_evaluation",
                "output": str(tmp_path / "orbit.json"),
                "input_sha256": {str(completed_path): study_b_server.sha256_file(completed_path)},
            }
        )
