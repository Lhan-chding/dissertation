"""CPU integration of the supplied package without confirmation or model execution."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.exposure_position import control
from src.exposure_position.schema import file_digest


def test_changed_design_bytes_fail_before_creating_run(tmp_path):
    design = tmp_path / "design.json"
    design.write_text("{}\n")
    with pytest.raises(ValueError, match="exact user-supplied"):
        control.bind_design(design, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_source_publish_never_overwrites_an_existing_binding(tmp_path):
    path = tmp_path / "source"
    control.write_once_bytes(path, b"first")
    control.write_once_bytes(path, b"first")
    with pytest.raises(ValueError, match="Immutable"):
        control.write_once_bytes(path, b"second")
    assert path.read_bytes() == b"first"


def test_training_modes_keep_all_fixed_parent_order_blocks():
    assert len(control.training_jobs("REUSE_12")) == 12
    assert len(control.training_jobs("RERUN_24")) == 24
    assert {r["logical_arm_id"] for r in control.training_jobs("REUSE_12")} == {
        "A3_LOCAL_C1_J3",
        "B3_FORWARD_C4_J3",
    }
    with pytest.raises(ValueError):
        control.training_jobs("AUTO")


def test_reference_does_not_predict_other_leaf_from_trigger_overlap():
    rows = control.structure_reference()["cells"]
    other_leaf = next(
        r
        for r in rows
        if r["logical_arm_id"] == "B3_FORWARD_C4_J3"
        and r["receiver_center_0based"] == 3
        and r["receiver_corrupted_index_0based"] == 1
    )
    center = next(
        r
        for r in rows
        if r["logical_arm_id"] == "B3_FORWARD_C4_J3"
        and r["receiver_center_0based"] == 3
        and r["receiver_corrupted_index_0based"] == 3
    )
    assert not other_leaf["same_center_support_subset"]
    assert center["same_center_support_subset"] and center["risk_reference"]


def test_worker_requires_explicit_model_permission_before_accessing_files(tmp_path):
    from src.exposure_position.worker import worker

    with pytest.raises(PermissionError, match="allow-gpu"):
        worker(tmp_path / "missing")
    assert list(tmp_path.iterdir()) == []


def test_worker_dispatch_preflight_uses_control_capability(tmp_path, monkeypatch):
    from src.exposure_position import schema, training, worker

    roles = []
    monkeypatch.setattr(control, "assert_code_bindings", lambda frozen: None)
    monkeypatch.setattr(
        schema, "verify_frozen", lambda run, *, role: roles.append(("frozen", role)) or {}
    )
    monkeypatch.setattr(
        training, "require_bridge", lambda run, *, role: roles.append(("bridge", role))
    )

    def stop_before_claim():
        raise RuntimeError("CPU test stops before GPU or queue access")

    monkeypatch.setattr(worker, "physical_gpu_key", stop_before_claim)
    with pytest.raises(RuntimeError, match="CPU test stops"):
        worker.worker(tmp_path, allow_gpu=True)
    assert roles == [("frozen", "control"), ("bridge", "control")]
    assert list(tmp_path.iterdir()) == []


def test_real_archive_cpu_pipeline_preserves_history_and_cannot_freeze(tmp_path):
    package, legacy = os.environ.get("SER_J23_SOURCE_PACKAGE"), os.environ.get("SER_J23_LEGACY_RUN")
    if not package or not legacy:
        pytest.skip("Explicit private source package and extracted historical run required")
    package, legacy = Path(package), Path(legacy)
    old_hashes = {p: file_digest(p) for p in (legacy / "manifests").rglob("*") if p.is_file()}
    run = tmp_path / "run"
    result = control.build_training(package / "SER_J23_DESIGN.json", legacy, run)
    assert result["model_calls"] == 0 and not result["confirmation_generated"]
    assert control.cpu_check(run)["status"] == "PASS"
    assert json.loads((run / "machine.json").read_text())[
        "legacy_frozen_plan_sha256"
    ] == file_digest(legacy / "FROZEN_PLAN.json")
    assert not (run / "FROZEN_PLAN.json").exists()
    assert not (run / "manifests/E_CONFIRM2").exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        control.freeze(run, tmp_path / "no_exclusion", tmp_path / "no_validation")
    assert not (run / "FROZEN_PLAN.json").exists()
    assert all(file_digest(path) == expected for path, expected in old_hashes.items())
