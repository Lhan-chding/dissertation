"""CPU wiring tests, explicitly incapable of certifying a scientific run.

G0 and the numerical bridge are mocked. The confirmation builder returns ZERO
tasks, and no checkpoint files or model backends exist. These tests exercise
freeze/queue field contracts only; their PASS is never experimental permission.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.exposure_position import bridge, control, data, evaluate, exclusions, queue, runtime
from src.exposure_position.schema import (
    EXPERIMENT_ID,
    TRAINING_PATHS,
    digest,
    file_digest,
    read_json,
    read_jsonl,
    verify_frozen,
    verify_source_bound,
    write_json,
)
from src.exposure_position.training import REQUIRED_BRIDGE_CHECKS, require_bridge
from src.modeling_v3.io import canonical_hash


@pytest.fixture
def wiring_only_run(tmp_path, monkeypatch):
    from src.exposure_position import legacy_trajectory

    run = tmp_path / "synthetic_wiring_only"
    run.mkdir()
    # Source verification is real, but the bound training files are empty CPU
    # fixtures. They cannot satisfy the real data reconstruction/G0 checks.
    for relative in TRAINING_PATHS:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            path.write_text("")
        else:
            write_json(path, {"fixture_only": True})
    for name in (
        "SER_J23_DESIGN.json",
        "DESIGN_SOURCE_SHA256.json",
        "SOURCE_PROVENANCE.json",
        "TRAINING_DATA_AUDIT.json",
        "DEVELOPMENT_DATA_AUDIT.json",
        "DIRECTIONAL_HYPOTHESES.json",
        "STRUCTURE_OVERLAP_REFERENCE.json",
        "GUARD_FEASIBILITY_ONLY.json",
        "MANUAL_FIXTURE_ORBIT_EXCLUSIONS.json",
    ):
        write_json(run / name, {"fixture_only": True})
    (run / "DESIGN_SOURCE.md").write_text("CPU wiring fixture, not a scientific design.\n")
    source = {
        "status": "SOURCE_BOUND",
        "phase_id": EXPERIMENT_ID,
        "files": {
            relative: {"sha256": file_digest(run / relative)} for relative in sorted(TRAINING_PATHS)
        },
    }
    source["source_hash"] = digest(source)
    write_json(run / "SOURCE_BOUND.json", source)
    assert verify_source_bound(run) == source
    code = control.code_bindings()
    validation = {"status": "PASS", "code_sha256": code, "failures": 0, "errors": 0}
    write_json(run / "VALIDATION_RECEIPT.json", validation)
    write_json(run / "CPU_CHECK.json", validation)
    g0 = {"status": "MOCK_CPU_WIRING_ONLY_NOT_SCIENTIFIC_PERMISSION"}
    monkeypatch.setattr(bridge, "validate_g0", Mock(return_value=g0))
    monkeypatch.setattr(control, "validate_design", Mock(return_value={"fixture_only": True}))
    write_json(
        run / "CONTINUITY_BRIDGE.json",
        {
            "status": "PASS",
            "fixture_only": True,
            "execution_kind": "REAL_CUDA_BRIDGE",
            "g0_receipt": g0,
            "code_sha256": code,
            "source_bound_hash": canonical_hash(source),
            "technical_only": True,
            "maximum_physical_updates": 32,
            "physical_updates_started": 32,
            "physical_updates_completed": 32,
            "generations": 0,
            "E_CONFIRM2_accessed": False,
            "legacy_compatibility_pass": True,
            "checks": dict.fromkeys(REQUIRED_BRIDGE_CHECKS, True),
        },
    )
    lookup = {}
    for job in evaluate.build_evaluation_jobs("REUSE_12"):
        if job["source"] == "NEW_TRAINING":
            continue
        key = (
            "PARENT." + job["parent"] if job["source"] == "PARENT" else evaluate.checkpoint_key(job)
        )
        lookup[key] = {
            "path": str(tmp_path / "NO_CHECKPOINT_EXISTS.pt"),
            "checkpoint_sha256": "0" * 64,
            "source_experiment_id": "CPU_WIRING_FIXTURE",
            "source_checkpoint_id": key,
            "expected_identity": {},
            "status": "VERIFIED_FULL_CPU_STATE",
        }
    write_json(
        run / "PARENT_AND_ENDPOINT_BINDINGS.json",
        {"checkpoint_lookup": lookup, "runtime_environment": {}, "base_snapshot": {}},
    )
    write_json(
        run / "REUSE_AUDIT.json",
        {
            "status": "REUSE_CANDIDATE",
            "fixture_only": True,
            "bindings_file": {"sha256": file_digest(run / "PARENT_AND_ENDPOINT_BINDINGS.json")},
        },
    )
    exclusion_path = tmp_path / "synthetic_exclusion.json"
    write_json(exclusion_path, {"fixture_only": True})
    monkeypatch.setattr(exclusions, "load_verified_exclusions", Mock(return_value=set()))
    monkeypatch.setattr(
        legacy_trajectory,
        "audit_historical_trajectory",
        Mock(return_value=({"fixture_only": True, "rows_reused": 0}, [])),
    )
    confirmation = Mock(
        return_value=(
            {"E_CONFIRM2/tasks_public.jsonl": [], "E_CONFIRM2/audit_only.jsonl": []},
            {"fixture_only": True, "tasks_generated": 0, "model_calls": 0},
        )
    )
    monkeypatch.setattr(data, "build_confirmation_from_audit", confirmation)
    model_loader = Mock(side_effect=AssertionError("CPU wiring test must never load a model"))
    monkeypatch.setattr(runtime, "load_parent_backend", model_loader)
    return run, exclusion_path, confirmation, model_loader


def _freeze_fixture(fixture):
    run, exclusion, _, _ = fixture
    return control.freeze(run, exclusion, run / "VALIDATION_RECEIPT.json")


def test_mock_freeze_to_registered_queue_preserves_complete_allocation(wiring_only_run):
    run, exclusion, confirmation, model_loader = wiring_only_run
    assert _freeze_fixture(wiring_only_run)["status"] == "FROZEN"
    confirmation.assert_called_once_with(exclusion, auditor_authorized=True)
    assert read_jsonl(run / "manifests/E_CONFIRM2/tasks_public.jsonl") == []
    assert read_jsonl(run / "manifests/E_CONFIRM2/audit_only.jsonl") == []
    assert read_json(run / "CONFIRMATION_GENERATION_AUDIT.json")["tasks_generated"] == 0
    assert verify_frozen(run, role="auditor")["status"] == "FROZEN"
    result = queue.build_queue(run)
    assert result["formal_training_jobs"] == 12
    assert result["formal_evaluation_jobs"] == 114
    assert result["formal_generations"] == 259656
    registered = queue.registered_queue(run)
    try:
        rows = registered.rows()
        assert len(rows) == 126
        assert all(row["status"] == "PENDING" for row in rows)
        training_ids = {row["id"] for row in rows if row["payload"]["kind"] == "train"}
        for row in rows:
            payload = row["payload"]
            if payload.get("panel") == "E_CONFIRM2":
                assert set(payload["dependencies"]) == training_ids
                assert payload["kind"] == "eval"
                assert payload["evaluation_kind"] == "confirm_eval_sealed"
    finally:
        registered.db.close()
    model_loader.assert_not_called()


def test_changed_g0_bridge_binding_rejects_before_confirmation(wiring_only_run):
    run, _, confirmation, model_loader = wiring_only_run
    receipt = read_json(run / "CONTINUITY_BRIDGE.json")
    receipt["g0_receipt"] = {"status": "DIFFERENT_MOCK_G0"}
    write_json(run / "CONTINUITY_BRIDGE.json", receipt)
    with pytest.raises(ValueError, match="bridge"):
        _freeze_fixture(wiring_only_run)
    assert not (run / "FROZEN_PLAN.json").exists()
    confirmation.assert_not_called()
    model_loader.assert_not_called()


def test_changed_frozen_training_source_rejects_before_execution(wiring_only_run):
    run, _, _, model_loader = wiring_only_run
    _freeze_fixture(wiring_only_run)
    write_json(run / "machine.json", {"tampered": True})
    with pytest.raises(ValueError, match="modified frozen file"):
        require_bridge(run)
    model_loader.assert_not_called()


def test_changed_registered_payload_is_rejected(wiring_only_run):
    run, _, _, model_loader = wiring_only_run
    _freeze_fixture(wiring_only_run)
    queue.build_queue(run)
    registered = queue.registered_queue(run)
    try:
        registered.db.execute("UPDATE jobs SET payload='{}' WHERE rowid=1")
    finally:
        registered.db.close()
    with pytest.raises(ValueError, match="Queue differs"):
        queue.registered_queue(run)
    model_loader.assert_not_called()
