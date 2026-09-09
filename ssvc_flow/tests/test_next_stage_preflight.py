"""A CPU prerequisite check cannot claim or start a model experiment."""

import pytest


def test_preflight_checks_prerequisites_and_reports_cpu_only(monkeypatch):
    from src import next_stage_preflight as module

    gate = {
        "config": {"data_root": "/data", "model": {}, "protocol_version": "test"},
        "binding": {"original": True},
    }
    monkeypatch.setattr(module, "validate_prerequisites", lambda *a: gate)
    monkeypatch.setattr(module, "validate_config_against_gate", lambda c, g: g["config"])
    monkeypatch.setattr(
        module,
        "validate_r2_gate",
        lambda *a: {"status": "PASS", "environment": {"packages": "test"}},
    )
    monkeypatch.setattr(module, "validate_runtime_environment", lambda *a: {"packages": "test"})
    monkeypatch.setattr(module, "_source", lambda: {"source_commit": "fixture"})
    monkeypatch.setattr(
        module,
        "_cold_plan",
        lambda *a: ({"plan_hash": "fixed", "counts": {"train_rollouts": 384}}, {"files": {}}),
    )
    result = module.preflight({}, "/data", "r0", "r1", "supp", "r2", phase="R3-cold")
    assert result["status"] == "PASS" and result["execution_kind"] == "CPU_AUDIT"
    assert result["model_loaded"] is False and result["gpu_work_requested"] is False
    assert result["plan_hash"] == "fixed"
    assert result["budgets"]["candidate_optimizer_updates"] == 60
    monkeypatch.setattr(module, "validate_runtime_environment", lambda *a: {"packages": "changed"})
    with pytest.raises(ValueError, match="environment"):
        module.preflight({}, "/data", "r0", "r1", "supp", "r2", phase="R3-cold")


def test_preflight_rejects_unknown_phase_or_missing_cold_evidence():
    from src.next_stage_preflight import preflight

    with pytest.raises(ValueError, match="phase"):
        preflight({}, "/data", "r0", "r1", "supp", "r2", phase="R5")
    with pytest.raises(ValueError, match="R3-cold"):
        preflight({}, "/data", "r0", "r1", "supp", "r2", phase="R4")
