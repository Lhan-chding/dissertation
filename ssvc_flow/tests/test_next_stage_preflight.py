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


def test_warm_preflight_requires_full_R4_binding_and_same_cold_plan(monkeypatch):
    from src import next_stage_preflight as module

    env = {"packages": "test"}
    gate = {"config": {"data_root": "/data"}, "binding": {"original": True}}
    cold = {"status": "PASS", "environment": env, "plan_hash": "same_fixed_plan"}
    warm = {
        "status": "PASS",
        "environment": env,
        "warm_checkpoint": {"arm": "X_BASE", "step": 64, "state_hash": "full_adam_state"},
    }
    calls = []
    monkeypatch.setattr(module, "validate_prerequisites", lambda *a: gate)
    monkeypatch.setattr(module, "validate_config_against_gate", lambda c, g: g["config"])
    monkeypatch.setattr(module, "validate_r2_gate", lambda *a: {"environment": env})
    monkeypatch.setattr(module, "validate_r3_cold_gate", lambda *a: cold)
    monkeypatch.setattr(module, "validate_runtime_environment", lambda *a: env)
    monkeypatch.setattr(module, "_source", lambda: {"source_commit": "fixture"})
    monkeypatch.setattr(
        module,
        "_cold_plan",
        lambda *a: ({"plan_hash": "same_fixed_plan", "counts": {"train_rollouts": 384}}, {}),
    )

    def verified_r4(*args):
        calls.append(args)
        return warm

    monkeypatch.setattr(module, "validate_r4_gate", verified_r4, raising=False)
    result = module.preflight(
        {}, "/data", "r0", "r1", "supp", "r2", phase="R3-warm", r3_dir="cold", r4_dir="r4"
    )
    assert calls[0] == ("r4", gate, {"environment": env}, cold)
    assert result["gate_binding"]["r4"]["warm_checkpoint"]["state_hash"] == "full_adam_state"
    assert result["budgets"]["new_outputs"] == 4224
    assert result["budgets"]["direct_resample_outputs"] == 3072
    assert result["budgets"]["control_sequence_scores"] == 7680
    assert not result["model_loaded"] and not result["gpu_work_requested"]
    with pytest.raises(ValueError, match="R4"):
        module.preflight({}, "/data", "r0", "r1", "supp", "r2", phase="R3-warm", r3_dir="cold")
    cold["plan_hash"] = "changed"
    with pytest.raises(ValueError, match="plan"):
        module.preflight(
            {}, "/data", "r0", "r1", "supp", "r2", phase="R3-warm", r3_dir="cold", r4_dir="r4"
        )
