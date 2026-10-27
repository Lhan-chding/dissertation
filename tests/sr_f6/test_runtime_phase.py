"""CPU provenance/mode fixtures; no real model loading or phase authorization."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mm_core.training import capture_rng, state_hash
from sr_f6 import phases, runtime
from sr_f6.transactions import RunTransactions


def source_context(phase="PILOT_V1"):
    return {
        "phase": phase,
        "status": "BOUND_NOT_AUTHORIZED",
        "training_allowed": False,
        "deployment": "/fixture/declared-deployment",
        "identity_sha256": "d" * 64,
        "origin_identity_sha256": "e" * 64,
        "identity": {
            "config_sha256": "c" * 64,
            "source_manifest": {"path": "SOURCE_MANIFEST.json", "sha256": "a" * 64},
            "data_manifest": {"path": "DATA/MANIFEST.json", "sha256": "b" * 64},
            "origin": {"identity": {"path": "IDENTITY.json", "sha256": "e" * 64}},
        },
    }


def test_phase_source_requires_pin_and_actual_deployment(monkeypatch, tmp_path):
    context = source_context()
    deployment = tmp_path / "actual"
    deployment.mkdir()
    (tmp_path / "SOURCE_MANIFEST.json").write_text(
        json.dumps({"files": [{"path": "src/fixture_only.py"}]})
    )
    monkeypatch.setattr(runtime, "_actual_deployment", lambda: deployment)
    calls = []

    def verify(root, phase, **kwargs):
        calls.append((root, phase, kwargs))
        assert kwargs == {"deployment": deployment, "expected_identity_sha256": "d" * 64}
        return copy.deepcopy(context)

    monkeypatch.setattr(phases, "verify_phase", verify)
    result = runtime.verify_runtime_source(
        tmp_path, phase="PILOT_V1", phase_identity_sha256="d" * 64
    )
    assert result == context
    assert len(calls) == 1
    for phase, pin in (("PILOT_V1", None), ("PILOT_V1", "short"), ("CENSUS_REVIEW_V1", "d" * 64)):
        with pytest.raises(PermissionError):
            runtime.verify_runtime_source(tmp_path, phase=phase, phase_identity_sha256=pin)
    assert len(calls) == 1


def test_default_new_deployment_cannot_claim_old_s0(monkeypatch, tmp_path):
    new = tmp_path / "new-deployment"
    monkeypatch.setattr(runtime, "_actual_deployment", lambda: new)

    def origin(root, *, deployment):
        assert deployment == new
        raise PermissionError("Not the historical S0 deployment")

    monkeypatch.setattr(phases, "verify_origin", origin)
    with pytest.raises(PermissionError, match="historical S0"):
        runtime.verify_runtime_source(tmp_path)
    with pytest.raises(PermissionError, match="explicit registered phase"):
        runtime.verify_runtime_source(tmp_path, phase_identity_sha256="d" * 64)


def test_loaded_module_must_come_from_declared_inventory(tmp_path):
    source = tmp_path / "deployment/src/local_package"
    source.mkdir(parents=True)
    file = source / "module.py"
    file.write_text("VALUE = 1\n")
    deployment = source.parents[1]
    manifest = {"files": [{"path": "src/local_package/module.py"}]}
    module = SimpleNamespace(__name__="local_package.module", __file__=str(file))
    runtime._verify_module_origins(deployment, manifest, [module])
    foreign = tmp_path / "foreign.py"
    foreign.write_text("VALUE = 1\n")
    with pytest.raises(PermissionError, match="outside"):
        runtime._verify_module_origins(
            deployment, manifest, [SimpleNamespace(__name__=module.__name__, __file__=str(foreign))]
        )


@pytest.fixture
def mocked_model_load(monkeypatch, tmp_path):
    """Keep real torch parameters while mocking unavailable GPU/model I/O."""
    import mm_dev.runtime as native

    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "adapter_config.json").write_text("{}")
    config = json.loads((Path(__file__).parents[2] / "configs/sr_f6.json").read_text())
    config["model"]["parents"] = {"b0": str(parent)}
    (tmp_path / "CONFIG.json").write_text(json.dumps(config))
    context, calls = source_context(), []

    class FakeRuntime(runtime.Runtime):
        def __init__(self, *args, **kwargs):
            calls.append("model")
            self.model = torch.nn.Linear(2, 1).bfloat16()
            self.identity = {"dtype": "bfloat16", "experiment_id": config["experiment_id"]}
            self.torch = torch

    monkeypatch.setattr(runtime, "Runtime", FakeRuntime)
    monkeypatch.setattr(
        runtime, "verify_runtime_source", lambda *args, **kwargs: copy.deepcopy(context)
    )
    monkeypatch.setattr(
        runtime,
        "verify_binding_inputs",
        lambda *args: ({"parents": {"b0": {"content_sha256": "f" * 64}}}, {"cached": True}),
    )
    monkeypatch.setattr(runtime, "configure_parent_training", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, "seed_all", lambda *args: None)
    monkeypatch.setattr(native, "configure_audited_backend", lambda: None)
    monkeypatch.setattr(native, "actual_cuda_identity", lambda: {"fixture_only": True})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    return SimpleNamespace(
        root=tmp_path, context=context, calls=calls, audit=tmp_path / "PHASE_RUNS/PILOT_V1/path"
    )


def test_training_source_run_and_attempt_identity_and_audit_are_separate(mocked_model_load):
    f = mocked_model_load
    r, optimizer, scheduler = runtime.load(
        f.root,
        "b0",
        "attempt-1",
        mode="TRAINING",
        phase="PILOT_V1",
        phase_identity_sha256="d" * 64,
        audit_root=f.audit,
        run_id="experiment/PILOT/b0/r0/U11",
    )
    assert optimizer is not None and scheduler is not None
    assert r.identity["run_id"] == "experiment/PILOT/b0/r0/U11"
    assert r.identity["task_id"] == "attempt-1"
    assert r.identity["source_sha256"] == "a" * 64
    assert r.identity["origin_identity_sha256"] == "e" * 64
    assert r.identity["phase_identity_sha256"] == "d" * 64
    assert r.identity["run_root"] == str(f.root)
    assert r.identity["audit_root"] == str(f.audit)
    assert r.identity["execution_authorization"] == "NOT_AUTHORIZED"
    assert (f.audit / "runtime_identity/attempt-1.json").is_file()
    assert not (f.root / "runtime_identity").exists()
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        r.generate_group({}, f.root, list(range(8)))


def test_new_phase_requires_independent_audit_and_stable_training_run(mocked_model_load):
    f = mocked_model_load
    args = dict(mode="TRAINING", phase="PILOT_V1", phase_identity_sha256="d" * 64)
    with pytest.raises(PermissionError, match="stable scientific run_id"):
        runtime.load(f.root, "b0", "attempt", audit_root=f.audit, **args)
    with pytest.raises(PermissionError, match="independent audit_root"):
        runtime.load(f.root, "b0", "attempt", run_id="science", **args)
    with pytest.raises(PermissionError, match="phase namespace"):
        runtime.load(f.root, "b0", "attempt", run_id="science", audit_root=f.root / "DATA", **args)
    assert f.calls == []


@pytest.mark.parametrize(
    "field",
    ["phase", "source_sha256", "data_hash", "origin_identity_sha256", "phase_identity_sha256"],
)
def test_phase_or_mode_relabel_cannot_bypass_loaded_binding(mocked_model_load, field):
    f = mocked_model_load
    r, _, _ = runtime.load(
        f.root,
        "b0",
        "attempt",
        mode="TRAINING",
        phase="PILOT_V1",
        phase_identity_sha256="d" * 64,
        audit_root=f.audit,
        run_id="science",
    )
    r.identity[field] = "CENSUS_S0" if field == "phase" else "0" * 64
    with pytest.raises(PermissionError, match="actual loaded source"):
        r.generate_group({}, f.root, list(range(8)))
    r.identity["execution_mode"] = "CENSUS"
    with pytest.raises(PermissionError, match="mode"):
        r.execution_mode()


def test_evaluation_parent_is_read_only_and_does_not_create_optimizer(
    mocked_model_load, monkeypatch
):
    import sr_f3_ce.training as native

    monkeypatch.setattr(
        native, "fresh_optimizer", lambda *args: pytest.fail("Evaluation created an optimizer")
    )
    f = mocked_model_load
    r, optimizer, scheduler = runtime.load(
        f.root,
        "b0",
        "eval-attempt",
        mode="EVALUATION",
        phase="PILOT_V1",
        phase_identity_sha256="d" * 64,
        audit_root=f.audit,
    )
    assert optimizer is scheduler is None
    assert all(not p.requires_grad for p in r.model.parameters())
    assert r.policy_tensor_identity() == r.identity["reference_hash"]
    row = {}
    r._annotate_runtime_row(row, "structured", r.policy_tensor_identity())
    assert row["step"] == 0 and row["execution_mode"] == "EVALUATION"
    with pytest.raises(PermissionError, match="gradient"):
        r._batch_forward({}, [[1]], purpose="training_policy", grad=True)
    r.identity["step"] = 12
    with pytest.raises(PermissionError, match="bound content"):
        r.policy_tensor_identity()


@pytest.mark.parametrize("construction_fails", [False, True])
def test_evaluation_parent_load_preserves_rng_on_success_and_error(
    mocked_model_load, monkeypatch, construction_fails
):
    import mm_dev.runtime as native

    f = mocked_model_load
    configured = []
    monkeypatch.setattr(native, "configure_audited_backend", lambda: configured.append(True))
    original_capture = runtime.capture_rng

    def capture_after_backend():
        assert configured, "Backend must be configured before CUDA RNG capture"
        return original_capture()

    monkeypatch.setattr(runtime, "capture_rng", capture_after_backend)
    if construction_fails:
        original = runtime.Runtime

        class FailingConstruction(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                raise RuntimeError("fixture construction failure after RNG consumption")

        monkeypatch.setattr(runtime, "Runtime", FailingConstruction)
    before = state_hash(capture_rng())
    args = dict(
        mode="EVALUATION",
        phase="PILOT_V1",
        phase_identity_sha256="d" * 64,
        audit_root=f.audit,
    )
    if construction_fails:
        with pytest.raises(RuntimeError, match="fixture construction failure"):
            runtime.load(f.root, "b0", "eval-rng", **args)
    else:
        runtime.load(f.root, "b0", "eval-rng", **args)
    assert state_hash(capture_rng()) == before


def test_audit_paths_cannot_follow_symlinks_or_other_phase(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "PHASE_RUNS/PILOT_V1/link"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)
    for path in (link / "child", tmp_path / "PHASE_RUNS/FORMAL_V1/path"):
        with pytest.raises(PermissionError):
            runtime._audit_directory(tmp_path, "PILOT_V1", path)


def transaction_fixture_module():
    # Reuse the existing CPU transaction fixture, not its test runner or GPU code.
    path = Path(__file__).with_name("test_transactions.py")
    spec = importlib.util.spec_from_file_location("_endpoint_cpu_transactions_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    f = transaction_fixture_module()
    training_runtime, optimizer, scheduler, identity = f.setup()
    parent_model = copy.deepcopy(training_runtime.model)
    directory = tmp_path / "PHASE_RUNS/PILOT_V1/training"
    with RunTransactions(directory, identity) as tx:
        tx.commit(training_runtime, optimizer, scheduler, step=0, sampler_state=f.zero_sampler())
        pending = f.rollout(directory, identity, training_runtime, 1, f.zero_sampler())
        update = f.tiny_update(training_runtime, optimizer, scheduler, identity, pending)
        marker = f.commit_step(tx, training_runtime, optimizer, scheduler, 1, pending, update)
    r = runtime.Runtime.__new__(runtime.Runtime)
    r.model, r.torch = parent_model, torch
    r._srf6_adapter_parameter_names = tuple(name for name, _ in parent_model.named_parameters())
    r.identity = {
        "run_root": str(tmp_path),
        "audit_root": str(tmp_path / "PHASE_RUNS/PILOT_V1/eval"),
        "phase": "PILOT_V1",
        "phase_identity_sha256": "d" * 64,
        "execution_mode": "EVALUATION",
        "run_id": "evaluation-science",
        "task_id": "evaluation-attempt",
        "parent_id": "b0",
        "step": 0,
        "source_sha256": identity["source_sha256"],
        "source_hash": identity["source_sha256"],
        "data_hash": identity["data_sha256"],
        "config_sha256": identity["config_sha256"],
        "reference_hash": identity["reference_sha256"],
        "reference_parent": "b0",
        "parent_content_hash": "f" * 64,
        "experiment_id": identity["experiment_id"],
        "origin_identity_sha256": "e" * 64,
    }
    r._srf6_source_context = source_context()
    r._srf6_execution_identity = {key: r.identity[key] for key in runtime.EXECUTION_IDENTITY_FIELDS}
    monkeypatch.setattr(runtime, "verify_runtime_source", lambda *args, **kwargs: source_context())
    runtime._freeze_evaluation(r, 0, identity["reference_sha256"])
    return SimpleNamespace(
        runtime=r, directory=directory, identity=identity, marker=marker, pending=pending
    )


def test_committed_endpoint_binds_actual_step_policy_without_restoring_training_rng(
    endpoint, monkeypatch
):
    import sr_f3_ce.training as native

    monkeypatch.setattr(
        native, "fresh_optimizer", lambda *args: pytest.fail("Endpoint created optimizer")
    )
    r = endpoint.runtime
    torch.rand(5)
    before_rng = state_hash(capture_rng())
    with RunTransactions(endpoint.directory, endpoint.identity) as tx:
        report = runtime.load_committed_endpoint(
            r, tx, expected_commit_sha256=endpoint.marker["commit_sha256"]
        )
    assert state_hash(capture_rng()) == before_rng
    assert report["status"] == "ENDPOINT_LOADED_NOT_AUTHORIZED"
    assert report["optimizer_created"] is False and report["training_rng_restored"] is False
    assert r.identity["step"] == 1 and r.identity["checkpoint_step"] == 1
    assert r.identity["checkpoint_run_id"] == endpoint.identity["run_id"]
    assert (
        r.identity["run_id"] == "evaluation-science"
        and r.identity["task_id"] == "evaluation-attempt"
    )
    assert r.policy_tensor_identity() == endpoint.marker["policy_sha256"]
    assert all(not p.requires_grad and p.grad is None for p in r.model.parameters())
    row = {}
    r._annotate_runtime_row(row, "structured", r.policy_tensor_identity())
    assert row["step"] == 1 and row["checkpoint_run_id"] == endpoint.identity["run_id"]
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        r.generate_group({}, Path(r.identity["run_root"]), list(range(8)))
    with torch.no_grad():
        next(r.model.parameters()).add_(0.1)
    with pytest.raises(PermissionError, match="bound content"):
        r.policy_tensor_identity()


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_pin",
        "unpublished",
        "state_content",
        "source",
        "same_run",
        "audit_outside",
        "task_traversal",
    ],
)
def test_endpoint_refuses_unbound_or_unpublished_checkpoint(endpoint, defect):
    r = endpoint.runtime
    pin = endpoint.marker["commit_sha256"]
    before = state_hash(dict(r.model.named_parameters()))
    if defect == "wrong_pin":
        pin = "0" * 64
    elif defect == "unpublished":
        current = endpoint.directory / "checkpoints/step-001"
        current.rename(current.with_name(".staging-step-001"))
    elif defect == "state_content":
        state = endpoint.directory / "checkpoints/step-001/state.pt"
        state.write_bytes(state.read_bytes() + b"changed")
    elif defect == "source":
        r.identity["source_sha256"] = "0" * 64
    elif defect == "audit_outside":
        r.identity["audit_root"] = str(Path(r.identity["run_root"]) / "outside-phase")
    elif defect == "task_traversal":
        r.identity["task_id"] = "../../escape"
    else:
        r.identity["run_id"] = endpoint.identity["run_id"]
    with (
        RunTransactions(endpoint.directory, endpoint.identity) as tx,
        pytest.raises(PermissionError),
    ):
        runtime.load_committed_endpoint(r, tx, expected_commit_sha256=pin)
    assert state_hash(dict(r.model.named_parameters())) == before
    assert r.identity["step"] == 0
    assert not (Path(r.identity["run_root"]) / "outside-phase").exists()


def test_endpoint_requires_live_transaction_lock(endpoint):
    tx = RunTransactions(endpoint.directory, endpoint.identity)
    with pytest.raises(PermissionError, match="lock"):
        runtime.load_committed_endpoint(
            endpoint.runtime, tx, expected_commit_sha256=endpoint.marker["commit_sha256"]
        )
