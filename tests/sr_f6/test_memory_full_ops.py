"""Synthetic full validation submission gates, no scheduler, SSH, or model."""

import importlib.util
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def load():
    p = Path(__file__).resolve().parents[2] / "scripts/sr_f6_memory_full_ops/operations.py"
    spec = importlib.util.spec_from_file_location("full_ops_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def setup(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    for key in ("OPS", "V4_OPS", "V6_OPS", "ROOT", "CODE"):
        path = tmp_path / key
        path.mkdir()
        monkeypatch.setattr(m, key, path)
    a = dict(revision="a" * 40, deployment=str(tmp_path / "source"), binding_sha256="b" * 64)
    c = m.OPS / "CONTRACT.json"
    c.write_text("{}")
    capacity = dict(gpu_limit=8, submit_limit=5, reserved_gpus=7, submitted_by_user=4)

    def write(p, v):
        with p.open("x") as f:
            json.dump(v, f)

    old = SimpleNamespace(
        clean_submission_environment=lambda: None,
        metadata=lambda: {},
        scheduler=lambda _: (SimpleNamespace(capacity=lambda **_: capacity), None),
        query=lambda _: "",
        write_once=write,
        now=lambda: "SYNTHETIC",
    )
    legacy = SimpleNamespace(original=lambda: old, lock=lambda *a, **k: nullcontext())
    monkeypatch.setattr(m, "frozen_v4", lambda: (legacy, {}))
    monkeypatch.setattr(m, "metadata", lambda *a: {})
    monkeypatch.setattr(m, "predecessor", lambda *a: {})
    monkeypatch.setattr(m, "transition_locks", lambda *a: nullcontext())
    monkeypatch.setattr(m.getpass, "getuser", lambda: m.USER)
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="90001\n", stderr="")

    monkeypatch.setattr(m.subprocess, "run", run)
    return m, a, c, capacity, calls, old, legacy


def test_unique_validate_last_legal_gpu(setup):
    m, a, c, _, calls, *_ = setup
    assert m.submit(a, c, m.sha(c))["status"] == "SUBMITTED"
    assert len(calls) == 1 and "--gres=gpu:pro6000:1" in calls[0]
    assert "--time=23:00:00" in calls[0] and "--no-requeue" in calls[0]
    with pytest.raises(PermissionError):
        m.submit(a, c, m.sha(c))
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [("reserved_gpus", 8), ("submitted_by_user", 5)])
def test_capacity_does_not_write_intent(setup, field, value):
    m, a, c, cap, calls, *_ = setup
    cap[field] = value
    assert m.submit(a, c, m.sha(c))["status"] == "NO_LEGAL_CAPACITY"
    assert not calls and not (m.OPS / "validate_ATTEMPT.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "validate_ATTEMPT.json",
        "validate_SUBMISSION.json",
        "validate_STARTED.json",
        "validate_COMPLETE.json",
        "validate_FAILURE.json",
        "VALIDATION",
    ],
)
def test_any_prior_partial_blocks(setup, name):
    m, a, c, _, calls, *_ = setup
    (m.OPS / name).touch()
    with pytest.raises(PermissionError):
        m.submit(a, c, m.sha(c))
    assert not calls


def test_unknown_submission_is_retained(setup, monkeypatch):
    m, a, c, *_ = setup

    def fail(*a, **k):
        raise subprocess.TimeoutExpired("sbatch", 25)

    monkeypatch.setattr(m.subprocess, "run", fail)
    assert m.submit(a, c, m.sha(c))["status"] == "SUBMISSION_UNKNOWN"
    with pytest.raises(PermissionError):
        m.submit(a, c, m.sha(c))


@pytest.mark.parametrize("bad", ["gpu", "cpu", "memory", "account", "user"])
def test_actual_allocation_must_match(setup, bad):
    m, _, _, _, _, old, _ = setup
    fields = dict(
        JobId="90001",
        JobName=m.JOB_NAME,
        QOS=m.QOS,
        Partition="cluster02",
        Account="rose",
        UserId="varun024(1)",
        ReqTRES="cpu=4,mem=33G,gres/gpu=1,gres/gpu:pro6000=1",
        AllocTRES="cpu=4,mem=33G,gres/gpu=1,gres/gpu:pro6000=1",
    )
    old.query = lambda _: " ".join(k + "=" + v for k, v in fields.items())
    assert m.allocation(old, "90001")["JobId"] == "90001"
    if bad == "gpu":
        fields["AllocTRES"] = fields["AllocTRES"].replace("gpu:pro6000=1", "gpu:pro6000=2")
    elif bad == "cpu":
        fields["ReqTRES"] = fields["ReqTRES"].replace("cpu=4", "cpu=8")
    elif bad == "memory":
        fields["AllocTRES"] = fields["AllocTRES"].replace("33G", "128G")
    elif bad == "account":
        fields["Account"] = "other"
    else:
        fields["UserId"] = "other(2)"
    with pytest.raises(PermissionError):
        m.allocation(old, "90001")


def reference_fixture(m, monkeypatch, old, legacy):
    actual = dict(
        revision=m.V6_REVISION, deployment=str(m.V6_DEPLOYMENT), binding_sha256=m.V6_BINDING
    )
    c = dict(path=str(m.V6_OPS / "CONTRACT.json"), sha256=m.V6_CONTRACT_SHA)
    r = dict(path=str(m.V6_OPS / "SMOKE/RESULT.json"), sha256=m.V6_RESULT_SHA)
    cold = dict(path=str(m.V6_OPS / "SMOKE/COLD_VERIFICATION.json"), sha256="a" * 64)
    required = dict(
        prior_smoke_contract_ref=c,
        prior_smoke_result_ref=r,
        prior_smoke_complete_ref=dict(path=str(m.V6_OPS / "smoke_COMPLETE.json"), sha256="a" * 64),
        prior_smoke_cold_ref=cold,
    )
    prior_checks = []
    prior_module = SimpleNamespace(predecessor=lambda *a: prior_checks.append(a))
    monkeypatch.setattr(m, "frozen_v6", lambda: (prior_module, actual))
    values = {
        "CONTRACT.json": {"schema": "frozen-prior"},
        "SMOKE/RESULT.json": {
            "status": "SMOKE_NUMERICAL_PERFORMANCE_MEASURED",
            "admission": "NOT_FULL_BATCH_ADMISSION",
            "cold_verification_ref": cold,
        },
        "SMOKE/COLD_VERIFICATION.json": {
            "status": "VERIFIED_ACTUAL_TWO_ROW_REUSED_REFERENCE_SMOKE"
        },
        "smoke_ATTEMPT.json": dict(source=actual, contract=c),
        "smoke_SUBMISSION.json": dict(
            status="SUBMITTED", job_id="235771", source=actual, attempt_sha256="a" * 64
        ),
        "smoke_COMPLETE.json": dict(
            status="COMPLETED_STAGED_SMOKE_NOT_FULL_BATCH_ADMISSION",
            job_id="235771",
            source=actual,
            contract_ref=c,
            result_ref=r,
        ),
    }
    monkeypatch.setattr(m, "read", lambda p: values[str(Path(p).relative_to(m.V6_OPS))])
    monkeypatch.setattr(m, "sha", lambda p: "a" * 64)
    monkeypatch.setattr(
        m, "small_ref", lambda base, ref: values[str(Path(ref["path"]).relative_to(m.V6_OPS))]
    )
    old.query = lambda a: "235771|COMPLETED|0:0\n" if a[0] == "sacct" else ""
    return required, values, prior_checks


def test_successful_v6_reference_and_original_failure_chain_required(setup, monkeypatch):
    m, _, _, _, _, old, legacy = setup
    check = load().predecessor
    contract, _, called = reference_fixture(m, monkeypatch, old, legacy)
    # Use the real implementation with this module's patched globals.
    monkeypatch.setattr(m, "predecessor", type(check)(check.__code__, m.__dict__))
    assert m.predecessor(legacy, old, contract)["job_id"] == "235771"
    assert len(called) == 1


@pytest.mark.parametrize(
    "problem",
    [
        "running",
        "queued",
        "failed",
        "exit",
        "reference",
        "source",
        "completion",
        "intent",
        "watchdog",
        "inner_failure",
    ],
)
def test_invalid_reference_never_allows_new_submit(setup, monkeypatch, problem):
    m, _, _, _, _, old, legacy = setup
    check = load().predecessor
    monkeypatch.setattr(m, "predecessor", type(check)(check.__code__, m.__dict__))
    c, v, _ = reference_fixture(m, monkeypatch, old, legacy)
    if problem == "running":
        old.query = lambda a: "235771|RUNNING|0:0\n" if a[0] == "sacct" else ""
    if problem == "queued":
        old.query = lambda a: "235771|COMPLETED|0:0\n" if a[0] == "sacct" else "235771\n"
    if problem == "failed":
        old.query = lambda a: "235771|FAILED|1:0\n" if a[0] == "sacct" else ""
    if problem == "exit":
        old.query = lambda a: "235771|COMPLETED|1:0\n" if a[0] == "sacct" else ""
    if problem == "reference":
        c["prior_smoke_result_ref"] = dict(c["prior_smoke_result_ref"], sha256="b" * 64)
    if problem == "source":
        v["smoke_SUBMISSION.json"]["source"] = {}
    if problem == "completion":
        v["smoke_COMPLETE.json"]["result_ref"] = {}
    if problem == "intent":
        v["smoke_ATTEMPT.json"]["contract"] = {}
    if problem == "watchdog":
        (m.V6_OPS / "smoke_WATCHDOG_ATTEMPT.json").touch()
    if problem == "inner_failure":
        (m.V6_OPS / "SMOKE").mkdir()
        (m.V6_OPS / "SMOKE/FAILURE.json").touch()
    with pytest.raises(PermissionError):
        m.predecessor(legacy, old, c)


@pytest.mark.parametrize(
    "problem", ["cold_status", "cold_ref", "complete_ref", "admission", "completing", "unknown"]
)
def test_v6_cold_proof_and_terminal_required(setup, monkeypatch, problem):
    m, _, _, _, _, old, legacy = setup
    check = load().predecessor
    monkeypatch.setattr(m, "predecessor", type(check)(check.__code__, m.__dict__))
    c, v, _ = reference_fixture(m, monkeypatch, old, legacy)
    if problem == "cold_status":
        v["SMOKE/COLD_VERIFICATION.json"]["status"] = "UNKNOWN"
    elif problem == "cold_ref":
        c["prior_smoke_cold_ref"] = {}
    elif problem == "complete_ref":
        c["prior_smoke_complete_ref"] = {}
    elif problem == "admission":
        v["SMOKE/RESULT.json"]["admission"] = "FULL_ADMISSION"
    else:
        old.query = lambda a: f"235771|{problem.upper()}|0:0\n" if a[0] == "sacct" else ""
    with pytest.raises(PermissionError):
        m.predecessor(legacy, old, c)


def terminal_fixture(setup, monkeypatch):
    m, a, c, _, _, old, _ = setup
    cr = dict(path=str(c), sha256=m.sha(c))
    rr = dict(path=str(m.OPS / "VALIDATION/RESULT.json"), sha256="d" * 64)
    cold = dict(path=str(m.OPS / "VALIDATION/COLD_VERIFICATION.json"), sha256="e" * 64)
    old.write_once(
        m.OPS / "validate_ATTEMPT.json",
        dict(source=a, contract=cr, argv=m.argv_for(a, c, cr["sha256"])),
    )
    old.write_once(
        m.OPS / "validate_SUBMISSION.json",
        dict(
            status="SUBMITTED",
            job_id="90001",
            source=a,
            attempt_sha256=m.sha(m.OPS / "validate_ATTEMPT.json"),
        ),
    )
    complete = dict(
        status="COMPLETED_FULL_VALIDATION_NO_TRAINING",
        job_id="90001",
        source=a,
        contract_ref=cr,
        result_ref=rr,
        cold_verification_ref=cold,
    )
    old.write_once(m.OPS / "validate_COMPLETE.json", complete)
    monkeypatch.setattr(
        m,
        "contract_module",
        lambda _: SimpleNamespace(
            verify_validation_metadata=lambda *a: dict(
                status="VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH", cold_verification_ref=cold
            )
        ),
    )
    old.query = lambda args: "90001|COMPLETED|0:0\n" if args[0] == "sacct" else ""
    return m, a, c, cr, rr, cold, old, complete


def test_terminal_requires_completed_bound_full_cold_proof(setup, monkeypatch):
    m, a, c, cr, rr, _, old, _ = terminal_fixture(setup, monkeypatch)
    assert m.validation_terminal(old, a, c, cr["sha256"], rr)["queue_gone"] is True


@pytest.mark.parametrize(
    "problem", ["failure", "partial_cold", "queued", "running", "failed", "source", "result"]
)
def test_terminal_rejects_partial_unknown_or_mismatched_results(setup, monkeypatch, problem):
    m, a, c, cr, rr, _cold, old, complete = terminal_fixture(setup, monkeypatch)
    if problem == "failure":
        (m.OPS / "validate_FAILURE.json").touch()
    elif problem == "queued":
        old.query = lambda x: "90001|COMPLETED|0:0\n" if x[0] == "sacct" else "90001"
    elif problem in ("running", "failed"):
        old.query = lambda x: f"90001|{problem.upper()}|0:0\n" if x[0] == "sacct" else ""
    else:
        if problem == "partial_cold":
            complete["cold_verification_ref"] = {}
        elif problem == "source":
            complete["source"] = {}
        else:
            complete["result_ref"] = {}
        (m.OPS / "validate_COMPLETE.json").write_text(json.dumps(complete))
    with pytest.raises(PermissionError):
        m.validation_terminal(old, a, c, cr["sha256"], rr)


def test_launch_uses_full_gpu_entry_without_smoke_timeout(setup, monkeypatch):
    m, a, c, _, _calls, _old, _ = setup
    cr = dict(path=str(c), sha256=m.sha(c))
    m.submit(a, c, cr["sha256"])
    monkeypatch.setenv("SLURM_JOB_ID", "90001")
    monkeypatch.setenv("SLURM_JOB_NAME", m.JOB_NAME)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(m, "allocation", lambda *a: {})
    cold = dict(path=str(m.OPS / "VALIDATION/COLD_VERIFICATION.json"), sha256="e" * 64)
    monkeypatch.setattr(
        m,
        "contract_module",
        lambda _: SimpleNamespace(
            verify_validation_metadata=lambda *a: dict(
                status="VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH", cold_verification_ref=cold
            )
        ),
    )
    child = []

    def run(argv, **kw):
        child.append((argv, kw))
        (m.OPS / "VALIDATION").mkdir()
        (m.OPS / "VALIDATION/RESULT.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(m.subprocess, "run", run)
    complete = m.launch(a, c, cr["sha256"])
    assert len(child) == 1 and "sr_f6_memory_repair.validation_full" in child[0][0][5]
    assert "timeout" not in child[0][1]
    assert complete["cold_verification_ref"] == cold
    assert complete["status"] == "COMPLETED_FULL_VALIDATION_NO_TRAINING"
    with pytest.raises(PermissionError):
        m.launch(a, c, cr["sha256"])
    assert len(child) == 1
