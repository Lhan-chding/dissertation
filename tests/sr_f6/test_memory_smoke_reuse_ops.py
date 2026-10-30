"""Synthetic smoke submission gates, no scheduler, SSH, or model."""

import importlib.util
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def load():
    p = Path(__file__).resolve().parents[2] / "scripts/sr_f6_memory_smoke_reuse_ops/operations.py"
    spec = importlib.util.spec_from_file_location("smoke_reuse_ops_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def setup(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    for key in ("OPS", "V4_OPS", "V5_OPS", "ROOT", "CODE"):
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


def test_unique_smoke_last_legal_gpu(setup):
    m, a, c, _, calls, *_ = setup
    assert m.submit(a, c, m.sha(c))["status"] == "SUBMITTED"
    assert len(calls) == 1 and "--gres=gpu:pro6000:1" in calls[0]
    assert "--time=00:35:00" in calls[0] and "--no-requeue" in calls[0]
    with pytest.raises(PermissionError):
        m.submit(a, c, m.sha(c))
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [("reserved_gpus", 8), ("submitted_by_user", 5)])
def test_capacity_does_not_write_intent(setup, field, value):
    m, a, c, cap, calls, *_ = setup
    cap[field] = value
    assert m.submit(a, c, m.sha(c))["status"] == "NO_LEGAL_CAPACITY"
    assert not calls and not (m.OPS / "smoke_ATTEMPT.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "smoke_ATTEMPT.json",
        "smoke_SUBMISSION.json",
        "smoke_STARTED.json",
        "smoke_COMPLETE.json",
        "smoke_FAILURE.json",
        "SMOKE",
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


@pytest.mark.parametrize("hard_kill", [False, True])
def test_watchdog_interrupts_exact_child_with_bounded_grace(setup, monkeypatch, hard_kill):
    m, _, _, _, _, old, _ = setup
    events = []

    class Child:
        pid = 1234

        def wait(self, timeout):
            events.append(("wait", timeout))
            if timeout == 1650 or (timeout == 150 and hard_kill):
                raise subprocess.TimeoutExpired("child", timeout)
            return -9 if hard_kill else -2

        def send_signal(self, sig):
            events.append(("signal", sig))

        def kill(self):
            events.append(("kill",))

    monkeypatch.setattr(m.subprocess, "Popen", lambda argv, env: Child())
    result = m.run_child(["python"], {}, old)
    assert result["timed_out"] and result["killed"] is hard_kill
    assert events[:3] == [("wait", 1650), ("signal", m.signal.SIGINT), ("wait", 150)]
    assert ("kill",) in events if hard_kill else ("kill",) not in events
    assert m.read(m.OPS / "smoke_WATCHDOG_ATTEMPT.json")["child_pid"] == 1234
    assert m.read(m.OPS / "smoke_WATCHDOG_RESULT.json")["retry_authorized"] is False


def test_watchdog_success_never_signals(setup, monkeypatch):
    m, _, _, _, _, old, _ = setup
    child = SimpleNamespace(pid=1234, wait=lambda timeout: 0)
    monkeypatch.setattr(m.subprocess, "Popen", lambda argv, env: child)
    assert m.run_child(["python"], {}, old) == dict(returncode=0, timed_out=False, killed=False)
    assert not (m.OPS / "smoke_WATCHDOG_ATTEMPT.json").exists()


def reference_fixture(m, monkeypatch, old, legacy):
    actual = dict(
        revision=m.V5_REVISION, deployment=str(m.V5_DEPLOYMENT), binding_sha256=m.V5_BINDING
    )
    c = dict(path=str(m.V5_OPS / "CONTRACT.json"), sha256=m.V5_CONTRACT_SHA)
    r = dict(path=str(m.V5_OPS / "SMOKE/RESULT.json"), sha256=m.V5_RESULT_SHA)
    required = dict(prior_smoke_contract_ref=c, prior_smoke_result_ref=r)
    prior_checks = []
    prior_module = SimpleNamespace(predecessor=lambda *a: prior_checks.append(a))
    monkeypatch.setattr(m, "frozen_v5", lambda: (prior_module, actual))
    values = {
        "CONTRACT.json": {"schema": "frozen-prior"},
        "SMOKE/RESULT.json": {"status": "SMOKE_NUMERICAL_PERFORMANCE_MEASURED"},
        "smoke_ATTEMPT.json": dict(source=actual, contract=c),
        "smoke_SUBMISSION.json": dict(
            status="SUBMITTED", job_id="235713", source=actual, attempt_sha256="a" * 64
        ),
        "smoke_COMPLETE.json": dict(
            status="COMPLETED_STAGED_SMOKE_NOT_FULL_BATCH_ADMISSION",
            job_id="235713",
            source=actual,
            contract_ref=c,
            result_ref=r,
        ),
    }
    monkeypatch.setattr(m, "read", lambda p: values[str(Path(p).relative_to(m.V5_OPS))])
    monkeypatch.setattr(m, "sha", lambda p: "a" * 64)
    monkeypatch.setattr(
        m, "small_ref", lambda base, ref: values[str(Path(ref["path"]).relative_to(m.V5_OPS))]
    )
    old.query = lambda a: "235713|COMPLETED|0:0\n" if a[0] == "sacct" else ""
    return required, values, prior_checks


def test_successful_v5_reference_and_original_failure_chain_required(setup, monkeypatch):
    m, _, _, _, _, old, legacy = setup
    check = load().predecessor
    contract, _, called = reference_fixture(m, monkeypatch, old, legacy)
    # Use the real implementation with this module's patched globals.
    monkeypatch.setattr(m, "predecessor", type(check)(check.__code__, m.__dict__))
    assert m.predecessor(legacy, old, contract)["job_id"] == "235713"
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
        old.query = lambda a: "235713|RUNNING|0:0\n" if a[0] == "sacct" else ""
    if problem == "queued":
        old.query = lambda a: "235713|COMPLETED|0:0\n" if a[0] == "sacct" else "235713\n"
    if problem == "failed":
        old.query = lambda a: "235713|FAILED|1:0\n" if a[0] == "sacct" else ""
    if problem == "exit":
        old.query = lambda a: "235713|COMPLETED|1:0\n" if a[0] == "sacct" else ""
    if problem == "reference":
        c["prior_smoke_result_ref"] = dict(c["prior_smoke_result_ref"], sha256="b" * 64)
    if problem == "source":
        v["smoke_SUBMISSION.json"]["source"] = {}
    if problem == "completion":
        v["smoke_COMPLETE.json"]["result_ref"] = {}
    if problem == "intent":
        v["smoke_ATTEMPT.json"]["contract"] = {}
    if problem == "watchdog":
        (m.V5_OPS / "smoke_WATCHDOG_ATTEMPT.json").touch()
    if problem == "inner_failure":
        (m.V5_OPS / "SMOKE").mkdir()
        (m.V5_OPS / "SMOKE/FAILURE.json").touch()
    with pytest.raises(PermissionError):
        m.predecessor(legacy, old, c)
