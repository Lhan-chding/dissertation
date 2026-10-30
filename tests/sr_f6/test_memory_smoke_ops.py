"""Synthetic smoke submission gates, no scheduler, SSH, or model."""

import importlib.util
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def load():
    p = Path(__file__).resolve().parents[2] / "scripts/sr_f6_memory_smoke_ops/operations.py"
    spec = importlib.util.spec_from_file_location("smoke_ops_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def setup(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    for key in ("OPS", "V4_OPS", "ROOT", "CODE"):
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


def predecessor_fixture(m, monkeypatch, old, legacy):
    # Function globals must be the patched module: reload its original function source.
    import inspect

    exec(inspect.getsource(load().predecessor), m.__dict__)
    legacy.prior_allocation_terminal = lambda *a: None
    legacy.predecessor_terminal = lambda *a: None
    legacy.quiescent = lambda *a: None
    actual = dict(
        revision=m.V4_REVISION, deployment=str(m.V4_DEPLOYMENT), binding_sha256=m.V4_BINDING
    )

    def put(p, v):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(v))
        return dict(path=str(p), sha256=m.sha(p))

    contract = dict(
        prior_v4_contract_ref=dict(path=str(m.V4_OPS / "CONTRACT.json"), sha256=m.V4_CONTRACT_SHA)
    )
    restoration = {
        k: True
        for k in (
            "adapter_unchanged_during_replay",
            "reference_unchanged_during_replay",
            "adapter_restored",
            "rng_restored",
            "gradients_restored",
            "requires_grad_restored",
            "module_modes_restored",
            "frozen_base_buffer_versions_unchanged",
            "parameter_objects_unchanged",
        )
    }
    restoration.update(restoration_errors=[], runtime_requires_reload=False)
    partial = dict(restoration=restoration, counts=dict(generated_responses=0, optimizer_updates=0))
    value = dict(
        schema="srf6-memory-performance-transition-v1",
        status="TECHNICAL_SUPERSESSION_PRESERVED",
        predecessor_job_id="235653",
        predecessor_source_revision=m.V4_REVISION,
        predecessor_contract_ref=contract["prior_v4_contract_ref"],
        terminal_state="FAILED",
        exit_code="1:0",
        reason="user_requested_staged_qualification",
    )
    value["inner_failure_ref"] = put(
        m.V4_OPS / "VALIDATION/FAILURE.json",
        dict(status="FAILED_PRESERVED", error_type="KeyboardInterrupt"),
    )
    value["outer_failure_ref"] = put(
        m.V4_OPS / "validate_FAILURE.json",
        dict(status="FAILED_PRESERVED", job_id="235653", retry_authorized=False),
    )
    value["interruption_ref"] = put(
        m.OPS / "INTERRUPTION_RESULT.json",
        dict(
            status="EXACT_CHILD_SIGINT_SENT",
            reason="user_requested_staged_qualification",
            job_id="235653",
            source_revision=m.V4_REVISION,
            child_pid=769119,
            child_start_time="285882331",
            signal="SIGINT",
            returncode=0,
        ),
    )
    value["partial_evidence_ref"] = put(m.V4_OPS / "VALIDATION/PARTIAL_EVIDENCE.json", partial)
    value["partial_tensors_ref"] = put(m.V4_OPS / "VALIDATION/PARTIAL_TENSORS.pt", {})
    contract["predecessor_disposition_ref"] = put(m.OPS / "PREDECESSOR_DISPOSITION.json", value)
    attempt = dict(source=actual, contract=contract["prior_v4_contract_ref"], validation=None)
    ar = put(m.V4_OPS / "validate_ATTEMPT.json", attempt)
    put(
        m.V4_OPS / "validate_SUBMISSION.json",
        dict(status="SUBMITTED", job_id="235653", source=actual, attempt_sha256=ar["sha256"]),
    )
    old.query = lambda argv: "235653|FAILED|1:0|\n" if argv[0] == "sacct" else ""
    return contract, value, partial, put


def test_controlled_terminal_preserved(setup, monkeypatch):
    m, _, _, _, _, old, legacy = setup
    c, *_ = predecessor_fixture(m, monkeypatch, old, legacy)
    assert m.predecessor(legacy, old, c)["status"] == "TECHNICAL_SUPERSESSION_PRESERVED"


@pytest.mark.parametrize("state", ["RUNNING", "COMPLETING", "COMPLETED", "UNKNOWN"])
def test_predecessor_not_controlled_failure_blocks(setup, monkeypatch, state):
    m, _, _, _, _, old, legacy = setup
    c, *_ = predecessor_fixture(m, monkeypatch, old, legacy)
    old.query = lambda argv: f"235653|{state}|1:0|\n" if argv[0] == "sacct" else ""
    with pytest.raises(PermissionError):
        m.predecessor(legacy, old, c)


@pytest.mark.parametrize(
    "problem",
    ["queued", "resume", "result", "numerical_failure", "restore", "generated", "wrong_stop"],
)
def test_predecessor_unsafe_evidence_blocks(setup, monkeypatch, problem):
    m, _, _, _, _, old, legacy = setup
    c, v, p, put = predecessor_fixture(m, monkeypatch, old, legacy)
    if problem == "queued":
        old.query = lambda argv: "235653|FAILED|1:0|\n" if argv[0] == "sacct" else "235653"
    elif problem == "resume":
        (m.V4_OPS / "resume_ATTEMPT.json").touch()
    elif problem == "result":
        (m.V4_OPS / "VALIDATION/RESULT.json").touch()
    elif problem == "numerical_failure":
        v["inner_failure_ref"] = put(
            m.V4_OPS / "VALIDATION/FAILURE.json",
            dict(status="FAILED_PRESERVED", error_type="AssertionError"),
        )
    elif problem in ("restore", "generated"):
        if problem == "restore":
            p["restoration"]["rng_restored"] = False
        else:
            p["counts"]["generated_responses"] = 1
        v["partial_evidence_ref"] = put(m.V4_OPS / "VALIDATION/PARTIAL_EVIDENCE.json", p)
    else:
        s = m.read(m.OPS / "INTERRUPTION_RESULT.json")
        s["child_pid"] = 1
        v["interruption_ref"] = put(m.OPS / "INTERRUPTION_RESULT.json", s)
    c["predecessor_disposition_ref"] = put(m.OPS / "PREDECESSOR_DISPOSITION.json", v)
    with pytest.raises(PermissionError):
        m.predecessor(legacy, old, c)


def test_smoke_module_import_is_standard_library_only():
    path = Path(__file__).resolve().parents[2] / "src/sr_f6_memory_repair/smoke.py"
    result = subprocess.run(
        [
            __import__("sys").executable,
            "-I",
            "-B",
            "-c",
            "import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('smoke_alone',{str(path)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "assert not any(k=='torch' or k=='sr_f6' or k.startswith('transformers') "
            "for k in sys.modules);"
            "assert m.SMOKE_PLAN['generated_responses']==0;"
            "assert len(m.SMOKE_PLAN['measurements'])==4",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


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
