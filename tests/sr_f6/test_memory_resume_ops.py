"""Synthetic resume submission gates, no scheduler, SSH, or model."""

import importlib.util
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def load():
    p = Path(__file__).resolve().parents[2] / "scripts/sr_f6_memory_resume_ops/operations.py"
    spec = importlib.util.spec_from_file_location("resume_ops_test", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def setup(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    for key in ("OPS", "FULL_OPS", "ROOT", "CODE"):
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
    monkeypatch.setattr(
        m,
        "metadata",
        lambda *a: {
            "execution_source": {},
            "full_result_ref": {"path": "frozen-result", "sha256": "f" * 64},
        },
    )
    monkeypatch.setattr(m, "predecessor", lambda *a: {})
    monkeypatch.setattr(m, "transition_locks", lambda *a: nullcontext())
    monkeypatch.setattr(m.getpass, "getuser", lambda: m.USER)
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="90001\n", stderr="")

    monkeypatch.setattr(m.subprocess, "run", run)
    return m, a, c, capacity, calls, old, legacy


def test_unique_resume_last_legal_gpu(setup):
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
    assert not calls and not (m.OPS / "resume_ATTEMPT.json").exists()


@pytest.mark.parametrize(
    "name",
    [
        "resume_ATTEMPT.json",
        "resume_SUBMISSION.json",
        "resume_STARTED.json",
        "resume_COMPLETE.json",
        "resume_FAILURE.json",
        "EXECUTIONS",
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


def restore(m, name):
    original = getattr(load(), name)
    setattr(m, name, type(original)(original.__code__, m.__dict__))


def full_fixture(m, monkeypatch, old, legacy):
    refs = {
        "full_contract_ref": dict(
            path=str(m.FULL_OPS / "CONTRACT.json"), sha256=m.FULL_CONTRACT_SHA
        ),
        "full_result_ref": dict(
            path=str(m.FULL_OPS / "VALIDATION/RESULT.json"), sha256=m.FULL_RESULT_SHA
        ),
        "full_complete_ref": dict(
            path=str(m.FULL_OPS / "validate_COMPLETE.json"), sha256=m.FULL_COMPLETE_SHA
        ),
        "full_cold_ref": dict(
            path=str(m.FULL_OPS / "VALIDATION/COLD_VERIFICATION.json"), sha256=m.FULL_COLD_SHA
        ),
    }
    terminal = dict(
        job_id="235864",
        state="COMPLETED",
        exit_code="0:0",
        queue_gone=True,
        result_ref=refs["full_result_ref"],
        cold_verification_ref=refs["full_cold_ref"],
    )
    calls = []
    full = SimpleNamespace(
        predecessor=lambda *a: calls.append("historical"),
        validation_terminal=lambda *a: calls.append("terminal") or terminal,
    )
    monkeypatch.setattr(m, "frozen_full", lambda: (full, {}))
    monkeypatch.setattr(m, "small_ref", lambda *a: {})
    monkeypatch.setattr(m, "read", lambda *a: {})
    legacy.quiescent = lambda query: calls.append("quiescent") or {"step_state": "FAILED"}
    restore(m, "predecessor")
    return refs, terminal, calls


def test_exact_full_and_old_failure_chain_required(setup, monkeypatch):
    m, _, _, _, _, old, legacy = setup
    refs, _, calls = full_fixture(m, monkeypatch, old, legacy)
    assert m.predecessor(legacy, old, refs)["full"]["job_id"] == "235864"
    assert calls == ["historical", "terminal", "quiescent"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "235771"),
        ("state", "RUNNING"),
        ("exit_code", "1:0"),
        ("queue_gone", False),
        ("result_ref", {}),
        ("cold_verification_ref", {}),
    ],
)
def test_partial_terminal_never_admits_resume(setup, monkeypatch, field, value):
    m, _, _, _, _, old, legacy = setup
    refs, terminal, _ = full_fixture(m, monkeypatch, old, legacy)
    terminal[field] = value
    with pytest.raises(PermissionError):
        m.predecessor(legacy, old, refs)


@pytest.mark.parametrize(
    "name", ["full_contract_ref", "full_result_ref", "full_complete_ref", "full_cold_ref"]
)
def test_exact_all_four_references(setup, monkeypatch, name):
    m, _, _, _, _, old, legacy = setup
    refs, _, _ = full_fixture(m, monkeypatch, old, legacy)
    refs[name] = dict(refs[name], sha256="e" * 64)
    with pytest.raises(PermissionError):
        m.predecessor(legacy, old, refs)


@pytest.mark.parametrize(
    "version,day,name",
    [
        (1, "20261029", "resume_ATTEMPT.json"),
        (2, "20261029", "resume_SUBMISSION.json"),
        (3, "20261030", "EXECUTIONS"),
        (4, "20261030", "resume_FAILURE.json"),
        (5, "20261030", "resume_STARTED.json"),
        (6, "20261030", "resume_COMPLETE.json"),
        (7, "20261030", "resume_WATCHDOG_ATTEMPT.json"),
    ],
)
def test_any_historical_resume_evidence_blocks(setup, version, day, name):
    m, *_ = setup
    p = m.PROJECT / f"sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V{version}_{day}"
    p.mkdir(parents=True)
    (p / name).touch()
    with pytest.raises(PermissionError):
        m.no_prior_resume()


def test_empty_old_lock_is_not_an_intent(setup):
    m, *_ = setup
    p = m.PROJECT / "sr_f6_operations_20261027/PILOT_MEMORY_REPAIR_V4_20261030"
    p.mkdir(parents=True)
    (p / "resume.lock").touch()
    m.no_prior_resume()


@pytest.mark.parametrize("task", ["PRIMARY-b1-U10", "PRIMARY-b1-C_SHUF", "PRIMARY-b1-U11"])
def test_original_complete_must_not_repeat(setup, task):
    m, *_ = setup
    p = m.ROOT / "PHASE_RUNS/PILOT_V3/JOBS/TASKS" / task
    p.mkdir(parents=True)
    (p / "COMPLETE.json").write_text("{}")
    with pytest.raises(PermissionError):
        m.no_prior_resume()


def publish_task(m, old, contract, c, task, job="90001"):
    p = m.ROOT / "PHASE_RUNS/PILOT_V3/JOBS/TASKS" / task / "COMPLETE.json"
    p.parent.mkdir(parents=True)
    ext = m.OPS / "EXECUTIONS" / task
    ext.mkdir(parents=True)
    receipt = dict(
        schema="original",
        status="TASK_COMPLETE_NOT_GATE_SIGNED",
        task={"task_id": task},
        plan_ref={},
        producer={},
        gate_status="NOT_SIGNED",
    )
    cr = dict(path=str(c), sha256=m.sha(c))
    old.write_once(p, receipt)
    facts = dict(
        status="ACTUAL_STORAGE_CONTEXT_COMPLETE_NOT_SCIENTIFIC_PASS",
        contract_ref=cr,
        validation_ref=contract["full_result_ref"],
        execution_id=f"LH-SRF6-A-1-MEM-{job}-{task}",
        offload={},
    )
    old.write_once(ext / "STORAGE_FACTS.json", facts)
    extra = dict(
        receipt,
        offload={},
        resume_contract_ref=cr,
        execution_source=contract["execution_source"],
        full_validation_ref=contract["full_result_ref"],
        full_trainer_numerical_pass=False,
        storage_facts_ref=dict(
            path=str(ext / "STORAGE_FACTS.json"), sha256=m.sha(ext / "STORAGE_FACTS.json")
        ),
    )
    old.write_once(ext / "COMPLETE.json", extra)
    return p, ext


def ready_launch(setup, monkeypatch):
    m, a, c, _, calls, old, _ = setup
    m.submit(a, c, m.sha(c))
    monkeypatch.setenv("SLURM_JOB_ID", "90001")
    monkeypatch.setenv("SLURM_JOB_NAME", m.JOB_NAME)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    monkeypatch.setattr(m, "allocation", lambda *a: {})
    calls.clear()
    contract = m.metadata(a, c, m.sha(c))
    return m, a, c, calls, old, contract


def test_three_fresh_children_in_fixed_order_complete_only_after_publication(setup, monkeypatch):
    m, a, c, calls, old, contract = ready_launch(setup, monkeypatch)

    def run(argv, **kw):
        calls.append(argv)
        task = argv[argv.index("--task-id") + 1]
        assert not (m.OPS / "resume_COMPLETE.json").exists()
        assert str(m.FULL_DEPLOYMENT / "src") in argv[5]
        assert "sr_f6_memory_resume.execution" in argv[5]
        assert "validation_full" not in argv[5]
        publish_task(m, old, contract, c, task)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(m.subprocess, "run", run)
    out = m.launch(a, c, m.sha(c))
    assert [x[x.index("--task-id") + 1] for x in calls] == list(m.QUEUE)
    assert len(out["completed"]) == 3 and out["status"] == "QUEUE_COMPLETE_NOT_GATE_SIGNED"
    with pytest.raises(PermissionError):
        m.launch(a, c, m.sha(c))
    assert len(calls) == 3


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_child_failure_stops_remaining_queue_and_preserves_failure(
    setup, monkeypatch, failure_index
):
    m, a, c, calls, old, contract = ready_launch(setup, monkeypatch)

    def run(argv, **kw):
        calls.append(argv)
        if len(calls) - 1 == failure_index:
            return SimpleNamespace(returncode=1)
        publish_task(m, old, contract, c, argv[argv.index("--task-id") + 1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(m.subprocess, "run", run)
    with pytest.raises(PermissionError):
        m.launch(a, c, m.sha(c))
    assert len(calls) == failure_index + 1
    assert not (m.OPS / "resume_COMPLETE.json").exists()
    assert m.read(m.OPS / "resume_FAILURE.json")["retry_authorized"] is False


@pytest.mark.parametrize(
    "problem",
    ["wrong_task", "source", "validation", "facts", "contract", "failure", "numerical_claim"],
)
def test_success_exit_is_not_sufficient_without_bound_task_evidence(setup, problem):
    m, a, c, _, _, old, _ = setup
    contract = m.metadata(a, c, m.sha(c))
    task = m.QUEUE[0]
    _p, ext = publish_task(m, old, contract, c, task)
    receipt = m.read(ext / "COMPLETE.json")
    if problem == "wrong_task":
        receipt["task"] = {"task_id": "other"}
    if problem == "source":
        receipt["execution_source"] = {"revision": "other"}
    if problem == "validation":
        receipt["full_validation_ref"] = {}
    if problem == "facts":
        receipt["storage_facts_ref"]["sha256"] = "e" * 64
    if problem == "contract":
        receipt["resume_contract_ref"] = {}
    if problem == "failure":
        (ext / "FAILURE.json").write_text("{}")
    if problem == "numerical_claim":
        receipt["full_trainer_numerical_pass"] = True
    (ext / "COMPLETE.json").write_text(json.dumps(receipt))
    with pytest.raises(PermissionError):
        m.task_complete(contract, c, m.sha(c), task, "90001")


def test_zero_exit_without_publication_is_preserved_failure(setup, monkeypatch):
    m, a, c, calls, _old, _contract = ready_launch(setup, monkeypatch)
    monkeypatch.setattr(
        m.subprocess, "run", lambda *a, **kw: calls.append(a) or SimpleNamespace(returncode=0)
    )
    with pytest.raises(FileNotFoundError):
        m.launch(a, c, m.sha(c))
    assert len(calls) == 1
    assert (m.OPS / "resume_FAILURE.json").exists()
    assert not (m.OPS / "resume_COMPLETE.json").exists()


@pytest.mark.parametrize(
    "env,value",
    [("SLURM_RESTART_COUNT", "1"), ("SLURM_JOB_NAME", "other"), ("CUDA_VISIBLE_DEVICES", "0,1")],
)
def test_requeue_or_wrong_assignment_never_executes(setup, monkeypatch, env, value):
    m, a, c, calls, _old, _contract = ready_launch(setup, monkeypatch)
    monkeypatch.setenv(env, value)
    with pytest.raises(PermissionError):
        m.launch(a, c, m.sha(c))
    assert not calls


def test_submit_intent_is_written_before_scheduler_call(setup, monkeypatch):
    m, a, c, _, _, _, _ = setup

    def run(argv, **kw):
        intent = m.read(m.OPS / "resume_ATTEMPT.json")
        assert intent["status"] == "SUBMISSION_INTENT_NOT_COMPLETION"
        assert intent["argv"] == argv and "predecessor" in intent
        return SimpleNamespace(returncode=0, stdout="90001\n", stderr="")

    monkeypatch.setattr(m.subprocess, "run", run)
    assert m.submit(a, c, m.sha(c))["status"] == "SUBMITTED"
