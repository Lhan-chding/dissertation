"""Submission, allocation and independent peer failure tests; no real scheduler."""

import hashlib
import importlib.util
import json
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def load():
    path = Path(__file__).resolve().parents[2] / "scripts/sr_f6_round_b_ops/operations.py"
    spec = importlib.util.spec_from_file_location("round_b_ops_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream)


@pytest.fixture
def f(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, "PROJECT", tmp_path)
    for key in ("OPS", "ROOT", "CODE", "JOBS"):
        path = tmp_path / key
        path.mkdir()
        monkeypatch.setattr(m, key, path)
    for task in m.QUEUES[0]:
        run = (
            m.ROOT
            / "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/runs/b2/r0"
            / task.removeprefix("PRIMARY-b2-")
        )
        save(run / "RUN_IDENTITY.json", {})
        save(run / "ROLLOUT_REGISTRATION.json", {})
    actual = dict(revision="a" * 40, deployment=str(tmp_path / "source"), binding_sha256="b" * 64)
    path = m.OPS / "CONTRACT.json"
    path.write_text("{}")
    capacity = dict(gpu_limit=8, submit_limit=5, reserved_gpus=6, submitted_by_user=4)
    old = SimpleNamespace(
        clean_submission_environment=lambda: None,
        metadata=lambda: {},
        scheduler=lambda _: (SimpleNamespace(capacity=lambda **_: capacity), None),
        query=lambda _: "",
        write_once=save,
        now=lambda: "synthetic",
    )
    legacy = SimpleNamespace(original=lambda: old, lock=lambda *a, **k: nullcontext())
    previous = SimpleNamespace(
        frozen_v4=lambda: (legacy, {}),
        sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
        read=lambda p: json.loads(p.read_bytes()),
        canonical=lambda p, **_: p,
        transition_locks=lambda _: nullcontext(),
        OPS=tmp_path / "v8",
    )
    monkeypatch.setattr(m, "frozen_resume", lambda: previous)
    contract = {"execution_source": {}, "a_delivery_ref": {}}
    monkeypatch.setattr(m, "metadata", lambda *a: contract)
    monkeypatch.setattr(m, "terminal", lambda *a: {"state": "COMPLETED"})
    monkeypatch.setattr(m.getpass, "getuser", lambda: m.USER)
    calls = []

    def run(argv, **kw):
        assert (m.OPS / "round_b_ATTEMPT.json").exists()
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="90001\n", stderr="")

    monkeypatch.setattr(m.subprocess, "run", run)
    return SimpleNamespace(**locals())


def test_first_submission_from_absent_worker_and_task_dirs_creates_only_locks(f):
    @contextmanager
    def real_parent_sensitive_lock(path, **kwargs):
        with path.open("a+b"):
            yield

    f.previous.OPS.mkdir()
    f.legacy.lock = real_parent_sensitive_lock
    assert not (f.m.JOBS / "WORKERS").exists()
    assert f.m.submit(f.actual, f.path, "c" * 64)["status"] == "SUBMITTED"
    assert len(f.calls) == 1 and "--gres=gpu:pro6000:2" in f.calls[0]
    assert "--cpus-per-task=8" in f.calls[0] and "--mem=66G" in f.calls[0]
    assert "--no-requeue" in f.calls[0] and "--time=23:00:00" in f.calls[0]
    with pytest.raises(PermissionError, match="prior/partial"):
        f.m.submit(f.actual, f.path, "c" * 64)
    assert len(f.calls) == 1


@pytest.mark.parametrize("field,value", [("reserved_gpus", 7), ("submitted_by_user", 5)])
def test_capacity_denial_is_reversible_without_intent(f, field, value):
    f.capacity[field] = value
    assert f.m.submit(f.actual, f.path, "c" * 64)["status"] == "NO_LEGAL_CAPACITY"
    assert not f.calls and not (f.m.OPS / "round_b_ATTEMPT.json").exists()


@pytest.mark.parametrize(
    "artifact",
    [
        "round_b_ATTEMPT.json",
        "round_b_SUBMISSION.json",
        "round_b_FAILURE.json",
        ".pending-round_b_ATTEMPT.json-incomplete",
        "EXECUTIONS",
        "LANES",
    ],
)
def test_unknown_or_partial_round_never_retries(f, artifact):
    (f.m.OPS / artifact).touch()
    with pytest.raises(PermissionError, match="prior/partial"):
        f.m.submit(f.actual, f.path, "c" * 64)
    assert not f.calls


def test_scheduler_timeout_retains_unknown_and_blocks_second_submission(f, monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("sbatch", 25)

    monkeypatch.setattr(f.m.subprocess, "run", timeout)
    assert f.m.submit(f.actual, f.path, "c" * 64)["status"] == "SUBMISSION_UNKNOWN"
    with pytest.raises(PermissionError):
        f.m.submit(f.actual, f.path, "c" * 64)


@pytest.mark.parametrize(
    "artifact", ["rollouts", "NUMERICAL_TOKEN_AUDIT.jsonl", "checkpoints", "unknown-output"]
)
def test_any_nonregistration_primary_output_blocks_submission(f, artifact):
    run = f.m.ROOT / "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/runs/b2/r0/U10"
    run.mkdir(parents=True, exist_ok=True)
    (run / artifact).touch()
    with pytest.raises(PermissionError, match="b2 model output"):
        f.m.submit(f.actual, f.path, "c" * 64)
    assert not f.calls


def test_recovery_staging_is_not_treated_as_new_recovery(f):
    run = f.m.ROOT / "PHASE_RUNS/PILOT_V3/RECOVERY/b0/r0"
    (run / ".recovery-staging-interrupted").mkdir(parents=True)
    with pytest.raises(PermissionError, match="staging"):
        f.m.submit(f.actual, f.path, "c" * 64)
    assert not f.calls


@pytest.mark.parametrize("codes", [[1, 0], [0, 1], [0, 0]])
def test_both_lanes_start_and_are_waited_without_killing_peer(f, monkeypatch, codes):
    events = []

    def popen(argv, **kwargs):
        worker = int(argv[-1])
        events.append(("start", worker))
        assert "--kill-on-bad-exit=0" in argv and "--gres=gpu:pro6000:1" in argv
        return SimpleNamespace(wait=lambda: events.append(("wait", worker)) or codes[worker])

    monkeypatch.setattr(f.m.subprocess, "Popen", popen)
    assert f.m.run_lanes(f.actual, f.path, "c" * 64, {}) == codes
    assert events == [("start", 0), ("start", 1), ("wait", 0), ("wait", 1)]


def test_failure_to_create_peer_still_waits_started_lane(f, monkeypatch):
    events = []

    def popen(argv, **kwargs):
        if int(argv[-1]) == 1:
            raise OSError("creation failed")
        return SimpleNamespace(wait=lambda: events.append("waited") or 0)

    monkeypatch.setattr(f.m.subprocess, "Popen", popen)
    with pytest.raises(OSError, match="creation failed"):
        f.m.run_lanes(f.actual, f.path, "c" * 64, {})
    assert events == ["waited"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("JobState", "PENDING"),
        ("Requeue", "1"),
        ("TimeLimit", "1-00:00:00"),
        ("AllocTRES", "cpu=8,mem=66G,gres/gpu=1,gres/gpu:pro6000=1"),
        ("ReqTRES", "cpu=4,mem=33G,gres/gpu=2,gres/gpu:pro6000=2"),
    ],
)
def test_actual_allocation_checks_both_requests_and_grants(f, key, value):
    fields = dict(
        JobId="90001",
        JobName=f.m.JOB_NAME,
        JobState="RUNNING",
        QOS=f.m.QOS,
        Partition="cluster02",
        Account="rose",
        UserId="varun024(1)",
        Requeue="0",
        TimeLimit="23:00:00",
        ReqTRES="cpu=8,mem=66G,gres/gpu=2,gres/gpu:pro6000=2",
        AllocTRES="cpu=8,mem=66G,gres/gpu=2,gres/gpu:pro6000=2",
    )
    f.old.query = lambda _: " ".join(k + "=" + v for k, v in fields.items())
    assert f.m.allocation(f.old, "90001")["JobId"] == "90001"
    fields[key] = value
    with pytest.raises(PermissionError):
        f.m.allocation(f.old, "90001")


@pytest.mark.parametrize("queue", ["", "90001\n90002\n"])
def test_terminal_uses_user_queue_without_querying_retired_job_id(queue):
    m = load()
    calls = []

    def query(argv):
        calls.append(argv)
        return "235953|COMPLETED|0:0|\n235953.batch|COMPLETED|0:0|" if argv[0] == "sacct" else queue

    assert m.terminal(SimpleNamespace(query=query))["queue_gone"] is True
    assert calls[-1] == ["squeue", "-h", "-u", m.USER, "-o", "%i"]


def test_terminal_still_queued_is_rejected():
    m = load()
    old = SimpleNamespace(
        query=lambda argv: "235953|COMPLETED|0:0|" if argv[0] == "sacct" else "90001\n235953\n"
    )
    with pytest.raises(PermissionError, match="still queued"):
        m.terminal(old)


def test_failed_accounting_or_unavailable_queue_cannot_prove_success():
    m = load()
    with pytest.raises(PermissionError, match="COMPLETED"):
        m.terminal(SimpleNamespace(query=lambda argv: "235953|FAILED|1:0|"))

    def query(argv):
        if argv[0] == "squeue":
            raise RuntimeError("scheduler unavailable")
        return "235953|COMPLETED|0:0|"

    with pytest.raises(RuntimeError, match="unavailable"):
        m.terminal(SimpleNamespace(query=query))


def test_inherited_slurm_export_none_cannot_drop_original_thread_environment(f, monkeypatch):
    monkeypatch.setenv("SLURM_EXPORT_ENV", "NONE")
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    env = f.m.child_environment(f.m.OPS / "cache")
    assert env["SLURM_EXPORT_ENV"] == "ALL" and env["OMP_NUM_THREADS"] == "1"
    assert "--export=ALL" in f.m.worker_argv(f.actual, f.path, "c" * 64, 0)


@pytest.mark.parametrize("failed_index", [None, 0, 1, 2])
def test_lane_fresh_processes_fixed_order_failure_stops_only_own_lane(f, monkeypatch, failed_index):
    m = f.m
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SLURM_JOB_ID", "90001")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(m, "allocation", lambda *a: {})
    f.previous.FULL_DEPLOYMENT = f.m.PROJECT / "v7-source"
    save(
        m.OPS / "round_b_STARTED.json",
        dict(job_id="90001", source=f.actual, contract_ref=dict(path=str(f.path), sha256="c" * 64)),
    )
    monkeypatch.setattr(m, "task_complete", lambda p, c, path, sha, t, j: {"task_id": t})
    children = []

    def run(argv, **kwargs):
        children.append(argv)
        assert argv[1:4] == ["-I", "-B", "-u"]
        assert "sr_f6_round_b.execution" in argv[5]
        assert str(f.previous.FULL_DEPLOYMENT / "src") in argv[5]
        return SimpleNamespace(returncode=1 if len(children) - 1 == failed_index else 0)

    monkeypatch.setattr(m.subprocess, "run", run)
    if failed_index is None:
        result = m.lane(f.actual, f.path, "c" * 64, 0)
        assert len(result["completed"]) == 3
    else:
        with pytest.raises(PermissionError, match="stop this lane"):
            m.lane(f.actual, f.path, "c" * 64, 0)
        assert not (m.OPS / "LANES/0/COMPLETE.json").exists()
        failure = json.loads((m.OPS / "LANES/0/FAILURE.json").read_bytes())
        assert failure["peer_cancellation_requested"] is False
    assert [a[a.index("--task-id") + 1] for a in children] == m.QUEUES[0][: len(children)]
    assert len(children) == (3 if failed_index is None else failed_index + 1)
    with pytest.raises(PermissionError, match="Prior B lane"):
        m.lane(f.actual, f.path, "c" * 64, 0)


@pytest.mark.parametrize(
    "tamper", [None, "validation", "facts-validation", "source", "task", "failure", "facts-sha"]
)
def test_exit_success_requires_bound_original_and_storage_publications(f, tamper):
    m = f.m
    task = m.QUEUES[0][0]
    ext = m.OPS / "EXECUTIONS" / task
    f.previous.FULL_OPS = m.PROJECT / "full-ops"
    f.previous.FULL_RESULT_SHA = "f" * 64
    f.previous.small_ref = lambda base, ref: (
        f.previous.read(Path(ref["path"]))
        if f.previous.sha(Path(ref["path"])) == ref["sha256"]
        else (_ for _ in ()).throw(PermissionError("changed reference"))
    )
    validation = dict(
        path=str(f.previous.FULL_OPS / "VALIDATION/RESULT.json"), sha256=f.previous.FULL_RESULT_SHA
    )
    cr = dict(path=str(f.path), sha256="c" * 64)
    original = dict(
        status="TASK_COMPLETE_NOT_GATE_SIGNED",
        task={"task_id": task},
        gate_status="NOT_SIGNED",
        producer={"synthetic": True},
    )
    save(m.JOBS / "TASKS" / task / "COMPLETE.json", original)
    facts = dict(
        status="ACTUAL_STORAGE_CONTEXT_COMPLETE_NOT_SCIENTIFIC_PASS",
        contract_ref=cr,
        execution_id="LH-SRF6-B-0-B-90001-" + task,
        offload={"synthetic": True},
        validation_ref=validation,
    )
    if tamper == "facts-validation":
        facts["validation_ref"] = {}
    save(ext / "STORAGE_FACTS.json", facts)
    receipt = dict(
        original,
        round_b_contract_ref=cr,
        execution_source=f.contract["execution_source"],
        full_trainer_numerical_pass=False,
        full_validation_ref=validation,
        storage_facts_ref=dict(
            path=str(ext / "STORAGE_FACTS.json"), sha256=f.previous.sha(ext / "STORAGE_FACTS.json")
        ),
        offload=facts["offload"],
    )
    if tamper == "validation":
        receipt["full_validation_ref"] = {}
    if tamper == "source":
        receipt["execution_source"] = {"wrong": True}
    if tamper == "task":
        receipt["task"] = {"task_id": "PRIMARY-b0-U10"}
    if tamper == "failure":
        save(ext / "FAILURE.json", {})
    if tamper == "facts-sha":
        receipt["storage_facts_ref"]["sha256"] = "0" * 64
    save(ext / "COMPLETE.json", receipt)
    if tamper:
        with pytest.raises(PermissionError):
            m.task_complete(f.previous, f.contract, f.path, "c" * 64, task, "90001")
    else:
        result = m.task_complete(f.previous, f.contract, f.path, "c" * 64, task, "90001")
        assert (
            result["task_id"] == task
            and result["storage_facts_ref"] == receipt["storage_facts_ref"]
        )
