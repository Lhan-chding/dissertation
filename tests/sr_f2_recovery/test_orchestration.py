"""Fake Slurm tests: no test submits work or reads scientific scores."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f2_recovery import orchestration as recovery


def fields(intent, job="100", *, held=True):
    count = intent["gpu_count"]
    return dict(
        JobId=job,
        UserId="alice(1)",
        Account="rose",
        Partition="cluster02",
        QOS=recovery.TEACHER_QOS,
        JobName=intent["job_name"],
        Comment=intent["comment"],
        Command=intent["script"],
        Features="highmem",
        JobState="PENDING" if held else "RUNNING",
        Reason="JobHeldUser" if held else "None",
        ReqTRES=f"cpu={4 * count},mem={90 * count}G,gres/gpu={count},gres/gpu:pro6000={count}",
    )


class FakeSlurm:
    def __init__(self):
        self.jobs, self.calls, self.accounting = {}, [], {}
        self.fail_submit = self.fail_release = False

    def queue(self, user):
        return dict(self.jobs)

    def show(self, job):
        return self.jobs[job]

    def terminal(self, job):
        return self.accounting.get(job)

    def run(self, args):
        self.calls.append(args)
        if args[0] == "sbatch":
            path = Path(args[-1]).parent
            intent = recovery.read_json(path / "INTENT.json")
            assert recovery.read_json(path / "STATE.json")["status"] == "SUBMISSION_UNKNOWN"
            if self.fail_submit:
                return SimpleNamespace(returncode=1, stdout="", stderr="transport lost")
            job = str(100 + sum(call[0] == "sbatch" for call in self.calls))
            self.jobs[job] = fields(intent, job)
            return SimpleNamespace(returncode=0, stdout=job + "\n", stderr="")
        if args[:2] == ["scontrol", "release"]:
            if self.fail_release:
                raise TimeoutError("release result unknown")
            self.jobs[args[2]].update(JobState="RUNNING", Reason="None")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(args)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root, code, extension = tmp_path / "original", tmp_path / "frozen", tmp_path / "extension"
    for path in (root, code, extension):
        path.mkdir()
    directory = root / "recovery" / "cancel-recovery"
    directory.mkdir(parents=True)
    reference = root / "original-evidence.json"
    reference.write_text('{"frozen": true}\n')
    refs = [{"path": reference.name, "sha256": recovery.file_hash(reference)}]
    matrix = recovery.global_task_matrix()
    # Leave every train and endpoint available; prior history/pilot/diagnostics inherited.
    rows = {
        recovery.task_id(task): dict(
            task=task,
            status="NEVER_ATTEMPTED"
            if task["kind"] in {"train", "endpoint"}
            else "INHERITED_COMPLETE",
            reference_files=refs,
        )
        for task in matrix
    }
    registration = dict(code_root=str(code), original=True)
    recovery.save_json(
        root / "orchestration/CONFIG.json", dict(code_root=str(code), python=sys.executable)
    )
    audit = dict(
        original_root=str(root), registration_sha256=recovery.object_hash(registration), tasks=rows
    )
    recovery.save_json(directory / "AUDIT.json", audit)
    manifest = {}
    for relative in (
        "src/sr_f2_recovery/__init__.py",
        "src/sr_f2_recovery/orchestration.py",
        "scripts/sr_f2/recovery.py",
    ):
        target = extension / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# bound extension\n")
        manifest[relative] = recovery.file_hash(target)
    (code / "src/sr_f2").mkdir(parents=True)
    (code / "src/sr_f2/protocol.py").write_text("# frozen\n")
    old = [
        dict(
            job_id=str(i),
            user="alice",
            account="rose",
            qos=recovery.TEACHER_QOS,
            name=f"original-{i}",
        )
        for i in range(1, 6)
    ]
    plan = dict(
        original_root=str(root),
        recovery_id="cancel-recovery",
        max_run_gpus=5,
        registration_sha256=recovery.object_hash(registration),
        frozen_code_root=str(code),
        extension_root=str(extension),
        extension_manifest=manifest,
        python=str(Path(sys.executable).resolve()),
        original_task_matrix=matrix,
        audit_path="AUDIT.json",
        audit_sha256=recovery.file_hash(directory / "AUDIT.json"),
        old_cancelled_jobs=old,
    )
    recovery.save_json(directory / "PLAN.json", plan)
    result = recovery.Recovery(directory / "PLAN.json")
    monkeypatch.setattr(
        recovery, "__file__", str(extension / "src/sr_f2_recovery/orchestration.py")
    )
    import sr_f2.protocol as protocol

    monkeypatch.setattr(protocol, "__file__", str(code / "src/sr_f2/protocol.py"))
    monkeypatch.setattr(
        recovery, "verify_registration", lambda root, require_technical: registration
    )
    slurm = FakeSlurm()
    slurm.accounting = {r["job_id"]: dict(r, state="CANCELLED", exit_code="0:0") for r in old}
    return result, recovery.Controller(result, slurm=slurm, user="alice")


def reseal(r):
    recovery.save_json(r.audit_path, r.audit)
    r.plan["audit_sha256"] = recovery.file_hash(r.audit_path)
    recovery.save_json(r.path, r.plan)
    r.sha256 = recovery.object_hash(r.plan)


def only_remaining(r, *tasks):
    keep = {recovery.task_id(task) for task in tasks}
    for key, row in r.allowed.items():
        if key not in keep:
            row["status"] = "INHERITED_COMPLETE"
    reseal(r)


def test_five_gpus_in_three_held_identity_bound_pools(experiment):
    r, controller = experiment
    result = controller.submit_once()
    assert result["capacity"] == dict(teacher_gpus=5, user_submissions=3, run_gpus=5)
    calls = controller.slurm.calls
    assert [
        arg for call in calls if call[0] == "sbatch" for arg in call if arg.startswith("--gres")
    ] == ["--gres=gpu:pro6000:2", "--gres=gpu:pro6000:2", "--gres=gpu:pro6000:1"]
    assert all(
        call[:3] == ["sbatch", "--hold", "--parsable"] for call in calls if call[0] == "sbatch"
    )
    controller.submit_once()
    assert len(controller.slurm.calls) == 6
    script = (controller.pool_dir("RECOVERY_POOL_3") / "run.sbatch").read_text()
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in script
    assert "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4" in script
    assert "scripts/sr_f2/recovery.py" in script
    assert not (r.root / "orchestration/queue").exists()


@pytest.mark.parametrize("foreign_gpus,expected", [(3, 5), (4, 4), (5, 3), (7, 1), (8, 0)])
def test_counts_all_teacher_pending_reservations(experiment, foreign_gpus, expected):
    _, c = experiment
    c.slurm.jobs["999"] = dict(
        QOS=recovery.TEACHER_QOS,
        UserId="bob(2)",
        ReqTRES=f"gres/gpu={foreign_gpus}",
        JobState="PENDING",
    )
    result = c.submit_once()
    assert result["capacity"]["run_gpus"] == expected
    assert result["capacity"]["teacher_gpus"] <= 8
    assert "999" in c.slurm.jobs


def test_other_qos_user_jobs_count_toward_five_submission_limit(experiment):
    _, c = experiment
    c.slurm.jobs = {
        str(900 + i): dict(QOS="other", UserId="alice(1)", ReqTRES="cpu=1") for i in range(4)
    }
    assert c.submit_once()["capacity"] == dict(teacher_gpus=2, user_submissions=5, run_gpus=2)


def test_unknown_submission_is_reserved_and_never_retried(experiment):
    _, c = experiment
    c.slurm.fail_submit = True
    assert c.submit_once()["capacity"]["run_gpus"] == 2
    assert c.submit_once()["status"] == "BLOCKED_UNKNOWN_SUBMISSION"
    assert len(c.slurm.calls) == 1


def test_orphan_intent_reserves_capacity_and_blocks_submission(experiment):
    _, c = experiment
    recovery.save_json(c.pool_dir("RECOVERY_POOL_1") / "INTENT.json", {"unknown": True})
    assert c.submit_once()["status"] == "BLOCKED_UNKNOWN_SUBMISSION"
    assert c.submit_once()["capacity"]["run_gpus"] == 2
    assert not c.slurm.calls


def test_unknown_release_is_not_replayed(experiment):
    _, c = experiment
    c.slurm.fail_release = True
    with pytest.raises(TimeoutError):
        c.submit_once()
    assert c.submit_once()["status"] == "BLOCKED_UNKNOWN_SUBMISSION"
    with pytest.raises(PermissionError, match="already attempted"):
        c.release("RECOVERY_POOL_1")
    assert len(c.slurm.calls) == 2


def test_stale_local_terminal_state_does_not_free_unknown_accounting():
    attempts = [dict(pool_id="RECOVERY_POOL_3", gpu_count=1, status="DRAINED", job_id="12")]
    assert recovery.capacity({}, attempts, "alice") == dict(
        teacher_gpus=1, user_submissions=1, run_gpus=1
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"QOS": recovery.TEACHER_QOS},
        {"QOS": recovery.TEACHER_QOS, "UserId": "bob(2)", "ReqTRES": "gres/gpu=?"},
    ],
)
def test_unknown_queue_capacity_fails_closed(bad):
    with pytest.raises(PermissionError):
        recovery.capacity({"3": bad}, [], "alice")


@pytest.mark.parametrize(
    "key,value",
    [
        ("QOS", "other"),
        ("UserId", "bob(2)"),
        ("Comment", "foreign"),
        ("Command", "/wrong"),
        ("JobId", "999"),
        ("ReqTRES", "mem=180G,gres/gpu=2,gres/gpu:a100=2"),
        ("Features", "None"),
        ("JobState", "RUNNING"),
    ],
)
def test_allocation_identity_guard(key, value):
    intent = dict(
        pool_id="RECOVERY_POOL_1",
        gpu_count=2,
        job_id="100",
        job_name="name",
        script="/script",
        comment="bound",
    )
    with pytest.raises(PermissionError):
        recovery.verify_allocation(dict(fields(intent), **{key: value}), intent, "alice", held=True)


def test_running_original_allocation_blocks_recovery(experiment):
    _, c = experiment
    c.slurm.accounting["1"]["state"] = "RUNNING"
    with pytest.raises(PermissionError, match="confirmed cancelled"):
        c.submit_once()
    assert not c.slurm.calls


def test_cancelled_but_completing_original_allocation_blocks_recovery(experiment):
    _, c = experiment
    c.slurm.jobs["1"] = dict(
        QOS=recovery.TEACHER_QOS, UserId="alice(1)", ReqTRES="cpu=1", JobState="COMPLETING"
    )
    with pytest.raises(PermissionError, match="remains live"):
        c.submit_once()
    assert not c.slurm.calls


def test_original_runtime_change_blocks_recovery(experiment):
    r, _ = experiment
    recovery.save_json(
        r.root / "orchestration/CONFIG.json",
        dict(code_root=str(r.extension), python=sys.executable),
    )
    with pytest.raises(PermissionError, match="runtime differs"):
        r.verify()


def test_changed_allowlist_or_frozen_registration_fails(experiment, monkeypatch):
    r, _ = experiment
    assert r.verify() == r.sha256
    monkeypatch.setattr(recovery, "verify_registration", lambda *args, **kwargs: {})
    with pytest.raises(PermissionError, match="registration"):
        r.verify()


def test_missing_allowlist_entry_fails_closed(experiment):
    r, _ = experiment
    r.audit["tasks"].pop(next(iter(r.allowed)))
    reseal(r)
    with pytest.raises(PermissionError, match="classify every"):
        recovery.Recovery(r.path)


def test_reference_escape_and_modified_extension_fail(experiment):
    r, _ = experiment
    target = r.extension / "src/sr_f2_recovery/orchestration.py"
    target.write_text("# modified\n")
    with pytest.raises(PermissionError, match="extension source changed"):
        r.verify()
    with pytest.raises(PermissionError, match="escapes"):
        recovery.within(r.root, "../extension/src/sr_f2_recovery/orchestration.py")


def test_queue_reuses_inherited_and_launches_endpoint_without_seed_barrier(experiment):
    r, _ = experiment
    train = next(t for t in r.matrix if t["kind"] == "train")
    endpoint = dict(kind="endpoint", model_id=train["model_id"])
    other = next(t for t in r.matrix if t["kind"] == "train" and t != train)
    only_remaining(r, train, other, endpoint)
    q = recovery.RecoveryQueue(r)
    first, _ = q.claim("RECOVERY_POOL_1", "0", "101")
    second, _ = q.claim("RECOVERY_POOL_1", "1", "101")
    assert first["task"] == train and second["task"] == other
    q.transition(first, pid=12)
    q.transition(first, result=dict(train, receipt_sha256="verified"))
    next_claim, _ = q.claim("RECOVERY_POOL_1", "0", "101")
    assert next_claim["task"] == endpoint
    assert q.snapshot()["completed"] == 51
    assert not (r.root / "orchestration/queue").exists()


def test_blocked_training_prevents_endpoint_and_does_not_hold_idle_gpus(experiment):
    r, _ = experiment
    train = next(t for t in r.matrix if t["kind"] == "train")
    endpoint = dict(kind="endpoint", model_id=train["model_id"])
    only_remaining(r, train, endpoint)
    r.allowed[recovery.task_id(train)].update(
        status="BLOCKED_INCOMPLETE_GENERATION", reason="draw was not committed"
    )
    reseal(r)
    q = recovery.RecoveryQueue(r)
    snapshot = q.snapshot()
    assert snapshot["idle_reason"] == "ONLY_BLOCKED_DEPENDENCIES_REMAIN"
    assert snapshot["may_exit"] and not snapshot["all_complete"]
    assert q.claim("RECOVERY_POOL_3", "0", "103")[0] is None


def test_active_training_dependency_keeps_useful_pool_available(experiment):
    r, _ = experiment
    train = next(t for t in r.matrix if t["kind"] == "train")
    endpoint = dict(kind="endpoint", model_id=train["model_id"])
    only_remaining(r, train, endpoint)
    q = recovery.RecoveryQueue(r)
    q.claim("RECOVERY_POOL_1", "0", "101")
    snapshot = q.snapshot()
    assert snapshot["idle_reason"] == "WAIT_ACTIVE_PREREQUISITES"
    assert not snapshot["may_exit"]


def test_duplicate_gpu_claim_or_orphan_claim_is_never_retried(experiment):
    r, _ = experiment
    q = recovery.RecoveryQueue(r)
    claim, _ = q.claim("RECOVERY_POOL_3", "0", "103")
    with pytest.raises(PermissionError, match="already has active"):
        q.claim("RECOVERY_POOL_3", "0", "103")
    (q.directory / "tasks" / recovery.task_id(claim["task"]) / "STATE.json").unlink()
    with pytest.raises(RuntimeError, match="Orphan"):
        q.snapshot()


def test_worker_failure_stops_every_pool_without_original_changes(experiment):
    r, _ = experiment
    before = (r.root / "original-evidence.json").read_bytes()
    q = recovery.RecoveryQueue(r)
    claim, _ = q.claim("RECOVERY_POOL_1", "0", "101")
    q.transition(claim, error="unexpected worker exit")
    assert q.snapshot()["failed"]
    with pytest.raises(RuntimeError, match="stopped"):
        q.claim("RECOVERY_POOL_2", "0", "102")
    with pytest.raises(PermissionError, match="cannot be resumed"):
        q.transition(claim, pid=10)
    assert (r.root / "original-evidence.json").read_bytes() == before
    assert not (r.root / "STOP").exists()


def test_mutable_guards_checked_only_for_task_before_launch(experiment):
    r, _ = experiment
    task = next(t for t in r.matrix if t["kind"] == "train")
    key = recovery.task_id(task)
    mutable = r.root / "LATEST.json"
    mutable.write_text("original")
    r.allowed[key]["mutable_prelaunch_files"] = [
        {"path": mutable.name, "sha256": recovery.file_hash(mutable)}
    ]
    reseal(r)
    q = recovery.RecoveryQueue(r)
    claim, _ = q.claim("RECOVERY_POOL_1", "0", "101")
    assert claim["task"] == task
    mutable.write_text("legitimate next committed step")
    assert r.verify() == r.sha256  # Other workers may continue after this task starts.
    with pytest.raises(PermissionError, match="changed since"):
        r.task_prelaunch(key)


def test_never_attempted_absence_guard_detects_external_writer(experiment):
    r, _ = experiment
    key = recovery.task_id(next(t for t in r.matrix if t["kind"] == "train"))
    r.allowed[key]["absent_prelaunch_paths"] = ["unexpected-output"]
    reseal(r)
    (r.root / "unexpected-output").mkdir()
    with pytest.raises(PermissionError, match="now exists"):
        recovery.RecoveryQueue(r).claim("RECOVERY_POOL_1", "0", "101")
    assert not list((r.directory / "queue/tasks").glob("*/CLAIM.json"))


def test_pool_runs_original_worker_one_visible_gpu_and_verifies_completion(experiment, monkeypatch):
    r, c = experiment
    task = next(t for t in r.matrix if t["kind"] == "train")
    only_remaining(r, task)
    c.submit_once()
    pool = "RECOVERY_POOL_1"
    state = recovery.read_json(c.pool_dir(pool) / "STATE.json")
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    calls, verified = [], []

    class Process:
        pid = 77

        def poll(self):
            return 0

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    def verify(root, received, registration):
        verified.append(received)
        return dict(received, receipt_sha256="verified")

    monkeypatch.setattr(recovery, "verify_task", verify)
    result = recovery.run_pool(r, pool, slurm=c.slurm, poll_seconds=0, popen=popen)
    assert result["all_tasks_complete"]
    assert len(calls) == 1 and verified == [task]
    command, kwargs = calls[0]
    assert command[1] == str(r.code / "scripts/sr_f2/run_worker.py")
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4"
    assert kwargs["env"]["PYTHONPATH"] == str(r.code / "src")
    assert kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert not (r.root / "orchestration/queue").exists()
    with pytest.raises(PermissionError, match="already started"):
        recovery.run_pool(r, pool, slurm=c.slurm, poll_seconds=0, popen=popen)


def test_live_pool_wrong_job_never_launches(experiment, monkeypatch):
    r, c = experiment
    c.submit_once()
    monkeypatch.setenv("SLURM_JOB_ID", "foreign")
    with pytest.raises(PermissionError, match="bound recovery allocation"):
        recovery.run_pool(r, "RECOVERY_POOL_1", slurm=c.slurm)


def test_pool_worker_failure_preserves_exit_and_creates_global_stop(experiment, monkeypatch):
    r, c = experiment
    task = next(t for t in r.matrix if t["kind"] == "train")
    only_remaining(r, task)
    c.submit_once()
    pool = "RECOVERY_POOL_1"
    state = recovery.read_json(c.pool_dir(pool) / "STATE.json")
    monkeypatch.setenv("SLURM_JOB_ID", state["job_id"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    class Process:
        pid = 77

        def poll(self):
            return 17

    with pytest.raises(RuntimeError, match="worker failed"):
        recovery.run_pool(
            r, pool, slurm=c.slurm, poll_seconds=0, popen=lambda *args, **kwargs: Process()
        )
    exit_path = c.pool_dir(pool) / "workers" / recovery.task_id(task) / "PROCESS_EXIT.json"
    assert recovery.read_json(exit_path)["returncode"] == 17
    assert recovery.RecoveryQueue(r).snapshot()["failed"]
    assert recovery.read_json(r.directory / "STOP.json")["pool_id"] == pool
    assert not (r.root / "STOP").exists()
