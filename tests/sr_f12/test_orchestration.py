"""Scheduler regression tests use fake Slurm; they never submit a real job."""

import json
import sys
from types import SimpleNamespace

import pytest

from sr_f12.orchestration import (
    TEACHER_QOS,
    Controller,
    controller_lease,
    gpu_count,
    memory_gib,
    save_json,
    teacher_usage,
    verify_allocation,
)


def fields(intent, *, job="100", user="alice", qos=TEACHER_QOS):
    return dict(
        JobId=job,
        UserId=user + "(1)",
        Account="rose",
        Partition="cluster02",
        QOS=qos,
        JobName=intent["job_name"],
        Comment=intent["comment"],
        Command=intent["script"],
        ReqTRES="cpu=4,mem=90G,gres/gpu=1,gres/gpu:pro6000=1",
        Features="highmem",
        JobState="PENDING",
        Reason="JobHeldUser",
    )


class FakeSlurm:
    def __init__(self):
        self.jobs = {}
        self.calls = []
        self.accounting = {}
        self.fail_submit = False

    def queue(self, user):
        return dict(self.jobs)

    def show(self, job):
        return self.jobs[job]

    def terminal(self, job):
        return self.accounting.get(job)

    def run(self, args):
        self.calls.append(args)
        if args[0] == "sbatch":
            if self.fail_submit:
                return SimpleNamespace(returncode=1, stdout="", stderr="ambiguous transport")
            job = str(100 + len(self.jobs))
            options = dict(
                arg[2:].split("=", 1) for arg in args if arg.startswith("--") and "=" in arg
            )
            intent = dict(job_name=options["job-name"], comment=options["comment"], script=args[-1])
            self.jobs[job] = fields(intent, job=job)
            return SimpleNamespace(returncode=0, stdout=job + "\n", stderr="")
        if args[:2] == ["scontrol", "release"]:
            self.jobs[args[2]].update(JobState="RUNNING", Reason="None")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(args)


@pytest.fixture
def controller(tmp_path, monkeypatch):
    root, code, model = [tmp_path / name for name in ("run", "code", "model")]
    for path in (root, code, model):
        path.mkdir()
    manifest = tmp_path / "SOURCE.json"
    manifest.write_text("{}")
    slurm = FakeSlurm()
    value = Controller(
        root, code, model, manifest, "a" * 40, python=sys.executable, slurm=slurm, user="alice"
    )
    monkeypatch.setattr(value, "identity", lambda: "b" * 64)
    return value


def test_gpu_count_generic_is_not_added_to_type():
    assert gpu_count("cpu=16,gres/gpu=4,gres/gpu:pro6000=4") == 4
    assert gpu_count("gres/gpu:pro6000=1,gres/gpu:a100=2") == 3
    assert gpu_count("cpu=1,mem=3G") == 0
    with pytest.raises(PermissionError):
        gpu_count("gres/gpu=1,gres/gpu:pro6000=4")
    with pytest.raises(PermissionError):
        gpu_count("gres/gpu=unknown")


def test_memory_uses_actual_tres():
    assert memory_gib("cpu=4,mem=92160M") == 90
    assert memory_gib("mem=0.09T") > 80
    with pytest.raises(PermissionError):
        memory_gib("cpu=4")


def test_capacity_includes_pending_and_unknown_only_teacher():
    queue = {
        "1": dict(QOS=TEACHER_QOS, ReqTRES="gres/gpu=3"),
        "2": dict(QOS="other", ReqTRES="gres/gpu=16"),
    }
    states = [
        dict(status="RUNNING", job_id="1"),
        dict(status="SUBMISSION_UNKNOWN"),
        dict(status="SUBMITTED", job_id="4"),
        dict(status="COMPLETE", job_id="5"),
    ]
    assert teacher_usage(queue, states) == 5


def test_held_validation_rejects_wrong_qos_owner_memory_type():
    intent = dict(job_name="mine", comment="unique", script="/run/job", job_id="100")
    valid = fields(intent)
    verify_allocation(valid, intent, "alice", held=True)
    for key, value in [
        ("QOS", "other"),
        ("UserId", "bob(2)"),
        ("ReqTRES", "gres/gpu=1,gres/gpu:pro6000=1,mem=33G"),
        ("ReqTRES", "gres/gpu=1,gres/gpu:a100=1,mem=90G"),
        ("JobState", "RUNNING"),
        ("Comment", "unrelated"),
    ]:
        with pytest.raises(PermissionError):
            verify_allocation(dict(valid, **{key: value}), intent, "alice", held=True)


def test_submit_is_held_then_verified_then_released(controller):
    assert controller.submit("TECHNICAL", {}) is True
    assert controller.slurm.calls[0][0:3] == ["sbatch", "--hold", "--parsable"]
    assert "--qos=" + TEACHER_QOS in controller.slurm.calls[0]
    assert "--gres=gpu:pro6000:1" in controller.slurm.calls[0]
    assert controller.slurm.calls[1][:2] == ["scontrol", "release"]
    state = controller.task_state("TECHNICAL")
    assert state["status"] == "SUBMITTED"
    assert controller.submit("TECHNICAL", controller.slurm.queue("alice")) is False
    assert len(controller.slurm.calls) == 2
    directory = controller.task_dir("TECHNICAL")
    assert (directory / "INTENT.json").exists()
    assert (directory / "ALLOCATION_VERIFIED.json").exists()
    assert (directory / "RELEASE_INTENT.json").exists()


def test_unknown_submission_never_resubmits_and_reserves(controller):
    controller.slurm.fail_submit = True
    assert controller.submit("TECHNICAL", {}) is False
    assert controller.task_state("TECHNICAL")["status"] == "SUBMISSION_UNKNOWN"
    assert teacher_usage({}, controller.attempts()) == 1
    assert controller.submit("TECHNICAL", {}) is False
    assert len(controller.slurm.calls) == 1
    result = controller.tick()
    assert result["stage"] == "BLOCKED_UNKNOWN_SUBMISSION"
    assert len(controller.slurm.calls) == 1


def test_foreign_jobs_are_only_counted_never_modified(controller):
    controller.slurm.jobs["999"] = dict(QOS=TEACHER_QOS, ReqTRES="gres/gpu=5")
    result = controller.tick()
    assert result["teacher_gpu_usage"] == 5
    assert controller.slurm.calls == []
    assert controller.slurm.jobs["999"]["ReqTRES"] == "gres/gpu=5"


def test_initial_four_jobs_use_available_capacity(controller):
    controller.slurm.jobs["999"] = dict(QOS=TEACHER_QOS, ReqTRES="gres/gpu=1")
    result = controller.tick()
    assert set(result["tasks"]) == {"TECHNICAL", "BASELINE_0", "BASELINE_1", "BASELINE_2"}
    assert result["teacher_gpu_usage"] == 5
    assert len([a for a in controller.slurm.calls if a[0] == "sbatch"]) == 4
    assert not result["experiment_complete"]


def test_files_alone_cannot_finish_technical(controller):
    controller.submit("TECHNICAL", {})
    directory = controller.root / "technical"
    directory.mkdir()
    (directory / "TECHNICAL_CHECK_SR_F1_2.json").write_text('{"status":"PASS"}')
    controller.refresh(controller.slurm.queue("alice"))
    assert controller.task_state("TECHNICAL")["status"] == "RUNNING"


def test_terminal_failure_blocks_further_jobs(controller):
    controller.submit("TECHNICAL", {})
    job = controller.task_state("TECHNICAL")["job_id"]
    fields_ = controller.slurm.jobs.pop(job)
    controller.slurm.accounting[job] = dict(
        job_id=job,
        state="FAILED",
        exit_code="1:0",
        user="alice",
        account="rose",
        qos=TEACHER_QOS,
        name=fields_["JobName"],
    )
    result = controller.tick()
    assert result["stage"] == "BLOCKED_TECHNICAL_FAILURE"
    assert len([c for c in controller.slurm.calls if c[0] == "sbatch"]) == 1


def test_terminal_success_requires_receipt_validation(controller, monkeypatch):
    controller.submit("TECHNICAL", {})
    job = controller.task_state("TECHNICAL")["job_id"]
    fields_ = controller.slurm.jobs.pop(job)
    controller.slurm.accounting[job] = dict(
        job_id=job,
        state="COMPLETED",
        exit_code="0:0",
        user="alice",
        account="rose",
        qos=TEACHER_QOS,
        name=fields_["JobName"],
    )
    with pytest.raises(FileNotFoundError):
        controller.refresh({})
    assert controller.task_state("TECHNICAL")["status"] != "COMPLETE"
    monkeypatch.setattr(controller, "verify_task", lambda key: dict(task_id=key, verified=True))
    controller.refresh({})
    assert controller.task_state("TECHNICAL")["status"] == "COMPLETE"


def test_controller_exclusive_lease(tmp_path):
    with controller_lease(tmp_path), pytest.raises(RuntimeError), controller_lease(tmp_path):
        pass


def test_immutable_receipt_cannot_be_replaced(tmp_path):
    path = tmp_path / "x.json"
    save_json(path, {"x": 1}, exclusive=True)
    save_json(path, {"x": 1}, exclusive=True)
    with pytest.raises(PermissionError):
        save_json(path, {"x": 2}, exclusive=True)
    assert json.loads(path.read_text()) == {"x": 1}


def test_science_waits_previous_wave(controller, monkeypatch):
    for key in ("TECHNICAL", "BASELINE_0", "BASELINE_1", "BASELINE_2"):
        save_json(controller.task_dir(key) / "STATE.json", dict(task_id=key, status="COMPLETE"))
    monkeypatch.setattr(controller, "refresh", lambda queue: None)
    monkeypatch.setattr(controller, "freeze", lambda: {})
    result = controller.tick()
    science = [key for key in result["tasks"] if key.startswith("SRF1_2_")]
    assert len(science) == 4
    assert all(key.endswith("s71001") for key in science)
    result = controller.tick()
    assert not any(key.endswith("s71002") for key in result["tasks"])


def test_orphan_intent_reserves_and_blocks_duplicate(controller):
    directory = controller.task_dir("TECHNICAL")
    save_json(
        directory / "INTENT.json",
        dict(
            task_id="TECHNICAL",
            source_sha256="b" * 64,
            comment="orphan",
            job_name="srf12-TECHNICAL",
            script="/old/intent",
        ),
        exclusive=True,
    )
    assert teacher_usage({}, controller.attempts()) == 1
    result = controller.tick()
    assert result["stage"] == "BLOCKED_UNKNOWN_SUBMISSION"
    assert controller.slurm.calls == []


def test_worker_script_fixes_deterministic_environment(controller):
    controller.submit("TECHNICAL", {})
    script = (controller.task_dir("TECHNICAL") / "run.sbatch").read_text()
    for text in (
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "PYTHONUNBUFFERED=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
        "OPENBLAS_NUM_THREADS=4",
    ):
        assert text in script
