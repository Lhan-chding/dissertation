"""Fail-closed scheduling tests; fake Slurm never submits a real job."""

import os
import sys
from types import SimpleNamespace

import pytest

from sr_f2 import orchestration as scheduler
from sr_f2 import protocol
from sr_f2.orchestration import (
    TEACHER_QOS,
    Controller,
    controller_lease,
    object_hash,
    pool_specs,
    read_json,
    run_pool,
    save_json,
    teacher_usage,
    verify_allocation,
    verify_task,
    visible_devices,
)


def fields(intent, *, job="100", user="alice", count=4):
    return dict(
        JobId=job,
        UserId=user + "(1)",
        Account="rose",
        Partition="cluster02",
        QOS=TEACHER_QOS,
        JobName=intent["job_name"],
        Comment=intent["comment"],
        Command=intent["script"],
        Features="highmem",
        JobState="PENDING",
        Reason="JobHeldUser",
        ReqTRES=f"cpu=16,mem=360G,gres/gpu={count},gres/gpu:pro6000={count}",
    )


class FakeSlurm:
    def __init__(self):
        self.jobs = {}
        self.calls = []
        self.accounting = {}
        self.fail_submit = False

    def queue(self):
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
            job = str(99 + sum(call[0] == "sbatch" for call in self.calls))
            options = dict(
                arg[2:].split("=", 1) for arg in args if arg.startswith("--") and "=" in arg
            )
            intent = dict(job_name=options["job-name"], comment=options["comment"], script=args[-1])
            # The durable intent must precede every external mutation.
            intent_path = os.path.join(os.path.dirname(args[-1]), "INTENT.json")
            assert read_json(intent_path)["script"] == args[-1]
            self.jobs[job] = fields(intent, job=job)
            return SimpleNamespace(returncode=0, stdout=job + "\n", stderr="")
        if args[:2] == ["scontrol", "release"]:
            self.jobs[args[2]].update(JobState="RUNNING", Reason="None")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(args)

    def finish(self, job, *, state="COMPLETED", exit_code="0:0"):
        row = self.jobs.pop(job)
        self.accounting[job] = dict(
            job_id=job,
            state=state,
            exit_code=exit_code,
            user="alice",
            account="rose",
            qos=TEACHER_QOS,
            name=row["JobName"],
        )


@pytest.fixture
def controller(tmp_path, monkeypatch):
    root, code = tmp_path / "run", tmp_path / "code"
    root.mkdir()
    code.mkdir()
    monkeypatch.setattr(
        protocol, "verify_registration", lambda root: {"registered": True, "code_root": str(code)}
    )
    monkeypatch.setattr(protocol, "finalize_readiness", lambda root: None, raising=False)
    return Controller(root, code, python=sys.executable, slurm=FakeSlurm(), user="alice")


def mark_complete(controller, *keys):
    for key in keys:
        save_json(
            controller.pool_dir(key) / "STATE.json",
            dict(pool_id=key, gpu_count=4, status="COMPLETE"),
        )


def test_capacity_counts_all_teacher_users_pending_and_unknown():
    queue = {
        "1": dict(QOS=TEACHER_QOS, ReqTRES="gres/gpu=4", UserId="bob(2)", JobState="PENDING"),
        "2": dict(QOS=TEACHER_QOS, ReqTRES="cpu=1", UserId="alice(1)"),
        "3": dict(QOS="other", ReqTRES="gres/gpu=16", UserId="alice(1)"),
    }
    states = [
        dict(status="SUBMISSION_UNKNOWN", gpu_count=4),
        dict(status="COMPLETE", job_id="9", gpu_count=4),
    ]
    assert teacher_usage(queue, states, "alice") == dict(gpus=8, submitted=2)
    states[0]["job_id"] = "1"
    assert teacher_usage(queue, states, "alice") == dict(gpus=4, submitted=1)


@pytest.mark.parametrize(
    "row",
    [
        dict(QOS=TEACHER_QOS, UserId="alice(1)"),
        dict(QOS=TEACHER_QOS, ReqTRES="gres/gpu=unknown"),
        dict(QOS=TEACHER_QOS, ReqTRES="gres/gpu=4"),
    ],
)
def test_missing_capacity_or_owner_fails_closed(row):
    with pytest.raises(PermissionError):
        teacher_usage({"1": row}, [], "alice")


def test_submit_held_verify_release_and_no_duplicate(controller):
    assert controller.submit("GPU_POOL_1", {})
    assert controller.slurm.calls[0][:3] == ["sbatch", "--hold", "--parsable"]
    for option in ("--gres=gpu:pro6000:4", "--cpus-per-task=16", "--mem=360G"):
        assert option in controller.slurm.calls[0]
    assert controller.slurm.calls[1][:2] == ["scontrol", "release"]
    assert controller.pool_state("GPU_POOL_1")["status"] == "SUBMITTED"
    assert not controller.submit("GPU_POOL_1", controller.slurm.queue())
    assert len(controller.slurm.calls) == 2
    script = (controller.pool_dir("GPU_POOL_1") / "run.sbatch").read_text()
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in script
    assert "scripts/sr_f2/gpu_pool.py" in script


def test_two_four_gpu_history_pools_fill_eight_gpus(controller):
    result = controller.tick()
    assert set(result["pools"]) == {"GPU_POOL_1", "GPU_POOL_2"}
    assert result["teacher_usage"] == dict(gpus=8, submitted=2)
    assert result["stage"] == "HISTORY_WITH_BACKFILL"
    assert not result["experiment_complete"]


def test_foreign_teacher_gpu_allocations_count_without_mutation(controller):
    controller.slurm.jobs["999"] = dict(QOS=TEACHER_QOS, UserId="bob(2)", ReqTRES="gres/gpu=5")
    result = controller.tick()
    assert result["teacher_usage"] == dict(gpus=5, submitted=0)
    assert controller.slurm.calls == []


def test_five_teacher_cpu_jobs_exhaust_user_submission_limit(controller):
    controller.slurm.jobs = {
        str(i): dict(QOS=TEACHER_QOS, UserId="alice(1)", ReqTRES="cpu=1") for i in range(5)
    }
    assert controller.tick()["teacher_usage"] == dict(gpus=0, submitted=5)
    assert controller.slurm.calls == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("QOS", "other"),
        ("UserId", "bob(2)"),
        ("ReqTRES", "mem=360G,gres/gpu=4,gres/gpu:a100=4"),
        ("ReqTRES", "mem=300G,gres/gpu:pro6000=4"),
        ("JobState", "RUNNING"),
        ("Comment", "foreign"),
    ],
)
def test_wrong_allocation_stays_held(key, value):
    intent = dict(job_name="name", comment="bound", script="/run/script", job_id="100", gpu_count=4)
    valid = fields(intent)
    verify_allocation(valid, intent, "alice", held=True)
    with pytest.raises(PermissionError):
        verify_allocation(dict(valid, **{key: value}), intent, "alice", held=True)


def test_unknown_submission_consumes_four_gpus_and_never_retries(controller):
    controller.slurm.fail_submit = True
    result = controller.tick()
    assert result["stage"] == "BLOCKED_UNKNOWN_SUBMISSION"
    assert result["teacher_usage"] == dict(gpus=4, submitted=1)
    assert len(controller.slurm.calls) == 1
    controller.tick()
    assert len(controller.slurm.calls) == 1


def test_orphan_intent_reserves_capacity_and_does_not_retry(controller):
    save_json(
        controller.pool_dir("GPU_POOL_1") / "INTENT.json",
        dict(pool_id="GPU_POOL_1", gpu_count=4, comment="never observed"),
    )
    assert controller.tick()["stage"] == "BLOCKED_UNKNOWN_SUBMISSION"
    assert teacher_usage({}, controller.attempts(), "alice")["gpus"] == 4
    assert controller.slurm.calls == []


def test_missing_accounting_blocks_new_work_until_observed(controller):
    controller.submit("GPU_POOL_1", {})
    controller.slurm.jobs.clear()
    assert controller.tick()["stage"] == "BLOCKED_UNKNOWN_SUBMISSION"
    assert controller.pool_state("GPU_POOL_1")["status"] == "ACCOUNTING_UNKNOWN"
    assert len(controller.slurm.calls) == 2


def test_terminal_failure_stops_new_submissions(controller):
    controller.submit("GPU_POOL_1", {})
    controller.slurm.finish("100", state="FAILED", exit_code="1:0")
    assert controller.tick()["stage"] == "BLOCKED_TECHNICAL_FAILURE"
    assert (controller.pool_dir("GPU_POOL_1") / "TERMINAL.json").exists()
    assert len(controller.slurm.calls) == 2


def test_slurm_success_without_artifacts_is_a_recorded_failure(controller):
    controller.submit("GPU_POOL_1", {})
    controller.slurm.finish("100")
    assert controller.tick()["stage"] == "BLOCKED_TECHNICAL_FAILURE"
    assert controller.pool_state("GPU_POOL_1")["failure"] == "ARTIFACT_VERIFICATION"


def test_files_alone_never_finish_a_running_pool(controller):
    controller.submit("GPU_POOL_1", {})
    save_json(controller.pool_dir("GPU_POOL_1") / "COMPLETE.json", dict(status="COMPLETE"))
    controller.refresh(controller.slurm.queue())
    assert controller.pool_state("GPU_POOL_1")["status"] == "RUNNING"


def test_controller_must_not_consume_teacher_qos(controller, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_QOS", TEACHER_QOS)
    with pytest.raises(PermissionError, match="CPU controller"):
        controller.tick()
    assert not controller.slurm.calls


def test_queue_error_never_submits(controller, monkeypatch):
    def unavailable():
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(controller.slurm, "queue", unavailable)
    with pytest.raises(RuntimeError, match="queue unavailable"):
        controller.tick()
    assert controller.slurm.calls == []


def test_controller_exclusive_lease(tmp_path):
    with controller_lease(tmp_path), pytest.raises(RuntimeError), controller_lease(tmp_path):
        pass


def test_worker_receipt_binds_artifact_bytes_and_pilot_pass(tmp_path, monkeypatch):
    # Exercise envelope authentication independently; kind-specific evidence has separate tests.
    monkeypatch.setattr(scheduler, "verify_worker_artifacts", lambda *args: None)
    root = tmp_path
    task = dict(kind="pilot", model_id="SRF2_PILOT_EARLY")
    data = root / "pilot.bin"
    data.write_bytes(b"recorded checkpoint")
    receipt = dict(
        status="PASS",
        **task,
        registration_sha256="a" * 64,
        artifacts=[dict(path="pilot.bin", sha256=scheduler.file_hash(data))],
    )
    save_json(scheduler.task_receipt_path(root, task), receipt)
    assert verify_task(root, task, "a" * 64)["receipt_sha256"] == object_hash(receipt)
    data.write_bytes(b"changed")
    with pytest.raises(PermissionError, match="changed"):
        verify_task(root, task, "a" * 64)
    receipt["status"] = "COMPLETE"
    save_json(scheduler.task_receipt_path(root, task), receipt)
    with pytest.raises(PermissionError, match="status"):
        verify_task(root, task, "a" * 64)


def test_worker_artifact_cannot_escape_run_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    task = dict(kind="train", model_id="model")
    save_json(
        scheduler.task_receipt_path(root, task),
        dict(
            status="COMPLETE",
            **task,
            registration_sha256="a",
            artifacts=[dict(path="../outside.bin", sha256=scheduler.file_hash(outside))],
        ),
    )
    with pytest.raises(PermissionError, match="escapes"):
        verify_task(root, task, "a")


@pytest.mark.parametrize("value", ["0,1,2", "0,0,1,2", "0,1,2,3,4", "", "MIG-a,1,2,3"])
def test_visible_devices_requires_four_physical_distinct_allocations(value):
    with pytest.raises(PermissionError):
        visible_devices(value, 4)


def test_gpu_pool_refuses_wrong_slurm_binding(controller, monkeypatch):
    controller.submit("GPU_POOL_1", {})
    monkeypatch.setenv("SLURM_JOB_ID", "unrelated")
    with pytest.raises(PermissionError, match="bound Slurm"):
        run_pool(controller.root, controller.pool_dir("GPU_POOL_1") / "PLAN.json")


def test_new_capacity_race_keeps_verified_job_held(controller, monkeypatch):
    original_queue = controller.slurm.queue

    def raced_queue():
        return {
            **original_queue(),
            "999": dict(QOS=TEACHER_QOS, UserId="bob(2)", ReqTRES="gres/gpu=8"),
        }

    monkeypatch.setattr(controller.slurm, "queue", raced_queue)
    with pytest.raises(PermissionError, match="capacity exceeded"):
        controller.submit("GPU_POOL_1", {})
    assert controller.pool_state("GPU_POOL_1")["status"] == "HELD"
    assert len(controller.slurm.calls) == 1


def test_ambiguous_release_is_observed_and_never_repeated(controller, monkeypatch):
    original_run = controller.slurm.run

    def run(args):
        if args[:2] == ["scontrol", "release"]:
            controller.slurm.calls.append(args)
            return SimpleNamespace(returncode=1, stdout="", stderr="ambiguous release")
        return original_run(args)

    monkeypatch.setattr(controller.slurm, "run", run)
    controller.submit("GPU_POOL_1", {})
    assert controller.pool_state("GPU_POOL_1")["status"] == "RELEASE_UNKNOWN"
    assert controller.tick()["stage"] == "BLOCKED_UNKNOWN_SUBMISSION"
    assert len(controller.slurm.calls) == 2


def raw_group(directory, model_id, step, start, count=4):
    rows = [
        dict(model_id=model_id, step=step, slot_id=f"{model_id}-{step}-{i}")
        for i in range(start, start + count)
    ]
    group_id = object_hash([model_id, step, start])
    path = directory / "groups" / (group_id + ".json")
    records_hash = object_hash(rows)
    save_json(path, dict(group_id=group_id, records=rows, records_sha256=records_hash))
    return dict(
        group_id=group_id, raw_sha256=scheduler.file_hash(path), records_sha256=records_hash
    )


def complete_pilot(root):
    from sr_f1.contract import digest

    model_id = "SRF2_PILOT_EARLY"
    directory = root / "technical" / model_id
    checkpoint = directory / "checkpoints" / "step-08.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"independently verified worker state")
    commit = dict(
        path=checkpoint.name, step=8, sha256=scheduler.file_hash(checkpoint), state_hash="d" * 64
    )
    save_json(checkpoint.parent / "commit-08.json", commit)
    save_json(checkpoint.parent / "LATEST.json", commit)
    save_json(
        directory / "COMPLETE.json",
        dict(
            status="COMPLETE",
            model_id=model_id,
            step=8,
            technical_only=True,
            checkpoint=commit,
            last_monitor_step=8,
        ),
    )
    for step in (0, 8):
        monitor_dir = directory / "monitor" / f"step-{step:02d}"
        refs = [raw_group(monitor_dir, model_id, step, i) for i in range(0, 512, 4)]
        measurement = dict(status="COMPLETE", model_id=model_id, step=step, count=512, groups=refs)
        identity = dict(
            model_id=model_id,
            step=step,
            checkpoint_sha256=commit["sha256"],
            checkpoint_state_hash=commit["state_hash"],
        )
        save_json(
            directory / "monitor_state" / f"step-{step:02d}.json",
            dict(identity=identity, measurement=measurement, measurement_hash=digest(measurement)),
        )
    artifacts = [directory / "COMPLETE.json", checkpoint.parent / "commit-08.json"]
    task = dict(kind="pilot", model_id=model_id)
    save_json(
        scheduler.task_receipt_path(root, task),
        dict(
            status="PASS",
            **task,
            step=8,
            registration_sha256="r",
            artifacts=[
                dict(path=str(p.relative_to(root)), sha256=scheduler.file_hash(p))
                for p in artifacts
            ],
        ),
    )
    return task, directory


def test_complete_pilot_requires_canonical_checkpoint_and_all_raw_monitors(tmp_path):
    task, directory = complete_pilot(tmp_path)
    assert verify_task(tmp_path, task, "r")["model_id"] == task["model_id"]
    (directory / "checkpoints" / "step-08.pt").write_bytes(b"tampered")
    with pytest.raises(PermissionError, match="checkpoint bytes"):
        verify_task(tmp_path, task, "r")


def test_pilot_without_initial_monitor_cannot_pass(tmp_path):
    task, directory = complete_pilot(tmp_path)
    (directory / "monitor_state/step-00.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_task(tmp_path, task, "r")


@pytest.mark.parametrize("kind", ["history", "diagnostic", "pilot", "train", "endpoint"])
def test_one_benign_artifact_can_never_qualify_any_worker_kind(tmp_path, kind):
    task = next(item for item in scheduler.global_task_matrix() if item["kind"] == kind)
    benign = tmp_path / "benign.txt"
    benign.write_text("finished")
    save_json(
        scheduler.task_receipt_path(tmp_path, task),
        dict(
            status="PASS" if kind == "pilot" else "COMPLETE",
            **task,
            registration_sha256="r",
            artifacts=[dict(path=benign.name, sha256=scheduler.file_hash(benign))],
        ),
    )
    with pytest.raises(PermissionError, match="evidence"):
        verify_task(tmp_path, task, "r")


def test_raw_group_verification_rejects_changed_bytes_and_unfinished_groups(tmp_path):
    ref = raw_group(tmp_path, "model", 8, 0)
    receipt = dict(groups=[ref])
    scheduler.verify_evaluation_groups(tmp_path, receipt, "model", 4, step=8)
    pending = tmp_path / "groups" / ("a" * 64 + ".pending.jsonl")
    pending.write_text("partial")
    with pytest.raises(PermissionError, match="unfinished"):
        scheduler.verify_evaluation_groups(tmp_path, receipt, "model", 4, step=8)
    pending.unlink()
    with pytest.raises(PermissionError, match="identity"):
        scheduler.verify_evaluation_groups(tmp_path, receipt, "model", 4, step=16)
    raw = tmp_path / "groups" / (ref["group_id"] + ".json")
    data = read_json(raw)
    data["records"][0]["slot_id"] = "changed"
    save_json(raw, data)
    with pytest.raises(PermissionError, match="bytes"):
        scheduler.verify_evaluation_groups(tmp_path, receipt, "model", 4, step=8)


def test_matrix_has_53_work_units_and_only_two_persistent_pools():
    pools = pool_specs()
    assert [p["pool_id"] for p in pools] == ["GPU_POOL_1", "GPU_POOL_2"]
    assert all(p["gpu_count"] == 4 and p["queue"] == "global" for p in pools)
    tasks = scheduler.global_task_matrix()
    assert len(tasks) == len({scheduler.task_id(t) for t in tasks}) == 53
    assert {kind: sum(t["kind"] == kind for t in tasks) for kind in scheduler.KINDS} == dict(
        history=12, pilot=4, diagnostic=13, train=12, endpoint=12
    )
    assert all(p["task_matrix_sha256"] == object_hash(tasks) for p in pools)


def queue_for(controller):
    return scheduler.GlobalQueue(controller.root, controller.identity())


def finish_claim(queue, claim):
    queue.transition(claim, pid=123)
    return queue.transition(
        claim, result=dict(**claim["task"], receipt_sha256=object_hash(claim["task"]))
    )


def readiness_receipts(controller):
    queue = queue_for(controller)
    states = queue._states()
    done = {key for key, state in states.items() if state["status"] == "COMPLETE"}
    matrix = scheduler.global_task_matrix()
    for kind, name, extra in (
        ("history", "HISTORY_COMPLETE.json", dict(status="COMPLETE", paths=12)),
        ("pilot", "TECHNICAL_CHECK_SR_F2.json", dict(status="PASS")),
    ):
        if {scheduler.task_id(t) for t in matrix if t["kind"] == kind} <= done:
            save_json(
                controller.root / name,
                dict(**extra, registration_sha256=controller.identity()),
                exclusive=True,
            )


def finish_before_science(controller):
    queue = queue_for(controller)
    for _ in range(16):
        claim, _ = queue.claim("GPU_POOL_1", "0", "100")
        assert claim["task"]["kind"] in {"history", "pilot"}
        finish_claim(queue, claim)
    readiness_receipts(controller)
    return queue


def test_history_prioritized_with_pilot_backfill_before_slowest_history_finishes(controller):
    queue = queue_for(controller)
    slow, _ = queue.claim("GPU_POOL_1", "0", "100")
    for _ in range(11):
        claim, _ = queue.claim("GPU_POOL_2", "4", "101")
        assert claim["task"]["kind"] == "history"
        finish_claim(queue, claim)
    pilot, snapshot = queue.claim("GPU_POOL_2", "4", "101")
    assert pilot["task"]["kind"] == "pilot"
    assert snapshot["active"] == 1
    assert queue._states()[scheduler.task_id(slow["task"])]["status"] == "CLAIMED"
    assert not any(t["kind"] == "train" for t in snapshot["ready"])


def test_diagnostics_backfill_without_science_gates_and_idle_explains_real_dependency(controller):
    queue = queue_for(controller)
    slow, _ = queue.claim("GPU_POOL_1", "0", "100")
    kinds = []
    for _ in range(28):
        claim, _ = queue.claim("GPU_POOL_2", "4", "101")
        kinds.append(claim["task"]["kind"])
        finish_claim(queue, claim)
    assert kinds.count("diagnostic") == 13
    claim, snapshot = queue.claim("GPU_POOL_2", "4", "101")
    assert claim is None and snapshot["idle_reason"] == "WAIT_TECHNICAL_CPU_READINESS"
    finish_claim(queue, slow)
    readiness_receipts(controller)
    assert queue.claim("GPU_POOL_2", "4", "101")[0]["task"]["kind"] == "train"


def test_science_gates_require_complete_history_pilots_and_cpu_receipts(controller):
    queue = queue_for(controller)
    for _ in range(16):
        claim, _ = queue.claim("GPU_POOL_1", "0", "100")
        finish_claim(queue, claim)
    assert not queue.snapshot()["science_ready"]
    assert all(t["kind"] == "diagnostic" for t in queue.snapshot()["ready"])
    readiness_receipts(controller)
    assert queue.snapshot()["science_ready"]
    assert len([t for t in queue.snapshot()["ready"] if t["kind"] == "train"]) == 12


def test_fast_slot_rolls_to_next_seed_without_waiting_for_slow_seed_peer(controller):
    queue = finish_before_science(controller)
    first = []
    for device in ("0", "1", "2", "3"):
        claim, _ = queue.claim("GPU_POOL_1", device, "100")
        first.append(claim)
    assert all(c["task"]["model_id"].endswith("s71001") for c in first)
    finish_claim(queue, first[0])
    new, snapshot = queue.claim("GPU_POOL_1", "0", "100")
    assert new["task"]["model_id"].endswith("s71002")
    assert snapshot["active"] == 3
    endpoint = dict(kind="endpoint", model_id=first[0]["task"]["model_id"])
    assert endpoint in snapshot["ready"]
    assert not any(
        t["kind"] == "endpoint" and t["model_id"] == first[1]["task"]["model_id"]
        for t in snapshot["ready"]
    )


def test_global_claim_lock_prevents_cross_pool_duplicates(controller):
    from concurrent.futures import ThreadPoolExecutor

    queue = queue_for(controller)
    assignments = [
        ("GPU_POOL_1" if i < 4 else "GPU_POOL_2", str(i), "100" if i < 4 else "101")
        for i in range(8)
    ]
    with ThreadPoolExecutor(max_workers=8) as workers:
        claims = list(workers.map(lambda args: queue.claim(*args)[0], assignments))
    assert len({scheduler.task_id(c["task"]) for c in claims}) == 8
    assert all(c["task"]["kind"] == "history" for c in claims)
    with pytest.raises(PermissionError, match="already owns"):
        queue.claim("GPU_POOL_1", "0", "100")


def test_failed_or_orphan_claim_never_retries(controller):
    queue = queue_for(controller)
    claim, _ = queue.claim("GPU_POOL_1", "0", "100")
    queue.transition(claim, error="synthetic failure")
    with pytest.raises(RuntimeError, match="stopped"):
        queue.claim("GPU_POOL_2", "4", "101")
    with pytest.raises(PermissionError, match="terminal"):
        queue.transition(claim, pid=123)


def test_endpoint_seal_needs_all53_tasks_and_both_pool_terminal_states(controller):
    queue = finish_before_science(controller)
    while True:
        claim, snapshot = queue.claim("GPU_POOL_1", "0", "100")
        if claim is None:
            assert snapshot["all_complete"]
            break
        finish_claim(queue, claim)
    mark_complete(controller, "GPU_POOL_1")
    # One resident allocation remains unsubmitted: task artifacts alone are insufficient.
    result = controller.tick()
    assert not result["endpoint_seal_release_ready"]
    mark_complete(controller, "GPU_POOL_2")
    result = controller.tick()
    assert result["all_gpu_tasks_complete"] and result["endpoint_seal_release_ready"]
    assert result["stage"] == "ALL_GPU_TASKS_COMPLETE_REPORT_PENDING"


def test_gpu_pool_dynamic_queue_reuses_single_visible_slots_until_all53(controller, monkeypatch):
    controller.submit("GPU_POOL_1", {})
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    launched = []
    monkeypatch.setattr(
        scheduler,
        "verify_task",
        lambda root, task, registration_hash: dict(**task, receipt_sha256=object_hash(task)),
    )
    monkeypatch.setattr(scheduler.time, "sleep", lambda _: readiness_receipts(controller))

    def popen(command, **kwargs):
        launched.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        return SimpleNamespace(pid=1000 + len(launched), poll=lambda: 0)

    result = run_pool(
        controller.root,
        controller.pool_dir("GPU_POOL_1") / "PLAN.json",
        poll_seconds=0,
        popen=popen,
    )
    assert result["status"] == "COMPLETE" and len(result["tasks"]) == 53
    assert len(launched) == 53 and {d for _, d in launched} == {"4", "5", "6", "7"}
    assert all(command[2:4] == ["--root", str(controller.root)] for command, _ in launched)
    assert scheduler.verify_pool_task_ownership(controller.root, "GPU_POOL_1", result["tasks"])
    with pytest.raises(PermissionError, match="already recorded"):
        run_pool(
            controller.root,
            controller.pool_dir("GPU_POOL_1") / "PLAN.json",
            poll_seconds=0,
            popen=popen,
        )
    assert len(launched) == 53


def test_empty_pool_releases_when_all_remaining_work_is_claimed_by_other_pool(
    controller, monkeypatch
):
    controller.submit("GPU_POOL_1", {})
    queue = queue_for(controller)
    for index in range(52):
        claim, _ = queue.claim("GPU_POOL_1", "4", "100")
        completed = finish_claim(queue, claim)
        directory = controller.pool_dir("GPU_POOL_1") / "workers" / scheduler.task_id(claim["task"])
        save_json(
            directory / "LAUNCH_INTENT.json",
            dict(task=claim["task"], device="4", claim_id=claim["claim_id"]),
        )
        save_json(directory / "PROCESS_EXIT.json", dict(returncode=0))
        assert completed["status"] == "COMPLETE"
        if index == 15:
            readiness_receipts(controller)
    tail, _ = queue.claim("GPU_POOL_2", "0", "101")
    queue.transition(tail, pid=54321)
    assert queue.snapshot()["unclaimed"] == 0
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    def no_launch(*args, **kwargs):
        raise AssertionError("An empty pool must not launch or wait for another pool's tail")

    monkeypatch.setattr(scheduler.time, "sleep", no_launch)
    result = run_pool(
        controller.root, controller.pool_dir("GPU_POOL_1") / "PLAN.json", popen=no_launch
    )
    assert result["status"] == "COMPLETE"
    assert result["completion_reason"] == "NO_UNCLAIMED_TASKS_AND_NO_OWN_ACTIVE"
    assert result["global_completed_tasks"] == len(result["tasks"]) == 52
    assert not queue.snapshot()["all_complete"]
    assert queue._states()[scheduler.task_id(tail["task"])]["status"] == "RUNNING"
    assert not controller.tick()["endpoint_seal_release_ready"]
