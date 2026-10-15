from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f3_ce import orchestration as scheduler
from sr_f3_ce.orchestration import (
    TEACHER_QOS,
    Controller,
    file_hash,
    read_json,
    save_json,
    task_matrix,
    teacher_usage,
    write_task_receipt,
)


class FakeSlurm:
    def __init__(self):
        self.rows, self.scripts, self.accounting = {}, {}, {}
        self.calls = []
        self.queue_calls = 0
        self.next_job = 100
        self.ambiguous_submit = self.ambiguous_release = False
        self.alter_script = False

    def run(self, command):
        self.calls.append(command)
        if command[0] == "sbatch":
            args = dict(value[2:].split("=", 1) for value in command[1:-1] if "=" in value)
            job = str(self.next_job)
            self.next_job += 1
            count = int(args["gres"].rsplit(":", 1)[1]) if "gres" in args else 0
            tres = f"cpu={args['cpus-per-task']},mem={args['mem']},node=1"
            if count:
                tres += f",gres/gpu={count},gres/gpu:pro6000={count}"
            self.rows[job] = dict(
                JobId=job, JobName=args["job-name"], Account=args["account"],
                Partition=args["partition"], QOS=args["qos"], Command=command[-1],
                Comment=args["comment"], NumCPUs=args["cpus-per-task"], UserId="alice(1)",
                NumNodes="1", ReqTRES=tres, Features="highmem" if count else "(null)",
                ReqNodeList=args.get("nodelist", "(null)"), JobState="PENDING",
                Reason="JobHeldUser", TimeLimit="3-00:00:00", Requeue="0",
            )
            self.scripts[job] = Path(command[-1]).read_text()
            return SimpleNamespace(returncode=1 if self.ambiguous_submit else 0,
                                   stdout="" if self.ambiguous_submit else job + "\n",
                                   stderr="ambiguous" if self.ambiguous_submit else "")
        if command[:2] == ["scontrol", "release"]:
            if not self.ambiguous_release:
                self.rows[command[2]].update(JobState="RUNNING", Reason="None")
            return SimpleNamespace(returncode=int(self.ambiguous_release), stdout="", stderr="")
        raise AssertionError(command)

    def show(self, job):
        return dict(self.rows[job])

    def queue(self, user):
        self.queue_calls += 1
        return {job: dict(row) for job, row in self.rows.items()}

    def batch_script(self, job):
        return self.scripts[job] + ("# changed\n" if self.alter_script else "")

    def terminal(self, job):
        return self.accounting.get(job)


@pytest.fixture
def controller(tmp_path, monkeypatch):
    root, code = tmp_path / "run", tmp_path / "run/code"
    root.mkdir()
    code.mkdir()
    scripts = code / "scripts/sr_f3_ce"
    scripts.mkdir(parents=True)
    for name in ("submit.py", "run_worker.py"):
        (scripts / name).write_text("# fake Slurm test entry point\n")
    save_json(root / "RUN_IDENTITY.json", dict(experiment_id=scheduler.EXPERIMENT_ID))
    monkeypatch.setattr("sr_f3_ce.protocol.verify_run", lambda *args, **kwargs: ({}, {}))
    return Controller(root, code, python=sys.executable, slurm=FakeSlurm(), user="alice",
                      pool_size=2, cpu_qos="override-limits-but-killable")


def evidence(task_id):
    task = scheduler.registered_task(task_id)
    if task_id == "p0":
        return dict.fromkeys(("cpu_tests", "independent_answers", "historical_dedup", "render",
                              "processor", "input_leakage"), "PASS")
    if task_id == "engine--resume":
        return dict(logical_updates=4, physical_updates=6, new_rollouts=512, restore="PASS",
                    masks="PASS", probabilities="PASS", nonzero_ce_gradient=True)
    if task["kind"] == "engine":
        return dict(parent_id=task["parent_id"], diagnostic_rows=960, protocol_gate="PASS")
    if task["kind"] == "train":
        return dict(updates=64, rollout_rows=8192,
                    support_examples=0 if "_RL_" in task_id else 256,
                    monitor_steps=[0, 16, 32, 48, 64], checkpoint_sha256="c" * 64)
    if task["kind"] == "eval":
        return dict(structured_rows=10752, alternate_protocol_rows=2048, chartqa_rows=2500,
                    chartqapro_rows=256, chartqapro_status="COMPLETE", sealed=True)
    return dict(training_paths=12, evaluated_models=15, raw_rescoring="PASS")


def complete_local_p0(controller):
    root = controller.root
    artifact = root / "p0-output.json"
    artifact.write_text('{"checks":"complete"}')
    return write_task_receipt(root, "p0", artifacts=[artifact.name], evidence=evidence("p0"),
                              local_p0=True)


def complete_allocation(controller, state, monkeypatch):
    job, root = state["job_id"], controller.root
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(state["gpu_count"])))

    def popen(command, env):
        task_id = command[-1]
        with monkeypatch.context() as patch:
            for key, value in env.items():
                patch.setenv(key, value)
            output = root / (task_id + ".raw")
            output.write_text("registered raw output")
            metadata = evidence(task_id)
            if scheduler.TASKS[task_id]["kind"] == "train":
                metadata["checkpoint_sha256"] = file_hash(output)
            write_task_receipt(root, task_id, artifacts=[output.name], evidence=metadata)
        return SimpleNamespace(wait=lambda: 0)

    scheduler.run_job(root, state["key"], popen=popen)
    row = controller.slurm.rows.pop(job)
    controller.slurm.accounting[job] = dict(
        job_id=job, state="COMPLETED", exit_code="0:0", user="alice", account="rose",
        qos=row["QOS"], name=row["JobName"],
    )
    return controller.status()


def finish_engines(controller, monkeypatch):
    complete_local_p0(controller)
    for _ in range(2):
        complete_allocation(controller, controller.submit("engine"), monkeypatch)
    save_json(controller.root / "EXECUTION_FREEZE.json", dict(
        run_identity_sha256=scheduler.run_identity(controller.root),
        engine_receipts={key: file_hash(scheduler.receipt_path(controller.root, key))
                         for key in scheduler.TASKS if key.startswith("engine--")},
    ))


def test_matrix_is_exact_and_every_new_job_name_has_lh_suffix(controller):
    rows = task_matrix()
    assert len(rows) == len({row["task_id"] for row in rows}) == 33
    assert {kind: sum(row["kind"] == kind for row in rows) for kind in scheduler.KINDS} == dict(
        p0=1, engine=4, train=12, eval=15, release=1,
    )
    state = controller.submit("p0")
    intent = read_json(controller.directory / "jobs" / state["key"] / "INTENT.json")
    assert intent["job_name"].endswith("-LH")
    assert intent["gpu_count"] == 0 and intent["qos"] != TEACHER_QOS
    assert controller.slurm.calls[0][1:3] == ["--hold", "--parsable"]
    assert controller.slurm.calls[1] == ["scontrol", "release", state["job_id"]]


def test_controller_job_is_cpu_only_and_lh(controller):
    state = controller.submit("controller")
    intent = read_json(controller.directory / "jobs/controller/INTENT.json")
    assert state["gpu_count"] == 0
    assert intent["job_name"].endswith("-LH") and intent["qos"] != TEACHER_QOS
    assert not any(arg.startswith("--gres") for arg in intent["sbatch"])
    assert controller.submit("controller") is None


def test_capacity_counts_all_qos_users_requested_allocated_and_unknown():
    queue = {
        "1": dict(QOS=TEACHER_QOS, UserId="bob(2)", ReqTRES="gres/gpu=1", AllocTRES="gres/gpu=2"),
        "2": dict(QOS=TEACHER_QOS, UserId="alice(1)", ReqTRES="gres/gpu=1"),
        "3": dict(QOS="other", UserId="alice(1)", ReqTRES="gres/gpu=99"),
    }
    attempts = [dict(status="SUBMISSION_UNKNOWN", gpu_count=2, qos=TEACHER_QOS)]
    assert teacher_usage(queue, attempts, "alice") == dict(gpus=5, submitted=2)
    with pytest.raises(PermissionError, match="unknown"):
        teacher_usage({"1": dict(QOS=TEACHER_QOS, UserId="bob(2)")}, [], "alice")


def test_fresh_queries_and_one_remaining_gpu_shrink_pool(controller):
    complete_local_p0(controller)
    controller.slurm.rows["9"] = dict(QOS=TEACHER_QOS, UserId="bob(2)", ReqTRES="gres/gpu=4")
    state = controller.submit("engine")
    assert state["gpu_count"] == 1 and controller.slurm.queue_calls >= 2
    intent = read_json(controller.directory / "jobs" / state["key"] / "INTENT.json")
    assert intent["job_name"].endswith("-LH") and len(intent["task_ids"]) == 1
    calls = controller.slurm.queue_calls
    assert controller.submit("engine") is None
    assert controller.slurm.queue_calls > calls
    assert sum(call[0] == "sbatch" for call in controller.slurm.calls) == 1


def test_max_five_teacher_submitted_jobs_even_if_cpu_only(controller):
    complete_local_p0(controller)
    for i in range(5):
        controller.slurm.rows[str(i)] = dict(QOS=TEACHER_QOS, UserId="alice(1)", ReqTRES="cpu=1")
    assert controller.submit("engine") is None
    assert controller.slurm.calls == []


def test_unknown_sbatch_result_reserves_capacity_and_never_retries(controller):
    complete_local_p0(controller)
    controller.slurm.ambiguous_submit = True
    state = controller.submit("engine")
    assert state["status"] == "SUBMISSION_UNKNOWN"
    with pytest.raises(RuntimeError, match="no automatic retry"):
        controller.submit("engine")
    assert len(controller.slurm.calls) == 1
    # The unbound result plus its visible allocation are conservatively double-counted.
    assert controller.status()["teacher_usage"]["gpus"] == 4


def test_exception_during_sbatch_leaves_durable_unknown_intent(controller, monkeypatch):
    complete_local_p0(controller)

    def disconnected(command):
        raise TimeoutError("transport failed")

    monkeypatch.setattr(controller.slurm, "run", disconnected)
    with pytest.raises(TimeoutError):
        controller.submit("engine")
    assert controller.attempts()[0]["status"] == "SUBMISSION_UNKNOWN"
    assert controller.status()["blocked"]


def test_release_unknown_is_never_retried(controller):
    complete_local_p0(controller)
    controller.slurm.ambiguous_release = True
    assert controller.submit("engine")["status"] == "RELEASE_UNKNOWN"
    assert controller.tick()["blocked"]
    assert len(controller.slurm.calls) == 2


def test_actual_stored_script_must_equal_local_script(controller):
    complete_local_p0(controller)
    controller.slurm.alter_script = True
    with pytest.raises(PermissionError, match="stored batch script"):
        controller.submit("engine")
    assert controller.attempts()[0]["status"] == "HELD"
    assert len(controller.slurm.calls) == 1


def test_scheduler_automatic_requeue_is_never_allowed(controller, monkeypatch):
    complete_local_p0(controller)
    original_show = controller.slurm.show
    monkeypatch.setattr(controller.slurm, "show", lambda job: original_show(job) | dict(
        Requeue="1"))
    with pytest.raises(PermissionError, match="requeue must be disabled"):
        controller.submit("engine")
    assert "--no-requeue" in controller.slurm.calls[0]
    assert len(controller.slurm.calls) == 1


def test_capacity_race_never_releases_over_limit(controller, monkeypatch):
    complete_local_p0(controller)
    original_queue = controller.slurm.queue

    def queue(user):
        rows = original_queue(user)
        if controller.slurm.calls:
            rows["999"] = dict(QOS=TEACHER_QOS, UserId="bob(2)", ReqTRES="gres/gpu=4")
        return rows

    monkeypatch.setattr(controller.slurm, "queue", queue)
    with pytest.raises(PermissionError, match="capacity exceeded"):
        controller.submit("engine")
    assert len(controller.slurm.calls) == 1


def test_missing_queue_never_submits(controller, monkeypatch):
    monkeypatch.setattr(controller.slurm, "queue", lambda user: (_ for _ in ()).throw(
        RuntimeError("queue unavailable")))
    with pytest.raises(RuntimeError, match="queue unavailable"):
        controller.submit("p0")
    assert controller.slurm.calls == []


def test_only_completion_and_frozen_identity_unlock_stages(controller, monkeypatch):
    assert controller.ready_tasks() == ["p0"]
    assert controller.submit("engine") is None
    complete_local_p0(controller)
    assert controller.ready_tasks() == ["engine--b0", "engine--b1", "engine--b2", "engine--resume"]
    for _ in range(2):
        complete_allocation(controller, controller.submit("engine"), monkeypatch)
    assert controller.ready_tasks() == []  # Final execution freeze is independently required.
    assert controller.submit("train") is None
    save_json(controller.root / "EXECUTION_FREEZE.json", dict(
        run_identity_sha256=scheduler.run_identity(controller.root),
        engine_receipts={key: file_hash(scheduler.receipt_path(controller.root, key))
                         for key in scheduler.TASKS if key.startswith("engine--")},
    ))
    assert len(controller.ready_tasks()) == 12
    assert all(key.startswith("train--") for key in controller.ready_tasks())
    assert controller.submit("eval") is None


def test_all_twelve_train_paths_before_fifteen_eval_then_release(controller, monkeypatch):
    finish_engines(controller, monkeypatch)
    for index in range(6):
        complete_allocation(controller, controller.submit("train"), monkeypatch)
        assert len([key for key in controller.ready_tasks() if key.startswith("eval--")]) == (
            15 if index == 5 else 0
        )
    for index in range(8):
        state = controller.submit("eval")
        assert state["gpu_count"] == (1 if index == 7 else 2)
        complete_allocation(controller, state, monkeypatch)
    assert controller.ready_tasks() == ["release"]
    complete_allocation(controller, controller.submit("release"), monkeypatch)
    assert controller.status()["all_complete"]
    assert all("-LH" in call[-3] or any(arg.endswith("-LH") for arg in call)
               for call in controller.slurm.calls if call[0] == "sbatch")


def test_spoofed_task_and_changed_artifact_cannot_unlock_engine(controller):
    receipt = complete_local_p0(controller)
    receipt["task"] = scheduler.registered_task("engine--b0")
    save_json(scheduler.receipt_path(controller.root, "p0"), receipt)
    with pytest.raises(PermissionError, match="identity/status"):
        controller.ready_tasks()


def test_artifact_escape_and_false_counts_are_rejected(controller):
    outside = controller.root.parent / "outside"
    outside.write_text("not an experiment artifact")
    with pytest.raises(PermissionError, match="escapes"):
        write_task_receipt(controller.root, "p0", artifacts=["../outside"],
                           evidence=evidence("p0"), local_p0=True)
    bad = evidence("engine--resume") | dict(physical_updates=7)
    with pytest.raises(PermissionError, match="physical_updates"):
        scheduler.validate_evidence("engine--resume", bad)


def test_changed_artifact_is_rehashed_on_every_readiness(controller):
    complete_local_p0(controller)
    (controller.root / "p0-output.json").write_text("changed")
    with pytest.raises(PermissionError, match="artifact bytes"):
        controller.ready_tasks()


def test_lookalike_job_owner_and_terminal_identity_are_rejected(controller):
    state = controller.submit("p0")
    controller.slurm.rows[state["job_id"]]["UserId"] = "someone_else(2)"
    with pytest.raises(PermissionError, match="owner"):
        controller.status()
    row = controller.slurm.rows.pop(state["job_id"])
    controller.slurm.accounting[state["job_id"]] = dict(
        job_id=state["job_id"], user="alice", account="another", qos=row["QOS"],
        name=row["JobName"], state="COMPLETED", exit_code="0:0",
    )
    with pytest.raises(PermissionError, match="accounting identity"):
        controller.status()


def test_unknown_accounting_blocks_and_preserves_reservation(controller):
    complete_local_p0(controller)
    state = controller.submit("engine")
    controller.slurm.rows.pop(state["job_id"])
    status = controller.status()
    assert status["blocked"] and status["teacher_usage"]["gpus"] == 2
    assert controller.attempts()[0]["status"] == "ACCOUNTING_UNKNOWN"


def test_completed_state_cannot_impersonate_another_account(controller, monkeypatch):
    state = controller.submit("p0")
    complete_allocation(controller, state, monkeypatch)
    terminal_path = controller.directory / "jobs" / state["key"] / "TERMINAL.json"
    terminal = read_json(terminal_path)
    terminal["user"] = "bob"
    save_json(terminal_path, terminal)
    with pytest.raises(PermissionError, match="accounting identity"):
        controller.ready_tasks()


def test_completed_state_without_real_process_evidence_fails(controller):
    state = controller.submit("p0")
    directory = controller.directory / "jobs" / state["key"]
    row = controller.slurm.rows.pop(state["job_id"])
    controller.slurm.accounting[state["job_id"]] = dict(
        job_id=state["job_id"], user="alice", account="rose", qos=row["QOS"],
        name=row["JobName"], state="COMPLETED", exit_code="0:0",
    )
    status = controller.status()
    assert status["blocked"] and controller.attempts()[0]["status"] == "FAILED"
    assert (directory / "VERIFICATION_FAILURE.json").is_file()
    assert controller.ready_tasks() == []


def test_orphan_state_and_modified_engine_freeze_are_rejected(controller, monkeypatch):
    finish_engines(controller, monkeypatch)
    freeze_path = controller.root / "EXECUTION_FREEZE.json"
    freeze = read_json(freeze_path)
    freeze["engine_receipts"]["engine--resume"] = "0" * 64
    save_json(freeze_path, freeze)
    with pytest.raises(PermissionError, match="ENGINE bindings"):
        controller.ready_tasks()
    save_json(controller.directory / "jobs/orphan/STATE.json", dict(status="COMPLETE"))
    with pytest.raises(PermissionError, match="Orphan submission"):
        controller.attempts()


def test_pool_exits_after_fixed_work_without_waiting_for_unready_tasks(controller, monkeypatch):
    complete_local_p0(controller)
    state = controller.submit("engine")

    def no_sleep(_):
        raise AssertionError("A GPU pool must exit when its fixed claims finish")

    monkeypatch.setattr(scheduler.time, "sleep", no_sleep)
    status = complete_allocation(controller, state, monkeypatch)
    assert status["ready"] == ["engine--b2", "engine--resume"]


def test_generation_scores_are_not_a_scheduler_input(controller, monkeypatch):
    finish_engines(controller, monkeypatch)
    for task_id in ("engine--b0", "engine--b1", "engine--b2", "engine--resume"):
        raw = controller.root / (task_id + ".scores.json")
        raw.write_text('{"accuracy":0,"best_model":"never_read"}')
    assert len(controller.ready_tasks()) == 12


def test_explicit_teacher_cpu_policy_uses_zero_gpu_and_counts_submission(controller):
    (controller.directory / "CONFIG.json").unlink()
    controller = Controller(controller.root, controller.code, python=sys.executable,
                            slurm=controller.slurm, user="alice", cpu_qos=TEACHER_QOS)
    state = controller.submit("p0")
    assert state["gpu_count"] == 0 and state["qos"] == TEACHER_QOS
    assert controller.status()["teacher_usage"] == dict(gpus=0, submitted=1)
    intent = read_json(controller.directory / "jobs" / state["key"] / "INTENT.json")
    assert not any(arg.startswith("--gres") for arg in intent["sbatch"])


def test_unknown_teacher_cpu_submission_reserves_a_submission_slot(controller):
    attempts = [dict(status="SUBMISSION_UNKNOWN", gpu_count=0, qos=TEACHER_QOS)]
    assert teacher_usage({}, attempts, "alice") == dict(gpus=0, submitted=1)


def test_cpu_qos_requires_explicit_current_authorization(controller):
    (controller.directory / "CONFIG.json").unlink()
    controller = Controller(controller.root, controller.code, python=sys.executable,
                            slurm=controller.slurm, user="alice")
    with pytest.raises(PermissionError, match="explicitly verified --cpu-qos"):
        controller.submit("p0")
    assert controller.slurm.calls == []


def test_missing_worker_is_detected_before_sbatch(controller):
    (controller.code / "scripts/sr_f3_ce/run_worker.py").unlink()
    with pytest.raises(PermissionError, match="entry point missing"):
        controller.submit("p0")
    assert controller.slurm.calls == []


def test_mutated_durable_intent_cannot_launch_another_task(controller):
    complete_local_p0(controller)
    state = controller.submit("engine")
    intent_path = controller.directory / "jobs" / state["key"] / "INTENT.json"
    intent = read_json(intent_path)
    intent["task_ids"] = ["engine--b2", "engine--resume"]
    save_json(intent_path, intent)
    with pytest.raises(PermissionError, match="intent changed"):
        controller.status()


def test_intent_without_state_remains_reserved_unknown(controller):
    complete_local_p0(controller)
    controller.slurm.ambiguous_submit = True
    state = controller.submit("engine")
    (controller.directory / "jobs" / state["key"] / "STATE.json").unlink()
    assert controller.status()["blocked"]
    assert controller.attempts()[0]["status"] == "SUBMISSION_UNKNOWN"
    with pytest.raises(RuntimeError, match="no automatic retry"):
        controller.submit("engine")


@pytest.mark.parametrize("tres", [
    "cpu=4,gres/gpu:pro6000:5", "gres/gpu=-1", "gres/gpu=1,gres/gpu=1", "gres/gpu=unknown",
])
def test_malformed_gpu_tres_is_never_interpreted_as_free_capacity(tres):
    with pytest.raises(PermissionError):
        teacher_usage({"9": dict(QOS=TEACHER_QOS, UserId="bob(2)", ReqTRES=tres)}, [], "alice")


def single_gpu_controller(controller):
    (controller.directory / "CONFIG.json").unlink()
    return Controller(controller.root, controller.code, python=sys.executable,
                      slurm=controller.slurm, user="alice", pool_size=1,
                      cpu_qos="override-limits-but-killable")


def expire_allocation(controller, state, outcome="TIMEOUT"):
    job = state["job_id"]
    row = controller.slurm.rows.pop(job)
    controller.slurm.accounting[job] = dict(
        job_id=job, state=outcome, exit_code="0:15", user="alice", account="rose",
        qos=row["QOS"], name=row["JobName"],
    )
    return controller.status()


def resume_verifier(root, task_id, previous_key):
    path = root / f"immutable-cache-{previous_key}.bin"
    path.write_bytes(b"verified atomic checkpoint or raw-group chain")
    previous = read_json(root / "orchestration/jobs" / previous_key / "STATE.json")
    value = dict(status="PASS", task_id=task_id, previous_job_key=previous_key,
                 previous_job_id=previous["job_id"],
                 run_identity_sha256=scheduler.run_identity(root),
                 resume_kind="checkpoint" if task_id.startswith("train--") or (
                     task_id == "engine--resume") else "generation_cache",
                 artifacts=[scheduler.artifact_record(root, path.name)],
                 resume_state_sha256=file_hash(path))
    if task_id.startswith(("train--", "eval--")):
        value["execution_freeze_sha256"] = scheduler.verify_freeze(root)
    return value


def expired_engine(controller):
    controller = single_gpu_controller(controller)
    complete_local_p0(controller)
    previous = controller.submit("engine")
    assert expire_allocation(controller, previous)["blocked"]
    return controller, previous


def test_explicit_lease_resume_preserves_old_attempt_and_finishes_same_task(
    controller, monkeypatch,
):
    controller, previous = expired_engine(controller)
    original = read_json(controller.directory / "jobs" / previous["key"] / "INTENT.json")
    with pytest.raises(RuntimeError, match="no automatic retry"):
        controller.submit("engine")
    resumed = controller.resume(previous["key"], verifier=resume_verifier)
    assert resumed["key"] != previous["key"] and resumed["job_id"] != previous["job_id"]
    assert read_json(controller.directory / "jobs" / previous["key"] / "INTENT.json") == original
    intent = read_json(controller.directory / "jobs" / resumed["key"] / "INTENT.json")
    assert intent["resume_of"] == previous["key"] and intent["task_ids"] == original["task_ids"]
    assert intent["gpu_count"] == 1 and intent["job_name"].endswith("-LH")
    assert not controller.status()["blocked"]
    calls = len(controller.slurm.calls)
    again = controller.resume(previous["key"], verifier=lambda *a: pytest.fail("Must not reverify"))
    assert again == resumed and len(controller.slurm.calls) == calls
    status = complete_allocation(controller, resumed, monkeypatch)
    assert "engine--b0" in status["completed"]
    assert status["resumed_lease_failures"] == {previous["key"]: resumed["key"]}
    assert read_json(controller.directory / "jobs" / previous["key"] / "STATE.json")["status"] == (
        "FAILED"
    )


@pytest.mark.parametrize("outcome", ["FAILED", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"])
def test_only_timeout_or_preemption_can_use_lease_resume(controller, outcome):
    controller = single_gpu_controller(controller)
    complete_local_p0(controller)
    previous = controller.submit("engine")
    expire_allocation(controller, previous, outcome)
    with pytest.raises(PermissionError, match="terminal accounting"):
        controller.resume(previous["key"], verifier=lambda *a: pytest.fail("Must not verify"))
    assert sum(command[0] == "sbatch" for command in controller.slurm.calls) == 1


def test_resume_refuses_still_visible_job_or_changed_terminal_identity(controller):
    controller = single_gpu_controller(controller)
    complete_local_p0(controller)
    previous = controller.submit("engine")
    with pytest.raises(PermissionError, match="terminal single-GPU"):
        controller.resume(previous["key"], verifier=resume_verifier)
    expire_allocation(controller, previous)
    controller.slurm.accounting[previous["job_id"]]["user"] = "imposter"
    with pytest.raises(PermissionError, match="terminal accounting"):
        controller.resume(previous["key"], verifier=resume_verifier)


def test_resume_verifier_failure_creates_no_new_submission(controller):
    controller, previous = expired_engine(controller)

    def broken_verifier(*args):
        return resume_verifier(*args) | dict(status="FAIL")

    with pytest.raises(PermissionError, match="verification identity/status"):
        controller.resume(previous["key"], verifier=broken_verifier)
    assert len(controller.attempts()) == 1 and controller.status()["blocked"]


def test_resume_capacity_rechecked_after_checkpoint_verification(controller):
    controller, previous = expired_engine(controller)

    def filling_verifier(*args):
        value = resume_verifier(*args)
        controller.slurm.rows["999"] = dict(QOS=TEACHER_QOS, UserId="bob(2)", ReqTRES="gres/gpu=5")
        return value

    assert controller.resume(previous["key"], verifier=filling_verifier) is None
    assert len(controller.attempts()) == 1


def test_resume_unknown_submission_is_reserved_and_never_repeated(controller):
    controller, previous = expired_engine(controller)
    controller.slurm.ambiguous_submit = True
    resumed = controller.resume(previous["key"], verifier=resume_verifier)
    assert resumed["status"] == "SUBMISSION_UNKNOWN"
    calls = len(controller.slurm.calls)
    assert controller.resume(previous["key"], verifier=resume_verifier) == resumed
    assert controller.status()["blocked"] and len(controller.slurm.calls) == calls


def test_resumption_evidence_is_rehashed_and_mutable_pointers_rejected(controller):
    controller, previous = expired_engine(controller)

    def mutable_verifier(*args):
        value = resume_verifier(*args)
        path = controller.root / "LATEST.json"
        path.write_text("mutable")
        return value | dict(artifacts=[scheduler.artifact_record(controller.root, path.name)])

    with pytest.raises(PermissionError, match="mutable pointer"):
        controller.resume(previous["key"], verifier=mutable_verifier)
    controller.resume(previous["key"], verifier=resume_verifier)
    (controller.root / f"immutable-cache-{previous['key']}.bin").write_bytes(b"changed")
    with pytest.raises(PermissionError, match="artifact bytes"):
        controller.status()


def test_repeated_verified_leases_form_one_chain_without_erasing_failures(controller):
    controller, first = expired_engine(controller)
    second = controller.resume(first["key"], verifier=resume_verifier)
    expire_allocation(controller, second, "PREEMPTED")
    third = controller.resume(second["key"], verifier=resume_verifier)
    status = controller.status()
    assert not status["blocked"]
    assert status["resumed_lease_failures"] == {
        first["key"]: second["key"], second["key"]: third["key"],
    }
    assert sum(row["status"] == "FAILED" for row in status["attempts"]) == 2


def test_slurm_bound_cpu_p0_is_not_mistaken_for_local_receipt(controller, monkeypatch):
    state = controller.submit("p0")
    complete_allocation(controller, state, monkeypatch)
    receipt = read_json(scheduler.receipt_path(controller.root, "p0"))
    assert receipt["job_id"] == state["job_id"] != "LOCAL_P0"
    assert controller.completed() == {"p0"}


def test_recovery_submission_limit_and_two_gpu_pool_are_fail_closed(controller):
    complete_local_p0(controller)
    previous = controller.submit("engine")
    expire_allocation(controller, previous)
    with pytest.raises(PermissionError, match="single-GPU"):
        controller.resume(previous["key"], verifier=resume_verifier)
    for index in range(5):
        controller.slurm.rows[str(index)] = dict(
            QOS=TEACHER_QOS, UserId="alice(1)", ReqTRES="cpu=1")
    assert controller.resume(previous["key"], verifier=lambda *a: pytest.fail(
        "No capacity")) is None


def test_existing_completion_is_never_replayed_after_lease_expiry(controller):
    controller, previous = expired_engine(controller)
    save_json(scheduler.receipt_path(controller.root, "engine--b0"), dict(status="PASS"))
    with pytest.raises(PermissionError, match="receipt-only reconciliation"):
        controller.resume(previous["key"], verifier=resume_verifier)


def test_worker_receives_bound_resume_handoff(controller, monkeypatch):
    controller, previous = expired_engine(controller)
    resumed = controller.resume(previous["key"], verifier=resume_verifier)
    monkeypatch.setenv("SLURM_JOB_ID", resumed["job_id"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    launched = []

    def popen(command, env):
        launched.append(env)
        assert env["SRF3CE_PREVIOUS_JOB_KEY"] == previous["key"]
        path = Path(env["SRF3CE_RESUME_RECEIPT"])
        assert path == controller.directory / "jobs" / resumed["key"] / "VERIFIED_RESUME.json"
        assert read_json(path)["verification"]["task_id"] == command[-1]
        with monkeypatch.context() as patch:
            for key, value in env.items():
                patch.setenv(key, value)
            output = controller.root / "resumed-output.raw"
            output.write_text("registered completed raw")
            write_task_receipt(controller.root, command[-1], artifacts=[output.name],
                               evidence=evidence(command[-1]))
        return SimpleNamespace(wait=lambda: 0)

    scheduler.run_job(controller.root, resumed["key"], popen=popen)
    assert len(launched) == 1


def test_unverified_other_failure_still_blocks_explicit_recovery(controller):
    controller = single_gpu_controller(controller)
    complete_local_p0(controller)
    first, other = controller.submit("engine"), controller.submit("engine")
    expire_allocation(controller, first)
    expire_allocation(controller, other, "FAILED")
    with pytest.raises(RuntimeError, match="Other unresolved/technical failure"):
        controller.resume(first["key"], verifier=resume_verifier)


@pytest.mark.parametrize("value", ["0,0", "0", "", "MIG-abc,1", "0,1,2"])
def test_pool_requires_distinct_physical_devices(value):
    with pytest.raises(PermissionError):
        scheduler.visible_devices(value, 2)


def test_pool_refuses_wrong_slurm_binding(controller, monkeypatch):
    complete_local_p0(controller)
    state = controller.submit("engine")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    with pytest.raises(PermissionError, match="bound Slurm"):
        scheduler.run_job(controller.root, state["key"])


def test_controller_lease_prevents_double_submitters(tmp_path):
    with scheduler.lease(tmp_path), pytest.raises(RuntimeError), scheduler.lease(tmp_path):
        pass
