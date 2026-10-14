"""Fake Slurm and child-process checks; no test submits scientific work."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f2_replay import orchestration as replay


def fields(intent, job, *, state="RUNNING"):
    count = intent["gpu_count"]
    return dict(
        JobId=job,
        UserId="alice(1)",
        Account="rose",
        Partition="cluster02",
        QOS=replay.TEACHER_QOS,
        JobName=intent["job_name"],
        Comment=intent["comment"],
        Command=intent["script"],
        Features="highmem",
        JobState=state,
        Reason="JobHeldUser" if state == "PENDING" else "None",
        Requeue="0",
        ReqTRES=f"cpu={count * 4},mem={count * 90}G,gres/gpu={count},gres/gpu:pro6000={count}",
    )


class FakeSlurm:
    def __init__(self):
        self.jobs, self.accounting, self.calls = {}, {}, []
        self.fail_submit = self.fail_release = False
        self.after_submit = lambda: None

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
            intent = replay.read_json(path / "INTENT.json")
            assert replay.read_json(path / "STATE.json")["status"] == "SUBMISSION_UNKNOWN"
            if self.fail_submit:
                return SimpleNamespace(returncode=1, stdout="", stderr="transport lost")
            job = str(200 + sum(call[0] == "sbatch" for call in self.calls))
            self.jobs[job] = fields(intent, job, state="PENDING")
            self.after_submit()
            return SimpleNamespace(returncode=0, stdout=job + "\n", stderr="")
        if args[:2] == ["scontrol", "release"]:
            if self.fail_release:
                raise TimeoutError("release outcome unknown")
            self.jobs[args[2]].update(JobState="RUNNING", Reason="None")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(args)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root, code, extension = (tmp_path / name for name in ("original", "frozen", "replay-code"))
    for path in (root, code, extension):
        path.mkdir()
    parent = SimpleNamespace(
        directory=root / "recovery/r1",
        extension=tmp_path / "parent-code",
        sha256="parent-plan-hash",
        plan={"python": sys.executable},
    )
    tasks = {
        f"endpoint--M{i}": {"task": {"kind": "endpoint", "model_id": f"M{i}"}, "group_id": "g96"}
        for i in range(5)
    }
    parent.allowed = {key: dict(status="BLOCKED_INCOMPLETE_GENERATION") for key in tasks}
    plan = SimpleNamespace(
        root=root,
        code=code,
        extension=extension,
        parent=parent,
        directory=root / "replay/authorized",
        sha256="replay-plan-hash",
        tasks=tasks,
    )
    plan.path = plan.directory / "PLAN.json"
    manifest = {}
    for relative in ("src/sr_f2_replay/orchestration.py", "scripts/sr_f2/replay_recovery.py"):
        path = extension / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# bound source\n")
        manifest[relative] = replay.file_hash(path)
    replay.save_json(plan.directory / "EXTENSION_MANIFEST.json", dict(files=manifest))
    plan.plan = dict(
        extension_manifest="EXTENSION_MANIFEST.json",
        extension_manifest_sha256=replay.file_hash(plan.directory / "EXTENSION_MANIFEST.json"),
    )
    plan.verify = lambda: True
    monkeypatch.setattr(replay, "__file__", str(extension / "src/sr_f2_replay/orchestration.py"))
    parent_states = {}
    monkeypatch.setattr(
        replay, "ParentQueue", lambda _: SimpleNamespace(states=lambda: parent_states)
    )
    slurm = FakeSlurm()
    for index, spec in enumerate(replay.POOL_SPECS.values(), 1):
        parent_id, count, job = spec["parent_pool_id"], spec["gpu_count"], str(100 + index)
        base = parent.directory / "pools" / parent_id
        base.mkdir(parents=True)
        script = base / "run.sbatch"
        script.write_text("# frozen parent\n")
        intent = dict(
            pool_id=parent_id,
            gpu_count=count,
            plan_sha256=parent.sha256,
            user="alice",
            job_name="parent-" + parent_id,
            comment="parent-" + job,
            script=str(script),
            script_sha256=replay.file_hash(script),
        )
        replay.save_json(base / "INTENT.json", intent)
        replay.save_json(
            base / "STATE.json",
            dict(
                pool_id=parent_id,
                gpu_count=count,
                job_id=job,
                plan_sha256=parent.sha256,
                status="SUBMITTED",
            ),
        )
        replay.save_json(
            base / "STARTED.json",
            dict(job_id=job, plan_sha256=parent.sha256, devices=[str(n) for n in range(count)]),
        )
        replay.save_json(
            base / "COMPLETE.json",
            dict(status="DRAINED", pool_id=parent_id, job_id=job, plan_sha256=parent.sha256),
        )
        task = dict(kind="train", model_id="parent-" + job)
        key = replay.task_id(task)
        row = dict(
            task=task,
            pool_id=parent_id,
            job_id=job,
            device="0",
            status="COMPLETE",
            claim_id="claim-" + job,
            claim_sha256="hash-" + job,
        )
        parent_states[key] = row
        replay.save_json(
            base / "workers" / key / "LAUNCH_INTENT.json",
            {k: row[k] for k in ("task", "claim_id", "device")},
        )
        replay.save_json(base / "workers" / key / "PROCESS_EXIT.json", dict(returncode=0))
        slurm.jobs[job] = fields(intent, job)
        slurm.accounting[job] = dict(
            job_id=job,
            state="COMPLETED",
            exit_code="0:0",
            user="alice",
            account="rose",
            qos=replay.TEACHER_QOS,
            name=intent["job_name"],
        )
    controller = replay.Controller(plan, slurm=slurm, user="alice")
    return plan, controller, parent_states


def drain(controller, index=1):
    pool, job = f"RECOVERY_POOL_{index}", str(100 + index)
    path = controller.plan.parent.directory / "pools" / pool / "STATE.json"
    replay.save_json(path, dict(replay.read_json(path), status="DRAINED"))
    controller.slurm.jobs.pop(job, None)


def finish_claim(queue, pool="REPLAY_POOL_1", device="0", job="201"):
    claim = queue.claim(pool, device, job)
    queue.transition(claim, result=dict(claim["task"], receipt_sha256="verified"))
    return claim


def bind_dispatcher(plan, controller, monkeypatch, *, status="SUBMITTED"):
    base = plan.directory / "dispatcher"
    base.mkdir(parents=True, exist_ok=True)
    script = base / "run.sbatch"
    script.write_text("# cpu admission\n")
    intent = dict(
        plan_sha256=plan.sha256,
        user="alice",
        job_name="replay-cpu",
        comment="cpu-binding",
        script=str(script),
        script_sha256=replay.file_hash(script),
        qos="override-limits-but-killable",
    )
    replay.save_json(base / "INTENT.json", intent)
    replay.save_json(base / "STATE.json", dict(job_id="900", status=status))
    replay.save_json(base / "RELEASE_INTENT.json", dict(job_id="900"))
    controller.slurm.jobs["900"] = dict(
        JobId="900",
        UserId="alice(1)",
        Account="rose",
        QOS=intent["qos"],
        JobName=intent["job_name"],
        Comment=intent["comment"],
        Command=str(script),
        JobState="RUNNING",
        ReqTRES="cpu=4,mem=12G",
        Requeue="0",
    )
    monkeypatch.setenv("SLURM_JOB_ID", "900")
    return base


def test_independent_replacement_does_not_wait_for_slower_parents(experiment):
    plan, controller, _ = experiment
    assert controller.submit_once()["capacity"]["run_gpus"] == 5
    assert not controller.slurm.calls
    drain(controller, 1)
    result = controller.submit_once()
    assert result["capacity"] == dict(teacher_gpus=5, user_submissions=3, run_gpus=5)
    assert {x["pool_id"] for x in result["attempts"]} == {"REPLAY_POOL_1"}
    assert "102" in controller.slurm.jobs and "103" in controller.slurm.jobs
    args = controller.slurm.calls[0]
    assert "--no-requeue" in args and "--gres=gpu:pro6000:2" in args
    script = (controller.pool_dir("REPLAY_POOL_1") / "run.sbatch").read_text()
    assert ":4096:8" in script
    assert f"{plan.extension}/src:{plan.parent.extension}/src:{plan.code}/src" in script


def test_three_drains_use_exact_221_and_do_not_resubmit(experiment):
    _, controller, _ = experiment
    for index in range(1, 4):
        drain(controller, index)
    controller.submit_once()
    controller.submit_once()
    submitted = [call for call in controller.slurm.calls if call[0] == "sbatch"]
    assert len(submitted) == 3
    assert [next(x for x in args if x.startswith("--gres")) for args in submitted] == [
        "--gres=gpu:pro6000:2",
        "--gres=gpu:pro6000:2",
        "--gres=gpu:pro6000:1",
    ]
    for pool in replay.POOL_SPECS:
        base = controller.pool_dir(pool)
        submit = replay.read_json(base / "CAPACITY_BEFORE_SUBMIT.json")
        released = replay.read_json(base / "CAPACITY_BEFORE_RELEASE.json")
        assert submit["capacity"]["run_gpus"] + submit["proposed_gpus"] <= 5
        assert released["capacity"]["run_gpus"] <= 5


@pytest.mark.parametrize("problem", ["completing", "unknown", "failed", "nonzero", "not_drained"])
def test_parent_must_be_absent_drained_and_successful(experiment, problem):
    plan, controller, _ = experiment
    original = controller.slurm.jobs["101"]
    drain(controller)
    if problem == "completing":
        controller.slurm.jobs["101"] = dict(original, JobState="COMPLETING")
    elif problem == "unknown":
        controller.slurm.accounting.pop("101")
    elif problem == "failed":
        controller.slurm.accounting["101"]["state"] = "FAILED"
    elif problem == "nonzero":
        controller.slurm.accounting["101"]["exit_code"] = "1:0"
    else:
        path = plan.parent.directory / "pools/RECOVERY_POOL_1/STATE.json"
        replay.save_json(path, dict(replay.read_json(path), status="SUBMITTED"))
    assert controller.submit_once()["capacity"]["run_gpus"] == 5
    assert not controller.slurm.calls


@pytest.mark.parametrize(
    "problem", ["owner", "command", "gpu", "worker_exit", "orphan_worker", "completion_plan"]
)
def test_parent_identity_and_receipt_guards(experiment, problem):
    plan, controller, _ = experiment
    if problem in {"owner", "command", "gpu"}:
        key, value = {
            "owner": ("UserId", "other(2)"),
            "command": ("Command", "/wrong"),
            "gpu": ("ReqTRES", "cpu=4,mem=90G,gres/gpu=1,gres/gpu:pro6000=1"),
        }[problem]
        controller.slurm.jobs["101"][key] = value
    else:
        drain(controller)
        base = plan.parent.directory / "pools/RECOVERY_POOL_1"
        if problem == "worker_exit":
            path = next((base / "workers").glob("*/PROCESS_EXIT.json"))
            replay.save_json(path, dict(returncode=1))
        elif problem == "orphan_worker":
            (base / "workers/orphan").mkdir()
        else:
            path = base / "COMPLETE.json"
            replay.save_json(path, dict(replay.read_json(path), plan_sha256="wrong"))
    with pytest.raises(PermissionError):
        controller.submit_once()
    assert not controller.slurm.calls


def add_foreign(controller, count, job="700", owner="other", qos=None):
    controller.slurm.jobs[job] = dict(
        UserId=owner + "(2)",
        QOS=qos or replay.TEACHER_QOS,
        ReqTRES=f"cpu=4,mem=90G,gres/gpu={count}",
    )


def test_whole_qos_pending_capacity_and_all_user_job_limit(experiment):
    _, controller, _ = experiment
    drain(controller)
    add_foreign(controller, 4)
    controller.submit_once()
    assert not controller.slurm.calls  # other 4 + live parents 3 + proposed 2 > 8
    controller.slurm.jobs.pop("700")
    for number in range(3):
        add_foreign(controller, 0, job=str(700 + number), owner="alice", qos="other-cpu-qos")
    controller.submit_once()
    assert not controller.slurm.calls  # two parent jobs + three CPU jobs already = five


def test_unknown_submission_reserves_and_never_retries(experiment):
    _, controller, _ = experiment
    drain(controller)
    controller.slurm.fail_submit = True
    result = controller.submit_once()
    assert result["capacity"]["run_gpus"] == 5
    assert controller.submit_once()["status"] == "BLOCKED_UNKNOWN_SUBMISSION"
    assert len(controller.slurm.calls) == 1


def test_unknown_release_reserves_and_never_retries(experiment):
    _, controller, _ = experiment
    drain(controller)
    controller.slurm.fail_release = True
    with pytest.raises(TimeoutError):
        controller.submit_once()
    assert controller.submit_once()["status"] == "BLOCKED_UNKNOWN_SUBMISSION"
    assert controller.context()["capacity"]["run_gpus"] == 5
    assert len(controller.slurm.calls) == 2


def test_capacity_refresh_before_release_leaves_job_held(experiment):
    _, controller, _ = experiment
    drain(controller)
    controller.slurm.after_submit = lambda: add_foreign(controller, 4)
    with pytest.raises(PermissionError, match="remains held"):
        controller.submit_once()
    assert controller.slurm.jobs["201"]["JobState"] == "PENDING"
    assert len(controller.slurm.calls) == 1


def test_capacity_unknown_missing_and_duplicate_id_guards():
    usage = replay.capacity(
        {},
        [
            dict(gpu_count=2),
            dict(gpu_count=2, job_id="1"),
            dict(gpu_count=1, job_id="2", verified_terminal=True),
        ],
        "alice",
    )
    assert usage == dict(run_gpus=4, teacher_gpus=4, user_submissions=2)
    with pytest.raises(PermissionError, match="overlaps"):
        replay.capacity({}, [dict(gpu_count=2, job_id="1")] * 2, "alice")


def test_unique_claims_failed_stop_and_no_retry(experiment):
    plan, _, _ = experiment
    queue = replay.ReplayQueue(plan)
    claim = queue.claim("REPLAY_POOL_1", "0", "201")
    with pytest.raises(PermissionError, match="unfinished"):
        queue.claim("REPLAY_POOL_1", "0", "201")
    queue.transition(claim, error="preserved row mismatch")
    assert queue.snapshot()["failed"]
    assert (plan.directory / "STOP.json").exists()
    assert not (plan.parent.directory / "STOP.json").exists()
    with pytest.raises(RuntimeError, match="stopped"):
        queue.claim("REPLAY_POOL_1", "1", "201")
    with pytest.raises(PermissionError, match="retried"):
        queue.transition(claim, result=claim["task"])


def test_orphan_claim_and_unregistered_endpoint_fail_closed(experiment):
    plan, _, _ = experiment
    queue = replay.ReplayQueue(plan)
    claim = queue.claim("REPLAY_POOL_1", "0", "201")
    (queue.directory / "tasks" / replay.task_id(claim["task"]) / "STATE.json").unlink()
    with pytest.raises(RuntimeError, match="Orphan"):
        queue.snapshot()
    plan.tasks[next(iter(plan.tasks))]["task"]["kind"] = "train"
    with pytest.raises(PermissionError):
        replay.ReplayQueue(plan)


def test_no_empty_replacements_after_all_five_tasks_claimed(experiment):
    plan, controller, _ = experiment
    drain(controller)
    controller.submit_once()
    queue = replay.ReplayQueue(plan)
    for _ in range(5):
        finish_claim(queue)
    result = controller.submit_once()
    assert result["task_claims_sha256"]
    for pool in ("REPLAY_POOL_2", "REPLAY_POOL_3"):
        base = controller.pool_dir(pool)
        skipped = replay.read_json(base / "NOT_NEEDED.json")
        assert skipped["task_claims_sha256"] == result["task_claims_sha256"]
        assert not (base / "INTENT.json").exists() and not (base / "STATE.json").exists()
    drain(controller, 2)
    drain(controller, 3)
    controller.submit_once()
    assert sum(call[0] == "sbatch" for call in controller.slurm.calls) == 1


def test_skip_proof_cannot_hide_unsubmitted_pool_claim(experiment):
    plan, controller, _ = experiment
    queue = replay.ReplayQueue(plan)
    for _ in range(5):
        finish_claim(queue)
    with pytest.raises(PermissionError, match="Unsubmitted pool"):
        controller.mark_not_needed(queue)


def test_parent_tree_unchanged_by_replacement_admission(experiment):
    plan, controller, _ = experiment
    drain(controller)
    before = {
        str(path): replay.file_hash(path)
        for path in plan.parent.directory.rglob("*")
        if path.is_file()
    }
    controller.submit_once()
    after = {
        str(path): replay.file_hash(path)
        for path in plan.parent.directory.rglob("*")
        if path.is_file()
    }
    assert before == after


@pytest.mark.parametrize("problem", ["missing_manifest", "changed_source", "wrong_import"])
def test_controller_source_binding(experiment, monkeypatch, problem):
    plan, controller, _ = experiment
    drain(controller)
    if problem == "missing_manifest":
        path = plan.directory / plan.plan["extension_manifest"]
        manifest = replay.read_json(path)
        manifest["files"].pop("src/sr_f2_replay/orchestration.py")
        replay.save_json(path, manifest)
        plan.plan["extension_manifest_sha256"] = replay.file_hash(path)
    elif problem == "changed_source":
        (plan.extension / "scripts/sr_f2/replay_recovery.py").write_text("changed")
    else:
        monkeypatch.setattr(replay, "__file__", "/tmp/unbound.py")
    with pytest.raises(PermissionError):
        controller.submit_once()
    assert not controller.slurm.calls


class FinishedProcess:
    def __init__(self, result=0):
        self.pid = 12345
        self.result = result

    def poll(self):
        return self.result

    def wait(self, timeout):
        return self.result


def test_pool_single_gpu_refills_and_verifies_all_five(experiment, monkeypatch):
    plan, controller, _ = experiment
    drain(controller)
    controller.submit_once()
    monkeypatch.setenv("SLURM_JOB_ID", "201")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    calls, verified = [], []

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return FinishedProcess()

    def verify(p, task):
        verified.append(task)
        return dict(task, receipt_sha256="verified", replay_provenance_sha256="bound")

    monkeypatch.setattr(replay, "verified_task", verify)
    result = replay.run_pool(
        plan, "REPLAY_POOL_1", slurm=controller.slurm, poll_seconds=0, popen=launch
    )
    assert result["status"] == "DRAINED" and result["all_tasks_complete"]
    assert len(calls) == len(verified) == 5
    assert {kwargs["env"]["CUDA_VISIBLE_DEVICES"] for _, kwargs in calls} == {"3", "7"}
    assert all(
        "--task-id" in command and "replay_endpoint.py" in command[1] for command, _ in calls
    )
    assert (controller.pool_dir("REPLAY_POOL_1") / "CAPACITY_AT_START.json").exists()
    assert replay.ReplayQueue(plan).snapshot()["completed"] == 5


def test_pool_failure_stops_new_queue_without_parent_mutation(experiment, monkeypatch):
    plan, controller, _ = experiment
    drain(controller)
    controller.submit_once()
    monkeypatch.setenv("SLURM_JOB_ID", "201")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="failed"):
        replay.run_pool(
            plan,
            "REPLAY_POOL_1",
            slurm=controller.slurm,
            poll_seconds=0,
            popen=lambda *a, **k: FinishedProcess(9),
        )
    assert (plan.directory / "STOP.json").exists()
    assert not (plan.parent.directory / "STOP.json").exists()
    states = replay.ReplayQueue(plan).states()
    assert all(row["status"] == "FAILED" for row in states.values())


def test_dispatcher_polls_replaces_and_exits_after_claims(experiment, monkeypatch):
    plan, controller, _ = experiment
    base = bind_dispatcher(plan, controller, monkeypatch, status="RELEASE_UNKNOWN")
    sleeps = []

    def tick(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            drain(controller)
        else:
            for _ in range(5):
                finish_claim(replay.ReplayQueue(plan))

    result = replay.run_dispatcher(plan, slurm=controller.slurm, sleep=tick)
    assert sleeps == [15, 15]
    assert result["reason"] == "ALL_ENDPOINTS_CLAIMED"
    assert replay.read_json(base / "STATE.json")["status"] == "COMPLETE"
    assert (controller.pool_dir("REPLAY_POOL_2") / "NOT_NEEDED.json").exists()
    assert replay.read_json(base / "STATUS.json")["capacity"]["user_submissions"] == 4
    with pytest.raises(PermissionError):
        replay.run_dispatcher(plan, slurm=controller.slurm, sleep=tick)


@pytest.mark.parametrize(
    "field,value",
    [
        ("UserId", "other(2)"),
        ("Requeue", "1"),
        ("ReqTRES", "cpu=4,gres/gpu=1"),
        ("Command", "/wrong"),
        ("JobState", "PENDING"),
    ],
)
def test_dispatcher_real_cpu_identity(experiment, monkeypatch, field, value):
    plan, controller, _ = experiment
    base = bind_dispatcher(plan, controller, monkeypatch)
    controller.slurm.jobs["900"][field] = value
    with pytest.raises(PermissionError):
        replay.run_dispatcher(plan, slurm=controller.slurm)
    assert not (base / "STARTED.json").exists()


def test_dispatcher_unknown_admission_exits_without_cancelling_workers(experiment, monkeypatch):
    plan, controller, _ = experiment
    base = bind_dispatcher(plan, controller, monkeypatch)
    drain(controller)
    controller.slurm.fail_submit = True
    with pytest.raises(RuntimeError, match="Unknown replacement"):
        replay.run_dispatcher(plan, slurm=controller.slurm, sleep=lambda _: None)
    assert replay.read_json(base / "STATE.json")["status"] == "FAILED"
    assert not (plan.directory / "STOP.json").exists()
    assert all(call[0] != "scancel" for call in controller.slurm.calls)


def test_claim_and_state_remain_small(experiment):
    plan, _, _ = experiment
    plan.tasks[next(iter(plan.tasks))]["baseline_files"] = [
        dict(path=str(i), sha256="a" * 64) for i in range(1000)
    ]
    claim = replay.ReplayQueue(plan).claim("REPLAY_POOL_1", "0", "201")
    assert len(json.dumps(claim)) < 1500
