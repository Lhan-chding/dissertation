import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from sr_f4 import pool


def task(key, deps=(), **extra):
    return dict(task_id=key, kind="probe", block="b0", dependencies=list(deps), **extra)


def receipt(root, key):
    path = root / "tasks" / key / "COMPLETE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "COMPLETE", "task_id": key}))


def complete(root, claimed, lease):
    receipt(root, claimed["task_id"])
    assert pool.finish(root, claimed, 0)
    lease.close()


def test_immediate_dependency_refill_without_batch_barrier(tmp_path):
    pool.init_pool(tmp_path, [task("B0"), task("long_probe"), task("A", ["B0"])])
    first, first_lease = pool.claim(tmp_path, "0")
    other, other_lease = pool.claim(tmp_path, "1")
    assert (first["task_id"], other["task_id"]) == ("B0", "long_probe")
    complete(tmp_path, first, first_lease)
    following, lease = pool.claim(tmp_path, "0")
    assert following["task_id"] == "A"
    assert pool.status(tmp_path)["tasks"]["long_probe"]["status"] == "RUNNING"
    lease.close()
    other_lease.close()


def test_competing_workers_claim_each_task_once(tmp_path):
    pool.init_pool(tmp_path, [task(str(index)) for index in range(8)])
    with ThreadPoolExecutor(max_workers=12) as executor:
        claims = list(executor.map(lambda index: pool.claim(tmp_path, str(index)), range(12)))
    keys = [claimed["task_id"] for claimed, lease in claims if claimed]
    assert len(keys) == len(set(keys)) == 8
    for _claimed, lease in claims:
        if lease:
            lease.close()


@pytest.mark.parametrize("returncode,has_receipt", [(0, False), (1, True)])
def test_completion_requires_exit_and_receipt(tmp_path, returncode, has_receipt):
    pool.init_pool(tmp_path, [task("a"), task("dependent", ["a"])])
    claimed, lease = pool.claim(tmp_path, "0")
    if has_receipt:
        receipt(tmp_path, "a")
    assert not pool.finish(tmp_path, claimed, returncode)
    lease.close()
    summary = pool.status(tmp_path)
    assert summary["tasks"]["a"]["status"] == "FAILED"
    assert summary["tasks"]["dependent"]["status"] == "BLOCKED_DEPENDENCY"
    assert summary["work_remaining"] is False


def test_init_never_resets_existing_progress(tmp_path):
    matrix = [task("a"), task("C", status="BLOCKED", blocked_reason="HUMAN_REVIEW_MISSING")]
    pool.init_pool(tmp_path, matrix)
    claimed, lease = pool.claim(tmp_path, "0")
    complete(tmp_path, claimed, lease)
    assert pool.init_pool(tmp_path, matrix)["tasks"]["a"]["status"] == "COMPLETE"
    with pytest.raises(ValueError, match="differs"):
        pool.init_pool(tmp_path, [task("new")])
    assert not pool.status(tmp_path)["work_remaining"]
    assert not pool.status(tmp_path)["all_complete"]


@pytest.mark.parametrize(
    "matrix",
    [[task("a", ["a"])], [task("a", ["missing"])], [task("a"), task("a")], [task("../escape")]],
)
def test_bad_matrix_rejected(tmp_path, matrix):
    with pytest.raises(ValueError):
        pool.init_pool(tmp_path, matrix)


def row(user="louis", gpus=2, allocated=0):
    return {
        "QOS": pool.QOS,
        "UserId": user + "(1)",
        "ReqTRES": f"cpu=8,gres/gpu={gpus},gres/gpu:pro6000={gpus}",
        "AllocTRES": f"cpu=8,gres/gpu={allocated},gres/gpu:pro6000={allocated}",
    }


def test_shared_qos_counts_requests_allocations_and_cpu_jobs():
    values = {"1": row("other", 2), "2": row("louis", 1, 2), "3": row("louis", 0)}
    assert pool.usage(values, "louis") == {"gpus": 4, "submitted": 2}
    assert pool.gpu_count("gres/gpu:pro6000=2,gres/gpu:l40=1") == 3
    with pytest.raises(ValueError):
        pool.gpu_count("gres/gpu=unknown")


class FakeSlurm:
    def __init__(self, queue=None, *, visible=True, terminal=None):
        self.rows = dict(queue or {})
        self.visible = visible
        self.terminal_record = terminal
        self.commands = []

    def queue(self):
        return dict(self.rows)

    def terminal(self, job_id):
        return self.terminal_record

    def run(self, command):
        self.commands.append(command)
        job_id = str(100 + len(self.commands))
        count = int(next(arg for arg in command if arg.startswith("--gres=")).rsplit(":", 1)[1])
        if self.visible:
            self.rows[job_id] = row(gpus=count)
        return SimpleNamespace(returncode=0, stdout=job_id + "\n", stderr="")


@pytest.mark.parametrize("visible", [True, False])
def test_submission_uses_eight_gpus_even_before_queue_visibility(tmp_path, visible):
    pool.init_pool(tmp_path, [task(str(index)) for index in range(10)])
    slurm = FakeSlurm(visible=visible)
    result = pool.submit(tmp_path, tmp_path, "/usr/bin/python3", user="louis", slurm=slurm)
    assert len(result["submitted"]) == 4
    assert sum(item["gpus"] for item in result["submitted"]) == 8
    assert all(item["name"].startswith("LH-") for item in result["submitted"])
    assert all(not any("constraint" in arg for arg in command) for command in slurm.commands)
    again = pool.submit(tmp_path, tmp_path, "/usr/bin/python3", user="louis", slurm=slurm)
    assert again["submitted"] == []


def test_submit_respects_cpu_job_cap_and_other_users_gpu_usage(tmp_path):
    pool.init_pool(tmp_path, [task(str(index)) for index in range(10)])
    slurm = FakeSlurm(
        {"1": row("other", 3), **{str(index): row("louis", 0) for index in range(2, 6)}}
    )
    result = pool.submit(tmp_path, tmp_path, "/usr/bin/python3", user="louis", slurm=slurm)
    assert len(result["submitted"]) == 1
    assert result["submitted"][0]["gpus"] == 2


@pytest.mark.parametrize(
    "state,resumable,expected",
    [("TIMEOUT", True, "PENDING"), ("TIMEOUT", False, "FAILED"), ("OUT_OF_MEMORY", True, "FAILED")],
)
def test_timeout_resume_only_after_lease_release(tmp_path, monkeypatch, state, resumable, expected):
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    pool.init_pool(tmp_path, [task("a", resume_supported=resumable)])
    _claimed, lease = pool.claim(tmp_path, "0")
    slurm = FakeSlurm(
        terminal={
            "job_id": "99",
            "state": state,
            "exit_code": "0:0",
            "qos": pool.QOS,
            "account": pool.ACCOUNT,
        }
    )
    assert pool.reconcile_timeouts(tmp_path, slurm=slurm) == []
    lease.close()
    assert pool.reconcile_timeouts(tmp_path, slurm=slurm)[0]["status"] == expected
    assert len(pool._read(tmp_path / "pool/STATE.json")["tasks"]["a"]["attempts"]) == 1


def test_worker_continues_after_independent_task_failure(tmp_path, monkeypatch):
    pool.init_pool(tmp_path, [task("fails"), task("works")])

    def run(command, **kwargs):
        key = command[-1]
        if key == "works":
            receipt(tmp_path, key)
        return SimpleNamespace(returncode=int(key == "fails"))

    monkeypatch.setattr(pool.subprocess, "run", run)
    assert pool.worker(tmp_path, "0", poll_seconds=0) == 0
    summary = pool.status(tmp_path)
    assert summary["tasks"]["fails"]["status"] == "FAILED"
    assert summary["tasks"]["works"]["status"] == "COMPLETE"
