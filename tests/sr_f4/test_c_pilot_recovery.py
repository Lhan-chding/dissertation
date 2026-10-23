import copy
from types import SimpleNamespace

import pytest

from sr_f4 import c_pilot, pool
from sr_f4 import c_pilot_recovery as recovery
from sr_f4.common import read, write


@pytest.fixture
def timed_out(tmp_path, monkeypatch):
    matrix = c_pilot.task_matrix()
    write(tmp_path / "TASKS.json", matrix)
    write(tmp_path / "FROZEN.json", dict(code_revision="baseline", seed=123))
    monkeypatch.setattr(
        recovery,
        "require_frozen",
        lambda root: read(root / "TASKS.json") == matrix or pytest.fail("Frozen tasks changed"),
    )
    pool.init_pool(tmp_path, matrix)
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    task, lease = pool.claim(tmp_path, "99:0")
    record = dict(
        job_id="99",
        state="TIMEOUT",
        exit_code="0:0",
        user="testuser",
        qos=pool.QOS,
        account=pool.ACCOUNT,
    )
    slurm = SimpleNamespace(queue=lambda: {}, terminal=lambda job: record)
    with pool._locked(tmp_path) as state:
        state["jobs"]["job-0001"] = dict(job_id="99", user="testuser", status="TERMINAL")
        pool._save(tmp_path, state)
    lease.close()
    assert pool.reconcile_timeouts(tmp_path, slurm=slurm)[0]["status"] == "FAILED"
    directory = tmp_path / "RUNS/C/b0/J"
    write(directory / "PROGRESS.json", dict(successful_updates=10))
    checkpoint = directory / "checkpoints/latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"retained optimizer and rng checkpoint")
    write(directory / "rollouts/011/00000.json", dict(rows=["existing response"]))
    return tmp_path, task, slurm


def test_repair_preserves_freeze_attempts_checkpoint_and_responses(timed_out):
    root, task, slurm = timed_out
    originals = {p: p.read_bytes() for p in (root / "RUNS").rglob("*") if p.is_file()}
    originals.update(
        {root / name: (root / name).read_bytes() for name in ("FROZEN.json", "TASKS.json")}
    )
    prior = copy.deepcopy(read(root / "pool/STATE.json")["tasks"][task["task_id"]]["attempts"])
    assert len(recovery.enable_resumption(root, "fix")) == 72
    assert recovery.enable_resumption(root, "fix") == []
    assert recovery.recover_misclassified_timeouts(root, slurm=slurm) == [task["task_id"]]
    row = read(root / "pool/STATE.json")["tasks"][task["task_id"]]
    assert row["status"] == "PENDING" and row["attempts"] == prior
    assert row["recoveries"][0]["failed_attempt"] == 1
    assert all(p.read_bytes() == original for p, original in originals.items())
    assert recovery.recover_misclassified_timeouts(root, slurm=slurm) == []
    # The existing, unmodified controller handles subsequent TIMEOUT boundaries.
    new_task, lease = pool.claim(root, "99:0")
    assert new_task["task_id"] == task["task_id"] and new_task["attempt"] == 2
    lease.close()
    assert pool.reconcile_timeouts(root, slurm=slurm)[0]["status"] == "PENDING"


@pytest.mark.parametrize("guard", ["queued", "not_timeout", "lease", "other_failure"])
def test_recovery_cannot_retry_live_leases_or_other_failures(timed_out, guard):
    root, task, slurm = timed_out
    recovery.enable_resumption(root, "fix")
    lease = None
    if guard == "queued":
        slurm.queue = lambda: {"99": {}}
    elif guard == "not_timeout":
        slurm.terminal = lambda job: dict(state="OUT_OF_MEMORY")
    elif guard == "lease":
        lease = pool._task_lease(root, task["task_id"])
    else:
        with pool._locked(root) as state:
            state["tasks"][task["task_id"]]["reason"] = "child_exit=1"
            pool._save(root, state)
    try:
        assert recovery.recover_misclassified_timeouts(root, slurm=slurm) == []
    finally:
        if lease:
            lease.close()


def test_recovery_rejects_foreign_job_and_missing_committed_checkpoint(timed_out):
    root, _task, slurm = timed_out
    recovery.enable_resumption(root, "fix")
    record = slurm.terminal("99")
    slurm.terminal = lambda job: dict(record, user="other")
    with pytest.raises(PermissionError, match="allocation"):
        recovery.recover_misclassified_timeouts(root, slurm=slurm)
    slurm.terminal = lambda job: record
    (root / "RUNS/C/b0/J/checkpoints/latest.pt").unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        recovery.recover_misclassified_timeouts(root, slurm=slurm)


def test_amendment_rejects_scientific_spec_change(timed_out):
    root, task, _ = timed_out
    with pool._locked(root) as state:
        state["tasks"][task["task_id"]]["spec"]["arm"] = "SELECT"
        pool._save(root, state)
    with pytest.raises(ValueError, match="matrix"):
        recovery.enable_resumption(root, "fix")
