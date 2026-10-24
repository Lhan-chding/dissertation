import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f4 import c_local, pool
from sr_f4 import c_local_recovery as recovery
from sr_f4.common import read, write


@pytest.fixture
def incident(tmp_path, monkeypatch):
    matrix = c_local.task_matrix()
    pool.init_pool(tmp_path, matrix)
    monkeypatch.setattr(recovery, "require_frozen", lambda root: None)
    monkeypatch.setattr(recovery, "check_imports", lambda *args: "model_calls=0")
    with pool._locked(tmp_path) as state:
        state["jobs"]["job-0002"] = dict(
            job_id="223622", user="varun024", status="FAILED", code="/code"
        )
        for key, row in state["tasks"].items():
            if row["spec"]["dependencies"]:
                continue
            row.update(status="FAILED", reason="child_exit=1; complete_receipt=False")
            row["attempts"] = [
                dict(
                    status="FAILED",
                    returncode=1,
                    complete_receipt=False,
                    owner=dict(job_id="223622"),
                )
            ]
            path = tmp_path / "pool/logs" / f"{key}.attempt-001.log"
            path.parent.mkdir(exist_ok=True)
            path.write_text(
                "from sr_f3_ce.training import fresh_optimizer\n"
                'reference_module("semantic_contract")\n'
                "FileNotFoundError: [Errno 2] No such file or directory: "
                f"'/code/{recovery.MISSING}'\n"
            )
        pool._save(tmp_path, state)
    terminal = dict(
        job_id="223622",
        state="FAILED",
        exit_code="1:0",
        user="varun024",
        qos=pool.QOS,
        account=pool.ACCOUNT,
    )
    slurm = SimpleNamespace(
        queue=lambda: {}, user_job_ids=lambda user: set(), terminal=lambda job: terminal
    )
    return tmp_path, slurm


def test_recovery_is_narrow_preserves_attempts_and_is_idempotent(incident):
    root, slurm = incident
    before = read(root / "pool/STATE.json")
    receipt = recovery.recover(root, "223622", "fixed", slurm=slurm)
    assert receipt["status"] == "APPLIED" and len(receipt["tasks"]) == 24
    after = read(root / "pool/STATE.json")
    for key, old in before["tasks"].items():
        assert after["tasks"][key]["attempts"] == old["attempts"]
        if old["status"] == "PENDING":
            assert after["tasks"][key] == old
        else:
            assert after["tasks"][key]["status"] == "PENDING"
    assert recovery.recover(root, "223622", "fixed", slurm=slurm) == receipt
    assert read(root / "pool/STATE.json") == after


@pytest.mark.parametrize(
    "case", ["queue", "foreign", "different_error", "output", "lease", "new_attempt", "reservation"]
)
def test_refuses_unverified_or_noninitial_failures(incident, case):
    root, slurm = incident
    key = "CLC-T-b0-J"
    lease = None
    if case == "queue":
        slurm.user_job_ids = lambda user: {"223622"}
    elif case == "foreign":
        terminal = slurm.terminal("223622")
        slurm.terminal = lambda job: dict(terminal, user="other")
    elif case == "different_error":
        (root / "pool/logs" / f"{key}.attempt-001.log").write_text("CUDA out of memory")
    elif case == "output":
        write(root / "accounting" / "answers.json", dict(count=1))
    elif case == "lease":
        lease = pool._task_lease(root, key)
    else:
        with pool._locked(root) as state:
            if case == "new_attempt":
                state["tasks"][key]["attempts"].append(
                    copy.deepcopy(state["tasks"][key]["attempts"][0])
                )
            else:
                state["jobs"]["unknown"] = dict(status="SUBMISSION_UNKNOWN")
            pool._save(root, state)
    before = read(root / "pool/STATE.json")
    try:
        with pytest.raises((RuntimeError, PermissionError, ValueError)):
            recovery.recover(root, "223622", "fixed", slurm=slurm)
    finally:
        if lease:
            lease.close()
    assert read(root / "pool/STATE.json") == before


def test_controller_checks_deployed_imports_before_gpu_submission():
    script = Path(__file__).resolve().parents[2] / "scripts/sr_f4/c_local_controller.sbatch"
    text = script.read_text()
    assert text.index('check_imports(".", sys.executable)') < text.index("if [ ! -f")
    assert "--gpus-per-job 2" in text
