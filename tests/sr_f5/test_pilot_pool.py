import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f4.common import read, write
from sr_f5 import pilot_pool as pool

SCRIPT = Path(__file__).parents[2] / "scripts/sr_f5/submit_g3.py"
SPEC = importlib.util.spec_from_file_location("srf5_submit_g3_test", SCRIPT)
dispatcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatcher)


def registered_inputs():
    config = dict(
        master_seed=2026102501,
        parents=["b0", "b1", "b2"],
        arms=["J", "HARD", "DENSE", "LOCAL", "SHUF"],
    )
    frozen = dict(
        code_revision="abcdef123",
        paths=[dict(parent=p, arm=a) for p in config["parents"] for a in config["arms"]],
    )
    return config, frozen


def test_plan_is_deterministic_balanced_and_keeps_all_registered_paths():
    config, frozen = registered_inputs()
    plan = pool.build_plan(config, frozen, "abcdef123")
    assert pool.build_plan(config, frozen, "abcdef123") == plan
    tasks = [task for worker in plan["workers"] for task in worker["tasks"]]
    assert {(t["parent"], t["arm"]) for t in tasks} == {
        (p, a) for p in config["parents"] for a in config["arms"]
    }
    assert len(tasks) == 15 and sum(t["recovery"] for t in tasks) == 5
    assert all(t["recovery"] == (t["parent"] == "b0") for t in tasks)
    assert max(w["weighted_steps"] for w in plan["workers"]) == 56
    assert min(w["weighted_steps"] for w in plan["workers"]) == 52
    assert len({row["seed"] for row in plan["arm_orders"].values()}) == 3
    assert config["arms"] == ["J", "HARD", "DENSE", "LOCAL", "SHUF"]


def test_pool_revision_must_match_frozen_science():
    config, frozen = registered_inputs()
    with pytest.raises(PermissionError, match="revision"):
        pool.build_plan(config, frozen, "different")


def test_pool_cannot_bypass_g2_or_freeze_after_outputs(tmp_path, monkeypatch):
    config, frozen = registered_inputs()
    write(tmp_path / "CONFIG.json", config)

    def reject(_root):
        raise PermissionError("actual G2 failed")

    monkeypatch.setattr(pool, "require_frozen", reject)
    with pytest.raises(PermissionError, match="G2 failed"):
        pool.freeze_pool(tmp_path, "abcdef123")
    assert not (tmp_path / "G3/POOL_PLAN.json").exists()
    monkeypatch.setattr(pool, "require_frozen", lambda _: frozen)
    (tmp_path / "G3/RUNS/b0/J/primary/rollouts/001").mkdir(parents=True)
    with pytest.raises(PermissionError, match="after pilot outputs"):
        pool.freeze_pool(tmp_path, "abcdef123")


def test_frozen_assignment_cannot_be_changed(tmp_path, monkeypatch):
    config, frozen = registered_inputs()
    write(tmp_path / "CONFIG.json", config)
    monkeypatch.setattr(pool, "require_frozen", lambda _: frozen)
    plan = pool.freeze_pool(tmp_path, "abcdef123")
    assert pool.require_plan(tmp_path, "abcdef123") == plan
    plan["workers"][0]["tasks"].reverse()
    write(tmp_path / "G3/POOL_PLAN.json", plan)
    with pytest.raises(PermissionError, match="Registered identity differs"):
        pool.require_plan(tmp_path, "abcdef123")


@pytest.fixture
def worker(tmp_path, monkeypatch):
    tasks = [
        dict(parent="b0", arm="LOCAL", recovery=True),
        dict(parent="b1", arm="J", recovery=False),
    ]
    plan = dict(workers=[dict(worker=0, tasks=tasks)])
    monkeypatch.setattr(pool, "require_plan", lambda *_: plan)
    monkeypatch.setattr(pool, "verify_code", lambda code: Path(code))
    monkeypatch.setenv("SRF5_SHARD", "0")
    monkeypatch.setenv("SRF5_SHARDS", "4")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    calls = []
    comparison = {"status": "PASS"}

    def run(command, **kwargs):
        calls.append((command, kwargs))
        parent, arm = (command[command.index(key) + 1] for key in ("--parent", "--arm"))
        directory = tmp_path / "G3/RUNS" / parent / arm
        if "--compare-recovery" in command:
            write(directory / "RECOVERY.json", comparison)
        else:
            lane = command[command.index("--lane") + 1]
            write(
                directory / lane / "COMPLETE.json",
                dict(
                    status="COMPLETE",
                    committed_step=12,
                    lane=lane,
                    identity=dict(parent=parent, arm=arm),
                ),
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pool.subprocess, "run", run)
    return SimpleNamespace(root=tmp_path, calls=calls, comparison=comparison)


def test_worker_runs_fresh_primary_recovery_comparison_then_next_path(worker):
    w = worker
    receipt = pool.run_worker(w.root, w.root, "abcdef123", Path(sys.executable))
    assert receipt["status"] == "COMPLETE"
    assert [row["phase"] for row in receipt["calls"]] == [
        "primary",
        "recovery",
        "compare_recovery",
        "primary",
    ]
    assert len(w.calls) == 4
    assert all(c[0][2:4] == ["-m", "sr_f5.pilot"] for c in w.calls)
    assert all(c[1]["env"]["CUDA_VISIBLE_DEVICES"] == "7" for c in w.calls)
    assert receipt["scientific_gate"] == "NOT_YET_AGGREGATED"


def test_recovery_failure_stops_worker_even_when_compare_process_exits_zero(worker):
    w = worker
    w.comparison["status"] = "FAIL"
    with pytest.raises(RuntimeError, match="Recovery comparison"):
        pool.run_worker(w.root, w.root, "abcdef123", Path(sys.executable))
    assert len(w.calls) == 3
    assert read(w.root / "G3/workers/W0-123.json")["status"] == "FAILED"


@pytest.fixture
def submission(tmp_path, monkeypatch):
    root, code = tmp_path / "run", tmp_path / "code"
    root.mkdir()
    (code / "scripts/sr_f5").mkdir(parents=True)
    (code / "scripts/sr_f5/pilot.sbatch").write_text("# fixture")
    monkeypatch.setattr(dispatcher, "PROJECT", tmp_path)
    monkeypatch.setattr(dispatcher, "verify_code", lambda _: code)
    monkeypatch.setattr(dispatcher, "freeze_pool", lambda *_: dict(status="frozen"))
    calls = []

    class Slurm:
        def __init__(self):
            self.queue_rows = {}

        def user_job_ids(self, user):
            return set()

        def queue(self):
            return dict(self.queue_rows)

        def show(self, job_id):
            return dict(
                JobName=f"LH-SRF5-G3-P{int(job_id) - 200}",
                QOS=dispatcher.QOS,
                Account=dispatcher.ACCOUNT,
                ReqTRES="gres/gpu=2",
                WorkDir=str(code),
            )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"{199 + len(calls)}\n", stderr="")

    monkeypatch.setattr(dispatcher.subprocess, "run", run)
    return SimpleNamespace(
        args=SimpleNamespace(root=root, code=code, revision="abcdef123"),
        calls=calls,
        slurm=Slurm(),
        root=root,
    )


def test_submit_two_pairs_and_refuse_duplicate(submission):
    s = submission
    receipt = dispatcher.submit(s.args, slurm=s.slurm, user="me")
    assert [r["job_id"] for r in receipt["jobs"]] == ["200", "201"]
    assert [r[1]["env"]["SRF5_PAIR"] for r in s.calls] == ["0", "1"]
    assert all(r[1]["env"]["SRF5_REVISION"] == "abcdef123" for r in s.calls)
    with pytest.raises(PermissionError, match="no duplicate"):
        dispatcher.submit(s.args, slurm=s.slurm, user="me")


def test_submission_requires_real_frozen_gate(submission, monkeypatch):
    s = submission

    def blocked(*args):
        raise PermissionError("G2 is not PASS")

    monkeypatch.setattr(dispatcher, "freeze_pool", blocked)
    with pytest.raises(PermissionError, match="not PASS"):
        dispatcher.submit(s.args, slurm=s.slurm, user="me")
    assert not s.calls
    assert not (s.root / "G3_SUBMISSION_DUAL_GPU.json").exists()


def test_unknown_submission_receipt_blocks_retry(submission, monkeypatch):
    s = submission
    monkeypatch.setattr(
        dispatcher.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="?", stderr="")
    )
    with pytest.raises(RuntimeError, match="Ambiguous"):
        dispatcher.submit(s.args, slurm=s.slurm, user="me")
    receipt = read(s.root / "G3_SUBMISSION_DUAL_GPU.json")
    assert receipt["jobs"][0]["status"] == "SUBMISSION_UNKNOWN"
    with pytest.raises(PermissionError, match="no duplicate"):
        dispatcher.submit(s.args, slurm=s.slurm, user="me")


def test_insufficient_capacity_does_not_submit(submission):
    s = submission
    s.slurm.queue_rows = {"other": dict(ReqTRES="gres/gpu=4")}
    with pytest.raises(RuntimeError, match="Capacity unavailable"):
        dispatcher.submit(s.args, slurm=s.slurm, user="me")
    assert not s.calls
