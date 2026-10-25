import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f5.dual_gpu import run_worker, supervise, telemetry_command, worker_environments

DISPATCH = runpy.run_path(str(Path(__file__).parents[2] / "scripts/sr_f5/submit_g2.py"))


def test_batch_worker_inherits_required_backend_environment(tmp_path):
    script = Path(__file__).parents[2] / "scripts/sr_f5/calibration.sbatch"
    # Execute the real batch exports, stopping before its real-model exec.
    prefix = script.read_text().split("exec /projects/", 1)[0]
    result = subprocess.run(
        [
            "bash",
            "-c",
            prefix + "\npython3 -c 'import os; print(os.environ[\"CUBLAS_WORKSPACE_CONFIG\"])'",
        ],
        env=dict(
            os.environ, SRF5_CODE=str(tmp_path), SRF5_LAUNCHER_CODE=str(tmp_path), SRF5_PAIR="0"
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ":4096:8"
    assert telemetry_command(["3", "7"])[1] == "--id=3,7"


def test_pairs_preserve_four_shards_and_one_device_each():
    source = {"CUDA_VISIBLE_DEVICES": "2,5", "KEEP": "identity"}
    results = [worker_environments(pair, "2,5", source) for pair in (0, 1)]
    assert [[e["SRF5_SHARD"] for e in pair] for pair in results] == [["0", "1"], ["2", "3"]]
    assert all([e["CUDA_VISIBLE_DEVICES"] for e in pair] == ["2", "5"] for pair in results)
    assert all(
        e["KEEP"] == "identity" and e["SRF5_SHARDS"] == "4" for pair in results for e in pair
    )
    assert source == {"CUDA_VISIBLE_DEVICES": "2,5", "KEEP": "identity"}


@pytest.mark.parametrize("devices", ["", "0", "0,0", "0,1,2", "0,", "0, 1", "all"])
def test_invalid_gpu_visibility_fails(devices):
    with pytest.raises(ValueError):
        worker_environments(0, devices, {})


def test_real_child_environments_and_successful_wait(tmp_path):
    environments = worker_environments(1, "GPU-first,GPU-second", os.environ)
    commands = []
    for i in range(2):
        output = tmp_path / f"result{i}.json"
        script = (
            "import json,os,pathlib; "
            f"pathlib.Path({str(output)!r}).write_text(json.dumps(dict("
            "gpu=os.environ['CUDA_VISIBLE_DEVICES'],shard=os.environ['SRF5_SHARD'])))"
        )
        commands.append([sys.executable, "-c", script])
    assert supervise(commands, environments, [tmp_path / "a.log", tmp_path / "b.log"]) == [0, 0]
    assert json.loads((tmp_path / "result0.json").read_text()) == {"gpu": "GPU-first", "shard": "2"}
    assert json.loads((tmp_path / "result1.json").read_text()) == {
        "gpu": "GPU-second",
        "shard": "3",
    }


def test_failed_worker_reaps_other_process_group(tmp_path):
    pidfile = tmp_path / "child.pid"
    sleeping = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    failing = "import time,sys; time.sleep(0.3); sys.exit(9)"
    with pytest.raises(RuntimeError, match="worker failed"):
        supervise(
            [[sys.executable, "-c", sleeping], [sys.executable, "-c", failing]],
            [dict(os.environ), dict(os.environ)],
            [tmp_path / "a.log", tmp_path / "b.log"],
            grace_seconds=0.2,
        )
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_worker_uses_frozen_science_and_stops_failed_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("SRF5_SHARD", "3")
    monkeypatch.setenv("SRF5_SHARDS", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0 if len(calls) == 1 else 17)

    monkeypatch.setattr(subprocess, "run", run)
    assert run_worker(tmp_path, tmp_path / "run", Path(sys.executable)) == 17
    assert [c[c.index("--parent") + 1] for c, _ in calls] == ["b0", "b1"]
    for command, kwargs in calls:
        assert command[2:4] == ["-m", "sr_f5.calibration"]
        assert command[-4:] == ["--shard", "3", "--shards", "4"]
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"]["PYTHONPATH"] == f"{tmp_path}/src:{tmp_path}"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "7"


class SlurmFixture:
    def __init__(self):
        self.active = set()
        self.rows = {}
        self.terminal_rows = {
            str(i): dict(
                job_id=str(i),
                state="CANCELLED",
                user="me",
                qos=DISPATCH["QOS"],
                account=DISPATCH["ACCOUNT"],
            )
            for i in range(4)
        }

    def user_job_ids(self, _user):
        return self.active

    def queue(self):
        return dict(self.rows)

    def terminal(self, job_id):
        return self.terminal_rows.get(job_id)

    def show(self, job_id):
        return dict(ReqTRES="cpu=8,gres/gpu=2,gres/gpu:pro6000=2", AllocTRES="")


def test_replacement_requires_absence_and_terminal_accounting():
    slurm = SlurmFixture()
    original = dict(jobs=[dict(job_id=str(i), shard=i) for i in range(4)])
    assert len(DISPATCH["verify_previous_jobs"](slurm, original, "me")) == 4
    slurm.active = {"0"}
    with pytest.raises(RuntimeError, match="still active"):
        DISPATCH["verify_previous_jobs"](slurm, original, "me")
    slurm.active.clear()
    slurm.terminal_rows.pop("0")
    with pytest.raises(RuntimeError, match="unknown"):
        DISPATCH["verify_previous_jobs"](slurm, original, "me")


def test_capacity_counts_pending_and_acknowledged_visibility_lag():
    slurm = SlurmFixture()
    slurm.active = {"elsewhere"}
    slurm.rows = {"teacher": dict(ReqTRES="cpu=4,gres/gpu=1", AllocTRES="gres/gpu=1")}
    jobs, gpus, queue = DISPATCH["capacity"](slurm, "me", {"new"})
    assert jobs == 2
    assert gpus == 5
    assert set(queue) == {"teacher", "elsewhere", "new"}
    slurm.rows["unknown"] = {}
    with pytest.raises(RuntimeError, match="Unknown reservation"):
        DISPATCH["capacity"](slurm, "me", set())
