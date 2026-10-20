"""Pause coordination tests use disposable processes, never Slurm or SSH."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts/sr_f3_ce/cpu_dispatcher_pause.py"
spec = importlib.util.spec_from_file_location("cpu_pause_test", SOURCE)
pause = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pause)


@pytest.fixture
def dispatcher(tmp_path, monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    row = {"pid": child.pid, "startticks": "bound", "state": "S"}

    def read(_before):
        result = subprocess.check_output(["ps", "-o", "stat=", "-p", str(child.pid)], text=True)
        return dict(row, state=result.strip()[0])

    monkeypatch.setattr(pause, "identify", lambda *a: dict(row))
    monkeypatch.setattr(pause, "same_process", read)
    try:
        yield child, read
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGCONT)
            child.kill()
        child.wait()


def test_success_stops_then_resumes_same_process(dispatcher, tmp_path):
    child, read = dispatcher

    def action():
        assert read(None)["state"] == "T"
        return {"job_id": "215520", "submitted": False}

    assert pause.run_guarded(tmp_path, action, tmp_path / "journal")["job_id"] == "215520"
    assert read(None)["state"] != "T"
    resumed = json.loads((tmp_path / "journal/RESUMED.json").read_text())
    assert resumed["worker_reaped_before_resume"] is True
    assert resumed["process"]["pid"] == child.pid


def test_busy_lock_failure_resumes_without_retry(dispatcher, tmp_path):
    _, read = dispatcher

    def action():
        raise RuntimeError("Another SR-F3-CE submitter owns this run")

    with pytest.raises(RuntimeError, match="worker failed"):
        pause.run_guarded(tmp_path, action, tmp_path / "journal")
    assert read(None)["state"] != "T"
    assert "Another SR" in (tmp_path / "journal/ACTION_RESULT.json").read_text()


def test_timeout_reaps_lock_owner_before_resume(dispatcher, tmp_path):
    _, read = dispatcher
    import fcntl

    lock = tmp_path / "SUBMIT.lock"

    def action():
        with lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            time.sleep(5)

    with pytest.raises(TimeoutError, match="deadline"):
        pause.run_guarded(tmp_path, action, tmp_path / "journal", timeout_seconds=0.3)
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert read(None)["state"] != "T"


def test_identity_failure_never_sends_signal(tmp_path, monkeypatch):
    def refuse(*args):
        raise PermissionError("wrong original process")

    monkeypatch.setattr(pause, "identify", refuse)
    monkeypatch.setattr(pause, "send", lambda *a: pytest.fail("must not signal"))
    with pytest.raises(PermissionError):
        pause.run_guarded(tmp_path, lambda: None, tmp_path / "journal")


def test_process_identity_change_refuses_signal(monkeypatch):
    before = {"pid": 123, "startticks": "1", "state": "S"}
    monkeypatch.setattr(pause, "process", lambda pid: dict(before, startticks="2"))
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("must not signal reused PID"))
    with pytest.raises(PermissionError, match="identity changed"):
        pause.send(before, None, signal.SIGCONT)


def test_pause_deadline_is_bounded(tmp_path):
    with pytest.raises(ValueError):
        pause.run_guarded(tmp_path, lambda: None, tmp_path, timeout_seconds=121)


@pytest.mark.parametrize("change", [
    {"slurm_job_id": "215520"}, {"uids": [-1] * 4}, {"argv": ["wrong"]},
    {"exe": "/wrong/python"}, {"state": "T"}, {"state": "Z"},
])
def test_identify_rejects_other_job_owner_command_or_stopped_process(monkeypatch, change):
    argv = [sys.executable, "original-dispatch"]
    row = dict(pid=123, startticks="1", state="S", argv=argv, uids=[os.getuid()] * 4,
               slurm_job_id="211333", exe=str(Path(sys.executable).resolve()))
    row.update(change)
    monkeypatch.setattr(pause, "expected", lambda root: argv)
    monkeypatch.setattr(pause, "process", lambda pid: row)
    with pytest.raises(PermissionError, match="exactly one"):
        pause.identify("unused", expected_pid=123)


def test_timeout_also_stops_worker_subprocess_mutation(dispatcher, tmp_path):
    _, read = dispatcher
    marker = tmp_path / "must-not-write"

    def action():
        subprocess.run([
            sys.executable, "-c",
            "import time,pathlib; time.sleep(1); pathlib.Path(" + repr(str(marker)) +
            ").write_text('late mutation')",
        ], check=True)

    with pytest.raises(TimeoutError):
        pause.run_guarded(tmp_path, action, tmp_path / "journal", timeout_seconds=0.3)
    time.sleep(1.1)
    assert not marker.exists()
    assert read(None)["state"] != "T"
