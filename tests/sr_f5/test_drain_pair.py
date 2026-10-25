"""No model calls: protect allocation/process identities and phase boundaries."""

import importlib.util
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def controller(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "drain_pair", Path(__file__).parents[2] / "scripts/sr_f5/drain_pair.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, code = tmp_path / "run", tmp_path / "code"
    code.mkdir()
    (root / "CONTINUATION").mkdir(parents=True)
    args = SimpleNamespace(
        root=root,
        code=code,
        python=Path("/python"),
        job="123",
        revision="frozen",
        controller_revision="controller",
        max_seconds=10,
        poll_seconds=0,
    )
    monkeypatch.setattr(module, "OWNER_ROOT", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", args.job)

    def put(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    put("G3/FROZEN.json", {"code_revision": "frozen"})
    put(
        "G3/dual_jobs/123.json",
        dict(code=str(code), code_revision="frozen", status="RUNNING", workers=[0, 1]),
    )
    tasks = [dict(parent="b0", arm=arm, recovery=True) for arm in ["J", "SHUF"]]
    put("G3/POOL_PLAN.json", {"workers": [dict(tasks=[t]) for t in tasks]})
    ps = {}
    for worker, task in enumerate(tasks):
        arm = task["arm"]
        parent = 100 + worker
        child = 200 + worker
        put(
            f"G3/workers/W{worker}-123.json",
            dict(
                status="RUNNING",
                calls=[dict(parent="b0", arm=arm, phase="recovery", status="STARTED")],
            ),
        )
        env = dict(SLURM_JOB_ID="123", SRF5_SHARD=str(worker))
        ps[parent] = dict(
            pid=parent,
            parent=1,
            start="10",
            state="S",
            uid=os.getuid(),
            env=env,
            args=[
                "python",
                "-m",
                "sr_f5.pilot_pool",
                "--worker",
                "--root",
                str(root),
                "--code",
                str(code),
                "--revision",
                "frozen",
            ],
        )
        ps[child] = dict(
            pid=child,
            parent=parent,
            start="20",
            state="Z",
            exit_code=0,
            uid=os.getuid(),
            env=env,
            args=[
                "python",
                "-m",
                "sr_f5.pilot",
                "--root",
                str(root),
                "--parent",
                "b0",
                "--arm",
                arm,
                "--lane",
                "recovery",
            ],
        )
        for lane, restored in [("primary", 0), ("recovery", 4)]:
            put(
                f"G3/RUNS/b0/{arm}/{lane}/COMPLETE.json",
                dict(
                    status="COMPLETE",
                    committed_step=12,
                    execution=dict(
                        restored_step=restored,
                        process=dict(pid=child, start_ticks="20", host=module.socket.gethostname()),
                    ),
                ),
            )
        put(f"G3/RUNS/b0/{arm}/RECOVERY.json", {"status": "PASS"})
    signals, commands, closed = [], [], []
    monkeypatch.setattr(module, "processes", lambda r: list(ps.values()))
    monkeypatch.setattr(module, "proc", lambda pid, **kw: ps[pid].copy())
    monkeypatch.setattr(module, "open_coordinator", lambda pid: pid)
    monkeypatch.setattr(module.os, "close", closed.append)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)

    def send(fd, sig):
        signals.append((fd, sig))
        ps[fd]["state"] = "T" if sig == signal.SIGSTOP else "S"

    monkeypatch.setattr(module, "signal_coordinator", send)
    monkeypatch.setattr(module.subprocess, "run", lambda cmd, **kwargs: commands.append(cmd))
    return SimpleNamespace(
        m=module, args=args, ps=ps, signals=signals, commands=commands, put=put, closed=closed
    )


def test_success_stops_only_coordinators_and_compares_before_cancel(controller):
    c = controller
    c.m.run(c.args)
    assert c.signals == [(100, signal.SIGSTOP), (101, signal.SIGSTOP)]
    assert len(c.commands) == 3
    assert all("--compare-recovery" in cmd for cmd in c.commands[:2])
    assert c.commands[-1] == ["scancel", "123"]
    assert (
        c.m.read(c.args.root / "CONTINUATION/DRAIN_123.json")["status"]
        == "BOUNDARY_COMPLETE_CANCEL_REQUESTED"
    )


def test_failed_child_resumes_and_never_cancels(controller):
    c = controller
    c.ps[200]["exit_code"] = 256
    with pytest.raises(RuntimeError, match="Scientific child failed"):
        c.m.run(c.args)
    assert c.signals == [(100, signal.SIGSTOP), (100, signal.SIGCONT)]
    assert not c.commands


def test_pid_reuse_never_signals_replacement_even_in_cleanup(controller, monkeypatch):
    c = controller
    original = c.m.proc

    def reused(pid, **kw):
        p = original(pid, **kw)
        if kw.get("environment"):
            p["start"] = "REUSED"
        return p

    monkeypatch.setattr(c.m, "proc", reused)
    with pytest.raises(RuntimeError, match="Coordinator changed"):
        c.m.run(c.args)
    assert not c.signals and c.closed == [100]


def test_failed_comparison_releases_coordinator(controller):
    c = controller
    c.put("G3/RUNS/b0/J/RECOVERY.json", {"status": "FAIL"})
    with pytest.raises(RuntimeError, match="Recovery comparison failed"):
        c.m.run(c.args)
    assert c.signals[-1] == (100, signal.SIGCONT)
    assert not any(cmd[0] == "scancel" for cmd in c.commands)


def test_wrong_job_and_late_phase_are_rejected(controller, monkeypatch):
    c = controller
    monkeypatch.setenv("SLURM_JOB_ID", "someone_else")
    with pytest.raises(PermissionError, match="exact existing allocation"):
        c.m.run(c.args)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    c.put(
        "G3/workers/W0-123.json",
        dict(
            status="RUNNING",
            calls=[dict(parent="b0", arm="J", phase="compare_recovery", status="STARTED")],
        ),
    )
    with pytest.raises(RuntimeError, match="already passed"):
        c.m.run(c.args)
    assert not c.signals


def test_child_binding_requires_original_fresh_recovery(controller):
    c = controller
    c.ps[200]["start"] = "WRONG"
    with pytest.raises(RuntimeError, match="Completion is not bound"):
        c.m.run(c.args)
    assert c.signals[-1] == (100, signal.SIGCONT)
    assert not c.commands


def test_unrelated_worker_is_not_selected(controller):
    c = controller
    c.ps[100]["env"]["SLURM_JOB_ID"] = "999"
    with pytest.raises(RuntimeError, match="identity is ambiguous"):
        c.m.run(c.args)
    assert not c.signals


@pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd API required")
def test_real_pidfd_pause_does_not_stop_scientific_child():
    import subprocess
    import sys
    import time

    spec = importlib.util.spec_from_file_location(
        "drain_live", Path(__file__).parents[2] / "scripts/sr_f5/drain_pair.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,time; p=os.fork(); "
            "\nif p==0: time.sleep(0.2); os._exit(0)"
            "\nprint(p,flush=True); os.waitpid(p,0)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    fd = module.open_coordinator(parent.pid)
    try:
        child = int(parent.stdout.readline())
        module.signal_coordinator(fd, signal.SIGSTOP)
        deadline = time.monotonic() + 3
        while module.proc(child)["state"] != "Z" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert module.proc(parent.pid)["state"] == "T"
        assert module.proc(child)["state"] == "Z"
        assert module.proc(child)["exit_code"] == 0
    finally:
        module.signal_coordinator(fd, signal.SIGCONT)
        os.close(fd)
        parent.communicate(timeout=3)
    assert parent.returncode == 0
