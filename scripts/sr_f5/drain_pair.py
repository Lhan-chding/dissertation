"""Drain only pool coordinators after a complete frozen recovery subprocess.

Run as a CPU-only overlapping step INSIDE the identified Slurm allocation. Scientific
children never receive a signal. A pidfd pins each coordinator identity; any error
resumes paused coordinators. The original immutable deployment is not edited.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

OWNER_ROOT = Path("/projects/_ssd/varunssd/louis-ssvc")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def proc(pid, *, environment=False):
    base = Path("/proc") / str(pid)
    fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
    return dict(
        pid=pid,
        state=fields[0],
        parent=int(fields[1]),
        start=fields[19],
        exit_code=int(fields[49]),
        uid=base.stat().st_uid,
        args=(base / "cmdline").read_bytes().decode().strip("\0").split("\0"),
        env=dict(
            x.split("=", 1)
            for x in (base / "environ").read_bytes().decode().split("\0")
            if "=" in x
        )
        if environment
        else {},
    )


def has_option(args, name, value):
    return any(a == name and b == str(value) for a, b in pairwise(args))


def matches_worker(p, job, worker, code, root, revision):
    args, env = p["args"], p["env"]
    return (
        p["uid"] == os.getuid()
        and has_option(args, "-m", "sr_f5.pilot_pool")
        and "--worker" in args
        and has_option(args, "--root", root)
        and has_option(args, "--code", code)
        and has_option(args, "--revision", revision)
        and env.get("SLURM_JOB_ID") == str(job)
        and env.get("SRF5_SHARD") == str(worker)
    )


def matching_child(p, parent, job, root, arm):
    args = p["args"]
    return (
        p["uid"] == os.getuid()
        and p["parent"] == parent
        and p["env"].get("SLURM_JOB_ID") == str(job)
        and has_option(args, "-m", "sr_f5.pilot")
        and has_option(args, "--root", root)
        and has_option(args, "--parent", "b0")
        and has_option(args, "--arm", arm)
        and has_option(args, "--lane", "recovery")
    )


def target_call(receipt, arm):
    calls = receipt.get("calls", [])
    return (
        receipt.get("status") == "RUNNING"
        and bool(calls)
        and all(c.get("parent") == "b0" and c.get("arm") == arm for c in calls)
        and calls[-1].get("phase") == "recovery"
        and calls[-1].get("status") == "STARTED"
    )


def processes(root):
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            p = proc(int(path.name))
            if not has_option(p["args"], "--root", root):
                continue
            result.append(proc(int(path.name), environment=True))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def libc_pidfd_call(name, arguments, types):
    """Portable CPython may omit wrappers even when the host libc exports pidfds."""
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, name)
    function.argtypes = types
    function.restype = ctypes.c_int
    result = function(*arguments)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def open_coordinator(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return libc_pidfd_call("pidfd_open", (pid, 0), (ctypes.c_int, ctypes.c_uint))


def signal_coordinator(handle, sig):
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(handle, sig)
    else:
        libc_pidfd_call(
            "pidfd_send_signal",
            (handle, int(sig), None, 0),
            (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint),
        )


def finish_evidence(root, arm, child):
    base = root / "G3/RUNS/b0" / arm
    a, b = [read(base / lane / "COMPLETE.json") for lane in ("primary", "recovery")]
    if any(x.get("status") != "COMPLETE" or x.get("committed_step") != 12 for x in (a, b)):
        raise RuntimeError("Incomplete scientific phase; cannot drain")
    execution = b["execution"]
    if (
        execution["process"]["pid"] != child["pid"]
        or str(execution["process"]["start_ticks"]) != child["start"]
        or execution["process"]["host"] != socket.gethostname()
        or execution["restored_step"] != 4
        or a["execution"]["restored_step"] != 0
    ):
        raise RuntimeError("Completion is not bound to the uninterrupted target child")
    return base


def run(args):
    root, code = args.root.resolve(strict=True), args.code.resolve(strict=True)
    if (
        not root.is_relative_to(OWNER_ROOT)
        or root == OWNER_ROOT
        or not code.is_relative_to(OWNER_ROOT)
        or code == OWNER_ROOT
    ):
        raise PermissionError("Only canonical project descendants are authorized")
    if os.environ.get("SLURM_JOB_ID") != str(args.job):
        raise PermissionError("Controller must run inside the exact existing allocation")
    frozen = read(root / "G3/FROZEN.json")
    allocation = read(root / f"G3/dual_jobs/{args.job}.json")
    plan = read(root / "G3/POOL_PLAN.json")
    if (
        allocation["code"] != str(code)
        or allocation["code_revision"] != args.revision
        or frozen["code_revision"] != args.revision
        or allocation["status"] != "RUNNING"
        or allocation["workers"] != [0, 1]
    ):
        raise PermissionError("Only the registered slow pair 0 may be drained")
    tasks = {w: plan["workers"][w]["tasks"][0] for w in allocation["workers"]}
    if any(t["parent"] != "b0" or not t["recovery"] for t in tasks.values()):
        raise PermissionError("Target must be the first b0 recovery in each lane")
    directory = root / "CONTINUATION"
    path = directory / f"DRAIN_{args.job}.json"
    handles = {}
    children = {}
    done = set()
    cancelled = False
    report = dict(
        status="ARMED",
        job_id=str(args.job),
        source_revision=args.revision,
        controller_revision=args.controller_revision,
        controller=dict(pid=os.getpid(), host=socket.gethostname()),
        targets=tasks,
        started_at=datetime.now(timezone.utc).isoformat(),
        signals_only_coordinators=True,
        scientific_children_signalled=False,
        events=[],
    )
    with (directory / f"DRAIN_{args.job}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            raise PermissionError("Drain attempt exists; inspect before retry")
        write(path, report)
        deadline = time.monotonic() + args.max_seconds
        try:
            while len(done) != 2:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Drain deadline reached; release coordinators")
                for worker, task in tasks.items():
                    arm = task["arm"]
                    if worker not in handles:
                        receipt = read(root / f"G3/workers/W{worker}-{args.job}.json")
                        if receipt.get("status") != "RUNNING":
                            raise RuntimeError("Worker ended before planned boundary")
                        if not target_call(receipt, arm):
                            if any(
                                c.get("arm") != arm or c.get("phase") == "compare_recovery"
                                for c in receipt.get("calls", [])
                            ):
                                raise RuntimeError("Target phase already passed; do not interrupt")
                            continue
                        ps = processes(root)
                        parents = [
                            p
                            for p in ps
                            if matches_worker(p, args.job, worker, code, root, args.revision)
                        ]
                        if len(parents) != 1:
                            raise RuntimeError("Coordinator identity is ambiguous")
                        parent = parents[0]
                        cs = [
                            p for p in ps if matching_child(p, parent["pid"], args.job, root, arm)
                        ]
                        if not cs:  # Receipt precedes subprocess creation.
                            continue
                        if len(cs) != 1:
                            raise RuntimeError("Scientific child identity is ambiguous")
                        fd = open_coordinator(parent["pid"])
                        try:
                            fresh = proc(parent["pid"], environment=True)
                            if fresh["start"] != parent["start"] or not matches_worker(
                                fresh, args.job, worker, code, root, args.revision
                            ):
                                raise RuntimeError("Coordinator changed before signal")
                        except BaseException:
                            os.close(fd)
                            raise
                        handles[worker] = fd
                        report["events"].append(
                            dict(
                                worker=worker,
                                event="COORDINATOR_PAUSE_INTENT",
                                parent_pid=parent["pid"],
                                parent_start=parent["start"],
                                child_pid=cs[0]["pid"],
                                child_start=cs[0]["start"],
                            )
                        )
                        write(path, report)
                        signal_coordinator(fd, signal.SIGSTOP)
                        for _ in range(100):
                            if proc(parent["pid"])["state"] in ("T", "t"):
                                break
                            time.sleep(0.01)
                        else:
                            raise RuntimeError("Coordinator did not stop")
                        fresh_child = proc(cs[0]["pid"])
                        if (
                            fresh_child["start"] != cs[0]["start"]
                            or fresh_child["parent"] != parent["pid"]
                            or not target_call(
                                read(root / f"G3/workers/W{worker}-{args.job}.json"), arm
                            )
                        ):
                            raise RuntimeError("Phase advanced during pause; resume immediately")
                        children[worker] = cs[0]
                        report["events"].append(
                            dict(
                                worker=worker,
                                event="COORDINATOR_PAUSED",
                                parent_pid=parent["pid"],
                                child_pid=cs[0]["pid"],
                            )
                        )
                        write(path, report)
                    if worker in done:
                        continue
                    child = children[worker]
                    state = proc(child["pid"])
                    if state["start"] != child["start"]:
                        raise RuntimeError("Child identity changed")
                    if state["state"] != "Z":
                        continue
                    if state["exit_code"] != 0:
                        raise RuntimeError(
                            "Scientific child failed; retain evidence and resume coordinator"
                        )
                    base = finish_evidence(root, arm, child)
                    env = dict(
                        os.environ,
                        PYTHONPATH=f"{code}/src:{code}",
                        PYTHONDONTWRITEBYTECODE="1",
                        CUDA_VISIBLE_DEVICES="",
                    )
                    with (directory / f"DRAIN_{args.job}_W{worker}_COMPARE.log").open("a") as log:
                        subprocess.run(
                            [
                                str(args.python),
                                "-m",
                                "sr_f5.pilot",
                                "--root",
                                str(root),
                                "--parent",
                                "b0",
                                "--arm",
                                arm,
                                "--compare-recovery",
                            ],
                            env=env,
                            cwd=code,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=300,
                        )
                    if read(base / "RECOVERY.json")["status"] != "PASS":
                        raise RuntimeError(
                            "Recovery comparison failed; do not proceed to continuation"
                        )
                    done.add(worker)
                    report["events"].append(dict(worker=worker, event="RECOVERY_PASS_AT_BOUNDARY"))
                    write(path, report)
                time.sleep(args.poll_seconds)
            report.update(
                status="BOUNDARY_COMPLETE_CANCEL_REQUESTED",
                completed_at=datetime.now(timezone.utc).isoformat(),
                next_action=(
                    "Verify terminal allocation; submit same frozen pair in new job; "
                    "preserve all complete paths"
                ),
            )
            write(path, report)
            # Both scientific children have exited successfully; only idle coordinators remain.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            subprocess.run(["scancel", str(args.job)], check=True)
            cancelled = True
        except BaseException as exc:
            report.update(status="ERROR_COORDINATORS_RELEASED", error=repr(exc))
            write(path, report)
            raise
        finally:
            for fd in handles.values():
                try:
                    if not cancelled:
                        signal_coordinator(fd, signal.SIGCONT)
                except ProcessLookupError:
                    pass
                os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "code", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("job", "revision", "controller-revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--max-seconds", type=float, default=9 * 3600)
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(InterruptedError(signum)))
    run(args)


if __name__ == "__main__":
    main()
