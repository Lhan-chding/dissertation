"""Bounded, process-specific pause for the one authorized CPU node migration.

Run on cpu-2 as the dispatcher owner. The parent supervises the migration child;
it always reaps that child (releasing its file locks) before resuming the original
dispatcher. Dispatcher signals target only its original PID, never its Slurm job
or process group. Failure cleanup may kill the isolated migration child group.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import time
from contextlib import suppress
from pathlib import Path

ROOT = "/projects/_ssd/varunssd/louis-ssvc/sr_f3_ce_protocol_fix_20261016/selected_v2"
COMMIT = "35800b02b1ef1eff2378e4c933393b4ea2668e4d"
IDENTITY_SHA = "d03abcbf16c72f362ed73168ef383269b40599f9423bc4d3b0ef5d61a96eb33d"
JOB = "211333"


def save(path, value):
    """Publish once; never overwrite a previous pause's evidence."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def process(pid, proc=Path("/proc")):
    directory = proc / str(pid)
    raw = (directory / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    status = (directory / "status").read_text()
    uids = next(row.split()[1:] for row in status.splitlines() if row.startswith("Uid:"))
    environ = dict(
        item.split("=", 1)
        for item in (directory / "environ").read_bytes().decode().split("\0")
        if "=" in item
    )
    return {
        "pid": pid,
        "startticks": fields[19],
        "state": fields[0],
        "uids": [int(value) for value in uids],
        "argv": (directory / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
        "slurm_job_id": environ.get("SLURM_JOB_ID"),
        "exe": str((directory / "exe").resolve(strict=True)),
    }


def expected(root):
    root = Path(root).resolve(strict=True)
    if str(root) != ROOT or socket.gethostname().split(".")[0] != "cpu-2":
        raise PermissionError("Dispatcher pause is restricted to the registered root on cpu-2")
    identity_path = root / "operations" / COMMIT / "OPERATION_IDENTITY.json"
    body = identity_path.read_bytes()
    if hashlib.sha256(body).hexdigest() != IDENTITY_SHA:
        raise PermissionError("Original dispatcher identity changed")
    identity = json.loads(body)
    attempt = root / "operations/dispatcher/attempts/0007"
    state = json.loads((attempt / "STATE.json").read_text())
    started = json.loads((attempt / "STARTED.json").read_text())
    if state.get("job_id") != JOB or started.get("job_id") != JOB or (
        (attempt / "END.json").exists()
    ):
        raise PermissionError("Original dispatcher is not an active registered attempt")
    return [
        identity["python"], str(identity_path.parent / "dispatch_registered.py"),
        "--mode", "run", "--identity", str(identity_path), "--sequence", "7",
    ]


def identify(root, expected_pid=None):
    argv = expected(root)
    candidates = [int(expected_pid)] if expected_pid is not None else [
        int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit()
    ]
    matches = []
    for pid in candidates:
        try:
            row = process(pid)
        except (OSError, ValueError, StopIteration, UnicodeError):
            continue
        if row["argv"] == argv and row["slurm_job_id"] == JOB and (
            row["uids"] == [os.getuid()] * 4
            and row["exe"] == str(Path(argv[0]).resolve(strict=True))
        ):
            matches.append(row)
    if len(matches) != 1 or matches[0]["state"] in {"T", "t", "Z", "X"}:
        raise PermissionError("Expected exactly one live, unpaused original dispatcher process")
    return matches[0]


def same_process(before):
    after = process(before["pid"])
    if any(after[key] != before[key] for key in before if key != "state"):
        raise PermissionError("Dispatcher process identity changed; refusing signal")
    return after


def send(before, pidfd, signum):
    same_process(before)
    if pidfd is not None:
        signal.pidfd_send_signal(pidfd, signum)
    else:
        # Revalidation narrows the fallback race; deployments with pidfd support
        # always use the kernel-pinned process handle above.
        os.kill(before["pid"], signum)


def _reap(child, *, terminate=False):
    if terminate:
        with suppress(ProcessLookupError):
            # The migration child owns a fresh process group. Kill its outstanding
            # scontrol subprocess too: it must not mutate after lock release.
            os.killpg(child, signal.SIGKILL)
    while True:
        try:
            return os.waitpid(child, 0)
        except InterruptedError:
            continue


def run_guarded(root, action, journal, *, expected_pid=None, timeout_seconds=120):
    """Pause only 211333's Python; execute action in a supervised forked child.

    ``action`` must acquire the original SUBMIT.lock nonblockingly and release
    it before returning. Do not spawn detached children from action. Its result
    must be JSON serializable. The returned result is read only after resuming.
    """
    if not 0 < timeout_seconds <= 120:
        raise ValueError("Pause deadline must be positive and at most 120 seconds")
    if not hasattr(os, "fork"):
        raise RuntimeError("Guard requires a POSIX supervisor")
    journal = Path(journal)
    journal.mkdir(parents=True, exist_ok=True)
    before = identify(root, expected_pid)
    pidfd = os.pidfd_open(before["pid"]) if hasattr(os, "pidfd_open") else None
    child = None
    paused = False
    old_handlers = {}
    outcome_path = journal / "ACTION_RESULT.json"
    began = time.monotonic()

    def interrupted(signum, _frame):
        raise RuntimeError(f"Migration supervisor interrupted by signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old_handlers[signum] = signal.signal(signum, interrupted)
        same_process(before)
        save(journal / "PAUSE_INTENT.json", {
            "job_id": JOB, "process": before, "at": time.time(),
            "signal": "SIGSTOP", "timeout_seconds": timeout_seconds,
            "pidfd": pidfd is not None,
        })
        # Set before sending so a concurrent exception still attempts recovery.
        paused = True
        send(before, pidfd, signal.SIGSTOP)
        stop_deadline = time.monotonic() + min(5, timeout_seconds)
        while same_process(before)["state"] not in {"T", "t"}:
            if time.monotonic() > stop_deadline:
                raise TimeoutError("Dispatcher did not acknowledge stop")
            time.sleep(0.02)
        save(journal / "PAUSED.json", {
            "job_id": JOB, "process": same_process(before), "at": time.time(),
            "root": str(root),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })
        child = os.fork()
        if child == 0:
            os.setpgid(0, 0)
            for signum in old_handlers:
                signal.signal(signum, signal.SIG_DFL)
            if pidfd is not None:
                os.close(pidfd)
            try:
                result = action()
                save(outcome_path, {"status": "RETURNED", "result": result})
            except BaseException as error:
                try:
                    save(outcome_path, {"status": "FAILED", "error": repr(error)})
                finally:
                    os._exit(1)
            os._exit(0)
        # Establish the isolated group from both sides before entering the wait.
        # ESRCH is harmless if the child already exited.
        with suppress(ProcessLookupError):
            os.setpgid(child, child)
        while True:
            exited, wait_status = os.waitpid(child, os.WNOHANG)
            if exited:
                child = None
                if wait_status != 0:
                    raise RuntimeError("Migration worker failed; inspect ACTION_RESULT.json")
                break
            if time.monotonic() - began > timeout_seconds:
                raise TimeoutError("Migration pause deadline exceeded; outcome requires audit")
            time.sleep(0.05)
    finally:
        # During cleanup a second external signal must not interrupt lock release
        # and dispatcher recovery.
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if child is not None and child != 0:
                _reap(child, terminate=True)
            if paused:
                send(before, pidfd, signal.SIGCONT)
                resume_deadline = time.monotonic() + 5
                while same_process(before)["state"] in {"T", "t"}:
                    if time.monotonic() > resume_deadline:
                        raise TimeoutError("Dispatcher did not acknowledge resume")
                    time.sleep(0.02)
                save(journal / "RESUMED.json", {
                    "job_id": JOB, "process": same_process(before), "at": time.time(),
                    "elapsed_seconds": time.monotonic() - began,
                    "worker_reaped_before_resume": True,
                })
        finally:
            if pidfd is not None:
                os.close(pidfd)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    result = json.loads(outcome_path.read_text())
    if result["status"] != "RETURNED":
        raise RuntimeError("Migration worker did not return a verified result")
    return result["result"]
