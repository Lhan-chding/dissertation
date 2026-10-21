"""Small durable task pool: one independent queue consumer per allocated GPU.

The queue changes only execution order. Task dependencies and scientific work are
registered before launch; scores never enter the scheduler. fcntl locks work on
the cluster's shared filesystem without SQLite WAL's local-host assumptions.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

QOS = "soujanya-poria-startfund-2026-03"
ACCOUNT = "rose"
PARTITION = "cluster02"
GPU_LIMIT = 8
SUBMIT_LIMIT = 5
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,160}\Z")
_JOB_ID = re.compile(r"[0-9]+(?:_[0-9]+)?\Z")


def now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def _locked(root):
    directory = Path(root) / "pool"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "LOCK").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "STATE.json"
        state = _read(path) if path.exists() else None
        yield state


def _save(root, state):
    state["updated_at"] = now()
    _write(Path(root) / "pool/STATE.json", state)


def _validate_tasks(tasks):
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("A nonempty registered task list is required")
    indexed = {}
    for original in tasks:
        task = dict(original)
        key = task.get("task_id", "")
        if not isinstance(key, str) or not _TASK_ID.fullmatch(key) or ".." in key:
            raise ValueError("Invalid task_id")
        if key in indexed:
            raise ValueError("Duplicate task_id: " + key)
        dependencies = task.setdefault("dependencies", [])
        if not isinstance(dependencies, list) or any(not isinstance(x, str) for x in dependencies):
            raise ValueError("dependencies must be a list of task IDs")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("Duplicate task dependency")
        if task.get("status", "PENDING") not in ("PENDING", "BLOCKED", "SKIPPED"):
            raise ValueError("Only prospective PENDING/BLOCKED/SKIPPED tasks may be registered")
        indexed[key] = task
    visited, active = set(), set()

    def visit(key):
        if key in active:
            raise ValueError("Cyclic task dependency")
        if key in visited:
            return
        if key not in indexed:
            raise ValueError("Unknown dependency: " + key)
        active.add(key)
        for dependency in indexed[key]["dependencies"]:
            visit(dependency)
        active.remove(key)
        visited.add(key)

    for key in indexed:
        visit(key)
    return indexed


def init_pool(root, tasks):
    """Register once; an identical call is idempotent and never resets progress."""
    indexed = _validate_tasks(tasks)
    with _locked(root) as state:
        if state is not None:
            existing = {key: row["spec"] for key, row in state["tasks"].items()}
            if existing != indexed:
                raise ValueError("Registered task matrix already exists and differs")
            return _summary(state)
        state = {
            "version": 1,
            "created_at": now(),
            "tasks": {
                key: {"spec": task, "status": task.get("status", "PENDING"), "attempts": []}
                for key, task in indexed.items()
            },
            "jobs": {},
        }
        _save(root, state)
        return _summary(state)


def _effective_status(state, key):
    row = state["tasks"][key]
    if row["status"] != "PENDING":
        return row["status"]
    dependencies = [_effective_status(state, dep) for dep in row["spec"]["dependencies"]]
    if any(
        value in ("FAILED", "BLOCKED", "SKIPPED", "BLOCKED_DEPENDENCY") for value in dependencies
    ):
        return "BLOCKED_DEPENDENCY"
    if all(value == "COMPLETE" for value in dependencies):
        return "READY"
    return "WAITING"


def _summary(state):
    if state is None:
        raise FileNotFoundError("Pool has not been initialized")
    tasks = {
        key: {
            "status": _effective_status(state, key),
            "attempts": len(row["attempts"]),
            "owner": row.get("owner"),
            "reason": row.get("reason", row["spec"].get("blocked_reason")),
        }
        for key, row in state["tasks"].items()
    }
    counts = {}
    for row in tasks.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "counts": counts,
        "tasks": tasks,
        "jobs": state["jobs"],
        "all_complete": all(row["status"] == "COMPLETE" for row in tasks.values()),
        "work_remaining": any(
            row["status"] in ("READY", "RUNNING", "WAITING") for row in tasks.values()
        ),
    }


def status(root):
    with _locked(root) as state:
        return _summary(state)


def _task_lease(root, key, *, blocking=False):
    directory = Path(root) / "pool/leases"
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / f"{key}.lock").open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError:
        stream.close()
        return None
    return stream


def claim(root, slot):
    """Return (task, held lease), or (None, None). Caller holds lease through exit."""
    with _locked(root) as state:
        if state is None:
            raise FileNotFoundError("Pool has not been initialized")
        for key, row in state["tasks"].items():
            if _effective_status(state, key) != "READY":
                continue
            lease = _task_lease(root, key)
            if lease is None:
                continue
            owner = {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "slot": str(slot),
                "started_at": now(),
            }
            row.update(status="RUNNING", owner=owner)
            row.pop("reason", None)
            row["attempts"].append({"owner": owner, "status": "RUNNING"})
            _save(root, state)
            return dict(row["spec"], attempt=len(row["attempts"])), lease
    return None, None


def finish(root, task, returncode, *, error=None):
    key = task["task_id"]
    receipt_path = Path(root) / "tasks" / key / "COMPLETE.json"
    valid_receipt = False
    try:
        receipt = _read(receipt_path)
        valid_receipt = receipt.get("status") == "COMPLETE" and receipt.get("task_id", key) == key
    except (OSError, ValueError, AttributeError):
        pass
    complete = returncode == 0 and valid_receipt and error is None
    reason = (
        None
        if complete
        else (error or f"child_exit={returncode}; complete_receipt={valid_receipt}")
    )
    with _locked(root) as state:
        row = state["tasks"][key]
        if row["status"] != "RUNNING" or len(row["attempts"]) != task["attempt"]:
            raise RuntimeError("Task ownership changed before child exit")
        row.update(status="COMPLETE" if complete else "FAILED", reason=reason)
        row["attempts"][-1].update(
            status=row["status"],
            finished_at=now(),
            returncode=returncode,
            complete_receipt=valid_receipt,
            reason=reason,
        )
        row.pop("owner", None)
        _save(root, state)
    return complete


def gpu_count(tres):
    counts = {}
    for entry in str(tres or "").split(","):
        if not entry.startswith("gres/gpu"):
            continue
        match = re.fullmatch(r"(gres/gpu(?::[A-Za-z0-9_-]+)?)=([0-9]+)", entry)
        if not match or match[1] in counts:
            raise ValueError("Unknown GPU reservation: " + str(tres))
        counts[match[1]] = int(match[2])
    typed = sum(count for name, count in counts.items() if name != "gres/gpu")
    return max(counts.get("gres/gpu", 0), typed)


class Slurm:
    def run(self, arguments):
        return subprocess.run(arguments, text=True, capture_output=True, check=False, timeout=60)

    def show(self, job_id):
        result = self.run(["scontrol", "show", "job", str(job_id), "--oneliner"])
        if result.returncode:
            raise RuntimeError("Cannot read Slurm job: " + result.stderr)
        return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", result.stdout))

    def queue(self):
        result = self.run(["squeue", "--noheader", "--array", "--qos", QOS, "--format=%i"])
        if result.returncode:
            raise RuntimeError("Cannot read teacher QoS queue: " + result.stderr)
        ids = result.stdout.split()
        if len(set(ids)) != len(ids) or any(not _JOB_ID.fullmatch(key) for key in ids):
            raise RuntimeError("Teacher QoS queue IDs are ambiguous")
        return {key: self.show(key) for key in ids}

    def terminal(self, job_id):
        result = self.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                str(job_id),
                "--format=JobIDRaw,State,ExitCode,User%80,Account%80,QOS%100",
            ]
        )
        if result.returncode:
            raise RuntimeError("Cannot read Slurm accounting: " + result.stderr)
        rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
        rows = [row for row in rows if row[0] == str(job_id)]
        if not rows:
            return None
        if len(rows) != 1 or len(rows[0]) < 6:
            raise RuntimeError("Ambiguous Slurm accounting")
        row = rows[0]
        return {
            "job_id": row[0],
            "state": row[1].split()[0].rstrip("+"),
            "exit_code": row[2],
            "user": row[3],
            "account": row[4],
            "qos": row[5],
        }


def usage(queue, user):
    gpus, submitted = 0, 0
    for row in queue.values():
        if row.get("QOS") != QOS:
            raise ValueError("Unexpected queue QoS")
        if not row.get("ReqTRES") or not row.get("UserId"):
            raise ValueError("Incomplete queue capacity information")
        gpus += max(gpu_count(row["ReqTRES"]), gpu_count(row.get("AllocTRES")))
        submitted += row["UserId"].split("(")[0] == user
    return {"gpus": gpus, "submitted": submitted}


def _reserved_usage(root, queue, slurm, user):
    capacity = usage(queue, user)
    with _locked(root) as state:
        absent = [
            (key, dict(row))
            for key, row in state["jobs"].items()
            if row["status"] in ("SUBMITTED", "RUNNING") and row.get("job_id") not in queue
        ]
    for key, row in absent:
        terminal = slurm.terminal(row["job_id"])
        if terminal and terminal["state"] not in (
            "RUNNING",
            "PENDING",
            "CONFIGURING",
            "COMPLETING",
        ):
            with _locked(root) as state:
                current = state["jobs"][key]
                if current["status"] in ("SUBMITTED", "RUNNING"):
                    current.update(status="TERMINAL", scheduler_terminal=terminal)
                    _save(root, state)
        else:
            # squeue disappearance alone is not evidence that a reservation ended.
            capacity["gpus"] += row["gpus"]
            capacity["submitted"] += row["user"] == user
    return capacity


def reconcile_timeouts(root, *, slurm=None):
    """Recover only released leases whose recorded job is terminal TIMEOUT.

    Failed jobs are recorded and blocked. A complete receipt alone cannot certify
    child exit; a resumable task re-enters its own idempotent entrypoint.
    """
    slurm = slurm or Slurm()
    with _locked(root) as state:
        running = [
            (key, dict(row)) for key, row in state["tasks"].items() if row["status"] == "RUNNING"
        ]
    jobs = {row["owner"].get("job_id") for _, row in running}
    jobs.discard(None)
    if not jobs:
        return []
    queue = slurm.queue()
    terminal = {job: slurm.terminal(job) for job in jobs if job not in queue}
    changes = []
    for key, previous in running:
        job = previous["owner"].get("job_id")
        record = terminal.get(job)
        if not record or record["state"] in ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING"):
            continue
        if record.get("qos") != QOS or record.get("account") != ACCOUNT:
            raise PermissionError("Stale task accounting is outside the authorized allocation")
        lease = _task_lease(root, key)
        if lease is None:
            continue
        try:
            with _locked(root) as state:
                row = state["tasks"][key]
                if row["status"] != "RUNNING" or row.get("owner") != previous["owner"]:
                    continue
                recover = (
                    record["state"] == "TIMEOUT" and row["spec"].get("resume_supported") is True
                )
                row["attempts"][-1].update(
                    status="TIMEOUT" if recover else "FAILED", terminal=record, finished_at=now()
                )
                row.update(
                    status="PENDING" if recover else "FAILED",
                    reason="lease ended: " + record["state"],
                )
                row.pop("owner", None)
                _save(root, state)
                changes.append({"task_id": key, "status": row["status"], "terminal": record})
        finally:
            lease.close()
    return changes


def worker(root, slot, *, python=None, poll_seconds=10):
    """Consume ready work independently until no active or eligible work remains."""
    root = Path(root).resolve()
    python = python or sys.executable
    last_reconcile = 0.0
    while True:
        if (root / "pool/STOP").exists():
            return 0
        task, lease = claim(root, slot)
        if task is None:
            summary = status(root)
            if not summary["work_remaining"]:
                return 0
            if os.environ.get("SLURM_JOB_ID") and time.monotonic() - last_reconcile >= 60:
                reconcile_timeouts(root)
                last_reconcile = time.monotonic()
            time.sleep(poll_seconds)
            continue
        try:
            log = root / "pool/logs" / f"{task['task_id']}.attempt-{task['attempt']:03d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [python, "-m", "sr_f4.run", "--root", str(root), "--task", task["task_id"]]
            with log.open("a") as output:
                output.write(
                    json.dumps({"command": command, "slot": str(slot), "at": now()}) + "\n"
                )
                output.flush()
                try:
                    result = subprocess.run(
                        command, stdout=output, stderr=subprocess.STDOUT, check=False
                    )
                except OSError as exc:
                    finish(root, task, -1, error=str(exc))
                else:
                    finish(root, task, result.returncode)
        finally:
            lease.close()


def job(root, job_key, *, python=None):
    root = Path(root).resolve()
    python = python or sys.executable
    # A fast allocation can start before sbatch's caller persists its returned ID.
    deadline = time.monotonic() + 30
    while True:
        with _locked(root) as state:
            registered = dict(state["jobs"][job_key])
        if registered.get("job_id") or registered["status"] != "SUBMITTING":
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Slurm submission has not been acknowledged")
        time.sleep(0.2)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if registered.get("job_id") != job_id:
        raise PermissionError("Pool job does not match its registered Slurm allocation")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (
        len(devices) != registered["gpus"]
        or len(set(devices)) != len(devices)
        or any(not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", value) for value in devices)
    ):
        raise PermissionError("Expected one distinct visible GPU per pool slot")
    fields = Slurm().show(job_id)
    if any(
        fields.get(key) != value
        for key, value in {
            "QOS": QOS,
            "Account": ACCOUNT,
            "Partition": PARTITION,
            "JobName": registered["name"],
        }.items()
    ) or max(gpu_count(fields.get("ReqTRES")), gpu_count(fields.get("AllocTRES"))) != len(devices):
        raise PermissionError("Allocation differs from the registered teacher QoS request")
    with _locked(root) as state:
        state["jobs"][job_key].update(status="RUNNING", started_at=now(), devices=devices)
        _save(root, state)
    children = []
    for index, device in enumerate(devices):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, SRF4_POOL_SLOT=f"{job_id}:{index}")
        command = [
            python,
            "-m",
            "sr_f4.pool",
            "worker",
            "--root",
            str(root),
            "--slot",
            f"{job_id}:{index}",
        ]
        children.append(subprocess.Popen(command, env=env))
    codes = [child.wait() for child in children]
    with _locked(root) as state:
        state["jobs"][job_key].update(
            status="FINISHED" if not any(codes) else "FAILED",
            finished_at=now(),
            worker_exit_codes=codes,
        )
        _save(root, state)
    return int(any(codes))


def submit(
    root,
    code,
    python,
    *,
    slurm=None,
    user=None,
    cpus_per_gpu=8,
    memory_gib_per_gpu=96,
    highmem=False,
    maximum_new_jobs=4,
):
    """Fill available capacity with up to four two-GPU allocations.

    Queue capacity is refreshed immediately before every sbatch. All users in
    the shared QoS count toward its eight GPUs; this user's CPU jobs also count
    toward MaxSubmitPU=5. Unknown submission outcomes block repeat submission.
    """
    root, code = Path(root).resolve(), Path(code).resolve()
    python = str(Path(python).resolve())
    slurm, user = slurm or Slurm(), user or getpass.getuser()
    if cpus_per_gpu < 1 or memory_gib_per_gpu < 1 or not 1 <= maximum_new_jobs <= SUBMIT_LIMIT:
        raise ValueError("Invalid allocation resources")
    directory = root / "pool"
    directory.mkdir(parents=True, exist_ok=True)
    launched = []
    # Serialize submitters, while permitting workers to keep taking tasks.
    with (directory / "SUBMIT.lock").open("a+") as submit_lock:
        fcntl.flock(submit_lock, fcntl.LOCK_EX)
        reconcile_timeouts(root, slurm=slurm)
        for _ in range(maximum_new_jobs):
            summary = status(root)
            if not summary["work_remaining"]:
                break
            with _locked(root) as state:
                unknown = [
                    key
                    for key, row in state["jobs"].items()
                    if row["status"] in ("SUBMITTING", "SUBMISSION_UNKNOWN")
                ]
            if unknown:
                raise RuntimeError(
                    "Resolve unknown Slurm submission before retrying: " + ", ".join(unknown)
                )
            queue = slurm.queue()
            capacity = _reserved_usage(root, queue, slurm, user)
            available = GPU_LIMIT - capacity["gpus"]
            if available <= 0 or capacity["submitted"] >= SUBMIT_LIMIT:
                break
            count = min(2, available)
            with _locked(root) as state:
                sequence = len(state["jobs"]) + 1
                key = f"job-{sequence:04d}"
                name = f"LH-SRF4-{sequence:04d}"
                script = directory / "jobs" / f"{key}.sbatch"
                script.parent.mkdir(parents=True, exist_ok=True)
                output = directory / "logs" / f"{key}-%j.log"
                output.parent.mkdir(parents=True, exist_ok=True)
                lines = [
                    "#!/bin/bash",
                    "set -euo pipefail",
                    f"cd {shlex.quote(str(code))}",
                    f"export PYTHONPATH={shlex.quote(str(code / 'src') + ':' + str(code))}",
                    "export OMP_NUM_THREADS=4",
                    "export TOKENIZERS_PARALLELISM=false",
                    "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
                    "exec "
                    + shlex.join(
                        [python, "-m", "sr_f4.pool", "job", "--root", str(root), "--job-key", key]
                    ),
                ]
                script.write_text("\n".join(lines) + "\n")
                registration = {
                    "name": name,
                    "gpus": count,
                    "status": "SUBMITTING",
                    "created_at": now(),
                    "user": user,
                    "qos": QOS,
                    "account": ACCOUNT,
                    "partition": PARTITION,
                    "script": str(script),
                    "python": python,
                    "code": str(code),
                    "cpus": cpus_per_gpu * count,
                    "memory_gib": memory_gib_per_gpu * count,
                    "queue_usage": capacity,
                }
                state["jobs"][key] = registration
                _save(root, state)
            command = [
                "sbatch",
                "--parsable",
                f"--job-name={name}",
                f"--account={ACCOUNT}",
                f"--qos={QOS}",
                f"--partition={PARTITION}",
                "--nodes=1",
                "--ntasks=1",
                f"--gres=gpu:pro6000:{count}",
                f"--cpus-per-task={cpus_per_gpu * count}",
                f"--mem={memory_gib_per_gpu * count}G",
                "--time=3-00:00:00",
                "--no-requeue",
                f"--output={output}",
                "--export=ALL",
            ]
            if highmem:
                command.append("--constraint=highmem")
            command.append(str(script))
            try:
                result = slurm.run(command)
            except (OSError, subprocess.TimeoutExpired) as exc:
                with _locked(root) as state:
                    state["jobs"][key].update(status="SUBMISSION_UNKNOWN", error=str(exc))
                    _save(root, state)
                raise RuntimeError("Slurm submission outcome unknown; do not resubmit") from exc
            value = result.stdout.strip().split(";", 1)[0]
            with _locked(root) as state:
                record = state["jobs"][key]
                record.update(
                    submit_stdout=result.stdout, submit_stderr=result.stderr, command=command
                )
                if result.returncode:
                    record.update(status="REJECTED", returncode=result.returncode)
                elif not _JOB_ID.fullmatch(value):
                    record.update(status="SUBMISSION_UNKNOWN")
                else:
                    record.update(status="SUBMITTED", job_id=value)
                    launched.append(dict(record, job_key=key))
                _save(root, state)
            if result.returncode:
                break
            if not _JOB_ID.fullmatch(value):
                raise RuntimeError("Slurm returned an unrecognized job ID; do not resubmit")
    return {"submitted": launched, "pool": status(root)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "status", "worker", "job", "submit", "reconcile"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--slot", default="0")
    parser.add_argument("--job-key")
    parser.add_argument("--code", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cpus-per-gpu", type=int, default=8)
    parser.add_argument("--memory-gib-per-gpu", type=int, default=96)
    parser.add_argument("--maximum-new-jobs", type=int, default=4)
    parser.add_argument("--highmem", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "init":
        result = init_pool(args.root, _read(args.tasks or args.root / "TASKS.json"))
    elif args.mode == "status":
        result = status(args.root)
    elif args.mode == "worker":
        return worker(args.root, args.slot, python=args.python)
    elif args.mode == "job":
        if not args.job_key:
            parser.error("job requires --job-key")
        return job(args.root, args.job_key, python=args.python)
    elif args.mode == "reconcile":
        result = reconcile_timeouts(args.root)
    else:
        if not args.code:
            parser.error("submit requires --code")
        result = submit(
            args.root,
            args.code,
            args.python,
            cpus_per_gpu=args.cpus_per_gpu,
            memory_gib_per_gpu=args.memory_gib_per_gpu,
            highmem=args.highmem,
            maximum_new_jobs=args.maximum_new_jobs,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
