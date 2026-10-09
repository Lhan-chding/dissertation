"""Durable, identity-bound Slurm orchestration for the fixed MM-DEV F2 matrix.

The journal is authoritative. Submission intent is fsynced before sbatch; an
ambiguous outcome consumes a concurrency slot until Slurm positively identifies
the registered job. GPU time is accounting only, never a study stopping rule.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .contract import PLAN_ID, run_matrix

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "PREEMPTED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
}
AUTO_RESUME = {"PREEMPTED", "TIMEOUT"}
LIVE = {"SUBMITTING", "REGISTERED", "ACTIVE", "UNKNOWN"}
STATES = ("S0", "S_A_0", "S_AP_0", "S_A_1", "S_AP_1")
PERMISSION_PATH = "manifests/ALLOCATION_PERMISSION.json"
SACCT_FIELDS = (
    "JobIDRaw",
    "JobName",
    "State",
    "User",
    "Account",
    "QOS",
    "AllocTRES",
    "ElapsedRaw",
    "Start",
    "End",
    "ExitCode",
    "Comment",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def safe_id(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value), "UNSAFE_TASK_ID")
    return value


def contained(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    require(
        not relative.is_absolute() and relative.parts and ".." not in relative.parts,
        "UNSAFE_ARTIFACT_PATH",
    )
    path = root / relative
    require(path.resolve().is_relative_to(root), "ARTIFACT_ESCAPED_ROOT")
    require(
        not any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)),
        "SYMLINK_ARTIFACT",
    )
    return path


@contextmanager
def process_lease(path):
    """Kernel lease; never unlink a lock inode or infer death from timestamps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("LEASE_BUSY:" + str(path)) from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(
                json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "time": now()})
            )
            stream.flush()
            os.fsync(stream.fileno())
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def worker_lease(root, task_id):
    return process_lease(Path(root) / "orchestration/leases" / (safe_id(task_id) + ".lock"))


def _marker(root, task_id, status, artifacts, metadata=None, reason=None):
    root, task_id = Path(root), safe_id(task_id)
    registration = read_json(root / "orchestration/REGISTRATION.json")
    require(task_id in registration["tasks"], "UNREGISTERED_TASK")
    attempt_id = os.environ.get("MM_DEV_ATTEMPT_ID")
    require(attempt_id and os.environ.get("MM_DEV_TASK_ID") == task_id, "MISSING_WORKER_IDENTITY")
    require(
        os.environ.get("MM_DEV_REGISTRATION_HASH") == digest(registration),
        "WORKER_REGISTRATION_MISMATCH",
    )
    require(artifacts and len(artifacts) == len(set(map(str, artifacts))), "ARTIFACTS_REQUIRED")
    descriptors = {}
    for relative in artifacts:
        path = contained(root, relative)
        require(path.is_file(), "MISSING_ARTIFACT:" + str(relative))
        descriptors[str(relative)] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    marker = {
        "plan_id": PLAN_ID,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "registration_hash": digest(registration),
        "status": status,
        "time": now(),
        "artifacts": descriptors,
        "metadata": metadata or {},
        "reason": reason,
    }
    relative = (
        f"orchestration/completions/{task_id}.json"
        if status == "COMPLETE"
        else f"orchestration/checkpoints/{safe_id(attempt_id)}.json"
    )
    atomic_json(root / relative, marker, exclusive=True)
    return marker


def complete_task(root, task_id, artifacts, metadata=None):
    return _marker(root, task_id, "COMPLETE", artifacts, metadata)


def checkpoint_task(root, task_id, reason, artifacts, metadata=None):
    require(reason in {"PREEMPTION", "TIME_LEASE_END"}, "NONRESUMABLE_REASON")
    return _marker(root, task_id, "CHECKPOINTED", artifacts, metadata, reason)


def fail_task(root, task_id, error):
    """Preserve a software/identity failure even if Slurm later reports preemption."""
    root, task_id = Path(root), safe_id(task_id)
    registration = read_json(root / "orchestration/REGISTRATION.json")
    attempt_id = safe_id(os.environ.get("MM_DEV_ATTEMPT_ID"))
    require(
        task_id in registration["tasks"]
        and os.environ.get("MM_DEV_TASK_ID") == task_id
        and os.environ.get("MM_DEV_REGISTRATION_HASH") == digest(registration),
        "WORKER_REGISTRATION_MISMATCH",
    )
    atomic_json(
        root / f"orchestration/failures/{attempt_id}.json",
        {
            "plan_id": PLAN_ID,
            "task_id": task_id,
            "attempt_id": attempt_id,
            "registration_hash": digest(registration),
            "time": now(),
            "status": "TECHNICAL_BLOCKED",
            "error": str(error),
        },
        exclusive=True,
    )


def tasks_for(matrix, measurement_shards=1):
    require(matrix == run_matrix(), "RUN_MATRIX_DIFFERS_FROM_FROZEN_CONTRACT")
    require(type(measurement_shards) is int and 1 <= measurement_shards <= 5, "INVALID_SHARDS")
    tasks = {
        "ENGINE_F2": {
            "phase": "ENGINE",
            "dependencies": [],
            "gpus": 1,
            "script": "run_engine_f2.py",
            "arguments": [],
        }
    }
    preps = [row["run_id"] for row in matrix if row["phase"] == "PREP"]
    producers = {row["output_state"]: row["run_id"] for row in matrix if row["phase"] == "PREP"}
    for row in matrix[:4]:
        tasks[row["run_id"]] = {
            "phase": "PREP",
            "dependencies": ["ENGINE_F2"],
            "gpus": 1,
            "script": "run_train_path.py",
            "arguments": ["--run-id", row["run_id"]],
        }

    def measurement(phase, state, dependencies):
        identifiers = []
        for shard in range(measurement_shards):
            task = f"{phase.lower()}_{state}"
            if measurement_shards > 1:
                task += f"_shard{shard}of{measurement_shards}"
            tasks[task] = {
                "phase": phase,
                "dependencies": dependencies,
                "gpus": 1,
                "script": "run_probe.py" if phase == "PROBE" else "run_eval.py",
                "arguments": [
                    "--state-id",
                    state,
                    "--shard",
                    str(shard),
                    "--shards",
                    str(measurement_shards),
                ],
            }
            identifiers.append(task)
        return identifiers

    probes = []
    for state in STATES:
        probes.extend(measurement("PROBE", state, [producers.get(state, "ENGINE_F2")]))
    # All starts/probes must be fixed before any continuation; completion speed cannot select paths.
    for row in matrix[4:]:
        tasks[row["run_id"]] = {
            "phase": "CONTINUE",
            "dependencies": [*preps, *probes],
            "gpus": 1,
            "script": "run_train_path.py",
            "arguments": ["--run-id", row["run_id"]],
        }
    evaluations = []
    for state in STATES:
        evaluations.extend(measurement("EVAL", state, [producers.get(state, "ENGINE_F2")]))
    for row in matrix[4:]:
        evaluations.extend(measurement("EVAL", row["run_id"], [row["run_id"]]))
    science = [row["run_id"] for row in matrix]
    tasks["ANALYZE"] = {
        "phase": "ANALYZE",
        "dependencies": [*science, *probes, *evaluations],
        "gpus": 0,
        "script": "score_and_analyze.py",
        "arguments": [],
    }
    tasks["RELEASE"] = {
        "phase": "RELEASE",
        "dependencies": ["ANALYZE", *science, *probes, *evaluations],
        "gpus": 0,
        "script": "verify_release.py",
        "arguments": [],
    }
    return tasks


def validate_permission(permission, root):
    require(
        permission.get("plan_id") == PLAN_ID and permission.get("authorized") is True,
        "ALLOCATION_PERMISSION_REQUIRED",
    )
    require(
        Path(permission.get("run_root", "")).resolve() == Path(root).resolve(),
        "PERMISSION_ROOT_MISMATCH",
    )
    for key in ("owner", "account", "qos"):
        require(
            isinstance(permission.get(key), str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+", permission[key]),
            "INVALID_PERMISSION:" + key,
        )
    require(
        permission.get("gres") == "gpu:pro6000:1"
        and permission.get("gpus_per_worker") == 1
        and permission.get("max_gpus_concurrent") == 5,
        "INVALID_GPU_PERMISSION",
    )
    require(permission.get("verified_at"), "UNVERIFIED_PERMISSION")
    if permission.get("partition") is not None:
        require(re.fullmatch(r"[A-Za-z0-9_,.-]+", permission["partition"]), "INVALID_PARTITION")


class SlurmBackend:
    """No cancellation interface: this controller cannot cancel another study."""

    def _run(self, command):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=45,
            env={**os.environ, "TZ": "UTC", "SLURM_TIME_FORMAT": "standard"},
        )
        return {
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "time": now(),
        }

    def submit(self, command):
        return self._run(command)

    def release(self, job_id):
        return self._run(["scontrol", "release", str(job_id)])

    def submission_capacity(self, permission):
        """Count every submitted job for this user/QoS, including CPU controllers."""
        limits = self._run(
            [
                "sacctmgr",
                "--noheader",
                "--parsable2",
                "show",
                "qos",
                "where",
                "name=" + permission["qos"],
                "format=Name%256,MaxSubmitJobsPU",
            ]
        )
        # Expand pending arrays so a compressed array cannot be mistaken for one slot.
        # Do not filter by account, study name, dependency, held state, or GPU request.
        queue = self._run(
            [
                "squeue",
                "--noheader",
                "--array",
                "--states=all",
                "--user",
                permission["owner"],
                "--format=%i|%u|%q|%T",
            ]
        )
        receipts = {"sacctmgr": limits, "squeue": queue}
        if limits["returncode"] != 0 or queue["returncode"] != 0:
            raise RuntimeError("SLURM_CAPACITY_QUERY_FAILED:" + json.dumps(receipts))
        rows = [line.split("|") for line in limits["stdout"].splitlines() if line.strip()]
        for values in rows:
            if len(values) == 3 and values[-1] == "":
                values.pop()
        require(
            len(rows) == 1 and len(rows[0]) == 2 and rows[0][0].strip() == permission["qos"],
            "INVALID_QOS_CAPACITY_ROW",
        )
        limit = rows[0][1].strip()
        require(
            limit in {"", "UNLIMITED"} or re.fullmatch(r"[0-9]+", limit), "INVALID_QOS_SUBMIT_LIMIT"
        )
        maximum = None if limit in {"", "UNLIMITED"} else int(limit)
        job_ids = []
        for line in queue["stdout"].splitlines():
            values = [value.strip() for value in line.split("|")]
            require(len(values) == 4 and all(values), "INVALID_CAPACITY_QUEUE_ROW")
            job_id, owner, qos, state = values
            require(owner == permission["owner"], "CAPACITY_QUEUE_OWNER_MISMATCH")
            # Slurm can retain terminal jobs in its short-lived squeue cache.
            # Those no longer consume MaxSubmitJobsPU, even with --states=all.
            if qos == permission["qos"] and state.split()[0].rstrip("+") not in TERMINAL:
                require(job_id not in job_ids, "DUPLICATE_CAPACITY_QUEUE_JOB")
                job_ids.append(job_id)
        return {
            "owner": permission["owner"],
            "qos": permission["qos"],
            "max_submit_jobs_per_user": maximum,
            "queued_job_ids": job_ids,
            "receipts": receipts,
        }

    def observe(self, attempt, permission):
        queue = self._run(
            ["squeue", "--noheader", "--user", permission["owner"], "--format=%i|%j|%u|%T|%k|%r"]
        )
        fields = ",".join(f + ("%256" if f in {"JobName", "Comment"} else "") for f in SACCT_FIELDS)
        command = [
            "sacct",
            "--duplicates",
            "--noheader",
            "--parsable2",
            "--starttime",
            datetime.fromisoformat(attempt["created_at"])
            .astimezone(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S"),
            "--format",
            fields,
        ]
        if attempt.get("job_id"):
            command += ["--jobs", attempt["job_id"]]
        else:
            command += ["--user", permission["owner"], "--name", attempt["job_name"]]
        accounting = self._run(command)
        if queue["returncode"] != 0 or accounting["returncode"] != 0:
            raise RuntimeError(
                "SLURM_OBSERVATION_FAILED:" + json.dumps({"squeue": queue, "sacct": accounting})
            )
        queue_rows = []
        for line in queue["stdout"].splitlines():
            values = line.split("|")
            require(len(values) == 6, "INVALID_SQUEUE_ROW")
            job_id, name, user, state, comment, reason = values
            if (
                name == attempt["job_name"]
                and comment == attempt["comment"]
                and user == permission["owner"]
            ):
                queue_rows.append({"job_id": job_id, "state": state, "reason": reason})
        accounting_rows = []
        for line in accounting["stdout"].splitlines():
            values = line.split("|")
            if len(values) == len(SACCT_FIELDS) + 1 and values[-1] == "":
                values.pop()
            require(len(values) == len(SACCT_FIELDS), "INVALID_SACCT_ROW")
            row = dict(zip(SACCT_FIELDS, values, strict=True))
            if "." in row["JobIDRaw"]:
                continue
            if attempt.get("job_id") == row["JobIDRaw"]:
                require(
                    row["JobName"] == attempt["job_name"]
                    and row["Comment"] in {"", attempt["comment"]},
                    "SLURM_IDENTITY_MISMATCH",
                )
            if row["JobName"] == attempt["job_name"] and (
                row["Comment"] == attempt["comment"]
                or (row["Comment"] == "" and attempt.get("job_id") == row["JobIDRaw"])
            ):
                require(row["User"] == permission["owner"], "SLURM_OWNER_MISMATCH")
                accounting_rows.append(row)
        return {
            "queue": queue_rows,
            "accounting": accounting_rows,
            "receipts": {"squeue": queue, "sacct": accounting},
        }


def summarize_accounting(rows, attempt, permission, gpus):
    require(rows, "MISSING_ACCOUNTING")
    seen, total = set(), 0
    for row in rows:
        require(
            row["JobIDRaw"] == attempt["job_id"]
            and row["JobName"] == attempt["job_name"]
            and row["Comment"] in {"", attempt["comment"]}
            and row["User"] == permission["owner"]
            and row["Account"] == permission["account"]
            and row["QOS"] == permission["qos"],
            "ACCOUNTING_IDENTITY_MISMATCH",
        )
        require(digest(row) not in seen, "DUPLICATE_ACCOUNTING_EPOCH")
        seen.add(digest(row))
        state = row["State"].split()[0].rstrip("+")
        require(state in TERMINAL, "ACCOUNTING_NOT_TERMINAL")
        require(re.fullmatch(r"[0-9]+", row["ElapsedRaw"]), "INVALID_ELAPSED")
        elapsed = int(row["ElapsedRaw"])
        tres = dict(item.split("=", 1) for item in row["AllocTRES"].split(",") if "=" in item)
        allocated = int(tres.get("gres/gpu", "0"))
        if elapsed:
            require(allocated == gpus, "GPU_ALLOCATION_MISMATCH")
            if gpus:
                require(tres.get("gres/gpu:pro6000") == str(gpus), "UNVERIFIED_GPU_TYPE")
            start, end = (datetime.fromisoformat(row[key]) for key in ("Start", "End"))
            require(
                abs((end - start).total_seconds() - elapsed) <= 2, "INVALID_ALLOCATION_INTERVAL"
            )
        else:
            require(allocated in {0, gpus}, "GPU_ALLOCATION_MISMATCH")
        total += elapsed * allocated
    last = max(rows, key=lambda row: (row["End"], row["Start"]))
    return {
        "gpu_seconds": total,
        "gpu_hours": total / 3600,
        "epochs": len(rows),
        "terminal_state": last["State"].split()[0].rstrip("+"),
        "exit_code": last["ExitCode"],
        "comment_recorded_by_sacct": all(bool(row["Comment"]) for row in rows),
    }


class Scheduler:
    def __init__(
        self,
        plan,
        root,
        *,
        code_root,
        python,
        lease_minutes,
        engine_lease_minutes=720,
        measurement_shards=1,
        backend=None,
        verify_execution=None,
    ):
        self.plan, self.root = Path(plan).resolve(), Path(root).resolve()
        self.code_root, self.python = Path(code_root).resolve(), Path(python).resolve()
        require(type(lease_minutes) is int and lease_minutes > 0, "INVALID_ALLOCATION_LEASE")
        require(
            type(engine_lease_minutes) is int and engine_lease_minutes > 0,
            "INVALID_ENGINE_ALLOCATION_LEASE",
        )
        self.lease_minutes, self.shards = lease_minutes, measurement_shards
        self.engine_lease_minutes = engine_lease_minutes
        self.backend = backend or SlurmBackend()
        if verify_execution is None:
            from .common import verify_execution
        self.verify_execution = verify_execution
        self.folder = self.root / "orchestration"
        self.state = None
        self.registration = None
        self.sequence, self.previous = 0, None
        self._journal_signature = None

    def _identity(self):
        self.verify_execution(self.plan, self.root, require_engine=False)
        plan = read_json(self.plan)
        require(plan["plan_id"] == PLAN_ID, "WRONG_PLAN")
        resource = plan["resource"]
        require(
            resource["gpu_hours_policy"] == "ACCOUNTING_ONLY"
            and resource["max_allocated_gpu_hours"] is None
            and resource["max_wallclock_hours_for_study"] is None
            and resource["max_gpus_concurrent"] == 5
            and resource["gpus_per_worker"] == 1
            and resource["gpu_hour_triggered_stop"] is False,
            "RESOURCE_CONTRACT_MISMATCH",
        )
        matrix_path = self.plan.parent / "run_matrix.json"
        permission = read_json(self.root / PERMISSION_PATH)
        validate_permission(permission, self.root)
        tasks = tasks_for(read_json(matrix_path), self.shards)
        sources = {}
        for task in tasks.values():
            path = self.code_root / "scripts/mm_dev" / task["script"]
            sources[str(path.relative_to(self.code_root))] = file_hash(path)
        return {
            "schema_version": 1,
            "plan_id": PLAN_ID,
            "plan_path": str(self.plan),
            "plan_sha256": file_hash(self.plan),
            "matrix_sha256": file_hash(matrix_path),
            "freeze_sha256": file_hash(self.root / "manifests/F2_FREEZE.json"),
            "permission_receipt_sha256": file_hash(self.root / PERMISSION_PATH),
            "permission": permission,
            "tasks": tasks,
            "task_order": list(tasks),
            "code_root": str(self.code_root),
            "python": str(self.python),
            "lease_minutes": self.lease_minutes,
            "engine_lease_minutes": self.engine_lease_minutes,
            "measurement_shards": self.shards,
            "source_hashes": sources,
        }

    def _load(self):
        identity = self._identity()
        registry = self.folder / "REGISTRATION.json"
        if registry.exists():
            self.registration = read_json(registry)
            require(self.registration == identity, "IMMUTABLE_REGISTRATION_CHANGED")
        else:
            atomic_json(registry, identity, exclusive=True)
            self.registration = identity
        journal = self.folder / "journal.jsonl"
        if self.state is not None and journal.exists():
            info = journal.stat()
            signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            if signature == self._journal_signature:
                # This process has already fsynced and verified this exact prefix.
                # New processes or any externally changed journal replay the chain.
                self._projections()
                return
        self.sequence, self.previous, self.state = 0, None, None
        if journal.exists():
            with journal.open("rb") as stream:
                for line in stream:
                    require(line.endswith(b"\n"), "JOURNAL_TRUNCATED_REQUIRES_REPAIR")
                    row = json.loads(line)
                    row_hash = row.pop("sha256")
                    require(
                        row_hash == digest(row)
                        and row["previous"] == self.previous
                        and row["sequence"] == self.sequence + 1,
                        "JOURNAL_INTEGRITY_FAILURE",
                    )
                    require(
                        row["registration_hash"] == digest(self.registration),
                        "JOURNAL_IDENTITY_FAILURE",
                    )
                    self.sequence, self.previous, self.state = (
                        row["sequence"],
                        row_hash,
                        row["state"],
                    )
        if self.state is None:
            self.state = {
                "phase": "F2_FROZEN",
                "tasks": {key: {"status": "WAITING", "attempts": []} for key in identity["tasks"]},
            }
            self._save("REGISTERED_FIXED_MATRIX")
        else:
            self._projections()
            info = journal.stat()
            self._journal_signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)

    def _save(self, event, **details):
        row = {
            "sequence": self.sequence + 1,
            "previous": self.previous,
            "event": event,
            "details": details,
            "time": now(),
            "registration_hash": digest(self.registration),
            "state": copy.deepcopy(self.state),
        }
        row["sha256"] = digest(row)
        with (self.folder / "journal.jsonl").open("ab") as stream:
            stream.write(encoded(row))
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence, self.previous = row["sequence"], row["sha256"]
        self._projections()
        info = (self.folder / "journal.jsonl").stat()
        self._journal_signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)

    def _projections(self):
        atomic_json(self.folder / "STATE.json", self.state)
        permission = self.registration["permission"]
        allocations = []
        for task_id, task_state in self.state["tasks"].items():
            for attempt in task_state["attempts"]:
                if attempt.get("job_id"):
                    allocations.append(
                        {
                            "task_id": task_id,
                            "attempt_id": attempt["attempt_id"],
                            "job_id": attempt["job_id"],
                            "owner": permission["owner"],
                            "job_name": attempt["job_name"],
                            "comment": attempt["comment"],
                            "account": permission["account"],
                            "qos": permission["qos"],
                            "gres": permission["gres"]
                            if self.registration["tasks"][task_id]["gpus"]
                            else None,
                            "gpus": self.registration["tasks"][task_id]["gpus"],
                            # Controller connectivity uncertainty does not revoke a bound
                            # worker's permission; common.py still verifies live Slurm.
                            "status": "REGISTERED"
                            if attempt["status"] == "UNKNOWN" and "accounting" not in attempt
                            else attempt["status"],
                            "controller_status": attempt["status"],
                            "attempt_manifest_path": attempt["manifest_path"],
                            "actual_gpu_seconds": attempt.get("accounting", {}).get("gpu_seconds"),
                        }
                    )
        atomic_json(
            self.root / "manifests/ALLOCATIONS.json",
            {
                "schema_version": 1,
                "plan_id": PLAN_ID,
                "registration_hash": digest(self.registration),
                "permission_receipt_sha256": self.registration["permission_receipt_sha256"],
                "allocations": allocations,
            },
        )

    def _marker_valid(self, task_id, attempt, checkpoint=False):
        relative = (
            f"checkpoints/{attempt['attempt_id']}.json"
            if checkpoint
            else f"completions/{task_id}.json"
        )
        path = self.folder / relative
        if not path.exists():
            return False
        marker = read_json(path)
        require(
            marker["plan_id"] == PLAN_ID
            and marker["task_id"] == task_id
            and marker["attempt_id"] == attempt["attempt_id"]
            and marker["registration_hash"] == digest(self.registration),
            "COMPLETION_IDENTITY_MISMATCH",
        )
        require(
            marker["status"] == ("CHECKPOINTED" if checkpoint else "COMPLETE"),
            "INVALID_COMPLETION_STATUS",
        )
        if checkpoint:
            require(marker["reason"] in {"PREEMPTION", "TIME_LEASE_END"}, "INVALID_RESUME_REASON")
        require(marker["artifacts"], "MISSING_COMPLETION_ARTIFACTS")
        for relative, descriptor in marker["artifacts"].items():
            artifact = contained(self.root, relative)
            require(
                artifact.is_file()
                and artifact.stat().st_size == descriptor["bytes"]
                and file_hash(artifact) == descriptor["sha256"],
                "COMPLETION_ARTIFACT_CHANGED",
            )
        return True

    def _release(self, task_id, attempt):
        try:
            receipt = self.backend.release(attempt["job_id"])
            attempt["release_receipt"] = receipt
            if receipt["returncode"] != 0:
                raise RuntimeError("RELEASE_RESULT_UNKNOWN")
            attempt["released"] = True
            self._save("ALLOCATION_RELEASED", task_id=task_id)
        except Exception as exc:
            attempt["status"] = self.state["tasks"][task_id]["status"] = "UNKNOWN"
            self._save("RELEASE_UNKNOWN", task_id=task_id, error=repr(exc))

    def _observe(self, task_id):
        task_state = self.state["tasks"][task_id]
        attempt = task_state["attempts"][-1]
        try:
            observation = self.backend.observe(attempt, self.registration["permission"])
            # Persist raw scheduler evidence, including unsuccessful/empty observations.
            relative = f"observations/{attempt['attempt_id']}/{self.sequence + 1:08d}.json"
            atomic_json(self.folder / relative, observation, exclusive=True)
            attempt["observation"] = {
                "path": "orchestration/" + relative,
                "sha256": file_hash(self.folder / relative),
            }
            ids = {row["job_id"] for row in observation["queue"]}
            ids |= {row["JobIDRaw"] for row in observation["accounting"]}
            require(len(ids) <= 1, "MULTIPLE_JOBS_MATCH_SUBMISSION_INTENT")
            if not ids:
                raise RuntimeError("NO_POSITIVE_SLURM_IDENTITY")
            job_id = next(iter(ids))
            require(re.fullmatch(r"[0-9]+", job_id), "INVALID_SLURM_JOB_ID")
            require(attempt.get("job_id") in (None, job_id), "SLURM_JOB_IDENTITY_CHANGED")
            if not attempt.get("job_id"):
                attempt["job_id"] = job_id
                attempt["status"] = task_state["status"] = "REGISTERED"
                self._save("AMBIGUOUS_SUBMISSION_RECONCILED", task_id=task_id, job_id=job_id)
            if observation["queue"]:
                attempt["status"] = task_state["status"] = "ACTIVE"
                self._save("ALLOCATION_ACTIVE", task_id=task_id)
                if any(row["reason"] == "JobHeldUser" for row in observation["queue"]):
                    self._release(task_id, attempt)
                return
            if not observation["accounting"] or any(
                row["State"].split()[0].rstrip("+") not in TERMINAL
                for row in observation["accounting"]
            ):
                raise RuntimeError("ACCOUNTING_NOT_YET_TERMINAL")
            accounting = summarize_accounting(
                observation["accounting"],
                attempt,
                self.registration["permission"],
                self.registration["tasks"][task_id]["gpus"],
            )
            attempt["accounting"] = accounting
            terminal = accounting["terminal_state"]
            # A scheduler terminal record does not override a live process lease.
            with worker_lease(self.root, task_id):
                complete = self._marker_valid(task_id, attempt)
                checkpointed = self._marker_valid(task_id, attempt, checkpoint=True)
            failure_path = self.folder / "failures" / (attempt["attempt_id"] + ".json")
            if failure_path.exists():
                failure = read_json(failure_path)
                require(
                    failure["plan_id"] == PLAN_ID
                    and failure["task_id"] == task_id
                    and failure["attempt_id"] == attempt["attempt_id"]
                    and failure["registration_hash"] == digest(self.registration),
                    "FAILURE_MARKER_IDENTITY_MISMATCH",
                )
                attempt["status"], task_state["status"] = terminal, "BLOCKED"
                task_state["blocker"] = failure["error"]
            elif complete and terminal == "COMPLETED" and accounting["exit_code"] == "0:0":
                attempt["status"], task_state["status"] = terminal, "COMPLETE"
                task_state["completion_marker_sha256"] = file_hash(
                    self.folder / "completions" / (task_id + ".json")
                )
            elif terminal in AUTO_RESUME or (
                terminal == "COMPLETED" and accounting["exit_code"] == "0:0" and checkpointed
            ):
                # A complete marker plus a non-successful scheduler state requires review;
                # do not rerun already complete work merely to erase the failed allocation.
                require(not complete, "COMPLETED_ARTIFACT_WITH_UNSUCCESSFUL_ALLOCATION")
                attempt["status"] = terminal
                if task_id == "ENGINE_F2":
                    task_state["status"] = "BLOCKED"
                    task_state["blocker"] = "ENGINE_CONTINUOUS_TRACE_REQUIRES_EXPLICIT_REPAIR"
                else:
                    task_state["status"] = "RETRYABLE"
            else:
                attempt["status"], task_state["status"] = terminal, "BLOCKED"
                task_state["blocker"] = "TECHNICAL_TERMINAL_OR_MISSING_COMPLETE_MARKER"
            self._save("ALLOCATION_TERMINAL", task_id=task_id)
        except (ValueError, KeyError, TypeError) as exc:
            attempt["status"] = task_state["status"] = "BLOCKED"
            task_state["blocker"] = repr(exc)
            self._save("TECHNICAL_IDENTITY_BLOCK", task_id=task_id, error=repr(exc))
        except Exception as exc:
            attempt["status"] = task_state["status"] = "UNKNOWN"
            self._save("OBSERVATION_UNKNOWN", task_id=task_id, error=repr(exc))

    def _submission_slot_available(self, task_id):
        permission = self.registration["permission"]
        evidence = {"time": now(), "task_id": task_id}
        try:
            capacity = self.backend.submission_capacity(permission)
            evidence["observation"] = capacity
            maximum, queued = capacity["max_submit_jobs_per_user"], capacity["queued_job_ids"]
            require(
                capacity["owner"] == permission["owner"]
                and capacity["qos"] == permission["qos"]
                and (maximum is None or (type(maximum) is int and maximum >= 0))
                and isinstance(queued, list)
                and all(isinstance(j, str) and j for j in queued)
                and len(queued) == len(set(queued)),
                "INVALID_SUBMISSION_CAPACITY",
            )
            # A just-submitted or ambiguous job may not appear in squeue yet. Preserve
            # its slot until positive accounting, without double-counting visible IDs.
            reserved = [
                attempt["attempt_id"]
                for task in self.state["tasks"].values()
                for attempt in task["attempts"]
                if "accounting" not in attempt and attempt.get("job_id") not in queued
            ]
            used = len(queued) + len(reserved)
            available = maximum is None or used < maximum
            summary = {
                "status": "AVAILABLE" if available else "WAITING",
                "owner": permission["owner"],
                "qos": permission["qos"],
                "max_submit_jobs_per_user": maximum,
                "queued_jobs": len(queued),
                "reserved_attempt_ids": reserved,
                "available_slots": None if maximum is None else max(0, maximum - used),
            }
        except Exception as exc:
            available = False
            summary = {"status": "UNKNOWN", "error": repr(exc)}
        evidence["summary"] = summary
        relative = f"observations/capacity/{self.sequence + 1:08d}.json"
        atomic_json(self.folder / relative, evidence, exclusive=True)
        self.state["submission_capacity"] = {
            **summary,
            "time": evidence["time"],
            "observation": {
                "path": "orchestration/" + relative,
                "sha256": file_hash(self.folder / relative),
            },
        }
        self._save("SUBMISSION_CAPACITY_" + summary["status"], task_id=task_id)
        return available

    def _submit(self, task_id):
        task_state, task = self.state["tasks"][task_id], self.registration["tasks"][task_id]
        require(task_state["status"] in {"WAITING", "RETRYABLE"}, "DUPLICATE_SUBMISSION_GUARD")
        if task_id != "ENGINE_F2":
            self.verify_execution(self.plan, self.root, require_engine=True)
        if not self._submission_slot_available(task_id):
            return False
        attempt_id = f"{task_id}_attempt{len(task_state['attempts']):04d}"
        registration_hash = digest(self.registration)
        prefix = registration_hash[:12]
        attempt = {
            "attempt_id": attempt_id,
            "created_at": now(),
            "status": "SUBMITTING",
            "job_id": None,
            "job_name": f"mmdev-{prefix}-{attempt_id}",
            "comment": f"mmdev:{prefix}:{attempt_id}",
            "manifest_path": f"orchestration/attempts/{attempt_id}.json",
        }
        permission = self.registration["permission"]
        worker = [
            str(self.python),
            str(self.code_root / "scripts/mm_dev" / task["script"]),
            "--plan",
            str(self.plan),
            "--run-root",
            str(self.root),
            *task["arguments"],
        ]
        environment = {
            "PYTHONPATH": str(self.code_root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
            "MM_DEV_TASK_ID": task_id,
            "MM_DEV_ATTEMPT_ID": attempt_id,
            "MM_DEV_REGISTRATION_HASH": registration_hash,
        }
        script = "#!/bin/bash\nset -euo pipefail\n" + "\n".join(
            f"export {key}={shlex.quote(value)}" for key, value in environment.items()
        )
        script += "\ncd " + shlex.quote(str(self.code_root)) + "\nexec " + shlex.join(worker) + "\n"
        script_path = self.folder / "attempts" / (attempt_id + ".sh")
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with script_path.open("x") as stream:
            stream.write(script)
            stream.flush()
            os.fsync(stream.fileno())
        command = [
            "sbatch",
            "--hold",
            "--no-requeue",
            "--nodes=1",
            "--cpus-per-task=4",
            "--mem=64G",
            "--parsable",
            "--account",
            permission["account"],
            "--qos",
            permission["qos"],
            "--job-name",
            attempt["job_name"],
            "--comment",
            attempt["comment"],
            "--time",
            str(self.engine_lease_minutes if task_id == "ENGINE_F2" else self.lease_minutes),
            "--signal=B:USR1@600",
            "--output",
            str(self.folder / "attempts" / (attempt_id + "_%j.log")),
        ]
        if task["gpus"]:
            command += ["--gres", permission["gres"]]
        if permission.get("partition"):
            command += ["--partition", permission["partition"]]
        command.append(str(script_path))
        attempt["command"] = command
        atomic_json(
            self.root / attempt["manifest_path"],
            {
                **attempt,
                "task_id": task_id,
                "plan_id": PLAN_ID,
                "registration_hash": registration_hash,
                "script_sha256": file_hash(script_path),
                "worker_command": worker,
            },
            exclusive=True,
        )
        task_state["attempts"].append(attempt)
        task_state["status"] = "SUBMITTING"
        self._save("SUBMISSION_INTENT_DURABLE", task_id=task_id, attempt_id=attempt_id)
        try:
            receipt = self.backend.submit(command)
            attempt["submission_receipt"] = receipt
            require(receipt["returncode"] == 0, "SBATCH_RESULT_NOT_SUCCESSFUL")
            match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", receipt["stdout"])
            require(match, "AMBIGUOUS_SBATCH_STDOUT")
            attempt["job_id"] = match.group(1)
            attempt["status"] = task_state["status"] = "REGISTERED"
            self._save("ALLOCATION_BOUND_BEFORE_RELEASE", task_id=task_id)
            self._release(task_id, attempt)
        except Exception as exc:
            attempt["status"] = task_state["status"] = "UNKNOWN"
            self._save("SUBMISSION_UNKNOWN_NO_RETRY", task_id=task_id, error=repr(exc))
        return True

    def _phase(self):
        tasks = self.state["tasks"]

        def done(phase):
            return all(
                tasks[key]["status"] == "COMPLETE"
                for key, task in self.registration["tasks"].items()
                if task["phase"] == phase
            )

        phase = "F2_FROZEN"
        for group, value in (
            ("ENGINE", "ENGINE_VERIFIED"),
            ("PREP", "PREP_COMPLETE"),
            ("PROBE", "PROBE_COMPLETE"),
            ("CONTINUE", "RESPONSES_COMPLETE"),
            ("EVAL", "EVAL_COMPLETE"),
            ("ANALYZE", "ANALYZED"),
            ("RELEASE", "RELEASED"),
        ):
            if not done(group):
                break
            phase = value
        self.state["phase"] = phase
        self.state["technical_blockers"] = [
            key for key, state in tasks.items() if state["status"] == "BLOCKED"
        ]
        attempts = [a for task in tasks.values() for a in task["attempts"]]
        self.state["accounting"] = {
            "policy": "ACCOUNTING_ONLY",
            "max_allocated_gpu_hours": None,
            "max_wallclock_hours_for_study": None,
            "known_actual_gpu_seconds": sum(
                a.get("accounting", {}).get("gpu_seconds", 0) for a in attempts
            ),
            "unresolved_attempt_ids": [a["attempt_id"] for a in attempts if "accounting" not in a],
        }

    def _final_gpu_accounting(self):
        allocations = []
        gpu_tasks = [
            key
            for key in self.registration["task_order"]
            if self.registration["tasks"][key]["gpus"]
        ]
        for task_id in gpu_tasks:
            task = self.state["tasks"][task_id]
            require(task["status"] == "COMPLETE", "GPU_TASK_INCOMPLETE")
            require(
                file_hash(self.folder / "completions" / (task_id + ".json"))
                == task["completion_marker_sha256"],
                "COMPLETION_MARKER_CHANGED",
            )
            require(self._marker_valid(task_id, task["attempts"][-1]), "GPU_COMPLETION_CHANGED")
            for attempt in task["attempts"]:
                require("accounting" in attempt, "GPU_COST_UNRESOLVED")
                observation = attempt["observation"]
                require(
                    file_hash(contained(self.root, observation["path"])) == observation["sha256"],
                    "SCHEDULER_OBSERVATION_CHANGED",
                )
                allocations.append(
                    {
                        "task_id": task_id,
                        "attempt_id": attempt["attempt_id"],
                        "job_id": attempt["job_id"],
                        **attempt["accounting"],
                        "scheduler_observation": observation,
                    }
                )
        total = sum(a["gpu_seconds"] for a in allocations)
        receipt = {
            "schema_version": 1,
            "plan_id": PLAN_ID,
            "registration_hash": digest(self.registration),
            "status": "COMPLETE",
            "policy": "ACCOUNTING_ONLY",
            "actual_gpu_seconds": total,
            "actual_gpu_hours": total / 3600,
            "allocations": allocations,
            "gpu_task_count": len(gpu_tasks),
        }
        path = self.root / "manifests/GPU_ACCOUNTING_FINAL.json"
        if path.exists():
            require(read_json(path) == receipt, "FINAL_GPU_ACCOUNTING_CHANGED")
        else:
            atomic_json(path, receipt, exclusive=True)

    def tick(self):
        with process_lease(self.folder / "controller.lock"):
            self._load()
            for task_id, state in self.state["tasks"].items():
                if state["status"] in LIVE:
                    self._observe(task_id)
            for task_id in self.registration["task_order"]:
                task = self.registration["tasks"][task_id]
                state = self.state["tasks"][task_id]
                if state["status"] not in {"WAITING", "RETRYABLE"}:
                    continue
                if not all(
                    self.state["tasks"][key]["status"] == "COMPLETE" for key in task["dependencies"]
                ):
                    continue
                used = sum(
                    self.registration["tasks"][key]["gpus"]
                    for key, item in self.state["tasks"].items()
                    if item["status"] in LIVE
                )
                # Unknown submissions and unresolved blocked allocations remain reserved.
                used += sum(
                    self.registration["tasks"][key]["gpus"]
                    for key, item in self.state["tasks"].items()
                    if item["status"] == "BLOCKED"
                    and item["attempts"]
                    and "accounting" not in item["attempts"][-1]
                )
                if used + task["gpus"] > 5:
                    continue
                if task["phase"] in {"ANALYZE", "RELEASE"} and any(
                    "accounting" not in a
                    for item in self.state["tasks"].values()
                    for a in item["attempts"]
                ):
                    continue
                if task["phase"] in {"ANALYZE", "RELEASE"}:
                    self._final_gpu_accounting()
                if not self._submit(task_id):
                    # Every task uses the same registered QoS; defer until the next
                    # fresh observation, keeping the scientific task state unchanged.
                    break
            self._phase()
            self._save("TICK_FINISHED")
            return copy.deepcopy(self.state)
