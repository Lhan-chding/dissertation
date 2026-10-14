"""Five-GPU recovery with immutable lineage and no retry of ambiguous work.

Only an independently audited allowlist can authorize an original frozen worker.
The old queue is read-only; new claims, process receipts and incidents live below
``recovery/<id>``. Completion here does not release scientific scores.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import shlex
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from sr_f2.orchestration import (
    Slurm as FrozenSlurm,
)
from sr_f2.orchestration import (
    controller_lease,
    global_task_matrix,
    queue_lease,
    safe_id,
    task_id,
    verify_task,
    visible_devices,
)
from sr_f2.protocol import verify_registration
from sr_f12.orchestration import (
    file_hash,
    gpu_count,
    memory_gib,
    now,
    read_json,
    save_json,
)
from sr_f12.protocol import TEACHER_QOS, object_hash

RUN_GPU_LIMIT = 5
TEACHER_GPU_LIMIT = 8
SUBMIT_LIMIT = 5
POOL_SPECS = {"RECOVERY_POOL_1": 2, "RECOVERY_POOL_2": 2, "RECOVERY_POOL_3": 1}
AUDIT_STATES = frozenset(
    {
        "INHERITED_COMPLETE",
        "RESUME_ALLOWED",
        "NEVER_ATTEMPTED",
        "BLOCKED_INCOMPLETE_GENERATION",
    }
)
ACTIVE = frozenset({"CLAIMED", "RUNNING"})


def within(root, relative):
    relative = Path(relative)
    target = (root / relative).resolve(strict=True)
    if relative.is_absolute() or ".." in relative.parts or root not in target.parents:
        raise PermissionError("Evidence path escapes its bound root")
    return target


class Slurm(FrozenSlurm):
    def queue(self, user):
        """All teacher reservations plus all this user's submissions, without overlap."""
        ids = set()
        for selector in ("--qos=" + TEACHER_QOS, "--user=" + user):
            result = self.run(["squeue", "--noheader", selector, "--format=%i"])
            if result.returncode:
                raise RuntimeError("Cannot establish reservation capacity: " + result.stderr)
            values = result.stdout.split()
            if len(values) != len(set(values)) or any(not job.isdigit() for job in values):
                raise PermissionError("Unknown queue reservation identifier")
            ids.update(values)
        return {job: self.show(job) for job in sorted(ids)}

    def terminal(self, job):
        result = self.run(
            [
                "sacct",
                "-X",
                "-D",
                "-n",
                "-P",
                "-j",
                str(job),
                "--format=JobIDRaw,State%60,ExitCode,User%100,Account%100,QOS%100,JobName%100,End",
            ]
        )
        if result.returncode:
            raise RuntimeError("Terminal accounting unavailable")
        rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
        rows = [row for row in rows if row[0] == str(job)]
        if not rows or any(len(row) < 8 for row in rows):
            return None
        # Requeued CPU allocations retain multiple accounting rows. The latest
        # terminal end is the actual final allocation; duplicate latest rows fail closed.
        rows.sort(key=lambda row: row[7])
        row = rows[-1]
        if len(rows) > 1 and rows[-2][7] == row[7]:
            return None
        return dict(
            job_id=row[0],
            state=row[1].split()[0].rstrip("+"),
            exit_code=row[2],
            user=row[3],
            account=row[4],
            qos=row[5],
            name=row[6],
            end=row[7],
        )


def capacity(queue, attempts, user):
    teacher_gpus = user_jobs = run_gpus = 0
    known = {}
    for state in attempts:
        count = state.get("gpu_count")
        if count != POOL_SPECS.get(state.get("pool_id")):
            raise PermissionError("Unknown recovery reservation size")
        if state.get("job_id"):
            if state["job_id"] in known:
                raise PermissionError("Duplicate bound recovery allocation")
            known[state["job_id"]] = state
    for job, row in queue.items():
        owner = row.get("UserId", "").split("(")[0]
        if not owner or not row.get("ReqTRES") or not row.get("QOS"):
            raise PermissionError("Unknown queue owner, QoS or reservation")
        count = gpu_count(row["ReqTRES"])
        teacher_gpus += count if row["QOS"] == TEACHER_QOS else 0
        user_jobs += owner == user
        if job in known:
            if count != known[job]["gpu_count"] or owner != user or row["QOS"] != TEACHER_QOS:
                raise PermissionError("Bound recovery reservation identity changed")
            run_gpus += count
    for state in attempts:
        if state.get("job_id") not in queue:
            # Transport errors, orphan intents and missing accounting reserve all capacity.
            teacher_gpus += state["gpu_count"]
            run_gpus += state["gpu_count"]
            user_jobs += 1
    return dict(teacher_gpus=teacher_gpus, user_submissions=user_jobs, run_gpus=run_gpus)


def verify_allocation(fields, intent, user, *, held=False):
    expected = dict(
        Account="rose",
        Partition="cluster02",
        QOS=TEACHER_QOS,
        JobName=intent["job_name"],
        Comment=intent["comment"],
        Command=intent["script"],
        JobId=intent["job_id"],
    )
    if any(fields.get(key) != value for key, value in expected.items()):
        raise PermissionError("Recovery allocation identity differs")
    if fields.get("UserId", "").split("(")[0] != user:
        raise PermissionError("Recovery allocation owner differs")
    count = intent["gpu_count"]
    tres = fields.get("ReqTRES", "")
    if (
        count != POOL_SPECS.get(intent["pool_id"])
        or gpu_count(tres) != count
        or not re.search(rf"(?:^|,)gres/gpu:pro6000={count}(?:,|$)", tres)
    ):
        raise PermissionError("Recovery allocation must use its exact PRO6000 count")
    if memory_gib(tres) < count * 80 or "highmem" not in fields.get("Features", "").split("&"):
        raise PermissionError("Recovery allocation lacks frozen per-worker memory resources")
    if held and (fields.get("JobState"), fields.get("Reason")) != ("PENDING", "JobHeldUser"):
        raise PermissionError("New recovery allocation is not held")
    if not held and fields.get("JobState") != "RUNNING":
        raise PermissionError("Recovery worker allocation is not running")


class Recovery:
    def __init__(self, plan_path):
        self.path = Path(plan_path).resolve(strict=True)
        self.plan = read_json(self.path)
        self.sha256 = object_hash(self.plan)
        self.root = Path(self.plan["original_root"]).resolve(strict=True)
        self.directory = self.path.parent
        self.code = Path(self.plan["frozen_code_root"]).resolve(strict=True)
        self.extension = Path(self.plan["extension_root"]).resolve(strict=True)
        expected = self.root / "recovery" / safe_id(self.plan["recovery_id"]) / "PLAN.json"
        if self.path != expected or self.plan.get("max_run_gpus") != RUN_GPU_LIMIT:
            raise PermissionError("Recovery location or five-GPU limit differs")
        if self.code == self.extension or self.code in self.extension.parents:
            raise PermissionError("Recovery extension must live outside the frozen source tree")
        if self.plan.get("original_task_matrix") != global_task_matrix():
            raise PermissionError("Recovery matrix differs from the original 53 tasks")
        if len(self.plan.get("old_cancelled_jobs", [])) != 5:
            raise PermissionError("Recovery requires all five original cancellation identities")
        self.matrix = self.plan["original_task_matrix"]
        self.tasks = {task_id(task): task for task in self.matrix}
        self.audit_path = within(self.directory, self.plan["audit_path"])
        self.audit = read_json(self.audit_path)
        if self.audit.get("registration_sha256") != self.plan["registration_sha256"] or (
            self.audit.get("original_root") != str(self.root)
        ):
            raise PermissionError("Recovery audit registration identity differs")
        self.allowed = self.audit.get("tasks", {})
        if set(self.allowed) != set(self.tasks):
            raise PermissionError("Recovery audit must classify every original task exactly once")
        for key, row in self.allowed.items():
            if row.get("task") != self.tasks[key] or row.get("status") not in AUDIT_STATES:
                raise PermissionError("Unknown allowlist task or classification")
            refs = row.get("reference_files")
            if not isinstance(refs, list) or len({r["path"] for r in refs}) != len(refs):
                raise PermissionError("Invalid recovery lineage references")
            if row["status"] in {"INHERITED_COMPLETE", "RESUME_ALLOWED"} and not refs:
                raise PermissionError("Attempted work requires immutable original lineage")
            if row["status"] == "BLOCKED_INCOMPLETE_GENERATION" and not row.get("reason"):
                raise PermissionError("Blocked work must retain its technical reason")
            if not isinstance(row.get("mutable_prelaunch_files", []), list) or not isinstance(
                row.get("absent_prelaunch_paths", []), list
            ):
                raise PermissionError("Prelaunch evidence guards must be lists")

    def task_prelaunch(self, key):
        """Check mutable baselines only before this task's single recovery launch."""
        row = self.allowed[key]
        if row["status"] not in {"RESUME_ALLOWED", "NEVER_ATTEMPTED"}:
            raise PermissionError("Task is not allowlisted for launch")
        for ref in row.get("mutable_prelaunch_files", []):
            if file_hash(within(self.root, ref["path"])) != ref["sha256"]:
                raise PermissionError("Task output changed since its recovery audit")
        for relative in row.get("absent_prelaunch_paths", []):
            relative = Path(relative)
            target = (self.root / relative).resolve(strict=False)
            if relative.is_absolute() or ".." in relative.parts or self.root not in target.parents:
                raise PermissionError("Absent evidence path escapes original root")
            if target.exists() or (self.root / relative).is_symlink():
                raise PermissionError("Previously unattempted output now exists")

    def verify(self):
        if (self.root / "STOP").exists() or (self.directory / "STOP.json").exists():
            raise PermissionError("Recovery stopped; no retry")
        if object_hash(read_json(self.path)) != self.sha256:
            raise PermissionError("Recovery plan changed")
        if file_hash(self.audit_path) != self.plan["audit_sha256"]:
            raise PermissionError("Recovery allowlist changed")
        manifest = self.plan.get("extension_manifest", {})
        required = {
            "src/sr_f2_recovery/__init__.py",
            "src/sr_f2_recovery/orchestration.py",
            "scripts/sr_f2/recovery.py",
        }
        if not required <= set(manifest):
            raise PermissionError("Recovery extension manifest omits executable source")
        for path, sha in manifest.items():
            if file_hash(within(self.extension, path)) != sha:
                raise PermissionError("Recovery extension source changed")
        if Path(__file__).resolve() != self.extension / "src/sr_f2_recovery/orchestration.py":
            raise PermissionError("Loaded recovery module differs from the bound extension")
        import sr_f2.protocol as frozen_protocol

        if Path(frozen_protocol.__file__).resolve() != self.code / "src/sr_f2/protocol.py":
            raise PermissionError("Loaded scientific module differs from frozen source")
        registration = verify_registration(self.root, require_technical=True)
        if object_hash(registration) != self.plan["registration_sha256"] or (
            Path(registration["code_root"]).resolve(strict=True) != self.code
        ):
            raise PermissionError("Frozen registration or scientific source differs")
        if str(Path(self.plan["python"]).resolve(strict=True)) != self.plan["python"]:
            raise PermissionError("Recovery Python must be a canonical existing executable")
        original_config = read_json(self.root / "orchestration/CONFIG.json")
        if Path(original_config["python"]).resolve(strict=True) != Path(self.plan["python"]) or (
            Path(original_config["code_root"]).resolve(strict=True) != self.code
        ):
            raise PermissionError("Recovery runtime differs from the original controller")
        for row in self.allowed.values():
            for ref in row["reference_files"]:
                if file_hash(within(self.root, ref["path"])) != ref["sha256"]:
                    raise PermissionError("Original recovery lineage changed")
        return self.sha256

    def old_jobs_terminal(self, slurm, user):
        live = slurm.queue(user)
        ids = set()
        for expected in self.plan["old_cancelled_jobs"]:
            job = expected.get("job_id", "")
            if not str(job).isdigit() or job in ids or expected.get("user") != user:
                raise PermissionError("Invalid original cancellation identity")
            ids.add(job)
            if job in live:
                raise PermissionError("Original allocation remains live or completing")
            actual = slurm.terminal(job)
            if (
                actual is None
                or actual.get("state") != "CANCELLED"
                or any(
                    actual.get(key) != expected.get(key)
                    for key in ("job_id", "user", "account", "qos", "name")
                )
            ):
                raise PermissionError("Original allocation is not confirmed cancelled")


class RecoveryQueue:
    def __init__(self, recovery):
        self.recovery = recovery
        self.directory = recovery.directory / "queue"

    def states(self):
        result = {}
        for path in (self.directory / "tasks").glob("*/STATE.json"):
            key, row = path.parent.name, read_json(path)
            claim = read_json(path.with_name("CLAIM.json"))
            if (
                key not in self.recovery.tasks
                or row.get("task") != self.recovery.tasks[key]
                or (row.get("status") not in ACTIVE | {"COMPLETE", "FAILED"})
            ):
                raise PermissionError("Recovery task state differs")
            if self.recovery.allowed[key]["status"] not in {"RESUME_ALLOWED", "NEVER_ATTEMPTED"}:
                raise PermissionError("Non-allowlisted task has a recovery attempt")
            if (
                claim.get("plan_sha256") != self.recovery.sha256
                or (claim.get("audit_entry_sha256") != object_hash(self.recovery.allowed[key]))
                or row.get("claim_sha256") != object_hash(claim)
                or any(
                    row.get(k) != claim.get(k)
                    for k in ("task", "pool_id", "device", "job_id", "claim_id")
                )
                or claim.get("pool_id") not in POOL_SPECS
            ):
                raise PermissionError("Recovery claim or lineage changed")
            result[key] = row
        for path in (self.directory / "tasks").glob("*/CLAIM.json"):
            if path.parent.name not in result:
                raise RuntimeError("Orphan recovery claim; launch unknown, no retry")
        return result

    def _snapshot(self, states):
        complete = {
            key
            for key, row in self.recovery.allowed.items()
            if row["status"] == "INHERITED_COMPLETE"
        }
        complete.update(key for key, row in states.items() if row["status"] == "COMPLETE")
        active = {key for key, row in states.items() if row["status"] in ACTIVE}
        blocked = {
            key
            for key, row in self.recovery.allowed.items()
            if row["status"] == "BLOCKED_INCOMPLETE_GENERATION"
        }
        candidates = [
            task
            for task in self.recovery.matrix
            if task_id(task) not in complete | active | blocked | set(states)
        ]
        histories = {task_id(t) for t in self.recovery.matrix if t["kind"] == "history"}
        pilots = {task_id(t) for t in self.recovery.matrix if t["kind"] == "pilot"}
        prerequisites = histories | pilots

        def deps(task):
            base = prerequisites if task["kind"] in {"train", "endpoint"} else set()
            return base | (
                {task_id(dict(kind="train", model_id=task["model_id"]))}
                if task["kind"] == "endpoint"
                else set()
            )

        ready = [task for task in candidates if deps(task) <= complete]
        possible = complete | active
        while True:
            expanded = possible | {task_id(task) for task in candidates if deps(task) <= possible}
            if expanded == possible:
                break
            possible = expanded
        waiting = any(task_id(task) in possible for task in candidates)
        ready.sort(
            key=lambda t: ("history", "pilot", "train", "endpoint", "diagnostic").index(t["kind"])
        )
        failed = (self.recovery.directory / "STOP.json").exists() or any(
            row["status"] == "FAILED" for row in states.values()
        )
        if failed:
            ready = []
        reason = (
            "TECHNICAL_FAILURE"
            if failed
            else None
            if ready
            else (
                "ALL_TASKS_COMPLETE"
                if len(complete) == len(self.recovery.matrix)
                else "WAIT_ACTIVE_PREREQUISITES"
                if waiting and active
                else "ONLY_BLOCKED_DEPENDENCIES_REMAIN"
                if candidates or blocked
                else "NO_UNCLAIMED_WORK"
            )
        )
        return dict(
            ready=ready,
            completed=len(complete),
            active=len(active),
            blocked=len(blocked),
            unclaimed=len(candidates),
            all_complete=len(complete) == len(self.recovery.matrix),
            failed=failed,
            idle_reason=reason,
            may_exit=not ready and not waiting,
        )

    def snapshot(self):
        with queue_lease(self.directory):
            return self._snapshot(self.states())

    def claim(self, pool_id, device, job):
        if pool_id not in POOL_SPECS or not str(job).isdigit():
            raise PermissionError("Invalid recovery allocation owner")
        with queue_lease(self.directory):
            states = self.states()
            snapshot = self._snapshot(states)
            if snapshot["failed"] or (self.recovery.root / "STOP").exists():
                raise RuntimeError("Recovery queue stopped")
            owned = [
                row
                for row in states.values()
                if row["pool_id"] == pool_id and row["status"] in ACTIVE
            ]
            if any(row["device"] == device for row in owned) or len(owned) >= POOL_SPECS[pool_id]:
                raise PermissionError("Recovery GPU already has active work")
            if not snapshot["ready"]:
                return None, snapshot
            task = snapshot["ready"][0]
            key = task_id(task)
            self.recovery.task_prelaunch(key)
            claim = dict(
                task=task,
                pool_id=pool_id,
                device=device,
                job_id=str(job),
                plan_sha256=self.recovery.sha256,
                audit_entry_sha256=object_hash(self.recovery.allowed[key]),
                original_lineage=self.recovery.allowed[key],
                at=now(),
                claim_id=object_hash([task, pool_id, device, str(job), time.time_ns()]),
            )
            path = self.directory / "tasks" / key
            save_json(path / "CLAIM.json", claim, exclusive=True)
            state = dict(claim, status="CLAIMED", claim_sha256=object_hash(claim))
            save_json(path / "STATE.json", state, exclusive=True)
            return state, snapshot

    def transition(self, claim, *, pid=None, result=None, error=None):
        with queue_lease(self.directory):
            key = task_id(claim["task"])
            row = self.states()[key]
            if row["claim_id"] != claim["claim_id"] or row["status"] not in ACTIVE:
                raise PermissionError("Terminal or foreign recovery claim cannot be resumed")
            if error is not None:
                row.update(status="FAILED", error=str(error), finished_at=now())
                self.stop(str(error), claim["pool_id"])
            elif result is not None:
                if {k: result.get(k) for k in ("kind", "model_id")} != claim["task"]:
                    raise PermissionError("Completion is for a different task")
                row.update(status="COMPLETE", result=result, finished_at=now())
            elif type(pid) is int and pid > 0 and row["status"] == "CLAIMED":
                row.update(status="RUNNING", pid=pid, started_at=now())
            else:
                raise ValueError("Expected a new PID or verified terminal result")
            save_json(self.directory / "tasks" / key / "STATE.json", row)

    def stop(self, error, pool_id):
        path = self.recovery.directory / "STOP.json"
        # The first failure is authoritative, and each pool keeps its own receipt.
        if not path.exists():
            try:
                save_json(path, dict(error=str(error), pool_id=pool_id, at=now()), exclusive=True)
            except PermissionError:
                if not path.exists():
                    raise


class Controller:
    def __init__(self, recovery, *, slurm=None, user=None):
        self.recovery = recovery
        self.slurm = slurm or Slurm()
        self.user = user or getpass.getuser()

    def pool_dir(self, pool):
        if pool not in POOL_SPECS:
            raise PermissionError("Unknown recovery pool")
        return self.recovery.directory / "pools" / pool

    def attempts(self):
        values = []
        for pool, count in POOL_SPECS.items():
            path = self.pool_dir(pool)
            if (path / "STATE.json").exists():
                state = read_json(path / "STATE.json")
                if state.get("pool_id") != pool or state.get("gpu_count") != count:
                    raise PermissionError("Recovery pool state identity differs")
                values.append(state)
            elif (path / "INTENT.json").exists():
                values.append(dict(pool_id=pool, gpu_count=count, status="SUBMISSION_UNKNOWN"))
        if any(
            p.parent.name not in POOL_SPECS
            for p in (self.recovery.directory / "pools").glob("*/INTENT.json")
        ):
            raise PermissionError("Unregistered recovery allocation intent")
        return values

    def submit_once(self):
        """One bounded pass; never resubmit an existing or ambiguous intent."""
        recovery = self.recovery
        with controller_lease(recovery.directory):
            recovery.verify()
            recovery.old_jobs_terminal(self.slurm, self.user)
            if any(
                s["status"] in {"SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN"} for s in self.attempts()
            ):
                return dict(
                    status="BLOCKED_UNKNOWN_SUBMISSION",
                    capacity=capacity(self.slurm.queue(self.user), self.attempts(), self.user),
                )
            if not RecoveryQueue(recovery).snapshot()["ready"]:
                return dict(
                    status="NO_READY_WORK",
                    capacity=capacity(self.slurm.queue(self.user), self.attempts(), self.user),
                )
            for pool, count in POOL_SPECS.items():
                path = self.pool_dir(pool)
                if (path / "INTENT.json").exists() or (path / "STATE.json").exists():
                    continue
                usage = capacity(self.slurm.queue(self.user), self.attempts(), self.user)
                if (
                    usage["teacher_gpus"] + count > TEACHER_GPU_LIMIT
                    or usage["run_gpus"] + count > RUN_GPU_LIMIT
                    or usage["user_submissions"] >= SUBMIT_LIMIT
                ):
                    continue
                self._submit(pool, count)
                if any(
                    s["status"] in {"SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN"}
                    for s in self.attempts()
                ):
                    break
            return dict(
                status="SUBMISSION_PASS_FINISHED",
                pools=self.attempts(),
                capacity=capacity(self.slurm.queue(self.user), self.attempts(), self.user),
            )

    def _submit(self, pool, count):
        r = self.recovery
        r.verify()
        path = self.pool_dir(pool)
        path.mkdir(parents=True, exist_ok=True)
        command = [
            r.plan["python"],
            str(r.extension / "scripts/sr_f2/recovery.py"),
            "pool",
            "--plan",
            str(r.path),
            "--pool-id",
            pool,
        ]
        body = "#!/bin/bash\nset -euo pipefail\n"
        body += "export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
        body += "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1\n"
        body += "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n"
        body += (
            "export PYTHONPATH="
            + shlex.quote(str(r.extension / "src") + ":" + str(r.code / "src"))
            + "\n"
        )
        body += "cd " + shlex.quote(str(r.code)) + "\n"
        body += "test ! -e " + shlex.quote(str(r.root / "STOP")) + "\n"
        body += "test ! -e " + shlex.quote(str(r.directory / "STOP.json")) + "\n"
        body += "exec " + shlex.join(command) + "\n"
        script = path / "run.sbatch"
        with script.open("x") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        intent = dict(
            pool_id=pool,
            gpu_count=count,
            plan_sha256=r.sha256,
            script=str(script),
            script_sha256=file_hash(script),
            command=command,
            job_name="srf2r-" + pool,
            user=self.user,
            comment="srf2r-" + object_hash([str(r.path), pool, r.sha256])[:20],
            at=now(),
        )
        args = [
            "sbatch",
            "--hold",
            "--parsable",
            "--account=rose",
            "--partition=cluster02",
            "--qos=" + TEACHER_QOS,
            "--nodes=1",
            "--ntasks=1",
            f"--gres=gpu:pro6000:{count}",
            "--constraint=highmem",
            f"--cpus-per-task={4 * count}",
            f"--mem={90 * count}G",
            "--time=4320",
            "--job-name=" + intent["job_name"],
            "--comment=" + intent["comment"],
            "--output=" + str(path / "slurm-%j.log"),
            str(script),
        ]
        save_json(path / "INTENT.json", dict(intent, sbatch=args), exclusive=True)
        state = dict(
            pool_id=pool, gpu_count=count, status="SUBMISSION_UNKNOWN", plan_sha256=r.sha256
        )
        save_json(path / "STATE.json", state, exclusive=True)
        try:
            response = self.slurm.run(args)
        except Exception as error:
            save_json(
                path / "SUBMISSION_EXCEPTION.json",
                dict(error=repr(error), at=now()),
                exclusive=True,
            )
            raise
        save_json(
            path / "SUBMISSION.json",
            dict(
                returncode=response.returncode,
                stdout=response.stdout,
                stderr=response.stderr,
                at=now(),
            ),
            exclusive=True,
        )
        match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", response.stdout)
        if response.returncode or not match:
            return
        state.update(job_id=match[1], status="HELD")
        save_json(path / "STATE.json", state)
        self.release(pool)

    def release(self, pool):
        r, path = self.recovery, self.pool_dir(pool)
        state, intent = read_json(path / "STATE.json"), read_json(path / "INTENT.json")
        if state["status"] != "HELD" or (path / "RELEASE_INTENT.json").exists():
            raise PermissionError("Release is already attempted or allocation is ambiguous")
        r.verify()
        r.old_jobs_terminal(self.slurm, self.user)
        if (
            intent.get("plan_sha256") != r.sha256
            or file_hash(intent["script"]) != intent["script_sha256"]
        ):
            raise PermissionError("Held recovery script or plan changed")
        fields = self.slurm.show(state["job_id"])
        verify_allocation(fields, dict(intent, job_id=state["job_id"]), self.user, held=True)
        usage = capacity(self.slurm.queue(self.user), self.attempts(), self.user)
        if (
            usage["teacher_gpus"] > TEACHER_GPU_LIMIT
            or usage["run_gpus"] > RUN_GPU_LIMIT
            or usage["user_submissions"] > SUBMIT_LIMIT
        ):
            raise PermissionError("Reservation capacity exceeded; retaining held job")
        save_json(path / "ALLOCATION_VERIFIED.json", dict(fields=fields, at=now()), exclusive=True)
        save_json(
            path / "RELEASE_INTENT.json", dict(job_id=state["job_id"], at=now()), exclusive=True
        )
        # Save uncertainty before external mutation; a crash never permits replay.
        state["status"] = "RELEASE_UNKNOWN"
        save_json(path / "STATE.json", state)
        response = self.slurm.run(["scontrol", "release", state["job_id"]])
        save_json(
            path / "RELEASE.json",
            dict(
                returncode=response.returncode,
                stdout=response.stdout,
                stderr=response.stderr,
                at=now(),
            ),
            exclusive=True,
        )
        if response.returncode == 0:
            with queue_lease(path / "state_lock"):
                current = read_json(path / "STATE.json")
                if current["status"] == "RELEASE_UNKNOWN":
                    save_json(path / "STATE.json", dict(current, status="SUBMITTED"))


def run_pool(recovery, pool_id, *, slurm=None, poll_seconds=1, popen=subprocess.Popen):
    r = recovery
    controller = Controller(r, slurm=slurm)
    path = controller.pool_dir(pool_id)
    state, intent = read_json(path / "STATE.json"), read_json(path / "INTENT.json")
    job = os.environ.get("SLURM_JOB_ID")
    if (
        job != state.get("job_id")
        or state.get("plan_sha256") != r.sha256
        or intent.get("plan_sha256") != r.sha256
    ):
        raise PermissionError("Pool is not in its bound recovery allocation")
    controller.user = intent["user"]
    verify_allocation(controller.slurm.show(job), dict(intent, job_id=job), controller.user)
    if file_hash(intent["script"]) != intent["script_sha256"]:
        raise PermissionError("Recovery batch script changed")
    r.verify()
    r.old_jobs_terminal(controller.slurm, controller.user)
    usage = capacity(
        controller.slurm.queue(controller.user), controller.attempts(), controller.user
    )
    if (
        usage["teacher_gpus"] > TEACHER_GPU_LIMIT
        or usage["run_gpus"] > RUN_GPU_LIMIT
        or usage["user_submissions"] > SUBMIT_LIMIT
    ):
        raise PermissionError("Live recovery reservation capacity exceeds its limits")
    devices = visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""), POOL_SPECS[pool_id])
    if (path / "STARTED.json").exists():
        raise PermissionError("Recovery allocation already started; no automatic retry")
    save_json(
        path / "STARTED.json",
        dict(job_id=job, plan_sha256=r.sha256, devices=devices, pid=os.getpid(), at=now()),
        exclusive=True,
    )
    queue = RecoveryQueue(r)
    active, slots = {}, {}
    stopping = False

    def signal_stop(signum, frame):
        nonlocal stopping
        stopping = True

    def record_slot(device, status, **details):
        value = dict(device=device, status=status, **details)
        if slots.get(device) == value:
            return
        slots[device] = value
        with (path / "SLOT_EVENTS.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(value, at=now()), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        save_json(path / "SLOTS.json", dict(slots=slots, at=now()))

    handlers = {sig: signal.signal(sig, signal_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            if stopping or (r.root / "STOP").exists() or queue.snapshot()["failed"]:
                raise RuntimeError("Recovery allocation stopped; no automatic retry")
            for device, (process, log, claim, attempt) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[device]
                save_json(
                    attempt / "PROCESS_EXIT.json", dict(returncode=code, at=now()), exclusive=True
                )
                if code:
                    queue.transition(claim, error=f"Original frozen worker exited {code}")
                    raise RuntimeError("Recovery worker failed; preserving partial output")
                result = verify_task(r.root, claim["task"], r.plan["registration_sha256"])
                queue.transition(claim, result=result)
                record_slot(device, "AVAILABLE", completed_task=claim["task"])
            snapshot = queue.snapshot()
            save_json(path / "QUEUE_SNAPSHOT.json", dict(snapshot, at=now()))
            if not active and (snapshot["all_complete"] or snapshot["may_exit"]):
                break
            for device in devices:
                if device in active:
                    continue
                claim, snapshot = queue.claim(pool_id, device, job)
                if claim is None:
                    record_slot(device, "IDLE", reason=snapshot["idle_reason"])
                    continue
                r.verify()
                task = claim["task"]
                r.task_prelaunch(task_id(task))
                attempt = path / "workers" / task_id(task)
                attempt.mkdir(parents=True, exist_ok=False)
                command = [
                    r.plan["python"],
                    str(r.code / "scripts/sr_f2/run_worker.py"),
                    "--root",
                    str(r.root),
                    "--kind",
                    task["kind"],
                    "--model-id",
                    task["model_id"],
                ]
                save_json(
                    attempt / "LAUNCH_INTENT.json",
                    dict(
                        command=command,
                        claim_id=claim["claim_id"],
                        task=task,
                        device=device,
                        plan_sha256=r.sha256,
                        audit_entry_sha256=claim["audit_entry_sha256"],
                        at=now(),
                    ),
                    exclusive=True,
                )
                log = (attempt / "worker.log").open("xb")
                try:
                    if queue.snapshot()["failed"]:
                        raise RuntimeError("Recovery stopped before child launch")
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=device,
                        PYTHONPATH=str(r.code / "src"),
                        CUBLAS_WORKSPACE_CONFIG=":4096:8",
                        PYTHONUNBUFFERED="1",
                        PYTHONDONTWRITEBYTECODE="1",
                        OMP_NUM_THREADS="4",
                        MKL_NUM_THREADS="4",
                        OPENBLAS_NUM_THREADS="4",
                    )
                    process = popen(
                        command,
                        cwd=r.code,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException as error:
                    log.close()
                    queue.transition(claim, error=repr(error))
                    raise
                active[device] = (process, log, claim, attempt)
                queue.transition(claim, pid=process.pid)
                record_slot(device, "BUSY", task=task)
            time.sleep(min(max(poll_seconds, 0), 60))
        for device in devices:
            record_slot(device, "FINISHED", reason=snapshot["idle_reason"])
        result = dict(
            status="DRAINED",
            plan_sha256=r.sha256,
            job_id=job,
            pool_id=pool_id,
            all_tasks_complete=snapshot["all_complete"],
            completion_reason=snapshot["idle_reason"],
            at=now(),
        )
        save_json(path / "COMPLETE.json", result, exclusive=True)
        with queue_lease(path / "state_lock"):
            save_json(path / "STATE.json", dict(state, status="DRAINED"))
        return result
    except BaseException as error:
        queue.stop(repr(error), pool_id)
        save_json(path / "FAILURE.json", dict(error=repr(error), at=now()), exclusive=True)
        with queue_lease(path / "state_lock"):
            save_json(path / "STATE.json", dict(state, status="FAILED"))
        raise
    finally:
        for process, _, _, _ in active.values():
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for process, log, claim, attempt in active.values():
            try:
                result = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                result = process.wait(timeout=10)
            log.close()
            save_json(
                attempt / "PROCESS_EXIT.json",
                dict(returncode=result, pool_aborted=True, at=now()),
                exclusive=True,
            )
            queue.transition(claim, error="Pool aborted; no automatic recovery retry")
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
