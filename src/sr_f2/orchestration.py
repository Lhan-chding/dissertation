"""SR-F2 persistent GPU pools with held, identity-bound Slurm submission.

A pool is a queue of independent one-GPU processes, never data parallel training.
Unknown submissions reserve their complete allocation and are never retried.
Only jobs bound to this run's durable intents can be released. Scores remain sealed.
"""

from __future__ import annotations

import fcntl
import getpass
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path

from sr_f12.orchestration import (
    TERMINAL,
    file_hash,
    gpu_count,
    memory_gib,
    now,
    read_json,
    save_json,
)
from sr_f12.orchestration import (
    Slurm as PreviousSlurm,
)
from sr_f12.protocol import BASELINE, TEACHER_QOS, object_hash

GPU_LIMIT = 8
SUBMIT_LIMIT = 5
POOL_GPUS = 2
FINAL_STATES = frozenset({"COMPLETE", "FAILED"})
UNKNOWN_STATES = frozenset({"SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN", "ACCOUNTING_UNKNOWN"})
KINDS = frozenset({"history", "diagnostic", "pilot", "train", "endpoint"})


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise PermissionError("Unsafe registered identifier")
    return value


def global_task_matrix():
    """The frozen 53 scientific/technical work units, independent of GPU ownership."""
    from sr_f12.protocol import scientific_matrix as historical_matrix

    from .protocol import scientific_matrix, technical_matrix

    old = [row["model_id"] for row in historical_matrix()]
    new = [row["model_id"] for row in scientific_matrix()]
    tasks = []
    for kind, ids in (
        ("history", old),
        ("pilot", [row["model_id"] for row in technical_matrix()]),
        ("diagnostic", [BASELINE, *old]),
        ("train", new),
        ("endpoint", new),
    ):
        tasks.extend(dict(kind=kind, model_id=safe_id(mid)) for mid in ids)
    if len(tasks) != 53 or len({task_id(task) for task in tasks}) != 53:
        raise PermissionError("SR-F2 registered task matrix differs")
    return tasks


def task_id(task):
    if task.get("kind") not in KINDS or set(task) != {"kind", "model_id"}:
        raise PermissionError("Invalid registered task identity")
    return task["kind"] + "--" + safe_id(task["model_id"])


def pool_specs():
    """Four persistent two-GPU allocations draw from one global ready queue."""
    matrix_hash = object_hash(global_task_matrix())
    return [
        dict(
            pool_id=f"GPU_POOL_{index}",
            gpu_count=POOL_GPUS,
            queue="global",
            task_matrix_sha256=matrix_hash,
        )
        for index in range(1, GPU_LIMIT // POOL_GPUS + 1)
    ]


@contextmanager
def queue_lease(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "QUEUE.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class GlobalQueue:
    """Claim-before-launch shared scheduler; no failed or ambiguous task is retried."""

    def __init__(self, root, registration_hash):
        self.root = Path(root).resolve(strict=True)
        self.directory = self.root / "orchestration/queue"
        self.registration_hash = registration_hash
        self.matrix = global_task_matrix()
        self.tasks = {task_id(task): task for task in self.matrix}
        save_json(
            self.directory / "CONFIG.json",
            dict(
                tasks=self.matrix,
                task_matrix_sha256=object_hash(self.matrix),
                registration_sha256=registration_hash,
                priorities=["history", "pilot", "train", "endpoint", "diagnostic"],
                strict_seed_waves=False,
                failure_policy="stop_without_retry",
            ),
            exclusive=True,
        )

    def _states(self):
        rows = {}
        for path in (self.directory / "tasks").glob("*/STATE.json"):
            key = path.parent.name
            row = read_json(path)
            claim_path = path.with_name("CLAIM.json")
            if (
                key not in self.tasks
                or row.get("task") != self.tasks[key]
                or (
                    row.get("registration_sha256") != self.registration_hash
                    or row.get("status") not in {"CLAIMED", "RUNNING", "COMPLETE", "FAILED"}
                    or row.get("pool_id") not in {p["pool_id"] for p in pool_specs()}
                )
            ):
                raise PermissionError("Global queue task state identity differs")
            claim = read_json(claim_path)
            if any(
                row.get(k) != claim.get(k)
                for k in ("task", "pool_id", "device", "job_id", "registration_sha256", "claim_id")
            ) or object_hash(claim) != row.get("claim_sha256"):
                raise PermissionError("Global queue claim changed")
            rows[key] = row
        for path in (self.directory / "tasks").glob("*/CLAIM.json"):
            if path.parent.name not in rows:
                raise RuntimeError("Orphan global claim: launch state unknown; no retry")
        return rows

    def _gate(self, name, expected):
        path = self.root / name
        if not path.exists():
            return False
        value = read_json(path)
        if any(value.get(k) != v for k, v in expected.items()) or (
            value.get("registration_sha256") != self.registration_hash
        ):
            raise PermissionError("Global readiness receipt differs: " + name)
        return True

    def _snapshot(self, states):
        completed = {key for key, row in states.items() if row["status"] == "COMPLETE"}
        active = [row for row in states.values() if row["status"] in {"CLAIMED", "RUNNING"}]
        if (self.directory / "STOP.json").exists() or any(
            row["status"] == "FAILED" for row in states.values()
        ):
            return dict(
                stage="FAILED",
                ready=[],
                active=len(active),
                completed=len(completed),
                all_complete=False,
                idle_reason="GLOBAL_TECHNICAL_FAILURE",
            )
        histories = {task_id(t) for t in self.matrix if t["kind"] == "history"}
        pilots = {task_id(t) for t in self.matrix if t["kind"] == "pilot"}
        history_done, pilot_done = histories <= completed, pilots <= completed
        history_gate = history_done and self._gate(
            "HISTORY_COMPLETE.json", dict(status="COMPLETE", paths=12)
        )
        technical_gate = pilot_done and self._gate(
            "TECHNICAL_CHECK_SR_F2.json", dict(status="PASS")
        )
        science_ready = history_gate and technical_gate
        ready = []
        for task in self.matrix:
            key = task_id(task)
            if key in states:
                continue
            kind = task["kind"]
            allowed = kind in {"history", "pilot", "diagnostic"}
            if kind == "train":
                allowed = science_ready
            elif kind == "endpoint":
                allowed = (
                    science_ready
                    and task_id(dict(kind="train", model_id=task["model_id"])) in completed
                )
            if allowed:
                ready.append(task)
        priority = {
            kind: index
            for index, kind in enumerate(("history", "pilot", "train", "endpoint", "diagnostic"))
        }
        ready.sort(key=lambda task: priority[task["kind"]])
        all_done = len(completed) == len(self.matrix)
        stage = (
            "COMPLETE"
            if all_done
            else "TRAINING_AND_SEALED_ENDPOINTS"
            if science_ready
            else "TECHNICAL_PILOT"
            if history_done
            else "HISTORY_WITH_BACKFILL"
        )
        if ready:
            reason = None
        elif all_done:
            reason = "ALL_TASKS_COMPLETE"
        elif history_done and not history_gate:
            reason = "WAIT_HISTORY_CPU_READINESS"
        elif pilot_done and not technical_gate:
            reason = "WAIT_TECHNICAL_CPU_READINESS"
        elif active:
            reason = "WAIT_ACTIVE_PREREQUISITES"
        else:
            reason = "WAIT_REQUIRED_READINESS"
        return dict(
            stage=stage,
            ready=ready,
            active=len(active),
            completed=len(completed),
            unclaimed=len(self.matrix) - len(states),
            all_complete=all_done,
            idle_reason=reason,
            history_complete=history_done,
            pilots_complete=pilot_done,
            science_ready=science_ready,
        )

    def snapshot(self):
        with queue_lease(self.directory):
            return self._snapshot(self._states())

    def claim(self, pool_id, device, job_id):
        if pool_id not in {p["pool_id"] for p in pool_specs()} or not str(job_id).isdigit():
            raise PermissionError("Invalid queue owner")
        with queue_lease(self.directory):
            states = self._states()
            if (self.root / "STOP").exists():
                raise RuntimeError("SR-F2 stopped; no new claims")
            if any(
                row["pool_id"] == pool_id
                and row["device"] == device
                and row["status"] in {"CLAIMED", "RUNNING"}
                for row in states.values()
            ):
                raise PermissionError("A physical GPU already owns an unfinished task")
            if (
                sum(
                    row["pool_id"] == pool_id and row["status"] in {"CLAIMED", "RUNNING"}
                    for row in states.values()
                )
                >= POOL_GPUS
            ):
                raise PermissionError("Pool already owns its full physical GPU allocation")
            snapshot = self._snapshot(states)
            if snapshot["stage"] == "FAILED":
                raise RuntimeError("Global queue stopped after a technical failure")
            if not snapshot["ready"]:
                return None, snapshot
            task = snapshot["ready"][0]
            directory = self.directory / "tasks" / task_id(task)
            claim = dict(
                task=task,
                pool_id=pool_id,
                device=device,
                job_id=str(job_id),
                registration_sha256=self.registration_hash,
                claimed_at=now(),
                claim_id=object_hash([task, pool_id, device, str(job_id), time.time_ns()]),
            )
            save_json(directory / "CLAIM.json", claim, exclusive=True)
            row = dict(claim, status="CLAIMED", claim_sha256=object_hash(claim))
            save_json(directory / "STATE.json", row, exclusive=True)
            return row, snapshot

    def transition(self, claim, *, pid=None, result=None, error=None):
        with queue_lease(self.directory):
            key = task_id(claim["task"])
            row = self._states()[key]
            if row["claim_id"] != claim["claim_id"]:
                raise PermissionError("Claim ownership changed")
            if row["status"] not in {"CLAIMED", "RUNNING"}:
                raise PermissionError("A terminal task cannot be retried or recommitted")
            if error is not None:
                row.update(status="FAILED", error=str(error), finished_at=now())
                stop_path = self.directory / "STOP.json"
                if not stop_path.exists():
                    save_json(
                        stop_path,
                        dict(
                            task=claim["task"], pool_id=claim["pool_id"], error=str(error), at=now()
                        ),
                        exclusive=True,
                    )
            elif result is not None:
                if {k: result.get(k) for k in ("kind", "model_id")} != claim["task"]:
                    raise PermissionError("Verified task completion differs from claim")
                row.update(status="COMPLETE", result=result, finished_at=now())
            elif type(pid) is int and pid > 0:
                row.update(status="RUNNING", pid=pid, started_at=now())
            else:
                raise ValueError("A running PID or verified completion is required")
            save_json(self.directory / "tasks" / key / "STATE.json", row)
            return row

    def stop_pool(self, pool_id, error):
        with queue_lease(self.directory):
            stop = self.directory / "STOP.json"
            if not stop.exists():
                save_json(stop, dict(pool_id=pool_id, error=str(error), at=now()), exclusive=True)
            for key, row in self._states().items():
                if row["pool_id"] == pool_id and row["status"] in {"CLAIMED", "RUNNING"}:
                    row.update(status="FAILED", error=str(error), finished_at=now())
                    save_json(self.directory / "tasks" / key / "STATE.json", row)


def verify_pool_task_ownership(root, pool_id, tasks):
    """Bind an actual pool's disjoint completion subset to its global claims and exits."""
    root = Path(root).resolve(strict=True)
    config = read_json(root / "orchestration/queue/CONFIG.json")
    queue = GlobalQueue(root, config["registration_sha256"])
    pool_directory = root / "orchestration/pools" / pool_id
    pool_state = read_json(pool_directory / "STATE.json")
    started = read_json(pool_directory / "STARTED.json")
    if started.get("job_id") != pool_state.get("job_id") or (
        len(set(started.get("devices", []))) != POOL_GPUS
    ):
        raise PermissionError("Pool process allocation or physical-device receipt differs")
    with queue_lease(queue.directory):
        states = queue._states()
    owned = {key: row for key, row in states.items() if row["pool_id"] == pool_id}
    ids = [task_id({k: receipt[k] for k in ("kind", "model_id")}) for receipt in tasks]
    if len(ids) != len(set(ids)) or set(ids) != set(owned):
        raise PermissionError("Pool completion has missing or duplicate globally claimed tasks")
    for key, result in zip(ids, tasks, strict=True):
        row = owned[key]
        directory = root / "orchestration/pools" / pool_id / "workers" / key
        launch = read_json(directory / "LAUNCH_INTENT.json")
        exit_receipt = read_json(directory / "PROCESS_EXIT.json")
        if (
            row["status"] != "COMPLETE"
            or row.get("result") != result
            or row["job_id"] != pool_state.get("job_id")
            or row["device"] not in started["devices"]
            or (
                launch.get("claim_id") != row["claim_id"]
                or launch.get("task") != row["task"]
                or launch.get("device") != row["device"]
                or exit_receipt.get("returncode") != 0
                or exit_receipt.get("pool_aborted") is True
            )
        ):
            raise PermissionError("Pool claim/process/verified completion binding differs")
    return True


class Slurm(PreviousSlurm):
    def terminal(self, job_id):
        result = self.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                str(job_id),
                "--format=JobIDRaw,State,ExitCode,User%100,Account%100,QOS%100,JobName%100",
            ]
        )
        if result.returncode:
            raise RuntimeError("sacct unavailable: " + result.stderr)
        rows = [row.split("|") for row in result.stdout.splitlines() if row.strip()]
        rows = [row for row in rows if row[0] == str(job_id)]
        if len(rows) != 1:
            return None
        row = rows[0]
        if len(row) < 7:
            raise PermissionError("Incomplete accounting identity")
        return dict(
            job_id=row[0],
            state=row[1].split()[0].rstrip("+"),
            exit_code=row[2],
            user=row[3],
            account=row[4],
            qos=row[5],
            name=row[6],
        )

    def queue(self):
        # All users in the teacher QoS share GrpTRES. --user would undercount it.
        result = self.run(["squeue", "--noheader", "--qos", TEACHER_QOS, "--format=%i"])
        if result.returncode:
            raise RuntimeError("Teacher-QoS queue unavailable: " + result.stderr)
        ids = result.stdout.split()
        if len(ids) != len(set(ids)) or any(not job.isdigit() for job in ids):
            raise PermissionError("Unrecognized queue identifier; capacity unknown")
        return {job: self.show(job) for job in ids}


def teacher_usage(queue, attempts, user):
    """Count requested physical GPUs globally, and submitted jobs for this user."""
    gpus = submitted = 0
    for row in queue.values():
        if row.get("QOS") != TEACHER_QOS:
            continue
        if not row.get("ReqTRES") or not row.get("UserId"):
            raise PermissionError("Teacher-QoS reservation identity/capacity unknown")
        gpus += gpu_count(row["ReqTRES"])
        submitted += row["UserId"].split("(")[0] == user
    for state in attempts:
        if state["status"] not in FINAL_STATES and state.get("job_id") not in queue:
            count = state.get("gpu_count")
            if type(count) is not int or count != POOL_GPUS:
                raise PermissionError("Unknown intent allocation size")
            gpus += count
            submitted += 1
    return dict(gpus=gpus, submitted=submitted)


def verify_allocation(fields, intent, user, *, held=False):
    for key, value in dict(
        Account="rose",
        Partition="cluster02",
        QOS=TEACHER_QOS,
        JobName=intent["job_name"],
        Comment=intent["comment"],
        Command=intent["script"],
        JobId=intent["job_id"],
    ).items():
        if fields.get(key) != value:
            raise PermissionError("Allocation identity differs: " + key)
    if fields.get("UserId", "").split("(")[0] != user:
        raise PermissionError("Allocation owner differs")
    tres = fields.get("ReqTRES", "")
    count = intent["gpu_count"]
    if (
        count != POOL_GPUS
        or gpu_count(tres) != count
        or not re.search(rf"(?:^|,)gres/gpu:pro6000={count}(?:,|$)", tres)
    ):
        raise PermissionError("Expected two physical PRO6000 GPUs per pool")
    if memory_gib(tres) < 80 * count:
        raise PermissionError("Actual pool memory is below 80 GiB per worker")
    if "highmem" not in fields.get("Features", "").split("&"):
        raise PermissionError("Expected highmem constraint")
    if held and (fields.get("JobState"), fields.get("Reason")) != ("PENDING", "JobHeldUser"):
        raise PermissionError("New submission is not held")


@contextmanager
def controller_lease(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "CONTROLLER.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another SR-F2 controller owns the lease") from None
        yield


class Controller:
    def __init__(self, root, code_root, *, python=sys.executable, slurm=None, user=None):
        self.root = Path(root).resolve(strict=True)
        self.code = Path(code_root).resolve(strict=True)
        self.python = str(Path(python).resolve(strict=True))
        self.slurm = slurm or Slurm()
        self.user = user or getpass.getuser()
        self.directory = self.root / "orchestration"
        self.specs = {row["pool_id"]: row for row in pool_specs()}
        self.directory.mkdir(parents=True, exist_ok=True)
        save_json(
            self.directory / "CONFIG.json",
            dict(
                root=str(self.root),
                code_root=str(self.code),
                python=self.python,
                user=self.user,
                teacher_qos=TEACHER_QOS,
                max_teacher_qos_gpus=GPU_LIMIT,
                max_user_teacher_submissions=SUBMIT_LIMIT,
                pools=list(self.specs.values()),
                strict_seed_waves=False,
                global_tasks=global_task_matrix(),
                global_task_matrix_sha256=object_hash(global_task_matrix()),
                scores_sealed_until_all_training_and_endpoints=True,
            ),
            exclusive=True,
        )

    def identity(self):
        from .protocol import verify_registration

        if (self.root / "STOP").exists():
            raise PermissionError("SR-F2 STOP exists")
        registration = verify_registration(self.root)
        if Path(registration["code_root"]).resolve(strict=True) != self.code:
            raise PermissionError("Controller source directory differs from registration")
        return object_hash(registration)

    def pool_dir(self, key):
        if safe_id(key) not in self.specs:
            raise PermissionError("Unregistered pool")
        return self.directory / "pools" / key

    def pool_state(self, key):
        path = self.pool_dir(key) / "STATE.json"
        return read_json(path) if path.exists() else None

    def attempts(self):
        result = []
        for key, spec in self.specs.items():
            state = self.pool_state(key)
            if state is not None:
                if state.get("pool_id") != key or state.get("gpu_count") != spec["gpu_count"]:
                    raise PermissionError("Pool state identity differs")
                result.append(state)
            elif (self.pool_dir(key) / "INTENT.json").exists():
                intent = read_json(self.pool_dir(key) / "INTENT.json")
                result.append(
                    dict(pool_id=key, gpu_count=intent["gpu_count"], status="SUBMISSION_UNKNOWN")
                )
        known = set(self.specs)
        intent_paths = (self.directory / "pools").glob("*/INTENT.json")
        if any(p.parent.name not in known for p in intent_paths):
            raise PermissionError("Unregistered pool intent exists")
        return result

    def command(self, key):
        return [
            self.python,
            str(self.code / "scripts/sr_f2/gpu_pool.py"),
            "--root",
            str(self.root),
            "--plan",
            str(self.pool_dir(key) / "PLAN.json"),
        ]

    def submit(self, key, queue):
        directory = self.pool_dir(key)
        if (directory / "INTENT.json").exists() or self.pool_state(key):
            return False
        if key not in self.ready_pools()[0]:
            raise PermissionError("Pool prerequisites are not complete")
        usage = teacher_usage(queue, self.attempts(), self.user)
        if usage["gpus"] + POOL_GPUS > GPU_LIMIT or usage["submitted"] >= SUBMIT_LIMIT:
            return False
        registration_hash = self.identity()
        directory.mkdir(parents=True, exist_ok=True)
        plan = dict(
            self.specs[key],
            root=str(self.root),
            registration_sha256=registration_hash,
            code_root=str(self.code),
            python=self.python,
        )
        save_json(directory / "PLAN.json", plan, exclusive=True)
        script = directory / "run.sbatch"
        command = self.command(key)
        body = "#!/bin/bash\nset -euo pipefail\n"
        body += "export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
        body += "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1\n"
        body += "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n"
        body += "export PYTHONPATH=" + shlex.quote(str(self.code / "src")) + "\n"
        body += "cd " + shlex.quote(str(self.code)) + "\n"
        body += "test ! -e " + shlex.quote(str(self.root / "STOP")) + "\n"
        body += "exec " + shlex.join(command) + "\n"
        with script.open("x") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        intent = dict(
            pool_id=key,
            created=now(),
            registration_sha256=registration_hash,
            script=str(script),
            script_sha256=file_hash(script),
            command=command,
            plan_sha256=object_hash(plan),
            gpu_count=POOL_GPUS,
            job_name="srf2-" + key + "-LH",
            comment="srf2-" + object_hash([str(self.root), key, registration_hash])[:20],
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
            f"--gres=gpu:pro6000:{POOL_GPUS}",
            "--constraint=highmem",
            f"--cpus-per-task={4 * POOL_GPUS}",
            f"--mem={90 * POOL_GPUS}G",
            "--time=4320",
            "--job-name=" + intent["job_name"],
            "--comment=" + intent["comment"],
            "--output=" + str(directory / "slurm-%j.log"),
            str(script),
        ]
        intent["sbatch"] = args
        save_json(directory / "INTENT.json", intent, exclusive=True)
        state = dict(
            pool_id=key,
            gpu_count=POOL_GPUS,
            status="SUBMISSION_UNKNOWN",
            registration_sha256=registration_hash,
        )
        save_json(directory / "STATE.json", state)
        try:
            result = self.slurm.run(args)
        except Exception as error:
            save_json(
                directory / "SUBMISSION_EXCEPTION.json",
                dict(error=repr(error), at=now()),
                exclusive=True,
            )
            raise
        save_json(
            directory / "SUBMISSION.json",
            dict(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, at=now()
            ),
            exclusive=True,
        )
        match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout)
        if result.returncode or not match:
            return False
        state.update(job_id=match[1], status="HELD")
        save_json(directory / "STATE.json", state)
        self.release(key, state, intent)
        return True

    def release(self, key, state, intent):
        directory = self.pool_dir(key)
        intent = dict(intent, job_id=state["job_id"])
        if file_hash(intent["script"]) != intent["script_sha256"]:
            raise PermissionError("Submitted script changed")
        if object_hash(read_json(directory / "PLAN.json")) != intent["plan_sha256"]:
            raise PermissionError("Submitted pool plan changed")
        if self.identity() != intent["registration_sha256"]:
            raise PermissionError("Registration differs from submission intent")
        fields = self.slurm.show(state["job_id"])
        verify_allocation(fields, intent, self.user, held=True)
        usage = teacher_usage(self.slurm.queue(), self.attempts(), self.user)
        if usage["gpus"] > GPU_LIMIT or usage["submitted"] > SUBMIT_LIMIT:
            raise PermissionError("Teacher-QoS capacity exceeded; retaining held allocation")
        if (directory / "RELEASE_INTENT.json").exists():
            state["status"] = "RELEASE_UNKNOWN"
            save_json(directory / "STATE.json", state)
            return
        save_json(
            directory / "ALLOCATION_VERIFIED.json",
            dict(job_id=state["job_id"], fields=fields, registration_sha256=self.identity()),
            exclusive=True,
        )
        save_json(
            directory / "RELEASE_INTENT.json",
            dict(job_id=state["job_id"], at=now()),
            exclusive=True,
        )
        result = self.slurm.run(["scontrol", "release", state["job_id"]])
        save_json(
            directory / "RELEASE.json",
            dict(
                job_id=state["job_id"],
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                at=now(),
            ),
            exclusive=True,
        )
        state["status"] = "SUBMITTED" if result.returncode == 0 else "RELEASE_UNKNOWN"
        save_json(directory / "STATE.json", state)

    def refresh(self, queue):
        for state in self.attempts():
            if state["status"] in FINAL_STATES:
                continue
            key, job = state["pool_id"], state.get("job_id")
            directory = self.pool_dir(key)
            intent = read_json(directory / "INTENT.json")
            if not job:
                matches = [
                    jid for jid, row in queue.items() if row.get("Comment") == intent["comment"]
                ]
                if len(matches) > 1:
                    raise PermissionError("Multiple Slurm jobs match one intent")
                if not matches:
                    continue
                job = matches[0]
                verify_allocation(queue[job], dict(intent, job_id=job), self.user)
                state.update(job_id=job, status="HELD")
                save_json(directory / "RECONCILED.json", dict(job_id=job, at=now()), exclusive=True)
                save_json(directory / "STATE.json", state)
            if job in queue:
                verify_allocation(queue[job], dict(intent, job_id=job), self.user)
                if queue[job].get("Reason") == "JobHeldUser":
                    self.release(key, state, intent)
                else:
                    state.update(status=queue[job]["JobState"])
                    save_json(directory / "STATE.json", state)
                continue
            terminal = self.slurm.terminal(job)
            if terminal is None or terminal["state"] not in TERMINAL:
                state["status"] = "ACCOUNTING_UNKNOWN"
                save_json(directory / "STATE.json", state)
                continue
            if any(
                terminal.get(k) != v
                for k, v in dict(
                    job_id=job,
                    user=self.user,
                    account="rose",
                    qos=TEACHER_QOS,
                    name=intent["job_name"],
                ).items()
            ):
                raise PermissionError("Terminal accounting identity differs")
            save_json(directory / "TERMINAL.json", terminal, exclusive=True)
            if terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0":
                try:
                    evidence = self.verify_pool(key)
                except Exception as error:
                    save_json(
                        directory / "VERIFICATION_FAILURE.json",
                        dict(error=repr(error), at=now()),
                        exclusive=True,
                    )
                    state.update(status="FAILED", failure="ARTIFACT_VERIFICATION")
                else:
                    save_json(directory / "VERIFIED_COMPLETE.json", evidence, exclusive=True)
                    state["status"] = "COMPLETE"
            else:
                state.update(status="FAILED", failure=terminal)
            save_json(directory / "STATE.json", state)

    def verify_pool(self, key):
        directory = self.pool_dir(key)
        receipt = read_json(directory / "COMPLETE.json")
        plan = read_json(directory / "PLAN.json")
        state = read_json(directory / "STATE.json")
        intent = read_json(directory / "INTENT.json")
        allocation = read_json(directory / "ALLOCATION_VERIFIED.json")
        if (
            object_hash(plan) != intent["plan_sha256"]
            or file_hash(intent["script"]) != intent["script_sha256"]
            or allocation.get("job_id") != state.get("job_id")
            or allocation.get("registration_sha256") != self.identity()
            or read_json(directory / "RELEASE_INTENT.json").get("job_id") != state["job_id"]
        ):
            raise PermissionError("Completed allocation binding differs")
        verify_allocation(allocation["fields"], dict(intent, job_id=state["job_id"]), self.user)
        if receipt.get("status") != "COMPLETE" or receipt.get("plan_sha256") != object_hash(plan):
            raise PermissionError("GPU pool completion differs")
        registration_hash = self.identity()
        actual = receipt.get("tasks")
        if not isinstance(actual, list):
            raise PermissionError("GPU pool completion tasks absent")
        expected = [
            verify_task(self.root, {k: row[k] for k in ("kind", "model_id")}, registration_hash)
            for row in actual
        ]
        if actual != expected or receipt.get("task_matrix_sha256") != object_hash(
            global_task_matrix()
        ):
            raise PermissionError("GPU pool task completion differs")
        verify_pool_task_ownership(self.root, key, actual)
        return dict(pool_id=key, receipt_sha256=object_hash(receipt), tasks=expected)

    def ready_pools(self):
        queue = GlobalQueue(self.root, self.identity())
        snapshot = queue.snapshot()
        return list(self.specs), snapshot["stage"]

    def advance_readiness(self):
        """Only this ordinary-QoS CPU controller publishes the scientific barriers."""
        from .protocol import finalize_readiness

        queue = GlobalQueue(self.root, self.identity())
        snapshot = queue.snapshot()
        history_missing = (
            snapshot.get("history_complete") and not (self.root / "HISTORY_COMPLETE.json").exists()
        )
        pilots_missing = (
            snapshot.get("pilots_complete")
            and not (self.root / "TECHNICAL_CHECK_SR_F2.json").exists()
        )
        if history_missing or pilots_missing:
            finalize_readiness(self.root)
        return queue.snapshot()

    def tick(self):
        if os.environ.get("SLURM_JOB_QOS") == TEACHER_QOS:
            raise PermissionError("CPU controller must use ordinary override QoS")
        with controller_lease(self.directory):
            registration_hash = self.identity()
            global_queue = GlobalQueue(self.root, registration_hash)
            queue = self.slurm.queue()
            self.refresh(queue)
            attempts = self.attempts()
            if any(s["status"] == "FAILED" for s in attempts):
                stage = "BLOCKED_TECHNICAL_FAILURE"
                for state in attempts:
                    if state["status"] == "FAILED":
                        global_queue.stop_pool(
                            state["pool_id"], "Allocation terminal/verification failure"
                        )
            elif any(s["status"] in UNKNOWN_STATES for s in attempts):
                stage = "BLOCKED_UNKNOWN_SUBMISSION"
            elif global_queue.snapshot()["stage"] == "FAILED":
                stage = "BLOCKED_TECHNICAL_FAILURE"
            else:
                readiness = self.advance_readiness()
                stage = readiness["stage"]
                for key in self.specs:
                    self.submit(key, queue)
                    queue = self.slurm.queue()
                    if any(s["status"] in UNKNOWN_STATES for s in self.attempts()):
                        stage = "BLOCKED_UNKNOWN_SUBMISSION"
                        break
            attempts = self.attempts()
            snapshot = global_queue.snapshot()
            complete = {s["pool_id"] for s in attempts if s["status"] == "COMPLETE"}
            all_complete = complete == set(self.specs) and snapshot["all_complete"]
            if all_complete:
                stage = "ALL_GPU_TASKS_COMPLETE_REPORT_PENDING"
            result = dict(
                stage=stage,
                at=now(),
                pools={s["pool_id"]: s for s in attempts},
                teacher_usage=teacher_usage(queue, attempts, self.user),
                global_queue=snapshot,
                all_gpu_tasks_complete=all_complete,
                endpoint_seal_release_ready=all_complete,
                experiment_complete=False,
            )
            save_json(self.directory / "STATUS.json", result)
            return result


def task_receipt_path(root, task):
    if task.get("kind") not in KINDS:
        raise PermissionError("Unknown worker kind")
    return Path(root) / "tasks" / task["kind"] / safe_id(task["model_id"]) / "COMPLETE.json"


def verify_task(root, task, registration_hash):
    """Authenticate worker completion and its artifact bytes without exposing scores."""
    root = Path(root).resolve(strict=True)
    path = task_receipt_path(root, task)
    receipt = read_json(path)
    expected_status = "PASS" if task["kind"] == "pilot" else "COMPLETE"
    if any(
        receipt.get(key) != value
        for key, value in dict(
            status=expected_status,
            kind=task["kind"],
            model_id=task["model_id"],
            registration_sha256=registration_hash,
        ).items()
    ):
        raise PermissionError("Worker completion identity/status differs")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise PermissionError("Worker completion has no artifact evidence")
    names = set()
    for artifact in artifacts:
        relative = Path(artifact["path"])
        target = (root / relative).resolve(strict=True)
        if relative.is_absolute() or ".." in relative.parts or root not in target.parents:
            raise PermissionError("Worker artifact escapes SR-F2 run")
        if str(relative) in names or file_hash(target) != artifact["sha256"]:
            raise PermissionError("Worker artifact duplicated or changed")
        names.add(str(relative))
    verify_worker_artifacts(root, task, receipt, names)
    return dict(kind=task["kind"], model_id=task["model_id"], receipt_sha256=object_hash(receipt))


def verify_evaluation_groups(directory, receipt, model_id, expected_records, *, step=None):
    """Hash and count raw records without reading any scientific score files."""
    refs = receipt.get("groups")
    if not isinstance(refs, list) or not refs:
        raise PermissionError("Evaluation completion has no raw groups")
    group_ids, slots = set(), set()
    for ref in refs:
        gid = ref.get("group_id", "")
        if not re.fullmatch(r"[0-9a-f]{64}", gid) or gid in group_ids:
            raise PermissionError("Evaluation raw group identity differs")
        group_ids.add(gid)
        path = Path(directory) / "groups" / (gid + ".json")
        raw = read_json(path)
        rows = raw.get("records")
        if (
            file_hash(path) != ref["raw_sha256"]
            or raw.get("group_id") != gid
            or not isinstance(rows, list)
            or not 1 <= len(rows) <= 8
            or object_hash(rows) != ref["records_sha256"]
            or raw.get("records_sha256") != ref["records_sha256"]
        ):
            raise PermissionError("Evaluation raw record bytes or group coverage differ")
        for row in rows:
            slot = row.get("slot_id")
            if (
                not isinstance(slot, str)
                or slot in slots
                or row.get("model_id") != model_id
                or (step is not None and row.get("step") != step)
            ):
                raise PermissionError("Evaluation slot identity or uniqueness differs")
            slots.add(slot)
    if len(slots) != expected_records:
        raise PermissionError("Evaluation record count differs")
    actual = {p.stem for p in (Path(directory) / "groups").glob("*.json")}
    if actual != group_ids or any((Path(directory) / "groups").glob("*.pending.jsonl")):
        raise PermissionError("Unregistered or unfinished raw evaluation groups remain")


def verify_worker_artifacts(root, task, receipt, names):
    """A COMPLETE label alone cannot stand in for the required kind-specific evidence."""
    kind, model_id = task["kind"], task["model_id"]
    if task not in global_task_matrix():
        raise PermissionError("Unregistered worker completion")
    if kind in {"train", "pilot"}:
        step = 96 if kind == "train" else 8
        relative = Path("runs" if kind == "train" else "technical") / model_id
        required = {
            str(relative / "COMPLETE.json"),
            str(relative / "checkpoints" / f"commit-{step:02d}.json"),
        }
        if kind == "train":
            required.add(str(relative / "monitor_state" / f"step-{step:02d}.json"))
        if names != required or receipt.get("step") != step:
            raise PermissionError("Training completion evidence or final step differs")
        directory = root / relative
        done = read_json(directory / "COMPLETE.json")
        checkpoint_dir = directory / "checkpoints"
        commit = read_json(checkpoint_dir / f"commit-{step:02d}.json")
        if (
            done.get("status") != "COMPLETE"
            or done.get("model_id") != model_id
            or done.get("step") != step
            or done.get("technical_only") != (kind == "pilot")
            or done.get("last_monitor_step") != step
            or done.get("checkpoint") != commit
            or read_json(checkpoint_dir / "LATEST.json") != commit
            or commit.get("step") != step
        ):
            raise PermissionError("Training/checkpoint completion binding differs")
        checkpoint_path = Path(commit["path"])
        target = (checkpoint_dir / checkpoint_path).resolve(strict=True)
        if (
            checkpoint_path.is_absolute()
            or ".." in checkpoint_path.parts
            or target.parent != checkpoint_dir.resolve()
            or file_hash(target) != commit["sha256"]
        ):
            raise PermissionError("Final checkpoint bytes or location differ")
        from sr_f1.contract import digest

        for monitor_step in range(0, step + 1, 8):
            monitor = read_json(directory / "monitor_state" / f"step-{monitor_step:02d}.json")
            identity, measure = monitor["identity"], monitor["measurement"]
            if (
                identity.get("model_id") != model_id
                or identity.get("step") != monitor_step
                or measure.get("model_id") != model_id
                or measure.get("step") != monitor_step
                or measure.get("status") != "COMPLETE"
                or measure.get("count") != 512
                or monitor.get("measurement_hash") != digest(measure)
            ):
                raise PermissionError("Required online monitor is incomplete or changed")
            if monitor_step == step and (
                identity.get("checkpoint_sha256") != commit["sha256"]
                or identity.get("checkpoint_state_hash") != commit["state_hash"]
            ):
                raise PermissionError("Final monitor is not bound to final checkpoint")
            verify_evaluation_groups(
                directory / "monitor" / f"step-{monitor_step:02d}",
                measure,
                model_id,
                512,
                step=monitor_step,
            )
        return
    relative = {
        "history": Path("history") / model_id,
        "diagnostic": Path("diagnostics") / model_id,
        "endpoint": Path("sealed/evaluation") / model_id,
    }[kind]
    if names != {str(relative / "COMPLETE.json")}:
        raise PermissionError("Evaluation completion evidence names differ")
    directory = root / relative
    done = read_json(directory / "COMPLETE.json")
    expected_records = {"history": 6656, "diagnostic": 2688, "endpoint": 15300}[kind]
    if (
        receipt.get("records") != expected_records
        or done.get("records") != expected_records
        or done.get("status") != "COMPLETE"
        or done.get("model_id") != model_id
    ):
        raise PermissionError("Evaluation completion identity or coverage differs")
    if kind == "history":
        summaries = done.get("summaries", [])
        steps = list(range(0, 97, 8))
        if done.get("steps") != steps or [s.get("step") for s in summaries] != steps:
            raise PermissionError("Historical monitor steps are incomplete")
        for summary in summaries:
            step_dir = directory / f"step-{summary['step']:02d}"
            if (
                summary != read_json(step_dir / "COMPLETE.json")
                or summary.get("count") != 512
                or summary.get("model_id") != model_id
            ):
                raise PermissionError("Historical monitor identity or count differs")
            verify_evaluation_groups(step_dir, summary, model_id, 512, step=summary["step"])
    else:
        if kind == "endpoint" and done.get("scores_sealed") is not True:
            raise PermissionError("New scientific endpoint scores must remain sealed")
        if kind == "endpoint" and any((directory / "scores").glob("*.json")):
            raise PermissionError("Scientific scores exist before endpoint seal release")
        if kind == "diagnostic" and (
            done.get("old_models_only") is not True
            or done.get("new_scientific_test_access") is not False
        ):
            raise PermissionError("Diagnostic source-model scope differs")
        verify_evaluation_groups(
            directory, done, model_id, expected_records, step=0 if model_id == BASELINE else 96
        )


def visible_devices(value, count):
    devices = value.split(",")
    if (
        len(devices) != count
        or len(set(devices)) != count
        or any(not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", device) for device in devices)
    ):
        raise PermissionError(f"Pool needs exactly {count} distinct allocated CUDA devices")
    return devices


def run_pool(root, plan_path, *, poll_seconds=1, popen=subprocess.Popen):
    """Keep two physical slots busy from the shared queue until all 53 tasks finish."""
    import json

    from .protocol import verify_registration

    root = Path(root).resolve(strict=True)
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read_json(plan_path)
    specs = {row["pool_id"]: row for row in pool_specs()}
    if plan.get("pool_id") not in specs or any(
        plan.get(k) != v for k, v in specs[plan["pool_id"]].items()
    ):
        raise PermissionError("Pool plan is not in the registered task matrix")
    pool_id = plan["pool_id"]
    directory = root / "orchestration/pools" / pool_id
    if plan_path != directory / "PLAN.json" or plan.get("root") != str(root):
        raise PermissionError("Pool plan location differs")
    state, intent = read_json(directory / "STATE.json"), read_json(directory / "INTENT.json")
    if state.get("job_id") != os.environ.get("SLURM_JOB_ID") or not state.get("job_id"):
        raise PermissionError("Pool is not running in its bound Slurm allocation")
    if object_hash(plan) != intent["plan_sha256"]:
        raise PermissionError("GPU pool plan changed after submission")
    devices = visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""), plan["gpu_count"])
    code = Path(plan["code_root"]).resolve(strict=True)
    config = read_json(root / "orchestration/CONFIG.json")
    if (
        plan["code_root"] != config["code_root"]
        or plan["python"] != config["python"]
        or state.get("pool_id") != pool_id
        or state.get("gpu_count") != plan["gpu_count"]
    ):
        raise PermissionError("Pool execution identity differs from controller configuration")
    if object_hash(verify_registration(root)) != plan["registration_sha256"]:
        raise PermissionError("Registration differs from pool allocation")
    if (directory / "STARTED.json").exists():
        raise PermissionError("Pool launch already recorded; no automatic retry")
    save_json(
        directory / "STARTED.json",
        dict(
            job_id=state["job_id"],
            pid=os.getpid(),
            devices=devices,
            plan_sha256=object_hash(plan),
            at=now(),
        ),
        exclusive=True,
    )
    queue = GlobalQueue(root, plan["registration_sha256"])
    active, slot_status = {}, {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    def record_slot(device, status, **details):
        value = dict(device=device, status=status, **details)
        if slot_status.get(device) == value:
            return
        slot_status[device] = value
        event = dict(value, at=now())
        with (directory / "SLOT_EVENTS.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        save_json(directory / "SLOTS.json", dict(slots=slot_status, at=now()))

    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while True:
            if stopping or (root / "STOP").exists():
                raise RuntimeError("SR-F2 pool stopped; no retries")
            if queue.snapshot()["stage"] == "FAILED":
                raise RuntimeError("Global task failure stops both GPU pools")
            # Harvest first, then refill immediately within the same scheduling turn.
            for device, (process, log, claim, task_dir) in list(active.items()):
                result = process.poll()
                if result is None:
                    continue
                log.close()
                del active[device]
                save_json(
                    task_dir / "PROCESS_EXIT.json",
                    dict(returncode=result, at=now()),
                    exclusive=True,
                )
                if result:
                    queue.transition(claim, error=f"Worker exited {result}")
                    raise RuntimeError(
                        "Worker failed; preserving all outputs and stopping both pools"
                    )
                evidence = verify_task(root, claim["task"], plan["registration_sha256"])
                queue.transition(claim, result=evidence)
                record_slot(device, "AVAILABLE", completed_task=claim["task"])
            snapshot = queue.snapshot()
            if snapshot["all_complete"]:
                if active:
                    raise PermissionError("Global completion conflicts with active child processes")
                completion_reason = "ALL_TASKS_COMPLETE"
                break
            if not active and snapshot["unclaimed"] == 0:
                completion_reason = "NO_UNCLAIMED_TASKS_AND_NO_OWN_ACTIVE"
                break
            for device in devices:
                if device in active:
                    continue
                claim, snapshot = queue.claim(pool_id, device, state["job_id"])
                if claim is None:
                    record_slot(device, "IDLE", idle_reason=snapshot["idle_reason"])
                    continue
                task = claim["task"]
                if object_hash(verify_registration(root)) != plan["registration_sha256"]:
                    raise PermissionError("Registered source changed during pool execution")
                task_dir = directory / "workers" / task_id(task)
                if task_dir.exists():
                    raise PermissionError("Worker launch already recorded; no automatic retry")
                task_dir.mkdir(parents=True)
                command = [
                    plan["python"],
                    str(code / "scripts/sr_f2/run_worker.py"),
                    "--root",
                    str(root),
                    "--kind",
                    task["kind"],
                    "--model-id",
                    task["model_id"],
                ]
                save_json(
                    task_dir / "LAUNCH_INTENT.json",
                    dict(
                        task=task,
                        device=device,
                        claim_id=claim["claim_id"],
                        command=command,
                        at=now(),
                    ),
                    exclusive=True,
                )
                log = (task_dir / "worker.log").open("xb")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
                try:
                    if (queue.directory / "STOP.json").exists():
                        raise RuntimeError("Global queue failed before child launch")
                    process = popen(
                        command,
                        cwd=code,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException as error:
                    log.close()
                    queue.transition(claim, error=repr(error))
                    raise
                active[device] = (process, log, claim, task_dir)
                queue.transition(claim, pid=process.pid)
                record_slot(device, "BUSY", task=task)
            time.sleep(min(max(poll_seconds, 0), 60))
        if object_hash(verify_registration(root)) != plan["registration_sha256"]:
            raise PermissionError("Registered source changed during pool execution")
        with queue_lease(queue.directory):
            states = queue._states()
        ordered = [
            states[task_id(task)]["result"]
            for task in global_task_matrix()
            if states[task_id(task)]["pool_id"] == pool_id
        ]
        verify_pool_task_ownership(root, pool_id, ordered)
        for device in devices:
            record_slot(device, "FINISHED", idle_reason=completion_reason)
        return save_json(
            directory / "COMPLETE.json",
            dict(
                status="COMPLETE",
                plan_sha256=object_hash(plan),
                task_matrix_sha256=object_hash(global_task_matrix()),
                global_completed_tasks=snapshot["completed"],
                completion_reason=completion_reason,
                tasks=ordered,
            ),
            exclusive=True,
        )
    except BaseException as error:
        save_json(
            directory / "POOL_FAILURE.json", dict(error=repr(error), at=now()), exclusive=True
        )
        queue.stop_pool(pool_id, error)
        raise
    finally:
        for process, _log, _claim, _task_dir in active.values():
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for process, log, _claim, task_dir in active.values():
            try:
                result = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                result = process.wait(timeout=10)
            log.close()
            save_json(
                task_dir / "PROCESS_EXIT.json",
                dict(returncode=result, at=now(), pool_aborted=True),
                exclusive=True,
            )
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def run_controller(controller, *, once=False, interval_seconds=30):
    while True:
        result = controller.tick()
        if once or result["stage"].startswith("BLOCKED_") or result["all_gpu_tasks_complete"]:
            return result
        time.sleep(min(interval_seconds, 60))
