"""Replace each naturally drained recovery allocation without an experiment-wide barrier.

The parent plan, code and receipts remain immutable. A replacement can use only
its own parent's released 2/2/1 GPUs, after fresh queue/accounting verification.
Capacity includes both generations and every unknown reservation.
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

from sr_f2.orchestration import controller_lease, queue_lease, task_id, visible_devices
from sr_f2_recovery.orchestration import RecoveryQueue as ParentQueue
from sr_f2_recovery.orchestration import Slurm
from sr_f12.orchestration import file_hash, gpu_count, memory_gib, now, read_json, save_json
from sr_f12.protocol import TEACHER_QOS, object_hash

RUN_GPU_LIMIT = 5
TEACHER_GPU_LIMIT = 8
USER_SUBMIT_LIMIT = 5
POOL_SPECS = {
    "REPLAY_POOL_1": {"parent_pool_id": "RECOVERY_POOL_1", "gpu_count": 2},
    "REPLAY_POOL_2": {"parent_pool_id": "RECOVERY_POOL_2", "gpu_count": 2},
    "REPLAY_POOL_3": {"parent_pool_id": "RECOVERY_POOL_3", "gpu_count": 1},
}
ACTIVE = frozenset({"CLAIMED", "RUNNING"})
UNKNOWN = frozenset({"SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN"})


def verify_controller_source(plan):
    from .replay import within

    required = ("src/sr_f2_replay/orchestration.py", "scripts/sr_f2/replay_recovery.py")
    manifest_path = within(plan.directory, plan.plan["extension_manifest"])
    if file_hash(manifest_path) != plan.plan["extension_manifest_sha256"]:
        raise PermissionError("Replay controller source manifest changed")
    manifest = read_json(manifest_path).get("files", {})
    if Path(__file__).resolve() != (plan.extension / required[0]).resolve():
        raise PermissionError("Replay controller was imported from an unbound extension")
    for relative in required:
        if relative not in manifest or file_hash(plan.extension / relative) != manifest[relative]:
            raise PermissionError("Replay controller code is absent or changed in its manifest")


def _identity(fields, intent, job, user):
    expected = dict(
        JobId=job,
        Account="rose",
        Partition="cluster02",
        QOS=TEACHER_QOS,
        JobName=intent["job_name"],
        Comment=intent["comment"],
        Command=intent["script"],
    )
    if (
        any(fields.get(k) != v for k, v in expected.items())
        or fields.get("UserId", "").split("(")[0] != user
    ):
        raise PermissionError("Bound allocation owner or submission identity differs")
    count, tres = intent["gpu_count"], fields.get("ReqTRES", "")
    if gpu_count(tres) != count or not re.search(rf"(?:^|,)gres/gpu:pro6000={count}(?:,|$)", tres):
        raise PermissionError("Bound allocation PRO6000 count differs")
    if memory_gib(tres) < count * 80 or "highmem" not in fields.get("Features", "").split("&"):
        raise PermissionError("Bound allocation lacks frozen per-worker resources")


def verify_allocation(fields, intent, job, user, *, held=False):
    _identity(fields, intent, job, user)
    if held and (fields.get("JobState"), fields.get("Reason")) != ("PENDING", "JobHeldUser"):
        raise PermissionError("Replacement allocation must remain held until verified")
    if not held and fields.get("JobState") != "RUNNING":
        raise PermissionError("Replacement allocation is not running")


def verify_dispatcher_allocation(fields, intent, job, user):
    """The admission loop runs only inside its separately bound, non-requeued CPU job."""
    expected = dict(
        JobId=job,
        Account="rose",
        QOS=intent["qos"],
        JobName=intent["job_name"],
        Comment=intent["comment"],
        Command=intent["script"],
        JobState="RUNNING",
        Requeue="0",
    )
    if (
        any(fields.get(k) != v for k, v in expected.items())
        or fields.get("UserId", "").split("(")[0] != user
    ):
        raise PermissionError("CPU dispatcher allocation identity differs")
    if not fields.get("ReqTRES") or gpu_count(fields["ReqTRES"]) != 0:
        raise PermissionError("Dispatcher must use zero GPUs")


def _terminal_matches(row, intent, job, user):
    expected = dict(
        job_id=job,
        state="COMPLETED",
        exit_code="0:0",
        user=user,
        account="rose",
        qos=TEACHER_QOS,
        name=intent["job_name"],
    )
    return isinstance(row, dict) and all(row.get(k) == v for k, v in expected.items())


def capacity(live, reservations, user):
    """Count live/pending allocations once; absent unverified identities reserve fully."""
    teacher = submitted = ours = 0
    known = {}
    for reservation in reservations:
        count = reservation.get("gpu_count")
        if type(count) is not int or count not in (1, 2):
            raise PermissionError("Unknown reservation count")
        job = reservation.get("job_id")
        if job:
            if job in known:
                raise PermissionError("Parent/replacement reservation identifier overlaps")
            known[job] = reservation
    for job, fields in live.items():
        owner = fields.get("UserId", "").split("(")[0]
        if not owner or not fields.get("QOS") or not fields.get("ReqTRES"):
            raise PermissionError("Unknown live queue ownership or capacity")
        count = gpu_count(fields["ReqTRES"])
        teacher += count if fields["QOS"] == TEACHER_QOS else 0
        submitted += owner == user
        if job in known:
            reservation = known[job]
            if count != reservation["gpu_count"] or owner != user or fields["QOS"] != TEACHER_QOS:
                raise PermissionError("Live reservation identity differs")
            ours += count
    for reservation in reservations:
        if reservation.get("job_id") not in live and not reservation.get("verified_terminal"):
            teacher += reservation["gpu_count"]
            submitted += 1
            ours += reservation["gpu_count"]
    return dict(teacher_gpus=teacher, user_submissions=submitted, run_gpus=ours)


class ReplayQueue:
    """One immutable claim per authorized endpoint, independent of parent queue ownership."""

    def __init__(self, plan):
        self.plan = plan
        self.directory = plan.directory / "queue"
        self.tasks = {}
        for key, definition in plan.tasks.items():
            task = definition.get("task")
            if not isinstance(task, dict) or task_id(task) != key or task["kind"] != "endpoint":
                raise PermissionError("Replay queue accepts only its exact registered endpoints")
            if plan.parent.allowed.get(key, {}).get("status") != "BLOCKED_INCOMPLETE_GENERATION":
                raise PermissionError(
                    "Replay endpoint was not blocked in the immutable parent audit"
                )
            self.tasks[key] = task
        if len(self.tasks) != 5:
            raise PermissionError("Exactly five explicitly authorized replay endpoints required")
        self.hashes = {key: object_hash(definition) for key, definition in plan.tasks.items()}

    def states(self):
        states = {}
        for path in (self.directory / "tasks").glob("*/STATE.json"):
            key, state = path.parent.name, read_json(path)
            claim = read_json(path.with_name("CLAIM.json"))
            if (
                key not in self.tasks
                or state.get("task") != self.tasks[key]
                or state.get("status") not in ACTIVE | {"COMPLETE", "FAILED"}
            ):
                raise PermissionError("Replay queue task state differs")
            if (
                claim.get("plan_sha256") != self.plan.sha256
                or claim.get("task_definition_sha256") != self.hashes[key]
                or state.get("claim_sha256") != object_hash(claim)
            ):
                raise PermissionError("Replay claim plan or task definition changed")
            if (
                any(
                    state.get(k) != claim.get(k)
                    for k in ("task", "pool_id", "job_id", "device", "claim_id")
                )
                or claim.get("pool_id") not in POOL_SPECS
            ):
                raise PermissionError("Replay claim ownership differs")
            states[key] = state
        for path in (self.directory / "tasks").glob("*/CLAIM.json"):
            if path.parent.name not in states:
                raise RuntimeError("Orphan replay claim; launch unknown, no retry")
        return states

    def _snapshot(self, states):
        failed = (self.plan.directory / "STOP.json").exists() or any(
            row["status"] == "FAILED" for row in states.values()
        )
        ready = [] if failed else [self.tasks[key] for key in self.tasks if key not in states]
        return dict(
            ready=ready,
            active=sum(row["status"] in ACTIVE for row in states.values()),
            completed=sum(row["status"] == "COMPLETE" for row in states.values()),
            unclaimed=len(self.tasks) - len(states),
            failed=failed,
            all_complete=len(states) == len(self.tasks)
            and all(row["status"] == "COMPLETE" for row in states.values()),
        )

    def snapshot(self):
        with queue_lease(self.directory):
            return self._snapshot(self.states())

    def claim(self, pool, device, job):
        if pool not in POOL_SPECS or not str(job).isdigit():
            raise PermissionError("Unknown replay owner")
        with queue_lease(self.directory):
            states = self.states()
            snapshot = self._snapshot(states)
            if snapshot["failed"]:
                raise RuntimeError("Replay queue stopped")
            owned = [
                row for row in states.values() if row["pool_id"] == pool and row["status"] in ACTIVE
            ]
            if (
                any(row["device"] == device for row in owned)
                or len(owned) >= POOL_SPECS[pool]["gpu_count"]
            ):
                raise PermissionError("Replay physical GPU already owns unfinished work")
            if not snapshot["ready"]:
                return None
            task = snapshot["ready"][0]
            key = task_id(task)
            claim = dict(
                task=task,
                pool_id=pool,
                job_id=str(job),
                device=device,
                plan_sha256=self.plan.sha256,
                task_definition_sha256=self.hashes[key],
                parent_plan_sha256=self.plan.parent.sha256,
                claim_id=object_hash([key, pool, str(job), device, time.time_ns()]),
                at=now(),
            )
            path = self.directory / "tasks" / key
            save_json(path / "CLAIM.json", claim, exclusive=True)
            state = dict(claim, claim_sha256=object_hash(claim), status="CLAIMED")
            save_json(path / "STATE.json", state, exclusive=True)
            return state

    def stop(self, error, pool):
        path = self.plan.directory / "STOP.json"
        if not path.exists():
            try:
                save_json(path, dict(error=str(error), pool_id=pool, at=now()), exclusive=True)
            except PermissionError:
                if not path.exists():
                    raise

    def transition(self, claim, *, pid=None, result=None, error=None):
        with queue_lease(self.directory):
            key = task_id(claim["task"])
            state = self.states()[key]
            if state["claim_id"] != claim["claim_id"] or state["status"] not in ACTIVE:
                raise PermissionError("Replay attempt cannot be retried or recommitted")
            if error is not None:
                state.update(status="FAILED", error=str(error), finished_at=now())
                self.stop(error, claim["pool_id"])
            elif result is not None:
                if {k: result.get(k) for k in ("kind", "model_id")} != claim["task"]:
                    raise PermissionError("Replay completion differs from its claim")
                state.update(status="COMPLETE", result=result, finished_at=now())
            elif type(pid) is int and pid > 0 and state["status"] == "CLAIMED":
                state.update(status="RUNNING", pid=pid, started_at=now())
            else:
                raise ValueError("Expected new process or verified completion")
            save_json(self.directory / "tasks" / key / "STATE.json", state)


class Controller:
    def __init__(self, plan, *, slurm=None, user=None):
        self.plan = plan
        self.slurm = slurm or Slurm()
        self.user = user or getpass.getuser()
        self.parent_queue = ParentQueue(plan.parent)

    def pool_dir(self, pool):
        if pool not in POOL_SPECS:
            raise PermissionError("Unknown replay pool")
        return self.plan.directory / "pools" / pool

    def attempts(self):
        result = []
        for pool, spec in POOL_SPECS.items():
            path = self.pool_dir(pool)
            if (path / "NOT_NEEDED.json").exists() and any(
                (path / name).exists() for name in ("INTENT.json", "STATE.json", "STARTED.json")
            ):
                raise PermissionError("Skipped pool has allocation evidence")
            if (path / "STATE.json").exists():
                state = read_json(path / "STATE.json")
                if (
                    state.get("pool_id") != pool
                    or state.get("gpu_count") != spec["gpu_count"]
                    or state.get("plan_sha256") != self.plan.sha256
                ):
                    raise PermissionError("Replay pool state differs")
                if state.get("status") not in UNKNOWN | {
                    "HELD",
                    "SUBMITTED",
                    "DRAINED",
                    "FAILED",
                } or (
                    state["status"] != "SUBMISSION_UNKNOWN"
                    and (not isinstance(state.get("job_id"), str) or not state["job_id"].isdigit())
                ):
                    raise PermissionError("Replay pool status or job identity is invalid")
                result.append(state)
            elif (path / "INTENT.json").exists():
                result.append(
                    dict(pool_id=pool, gpu_count=spec["gpu_count"], status="SUBMISSION_UNKNOWN")
                )
        if any(
            p.parent.name not in POOL_SPECS
            for p in (self.plan.directory / "pools").glob("*/INTENT.json")
        ):
            raise PermissionError("Unregistered replay allocation intent")
        return result

    def _parent(self, pool, live):
        spec = POOL_SPECS[pool]
        parent_id = spec["parent_pool_id"]
        base = self.plan.parent.directory / "pools" / parent_id
        state, intent = read_json(base / "STATE.json"), read_json(base / "INTENT.json")
        count, job = spec["gpu_count"], state.get("job_id")
        if (
            state.get("pool_id") != parent_id
            or intent.get("pool_id") != parent_id
            or state.get("gpu_count") != count
            or intent.get("gpu_count") != count
            or state.get("plan_sha256") != self.plan.parent.sha256
            or intent.get("plan_sha256") != self.plan.parent.sha256
            or intent.get("user") != self.user
            or not isinstance(job, str)
            or not job.isdigit()
        ):
            raise PermissionError("Parent pool identity changed or is unknown")
        reservation = dict(pool_id=parent_id, gpu_count=count, job_id=job, verified_terminal=False)
        if job in live:
            _identity(live[job], intent, job, self.user)
            return reservation, None
        terminal = self.slurm.terminal(job)
        if state.get("status") != "DRAINED" or not _terminal_matches(
            terminal, intent, job, self.user
        ):
            return reservation, None
        started, done = read_json(base / "STARTED.json"), read_json(base / "COMPLETE.json")
        if (
            done.get("status") != "DRAINED"
            or done.get("job_id") != job
            or done.get("pool_id") != parent_id
            or started.get("job_id") != job
            or done.get("plan_sha256") != self.plan.parent.sha256
            or started.get("plan_sha256") != self.plan.parent.sha256
            or len(set(started.get("devices", []))) != count
            or (base / "FAILURE.json").exists()
            or file_hash(intent["script"]) != intent["script_sha256"]
        ):
            raise PermissionError("Parent drain evidence is inconsistent")
        states = self.parent_queue.states()
        owned = {key: row for key, row in states.items() if row["pool_id"] == parent_id}
        actual = {p.name for p in (base / "workers").glob("*") if p.is_dir()}
        if actual != set(owned):
            raise PermissionError("Parent task/process ownership differs")
        exits = []
        for key, row in sorted(owned.items()):
            worker = base / "workers" / key
            launch, exited = (
                read_json(worker / "LAUNCH_INTENT.json"),
                read_json(worker / "PROCESS_EXIT.json"),
            )
            if (
                row["status"] != "COMPLETE"
                or row["job_id"] != job
                or row["device"] not in started["devices"]
                or any(launch.get(k) != row.get(k) for k in ("task", "claim_id", "device"))
                or exited.get("returncode") != 0
                or exited.get("pool_aborted") is True
            ):
                raise PermissionError("Parent workers are not all terminal successes")
            exits.append(
                dict(task_id=key, claim_sha256=row["claim_sha256"], exit_sha256=object_hash(exited))
            )
        proof = dict(
            parent_pool_id=parent_id,
            parent_job_id=job,
            gpu_count=count,
            parent_plan_sha256=self.plan.parent.sha256,
            terminal=terminal,
            parent_intent_sha256=object_hash(intent),
            parent_completion_sha256=object_hash(done),
            workers=exits,
        )
        return dict(reservation, verified_terminal=True), proof

    def context(self):
        live = self.slurm.queue(self.user)
        reservations, eligible = [], {}
        for pool in POOL_SPECS:
            reservation, proof = self._parent(pool, live)
            reservations.append(reservation)
            if proof is not None:
                eligible[pool] = proof
        attempts = self.attempts()
        for state in attempts:
            row = dict(state, verified_terminal=False)
            job = state.get("job_id")
            if job:
                path = self.pool_dir(state["pool_id"])
                intent = read_json(path / "INTENT.json")
                if job in live:
                    _identity(live[job], intent, job, self.user)
                elif state["status"] == "DRAINED" and _terminal_matches(
                    self.slurm.terminal(job), intent, job, self.user
                ):
                    done = read_json(path / "COMPLETE.json")
                    if (
                        done.get("job_id") != job
                        or done.get("plan_sha256") != self.plan.sha256
                        or done.get("status") != "DRAINED"
                    ):
                        raise PermissionError("Replacement terminal receipt differs")
                    row["verified_terminal"] = True
            reservations.append(row)
        return dict(
            live=live,
            eligible=eligible,
            attempts=attempts,
            capacity=capacity(live, reservations, self.user),
        )

    def check_stop(self):
        if any(
            path.exists()
            for path in (
                self.plan.directory / "STOP.json",
                self.plan.parent.directory / "STOP.json",
                self.plan.root / "STOP",
            )
        ):
            raise RuntimeError("Replay scheduling stopped")

    def capacity_receipt(self, context, *, proposed_gpus=0):
        return dict(
            plan_sha256=self.plan.sha256,
            capacity=context["capacity"],
            live_job_ids=sorted(context["live"]),
            proposed_gpus=proposed_gpus,
            parent_terminal_sha256={
                key: object_hash(proof) for key, proof in context["eligible"].items()
            },
            at=now(),
        )

    def mark_not_needed(self, queue):
        """Skip unused replacements only after every fixed task has an immutable claim."""
        with queue_lease(queue.directory):
            states = queue.states()
            if set(states) != set(queue.tasks) or any(
                row["status"] == "FAILED" for row in states.values()
            ):
                return None
            digest = object_hash({key: row["claim_sha256"] for key, row in states.items()})
            for pool in POOL_SPECS:
                path = self.pool_dir(pool)
                if (path / "INTENT.json").exists() or (path / "STATE.json").exists():
                    continue
                if (
                    any(row["pool_id"] == pool for row in states.values())
                    or (path / "STARTED.json").exists()
                ):
                    raise PermissionError("Unsubmitted pool owns a replay task or launch")
                record = dict(
                    pool_id=pool,
                    plan_sha256=self.plan.sha256,
                    task_ids=sorted(states),
                    task_claims_sha256=digest,
                    reason="ALL_ENDPOINTS_ALREADY_CLAIMED",
                )
                if (path / "NOT_NEEDED.json").exists():
                    existing = read_json(path / "NOT_NEEDED.json")
                    if any(existing.get(key) != value for key, value in record.items()):
                        raise PermissionError("Skipped pool coverage proof changed")
                else:
                    save_json(path / "NOT_NEEDED.json", dict(record, at=now()), exclusive=True)
            return digest

    def submit_once(self):
        with controller_lease(self.plan.directory):
            self.plan.verify()
            verify_controller_source(self.plan)
            self.check_stop()
            context = self.context()
            if any(state["status"] in UNKNOWN for state in context["attempts"]):
                return dict(status="BLOCKED_UNKNOWN_SUBMISSION", **context)
            queue = ReplayQueue(self.plan)
            for pool, spec in POOL_SPECS.items():
                if not queue.snapshot()["ready"]:
                    break
                path = self.pool_dir(pool)
                if (path / "NOT_NEEDED.json").exists():
                    raise PermissionError("Skipped replacement cannot be resubmitted")
                if (path / "INTENT.json").exists() or (path / "STATE.json").exists():
                    continue
                context = self.context()
                count, usage = spec["gpu_count"], context["capacity"]
                if (
                    pool not in context["eligible"]
                    or usage["teacher_gpus"] + count > TEACHER_GPU_LIMIT
                    or usage["run_gpus"] + count > RUN_GPU_LIMIT
                    or usage["user_submissions"] >= USER_SUBMIT_LIMIT
                ):
                    continue
                self._submit(pool, context)
                if any(state["status"] in UNKNOWN for state in self.attempts()):
                    break
            context = self.context()
            claimed = self.mark_not_needed(queue)
            return dict(
                status="SUBMISSION_PASS_FINISHED",
                queue=queue.snapshot(),
                task_claims_sha256=claimed,
                **context,
            )

    def _submit(self, pool, context):
        p, path = self.plan, self.pool_dir(pool)
        p.verify()
        verify_controller_source(p)
        self.check_stop()
        pending = ReplayQueue(p).snapshot()
        if pending["failed"]:
            raise RuntimeError("Replay failed before replacement submission")
        if pending["unclaimed"] == 0:
            return
        proof = context["eligible"][pool]
        count = POOL_SPECS[pool]["gpu_count"]
        path.mkdir(parents=True, exist_ok=True)
        save_json(
            path / "CAPACITY_BEFORE_SUBMIT.json",
            self.capacity_receipt(context, proposed_gpus=count),
            exclusive=True,
        )
        save_json(path / "PARENT_TERMINAL.json", proof, exclusive=True)
        command = [
            p.parent.plan["python"],
            str(p.extension / "scripts/sr_f2/replay_recovery.py"),
            "pool",
            "--plan",
            str(p.path),
            "--pool-id",
            pool,
        ]
        body = "#!/bin/bash\nset -euo pipefail\n"
        body += "export CUBLAS_WORKSPACE_CONFIG=:4096:8\n"
        body += "export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1\n"
        body += "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n"
        body += (
            "export PYTHONPATH="
            + shlex.quote(
                ":".join(str(root / "src") for root in (p.extension, p.parent.extension, p.code))
            )
            + "\n"
        )
        body += "cd " + shlex.quote(str(p.code)) + "\nexec " + shlex.join(command) + "\n"
        script = path / "run.sbatch"
        with script.open("x") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        intent = dict(
            pool_id=pool,
            gpu_count=count,
            user=self.user,
            plan_sha256=p.sha256,
            parent_pool_id=POOL_SPECS[pool]["parent_pool_id"],
            parent_job_id=proof["parent_job_id"],
            parent_terminal_sha256=object_hash(proof),
            script=str(script),
            script_sha256=file_hash(script),
            command=command,
            job_name="srf2replay-" + pool,
            comment="srf2replay-" + object_hash([str(p.path), pool, p.sha256])[:20],
            at=now(),
        )
        args = [
            "sbatch",
            "--hold",
            "--parsable",
            "--no-requeue",
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
            pool_id=pool, gpu_count=count, plan_sha256=p.sha256, status="SUBMISSION_UNKNOWN"
        )
        save_json(path / "STATE.json", state, exclusive=True)
        try:
            result = self.slurm.run(args)
        except BaseException as error:
            save_json(
                path / "SUBMISSION_EXCEPTION.json",
                dict(error=repr(error), at=now()),
                exclusive=True,
            )
            raise
        save_json(
            path / "SUBMISSION.json",
            dict(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, at=now()
            ),
            exclusive=True,
        )
        match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout)
        if result.returncode or not match:
            return
        save_json(path / "STATE.json", dict(state, job_id=match[1], status="HELD"))
        self.release(pool)

    def release(self, pool):
        p, path = self.plan, self.pool_dir(pool)
        state, intent = read_json(path / "STATE.json"), read_json(path / "INTENT.json")
        if state["status"] != "HELD" or (path / "RELEASE_INTENT.json").exists():
            raise PermissionError("Replacement release already attempted or is unknown")
        p.verify()
        verify_controller_source(p)
        self.check_stop()
        context = self.context()
        proof = context["eligible"].get(pool)
        if (
            proof is None
            or object_hash(proof) != intent["parent_terminal_sha256"]
            or read_json(path / "PARENT_TERMINAL.json") != proof
        ):
            raise PermissionError("Parent terminal proof changed before replacement release")
        if (
            intent.get("plan_sha256") != p.sha256
            or file_hash(intent["script"]) != intent["script_sha256"]
        ):
            raise PermissionError("Replacement script or plan changed")
        verify_allocation(
            self.slurm.show(state["job_id"]), intent, state["job_id"], self.user, held=True
        )
        usage = context["capacity"]
        if (
            usage["teacher_gpus"] > TEACHER_GPU_LIMIT
            or usage["run_gpus"] > RUN_GPU_LIMIT
            or usage["user_submissions"] > USER_SUBMIT_LIMIT
        ):
            raise PermissionError("Capacity exceeded; replacement remains held")
        save_json(
            path / "CAPACITY_BEFORE_RELEASE.json", self.capacity_receipt(context), exclusive=True
        )
        save_json(
            path / "ALLOCATION_VERIFIED.json",
            dict(fields=self.slurm.show(state["job_id"]), at=now()),
            exclusive=True,
        )
        save_json(
            path / "RELEASE_INTENT.json", dict(job_id=state["job_id"], at=now()), exclusive=True
        )
        save_json(path / "STATE.json", dict(state, status="RELEASE_UNKNOWN"))
        result = self.slurm.run(["scontrol", "release", state["job_id"]])
        save_json(
            path / "RELEASE.json",
            dict(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, at=now()
            ),
            exclusive=True,
        )
        if result.returncode == 0:
            with queue_lease(path / "state_lock"):
                current = read_json(path / "STATE.json")
                if current["status"] == "RELEASE_UNKNOWN":
                    save_json(path / "STATE.json", dict(current, status="SUBMITTED"))


def verified_task(plan, task):
    from .replay import verify_endpoint_task

    return verify_endpoint_task(plan, task)


def run_dispatcher(plan, *, slurm=None, poll_seconds=15, sleep=time.sleep):
    """Admit each drained parent's replacement promptly; exit once all tasks are owned."""
    if not 1 <= poll_seconds <= 60:
        raise ValueError("Dispatcher interval must be between 1 and 60 seconds")
    path = plan.directory / "dispatcher"
    intent, state = read_json(path / "INTENT.json"), read_json(path / "STATE.json")
    job = os.environ.get("SLURM_JOB_ID")
    release = read_json(path / "RELEASE_INTENT.json")
    if (
        not job
        or job != state.get("job_id")
        or release.get("job_id") != job
        or state.get("status") not in {"SUBMITTED", "RELEASE_UNKNOWN"}
        or intent.get("plan_sha256") != plan.sha256
        or file_hash(intent["script"]) != intent["script_sha256"]
    ):
        raise PermissionError("CPU dispatcher is outside its bound plan or allocation")
    controller = Controller(plan, slurm=slurm, user=intent["user"])
    allocation = controller.slurm.show(job)
    verify_dispatcher_allocation(allocation, intent, job, intent["user"])
    plan.verify()
    verify_controller_source(plan)
    controller.check_stop()
    save_json(path / "ALLOCATION_AT_START.json", dict(fields=allocation, at=now()), exclusive=True)
    save_json(
        path / "STARTED.json",
        dict(
            job_id=job,
            plan_sha256=plan.sha256,
            intent_sha256=object_hash(intent),
            pid=os.getpid(),
            at=now(),
        ),
        exclusive=True,
    )
    stopping = False

    def stop_signal(signum, frame):
        nonlocal stopping
        stopping = True

    handlers = {sig: signal.signal(sig, stop_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            if stopping:
                raise RuntimeError(
                    "CPU dispatcher interrupted; admission not retried automatically"
                )
            result = controller.submit_once()
            summary = {
                key: result[key]
                for key in ("status", "capacity", "task_claims_sha256", "queue")
                if key in result
            }
            save_json(
                path / "STATUS.json", dict(summary, job_id=job, plan_sha256=plan.sha256, at=now())
            )
            if result["status"] == "BLOCKED_UNKNOWN_SUBMISSION":
                raise RuntimeError(
                    "Unknown replacement submission; reserved capacity, no blind retry"
                )
            if result["queue"]["failed"]:
                raise RuntimeError("Replay task failed; admission stopped")
            if result["task_claims_sha256"] is not None:
                done = dict(
                    status="COMPLETE",
                    reason="ALL_ENDPOINTS_CLAIMED",
                    job_id=job,
                    plan_sha256=plan.sha256,
                    task_claims_sha256=result["task_claims_sha256"],
                    at=now(),
                )
                save_json(path / "COMPLETE.json", done, exclusive=True)
                with queue_lease(path / "state_lock"):
                    current = read_json(path / "STATE.json")
                    save_json(path / "STATE.json", dict(current, status="COMPLETE"))
                return done
            sleep(poll_seconds)
    except BaseException as error:
        # Admission failure does not cancel already running, independently bound GPU workers.
        save_json(
            path / "FAILURE.json",
            dict(job_id=job, plan_sha256=plan.sha256, error=repr(error), at=now()),
            exclusive=True,
        )
        with queue_lease(path / "state_lock"):
            current = read_json(path / "STATE.json")
            save_json(path / "STATE.json", dict(current, status="FAILED"))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def run_pool(plan, pool, *, slurm=None, poll_seconds=1, popen=subprocess.Popen):
    c = Controller(plan, slurm=slurm)
    path = c.pool_dir(pool)
    state, intent = read_json(path / "STATE.json"), read_json(path / "INTENT.json")
    job = os.environ.get("SLURM_JOB_ID")
    if (
        not job
        or job != state.get("job_id")
        or state.get("plan_sha256") != plan.sha256
        or intent.get("plan_sha256") != plan.sha256
    ):
        raise PermissionError("Replay pool is outside its bound allocation")
    c.user = intent["user"]
    verify_allocation(c.slurm.show(job), intent, job, c.user)
    plan.verify()
    verify_controller_source(plan)
    context = c.context()
    proof, usage = context["eligible"].get(pool), context["capacity"]
    if (
        proof is None
        or object_hash(proof) != intent["parent_terminal_sha256"]
        or file_hash(intent["script"]) != intent["script_sha256"]
    ):
        raise PermissionError("Replacement parent proof or script differs")
    if (
        usage["run_gpus"] > RUN_GPU_LIMIT
        or usage["teacher_gpus"] > TEACHER_GPU_LIMIT
        or usage["user_submissions"] > USER_SUBMIT_LIMIT
    ):
        raise PermissionError("Combined parent/replacement capacity exceeds limit")
    if (path / "STARTED.json").exists():
        raise PermissionError("Replay pool already started; no automatic retry")
    devices = visible_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES", ""), POOL_SPECS[pool]["gpu_count"]
    )
    save_json(path / "CAPACITY_AT_START.json", c.capacity_receipt(context), exclusive=True)
    save_json(
        path / "STARTED.json",
        dict(job_id=job, devices=devices, plan_sha256=plan.sha256, pid=os.getpid(), at=now()),
        exclusive=True,
    )
    queue, active, slots = ReplayQueue(plan), {}, {}
    stopping = False

    def stop_signal(signum, frame):
        nonlocal stopping
        stopping = True

    def slot(device, status, **details):
        value = dict(device=device, status=status, **details)
        if slots.get(device) != value:
            slots[device] = value
            with (path / "SLOT_EVENTS.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(value, at=now()), sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            save_json(path / "SLOTS.json", dict(slots=slots, at=now()))

    handlers = {sig: signal.signal(sig, stop_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            if (
                stopping
                or (plan.root / "STOP").exists()
                or (plan.parent.directory / "STOP.json").exists()
                or queue.snapshot()["failed"]
            ):
                raise RuntimeError("Replay pool stopped; no automatic retry")
            for device, (process, log, claim, attempt) in list(active.items()):
                result = process.poll()
                if result is None:
                    continue
                log.close()
                del active[device]
                save_json(
                    attempt / "PROCESS_EXIT.json", dict(returncode=result, at=now()), exclusive=True
                )
                if result:
                    queue.transition(claim, error=f"Replay endpoint exited {result}")
                    raise RuntimeError("Replay endpoint failed; preserving all attempts")
                try:
                    receipt = verified_task(plan, claim["task"])
                except BaseException as error:
                    queue.transition(claim, error=repr(error))
                    raise
                queue.transition(claim, result=receipt)
                slot(device, "AVAILABLE", completed_task=claim["task"])
            snapshot = queue.snapshot()
            save_json(path / "QUEUE_SNAPSHOT.json", dict(snapshot, at=now()))
            if not active and snapshot["unclaimed"] == 0:
                break
            for device in devices:
                if device in active:
                    continue
                claim = queue.claim(pool, device, job)
                if claim is None:
                    slot(device, "IDLE", reason="NO_UNCLAIMED_ENDPOINTS")
                    continue
                try:
                    plan.verify()
                except BaseException as error:
                    queue.transition(claim, error=repr(error))
                    raise
                key = task_id(claim["task"])
                attempt = path / "workers" / key
                attempt.mkdir(parents=True, exist_ok=False)
                command = [
                    plan.parent.plan["python"],
                    str(plan.extension / "scripts/sr_f2/replay_endpoint.py"),
                    "--plan",
                    str(plan.path),
                    "--task-id",
                    key,
                ]
                save_json(
                    attempt / "LAUNCH_INTENT.json",
                    dict(
                        task=claim["task"],
                        claim_id=claim["claim_id"],
                        device=device,
                        plan_sha256=plan.sha256,
                        task_definition_sha256=claim["task_definition_sha256"],
                        command=command,
                        at=now(),
                    ),
                    exclusive=True,
                )
                log = (attempt / "worker.log").open("xb")
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=device,
                    CUBLAS_WORKSPACE_CONFIG=":4096:8",
                    PYTHONUNBUFFERED="1",
                    PYTHONDONTWRITEBYTECODE="1",
                    OMP_NUM_THREADS="4",
                    MKL_NUM_THREADS="4",
                    OPENBLAS_NUM_THREADS="4",
                    PYTHONPATH=":".join(
                        str(root / "src")
                        for root in (plan.extension, plan.parent.extension, plan.code)
                    ),
                )
                try:
                    if queue.snapshot()["failed"]:
                        raise RuntimeError("Replay failed before child launch")
                    process = popen(
                        command,
                        cwd=plan.code,
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
                slot(device, "BUSY", task=claim["task"])
            time.sleep(min(max(poll_seconds, 0), 60))
        for device in devices:
            slot(device, "FINISHED", reason="NO_UNCLAIMED_ENDPOINTS")
        result = dict(
            status="DRAINED",
            pool_id=pool,
            job_id=job,
            plan_sha256=plan.sha256,
            parent_terminal_sha256=intent["parent_terminal_sha256"],
            all_tasks_complete=snapshot["all_complete"],
            at=now(),
        )
        save_json(path / "COMPLETE.json", result, exclusive=True)
        with queue_lease(path / "state_lock"):
            save_json(path / "STATE.json", dict(state, status="DRAINED"))
        return result
    except BaseException as error:
        queue.stop(repr(error), pool)
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
            queue.transition(claim, error="Replay pool aborted; no automatic retry")
        for sig, previous in handlers.items():
            signal.signal(sig, previous)
