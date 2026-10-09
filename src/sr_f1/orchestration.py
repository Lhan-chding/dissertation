"""Durable SR-F1 scheduling, with sealed TEST and conservative shared GPU accounting.

The hash-chained journal is authoritative. An ambiguous submission is UNKNOWN,
never retryable. Only a verified full-state checkpoint permits lease continuation.
The scheduler never reads old F2 results and has no cancellation API.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
from pathlib import Path

from mm_dev.orchestration import (
    AUTO_RESUME,
    TERMINAL,
    atomic_json,
    contained,
    digest,
    encoded,
    file_hash,
    now,
    process_lease,
    read_json,
    require,
    safe_id,
    summarize_accounting,
)
from mm_dev.orchestration import (
    SlurmBackend as BaseSlurmBackend,
)

PLAN_ID = "SR-F1-20261009"
LIVE = {"SUBMITTING", "REGISTERED", "ACTIVE", "UNKNOWN"}
SCIENCE_TERMINAL = {"COMPLETE", "TECHNICAL_FAILED"}
CHECKPOINT_REASONS = {"PREEMPTION", "TIME_LEASE_END", "STOP_REQUESTED"}
ARMS = ("A", "J", "PART", "DEC", "GATE")
SEEDS = (71001, 71002, 71003)
PERMISSION_PATH = "manifests/ALLOCATION_PERMISSION.json"


def worker_lease(root, task_id):
    return process_lease(Path(root) / "orchestration/leases" / (safe_id(task_id) + ".lock"))


def controller_incarnation(root, *, backend=None):
    """Verify this CPU controller allocation and positively bind lease restarts."""
    root, backend = Path(root), backend or SlurmBackend()
    job_id = os.environ.get("SLURM_JOB_ID", "")
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    require(
        re.fullmatch(r"[0-9]+", job_id) and re.fullmatch(r"[0-9]+", restart),
        "CONTROLLER_REQUEUE_REQUIRES_SLURM_IDENTITY",
    )
    receipt = backend._run(["scontrol", "show", "job", job_id, "--oneliner"])
    require(receipt["returncode"] == 0, "CONTROLLER_IDENTITY_UNKNOWN")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", receipt["stdout"]))
    permission = read_json(root / PERMISSION_PATH)
    require(
        fields.get("JobId") == job_id
        and fields.get("JobState") == "RUNNING"
        and fields.get("UserId", "").split("(")[0] == permission["owner"]
        and fields.get("Account") == permission["account"]
        and "AllocTRES" in fields
        and gpu_count(fields["AllocTRES"]) == 0,
        "CONTROLLER_MUST_BE_OWN_RUNNING_CPU_JOB",
    )
    identity = {
        "job_id": job_id,
        "restart": int(restart),
        "run_root": str(root.resolve()),
        "freeze_sha256": file_hash(root / "EXECUTION_FREEZE.json"),
    }
    folder = root / "orchestration/controller_leases"
    if int(restart):
        previous = folder / f"{job_id}_restart{int(restart) - 1:04d}_INTENT.json"
        require(previous.is_file(), "CONTROLLER_RESTART_WITHOUT_DURABLE_INTENT")
        prior = read_json(previous)
        require(
            prior["identity"] == {**identity, "restart": int(restart) - 1},
            "CONTROLLER_RESTART_IDENTITY_CHANGED",
        )
    observed = folder / f"{job_id}_restart{int(restart):04d}_START.json"
    if observed.exists():
        require(read_json(observed)["identity"] == identity, "CONTROLLER_INCARNATION_CHANGED")
    else:
        atomic_json(
            observed,
            {"identity": identity, "time": now(), "scheduler_receipt": receipt},
            exclusive=True,
        )
    return identity


def request_controller_requeue(root, identity, *, backend=None):
    """Requeue this CPU JobID once; uncertainty never creates a second request."""
    root, backend = Path(root), backend or SlurmBackend()
    require(
        identity == controller_incarnation(root, backend=backend), "CONTROLLER_IDENTITY_CHANGED"
    )
    stem = f"{identity['job_id']}_restart{identity['restart']:04d}"
    folder = root / "orchestration/controller_leases"
    intent = folder / (stem + "_INTENT.json")
    require(not intent.exists(), "CONTROLLER_REQUEUE_ALREADY_REQUESTED_NO_RETRY")
    command = ["scontrol", "requeue", identity["job_id"]]
    atomic_json(
        intent,
        {"status": "REQUEUE_INTENT", "identity": identity, "command": command, "time": now()},
        exclusive=True,
    )
    try:
        receipt = backend._run(command)
        status = "REQUEST_ACCEPTED" if receipt["returncode"] == 0 else "UNKNOWN"
        result = {"status": status, "identity": identity, "receipt": receipt, "time": now()}
    except Exception as exc:
        result = {"status": "UNKNOWN", "identity": identity, "error": repr(exc), "time": now()}
    atomic_json(folder / (stem + "_RESULT.json"), result, exclusive=True)
    return result


def tasks_for(matrix):
    expected = [(f"SRF1_{arm}_s{seed}", arm, seed) for seed in SEEDS for arm in ARMS]
    require(
        [(r["run_id"], r["arm"], r["paired_seed"]) for r in matrix] == expected,
        "FROZEN_15_PATH_MATRIX_CHANGED",
    )
    for row in matrix:
        require(
            row["common_start"] == "SRF1_COMMON_START"
            and (row["H"], row["B"], row["G"]) == (96, 16, 8)
            and row["checkpoint_steps"] == [0, 32, 64, 96]
            and row["schedule"] == f"manifests/schedules/train_seed_{row['paired_seed']}.jsonl",
            "FROZEN_TRAINING_BUDGET_CHANGED",
        )
    tasks = {}

    def add(key, phase, dependencies, gpus=1, **extra):
        tasks[key] = {"phase": phase, "dependencies": dependencies, "gpus": gpus, **extra}

    add("COMMON_START", "S2", [], operation="common_start")
    add("ENGINE", "S2", ["COMMON_START"], operation="engine")
    add(
        "BASELINE",
        "S3",
        ["ENGINE"],
        operation="evaluate",
        model_id="SRF1_COMMON_START",
        stage="baseline",
    )
    science = [row["run_id"] for row in matrix]
    for row in matrix:
        add(row["run_id"], "S4", ["BASELINE"], operation="train", run_id=row["run_id"])
    evaluations = []
    for model_id in ["SRF1_COMMON_START", *science]:
        key = "EVAL_" + model_id
        add(
            key,
            "S5",
            science,
            operation="evaluate",
            model_id=model_id,
            stage="final",
            terminal_dependencies=True,
        )
        evaluations.append(key)
    add(
        "ANALYZE",
        "S6",
        [*science, *evaluations],
        0,
        operation="analyze",
        terminal_dependencies=True,
    )
    add("RELEASE", "S6", ["ANALYZE"], 0, operation="release")
    return tasks


def _worker_identity(root, task_id):
    registration = read_json(Path(root) / "orchestration/REGISTRATION.json")
    attempt_id = safe_id(os.environ.get("SR_F1_ATTEMPT_ID"))
    require(
        task_id in registration["tasks"]
        and os.environ.get("SR_F1_TASK_ID") == task_id
        and os.environ.get("SR_F1_REGISTRATION_HASH") == digest(registration),
        "WORKER_IDENTITY_MISMATCH",
    )
    return registration, attempt_id


def _marker(root, task_id, status, artifacts, metadata=None, reason=None):
    root, task_id = Path(root), safe_id(task_id)
    registration, attempt_id = _worker_identity(root, task_id)
    require(artifacts and len(artifacts) == len(set(map(str, artifacts))), "ARTIFACTS_REQUIRED")
    descriptors = {}
    for relative in artifacts:
        path = contained(root, relative)
        require(path.is_file(), "MISSING_ARTIFACT:" + str(relative))
        descriptors[str(relative)] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    if status == "CHECKPOINTED":
        require(reason in CHECKPOINT_REASONS, "NONRESUMABLE_REASON")
        require(
            metadata
            and metadata.get("full_state") is True
            and metadata.get("identity_verified") is True,
            "FULL_STATE_RESUME_REQUIRED",
        )
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
        f"completions/{task_id}.json" if status == "COMPLETE" else f"checkpoints/{attempt_id}.json"
    )
    atomic_json(root / "orchestration" / relative, marker, exclusive=True)
    return marker


def complete_task(root, task_id, artifacts, metadata=None):
    return _marker(root, task_id, "COMPLETE", artifacts, metadata)


def checkpoint_task(root, task_id, reason, artifacts, metadata=None):
    return _marker(root, task_id, "CHECKPOINTED", artifacts, metadata, reason)


def fail_task(root, task_id, error):
    registration, attempt_id = _worker_identity(root, task_id)
    receipt = {
        "plan_id": PLAN_ID,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "registration_hash": digest(registration),
        "time": now(),
        "status": "TECHNICAL_FAILED",
        "error": str(error),
    }
    atomic_json(Path(root) / f"orchestration/failures/{attempt_id}.json", receipt, exclusive=True)
    return receipt


def validate_permission(permission, root):
    require(
        permission.get("plan_id") == PLAN_ID and permission.get("authorized") is True,
        "ALLOCATION_PERMISSION_REQUIRED",
    )
    require(
        Path(permission.get("run_root", "")).resolve() == Path(root).resolve(),
        "PERMISSION_ROOT_MISMATCH",
    )
    for field in ("owner", "account", "qos"):
        require(
            isinstance(permission.get(field), str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+", permission[field]),
            "INVALID_PERMISSION:" + field,
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


def gpu_count(tres):
    """Count untyped total once, otherwise sum typed GPU requests."""
    fields = dict(item.strip().split("=", 1) for item in tres.split(",") if "=" in item)
    values = {
        key: value
        for key, value in fields.items()
        if key == "gres/gpu" or key.startswith("gres/gpu:")
    }
    require(all(re.fullmatch(r"[0-9]+", value) for value in values.values()), "INVALID_GPU_TRES")
    return int(values["gres/gpu"]) if "gres/gpu" in values else sum(map(int, values.values()))


class SlurmBackend(BaseSlurmBackend):
    def project_capacity(self, permission):
        # All of the user's allocations count, regardless of study name or QoS.
        # Pending GPU requests reserve capacity conservatively as well.
        receipt = self._run(
            [
                "squeue",
                "--noheader",
                "--user",
                permission["owner"],
                "--states=all",
                "--format=%i|%u|%T",
            ]
        )
        require(receipt["returncode"] == 0, "PROJECT_QUEUE_UNKNOWN:" + json.dumps(receipt))
        jobs = []
        for line in receipt["stdout"].splitlines():
            values = [value.strip() for value in line.split("|")]
            require(
                len(values) == 3 and values[1] == permission["owner"], "INVALID_PROJECT_QUEUE_ROW"
            )
            job_id, _, state = values
            if state.split()[0].rstrip("+") in TERMINAL:
                continue
            require(re.fullmatch(r"[0-9]+", job_id), "UNSUPPORTED_PROJECT_JOB_ID")
            # %b only covers per-node requests. ReqTRES/AllocTRES include total
            # job GPUs for --gpus-per-task, multi-node and typed GRES as well.
            detail = self._run(["scontrol", "show", "job", job_id, "--oneliner"])
            require(detail["returncode"] == 0, "PROJECT_JOB_DETAIL_UNKNOWN:" + json.dumps(detail))
            fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", detail["stdout"]))
            require(
                fields.get("JobId") == job_id
                and fields.get("UserId", "").split("(")[0] == permission["owner"]
                and ("ReqTRES" in fields or "AllocTRES" in fields),
                "INVALID_PROJECT_JOB_DETAIL",
            )
            count = max(
                gpu_count(fields.get("ReqTRES", "")), gpu_count(fields.get("AllocTRES", ""))
            )
            jobs.append({"job_id": job_id, "gpus": count, "state": state, "receipt": detail})
        require(len({row["job_id"] for row in jobs}) == len(jobs), "DUPLICATE_PROJECT_JOB")
        return {"owner": permission["owner"], "jobs": jobs, "receipt": receipt}


class Scheduler:
    def __init__(
        self,
        plan,
        root,
        *,
        code_root,
        python,
        lease_minutes=4320,
        engine_lease_minutes=4320,
        backend=None,
        verify_execution=None,
    ):
        self.plan, self.root = Path(plan).resolve(), Path(root).resolve()
        self.code_root, self.python = Path(code_root).resolve(), Path(python).resolve()
        require(type(lease_minutes) is int and lease_minutes > 0, "INVALID_ALLOCATION_LEASE")
        require(
            type(engine_lease_minutes) is int and engine_lease_minutes > 0, "INVALID_ENGINE_LEASE"
        )
        self.lease_minutes, self.engine_lease_minutes = lease_minutes, engine_lease_minutes
        self.backend = backend or SlurmBackend()
        if verify_execution is None:
            from .freeze import verify_execution
        self.verify_execution = verify_execution
        self.folder = self.root / "orchestration"
        self.state = self.registration = None
        self.sequence, self.previous = 0, None
        self._journal_signature = None

    def _identity(self):
        self.verify_execution(self.plan, self.root, require_engine=False)
        plan = read_json(self.plan)
        require(plan["version"] == PLAN_ID, "WRONG_PLAN")
        resource = plan["resources"]
        require(
            resource["max_workers_across_active_project_runs"] == 5
            and resource["cumulative_gpu_hours_limit"] is None
            and resource["research_wallclock_limit"] is None
            and resource["stop_on_elapsed_time"] is False,
            "RESOURCE_CONTRACT_CHANGED",
        )
        permission = read_json(self.root / PERMISSION_PATH)
        validate_permission(permission, self.root)
        matrix_path = self.root / "manifests/RUN_MATRIX.json"
        tasks = tasks_for(read_json(matrix_path))
        script = self.code_root / "scripts/sr_f1/run_worker.py"
        return {
            "schema_version": 1,
            "plan_id": PLAN_ID,
            "plan_path": str(self.plan),
            "plan_sha256": file_hash(self.plan),
            "matrix_sha256": file_hash(matrix_path),
            "freeze_sha256": file_hash(self.root / "EXECUTION_FREEZE.json"),
            "permission_receipt_sha256": file_hash(self.root / PERMISSION_PATH),
            "permission": permission,
            "tasks": tasks,
            "task_order": list(tasks),
            "code_root": str(self.code_root),
            "python": str(self.python),
            "lease_minutes": self.lease_minutes,
            "engine_lease_minutes": self.engine_lease_minutes,
            "source_hashes": {"scripts/sr_f1/run_worker.py": file_hash(script)},
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
                # This process already verified and fsynced this exact prefix.
                # A new incarnation or changed journal always replays the chain.
                self._projections()
                return
        self.sequence, self.previous, self.state = 0, None, None
        if journal.exists():
            with journal.open("rb") as stream:
                for line in stream:
                    require(line.endswith(b"\n"), "JOURNAL_TRUNCATED_REQUIRES_REPAIR")
                    row = json.loads(line)
                    recorded = row.pop("sha256")
                    require(
                        recorded == digest(row)
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
                        recorded,
                        row["state"],
                    )
        if self.state is None:
            self.state = {
                "phase": "S1_VERIFIED",
                "test_sealed": True,
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
        for task_id, task in self.state["tasks"].items():
            for attempt in task["attempts"]:
                if not attempt.get("job_id"):
                    continue
                allocations.append(
                    {
                        "task_id": task_id,
                        "attempt_id": attempt["attempt_id"],
                        "job_id": attempt["job_id"],
                        "owner": permission["owner"],
                        "account": permission["account"],
                        "qos": permission["qos"],
                        "job_name": attempt["job_name"],
                        "comment": attempt["comment"],
                        "gpus": self.registration["tasks"][task_id]["gpus"],
                        "status": attempt["status"],
                        "attempt_manifest_path": attempt["manifest_path"],
                        "actual_gpu_seconds": attempt.get("accounting", {}).get("gpu_seconds"),
                    }
                )
        atomic_json(
            self.root / "manifests/ALLOCATIONS.json",
            {
                "plan_id": PLAN_ID,
                "registration_hash": digest(self.registration),
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
            "MARKER_IDENTITY_MISMATCH",
        )
        require(
            marker["status"] == ("CHECKPOINTED" if checkpoint else "COMPLETE"),
            "MARKER_STATUS_MISMATCH",
        )
        if checkpoint:
            require(
                marker["reason"] in CHECKPOINT_REASONS
                and marker["metadata"].get("full_state") is True
                and marker["metadata"].get("identity_verified") is True,
                "FULL_STATE_RESUME_REQUIRED",
            )
        require(marker["artifacts"], "EMPTY_MARKER")
        for relative, descriptor in marker["artifacts"].items():
            artifact = contained(self.root, relative)
            require(
                artifact.is_file()
                and artifact.stat().st_size == descriptor["bytes"]
                and file_hash(artifact) == descriptor["sha256"],
                "MARKER_ARTIFACT_CHANGED",
            )
        return True

    def _release(self, task_id, attempt):
        try:
            receipt = self.backend.release(attempt["job_id"])
            attempt["release_receipt"] = receipt
            require(receipt["returncode"] == 0, "RELEASE_RESULT_UNKNOWN")
            attempt["released"] = True
            self._save("ALLOCATION_RELEASED", task_id=task_id)
        except Exception as exc:
            attempt["status"] = self.state["tasks"][task_id]["status"] = "UNKNOWN"
            self._save("RELEASE_UNKNOWN", task_id=task_id, error=repr(exc))

    def _observe(self, task_id):
        task = self.state["tasks"][task_id]
        attempt = task["attempts"][-1]
        try:
            observation = self.backend.observe(attempt, self.registration["permission"])
            relative = f"observations/{attempt['attempt_id']}/{self.sequence + 1:08d}.json"
            atomic_json(self.folder / relative, observation, exclusive=True)
            attempt["observation"] = {
                "path": "orchestration/" + relative,
                "sha256": file_hash(self.folder / relative),
            }
            ids = {row["job_id"] for row in observation["queue"]} | {
                row["JobIDRaw"] for row in observation["accounting"]
            }
            require(len(ids) <= 1, "MULTIPLE_JOBS_MATCH_SUBMISSION")
            if not ids:
                raise RuntimeError("NO_POSITIVE_SLURM_IDENTITY")
            job_id = next(iter(ids))
            require(
                re.fullmatch(r"[0-9]+", job_id) and attempt.get("job_id") in (None, job_id),
                "SLURM_IDENTITY_CHANGED",
            )
            if not attempt.get("job_id"):
                attempt["job_id"] = job_id
                self._save("AMBIGUOUS_SUBMISSION_RECONCILED", task_id=task_id, job_id=job_id)
            if observation["queue"]:
                attempt["status"] = task["status"] = "ACTIVE"
                self._save("ALLOCATION_ACTIVE", task_id=task_id)
                if any(row["reason"] == "JobHeldUser" for row in observation["queue"]):
                    self._release(task_id, attempt)
                return
            if not observation["accounting"] or any(
                r["State"].split()[0].rstrip("+") not in TERMINAL for r in observation["accounting"]
            ):
                raise RuntimeError("ACCOUNTING_NOT_YET_TERMINAL")
            accounting = summarize_accounting(
                observation["accounting"],
                attempt,
                self.registration["permission"],
                self.registration["tasks"][task_id]["gpus"],
            )
            attempt["accounting"] = accounting
            with worker_lease(self.root, task_id):
                complete = self._marker_valid(task_id, attempt)
                checkpointed = self._marker_valid(task_id, attempt, checkpoint=True)
            terminal = accounting["terminal_state"]
            failure = self.folder / "failures" / (attempt["attempt_id"] + ".json")
            if failure.exists():
                receipt = read_json(failure)
                require(
                    receipt["registration_hash"] == digest(self.registration)
                    and receipt["attempt_id"] == attempt["attempt_id"]
                    and receipt["task_id"] == task_id,
                    "FAILURE_IDENTITY_CHANGED",
                )
                self._terminal_failure(task_id, attempt, receipt["error"])
            elif complete and terminal == "COMPLETED" and accounting["exit_code"] == "0:0":
                attempt["status"], task["status"] = terminal, "COMPLETE"
                task["completion_marker_sha256"] = file_hash(
                    self.folder / "completions" / (task_id + ".json")
                )
            elif (
                checkpointed
                and not complete
                and (
                    terminal in AUTO_RESUME
                    or (terminal == "COMPLETED" and accounting["exit_code"] == "0:0")
                )
            ):
                attempt["status"] = terminal
                checkpoint_path = f"orchestration/checkpoints/{attempt['attempt_id']}.json"
                checkpoint = read_json(contained(self.root, checkpoint_path))
                if (
                    task_id == "ENGINE"
                    and checkpoint["metadata"].get("restart_engine_track_required") is not True
                ):
                    self._terminal_failure(
                        task_id, attempt, "ENGINE_CONTINUOUS_TRACE_REQUIRES_EXPLICIT_REPAIR"
                    )
                else:
                    # ENGINE's explicit flag authorizes preservation of the old
                    # trace and a complete 4-vs-2+2 technical comparison. It does
                    # not permit treating an interrupted prefix as continuous.
                    task["resume_checkpoint"] = checkpoint_path
                    task["status"] = (
                        "STOPPED" if checkpoint["reason"] == "STOP_REQUESTED" else "RETRYABLE"
                    )
            else:
                self._terminal_failure(
                    task_id, attempt, "TERMINAL_WITHOUT_VERIFIED_COMPLETION_OR_FULL_STATE"
                )
            self._save("ALLOCATION_TERMINAL", task_id=task_id)
        except (ValueError, KeyError, TypeError) as exc:
            # Identity corruption is a project gate, not permission to drop an arm.
            attempt["status"] = task["status"] = "BLOCKED"
            task["blocker"] = repr(exc)
            self._save("IDENTITY_BLOCK", task_id=task_id, error=repr(exc))
        except Exception as exc:
            attempt["status"] = task["status"] = "UNKNOWN"
            self._save("OBSERVATION_UNKNOWN_NO_RETRY", task_id=task_id, error=repr(exc))

    def _terminal_failure(self, task_id, attempt, reason):
        task = self.state["tasks"][task_id]
        attempt["status"] = attempt["accounting"]["terminal_state"]
        task["status"] = (
            "TECHNICAL_FAILED"
            if self.registration["tasks"][task_id]["phase"] == "S4"
            else "BLOCKED"
        )
        task["blocker"] = reason

    def _capacity_available(self, task_id):
        permission = self.registration["permission"]
        evidence = {"time": now(), "task_id": task_id}
        try:
            capacity = self.backend.submission_capacity(permission)
            project = self.backend.project_capacity(permission)
            evidence.update(submission=capacity, project=project)
            require(
                capacity["owner"] == project["owner"] == permission["owner"]
                and capacity["qos"] == permission["qos"],
                "CAPACITY_OWNER_MISMATCH",
            )
            maximum, queued = capacity["max_submit_jobs_per_user"], capacity["queued_job_ids"]
            require(
                maximum is None or (type(maximum) is int and maximum >= 0), "INVALID_SUBMIT_LIMIT"
            )
            require(len(queued) == len(set(queued)), "DUPLICATE_QUEUE_ID")
            jobs = {row["job_id"]: row["gpus"] for row in project["jobs"]}
            require(
                len(jobs) == len(project["jobs"])
                and all(type(n) is int and n >= 0 for n in jobs.values()),
                "INVALID_PROJECT_CAPACITY",
            )
            reservations = [
                (key, attempt)
                for key, task in self.state["tasks"].items()
                for attempt in task["attempts"]
                if "accounting" not in attempt
            ]
            submit_used = len(queued) + sum(
                attempt.get("job_id") not in queued for _, attempt in reservations
            )
            gpu_used = sum(jobs.values()) + sum(
                self.registration["tasks"][key]["gpus"]
                for key, attempt in reservations
                if attempt.get("job_id") not in jobs
            )
            available = (maximum is None or submit_used < maximum) and gpu_used + self.registration[
                "tasks"
            ][task_id]["gpus"] <= 5
            summary = {
                "status": "AVAILABLE" if available else "WAITING",
                "project_reserved_gpus": gpu_used,
                "submission_slots_used": submit_used,
                "max_submit_jobs_per_user": maximum,
            }
        except Exception as exc:
            available, summary = False, {"status": "UNKNOWN", "error": repr(exc)}
        evidence["summary"] = summary
        atomic_json(
            self.folder / f"observations/capacity/{self.sequence + 1:08d}.json",
            evidence,
            exclusive=True,
        )
        self.state["capacity"] = summary
        self._save("CAPACITY_" + summary["status"], task_id=task_id)
        return available

    def _submit(self, task_id):
        task, spec = self.state["tasks"][task_id], self.registration["tasks"][task_id]
        require(task["status"] in {"WAITING", "RETRYABLE"}, "DUPLICATE_SUBMISSION_GUARD")
        if (self.root / "STOP").exists():
            return False
        self.verify_execution(
            self.plan, self.root, require_engine=spec["phase"] in {"S3", "S4", "S5", "S6"}
        )
        if task["status"] == "RETRYABLE":
            require(
                self._marker_valid(task_id, task["attempts"][-1], checkpoint=True),
                "RESUME_CHECKPOINT_REQUIRED",
            )
        if not self._capacity_available(task_id):
            return False
        attempt_id = f"{task_id}_attempt{len(task['attempts']):04d}"
        registration_hash = digest(self.registration)
        prefix = registration_hash[:12]
        attempt = {
            "attempt_id": attempt_id,
            "created_at": now(),
            "status": "SUBMITTING",
            "job_id": None,
            "job_name": f"srf1-{prefix}-{attempt_id}",
            "comment": f"srf1:{prefix}:{attempt_id}",
            "manifest_path": f"orchestration/attempts/{attempt_id}.json",
        }
        worker = [
            str(self.python),
            str(self.code_root / "scripts/sr_f1/run_worker.py"),
            "--plan",
            str(self.plan),
            "--run-root",
            str(self.root),
            "--task-id",
            task_id,
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
            "SR_F1_TASK_ID": task_id,
            "SR_F1_ATTEMPT_ID": attempt_id,
            "SR_F1_REGISTRATION_HASH": registration_hash,
        }
        if task.get("resume_checkpoint"):
            environment["SR_F1_RESUME_RECEIPT"] = str(
                contained(self.root, task["resume_checkpoint"])
            )
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
        permission = self.registration["permission"]
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
            str(self.engine_lease_minutes if task_id == "ENGINE" else self.lease_minutes),
            "--signal=B:USR1@600",
            "--output",
            str(self.folder / "attempts" / (attempt_id + "_%j.log")),
        ]
        if spec["gpus"]:
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
        task["attempts"].append(attempt)
        task["status"] = "SUBMITTING"
        self._save("SUBMISSION_INTENT_DURABLE", task_id=task_id, attempt_id=attempt_id)
        try:
            receipt = self.backend.submit(command)
            attempt["submission_receipt"] = receipt
            require(receipt["returncode"] == 0, "SBATCH_RESULT_UNKNOWN")
            match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", receipt["stdout"])
            require(match, "AMBIGUOUS_SBATCH_STDOUT")
            attempt["job_id"] = match.group(1)
            attempt["status"] = task["status"] = "REGISTERED"
            self._save("ALLOCATION_BOUND_BEFORE_RELEASE", task_id=task_id)
            self._release(task_id, attempt)
        except Exception as exc:
            attempt["status"] = task["status"] = "UNKNOWN"
            self._save("SUBMISSION_UNKNOWN_NO_RETRY", task_id=task_id, error=repr(exc))
        return True

    def _dependencies_ready(self, spec):
        accepted = SCIENCE_TERMINAL if spec.get("terminal_dependencies") else {"COMPLETE"}
        return all(self.state["tasks"][key]["status"] in accepted for key in spec["dependencies"])

    def _science_terminal(self):
        return all(
            self.state["tasks"][key]["status"] in SCIENCE_TERMINAL
            for key, spec in self.registration["tasks"].items()
            if spec["phase"] == "S4"
        )

    def _matrix_final(self):
        require(self._science_terminal(), "TEST_SEALED_UNTIL_ALL_SCIENCE_TERMINAL")
        rows = []
        for row in read_json(self.root / "manifests/RUN_MATRIX.json"):
            task = self.state["tasks"][row["run_id"]]
            rows.append(
                {
                    **row,
                    "status": task["status"],
                    "attempts": task["attempts"],
                    "failure": task.get("blocker"),
                    "completion_marker_sha256": task.get("completion_marker_sha256"),
                }
            )
        receipt = {
            "plan_id": PLAN_ID,
            "registration_hash": digest(self.registration),
            "all_runs_terminal": True,
            "runs": rows,
        }
        target = self.root / "RUN_MATRIX_FINAL.json"
        if target.exists():
            require(read_json(target) == receipt, "FINAL_MATRIX_CHANGED")
        else:
            atomic_json(target, receipt, exclusive=True)
        return receipt

    def _accounting(self, *, gpu_only=False):
        allocations = []
        for task_id, task in self.state["tasks"].items():
            if gpu_only and not self.registration["tasks"][task_id]["gpus"]:
                continue
            for attempt in task["attempts"]:
                allocations.append(
                    {
                        "task_id": task_id,
                        "attempt_id": attempt["attempt_id"],
                        "job_id": attempt.get("job_id"),
                        "status": attempt["status"],
                        "accounting": attempt.get("accounting"),
                        "observation": attempt.get("observation"),
                    }
                )
        unresolved = [row["attempt_id"] for row in allocations if row["accounting"] is None]
        total = sum(row["accounting"]["gpu_seconds"] for row in allocations if row["accounting"])
        return {
            "plan_id": PLAN_ID,
            "policy": "ACCOUNTING_ONLY",
            "status": "UNRESOLVED" if unresolved else "COMPLETE",
            "known_actual_gpu_seconds": total,
            "known_actual_gpu_hours": total / 3600,
            "unresolved_attempt_ids": unresolved,
            "allocations": allocations,
            "cumulative_gpu_hours_limit": None,
            "research_wallclock_limit": None,
        }

    def _phase(self):
        phase = "S1_VERIFIED"
        for stage in ("S2", "S3", "S4", "S5", "S6"):
            statuses = [
                self.state["tasks"][key]["status"]
                for key, spec in self.registration["tasks"].items()
                if spec["phase"] == stage
            ]
            if not all(
                status in (SCIENCE_TERMINAL if stage == "S4" else {"COMPLETE"})
                for status in statuses
            ):
                break
            phase = stage + "_COMPLETE"
        stopped = (self.root / "STOP").exists() or any(
            task["status"] == "STOPPED" for task in self.state["tasks"].values()
        )
        self.state["phase"] = (
            "STOPPED" if stopped else ("RELEASED" if phase == "S6_COMPLETE" else phase)
        )
        self.state["test_sealed"] = not self._science_terminal()
        self.state["technical_blockers"] = [
            key for key, task in self.state["tasks"].items() if task["status"] == "BLOCKED"
        ]
        self.state["technical_failed_runs"] = [
            key for key, task in self.state["tasks"].items() if task["status"] == "TECHNICAL_FAILED"
        ]
        self.state["accounting"] = self._accounting()

    def resume_stopped(self):
        """Explicit CLI action only; ordinary ticks never clear a requested stop."""
        with (
            process_lease(self.root.parent / ".sr_f1_project_controller.lock"),
            process_lease(self.folder / "controller.lock"),
        ):
            self._load()
            require(not (self.root / "STOP").exists(), "REMOVE_STOP_BEFORE_EXPLICIT_RESUME")
            resumed = []
            for task_id, task in self.state["tasks"].items():
                if task["status"] == "STOPPED":
                    require(
                        self._marker_valid(task_id, task["attempts"][-1], checkpoint=True),
                        "STOPPED_FULL_STATE_REQUIRED",
                    )
                    task["status"] = "RETRYABLE"
                    resumed.append(task_id)
            self._phase()
            self._save("EXPLICIT_STOPPED_PATH_RESUME", task_ids=resumed)
            return resumed

    def tick(self):
        # All SR-F1 run roots in the authorized project parent use this lock.
        # Existing F2 allocations are counted through the whole-owner queue.
        with (
            process_lease(self.root.parent / ".sr_f1_project_controller.lock"),
            process_lease(self.folder / "controller.lock"),
        ):
            self._load()
            for task_id, task in self.state["tasks"].items():
                if task["status"] in LIVE:
                    self._observe(task_id)
            if not (self.root / "STOP").exists() and not any(
                task["status"] == "STOPPED" for task in self.state["tasks"].values()
            ):
                for task_id in self.registration["task_order"]:
                    spec, task = self.registration["tasks"][task_id], self.state["tasks"][task_id]
                    if task["status"] not in {
                        "WAITING",
                        "RETRYABLE",
                    } or not self._dependencies_ready(spec):
                        continue
                    if spec["phase"] in {"S5", "S6"}:
                        self._matrix_final()
                    if spec["phase"] == "S6":
                        accounting = self._accounting(gpu_only=True)
                        if accounting["unresolved_attempt_ids"]:
                            continue
                        atomic_json(self.root / "GPU_ACCOUNTING.json", accounting)
                    if not self._submit(task_id):
                        break
            self._phase()
            self._save("TICK_FINISHED")
            return copy.deepcopy(self.state)
