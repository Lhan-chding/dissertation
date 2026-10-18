"""Operational-only reconciliation of explicit no-allocation submissions.

Scientific code, task claims, worker commands, and the frozen extension are not
replaced. Original immutable receipts stay in place; mutable STATE transitions
have byte-exact before images in an append-only operational journal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

SCHEMA = 1
QOS_REJECTION = (
    "sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
    "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy "
    "(job submit limit, user's size and/or time limits)"
)
LOCAL_DEFERRED = "SRF3CE_LOCAL_CAPACITY_DEFERRED: sbatch was not invoked\n"
ACTIONS = ["reconcile_verified_completion", "retry_explicit_no_allocation", "fresh_capacity_guard"]


def _hash(body):
    return hashlib.sha256(body).hexdigest()


def _read(path):
    return json.loads(Path(path).read_bytes())


def _raw(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise PermissionError("Preserved original bytes differ")
        return
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def check_authorization(root, path):
    root = Path(root).resolve(strict=True)
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(root / "operations"):
        raise PermissionError("Repair authorization must be retained under this run's operations")
    body = path.read_bytes()
    value = json.loads(body)
    expected = dict(
        schema_version=SCHEMA,
        scope="SRF3CE_OPERATIONAL_SUBMISSION_REPAIR",
        authorized_actions=ACTIONS,
        scientific_settings_changed=False,
        run_identity_sha256=_hash((root / "RUN_IDENTITY.json").read_bytes()),
        execution_freeze_sha256=_hash((root / "EXECUTION_FREEZE.json").read_bytes()),
    )
    if any(value.get(k) != v for k, v in expected.items()) or not value.get("user_request"):
        raise PermissionError("Repair authorization/frozen identity differs")
    prior = value.get("predecessor", {})
    old_path = Path(prior.get("identity_path", "")).resolve(strict=True)
    if not old_path.is_relative_to(root / "operations") or (
        _hash(old_path.read_bytes()) != prior.get("identity_sha256")
    ):
        raise PermissionError("Previous operation identity differs")
    old = _read(old_path)
    if (
        old.get("root") != str(root)
        or old.get("run_identity_sha256") != expected["run_identity_sha256"]
    ):
        raise PermissionError("Previous operation belongs to another scientific run")
    sequence = prior.get("sequence")
    if type(sequence) is not int or sequence < 1 or not str(prior.get("job_id", "")).isdigit():
        raise PermissionError("Repair predecessor identity is incomplete")
    directory = root / "operations/dispatcher/attempts" / f"{sequence:04d}"
    records = prior.get("records", {})
    if (
        not {"INTENT.json", "STATE.json", "STARTED.json", "END.json", "RELEASE.json", "run.sbatch"}
        <= records.keys()
    ):
        raise PermissionError("Repair predecessor originals are incomplete")
    for name, digest in records.items():
        if Path(name).name != name or _hash((directory / name).read_bytes()) != digest:
            raise PermissionError("Repair predecessor original changed")
    return dict(
        authorization_path=str(path), authorization_sha256=_hash(body), schema_version=SCHEMA
    )


def explicit_rejection(response):
    return (
        isinstance(response, dict)
        and type(response.get("returncode")) is int
        and response["returncode"] == 1
        and response.get("stdout") == ""
        and response.get("stderr", "").strip() == QOS_REJECTION
    )


class Repair:
    def __init__(self, value, api, io):
        self.value, self.api, self.io = value, api, io
        self.root = Path(value["root"])
        self.base = self.root / "operations/submission_reconciliation"
        binding = value["submission_repair"]
        if check_authorization(self.root, binding["authorization_path"]) != binding:
            raise PermissionError("Bound repair authorization changed")
        self.authorization = _read(binding["authorization_path"])
        self.raw_slurm = api.slurm
        self.slurm = GuardedSlurm(self)
        api.slurm = api.controller.slurm = self.slurm

    def rows(self):
        rows = self.io.state_rows(self.root)
        pool = getattr(self.api, "task_pool", None)
        if pool is not None and hasattr(pool, "filter_rows"):
            rows = pool.filter_rows(rows)
        for intent, state in rows:
            directory = self.root / "orchestration/jobs" / state["key"]
            if not re.fullmatch(r"job-[a-f0-9]{16}", state["key"]) or (
                state.get("intent_sha256") != self.io.sha(directory / "INTENT.json")
                or intent.get("key") != state["key"]
                or state.get("run_identity_sha256") != self.value["run_identity_sha256"]
                or any(
                    state.get(k) != intent.get(k)
                    for k in ("gpu_count", "qos", "run_identity_sha256")
                )
                or (
                    intent.get("execution_entry_schema") in {2, 3}
                    and (
                        type(state.get("gpu_count")) is not int
                        or type(intent.get("gpu_count")) is not int
                        or type(intent.get("max_tasks_per_job")) is not int
                        or intent["max_tasks_per_job"]
                        < (0 if intent.get("kind") == "release" else 1)
                        or not 0 <= state["gpu_count"] <= intent["max_tasks_per_job"]
                    )
                )
            ):
                raise PermissionError("Reconciliation ledger binding differs")
            if intent.get("execution_entry_schema") in {2, 3}:
                self._pool_capacity(intent)
        return rows

    def _pool_capacity(self, intent):
        """Validate a historical claim against its retained capacity, not today's."""
        capacity = getattr(self.api, "capacity", None)
        limits = intent.get("capacity_limits")
        if (
            capacity is None
            or not isinstance(limits, dict)
            or type(intent.get("max_tasks_per_job")) is not int
            or intent["max_tasks_per_job"] < (0 if intent.get("kind") == "release" else 1)
            or intent["max_tasks_per_job"] != limits.get("max_tasks_per_job")
        ):
            raise PermissionError("Pool repair requires bound scheduler capacity")
        capacity.verify_snapshot(intent)

    def _attempts(self, key):
        base = self.base / key / "attempts"
        paths = sorted(base.glob("[0-9][0-9][0-9][0-9]"))
        if [p.name for p in paths] != [f"{i:04d}" for i in range(1, len(paths) + 1)]:
            raise PermissionError("Submission journal has a gap or fork")
        return paths

    def _no_execution(self, intent, state):
        directory = self.root / "orchestration/jobs" / state["key"]
        if state.get("job_id") or state.get("status") != "SUBMISSION_UNKNOWN":
            raise PermissionError("Only an unassigned submission can be retried")
        forbidden = [
            "ALLOCATION_VERIFIED.json",
            "RELEASE_INTENT.json",
            "RELEASE.json",
            "STARTED.json",
            "COMPLETE.json",
            "TERMINAL.json",
            "SUBMISSION_EXCEPTION.json",
        ]
        if (
            any((directory / name).exists() for name in forbidden)
            or list(directory.glob("*.LAUNCH.json"))
            or list(directory.glob("*.EXIT.json"))
        ):
            raise PermissionError(
                "An allocated/started or ambiguous attempt cannot reuse its claim"
            )
        pooled = intent.get("execution_entry_schema") in {2, 3}
        tasks = intent.get("task_ids", [])
        rolling = intent.get("execution_entry_schema") == 3
        if pooled:
            pool = getattr(self.api, "task_pool", None)
            # The operations adapter owns pooled schemas; a field on an arbitrary
            # intent cannot widen the original single-task repair contract.
            registered = self.api.orchestration.TASKS
            if (
                pool is None
                or type(intent.get("execution_entry_schema")) is not int
                or not isinstance(self.value.get("task_pool"), dict)
                or not self.value["task_pool"]
                or intent.get("pool_operation_identity_path") != self.value["identity_path"]
                or intent.get("pool_operation_identity_sha256")
                != self.io.sha(self.value["identity_path"])
                or intent.get("pool_authorization_sha256")
                != self.value["task_pool"].get("authorization_sha256")
                or type(intent.get("max_tasks_per_job")) is not int
                or intent["max_tasks_per_job"] < (0 if intent.get("kind") == "release" else 1)
                or intent.get("extension_manifest_path") != self.value["extension_manifest_path"]
                or intent.get("extension_manifest_sha256")
                != self.value["extension_manifest_sha256"]
                or intent.get("kind") not in {"train", "eval", "release"}
                or (
                    rolling
                    and (
                        intent.get("kind") != "eval"
                        or intent.get("rolling_endpoints") is not True
                        or self.value["task_pool"].get("rolling_endpoints") is not True
                    )
                )
                or not isinstance(tasks, list)
                or not 1
                <= len(tasks)
                <= (
                    3 * intent["max_tasks_per_job"]
                    if rolling
                    else (1 if intent["kind"] == "release" else intent["max_tasks_per_job"])
                )
                or any(not isinstance(task, str) or task not in registered for task in tasks)
                or len(set(tasks)) != len(tasks)
                or any(registered[task]["kind"] != intent["kind"] for task in tasks)
                or type(intent.get("gpu_count")) is not int
                or type(state.get("gpu_count")) is not int
                or state["gpu_count"] != intent["gpu_count"]
                or (
                    not 1 <= intent["gpu_count"] <= len(tasks) <= 3 * intent["gpu_count"]
                    or intent["gpu_count"] > intent["max_tasks_per_job"]
                    if rolling
                    else intent["gpu_count"] != (0 if intent["kind"] == "release" else len(tasks))
                )
                or "resume_of" in intent
            ):
                raise PermissionError("Pool repair requires bound fixed unstarted science tasks")
            self._pool_capacity(intent)
        elif intent.get("kind") not in {"train", "eval", "release"} or len(tasks) != 1:
            raise PermissionError("Repair is restricted to one existing frozen science task")
        if intent["kind"] == "release" and (self.root / "release").exists():
            raise PermissionError("Release output exists for the rejected claim")
        if (self.root / "accounting/observability" / state["key"]).exists() or any(
            (self.root / "tasks" / task).exists() for task in tasks
        ):
            raise PermissionError("Task/observer evidence prevents an unstarted retry")
        # Only the original single-task TIMEOUT successor can reuse outputs.
        # Pooled allocation failures require an explicit handoff, never replay.
        if "resume_of" in intent:
            self.api.orchestration.runtime_resume_receipt(self.root, intent)
            return
        for task in tasks:
            model = self.api.orchestration.TASKS[task].get("model_id")
            if model and intent["kind"] == "train" and (self.root / "runs" / model).exists():
                raise PermissionError("Training output exists for the rejected claim")
            if (
                model
                and intent["kind"] == "eval"
                and (self.root / "sealed/endpoint" / model).exists()
            ):
                raise PermissionError("Evaluation output exists for the rejected claim")
        # Rejection classification also runs while waiting for capacity.
        # Keep that path small; submit() and retry_pending() retain the full
        # frozen intent/science verification at the submission boundary.
        if pooled and pool.check() != self.value["task_pool"]:
            raise PermissionError("Pool repair authorization differs from its operation")

    def rejection(self, intent, state):
        """Positive retained evidence only; absence from Slurm is never sufficient."""
        if state.get("job_id") or state.get("status") != "SUBMISSION_UNKNOWN":
            return None
        self._no_execution(intent, state)
        directory = self.root / "orchestration/jobs" / state["key"]
        paths = self._attempts(state["key"])
        if paths:
            latest = paths[-1]
            if not (latest / "INTENT.json").is_file() or not (latest / "RESULT.json").is_file():
                return None
            journal, result = _read(latest / "INTENT.json"), _read(latest / "RESULT.json")
            before = (latest / "BEFORE_STATE.raw.json").read_bytes()
            original = directory / "SUBMISSION.json"
            if (
                journal.get("original_intent_sha256") != self.io.sha(directory / "INTENT.json")
                or journal.get("job_key") != state["key"]
                or journal.get("before_state_sha256") != _hash(before)
                or json.loads(before) != state
                or journal.get("command") != intent["sbatch"]
                or journal.get("operation_identity_sha256")
                != self.io.sha(self.value["identity_path"])
                or result.get("submission_kind") != journal.get("submission_kind")
                or (
                    journal.get("submission_kind") == "RETRY"
                    and (
                        not original.is_file()
                        or journal.get("original_submission_sha256") != self.io.sha(original)
                    )
                )
            ):
                raise PermissionError("Submission journal identity/original differs")
            if result.get("sbatch_invoked") is True:
                marker = latest / "MAY_HAVE_SUBMITTED.json"
                if not marker.is_file() or _read(marker).get("command") != intent["sbatch"]:
                    raise PermissionError("Invoked submission lacks its durable marker")
            if (
                result.get("status") == "EXPLICIT_QOS_REJECTION"
                and result.get("sbatch_invoked") is True
                and explicit_rejection(result.get("response"))
            ):
                return dict(
                    kind="EXPLICIT_QOS_REJECTION",
                    path=str(latest / "RESULT.json"),
                    sha256=self.io.sha(latest / "RESULT.json"),
                )
            if (
                result.get("status") == "LOCAL_CAPACITY_DEFERRED"
                and result.get("sbatch_invoked") is False
                and not (latest / "MAY_HAVE_SUBMITTED.json").exists()
            ):
                if result.get("response") != dict(returncode=75, stdout="", stderr=LOCAL_DEFERRED):
                    raise PermissionError("Local deferral response changed")
                return dict(
                    kind="LOCAL_CAPACITY_DEFERRED",
                    path=str(latest / "RESULT.json"),
                    sha256=self.io.sha(latest / "RESULT.json"),
                )
            return None
        original = directory / "SUBMISSION.json"
        if original.is_file() and explicit_rejection(_read(original)):
            return dict(
                kind="EXPLICIT_QOS_REJECTION", path=str(original), sha256=self.io.sha(original)
            )
        return None

    def effective_states(self, rows=None):
        rows = self.rows() if rows is None else rows
        pool = getattr(self.api, "task_pool", None)
        if pool is not None and hasattr(pool, "filter_rows"):
            rows = pool.filter_rows(rows)
        return [state for intent, state in rows if not self.rejection(intent, state)]

    def verify_driver_predecessor(self, prior, old_intent, old_state, queue):
        bound = self.authorization["predecessor"]
        if prior.name != f"{bound['sequence']:04d}" or old_state.get("job_id") != bound["job_id"]:
            raise PermissionError("Unregistered failed dispatcher predecessor")
        if old_intent.get("operation_identity_sha256") != bound["identity_sha256"] or (
            old_state.get("intent_sha256") != self.io.sha(prior / "INTENT.json")
            or old_state.get("status") != "SUBMITTED"
            or old_state["job_id"] in queue
            or _read(prior / "END.json").get("status") != "FAILED"
        ):
            raise PermissionError("Failed dispatcher was not retained and terminal")
        terminal = self.raw_slurm.terminal(old_state["job_id"])
        expected = dict(
            job_id=bound["job_id"],
            state="FAILED",
            exit_code="1:0",
            user=self.value["user"],
            account="rose",
            qos=old_intent["qos"],
            name=old_intent["job_name"],
        )
        if terminal != expected:
            raise PermissionError("Failed dispatcher live accounting differs")
        self.io.save(
            self.base / "DRIVER_PREDECESSOR_VERIFIED.json",
            dict(
                predecessor=bound,
                terminal=terminal,
                authorization_sha256=self.value["submission_repair"]["authorization_sha256"],
            ),
            immutable=True,
        )
        return terminal

    def reconcile_completed(self):
        with self.api.orchestration.lease(self.root / "orchestration"):
            queue = self.raw_slurm.queue(self.value["user"])
            changed = []
            for intent, state in self.rows():
                job = state.get("job_id")
                if not job or job in queue or state["status"] in self.api.orchestration.FINAL:
                    continue
                terminal = self.raw_slurm.terminal(job)
                if not terminal or (terminal.get("state"), terminal.get("exit_code")) != (
                    "COMPLETED",
                    "0:0",
                ):
                    continue
                expected = dict(
                    job_id=job,
                    state="COMPLETED",
                    exit_code="0:0",
                    user=self.value["user"],
                    account="rose",
                    qos=intent["qos"],
                    name=intent["job_name"],
                )
                if terminal != expected:
                    raise PermissionError("Completed accounting identity differs")
                directory = self.root / "orchestration/jobs" / state["key"]
                journal = self.base / state["key"] / "completion"
                before = (directory / "STATE.json").read_bytes()
                _raw(journal / "BEFORE_STATE.raw.json", before)
                self.io.save(directory / "TERMINAL.json", terminal, immutable=True)
                tasks = self.api.controller.verify_job(state)
                receipt = dict(
                    status="VERIFIED_COMPLETION_ONLY",
                    job_id=job,
                    task_ids=tasks,
                    before_state_sha256=_hash(before),
                    original_intent_sha256=self.io.sha(directory / "INTENT.json"),
                    terminal_sha256=self.io.sha(directory / "TERMINAL.json"),
                    task_receipts={
                        t: self.io.sha(self.api.orchestration.receipt_path(self.root, t))
                        for t in tasks
                    },
                    operation_identity_sha256=self.io.sha(self.value["identity_path"]),
                    new_model_calls=0,
                )
                self.io.save(journal / "RECONCILIATION.json", receipt, immutable=True)
                updated = dict(state, status="COMPLETE")
                self.io.save(directory / "STATE.json", updated)
                changed.append(state["key"])
            return changed

    def _check_no_matching_allocation(self, intent, queue):
        if any(
            row.get("JobName") == intent["job_name"] or row.get("Comment") == intent["comment"]
            for row in queue.values()
        ):
            raise PermissionError("Rejected intent has a matching live allocation")
        # A positive rejection is required independently of these corroborating queries.
        start = intent["created"][:10]
        command = [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-S",
            start,
            "-u",
            self.value["user"],
            "--format=JobIDRaw,JobName%150,Comment%200",
        ]
        result = self.raw_slurm.run(command)
        if result.returncode:
            raise RuntimeError("Accounting cross-check unavailable; no retry")
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            row = line.split("|")
            if len(row) < 3:
                raise PermissionError("Malformed accounting cross-check")
            if row[1] == intent["job_name"] or row[2] == intent["comment"]:
                raise PermissionError("Rejected intent has a matching accounting allocation")
        return dict(
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def retry_pending(self):
        """At most one submission per rejected claim and observed capacity transition."""
        if not any(s.get("status") == "SUBMISSION_UNKNOWN" for _, s in self.rows()):
            return None
        with self.api.orchestration.lease(self.root / "orchestration"):
            # Full frozen authentication precedes the final lightweight capacity query.
            self.api.controller.identity()
            verified = {s["key"]: s for s in self.api.controller.attempts()}
            rows = self.rows()
            intents = {state["key"]: intent for intent, state in rows}
            # A repair candidate must not bypass a different unresolved job. Keep
            # the frozen controller's global blocker and verified-resume semantics.
            for blocked in self.api.controller.blocking_attempts(list(verified.values())):
                if self.rejection(intents[blocked["key"]], blocked) is None:
                    raise RuntimeError(
                        "Submission outcome remains unknown or blocked; not retried: "
                        + blocked["key"]
                    )
            for intent, state in rows:
                if state.get("job_id") or state["status"] != "SUBMISSION_UNKNOWN":
                    continue
                evidence = self.rejection(intent, state)
                if evidence is None:
                    raise RuntimeError(
                        "Submission outcome remains unknown; not retried: " + state["key"]
                    )
                if state != verified[state["key"]]:
                    raise PermissionError("Submission state changed during reconciliation")
                self.api.extension._verify_intent(self.root, intent)
                queue = self.raw_slurm.queue(self.value["user"])
                crosscheck = self._check_no_matching_allocation(intent, queue)
                result = self.slurm.submit(
                    intent["sbatch"], evidence=evidence, accounting=crosscheck
                )
                match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout)
                if result.returncode or not match:
                    if (
                        explicit_rejection(
                            dict(
                                returncode=result.returncode,
                                stdout=result.stdout,
                                stderr=result.stderr,
                            )
                        )
                        or result.stderr == LOCAL_DEFERRED
                    ):
                        return "WAIT_CAPACITY_FOR_REJECTED_SUBMISSION"
                    raise RuntimeError("Retry outcome unknown; no further submission")
                updated = dict(state, job_id=match[1], status="HELD")
                self.io.save(
                    self.root / "orchestration/jobs" / state["key"] / "STATE.json", updated
                )
                self.api.controller.release(updated, intent)
                if updated.get("status") != "SUBMITTED":
                    raise RuntimeError("Retry release outcome is unresolved")
            return None

    def known_rejections_only(self, attempts):
        by_key = {state["key"]: intent for intent, state in self.rows()}
        blockers = self.api.controller.blocking_attempts(attempts)
        return bool(blockers) and all(self.rejection(by_key[s["key"]], s) for s in blockers)

    def prepare(self):
        bound = self.authorization["predecessor"]
        prior = self.root / "operations/dispatcher/attempts" / f"{bound['sequence']:04d}"
        self.verify_driver_predecessor(
            prior,
            _read(prior / "INTENT.json"),
            _read(prior / "STATE.json"),
            self.raw_slurm.queue(self.value["user"]),
        )
        completed = self.reconcile_completed()
        pending = [
            dict(key=s["key"], evidence=self.rejection(i, s))
            for i, s in self.rows()
            if s.get("status") == "SUBMISSION_UNKNOWN"
        ]
        if any(row["evidence"] is None for row in pending):
            raise RuntimeError("Unknown submission cannot be reconciled")
        return dict(
            status="PREPARED_NO_NEW_SUBMISSIONS",
            completed=completed,
            pending=pending,
            new_model_calls=0,
        )


class GuardedSlurm:
    def __init__(self, repair):
        self.repair = repair

    def __getattr__(self, name):
        return getattr(self.repair.raw_slurm, name)

    def run(self, args):
        if args and args[0] == "sbatch":
            return self.submit(args)
        if args[:2] == ["scontrol", "release"]:
            self.release_guard(args[2])
        return self.repair.raw_slurm.run(args)

    def _candidate(self, args):
        r = self.repair
        script = Path(args[-1]).resolve(strict=True)
        if script.name != "run.sbatch" or not (
            script.is_relative_to(r.root / "orchestration/jobs")
            or script.is_relative_to(r.root / "operations/dispatcher/attempts")
        ):
            raise PermissionError("Unregistered guarded submission path")
        intent = _read(script.with_name("INTENT.json"))
        if intent.get("sbatch") != args or intent.get("script_sha256") != r.io.sha(script):
            raise PermissionError("Guarded submit command/script differs")
        if intent.get("kind"):
            state = _read(script.with_name("STATE.json"))
            r._no_execution(intent, state)
        else:
            if intent.get("operation_identity_sha256") != r.io.sha(r.value["identity_path"]):
                raise PermissionError("CPU operation identity differs")
            dispatcher_qos = r.value.get("dispatcher_qos", r.value["qos"])
            if intent.get("qos") != dispatcher_qos or intent.get("gpu_count") != 0:
                raise PermissionError("CPU dispatcher QoS or GPU request differs")
            state = dict(
                key="dispatcher", status="SUBMISSION_UNKNOWN", gpu_count=0, qos=dispatcher_qos
            )
        return script.parent, intent, state

    def _ordinary_dispatcher(self, intent):
        r = self.repair
        return (
            not intent.get("kind")
            and r.value.get("dispatcher_qos", r.value["qos"]) != r.value["qos"]
        )

    def _dispatcher_boundary(self, intent, *, candidate, job=None):
        """Count only CPU dispatcher capacity, retaining live evidence at each write."""
        r = self.repair
        qos = r.value.get("dispatcher_qos", r.value["qos"])
        if (
            intent.get("operation_identity_sha256") != r.io.sha(r.value["identity_path"])
            or intent.get("qos") != qos
            or intent.get("gpu_count") != 0
        ):
            raise PermissionError("CPU dispatcher operation or allocation differs")
        capacity = getattr(r.api, "capacity", None)
        if capacity is None:
            raise PermissionError("Ordinary CPU dispatcher requires live scheduler capacity")
        limits = capacity.dispatcher_limits(qos, fresh=True)
        queue = r.raw_slurm.queue(r.value["user"])
        if candidate and any(
            row.get("JobName") == intent["job_name"] or row.get("Comment") == intent["comment"]
            for row in queue.values()
        ):
            raise PermissionError("Candidate already has a live allocation")
        if job is not None and (
            job not in queue
            or queue[job].get("QOS") != qos
            or queue[job].get("Command") != intent["script"]
        ):
            raise PermissionError("Held CPU dispatcher allocation differs")
        headroom = capacity.dispatcher_headroom(queue, limits)
        return queue, limits, headroom, headroom["submitted"] >= (1 if candidate else 0)

    def _fresh_capacity(self):
        capacity = getattr(self.repair.api, "capacity", None)
        if capacity is None:
            if "task_pool" in self.repair.value:
                raise PermissionError("Pool submission requires live scheduler capacity")
            return None, None
        return capacity, capacity.limits(fresh=True)

    @staticmethod
    def _within_capacity(usage, capacity, limits, gpu_count):
        if capacity is None:
            # Preserve the legacy operation's frozen allocation contract.
            return usage["gpus"] <= 5 and usage["submitted"] <= 5
        remaining = capacity.headroom(usage, limits)
        return (
            remaining["gpus"] >= 0
            and remaining["submitted"] >= 0
            and gpu_count <= limits["max_tasks_per_job"]
        )

    def submit(self, args, *, evidence=None, accounting=None):
        r = self.repair
        directory, intent, state = self._candidate(args)
        if self._ordinary_dispatcher(intent):
            queue, limits, headroom, allowed = self._dispatcher_boundary(intent, candidate=True)
            r.io.save(
                directory / "QUEUE_IMMEDIATELY_BEFORE_SUBMIT.json",
                dict(
                    at=time.time(),
                    jobs=queue,
                    dispatcher_capacity=limits,
                    dispatcher_headroom=headroom,
                    includes_candidate=True,
                    submitted_after_candidate=headroom["submitted"] - 1,
                ),
                immutable=True,
            )
            if not allowed:
                return SimpleNamespace(returncode=75, stdout="", stderr=LOCAL_DEFERRED)
            return r.raw_slurm.run(args)
        rows = r.rows()
        effective = r.effective_states(rows)
        science = bool(intent.get("kind"))
        if not science or state["key"] not in {s["key"] for s in effective}:
            effective.append(state)
        journal = None
        if science:
            paths = r._attempts(state["key"])
            if paths and evidence is None:
                raise PermissionError(
                    "Existing submission journal requires explicit reconciliation"
                )
            if (directory / "SUBMISSION.json").exists() and evidence is None:
                raise PermissionError("Original submission cannot be silently repeated")
            journal = r.base / state["key"] / "attempts" / f"{len(paths) + 1:04d}"
            before = (directory / "STATE.json").read_bytes()
            _raw(journal / "BEFORE_STATE.raw.json", before)
            kind = "RETRY" if (directory / "SUBMISSION.json").exists() else "INITIAL"
            r.io.save(
                journal / "INTENT.json",
                dict(
                    schema_version=SCHEMA,
                    job_key=state["key"],
                    submission_kind=kind,
                    original_intent_sha256=r.io.sha(directory / "INTENT.json"),
                    original_submission_sha256=r.io.sha(directory / "SUBMISSION.json")
                    if kind == "RETRY"
                    else None,
                    before_state_sha256=_hash(before),
                    command=args,
                    evidence=evidence,
                    accounting=accounting,
                    operation_identity_sha256=r.io.sha(r.value["identity_path"]),
                    at=time.time(),
                ),
                immutable=True,
            )
        # Capacity may require scheduler queries; observe the queue afterwards.
        # No full source, task, or checkpoint hashing after this queue observation.
        capacity, limits = self._fresh_capacity()
        queue = r.raw_slurm.queue(r.value["user"])
        if science and any(
            v.get("JobName") == intent["job_name"] or v.get("Comment") == intent["comment"]
            for v in queue.values()
        ):
            raise PermissionError("Candidate already has a live allocation")
        usage = r.api.orchestration.teacher_usage(queue, effective, r.value["user"])
        target = journal if science else directory
        r.io.save(
            target / "QUEUE_IMMEDIATELY_BEFORE_SUBMIT.json",
            dict(
                at=time.time(),
                jobs=queue,
                teacher_usage=usage,
                includes_candidate=True,
                **({"scheduler_capacity": limits} if limits is not None else {}),
            ),
            immutable=True,
        )
        if not self._within_capacity(usage, capacity, limits, state["gpu_count"]):
            response = dict(returncode=75, stdout="", stderr=LOCAL_DEFERRED)
            if science:
                r.io.save(
                    journal / "RESULT.json",
                    dict(
                        status="LOCAL_CAPACITY_DEFERRED",
                        submission_kind=kind,
                        sbatch_invoked=False,
                        response=response,
                        at=time.time(),
                    ),
                    immutable=True,
                )
            return SimpleNamespace(**response)
        if science:
            r.io.save(
                journal / "MAY_HAVE_SUBMITTED.json",
                dict(at=time.time(), command=args),
                immutable=True,
            )
        try:
            result = r.raw_slurm.run(args)
        except Exception as error:
            if science:
                r.io.save(
                    journal / "EXCEPTION.json",
                    dict(error=repr(error), at=time.time()),
                    immutable=True,
                )
            raise
        if science:
            response = dict(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
            )
            match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout)
            status = (
                "ACCEPTED"
                if result.returncode == 0 and match
                else ("EXPLICIT_QOS_REJECTION" if explicit_rejection(response) else "UNKNOWN")
            )
            record = dict(
                status=status,
                submission_kind=kind,
                sbatch_invoked=True,
                response=response,
                at=time.time(),
            )
            if status == "ACCEPTED":
                record["job_id"] = match[1]
            r.io.save(journal / "RESULT.json", record, immutable=True)
            if status == "ACCEPTED":
                r.io.save(
                    journal / "ACCEPTED.json",
                    dict(
                        job_id=match[1], original_intent_sha256=r.io.sha(directory / "INTENT.json")
                    ),
                    immutable=True,
                )
        return result

    def release_guard(self, job):
        r = self.repair
        if r.value.get("dispatcher_qos", r.value["qos"]) != r.value["qos"]:
            fields = r.raw_slurm.show(job)
            candidate_script = Path(fields["Command"]).resolve(strict=True)
            if candidate_script.is_relative_to(r.root / "operations/dispatcher/attempts"):
                intent = _read(candidate_script.with_name("INTENT.json"))
                if intent.get("kind"):
                    raise PermissionError("CPU dispatcher path contains a scientific intent")
                queue, limits, headroom, allowed = self._dispatcher_boundary(
                    intent, candidate=False, job=job
                )
                if not allowed:
                    raise PermissionError(
                        "Fresh CPU release capacity exceeded; allocation remains held"
                    )
                r.io.save(
                    candidate_script.with_name("QUEUE_IMMEDIATELY_BEFORE_RELEASE.json"),
                    dict(
                        at=time.time(),
                        job_id=job,
                        jobs=queue,
                        dispatcher_capacity=limits,
                        dispatcher_headroom=headroom,
                    ),
                    immutable=True,
                )
                return
        effective = r.effective_states()
        capacity, limits = self._fresh_capacity()
        queue = r.raw_slurm.queue(r.value["user"])
        if job not in queue:
            raise RuntimeError("Held allocation disappeared before release")
        usage = r.api.orchestration.teacher_usage(queue, effective, r.value["user"])
        script = Path(queue[job]["Command"]).resolve(strict=True)
        if not (
            script.is_relative_to(r.root / "orchestration/jobs")
            or script.is_relative_to(r.root / "operations/dispatcher/attempts")
        ):
            raise PermissionError("Release is outside this registered run")
        intent = _read(script.with_name("INTENT.json"))
        gpu_count = intent.get("gpu_count") if intent.get("kind") else 0
        if type(gpu_count) is not int or gpu_count < 0:
            raise PermissionError("Release allocation has an invalid GPU count")
        if not self._within_capacity(usage, capacity, limits, gpu_count):
            raise PermissionError("Fresh release capacity exceeded; allocation remains held")
        r.io.save(
            script.with_name("QUEUE_IMMEDIATELY_BEFORE_RELEASE.json"),
            dict(
                at=time.time(),
                job_id=job,
                jobs=queue,
                teacher_usage=usage,
                **({"scheduler_capacity": limits} if limits is not None else {}),
            ),
            immutable=True,
        )
