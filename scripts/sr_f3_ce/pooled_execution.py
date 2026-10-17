#!/usr/bin/env python3
"""Authorized packing of fixed tasks into real, bounded Slurm allocations.

The frozen configuration and native observation deployment stay unchanged.  This
operation adds an explicitly different entry schema and preserves the installed
native Controller class object, so newly constructed endpoint/release verifiers
understand both historical and pooled allocations.  Each worker still executes
the original native observation wrapper and scientific worker on one GPU.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

SCHEMA = 1
ENTRY_SCHEMA = 2
SCOPE = "SRF3CE_FIXED_TASK_POOL"
MAX_TASKS = 5
_OWNER = "_srf3ce_task_pool_owner"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_authorization(root, path):
    root, path = Path(root).resolve(strict=True), Path(path).resolve(strict=True)
    if not path.is_relative_to(root / "operations") or not path.is_file():
        raise PermissionError("Task pool authorization must be a retained operational file")
    value = _read(path)
    expected = dict(
        schema_version=SCHEMA,
        scope=SCOPE,
        max_tasks_per_job=MAX_TASKS,
        scientific_settings_changed=False,
        preserve_history=True,
        no_duplicate_tasks=True,
        one_task_per_gpu=True,
        run_identity_sha256=_sha(root / "RUN_IDENTITY.json"),
        execution_freeze_sha256=_sha(root / "EXECUTION_FREEZE.json"),
        config_sha256=_sha(root / "orchestration/CONFIG.json"),
    )
    if any(type(value.get(k)) is not type(v) or value[k] != v for k, v in expected.items()):
        raise PermissionError("Task pool authorization differs from the fixed experiment")
    if not isinstance(value.get("user_request"), str) or not value["user_request"].strip():
        raise PermissionError("Explicit task pool user authorization is required")
    if not re.fullmatch(r"[a-f0-9]{64}", str(value.get("extension_manifest_sha256", ""))):
        raise PermissionError("Task pool authorization must name the retained native extension")
    return dict(
        authorization_path=str(path),
        authorization_sha256=_sha(path),
        max_tasks_per_job=MAX_TASKS,
        scientific_settings_changed=False,
    )


def quota_usage(root, value, queue, attempts, user, *, original, tasks):
    """Extend only authenticated off-queue pool reservations to three through five GPUs.

    Real queue rows and legacy reservation semantics stay with the frozen
    implementation. This helper reads only small immutable allocation/binding
    records, and is also usable by the read-only monitor without installing a
    Controller or invoking full scientific validation.
    """
    legacy, expanded = [], []
    for state in attempts:
        if state.get("status") in {"COMPLETE", "FAILED"} or state.get("job_id") in queue:
            legacy.append(state)
            continue
        count = state.get("gpu_count")
        if type(count) is not int or count not in range(3, MAX_TASKS + 1):
            legacy.append(state)  # Preserve the original rejection of malformed counts.
            continue
        expanded.append(state)
    if expanded:
        root = Path(root).resolve(strict=True)
        identity_path = Path(value["identity_path"]).resolve(strict=True)
        commit = value.get("operation_commit", "")
        binding = value.get("task_pool", {})
        if not re.fullmatch(r"[a-f0-9]{40}", str(commit)) or (
            identity_path != root / "operations" / commit / "OPERATION_IDENTITY.json"
            or check_authorization(root, binding.get("authorization_path", "")) != binding
        ):
            raise PermissionError("Expanded GPU reservation lacks its operational authorization")
        identity, identity_sha256 = _read(identity_path), _sha(identity_path)
        if identity.get("task_pool") != binding or (
            identity.get("root") != str(root)
            or identity.get("operation_commit") != commit
            or identity.get("run_identity_sha256") != value.get("run_identity_sha256")
        ):
            raise PermissionError("Expanded GPU reservation operation identity differs")
        seen = set()
        for state in expanded:
            key = state.get("key", "")
            if not re.fullmatch(r"job-[a-f0-9]{16}", str(key)) or key in seen:
                raise PermissionError("Expanded GPU reservation key is invalid or duplicated")
            seen.add(key)
            path = root / "orchestration/jobs" / key / "INTENT.json"
            intent = _read(path)
            selected = intent.get("task_ids")
            if (
                type(intent.get("execution_entry_schema")) is not int
                or intent["execution_entry_schema"] != ENTRY_SCHEMA
                or intent.get("key") != key
                or state.get("intent_sha256") != _sha(path)
                or not isinstance(selected, list)
                or any(not isinstance(task, str) or task not in tasks for task in selected)
                or len(selected) != len(set(selected))
                or len(selected) != state["gpu_count"]
                or type(intent.get("gpu_count")) is not int
                or intent["gpu_count"] != state["gpu_count"]
                or intent.get("kind") not in {"train", "eval"}
                or any(tasks[task]["kind"] != intent["kind"] for task in selected)
                or intent.get("user") != user
                or intent.get("pool_operation_identity_path") != str(identity_path)
                or intent.get("pool_operation_identity_sha256") != identity_sha256
                or intent.get("pool_authorization_sha256") != binding["authorization_sha256"]
                or intent.get("max_tasks_per_job") != MAX_TASKS
                or state.get("qos") != "soujanya-poria-startfund-2026-03"
                or state.get("qos") != intent.get("qos")
                or state.get("run_identity_sha256") != value.get("run_identity_sha256")
                or state.get("run_identity_sha256") != intent.get("run_identity_sha256")
            ):
                raise PermissionError("Expanded GPU reservation differs from its bound pool intent")
    usage = original(queue, legacy, user)
    return dict(
        gpus=usage["gpus"] + sum(state["gpu_count"] for state in expanded),
        submitted=usage["submitted"] + len(expanded),
    )


class Pool:
    def __init__(self, value, extension):
        self.value, self.extension = copy.deepcopy(value), extension
        self.root = Path(value["root"]).resolve(strict=True)
        self.identity_path = Path(value["identity_path"]).resolve(strict=True)
        self.operation = self.identity_path.parent
        self.source = self.operation / "pooled_execution.py"
        self.identity_sha256 = _sha(self.identity_path)
        self.context = extension._context(self.root)
        self.controller_class = self.context.controller
        self.original_intent = extension._verify_intent
        self.original_submit = self.controller_class.submit
        self.original_verify_job = self.controller_class.verify_job
        self.original_prepare_resume = self.controller_class.prepare_resume
        self.original_teacher_usage = extension.teacher_usage
        self.check()

    def check(self):
        """Only small operation bindings; full science gates remain at boundaries."""
        if (self.root / "STOP").exists() or (self.root / "STOP").is_symlink():
            raise PermissionError("SR-F3-CE STOP exists")
        commit = self.value.get("operation_commit", "")
        if not re.fullmatch(r"[a-f0-9]{40}", str(commit)) or (
            self.operation != self.root / "operations" / commit
            or self.identity_path != self.operation / "OPERATION_IDENTITY.json"
            or Path(__file__).resolve(strict=True) != self.source
            or _sha(self.identity_path) != self.identity_sha256
        ):
            raise PermissionError("Task pool operation path/source identity changed")
        binding = self.value.get("task_pool")
        if (
            not isinstance(binding, dict)
            or check_authorization(self.root, binding.get("authorization_path", "")) != binding
        ):
            raise PermissionError("Registered task pool authorization differs")
        identity = _read(self.identity_path)
        sources_path = self.operation / "OPERATION_SOURCES.json"
        sources = _read(sources_path)
        if identity.get("task_pool") != binding or (
            identity.get("operation_commit") != commit
            or identity.get("root") != str(self.root)
            or identity.get("operation_sources_sha256") != _sha(sources_path)
            or sources.get("files", {}).get("pooled_execution.py") != _sha(self.source)
        ):
            raise PermissionError("Task pool source is not bound by the operation identity")
        manifest = Path(self.value["extension_manifest_path"]).resolve(strict=True)
        authorization = _read(binding["authorization_path"])
        if str(manifest) != self.context.manifest or (
            self.value.get("extension_manifest_sha256") != _sha(manifest)
            or authorization["extension_manifest_sha256"] != _sha(manifest)
            or self.context.api.orchestration.Controller is not self.controller_class
        ):
            raise PermissionError("Task pool changed the native observation deployment")
        config = _read(self.root / "orchestration/CONFIG.json")
        if config.get("pool_size") != 1 or (
            self.value.get("config_sha256") != _sha(self.root / "orchestration/CONFIG.json")
            or self.value.get("run_identity_sha256") != _sha(self.root / "RUN_IDENTITY.json")
        ):
            raise PermissionError("Task pool cannot replace the frozen configuration")
        if getattr(self.controller_class, _OWNER, None) is self and (
            self.extension.teacher_usage != self.teacher_usage
            or self.context.api.orchestration.teacher_usage != self.teacher_usage
        ):
            raise PermissionError("Task pool capacity accounting adapter changed")
        return binding

    @staticmethod
    def pooled(intent):
        return intent.get("execution_entry_schema") == ENTRY_SCHEMA

    def teacher_usage(self, queue, attempts, user):
        return quota_usage(
            self.root,
            self.value,
            queue,
            attempts,
            user,
            original=self.original_teacher_usage,
            tasks=self.extension.TASKS,
        )

    def command(self, mode, identifier, python):
        return [
            python,
            str(self.source),
            "--mode",
            mode,
            "--identity",
            str(self.identity_path),
            "--job-key" if mode == "job" else "--task-id",
            identifier,
        ]

    def key(self, selected):
        return (
            "job-"
            + self.extension.object_hash(["task-pool-v1", self.identity_sha256, selected])[:16]
        )

    def verify_intent(self, root, intent):
        if not self.pooled(intent):
            return self.original_intent(root, intent)
        if Path(root).resolve(strict=True) != self.root:
            raise PermissionError("Task pool intent belongs to another run")
        binding = self.check()
        e = self.extension
        checked = e.verify_extension(self.root, self.context.manifest, science=True)
        config = _read(self.root / "orchestration/CONFIG.json")
        tasks, kind = intent.get("task_ids"), intent.get("kind")
        if (
            not isinstance(tasks, list)
            or not tasks
            or len(tasks) != len(set(tasks))
            or (
                kind not in e.SCIENCE
                or any(not isinstance(t, str) or t not in e.TASKS for t in tasks)
                or any(e.TASKS[t]["kind"] != kind for t in tasks)
                or (len(tasks) > MAX_TASKS if kind in {"train", "eval"} else len(tasks) != 1)
                or "resume_of" in intent
            )
        ):
            raise PermissionError("Task pool requires distinct fixed tasks of one stage")
        gpu_count = len(tasks) if kind in {"train", "eval"} else 0
        key = self.key(tasks)
        expected = dict(
            execution_entry_schema=ENTRY_SCHEMA,
            key=key,
            root=str(self.root),
            code_root=str(self.root / "code"),
            python=config["python"],
            user=config["user"],
            run_identity_sha256=checked["run_identity_sha256"],
            extension_manifest_path=self.context.manifest,
            extension_manifest_sha256=checked["manifest_sha256"],
            amendment_sha256=checked["amendment_sha256"],
            execution_freeze_sha256=checked["execution_freeze_sha256"],
            pool_operation_identity_path=str(self.identity_path),
            pool_operation_identity_sha256=self.identity_sha256,
            pool_authorization_sha256=binding["authorization_sha256"],
            max_tasks_per_job=MAX_TASKS,
            gpu_count=gpu_count,
            cpus=4 * max(1, gpu_count),
            memory_gib=90 * gpu_count if gpu_count else 12,
            partition="cluster02" if gpu_count else config["cpu_policy"]["partition"],
            qos=e.TEACHER_QOS if gpu_count else config["cpu_policy"]["qos"],
            cpu_node=None if gpu_count else config["cpu_policy"]["cpu_node"],
            script=str(self.root / "orchestration/jobs" / key / "run.sbatch"),
            job_name=f"srf3ce-{kind}-{key}-LH",
            job_entry_command=self.command("job", key, config["python"]),
            worker_entry_commands={t: self.command("worker", t, config["python"]) for t in tasks},
            claim_id=e.object_hash(
                [
                    checked["run_identity_sha256"],
                    key,
                    checked["manifest_sha256"],
                    checked["amendment_sha256"],
                    self.identity_sha256,
                    binding["authorization_sha256"],
                ]
            ),
            comment="srf3ce-"
            + e.object_hash(
                [
                    str(self.root),
                    key,
                    checked["run_identity_sha256"],
                ]
            )[:24],
        )
        if any(type(intent.get(k)) is not type(v) or intent[k] != v for k, v in expected.items()):
            raise PermissionError("Actual pooled entry/source/allocation identity differs")
        if Path(intent["script"]).read_text() != e._script_body(
            self.root, self.root / "code", expected["job_entry_command"]
        ) or (
            intent.get("script_sha256") != _sha(intent["script"])
            or intent.get("sbatch") != e._sbatch_args(intent)
        ):
            raise PermissionError("Actual pooled batch script or submission fields changed")
        return checked

    def submit(self, controller, kind, *, resume_key=None, resume_verifier=None):
        e = self.extension
        if kind not in e.SCIENCE:
            return self.original_submit(
                controller, kind, resume_key=resume_key, resume_verifier=resume_verifier
            )
        if resume_key is not None:
            prior = _read(controller.directory / "jobs" / resume_key / "INTENT.json")
            if self.pooled(prior):
                raise RuntimeError("HANDOFF_REQUIRED: pooled allocations cannot be replayed")
            return self.original_submit(
                controller, kind, resume_key=resume_key, resume_verifier=resume_verifier
            )
        self.check()
        if Path(controller.root) != self.root or controller.pool_size != 1:
            raise PermissionError("Task pool controller differs from the frozen single-task config")
        checked = e.verify_extension(self.root, self.context.manifest, science=True)
        with e.lease(controller.directory):
            identity = controller.identity()
            queue = controller.slurm.queue(controller.user)
            controller.refresh(queue)
            attempts = controller.attempts()
            if controller.blocking_attempts(attempts):
                raise RuntimeError("Unresolved or failed submission; no automatic retry")
            ready = [key for key in controller.ready_tasks() if e.TASKS[key]["kind"] == kind]
            usage = e.teacher_usage(queue, attempts, controller.user)
            if not ready or usage["submitted"] >= e.SUBMIT_LIMIT:
                return None
            count = (
                min(MAX_TASKS, e.GPU_LIMIT - usage["gpus"], len(ready))
                if kind in {"train", "eval"}
                else 0
            )
            if kind in {"train", "eval"} and count < 1:
                return None
            selected = ready[: count if count else 1]
            key = self.key(selected)
            directory = controller.directory / "jobs" / key
            # A deterministic claim is never overwritten or issued a second time.
            if directory.exists():
                raise RuntimeError("Existing pooled claim requires explicit reconciliation")
            directory.mkdir(parents=True)
            e.save_json(
                directory / "QUEUE_BEFORE_SUBMIT.json",
                dict(
                    at=e.now(),
                    jobs=queue,
                    teacher_usage=usage,
                ),
                exclusive=True,
            )
            binding = self.value["task_pool"]
            intent = dict(
                key=key,
                kind=kind,
                task_ids=selected,
                gpu_count=count,
                created=e.now(),
                run_identity_sha256=identity,
                root=str(self.root),
                code_root=str(controller.code),
                python=controller.python,
                user=controller.user,
                job_name=f"srf3ce-{kind}-{key}-LH",
                cpus=4 * max(1, count),
                memory_gib=90 * count if count else 12,
                partition="cluster02" if count else controller.cpu_policy["partition"],
                qos=e.TEACHER_QOS if count else controller.cpu_policy["qos"],
                cpu_node=None if count else controller.cpu_policy["cpu_node"],
                execution_entry_schema=ENTRY_SCHEMA,
                extension_manifest_path=self.context.manifest,
                extension_manifest_sha256=checked["manifest_sha256"],
                amendment_sha256=checked["amendment_sha256"],
                execution_freeze_sha256=e.verify_freeze(self.root),
                pool_operation_identity_path=str(self.identity_path),
                pool_operation_identity_sha256=self.identity_sha256,
                pool_authorization_sha256=binding["authorization_sha256"],
                max_tasks_per_job=MAX_TASKS,
                job_entry_command=self.command("job", key, controller.python),
                worker_entry_commands={
                    t: self.command("worker", t, controller.python) for t in selected
                },
            )
            intent["claim_id"] = e.object_hash(
                [
                    identity,
                    key,
                    checked["manifest_sha256"],
                    checked["amendment_sha256"],
                    self.identity_sha256,
                    binding["authorization_sha256"],
                ]
            )
            intent["comment"] = "srf3ce-" + e.object_hash([str(self.root), key, identity])[:24]
            script = directory / "run.sbatch"
            intent["script"] = str(script)
            with script.open("x") as stream:
                stream.write(
                    e._script_body(self.root, controller.code, intent["job_entry_command"])
                )
                stream.flush()
                os.fsync(stream.fileno())
            intent["script_sha256"] = _sha(script)
            intent["sbatch"] = e._sbatch_args(intent)
            self.verify_intent(self.root, intent)
            e.save_json(directory / "INTENT.json", intent, exclusive=True)
            state = dict(
                key=key,
                status="SUBMISSION_UNKNOWN",
                gpu_count=count,
                run_identity_sha256=identity,
                intent_sha256=_sha(directory / "INTENT.json"),
                qos=intent["qos"],
            )
            e.save_json(directory / "STATE.json", state)
            try:
                result = controller.slurm.run(intent["sbatch"])
            except Exception as error:
                e.save_json(
                    directory / "SUBMISSION_EXCEPTION.json",
                    dict(
                        error=repr(error),
                        at=e.now(),
                    ),
                    exclusive=True,
                )
                raise
            e.save_json(
                directory / "SUBMISSION.json",
                dict(
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    at=e.now(),
                ),
                exclusive=True,
            )
            match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout)
            if result.returncode or not match:
                return state
            state.update(job_id=match[1], status="HELD")
            e.save_json(directory / "STATE.json", state)
            controller.release(state, intent)
            return state

    def prepare_resume(self, controller, previous_key, attempts, queue, verifier):
        intent = _read(controller.directory / "jobs" / previous_key / "INTENT.json")
        if self.pooled(intent):
            raise RuntimeError("HANDOFF_REQUIRED: preserve every pooled task before any recovery")
        return self.original_prepare_resume(controller, previous_key, attempts, queue, verifier)

    def verify_job(self, controller, state):
        e = self.extension
        directory = controller.directory / "jobs" / state["key"]
        intent = _read(directory / "INTENT.json")
        if not self.pooled(intent):
            return self.original_verify_job(controller, state)
        self.verify_intent(self.root, intent)
        job = state.get("job_id")
        terminal = _read(directory / "TERMINAL.json")
        expected = dict(
            job_id=job,
            state="COMPLETED",
            exit_code="0:0",
            user=controller.user,
            account="rose",
            qos=intent["qos"],
            name=intent["job_name"],
        )
        if any(terminal.get(k) != v for k, v in expected.items()):
            raise PermissionError("Completed pooled accounting identity differs")
        allocation = _read(directory / "ALLOCATION_VERIFIED.json")
        if allocation.get("job_id") != job or (
            allocation.get("script_sha256") != intent["script_sha256"]
            or _sha(intent["script"]) != intent["script_sha256"]
        ):
            raise PermissionError("Completed pooled allocation/script identity differs")
        e.verify_allocation(
            allocation["fields"], dict(intent, job_id=job), controller.user, held=True
        )
        release = _read(directory / "RELEASE.json")
        if release.get("job_id") != job or release.get("returncode") != 0:
            raise PermissionError("Completed pooled job lacks confirmed release")
        started, complete = _read(directory / "STARTED.json"), _read(directory / "COMPLETE.json")
        if started.get("job_id") != job or complete != dict(
            status="COMPLETE",
            job_id=job,
            task_ids=intent["task_ids"],
            claim_id=intent["claim_id"],
        ):
            raise PermissionError("Pooled process completion identity differs")
        devices = started.get("devices")
        if not isinstance(devices, list):
            raise PermissionError("Pooled physical device receipt missing")
        if intent["gpu_count"]:
            e.visible_devices(",".join(devices), intent["gpu_count"])
        elif devices:
            raise PermissionError("CPU release unexpectedly owns GPU devices")
        for index, task in enumerate(intent["task_ids"]):
            launch, exited = (
                _read(directory / (task + ".LAUNCH.json")),
                _read(directory / (task + ".EXIT.json")),
            )
            if (
                launch
                != dict(
                    command=intent["worker_entry_commands"][task],
                    claim_id=intent["claim_id"],
                    device=devices[index] if devices else "",
                )
                or exited.get("returncode") != 0
            ):
                raise PermissionError("Pooled task process identity/exit differs")
            e.verify_task(self.root, task, binding=dict(job_id=job, claim_id=intent["claim_id"]))
            e.verify_observation_receipt(self.root, intent, task, job)
        return list(intent["task_ids"])

    def run_job(self, job_key, *, popen=subprocess.Popen, sleep=time.sleep):
        e, root = self.extension, self.root
        self.context.api.protocol.verify_run(root)
        if not re.fullmatch(r"job-[a-f0-9]{16}", job_key):
            raise PermissionError("Unsafe pooled allocation key")
        directory = root / "orchestration/jobs" / job_key
        intent, state = _read(directory / "INTENT.json"), _read(directory / "STATE.json")
        if not self.pooled(intent):
            raise PermissionError("Pooled entry cannot replace a historical allocation")
        self.verify_intent(root, intent)
        job = os.environ.get("SLURM_JOB_ID", "")
        if (
            state.get("job_id") != job
            or state.get("status") not in {"SUBMITTED", "RELEASE_UNKNOWN"}
            or (
                intent["run_identity_sha256"] != e.run_identity(root)
                or state.get("intent_sha256") != _sha(directory / "INTENT.json")
            )
        ):
            raise PermissionError("Pooled runner is not the bound released Slurm job")
        release = _read(directory / "RELEASE_INTENT.json")
        if release.get("job_id") != job or (directory / "STARTED.json").exists():
            raise PermissionError("Pooled allocation already started or lacks verified release")
        count = intent["gpu_count"]
        devices = (
            e.visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""), count) if count else []
        )
        if count and count != len(intent["task_ids"]):
            raise PermissionError("Each pooled GPU requires exactly one fixed task")
        allocation = _read(directory / "ALLOCATION_VERIFIED.json")
        if allocation.get("job_id") != job or _sha(intent["script"]) != intent["script_sha256"]:
            raise PermissionError("Pooled runtime allocation/script identity differs")
        e.verify_allocation(
            allocation["fields"], dict(intent, job_id=job), intent["user"], held=True
        )
        e.save_json(
            directory / "STARTED.json",
            dict(job_id=job, devices=devices, at=e.now()),
            exclusive=True,
        )
        children, exits, verified, failures = {}, {}, [], {}
        received, forwarded, handlers = [], set(), {}

        def receive(signum, _frame):
            received.append(signum)

        def forward(signum):
            for task, child in children.items():
                if task in exits:
                    continue
                with suppress(ProcessLookupError):
                    # Every child was created in its own session below. Never signal
                    # the scheduler, another job, or an inherited process group.
                    os.killpg(child.pid, signum)

        def reap():
            for task, child in children.items():
                if task in exits:
                    continue
                code = child.poll()
                if code is not None:
                    exits[task] = code
                    e.save_json(
                        directory / (task + ".EXIT.json"), dict(returncode=code), exclusive=True
                    )

        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                handlers[signum] = signal.signal(signum, receive)
            for index, task in enumerate(intent["task_ids"]):
                if received:
                    raise RuntimeError(
                        "Parent termination arrived before all fixed workers launched"
                    )
                env = dict(os.environ, SRF3CE_JOB_KEY=job_key, SRF3CE_CLAIM_ID=intent["claim_id"])
                env["CUDA_VISIBLE_DEVICES"] = devices[index] if count else ""
                env.pop("SRF3CE_RESUME_RECEIPT", None)
                env.pop("SRF3CE_PREVIOUS_JOB_KEY", None)
                command = intent["worker_entry_commands"][task]
                e.save_json(
                    directory / (task + ".LAUNCH.json"),
                    dict(
                        command=command,
                        device=env["CUDA_VISIBLE_DEVICES"],
                        claim_id=intent["claim_id"],
                    ),
                    exclusive=True,
                )
                try:
                    children[task] = popen(command, env=env, start_new_session=True)
                except BaseException as error:
                    e.save_json(
                        directory / (task + ".LAUNCH_EXCEPTION.json"),
                        dict(
                            error=repr(error),
                            at=e.now(),
                        ),
                        exclusive=True,
                    )
                    raise
            signal_started = None
            while len(exits) < len(children):
                reap()
                if received:
                    if signal_started is None:
                        signal_started = time.monotonic()
                    for signum in received:
                        if signum not in forwarded:
                            forward(signum)
                            forwarded.add(signum)
                    if time.monotonic() - signal_started > 10 and signal.SIGKILL not in forwarded:
                        forward(signal.SIGKILL)
                        forwarded.add(signal.SIGKILL)
                if len(exits) < len(children):
                    sleep(0.1)
            for task, code in exits.items():
                if code:
                    failures[task] = dict(returncode=code)
                    continue
                try:
                    e.verify_task(root, task, binding=dict(job_id=job, claim_id=intent["claim_id"]))
                    e.verify_observation_receipt(root, intent, task, job)
                    verified.append(task)
                except Exception as error:
                    failures[task] = dict(error=repr(error))
                    e.save_json(
                        directory / (task + ".VERIFICATION_FAILURE.json"),
                        dict(
                            error=repr(error),
                            at=e.now(),
                        ),
                        exclusive=True,
                    )
            if failures or received:
                raise RuntimeError(
                    "HANDOFF_REQUIRED: pooled worker failure/termination; no retries"
                )
            e.save_json(
                directory / "COMPLETE.json",
                dict(
                    status="COMPLETE",
                    job_id=job,
                    task_ids=intent["task_ids"],
                    claim_id=intent["claim_id"],
                ),
                exclusive=True,
            )
            e.save_json(
                directory / "POOL_END.json",
                dict(
                    status="COMPLETE",
                    job_id=job,
                    verified_tasks=verified,
                    exits=exits,
                    at=e.now(),
                ),
                exclusive=True,
            )
            return _read(directory / "COMPLETE.json")
        except BaseException as error:
            # A launcher exception must not orphan this allocation's own children.
            reap()
            if len(exits) < len(children):
                forward(signal.SIGTERM)
                deadline = time.monotonic() + 10
                while len(exits) < len(children) and time.monotonic() < deadline:
                    reap()
                    if len(exits) < len(children):
                        sleep(0.1)
                if len(exits) < len(children):
                    forward(signal.SIGKILL)
                    for task, child in children.items():
                        if task not in exits:
                            exits[task] = child.wait(timeout=5)
                            e.save_json(
                                directory / (task + ".EXIT.json"),
                                dict(
                                    returncode=exits[task],
                                ),
                                exclusive=True,
                            )
            e.save_json(
                directory / "POOL_END.json",
                dict(
                    status="HANDOFF_REQUIRED",
                    job_id=job,
                    error=repr(error),
                    exits=exits,
                    verified_tasks=verified,
                    failures=failures,
                    signals=received,
                    unlaunched_tasks=[t for t in intent["task_ids"] if t not in children],
                    new_submissions=0,
                    automatic_retries=0,
                    at=e.now(),
                ),
                exclusive=True,
            )
            raise
        finally:
            for signum, previous in handlers.items():
                signal.signal(signum, previous)


def install(value, extension):
    """Add operational methods while retaining the exact native class object."""
    cls = extension._context(value["root"]).controller
    previous = getattr(cls, _OWNER, None)
    if previous is not None:
        if previous.identity_path != Path(value["identity_path"]).resolve(strict=True) or (
            previous.extension is not extension
        ):
            raise PermissionError("Another task pool operation is already installed")
        previous.check()
        return previous
    pool = Pool(value, extension)

    def submit(controller, kind, *, resume_key=None, resume_verifier=None):
        return pool.submit(controller, kind, resume_key=resume_key, resume_verifier=resume_verifier)

    def verify_job(controller, state):
        return pool.verify_job(controller, state)

    def prepare_resume(controller, previous_key, attempts, queue, verifier):
        return pool.prepare_resume(controller, previous_key, attempts, queue, verifier)

    cls.submit, cls.verify_job, cls.prepare_resume = submit, verify_job, prepare_resume
    extension._verify_intent = pool.verify_intent
    extension.teacher_usage = pool.teacher_usage
    pool.context.api.orchestration.teacher_usage = pool.teacher_usage
    setattr(cls, _OWNER, pool)
    return pool


def _load_helper(identity_path):
    identity_path = Path(identity_path).resolve(strict=True)
    value = _read(identity_path)
    operation = identity_path.parent
    root = Path(value["root"]).resolve(strict=True)
    commit = value.get("operation_commit", "")
    if not re.fullmatch(r"[a-f0-9]{40}", str(commit)) or operation != root / "operations" / commit:
        raise PermissionError("Task pool CLI requires the registered operation directory")
    sources_path = operation / "OPERATION_SOURCES.json"
    if value.get("operation_sources_sha256") != _sha(sources_path):
        raise PermissionError("Task pool CLI source manifest changed")
    sources = _read(sources_path)["files"]
    if Path(__file__).resolve(strict=True) != operation / "pooled_execution.py" or (
        sources.get("pooled_execution.py") != _sha(__file__)
    ):
        raise PermissionError("Task pool CLI source is not registered")
    path = operation / "dispatch_registered.py"
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != sources.get(path.name):
        raise PermissionError("Task pool CLI dispatcher source changed")
    name = "srf3ce_pool_dispatch_" + sources[path.name]
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
    sys.modules[name] = module
    exec(compile(body, str(path), "exec"), module.__dict__)
    checked = module.identity(identity_path)
    checked["identity_path"] = str(identity_path)
    return module, checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("job", "worker"), required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--job-key")
    parser.add_argument("--task-id")
    args = parser.parse_args(argv)
    if (args.mode == "job" and (not args.job_key or args.task_id)) or (
        args.mode == "worker" and (not args.task_id or args.job_key)
    ):
        parser.error("job needs only --job-key; worker needs only --task-id")
    helper, value = _load_helper(args.identity)
    api = helper.load_science(value)
    pool = install(value, api.extension)
    if args.mode == "job":
        result = pool.run_job(args.job_key)
    else:
        job_key = os.environ.get("SRF3CE_JOB_KEY", "")
        if not re.fullmatch(r"job-[a-f0-9]{16}", job_key):
            raise PermissionError("Pooled worker lacks its registered allocation")
        intent = _read(pool.root / "orchestration/jobs" / job_key / "INTENT.json")
        if not pool.pooled(intent):
            raise PermissionError("Pooled worker cannot replace historical execution")
        pool.verify_intent(pool.root, intent)
        # This is the retained native observation implementation, which then
        # loads and invokes the original frozen scientific worker.
        result = api.extension.run_worker(pool.root, args.task_id)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
