"""Bounded SR-F3-CE Slurm submission; scores never participate in scheduling.

Every allocation has an immutable intent before sbatch, is submitted held, and
is released only after checking Slurm's actual fields and stored batch script.
Unknown submissions are reservations, never retry opportunities. GPU pools run
one fixed task per physical device and exit; they never wait for prerequisites.
"""

from __future__ import annotations

import fcntl
import getpass
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from sr_f12.orchestration import (
    TERMINAL,
    file_hash,
    gpu_count,
    memory_gib,
    now,
    read_json,
    save_json,
    slurm_fields,
)

EXPERIMENT_ID = "SR-F3-CE-20261016"
TEACHER_QOS = "soujanya-poria-startfund-2026-03"
GPU_LIMIT = 5
SUBMIT_LIMIT = 5
ARMS = ("RL", "GOLD", "PREFIX", "EXEC")
PARENTS = tuple(f"SRF2_J6_s{seed}" for seed in (71001, 71002, 71003))
RUNS = tuple(f"SRF3CE_{arm}_b{block}" for block in range(3) for arm in ARMS)
FINAL = frozenset({"COMPLETE", "FAILED"})
UNKNOWN = frozenset({"SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN", "ACCOUNTING_UNKNOWN"})
BLOCKING = UNKNOWN | {"HELD", "FAILED"}
KINDS = ("p0", "engine", "train", "eval", "release")
RESUMABLE_STATES = frozenset({"TIMEOUT", "PREEMPTED"})


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def task_matrix():
    tasks = [dict(task_id="p0", kind="p0")]
    tasks += [
        dict(task_id=f"engine--b{b}", kind="engine", parent_id=parent)
        for b, parent in enumerate(PARENTS)
    ]
    tasks.append(dict(task_id="engine--resume", kind="engine"))
    tasks += [dict(task_id=f"train--{run}", kind="train", model_id=run) for run in RUNS]
    tasks += [dict(task_id=f"eval--{mid}", kind="eval", model_id=mid) for mid in (*PARENTS, *RUNS)]
    tasks.append(dict(task_id="release", kind="release"))
    return tasks


TASKS = {task["task_id"]: task for task in task_matrix()}


def registered_task(task_id):
    if task_id not in TASKS:
        raise PermissionError("Unregistered task identity")
    return dict(TASKS[task_id])


def run_identity(root):
    path = Path(root) / "RUN_IDENTITY.json"
    identity = read_json(path)
    if identity.get("experiment_id") != EXPERIMENT_ID:
        raise PermissionError("RUN_IDENTITY experiment differs")
    return file_hash(path)


def receipt_path(root, task_id):
    registered_task(task_id)
    return Path(root) / "tasks" / task_id / "COMPLETE.json"


def artifact_record(root, path):
    root, relative = Path(root).resolve(strict=True), Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("Artifact path escapes run root")
    target = (root / relative).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise PermissionError("Artifact path escapes run root or is not a file")
    return dict(path=str(relative), sha256=file_hash(target), bytes=target.stat().st_size)


def validate_evidence(task_id, evidence):
    """Only completion metadata, never accuracy or other model scores."""
    task = registered_task(task_id)
    if not isinstance(evidence, dict):
        raise PermissionError("Task completion evidence missing")
    expected = {}
    if task_id == "p0":
        expected = {key: "PASS" for key in (
            "cpu_tests", "independent_answers", "historical_dedup", "render", "processor",
            "input_leakage",
        )}
    elif task_id == "engine--resume":
        expected = dict(
            logical_updates=4, physical_updates=6, new_rollouts=512, restore="PASS",
            masks="PASS", probabilities="PASS", nonzero_ce_gradient=True,
        )
    elif task["kind"] == "engine":
        expected = dict(parent_id=task["parent_id"], diagnostic_rows=960, protocol_gate="PASS")
    elif task["kind"] == "train":
        expected = dict(
            updates=64, rollout_rows=8192,
            support_examples=0 if task["model_id"].startswith("SRF3CE_RL_") else 256,
            monitor_steps=[0, 16, 32, 48, 64],
        )
        if not re.fullmatch(r"[a-f0-9]{64}", str(evidence.get("checkpoint_sha256", ""))):
            raise PermissionError("Final checkpoint identity missing")
    elif task["kind"] == "eval":
        expected = dict(
            structured_rows=10752, alternate_protocol_rows=2048, chartqa_rows=2500, sealed=True,
        )
        count, status = evidence.get("chartqapro_rows"), evidence.get("chartqapro_status")
        if type(count) is not int or not 0 <= count <= 256 or (
            status not in {"COMPLETE", "UNAVAILABLE"}
            or (status == "UNAVAILABLE" and count != 0)
            or (status == "COMPLETE" and count == 0)
        ):
            raise PermissionError("External evaluation completion/omission differs")
    elif task_id == "release":
        expected = dict(training_paths=12, evaluated_models=15, raw_rescoring="PASS")
    for key, value in expected.items():
        if type(evidence.get(key)) is not type(value) or evidence[key] != value:
            raise PermissionError("Task completion evidence differs: " + key)


def write_task_receipt(root, task_id, *, artifacts, evidence, local_p0=False):
    """Called by a worker only after its kind-specific raw/checkpoint validation.

The evidence sidecar is independently hashed with the actual output artifacts.
Local P0 is the only task allowed outside an authenticated Slurm allocation.
"""
    root, task = Path(root).resolve(strict=True), registered_task(task_id)
    validate_evidence(task_id, evidence)
    if local_p0:
        if task_id != "p0":
            raise PermissionError("Only P0 may complete locally")
        binding = dict(job_id="LOCAL_P0", claim_id="LOCAL_P0")
    else:
        binding = worker_binding(root, task_id)
    records = [artifact_record(root, path) for path in artifacts]
    if not records or len({row["path"] for row in records}) != len(records) or (
        f"tasks/{task_id}/EVIDENCE.json" in {row["path"] for row in records}
    ):
        raise PermissionError("Nonempty unique artifact evidence required")
    if task["kind"] == "train" and evidence["checkpoint_sha256"] not in {
        row["sha256"] for row in records
    }:
        raise PermissionError("Final checkpoint is not among the bound output artifacts")
    sidecar = root / "tasks" / task_id / "EVIDENCE.json"
    save_json(sidecar, dict(task=task, evidence=evidence), exclusive=True)
    records.append(artifact_record(root, sidecar.relative_to(root)))
    result = dict(
        task=task, status="PASS" if task["kind"] in {"p0", "engine"} else "COMPLETE",
        run_identity_sha256=run_identity(root), artifacts=records, evidence=evidence, **binding,
    )
    if task["kind"] in {"train", "eval", "release"}:
        result["execution_freeze_sha256"] = verify_freeze(root)
    save_json(receipt_path(root, task_id), result, exclusive=True)
    if task_id == "p0":
        save_json(root / "P0_RECEIPT.json", result, exclusive=True)
    return result


def verify_task(root, task_id, *, binding=None):
    root, task = Path(root).resolve(strict=True), registered_task(task_id)
    receipt = read_json(receipt_path(root, task_id))
    expected = dict(
        task=task, status="PASS" if task["kind"] in {"p0", "engine"} else "COMPLETE",
        run_identity_sha256=run_identity(root),
    )
    if binding:
        expected.update({key: binding[key] for key in ("job_id", "claim_id")})
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise PermissionError("Task receipt identity/status differs")
    validate_evidence(task_id, receipt.get("evidence"))
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) < 2:
        raise PermissionError("Task receipt lacks output artifact evidence")
    if len({row.get("path") for row in artifacts}) != len(artifacts):
        raise PermissionError("Duplicate artifact evidence")
    for record in artifacts:
        if record != artifact_record(root, record.get("path", "")):
            raise PermissionError("Task artifact bytes changed")
    if task["kind"] == "train" and receipt["evidence"]["checkpoint_sha256"] not in {
        row["sha256"] for row in artifacts
    }:
        raise PermissionError("Final checkpoint is not among the bound output artifacts")
    sidecar = f"tasks/{task_id}/EVIDENCE.json"
    if sidecar not in {row["path"] for row in artifacts} or read_json(root / sidecar) != dict(
        task=task, evidence=receipt["evidence"]
    ):
        raise PermissionError("Task evidence sidecar identity differs")
    if task_id == "p0" and read_json(root / "P0_RECEIPT.json") != receipt:
        raise PermissionError("P0 receipt binding differs")
    if task["kind"] in {"train", "eval", "release"} and (
        receipt.get("execution_freeze_sha256") != verify_freeze(root)
    ):
        raise PermissionError("Scientific receipt execution freeze differs")
    return dict(task_id=task_id, receipt_sha256=file_hash(receipt_path(root, task_id)))


def verify_freeze(root):
    """Require the final freeze to bind every already-completed ENGINE receipt."""
    from .protocol import verify_run

    root = Path(root)
    verify_run(root, science=True)
    freeze = read_json(root / "EXECUTION_FREEZE.json")
    if freeze.get("run_identity_sha256") != run_identity(root):
        raise PermissionError("Execution freeze identity differs")
    expected = {
        key: file_hash(receipt_path(root, key)) for key in TASKS if key.startswith("engine--")
    }
    if freeze.get("engine_receipts") != expected:
        raise PermissionError("Execution freeze ENGINE bindings differ")
    return file_hash(root / "EXECUTION_FREEZE.json")


def validate_resume_verification(root, value, previous_intent, previous_state):
    """Authenticate the runner's CPU verification of immutable recovery artifacts.

The runner owns checkpoint/cached-record semantics. Artifacts must be immutable
commits or raw groups, not mutable pointers such as LATEST.json. No model score
is accepted as a reason to resume or as a checkpoint selector.
"""
    task_id = previous_intent["task_ids"][0]
    expected = dict(
        status="PASS", task_id=task_id, previous_job_key=previous_state["key"],
        previous_job_id=previous_state["job_id"], run_identity_sha256=run_identity(root),
        resume_kind="checkpoint" if TASKS[task_id]["kind"] == "train" or (
            task_id == "engine--resume"
        ) else "generation_cache",
    )
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in expected.items()):
        raise PermissionError("Recovery verification identity/status differs")
    if not re.fullmatch(r"[a-f0-9]{64}", str(value.get("resume_state_sha256", ""))):
        raise PermissionError("Recovery state identity missing")
    if TASKS[task_id]["kind"] in {"train", "eval"} and (
        value.get("execution_freeze_sha256") != verify_freeze(root)
    ):
        raise PermissionError("Recovery execution freeze differs")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts or (
        len({row.get("path") for row in artifacts}) != len(artifacts)
    ):
        raise PermissionError("Recovery requires unique immutable artifact evidence")
    for record in artifacts:
        path = record.get("path", "")
        if Path(path).name == "LATEST.json" or record != artifact_record(root, path):
            raise PermissionError("Recovery artifact bytes changed or mutable pointer supplied")
    return value


@contextmanager
def lease(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "SUBMIT.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another SR-F3-CE submitter owns this run") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Slurm:
    def run(self, args):
        return subprocess.run(args, text=True, capture_output=True, check=False, timeout=60)

    def show(self, job_id):
        result = self.run(["scontrol", "show", "job", str(job_id), "--oneliner"])
        if result.returncode:
            raise RuntimeError("Slurm job identity unavailable: " + result.stderr)
        return slurm_fields(result.stdout)

    def queue(self, user):
        # The QoS is shared. Its jobs are counted irrespective of user or experiment.
        ids = set()
        for selector in (["--qos", TEACHER_QOS], ["--user", user]):
            result = self.run(["squeue", "--noheader", *selector, "--format=%i"])
            if result.returncode:
                raise RuntimeError("Slurm queue unavailable: " + result.stderr)
            values = result.stdout.split()
            if len(set(values)) != len(values) or any(not jid.isdigit() for jid in values):
                raise PermissionError("Queue identity/capacity unknown")
            ids.update(values)
        return {jid: self.show(jid) for jid in sorted(ids)}

    def batch_script(self, job_id):
        result = self.run(["scontrol", "write", "batch_script", str(job_id), "-"])
        if result.returncode:
            raise RuntimeError("Submitted batch script unavailable: " + result.stderr)
        return result.stdout

    def terminal(self, job_id):
        result = self.run([
            "sacct", "-X", "-n", "-P", "-j", str(job_id),
            "--format=JobIDRaw,State,ExitCode,User%100,Account%100,QOS%100,JobName%100",
        ])
        if result.returncode:
            raise RuntimeError("Slurm accounting unavailable: " + result.stderr)
        rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
        rows = [row for row in rows if row[0] == str(job_id)]
        if not rows:
            return None
        if len(rows) != 1 or len(rows[0]) < 7:
            raise PermissionError("Ambiguous accounting identity")
        row = rows[0]
        return dict(job_id=row[0], state=row[1].split()[0].rstrip("+"), exit_code=row[2],
                    user=row[3], account=row[4], qos=row[5], name=row[6])


def physical_gpu_count(tres):
    """Reject malformed/duplicate GPU fields instead of interpreting them as zero."""
    seen = set()
    for entry in str(tres).split(","):
        if entry.startswith("gres/gpu"):
            if not re.fullmatch(r"gres/gpu(?::[A-Za-z0-9_-]+)?=[0-9]+", entry):
                raise PermissionError("Unparseable physical GPU reservation")
            name = entry.split("=", 1)[0]
            if name in seen:
                raise PermissionError("Duplicate GPU reservation field")
            seen.add(name)
    return gpu_count(tres)


def teacher_usage(queue, attempts, user):
    gpus = submitted = 0
    for row in queue.values():
        if row.get("QOS") != TEACHER_QOS:
            continue
        if not row.get("ReqTRES") or not row.get("UserId") or "=" not in row["ReqTRES"]:
            raise PermissionError("Teacher QoS reservation capacity unknown")
        requested = physical_gpu_count(row["ReqTRES"])
        allocated = physical_gpu_count(row.get("AllocTRES", ""))
        gpus += max(requested, allocated)
        submitted += row["UserId"].split("(")[0] == user
    for state in attempts:
        if state["status"] not in FINAL and state.get("job_id") not in queue:
            count = state.get("gpu_count")
            if type(count) is not int or count not in (0, 1, 2):
                raise PermissionError("Unknown intent GPU reservation")
            qos = state.get("qos")
            if not isinstance(qos, str) or not qos or (count and qos != TEACHER_QOS):
                raise PermissionError("Unknown intent QoS reservation")
            if qos == TEACHER_QOS:
                gpus += count
                submitted += 1
    return dict(gpus=gpus, submitted=submitted)


def verify_allocation(fields, intent, user, *, held=False):
    expected = dict(
        JobId=intent["job_id"], JobName=intent["job_name"], Account="rose",
        Partition=intent["partition"], QOS=intent["qos"], Command=intent["script"],
        Comment=intent["comment"], NumCPUs=str(intent["cpus"]),
    )
    if not intent["job_name"].endswith("-LH"):
        raise PermissionError("Job name must end in -LH")
    for key, value in expected.items():
        if fields.get(key) != value:
            raise PermissionError("Allocation identity differs: " + key)
    if fields.get("UserId", "").split("(")[0] != user:
        raise PermissionError("Allocation owner differs")
    if fields.get("NumNodes") not in ("1", "1-1"):
        raise PermissionError("Allocation node count differs")
    count, tres = intent["gpu_count"], fields.get("ReqTRES", "")
    if physical_gpu_count(tres) != count or memory_gib(tres) != intent["memory_gib"]:
        raise PermissionError("Actual GPU/memory allocation differs")
    if count and (
        intent["qos"] != TEACHER_QOS or intent["partition"] != "cluster02"
        or not re.search(rf"(?:^|,)gres/gpu:pro6000={count}(?:,|$)", tres)
        or "highmem" not in fields.get("Features", "").split("&")
    ):
        raise PermissionError("Expected teacher QoS physical PRO6000 allocation")
    if not count and intent.get("cpu_node") and fields.get("ReqNodeList") != intent["cpu_node"]:
        raise PermissionError("CPU allocation policy differs")
    if fields.get("TimeLimit") != "3-00:00:00":
        raise PermissionError("Scheduler lease differs")
    if fields.get("Requeue") != "0":
        raise PermissionError("Automatic Slurm requeue must be disabled")
    if held and (fields.get("JobState"), fields.get("Reason")) != ("PENDING", "JobHeldUser"):
        raise PermissionError("Submission is not held")


class Controller:
    def __init__(self, root, code_root, *, python=sys.executable, slurm=None, user=None,
                 cpu_partition="cluster02", cpu_qos=None,
                 cpu_node="cpu-2", pool_size=1):
        self.root, self.code = Path(root).resolve(strict=True), Path(code_root).resolve(strict=True)
        self.python = str(Path(python).resolve(strict=True))
        self.slurm, self.user = slurm or Slurm(), user or getpass.getuser()
        self.directory = self.root / "orchestration"
        if pool_size not in (1, 2) or (cpu_qos is not None and (
            not isinstance(cpu_qos, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", cpu_qos)
        )):
            raise PermissionError("Invalid pool size or CPU QoS")
        self.pool_size = pool_size
        self.cpu_policy = dict(partition=cpu_partition, qos=cpu_qos, cpu_node=cpu_node)
        save_json(self.directory / "CONFIG.json", dict(
            root=str(self.root), code_root=str(self.code), python=self.python, user=self.user,
            run_identity_sha256=run_identity(self.root), cpu_policy=self.cpu_policy,
            pool_size=pool_size, gpu_limit=GPU_LIMIT, submitted_limit=SUBMIT_LIMIT,
            tasks=task_matrix(),
        ), exclusive=True)

    def identity(self):
        from .protocol import verify_run

        if (self.root / "STOP").exists():
            raise PermissionError("SR-F3-CE STOP exists")
        verify_run(self.root)
        if (self.root / "code").resolve(strict=True) != self.code:
            raise PermissionError("Scheduler source differs from the run-bound source")
        identity = run_identity(self.root)
        if read_json(self.directory / "CONFIG.json")["run_identity_sha256"] != identity:
            raise PermissionError("RUN_IDENTITY changed after scheduler registration")
        return identity

    def attempts(self):
        result = []
        for path in sorted((self.directory / "jobs").glob("*/INTENT.json")):
            intent = read_json(path)
            state_path = path.with_name("STATE.json")
            state = read_json(state_path) if state_path.exists() else dict(
                key=path.parent.name, status="SUBMISSION_UNKNOWN", gpu_count=intent["gpu_count"],
                run_identity_sha256=intent["run_identity_sha256"],
                intent_sha256=file_hash(path),
                qos=intent["qos"],
            )
            if state.get("key") != path.parent.name or any(
                state.get(k) != intent[k] for k in ("gpu_count", "qos", "run_identity_sha256")
            ) or intent.get("key") != path.parent.name or (
                intent["run_identity_sha256"] != self.identity()
            ):
                raise PermissionError("Submission state identity differs")
            if state.get("status") not in FINAL | UNKNOWN | {"HELD", "SUBMITTED"}:
                raise PermissionError("Unknown submission state")
            if state.get("intent_sha256") != file_hash(path):
                raise PermissionError("Durable submission intent changed")
            if intent["task_ids"] and (
                len(set(intent["task_ids"])) != len(intent["task_ids"])
                or any(registered_task(key)["kind"] != intent["kind"] for key in intent["task_ids"])
            ):
                raise PermissionError("Submission task identity differs")
            result.append(state)
        if any(not path.with_name("INTENT.json").exists()
               for path in (self.directory / "jobs").glob("*/STATE.json")):
            raise PermissionError("Orphan submission state without durable intent")
        return result

    def resume_links(self, attempts):
        """A verified new attempt supersedes one lease failure without erasing it."""
        by_key = {row["key"]: row for row in attempts}
        links = {}
        for state in attempts:
            directory = self.directory / "jobs" / state["key"]
            intent = read_json(directory / "INTENT.json")
            previous_key = intent.get("resume_of")
            if previous_key is None:
                continue
            if previous_key not in by_key or previous_key in links or previous_key == state["key"]:
                raise PermissionError("Recovery attempt forks or has an unknown predecessor")
            previous = by_key[previous_key]
            prior_dir = self.directory / "jobs" / previous_key
            prior_intent = read_json(prior_dir / "INTENT.json")
            terminal = read_json(prior_dir / "TERMINAL.json")
            verification_path = directory / "VERIFIED_RESUME.json"
            envelope = read_json(verification_path)
            if previous["status"] != "FAILED" or terminal.get("state") not in RESUMABLE_STATES:
                raise PermissionError("Only a recorded lease failure can have a successor")
            if intent["task_ids"] != prior_intent["task_ids"] or (
                intent["gpu_count"] != 1 or prior_intent["gpu_count"] != 1
                or len(intent["task_ids"]) != 1
                or intent.get("resume_verification_sha256") != file_hash(verification_path)
                or envelope.get("previous_intent_sha256") != file_hash(prior_dir / "INTENT.json")
                or envelope.get("previous_terminal_sha256") != file_hash(
                    prior_dir / "TERMINAL.json")
            ):
                raise PermissionError("Recovery predecessor/terminal/task binding differs")
            expected_terminal = dict(job_id=previous["job_id"], user=self.user, account="rose",
                                     qos=prior_intent["qos"], name=prior_intent["job_name"])
            if any(terminal.get(k) != v for k, v in expected_terminal.items()):
                raise PermissionError("Recovery predecessor accounting identity differs")
            validate_resume_verification(self.root, envelope.get("verification"),
                                         prior_intent, previous)
            links[previous_key] = state["key"]
        return links

    def blocking_attempts(self, attempts):
        resumed = self.resume_links(attempts)
        return [row for row in attempts if row["status"] in BLOCKING and row["key"] not in resumed]

    def recovery_blockers(self, attempts, previous_key):
        # Other lease failures still block ordinary new work, but do not prevent
        # explicitly verifying their independent peers one at a time.
        for row in self.blocking_attempts(attempts):
            if row["key"] == previous_key:
                continue
            terminal_path = self.directory / "jobs" / row["key"] / "TERMINAL.json"
            if row["status"] != "FAILED" or not terminal_path.is_file() or (
                read_json(terminal_path).get("state") not in RESUMABLE_STATES
            ):
                raise RuntimeError("Other unresolved/technical failure blocks recovery")

    def resume(self, previous_job_key, *, verifier=None):
        """Explicitly continue one verified lease-expired single-GPU task."""
        if not re.fullmatch(r"job-[a-f0-9]{16}", previous_job_key):
            raise PermissionError("Invalid recovery predecessor")
        intent = read_json(self.directory / "jobs" / previous_job_key / "INTENT.json")
        return self.submit(intent["kind"], resume_key=previous_job_key, resume_verifier=verifier)

    def prepare_resume(self, previous_key, attempts, queue, verifier):
        by_key = {row["key"]: row for row in attempts}
        if previous_key not in by_key:
            raise PermissionError("Recovery predecessor is not a registered allocation")
        previous = by_key[previous_key]
        directory = self.directory / "jobs" / previous_key
        intent = read_json(directory / "INTENT.json")
        if previous["status"] != "FAILED" or previous.get("job_id") in queue or (
            intent["gpu_count"] != 1 or len(intent["task_ids"]) != 1
            or intent["kind"] not in {"engine", "train", "eval"}
        ):
            raise PermissionError("Recovery requires a terminal single-GPU lease failure")
        terminal = self.slurm.terminal(previous["job_id"])
        expected = dict(job_id=previous["job_id"], user=self.user, account="rose",
                        qos=intent["qos"], name=intent["job_name"])
        if terminal is None or terminal.get("state") not in RESUMABLE_STATES or any(
            terminal.get(k) != v for k, v in expected.items()
        ) or terminal != read_json(directory / "TERMINAL.json"):
            raise PermissionError("Recovery terminal accounting is missing or differs")
        allocation = read_json(directory / "ALLOCATION_VERIFIED.json")
        verify_allocation(allocation["fields"], dict(intent, job_id=previous["job_id"]),
                          self.user, held=True)
        if file_hash(intent["script"]) != intent["script_sha256"] or (
            read_json(directory / "RELEASE.json").get("returncode") != 0
        ):
            raise PermissionError("Recovery predecessor source/release differs")
        task_id = intent["task_ids"][0]
        if receipt_path(self.root, task_id).exists():
            raise PermissionError("Existing task completion needs receipt-only reconciliation")
        if verifier is None:
            from .runner import verify_resume as verifier
        verified = verifier(self.root, task_id, previous_key)
        validate_resume_verification(self.root, verified, intent, previous)
        return dict(verification=verified,
                    previous_intent_sha256=file_hash(directory / "INTENT.json"),
                    previous_terminal_sha256=file_hash(directory / "TERMINAL.json"))

    def verify_job(self, state):
        """Authenticate accounting and process evidence before accepting any task."""
        directory = self.directory / "jobs" / state["key"]
        intent = read_json(directory / "INTENT.json")
        job = state.get("job_id")
        terminal = read_json(directory / "TERMINAL.json")
        expected = dict(job_id=job, state="COMPLETED", exit_code="0:0", user=self.user,
                        account="rose", qos=intent["qos"], name=intent["job_name"])
        if any(terminal.get(key) != value for key, value in expected.items()):
            raise PermissionError("Completed accounting identity differs")
        allocation = read_json(directory / "ALLOCATION_VERIFIED.json")
        if allocation.get("job_id") != job or allocation.get("script_sha256") != (
            intent["script_sha256"]
        ) or file_hash(intent["script"]) != intent["script_sha256"]:
            raise PermissionError("Completed allocation/script identity differs")
        verify_allocation(allocation["fields"], dict(intent, job_id=job), self.user, held=True)
        release = read_json(directory / "RELEASE.json")
        if release.get("job_id") != job or release.get("returncode") != 0:
            raise PermissionError("Completed job lacks confirmed release")
        started = read_json(directory / "STARTED.json")
        complete = read_json(directory / "COMPLETE.json")
        if started.get("job_id") != job or complete != dict(
            status="COMPLETE", job_id=job, task_ids=intent["task_ids"], claim_id=intent["claim_id"]
        ):
            raise PermissionError("Job process completion identity differs")
        devices = started.get("devices")
        if not isinstance(devices, list):
            raise PermissionError("Physical device receipt missing")
        if intent["gpu_count"]:
            visible_devices(",".join(devices), intent["gpu_count"])
        elif devices:
            raise PermissionError("CPU job unexpectedly owns GPU devices")
        for index, task_id in enumerate(intent["task_ids"]):
            launch = read_json(directory / (task_id + ".LAUNCH.json"))
            process_exit = read_json(directory / (task_id + ".EXIT.json"))
            expected_command = [intent["python"],
                                str(Path(intent["code_root"]) / "scripts/sr_f3_ce/run_worker.py"),
                                "--root", str(self.root), "--task-id", task_id]
            if launch != dict(command=expected_command, claim_id=intent["claim_id"],
                              device=devices[index] if devices else "") or (
                process_exit.get("returncode") != 0
            ):
                raise PermissionError("Task process identity/exit differs")
            verify_task(self.root, task_id, binding=dict(job_id=job, claim_id=intent["claim_id"]))
        return intent["task_ids"]

    def completed(self):
        done = set()
        p0_path = receipt_path(self.root, "p0")
        if p0_path.exists() and read_json(p0_path).get("job_id") == "LOCAL_P0":
            verify_task(self.root, "p0", binding=dict(job_id="LOCAL_P0", claim_id="LOCAL_P0"))
            done.add("p0")
        for state in self.attempts():
            if state["status"] == "COMPLETE":
                done.update(self.verify_job(state))
        return done

    def ready_tasks(self):
        done = self.completed()
        claimed = {
            key for state in self.attempts()
            for key in read_json(self.directory / "jobs" / state["key"] / "INTENT.json")["task_ids"]
        }
        engines = {key for key in TASKS if key.startswith("engine--")}
        trains = {key for key in TASKS if key.startswith("train--")}
        evaluations = {key for key in TASKS if key.startswith("eval--")}
        if engines <= done:
            if not (self.root / "EXECUTION_FREEZE.json").exists():
                return []
            verify_freeze(self.root)
        ready = []
        for key, task in TASKS.items():
            if key in done or key in claimed:
                continue
            kind = task["kind"]
            allowed = kind == "p0" or (kind == "engine" and "p0" in done)
            if kind == "train":
                allowed = engines <= done and "p0" in done
            if kind == "eval":
                allowed = engines <= done and trains <= done
            if kind == "release":
                allowed = engines <= done and trains <= done and evaluations <= done
            if allowed:
                ready.append(key)
        return ready

    def submit(self, kind, *, resume_key=None, resume_verifier=None):
        if kind not in (*KINDS, "controller"):
            raise ValueError("Unknown submission kind")
        with lease(self.directory):
            self.identity()
            required = [self.code / "scripts/sr_f3_ce/submit.py"]
            if kind != "controller":
                required.append(self.code / "scripts/sr_f3_ce/run_worker.py")
            if any(not path.is_file() for path in required):
                raise PermissionError("Deployed scheduler/worker entry point missing")
            if kind in {"p0", "release", "controller"} and not self.cpu_policy["qos"]:
                raise PermissionError("CPU submission requires an explicitly verified --cpu-qos")
            queue = self.slurm.queue(self.user)  # Fresh observation before every sbatch.
            self.refresh(queue)
            attempts = self.attempts()
            blockers = self.blocking_attempts(attempts)
            resume_envelope = None
            if resume_key is not None:
                key = "job-" + object_hash(["resume", resume_key, self.identity()])[:16]
                existing = next((row for row in attempts if row["key"] == key), None)
                if existing is not None:
                    return existing  # An explicit repeat observes; it never submits twice.
                self.recovery_blockers(attempts, resume_key)
                old_intent = read_json(self.directory / "jobs" / resume_key / "INTENT.json")
                selected = old_intent["task_ids"]
                if old_intent["kind"] != kind:
                    raise PermissionError("Recovery kind differs from its predecessor")
                usage = teacher_usage(queue, attempts, self.user)
                if usage["gpus"] >= GPU_LIMIT or usage["submitted"] >= SUBMIT_LIMIT:
                    return None
                resume_envelope = self.prepare_resume(resume_key, attempts, queue, resume_verifier)
                # CPU checkpoint verification can be slow. Re-observe before reserving anything.
                queue = self.slurm.queue(self.user)
                self.refresh(queue)
                attempts = self.attempts()
                self.recovery_blockers(attempts, resume_key)
                old_job = next(row["job_id"] for row in attempts if row["key"] == resume_key)
                if old_job in queue:
                    raise PermissionError("Recovery predecessor unexpectedly reappeared in queue")
            elif blockers:
                raise RuntimeError("Unresolved or failed submission; no automatic retry")
            elif kind == "controller":
                if any(row["key"] == "controller" for row in attempts):
                    return None
                selected, key = [], "controller"
            else:
                selected = [key for key in self.ready_tasks() if TASKS[key]["kind"] == kind]
                if not selected:
                    return None
                selected = selected[:self.pool_size if kind in {"engine", "train", "eval"} else 1]
                key = "job-" + object_hash(selected)[:16]
            count = len(selected) if kind in {"engine", "train", "eval"} else 0
            usage = teacher_usage(queue, attempts, self.user)
            if count:
                count = min(count, GPU_LIMIT - usage["gpus"])
                if count < 1 or usage["submitted"] >= SUBMIT_LIMIT:
                    return None
                selected = selected[:count]
                if resume_key is None:
                    key = "job-" + object_hash(selected)[:16]
            elif self.cpu_policy["qos"] == TEACHER_QOS and usage["submitted"] >= SUBMIT_LIMIT:
                return None
            directory = self.directory / "jobs" / key
            directory.mkdir(parents=True, exist_ok=True)
            if resume_envelope is not None:
                save_json(directory / "VERIFIED_RESUME.json", resume_envelope, exclusive=True)
            save_json(directory / "QUEUE_BEFORE_SUBMIT.json", dict(at=now(), jobs=queue,
                      teacher_usage=usage), exclusive=True)
            intent = dict(
                key=key, kind=kind, task_ids=selected, gpu_count=count, created=now(),
                run_identity_sha256=self.identity(), root=str(self.root), code_root=str(self.code),
                python=self.python, user=self.user, job_name=f"srf3ce-{kind}-{key}-LH",
                claim_id=object_hash([self.identity(), key]),
                cpus=4 * max(1, count), memory_gib=90 * count if count else 12,
                partition="cluster02" if count else self.cpu_policy["partition"],
                qos=TEACHER_QOS if count else self.cpu_policy["qos"],
                cpu_node=None if count else self.cpu_policy["cpu_node"],
            )
            if kind in {"train", "eval", "release"}:
                intent["execution_freeze_sha256"] = verify_freeze(self.root)
            if resume_envelope is not None:
                intent.update(resume_of=resume_key, resume_verification_sha256=file_hash(
                    directory / "VERIFIED_RESUME.json"))
            intent["comment"] = "srf3ce-" + object_hash([str(self.root), key, self.identity()])[:24]
            script = directory / "run.sbatch"
            intent["script"] = str(script)
            command = [self.python, str(self.code / "scripts/sr_f3_ce/submit.py"),
                       "--root", str(self.root), "--code-root", str(self.code),
                       "--python", self.python, "--kind", "run-job", "--job-key", key]
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
            intent["script_sha256"] = file_hash(script)
            args = ["sbatch", "--hold", "--parsable", "--no-requeue", "--open-mode=append",
                    "--account=rose",
                    "--partition=" + intent["partition"], "--qos=" + intent["qos"],
                    "--nodes=1", "--ntasks=1", f"--cpus-per-task={intent['cpus']}",
                    f"--mem={intent['memory_gib']}G", "--time=4320",
                    "--job-name=" + intent["job_name"], "--comment=" + intent["comment"],
                    "--output=" + str(directory / "slurm-%j.log")]
            args += [f"--gres=gpu:pro6000:{count}", "--constraint=highmem"] if count else (
                ["--nodelist=" + intent["cpu_node"]] if intent["cpu_node"] else []
            )
            args.append(str(script))
            intent["sbatch"] = args
            save_json(directory / "INTENT.json", intent, exclusive=True)
            state = dict(key=key, status="SUBMISSION_UNKNOWN", gpu_count=count,
                         run_identity_sha256=self.identity(),
                         intent_sha256=file_hash(directory / "INTENT.json"), qos=intent["qos"])
            save_json(directory / "STATE.json", state)
            try:
                result = self.slurm.run(args)
            except Exception as error:
                save_json(directory / "SUBMISSION_EXCEPTION.json",
                          dict(error=repr(error), at=now()), exclusive=True)
                raise
            save_json(directory / "SUBMISSION.json", dict(
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, at=now(),
            ), exclusive=True)
            match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout)
            if result.returncode or not match:
                return state
            state.update(job_id=match[1], status="HELD")
            save_json(directory / "STATE.json", state)
            self.release(state, intent)
            return state

    def release(self, state, intent):
        directory = self.directory / "jobs" / state["key"]
        if (directory / "RELEASE_INTENT.json").exists():
            raise RuntimeError("Release already attempted; no automatic retry")
        if self.identity() != intent["run_identity_sha256"] or (
            file_hash(intent["script"]) != intent["script_sha256"]
        ):
            raise PermissionError("Submission source identity changed")
        runtime_resume_receipt(self.root, intent)
        fields = self.slurm.show(state["job_id"])
        verify_allocation(fields, dict(intent, job_id=state["job_id"]), self.user, held=True)
        actual_script = self.slurm.batch_script(state["job_id"])
        if hashlib.sha256(actual_script.encode()).hexdigest() != intent["script_sha256"]:
            raise PermissionError("Slurm stored batch script differs")
        release_queue = self.slurm.queue(self.user)
        usage = teacher_usage(release_queue, self.attempts(), self.user)
        save_json(directory / "QUEUE_BEFORE_RELEASE.json", dict(at=now(), jobs=release_queue,
                  teacher_usage=usage), exclusive=True)
        if usage["gpus"] > GPU_LIMIT or usage["submitted"] > SUBMIT_LIMIT:
            raise PermissionError("Teacher QoS capacity exceeded; allocation stays held")
        save_json(directory / "ALLOCATION_VERIFIED.json", dict(
            job_id=state["job_id"], fields=fields, script_sha256=intent["script_sha256"],
        ), exclusive=True)
        save_json(directory / "RELEASE_INTENT.json", dict(job_id=state["job_id"], at=now()),
                  exclusive=True)
        state["status"] = "RELEASE_UNKNOWN"
        save_json(directory / "STATE.json", state)
        result = self.slurm.run(["scontrol", "release", state["job_id"]])
        save_json(directory / "RELEASE.json", dict(
            job_id=state["job_id"], returncode=result.returncode, stdout=result.stdout,
            stderr=result.stderr, at=now(),
        ), exclusive=True)
        if result.returncode == 0:
            state["status"] = "SUBMITTED"
        save_json(directory / "STATE.json", state)

    def refresh(self, queue):
        for state in self.attempts():
            if state["status"] in FINAL or state["status"] in {
                "SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN",
            }:
                continue  # Unknown outcomes require explicit human reconciliation, never resubmit.
            directory = self.directory / "jobs" / state["key"]
            intent = read_json(directory / "INTENT.json")
            job = state.get("job_id")
            if not job:
                raise PermissionError("Known allocation is missing job identity")
            if job in queue:
                verify_allocation(queue[job], dict(intent, job_id=job), self.user)
                if state["status"] == "ACCOUNTING_UNKNOWN":
                    state["status"] = "SUBMITTED"
                    save_json(directory / "STATE.json", state)
                continue
            terminal = self.slurm.terminal(job)
            if terminal is None or terminal["state"] not in TERMINAL:
                state["status"] = "ACCOUNTING_UNKNOWN"
            else:
                expected = dict(job_id=job, user=self.user, account="rose", qos=intent["qos"],
                                name=intent["job_name"])
                if any(terminal.get(k) != v for k, v in expected.items()):
                    raise PermissionError("Terminal accounting identity differs")
                save_json(directory / "TERMINAL.json", terminal, exclusive=True)
                if terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0":
                    try:
                        self.verify_job(state)
                    except Exception as error:
                        save_json(directory / "VERIFICATION_FAILURE.json",
                                  dict(error=repr(error), at=now()), exclusive=True)
                        state["status"] = "FAILED"
                    else:
                        state["status"] = "COMPLETE"
                else:
                    state["status"] = "FAILED"
            save_json(directory / "STATE.json", state)

    def status(self):
        with lease(self.directory):
            queue = self.slurm.queue(self.user)
            self.refresh(queue)
            attempts, done = self.attempts(), self.completed()
            blocked = bool(self.blocking_attempts(attempts))
            result = dict(
                at=now(), completed=sorted(done), ready=[] if blocked else self.ready_tasks(),
                attempts=attempts, teacher_usage=teacher_usage(queue, attempts, self.user),
                blocked=blocked, all_complete=set(TASKS) <= done,
                resumed_lease_failures=self.resume_links(attempts),
            )
            save_json(self.directory / "STATUS.json", result)
            return result

    def tick(self):
        status = self.status()
        if not status["blocked"]:
            for kind in KINDS:
                while any(TASKS[key]["kind"] == kind for key in status["ready"]):
                    result = self.submit(kind)
                    status = self.status()
                    if result is None or status["blocked"]:
                        break
        return status


def visible_devices(value, count):
    values = value.split(",") if value else []
    if len(values) != count or len(set(values)) != count or any(
        not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", item) for item in values
    ):
        raise PermissionError("Expected distinct physical CUDA allocations")
    return values


def worker_binding(root, task_id):
    key, job = os.environ.get("SRF3CE_JOB_KEY", ""), os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"job-[a-f0-9]{16}", key) or not job.isdigit():
        raise PermissionError("Worker is not bound to an SR-F3-CE allocation")
    directory = Path(root) / "orchestration/jobs" / key
    intent, state = read_json(directory / "INTENT.json"), read_json(directory / "STATE.json")
    if task_id not in intent["task_ids"] or state.get("job_id") != job or (
        intent["run_identity_sha256"] != run_identity(root)
        or os.environ.get("SRF3CE_CLAIM_ID") != intent["claim_id"]
        or state.get("intent_sha256") != file_hash(directory / "INTENT.json")
    ):
        raise PermissionError("Worker task/allocation binding differs")
    return dict(job_id=job, claim_id=intent["claim_id"])


def runtime_resume_receipt(root, intent):
    """Recheck the immutable recovery handoff immediately before the worker starts."""
    if "resume_of" not in intent:
        return None
    root = Path(root)
    directory = root / "orchestration/jobs" / intent["key"]
    prior = root / "orchestration/jobs" / intent["resume_of"]
    path = directory / "VERIFIED_RESUME.json"
    envelope = read_json(path)
    if file_hash(path) != intent["resume_verification_sha256"] or (
        envelope.get("previous_intent_sha256") != file_hash(prior / "INTENT.json")
        or envelope.get("previous_terminal_sha256") != file_hash(prior / "TERMINAL.json")
    ):
        raise PermissionError("Runtime recovery receipt/predecessor changed")
    validate_resume_verification(root, envelope.get("verification"),
                                 read_json(prior / "INTENT.json"), read_json(prior / "STATE.json"))
    return path


def run_job(root, job_key, *, popen=subprocess.Popen):
    """Run exactly the fixed claims; no dynamic scientific routing or idle GPUs."""
    from .protocol import verify_run

    root = Path(root).resolve(strict=True)
    verify_run(root)
    if not re.fullmatch(r"job-[a-f0-9]{16}|controller", job_key):
        raise PermissionError("Unsafe allocation key")
    directory = root / "orchestration/jobs" / job_key
    intent, state = read_json(directory / "INTENT.json"), read_json(directory / "STATE.json")
    job = os.environ.get("SLURM_JOB_ID", "")
    if state.get("job_id") != job or intent["run_identity_sha256"] != run_identity(root):
        raise PermissionError("Runner is not the bound Slurm job")
    if state.get("intent_sha256") != file_hash(directory / "INTENT.json"):
        raise PermissionError("Durable submission intent changed")
    release = read_json(directory / "RELEASE_INTENT.json")
    if release.get("job_id") != job or (directory / "STARTED.json").exists():
        raise PermissionError("Allocation already started or lacks verified release")
    count = intent["gpu_count"]
    devices = visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""), count) if count else []
    if count != len(intent["task_ids"]) and count:
        raise PermissionError("Each GPU must have exactly one ready task")
    allocation = read_json(directory / "ALLOCATION_VERIFIED.json")
    if allocation.get("job_id") != job or file_hash(intent["script"]) != intent["script_sha256"]:
        raise PermissionError("Runtime allocation/script identity differs")
    verify_allocation(allocation["fields"], dict(intent, job_id=job), intent["user"], held=True)
    resume_receipt = runtime_resume_receipt(root, intent)
    save_json(directory / "STARTED.json", dict(job_id=job, devices=devices, at=now()),
              exclusive=True)
    if intent["kind"] == "controller":
        config = read_json(root / "orchestration/CONFIG.json")
        controller = Controller(root, config["code_root"], python=config["python"],
                                user=config["user"], pool_size=config["pool_size"],
                                cpu_partition=config["cpu_policy"]["partition"],
                                cpu_qos=config["cpu_policy"]["qos"],
                                cpu_node=config["cpu_policy"]["cpu_node"])
        while True:
            status = controller.tick()
            if status["blocked"]:
                raise RuntimeError("Controller stopped on unresolved technical state")
            if status["all_complete"]:
                break
            time.sleep(30)
    else:
        children = []
        for index, task_id in enumerate(intent["task_ids"]):
            env = dict(os.environ, SRF3CE_JOB_KEY=job_key, SRF3CE_CLAIM_ID=intent["claim_id"])
            env["CUDA_VISIBLE_DEVICES"] = devices[index] if count else ""
            if resume_receipt is not None:
                env["SRF3CE_RESUME_RECEIPT"] = str(resume_receipt)
                env["SRF3CE_PREVIOUS_JOB_KEY"] = intent["resume_of"]
            else:
                env.pop("SRF3CE_RESUME_RECEIPT", None)
                env.pop("SRF3CE_PREVIOUS_JOB_KEY", None)
            command = [intent["python"],
                       str(Path(intent["code_root"]) / "scripts/sr_f3_ce/run_worker.py"),
                       "--root", str(root), "--task-id", task_id]
            save_json(directory / (task_id + ".LAUNCH.json"), dict(
                command=command, device=env["CUDA_VISIBLE_DEVICES"], claim_id=intent["claim_id"],
            ), exclusive=True)
            children.append((task_id, popen(command, env=env)))
        failed = []
        for task_id, child in children:
            code = child.wait()
            save_json(directory / (task_id + ".EXIT.json"), dict(returncode=code), exclusive=True)
            if code:
                failed.append(task_id)
            else:
                verify_task(root, task_id, binding=dict(job_id=job, claim_id=intent["claim_id"]))
        if failed:
            raise RuntimeError("Workers failed; preserved without retry: " + ", ".join(failed))
    save_json(directory / "COMPLETE.json", dict(status="COMPLETE", job_id=job,
              task_ids=intent["task_ids"], claim_id=intent["claim_id"]), exclusive=True)
    return read_json(directory / "COMPLETE.json")
