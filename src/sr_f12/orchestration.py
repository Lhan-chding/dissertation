"""Fail-closed SR-F1.2 scheduling, with durable held submission and no automatic retry.

Only this experiment's jobs are released. Other jobs are read solely to count the
teacher-QoS GPU capacity. An unresolved submission consumes a reservation and is
never submitted again. Scientific progress needs both Slurm terminal success and
validated artifacts; a running job or an existing file alone is insufficient.
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
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .protocol import (
    AMENDMENT_ID,
    BASELINE,
    TEACHER_QOS,
    object_hash,
    scientific_matrix,
    validate_config,
)

TERMINAL = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
    }
)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value, *, exclusive=False):
    """Durable atomic replace; exclusive artifacts can only be replayed identically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if read_json(path) != value:
                    raise PermissionError("Immutable receipt differs: " + str(path)) from None
        else:
            os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def slurm_fields(text):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", text))


def gpu_count(tres):
    """Do not double-count generic and typed GPU TRES in the same allocation."""
    values = {}
    for entry in str(tres).split(","):
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        if key == "gres/gpu" or key.startswith("gres/gpu:"):
            if not value.isdigit():
                raise PermissionError("Unparseable GPU reservation")
            values[key] = int(value)
    if "gres/gpu" in values:
        typed = sum(v for k, v in values.items() if k != "gres/gpu")
        if typed and typed != values["gres/gpu"]:
            raise PermissionError("Inconsistent GPU TRES")
        return values["gres/gpu"]
    return sum(values.values())


def memory_gib(tres):
    match = re.search(r"(?:^|,)mem=([0-9.]+)([KMGTP]?)", tres)
    if not match:
        raise PermissionError("Actual requested memory missing")
    return (
        float(match[1])
        * {"K": 1 / 1048576, "M": 1 / 1024, "G": 1, "T": 1024, "P": 1048576, "": 1 / 1024}[match[2]]
    )


class Slurm:
    def run(self, args):
        return subprocess.run(args, text=True, capture_output=True, check=False, timeout=60)

    def queue(self, user):
        result = self.run(["squeue", "--noheader", "--user", user, "--format=%i"])
        if result.returncode:
            raise RuntimeError("squeue unavailable: " + result.stderr)
        ids = result.stdout.split()
        if any(not job.isdigit() for job in ids):
            raise PermissionError("Unrecognized queue identifier; capacity unknown")
        return {job: self.show(job) for job in ids}

    def show(self, job_id):
        result = self.run(["scontrol", "show", "job", str(job_id), "--oneliner"])
        if result.returncode:
            raise RuntimeError("Job identity unavailable: " + result.stderr)
        return slurm_fields(result.stdout)

    def terminal(self, job_id):
        result = self.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                str(job_id),
                "--format=JobIDRaw,State,ExitCode,User,Account,QOS,JobName",
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
        state = row[1].split()[0].rstrip("+")
        return dict(
            job_id=row[0],
            state=state,
            exit_code=row[2],
            user=row[3],
            account=row[4],
            qos=row[5],
            name=row[6],
        )


def verify_allocation(fields, intent, user, *, held=False):
    expected = dict(
        Account="rose",
        Partition="cluster02",
        QOS=TEACHER_QOS,
        JobName=intent["job_name"],
        Comment=intent["comment"],
        Command=intent["script"],
    )
    for key, value in expected.items():
        if fields.get(key) != value:
            raise PermissionError("Allocation identity differs: " + key)
    if fields.get("UserId", "").split("(")[0] != user:
        raise PermissionError("Allocation owner differs")
    if fields.get("JobId") != intent.get("job_id", fields.get("JobId")):
        raise PermissionError("Allocation id differs")
    if gpu_count(fields.get("ReqTRES", "")) != 1:
        raise PermissionError("Expected exactly one GPU per path")
    if not re.search(r"gres/gpu:pro6000=1(?:,|$)", fields.get("ReqTRES", "")):
        raise PermissionError("Expected actual PRO6000 GPU TRES")
    if memory_gib(fields.get("ReqTRES", "")) < 80:
        raise PermissionError("Actual highmem allocation is below 80 GiB")
    if "highmem" not in fields.get("Features", "").split("&"):
        raise PermissionError("Expected highmem constraint")
    if held and (fields.get("JobState") != "PENDING" or fields.get("Reason") != "JobHeldUser"):
        raise PermissionError("New submission is not held")


def teacher_usage(queue, attempts):
    actual = sum(
        gpu_count(row.get("ReqTRES", "")) for row in queue.values() if row.get("QOS") == TEACHER_QOS
    )
    # Known jobs absent from squeue still reserve until an authenticated terminal receipt.
    reserved = sum(
        1
        for a in attempts
        if a["status"] not in ("COMPLETE", "FAILED") and a.get("job_id") not in queue
    )
    return actual + reserved


@contextmanager
def controller_lease(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "CONTROLLER.lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another SR-F1.2 controller owns the lease") from None
        yield


class Controller:
    def __init__(
        self,
        root,
        code_root,
        model_path,
        source_manifest,
        source_commit,
        *,
        python=sys.executable,
        slurm=None,
        user=None,
    ):
        self.root = Path(root).resolve(strict=True)
        self.code = Path(code_root).resolve(strict=True)
        self.model = Path(model_path).resolve(strict=True)
        self.source_manifest = Path(source_manifest).resolve(strict=True)
        self.source_commit = source_commit
        self.python = str(Path(python).resolve(strict=True))
        self.directory = self.root / "orchestration"
        self.slurm = slurm or Slurm()
        self.user = user or getpass.getuser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = dict(
            version=AMENDMENT_ID,
            root=str(self.root),
            code_root=str(self.code),
            model_path=str(self.model),
            source_manifest=str(self.source_manifest),
            source_commit=source_commit,
            python=self.python,
            user=self.user,
            baseline_shards=3,
            gpu_qos=TEACHER_QOS,
            gpu_limit=5,
            main_matrix=scientific_matrix(),
        )
        save_json(self.directory / "CONFIG.json", self.config, exclusive=True)

    def identity(self):
        from .evaluation import verify_source_manifest

        if (self.root / "STOP").exists():
            raise PermissionError("Experiment STOP exists")
        manifest = read_json(self.source_manifest)
        if manifest.get("git_commit") != self.source_commit:
            raise PermissionError("Source manifest commit differs")
        source_hash = verify_source_manifest(self.source_manifest, self.code)
        registration = read_json(self.root / "AMENDMENT.json")
        if (
            registration.get("amendment_id") != AMENDMENT_ID
            or registration.get("source_commit") != self.source_commit
        ):
            raise PermissionError("Independent SR-F1.2 source registration differs")
        plan = validate_config(self.root / "config/SR_F1_2.json")
        if registration.get("config_sha256") != object_hash(plan):
            raise PermissionError("Provisional configuration differs from registration")
        model_identity = read_json(self.root / "MODEL_ENVIRONMENT_IDENTITY.json")
        if Path(model_identity["model_path"]).resolve(strict=True) != self.model:
            raise PermissionError("Baseline/scientific model path differs")
        return source_hash

    def attempts(self):
        states = {
            p.parent.name: read_json(p)
            for p in sorted((self.directory / "tasks").glob("*/STATE.json"))
        }
        for path in sorted((self.directory / "tasks").glob("*/INTENT.json")):
            if path.parent.name not in states:
                intent = read_json(path)
                states[path.parent.name] = dict(
                    task_id=intent["task_id"],
                    status="SUBMISSION_UNKNOWN",
                    source_sha256=intent["source_sha256"],
                )
        return list(states.values())

    def task_dir(self, key):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("Unsafe task id")
        return self.directory / "tasks" / key

    def task_state(self, key):
        path = self.task_dir(key) / "STATE.json"
        return read_json(path) if path.exists() else None

    def command(self, key):
        base = [self.python]
        if key == "TECHNICAL":
            return [
                *base,
                str(self.code / "scripts/sr_f12/technical_check.py"),
                "--run-root",
                str(self.root),
                "--plan",
                str(self.root / "config/SR_F1_2.json"),
            ]
        if key.startswith("BASELINE_") and key[-1] in "012" and key == "BASELINE_" + key[-1]:
            return [
                *base,
                str(self.code / "scripts/sr_f12/run_baseline.py"),
                "--run-root",
                str(self.root),
                "--model-path",
                str(self.model),
                "--shard-index",
                key[-1],
                "--shard-count",
                "3",
                "--source-manifest",
                str(self.source_manifest),
            ]
        if key in {row["model_id"] for row in scientific_matrix()}:
            return [
                *base,
                str(self.code / "scripts/sr_f12/run_worker.py"),
                "--run-root",
                str(self.root),
                "--plan",
                str(self.root / "config/SR_F1_2_FROZEN.json"),
                "--run-id",
                key,
            ]
        raise PermissionError("Unregistered task")

    def submit(self, key, queue):
        """An intent exists before sbatch; any ambiguous result is never retried."""
        directory = self.task_dir(key)
        if (directory / "INTENT.json").exists() or self.task_state(key):
            return False
        if teacher_usage(queue, self.attempts()) >= 5:
            return False
        source_hash = self.identity()
        directory.mkdir(parents=True, exist_ok=True)
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
        token = object_hash([str(self.root), key, source_hash])[:20]
        intent = dict(
            task_id=key,
            created=now(),
            source_sha256=source_hash,
            script=str(script),
            script_sha256=file_hash(script),
            command=command,
            job_name="srf12-" + key,
            comment="srf12-" + token,
            gpu_count=1,
        )
        args = [
            "sbatch",
            "--hold",
            "--parsable",
            "--account=rose",
            "--partition=cluster02",
            "--qos=" + TEACHER_QOS,
            "--nodes=1",
            "--gres=gpu:pro6000:1",
            "--constraint=highmem",
            "--cpus-per-task=4",
            "--time=4320",
            "--job-name=" + intent["job_name"],
            "--comment=" + intent["comment"],
            "--output=" + str(directory / "slurm-%j.log"),
            str(script),
        ]
        intent["sbatch"] = args
        save_json(directory / "INTENT.json", intent, exclusive=True)
        state = dict(task_id=key, status="SUBMISSION_UNKNOWN", source_sha256=source_hash)
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
        directory = self.task_dir(key)
        intent = dict(intent, job_id=state["job_id"])
        if file_hash(intent["script"]) != intent["script_sha256"]:
            raise PermissionError("Submitted script changed")
        fields = self.slurm.show(state["job_id"])
        verify_allocation(fields, intent, self.user, held=True)
        queue = self.slurm.queue(self.user)
        if teacher_usage(queue, self.attempts()) > 5:
            raise PermissionError("Teacher-QoS GPU capacity exceeded; keeping job held")
        save_json(
            directory / "ALLOCATION_VERIFIED.json",
            dict(job_id=state["job_id"], fields=fields, source_sha256=self.identity()),
            exclusive=True,
        )
        if (directory / "RELEASE_INTENT.json").exists():
            # The previous command might have succeeded: observation, never blind repetition.
            state["status"] = "RELEASE_UNKNOWN"
            save_json(directory / "STATE.json", state)
            return
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
            key, job = state["task_id"], state.get("job_id")
            directory = self.task_dir(key)
            intent = read_json(directory / "INTENT.json")
            if state["status"] in ("COMPLETE", "FAILED"):
                continue
            if not job:
                matches = [
                    jid
                    for jid, fields in queue.items()
                    if fields.get("Comment") == intent["comment"]
                ]
                if len(matches) > 1:
                    raise PermissionError("Duplicate jobs for one submission intent")
                if not matches:
                    # Unresolved intent reserves capacity pending investigation.
                    continue
                job = matches[0]
                state.update(job_id=job, status="HELD")
                save_json(directory / "RECONCILED.json", dict(job_id=job, at=now()), exclusive=True)
                save_json(directory / "STATE.json", state)
            if job in queue:
                verify_allocation(queue[job], dict(intent, job_id=job), self.user)
                if queue[job].get("Reason") == "JobHeldUser":
                    if not (directory / "RELEASE_INTENT.json").exists():
                        self.release(key, state, intent)
                    continue
                state.update(status=queue[job]["JobState"])
                save_json(directory / "STATE.json", state)
                continue
            terminal = self.slurm.terminal(job)
            if terminal is None or terminal["state"] not in TERMINAL:
                continue
            if any(
                terminal[k] != v
                for k, v in dict(
                    user=self.user, account="rose", qos=TEACHER_QOS, name=intent["job_name"]
                ).items()
            ):
                raise PermissionError("Terminal accounting identity differs")
            save_json(directory / "TERMINAL.json", terminal, exclusive=True)
            if terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0":
                evidence = self.verify_task(key)
                save_json(directory / "VERIFIED_COMPLETE.json", evidence, exclusive=True)
                state["status"] = "COMPLETE"
            else:
                state.update(status="FAILED", failure=terminal)
            save_json(directory / "STATE.json", state)

    def verify_task(self, key):
        if key == "TECHNICAL":
            from .runner import validate_technical_receipt

            receipt = read_json(self.root / "technical/TECHNICAL_CHECK_SR_F1_2.json")
            plan = read_json(self.root / "technical/SELECTED_CONFIG.json")
            validate_technical_receipt(receipt, plan)
            return dict(
                task_id=key, receipt_sha256=object_hash(receipt), config_sha256=object_hash(plan)
            )
        if key.startswith("BASELINE_"):
            directory = (
                self.root
                / "raw/evaluation"
                / f"{BASELINE}_step0_baseline"
                / f"shard{int(key[-1]):03d}"
            )
            receipt, plan = (
                read_json(directory / "COMPLETE.json"),
                read_json(directory / "PLAN.json"),
            )
            if receipt.get("status") != "COMPLETE" or receipt.get("plan_sha256") != object_hash(
                plan
            ):
                raise PermissionError("Baseline shard completion differs")
            if plan.get("shard_count") != 3 or plan.get("shard_index") != int(key[-1]):
                raise PermissionError("Baseline shard topology differs")
            return dict(
                task_id=key, receipt_sha256=object_hash(receipt), plan_sha256=object_hash(plan)
            )
        if key not in {row["model_id"] for row in scientific_matrix()}:
            raise PermissionError("Unknown completion task")
        import torch

        from mm_core.training import state_hash

        from .training import CHECKPOINT_FIELDS

        directory = self.root / "runs" / key
        receipt = read_json(directory / "COMPLETE.json")
        if (
            receipt.get("status"),
            receipt.get("model_id"),
            receipt.get("step"),
            receipt.get("technical_only"),
        ) != ("COMPLETE", key, 96, False):
            raise PermissionError("Incomplete scientific path")
        checkpoint = receipt["checkpoint"]
        cp_dir = directory / "checkpoints"
        if (
            checkpoint != read_json(cp_dir / "LATEST.json")
            or checkpoint != read_json(cp_dir / "commit-96.json")
            or checkpoint.get("step") != 96
        ):
            raise PermissionError("Checkpoint completion binding differs")
        path = (cp_dir / checkpoint["path"]).resolve(strict=True)
        if path.parent != cp_dir.resolve() or file_hash(path) != checkpoint["sha256"]:
            raise PermissionError("Final checkpoint bytes differ")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if set(data) != CHECKPOINT_FIELDS or data["committed_logical_step"] != 96:
            raise PermissionError("Final checkpoint fields/step differ")
        if (
            state_hash(data) != checkpoint["state_hash"]
            or {k: state_hash(v) for k, v in data.items()} != checkpoint["field_hashes"]
        ):
            raise PermissionError("Final checkpoint complete-state hashes differ")
        if (
            data["run_identity"] != read_json(directory / "RUN_MANIFEST.json")
            or data["run_identity"]["model_id"] != key
        ):
            raise PermissionError("Final checkpoint model identity differs")
        return dict(
            task_id=key, receipt_sha256=object_hash(receipt), checkpoint_sha256=checkpoint["sha256"]
        )

    def freeze(self):
        from .analysis import freeze_predictions, preregister_predictions, semantic_state_table
        from .evaluation import collect_scored_shards
        from .protocol import freeze_scientific

        if (self.root / "SCIENCE_FREEZE.json").exists():
            freeze = read_json(self.root / "SCIENCE_FREEZE.json")
            plan = read_json(self.root / "config/SR_F1_2_FROZEN.json")
            technical = read_json(self.root / "TECHNICAL_CHECK_SR_F1_2.json")
            predictions = read_json(self.root / "SEMANTIC_PREDICTIONS.json")
            if (
                freeze.get("status") != "FROZEN_BEFORE_SCIENCE"
                or freeze.get("source_commit") != self.source_commit
                or freeze.get("config_sha256") != object_hash(plan)
                or freeze.get("technical_check_sha256") != object_hash(technical)
                or freeze.get("predictions_sha256") != object_hash(predictions)
            ):
                raise PermissionError("Scientific freeze binding differs")
            return freeze
        baseline = collect_scored_shards(
            self.root / "raw/evaluation" / f"{BASELINE}_step0_baseline", expected_shards=3
        )
        tasks = [
            json.loads(line)
            for line in (self.root / "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        families = Counter(row["family"] for row in tasks if row["pool"] == "TRAIN")
        state = semantic_state_table(baseline)
        predictions = preregister_predictions(state, dict(families))
        freeze_predictions(self.root, state, predictions)
        technical = read_json(self.root / "technical/TECHNICAL_CHECK_SR_F1_2.json")
        plan = read_json(self.root / "technical/SELECTED_CONFIG.json")
        return freeze_scientific(
            self.root,
            plan,
            technical,
            predictions,
            source_commit=self.source_commit,
            actual_lora_modules=sorted(technical["preflight"]["lora_modules"]["target_modules"]),
        )

    def tick(self):
        source_hash = self.identity()
        queue = self.slurm.queue(self.user)
        self.refresh(queue)
        states = {row["task_id"]: row for row in self.attempts()}
        failures = [row for row in states.values() if row["status"] == "FAILED"]
        unknown = [
            row
            for row in states.values()
            if row["status"] in ("SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN")
        ]
        stage, desired = (
            "TECHNICAL_AND_BASELINE",
            ["TECHNICAL", "BASELINE_0", "BASELINE_1", "BASELINE_2"],
        )
        if failures:
            stage, desired = "BLOCKED_TECHNICAL_FAILURE", []
        elif unknown:
            stage, desired = "BLOCKED_UNKNOWN_SUBMISSION", []
        elif all(states.get(key, {}).get("status") == "COMPLETE" for key in desired):
            self.freeze()
            stage, desired = "MAIN_TRAINING", []
            for wave in (1, 2, 3):
                runs = [row["model_id"] for row in scientific_matrix() if row["wave"] == wave]
                if not all(states.get(key, {}).get("status") == "COMPLETE" for key in runs):
                    desired = runs
                    break
            if not desired:
                stage = "BLOCKED_ENDPOINT_EVALUATION_NOT_CONFIGURED"
        for key in desired:
            if key not in states:
                queue = self.slurm.queue(self.user)
                self.submit(key, queue)
                states = {row["task_id"]: row for row in self.attempts()}
                if any(
                    row["status"] in ("SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN")
                    for row in states.values()
                ):
                    break
        snapshot = dict(
            version=AMENDMENT_ID,
            at=now(),
            stage=stage,
            source_sha256=source_hash,
            tasks=states,
            scientific_complete=sum(
                states.get(row["model_id"], {}).get("status") == "COMPLETE"
                for row in scientific_matrix()
            ),
            scientific_total=12,
            experiment_complete=False,
            failures=failures,
            unknown_submissions=unknown,
            teacher_gpu_usage=teacher_usage(self.slurm.queue(self.user), list(states.values())),
        )
        save_json(self.directory / "STATE.json", snapshot)
        return snapshot


def run_controller(controller, *, once=False):
    with controller_lease(controller.directory):
        while True:
            result = controller.tick()
            print(json.dumps(result, sort_keys=True), flush=True)
            if once or result["stage"].startswith("BLOCKED_"):
                return result
            time.sleep(60)
