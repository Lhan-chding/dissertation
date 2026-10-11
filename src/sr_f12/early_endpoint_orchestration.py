"""Training-priority sealed evaluation after all training jobs have been dispatched."""

from __future__ import annotations

import getpass
import os
import re
import sys
from pathlib import Path

from .endpoint_orchestration import MODEL_IDS, EndpointController
from .orchestration import (
    Controller,
    Slurm,
    file_hash,
    now,
    read_json,
    save_json,
    teacher_usage,
    verify_allocation,
)
from .protocol import AMENDMENT_ID, BASELINE, TEACHER_QOS, object_hash, scientific_matrix


def submission_usage(queue, attempts):
    return sum(r.get("QOS") == TEACHER_QOS for r in queue.values()) + sum(
        r["status"] not in ("COMPLETE", "FAILED") and r.get("job_id") not in queue for r in attempts
    )


class EarlyEndpointController(EndpointController):
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
        self.slurm, self.user = slurm or Slurm(), user or getpass.getuser()
        self.directory = self.root / "endpoint_orchestration"
        self.config = dict(
            version=AMENDMENT_ID,
            root=str(self.root),
            code_root=str(self.code),
            model_path=str(self.model),
            source_manifest=str(self.source_manifest),
            source_commit=source_commit,
            python=self.python,
            user=self.user,
            model_ids=list(MODEL_IDS),
            gpu_qos=TEACHER_QOS,
            gpu_limit=5,
            policy="TRAINING_PRIORITY_SEALED_UNTIL_ALL_MAIN_COMPLETE",
        )
        save_json(
            self.directory / "revisions/early_sealed/CONFIG.json", self.config, exclusive=True
        )
        self.excluded_nodes = []

    def identity(self):
        from .early_authorization import verify_early_implementation

        registered = verify_early_implementation(
            self.root, self.code, self.source_manifest, self.source_commit
        )
        if (
            Path(read_json(self.root / "MODEL_ENVIRONMENT_IDENTITY.json")["model_path"]).resolve()
            != self.model
        ):
            raise PermissionError("Early endpoint model path differs")
        return registered["source_manifest_sha256"]

    def training_priority(self, queue):
        """No evaluation while training could still need a submission or allocation.

        With all twelve immutable intents present, the production main controller
        has no unsubmitted matrix key left. Only this endpoint lease can submit.
        """
        states = {r["task_id"]: r for r in self.main_attempts()}
        snapshot = read_json(self.root / "orchestration/STATE.json")
        if snapshot.get("failures") or snapshot.get("unknown_submissions"):
            return None
        jobs, nodes = set(), set()
        for run in scientific_matrix():
            key = run["model_id"]
            state = states.get(key, {})
            cached = snapshot.get("tasks", {}).get(key, {})
            job = state.get("job_id")
            if (
                state.get("status") not in ("RUNNING", "COMPLETE")
                or cached.get("status") not in ("RUNNING", "COMPLETE")
                or cached.get("job_id") != job
                or not isinstance(job, str)
                or not job.isdigit()
                or job in jobs
            ):
                return None
            jobs.add(job)
            directory = self.root / "orchestration/tasks" / key
            intent = read_json(directory / "INTENT.json")
            script = Path(intent["script"]).resolve(strict=True)
            if (
                intent.get("task_id") != key
                or intent.get("gpu_count") != 1
                or intent.get("job_name") != "srf12-" + key
                or not script.is_relative_to(directory.resolve())
                or file_hash(script) != intent["script_sha256"]
            ):
                raise PermissionError("Training submission identity differs")
            if state["status"] == "RUNNING":
                fields = queue.get(job)
                if not fields or fields.get("JobState") != "RUNNING":
                    return None
                verify_allocation(fields, dict(intent, job_id=job), self.user)
                node = fields.get("NodeList", "")
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", node) or node in ("None", "null"):
                    raise PermissionError("Cannot exclude active training node")
                nodes.add(node)
            else:
                if job in queue:
                    return None
                expected = dict(
                    job_id=job,
                    state="COMPLETED",
                    exit_code="0:0",
                    user=self.user,
                    account="rose",
                    qos=TEACHER_QOS,
                    name=intent["job_name"],
                )
                if (
                    self.slurm.terminal(job) != expected
                    or read_json(directory / "TERMINAL.json") != expected
                ):
                    raise PermissionError("Completed training accounting differs")
        return sorted(nodes)

    def command(self, key):
        if not key.startswith("EVAL_") or key[5:] not in MODEL_IDS:
            raise PermissionError("Unregistered early endpoint task")
        command = [
            self.python,
            str(self.code / "scripts/sr_f12/run_early_evaluation.py"),
            "--run-root",
            str(self.root),
            "--model-path",
            str(self.model),
            "--model-id",
            key[5:],
            "--source-manifest",
            str(self.source_manifest),
        ]
        return command

    def submission_options(self, key):
        return ["--exclude=" + ",".join(self.excluded_nodes)] if self.excluded_nodes else []

    def model_gate(self, model):
        if model == BASELINE:
            gate = self.main_gate(self.slurm.queue(self.user))
            return gate if gate else None
        state = self.task_main_state(model)
        if not state or state.get("status") != "COMPLETE":
            return None
        from .early_authorization import create_model_acceptance

        return create_model_acceptance(self.root, model, slurm=self.slurm)

    def task_main_state(self, model):
        path = self.root / "orchestration/tasks" / model / "STATE.json"
        return read_json(path) if path.exists() else None

    def submit(self, key, queue):
        # Fresh observation after potential expensive state validation.
        nodes = self.training_priority(queue)
        attempts = [*self.attempts(), *self.main_attempts()]
        if (
            nodes is None
            or teacher_usage(queue, attempts) >= 5
            or submission_usage(queue, attempts) >= 5
        ):
            return False
        if self.model_gate(key[5:]) is None:
            return False
        queue = self.slurm.queue(self.user)
        nodes = self.training_priority(queue)
        attempts = [*self.attempts(), *self.main_attempts()]
        if (
            nodes is None
            or teacher_usage(queue, attempts) >= 5
            or submission_usage(queue, attempts) >= 5
        ):
            return False
        self.excluded_nodes = nodes
        return Controller.submit(self, key, queue)

    def release(self, key, state, intent):
        if self.model_gate(key[5:]) is None:
            raise PermissionError("Model generation gate disappeared")
        queue = self.slurm.queue(self.user)
        nodes = self.training_priority(queue)
        attempts = [*self.attempts(), *self.main_attempts()]
        if (
            nodes is None
            or teacher_usage(queue, attempts) > 5
            or submission_usage(queue, attempts) > 5
        ):
            raise PermissionError("Training priority/capacity changed; endpoint stays held")
        fields = self.slurm.show(state["job_id"])
        excluded = fields.get("ExcNodeList", "(null)")
        actual = set()
        if excluded != "(null)":
            result = self.slurm.run(["scontrol", "show", "hostnames", excluded])
            if result.returncode:
                raise PermissionError("Cannot verify excluded training nodes")
            actual = set(result.stdout.split())
        if not set(nodes) <= actual:
            raise PermissionError("Endpoint does not exclude every active training node")
        return Controller.release(self, key, state, intent)

    def verify_task(self, key):
        from .early_authorization import verify_generation_authorization
        from .endpoint import read_completed_panel

        if key == "EVAL_" + BASELINE:
            return super().verify_task(key)
        model = key[5:]
        gate = verify_generation_authorization(self.root, model, slurm=self.slurm)
        directory = self.root / "sealed/evaluation" / model
        receipt = read_json(directory / "COMPLETE.json")
        if (
            receipt.get("status") != "COMPLETE"
            or receipt.get("model_id") != model
            or receipt.get("records") != 17604
            or receipt.get("source_sha256") != self.identity()
            or receipt.get("generation_gate") != "MODEL_TERMINAL_SEALED"
            or receipt.get("generation_authorization_sha256") != object_hash(gate)
            or "main_completion_sha256" in receipt
            or receipt.get("aggregate_scores_released") is not False
        ):
            raise PermissionError("Sealed endpoint completion differs")
        panels = receipt.get("panels", [])
        if len(panels) != 2 or {p["panel"]: p["records"] for p in panels} != {
            "final": 15104,
            "chartqa": 2500,
        }:
            raise PermissionError("Sealed endpoint coverage differs")
        seen = set()
        for panel in panels:
            path = directory / panel["panel"]
            if file_hash(path / "COMPLETE.json") != panel["completion_sha256"]:
                raise PermissionError("Sealed panel receipt changed")
            rows = read_completed_panel(path)
            if len(rows) != panel["records"]:
                raise PermissionError("Sealed panel record count differs")
            for row in rows:
                if row["model_id"] != model or row["slot_id"] in seen or "score" in row:
                    raise PermissionError(
                        "Sealed record identity/uniqueness/unscored state differs"
                    )
                seen.add(row["slot_id"])
        return dict(
            task_id=key,
            model_id=model,
            records=17604,
            receipt_sha256=object_hash(receipt),
            generation_authorization_sha256=object_hash(gate),
            scores_released=False,
        )

    def tick(self):
        source_hash = self.identity()
        queue = self.slurm.queue(self.user)
        self.refresh(queue)
        states = {r["task_id"]: r for r in self.attempts()}
        all_attempts = [*states.values(), *self.main_attempts()]
        failed = [r for r in all_attempts if r["status"] == "FAILED"]
        unknown = [
            r for r in all_attempts if r["status"] in ("SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN")
        ]
        stage = "WAITING_FOR_TRAINING_PRIORITY"
        acceptance = None
        if failed:
            stage = "BLOCKED_ENDPOINT_OR_TRAINING_FAILURE"
        elif unknown:
            stage = "BLOCKED_UNKNOWN_SUBMISSION"
        elif self.training_priority(self.slurm.queue(self.user)) is not None:
            stage = "SEALED_EVALUATION_TRAINING_PRIORITY"
            acceptance = self.main_gate(self.slurm.queue(self.user))
            # Trained models can enter early; base waits for the original complete-matrix gate.
            for model in [m for m in MODEL_IDS if m != BASELINE] + [BASELINE]:
                key = "EVAL_" + model
                if key not in states:
                    self.submit(key, self.slurm.queue(self.user))
                    states = {r["task_id"]: r for r in self.attempts()}
                    if any(
                        r["status"] in ("SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN")
                        for r in states.values()
                    ):
                        stage = "BLOCKED_UNKNOWN_SUBMISSION"
                        break
            if (
                acceptance
                and len(states) == 13
                and all(r["status"] == "COMPLETE" for r in states.values())
            ):
                self.release_all()
                stage = "EVALUATIONS_COMPLETE_REPORT_PENDING"
        result = dict(
            version=AMENDMENT_ID,
            at=now(),
            stage=stage,
            source_sha256=source_hash,
            scheduler_job_id=os.environ.get("SLURM_JOB_ID"),
            tasks=states,
            endpoints_complete=sum(r["status"] == "COMPLETE" for r in states.values()),
            endpoints_total=13,
            failures=failed,
            unknown_submissions=unknown,
            report_pending=True,
            experiment_delivered=False,
            teacher_gpu_usage=teacher_usage(self.slurm.queue(self.user), all_attempts),
            scores_sealed=acceptance is None
            or len(states) != 13
            or any(r["status"] != "COMPLETE" for r in states.values()),
        )
        save_json(self.directory / "STATE.json", result)
        return result
