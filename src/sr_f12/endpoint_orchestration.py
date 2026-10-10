"""Independent endpoint scheduler; never modifies or supersedes the training source.

The original controller, science registration and checkpoint recipe remain frozen.
This controller observes all twelve authenticated terminal jobs before submitting
any endpoint, then releases verified scores. Reporting remains a separate stage.
"""

from __future__ import annotations

import getpass
import json
import sys
import time
from pathlib import Path

from .orchestration import (
    Controller,
    Slurm,
    controller_lease,
    file_hash,
    now,
    read_json,
    save_json,
    teacher_usage,
)
from .protocol import AMENDMENT_ID, BASELINE, TEACHER_QOS, object_hash, scientific_matrix

MODEL_IDS = (BASELINE, *(row["model_id"] for row in scientific_matrix()))


class EndpointController(Controller):
    """Reuse held-submit mechanics while keeping a separate registry and source chain."""

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
        self.directory = self.root / "endpoint_orchestration"
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
            model_ids=list(MODEL_IDS),
            gpu_qos=TEACHER_QOS,
            gpu_limit=5,
        )
        save_json(self.directory / "CONFIG.json", self.config, exclusive=True)

    def identity(self):
        from .evaluation import verify_source_manifest

        if (self.root / "STOP").exists():
            raise PermissionError("Experiment STOP exists")
        manifest = read_json(self.source_manifest)
        if manifest.get("git_commit") != self.source_commit:
            raise PermissionError("Endpoint source manifest commit differs")
        source_hash = verify_source_manifest(self.source_manifest, self.code)
        amendment = read_json(self.root / "AMENDMENT.json")
        registration = read_json(self.root / "ENDPOINT_IMPLEMENTATION.json")
        expected = dict(
            status="REGISTERED_ENDPOINT_IMPLEMENTATION",
            amendment_id=AMENDMENT_ID,
            amendment_sha256=object_hash(amendment),
            training_source_commit=amendment["source_commit"],
            source_commit=self.source_commit,
            source_sha256=source_hash,
            scientific_settings_unchanged=True,
            model_ids=list(MODEL_IDS),
        )
        if amendment.get("amendment_id") != AMENDMENT_ID or any(
            registration.get(key) != value for key, value in expected.items()
        ):
            raise PermissionError("Endpoint implementation is not bound to the original amendment")
        model = read_json(self.root / "MODEL_ENVIRONMENT_IDENTITY.json")
        if Path(model["model_path"]).resolve(strict=True) != self.model:
            raise PermissionError("Endpoint base model path differs")
        return source_hash

    def main_attempts(self):
        directory = self.root / "orchestration/tasks"
        states = {p.parent.name: read_json(p) for p in directory.glob("*/STATE.json")}
        for path in directory.glob("*/INTENT.json"):
            if path.parent.name not in states:
                states[path.parent.name] = dict(
                    task_id=path.parent.name, status="SUBMISSION_UNKNOWN"
                )
        return list(states.values())

    def main_gate(self, queue):
        """The main registry, live accounting, and full state must all agree."""
        from .endpoint import verify_main_completion

        main_state = self.root / "orchestration/STATE.json"
        if not main_state.exists():
            return None
        snapshot = read_json(main_state)
        states = {row["task_id"]: row for row in self.main_attempts()}
        science = scientific_matrix()
        if any(
            states.get(run["model_id"], {}).get("status") != "COMPLETE"
            or snapshot.get("tasks", {}).get(run["model_id"], {}).get("status") != "COMPLETE"
            for run in science
        ):
            return None
        from .source_revision import resolve_source_revision

        revision = resolve_source_revision(self.root)
        training_commit = revision["source_commit"]
        if read_json(self.root / "SCIENCE_FREEZE.json").get("source_commit") != training_commit:
            raise PermissionError("Training source registration changed")
        terminals = []
        for run in science:
            key = run["model_id"]
            state = states[key]
            directory = self.root / "orchestration/tasks" / key
            intent = read_json(directory / "INTENT.json")
            job = state.get("job_id")
            if not job or job in queue:
                return None
            terminal = self.slurm.terminal(job)
            if terminal is None:
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
            if terminal != expected or terminal != read_json(directory / "TERMINAL.json"):
                raise PermissionError("A main path lacks authenticated terminal success")
            if file_hash(intent["script"]) != intent["script_sha256"]:
                raise PermissionError("Main job script identity changed")
            terminals.append(
                dict(
                    model_id=key,
                    terminal=terminal,
                    completion_receipt_sha256=file_hash(directory / "VERIFIED_COMPLETE.json"),
                )
            )
        result = verify_main_completion(self.root)
        acceptance = dict(
            status="ALL_TWELVE_TERMINAL_AND_STATE_VERIFIED",
            main_completion=result,
            main_completion_sha256=object_hash(result),
            terminals=terminals,
            training_source_commit=training_commit,
            registration_source_commit=revision["registration_source_commit"],
        )
        save_json(self.directory / "MAIN_ACCEPTANCE.json", acceptance, exclusive=True)
        return acceptance

    def command(self, key):
        if not key.startswith("EVAL_") or key[5:] not in MODEL_IDS:
            raise PermissionError("Unregistered endpoint task")
        return [
            self.python,
            str(self.code / "scripts/sr_f12/run_evaluation.py"),
            "--run-root",
            str(self.root),
            "--model-path",
            str(self.model),
            "--model-id",
            key[5:],
            "--source-manifest",
            str(self.source_manifest),
        ]

    def submit(self, key, queue):
        if teacher_usage(queue, [*self.attempts(), *self.main_attempts()]) >= 5:
            return False
        if self.main_gate(queue) is None:
            return False
        return super().submit(key, queue)

    def release(self, key, state, intent):
        queue = self.slurm.queue(self.user)
        if teacher_usage(queue, [*self.attempts(), *self.main_attempts()]) > 5:
            raise PermissionError("Combined teacher-QoS reservations exceed five; keeping held")
        if self.main_gate(queue) is None:
            raise PermissionError("Main terminal gate disappeared; keeping endpoint held")
        return super().release(key, state, intent)

    def verify_task(self, key):
        from .endpoint import read_completed_panel

        if not key.startswith("EVAL_") or key[5:] not in MODEL_IDS:
            raise PermissionError("Unregistered endpoint completion")
        model = key[5:]
        directory = self.root / "sealed/evaluation" / model
        receipt = read_json(directory / "COMPLETE.json")
        gate = read_json(self.directory / "MAIN_ACCEPTANCE.json")
        expected_panels = (
            {"final": 13824, "train_fit": 128, "chartqa": 2500}
            if model == BASELINE
            else {"final": 15104, "chartqa": 2500}
        )
        expected_count = sum(expected_panels.values())
        if (
            receipt.get("status") != "COMPLETE"
            or receipt.get("model_id") != model
            or receipt.get("records") != expected_count
            or receipt.get("main_completion_sha256") != gate["main_completion_sha256"]
            or receipt.get("source_sha256") != self.identity()
            or receipt.get("aggregate_scores_released") is not False
        ):
            raise PermissionError("Endpoint completion identity differs")
        panels = receipt.get("panels", [])
        if (
            len(panels) != len(expected_panels)
            or {row["panel"]: row["records"] for row in panels} != expected_panels
        ):
            raise PermissionError("Endpoint panels differ from registered coverage")
        seen = set()
        for panel in panels:
            panel_dir = directory / panel["panel"]
            if file_hash(panel_dir / "COMPLETE.json") != panel["completion_sha256"]:
                raise PermissionError("Endpoint panel receipt hash differs")
            rows = read_completed_panel(panel_dir)
            if len(rows) != panel["records"]:
                raise PermissionError("Endpoint panel count differs")
            for row in rows:
                if row["model_id"] != model or row["slot_id"] in seen:
                    raise PermissionError("Endpoint model identity or unique slot coverage differs")
                seen.add(row["slot_id"])
        return dict(
            task_id=key,
            model_id=model,
            records=expected_count,
            receipt_sha256=object_hash(receipt),
            main_completion_sha256=gate["main_completion_sha256"],
        )

    def release_all(self):
        from .endpoint import release_scores

        released = []
        for model in MODEL_IDS:
            state = self.task_state("EVAL_" + model)
            if not state or state["status"] != "COMPLETE":
                raise PermissionError(
                    "All endpoint jobs must be terminal and verified before release"
                )
            self.verify_task("EVAL_" + model)
            receipt = release_scores(self.root, model)
            expected = 16452 if model == BASELINE else 17604
            if (
                receipt.get("status") != "RELEASED_AFTER_ALL_MAIN_COMPLETE"
                or receipt.get("model_id") != model
                or receipt.get("count") != expected
                or receipt.get("records_sha256") != object_hash(receipt["records"])
            ):
                raise PermissionError("Released endpoint scores differ")
            released.append(
                dict(model_id=model, records=expected, release_sha256=object_hash(receipt))
            )
        return save_json(
            self.directory / "COMPLETE.json",
            dict(
                status="EVALUATIONS_COMPLETE_REPORT_PENDING",
                endpoint_models=13,
                endpoint_records=sum(row["records"] for row in released),
                releases=released,
                source_sha256=self.identity(),
                report_pending=True,
                experiment_delivered=False,
            ),
            exclusive=True,
        )

    def tick(self):
        source_hash = self.identity()
        queue = self.slurm.queue(self.user)
        acceptance = self.main_gate(queue)
        if acceptance is not None:
            self.refresh(queue)
        states = {row["task_id"]: row for row in self.attempts()}
        failed = [row for row in states.values() if row["status"] == "FAILED"]
        unknown = [
            row
            for row in [*states.values(), *self.main_attempts()]
            if row["status"] in ("SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN")
        ]
        stage = "WAITING_FOR_ALL_MAIN_TERMINAL"
        if failed:
            stage = "BLOCKED_ENDPOINT_FAILURE"
        elif unknown:
            stage = "BLOCKED_UNKNOWN_SUBMISSION"
        elif acceptance is not None:
            stage = "ENDPOINT_EVALUATION"
            for model in MODEL_IDS:
                key = "EVAL_" + model
                if key not in states:
                    self.submit(key, self.slurm.queue(self.user))
                    states = {row["task_id"]: row for row in self.attempts()}
                    if any(
                        row["status"] in ("SUBMISSION_UNKNOWN", "RELEASE_UNKNOWN")
                        for row in states.values()
                    ):
                        stage = "BLOCKED_UNKNOWN_SUBMISSION"
                        break
            if len(states) == 13 and all(row["status"] == "COMPLETE" for row in states.values()):
                self.release_all()
                stage = "EVALUATIONS_COMPLETE_REPORT_PENDING"
        snapshot = dict(
            version=AMENDMENT_ID,
            at=now(),
            stage=stage,
            source_sha256=source_hash,
            tasks=states,
            endpoints_complete=sum(row["status"] == "COMPLETE" for row in states.values()),
            endpoints_total=13,
            report_pending=True,
            experiment_delivered=False,
            failures=failed,
            unknown_submissions=unknown,
            teacher_gpu_usage=teacher_usage(
                self.slurm.queue(self.user), [*states.values(), *self.main_attempts()]
            ),
        )
        save_json(self.directory / "STATE.json", snapshot)
        return snapshot


def run_endpoint_controller(controller, *, once=False):
    with controller_lease(controller.directory):
        while True:
            result = controller.tick()
            print(json.dumps(result, sort_keys=True), flush=True)
            if (
                once
                or result["stage"].startswith("BLOCKED_")
                or result["stage"] == "EVALUATIONS_COMPLETE_REPORT_PENDING"
            ):
                return result
            time.sleep(60)
