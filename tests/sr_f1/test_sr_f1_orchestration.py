"""SR-F1 CPU state-machine simulations; these never submit a real Slurm job."""

import copy
import importlib.util
import json
import os
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from sr_f1.orchestration import (
    PLAN_ID,
    Scheduler,
    SlurmBackend,
    atomic_json,
    checkpoint_task,
    complete_task,
    controller_incarnation,
    digest,
    fail_task,
    read_json,
    request_controller_requeue,
    tasks_for,
)

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "docs/sr_f1/package"


class SimulatedSlurm:
    def __init__(self, root):
        self.root = root
        self.jobs = {}
        self.external_gpus = 0
        self.ambiguous = False
        self.outage = False
        self.capacity_outage = False
        self.maximum_live_gpus = 0

    def submission_capacity(self, permission):
        if self.capacity_outage:
            raise TimeoutError("quota unknown")
        return {
            "owner": permission["owner"],
            "qos": permission["qos"],
            "max_submit_jobs_per_user": 5,
            "queued_job_ids": [
                job_id
                for job_id, job in self.jobs.items()
                if job["state"] in {"PENDING", "RUNNING"}
            ],
        }

    def project_capacity(self, permission):
        return {
            "owner": permission["owner"],
            "jobs": [
                {"job_id": "9000", "gpus": self.external_gpus, "state": "RUNNING"},
                *[
                    {"job_id": job_id, "gpus": job["gpus"], "state": job["state"]}
                    for job_id, job in self.jobs.items()
                    if job["state"] in {"PENDING", "RUNNING"}
                ],
            ],
        }

    def submit(self, command):
        last = json.loads((self.root / "orchestration/journal.jsonl").read_text().splitlines()[-1])
        assert last["event"] == "SUBMISSION_INTENT_DURABLE"
        task_id = last["details"]["task_id"]
        assert last["state"]["tasks"][task_id]["status"] == "SUBMITTING"
        job_id = str(1000 + len(self.jobs))
        self.jobs[job_id] = {
            "task_id": task_id,
            "name": command[command.index("--job-name") + 1],
            "comment": command[command.index("--comment") + 1],
            "state": "PENDING",
            "held": True,
            "gpus": int("--gres" in command),
            "exit_code": "0:0",
        }
        current = self.external_gpus + sum(
            job["gpus"] for job in self.jobs.values() if job["state"] in {"PENDING", "RUNNING"}
        )
        assert current <= 5
        self.maximum_live_gpus = max(current, self.maximum_live_gpus)
        if self.ambiguous:
            self.ambiguous = False
            raise TimeoutError("accepted but SSH response lost")
        return {"returncode": 0, "stdout": job_id + "\n", "stderr": ""}

    def release(self, job_id):
        assert any(
            row["job_id"] == job_id
            for row in read_json(self.root / "manifests/ALLOCATIONS.json")["allocations"]
        )
        self.jobs[job_id].update(state="RUNNING", held=False)
        return {"returncode": 0, "stdout": "", "stderr": ""}

    def observe(self, attempt, permission):
        if self.outage:
            raise TimeoutError("SSH unknown")
        queue, accounting = [], []
        for job_id, job in self.jobs.items():
            if job["name"] != attempt["job_name"]:
                continue
            if job["state"] in {"PENDING", "RUNNING"}:
                queue.append(
                    {
                        "job_id": job_id,
                        "state": job["state"],
                        "reason": "JobHeldUser" if job["held"] else "None",
                    }
                )
            else:
                start = datetime(2026, 10, 9)
                accounting.append(
                    {
                        "JobIDRaw": job_id,
                        "JobName": job["name"],
                        "Comment": job["comment"],
                        "User": permission["owner"],
                        "Account": permission["account"],
                        "QOS": permission["qos"],
                        "State": job["state"],
                        "ElapsedRaw": "10",
                        "Start": start.isoformat(),
                        "End": (start + timedelta(seconds=10)).isoformat(),
                        "ExitCode": job["exit_code"],
                        "AllocTRES": "gres/gpu=1,gres/gpu:pro6000=1" if job["gpus"] else "cpu=1",
                    }
                )
        return {"queue": queue, "accounting": accounting}


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()
        self.root = self.base / "run"
        self.plan = PACKAGE / "config/SR_F1.json"
        self.matrix = read_json(PACKAGE / "manifests/RUN_MATRIX.json")
        atomic_json(self.root / "manifests/RUN_MATRIX.json", self.matrix)
        atomic_json(
            self.root / "EXECUTION_FREEZE.json",
            {"plan_id": PLAN_ID, "status": "FROZEN", "fixture": True},
        )
        atomic_json(
            self.root / "manifests/ALLOCATION_PERMISSION.json",
            {
                "plan_id": PLAN_ID,
                "authorized": True,
                "run_root": str(self.root),
                "owner": "owner",
                "account": "rose",
                "qos": "teacher",
                "gres": "gpu:pro6000:1",
                "gpus_per_worker": 1,
                "max_gpus_concurrent": 5,
                "verified_at": "2026-10-09T00:00:00+00:00",
            },
        )
        self.backend = SimulatedSlurm(self.root)
        self.engine_verified = False
        self.calls = []
        self.scheduler = self.new_scheduler()

    def verify(self, plan, root, require_engine=False):
        self.calls.append(require_engine)
        if require_engine and not self.engine_verified:
            raise ValueError("ENGINE_NOT_VERIFIED")

    def new_scheduler(self):
        return Scheduler(
            self.plan,
            self.root,
            code_root=REPO,
            python="/usr/bin/python3",
            backend=self.backend,
            verify_execution=self.verify,
        )

    def marker(
        self,
        task_id,
        *,
        checkpoint=False,
        failure=False,
        reason="TIME_LEASE_END",
        engine_restart=False,
    ):
        task = self.scheduler.state["tasks"][task_id]
        attempt = task["attempts"][-1]
        environment = {
            "SR_F1_TASK_ID": task_id,
            "SR_F1_ATTEMPT_ID": attempt["attempt_id"],
            "SR_F1_REGISTRATION_HASH": digest(self.scheduler.registration),
        }
        relative = "artifacts/" + attempt["attempt_id"] + ".json"
        atomic_json(self.root / relative, {"task": task_id, "update": 8})
        with patch.dict(os.environ, environment):
            if failure:
                fail_task(self.root, task_id, "preserved technical failure")
            elif checkpoint:
                checkpoint_task(
                    self.root,
                    task_id,
                    reason,
                    [relative],
                    {
                        "full_state": True,
                        "identity_verified": True,
                        "next_update": 8,
                        "restart_engine_track_required": engine_restart,
                    },
                )
            else:
                complete_task(self.root, task_id, [relative])
        return attempt

    def finish(self, task_id, *, checkpoint=False, failure=False, terminal="COMPLETED"):
        attempt = self.marker(task_id, checkpoint=checkpoint, failure=failure)
        self.backend.jobs[attempt["job_id"]]["state"] = terminal
        return attempt

    def through_baseline(self):
        self.scheduler.tick()
        self.finish("COMMON_START")
        self.scheduler.tick()
        self.finish("ENGINE")
        self.engine_verified = True
        self.scheduler.tick()
        self.finish("BASELINE")

    def test_fixed_matrix_and_common_start_gate(self):
        tasks = tasks_for(self.matrix)
        self.assertEqual(sum(task["phase"] == "S4" for task in tasks.values()), 15)
        self.assertEqual(sum(task["phase"] == "S5" for task in tasks.values()), 16)
        changed = copy.deepcopy(self.matrix)
        changed[0]["H"] = 97
        with self.assertRaisesRegex(ValueError, "BUDGET"):
            tasks_for(changed)
        state = self.scheduler.tick()
        self.assertEqual(next(iter(self.backend.jobs.values()))["task_id"], "COMMON_START")
        self.assertTrue(state["test_sealed"])
        self.assertEqual(state["tasks"]["ENGINE"]["status"], "WAITING")

    def test_engine_receipt_not_just_completed_job(self):
        self.scheduler.tick()
        self.finish("COMMON_START")
        self.scheduler.tick()
        self.finish("ENGINE")
        with self.assertRaisesRegex(ValueError, "ENGINE_NOT_VERIFIED"):
            self.scheduler.tick()
        self.assertEqual(len(self.backend.jobs), 2)

    def test_common_protocol_full_state_lease_resume_does_not_release_engine(self):
        self.scheduler.tick()
        self.finish("COMMON_START", checkpoint=True, terminal="TIMEOUT")
        state = self.scheduler.tick()
        self.assertEqual(len(state["tasks"]["COMMON_START"]["attempts"]), 2)
        self.assertEqual(state["tasks"]["ENGINE"]["status"], "WAITING")

    def test_engine_lease_requires_explicit_whole_trace_restart_capability(self):
        self.scheduler.tick()
        self.finish("COMMON_START")
        self.scheduler.tick()
        self.finish("ENGINE", checkpoint=True, terminal="TIMEOUT")
        state = self.scheduler.tick()
        self.assertEqual(state["tasks"]["ENGINE"]["status"], "BLOCKED")
        self.assertEqual(len(state["tasks"]["ENGINE"]["attempts"]), 1)
        self.assertEqual(state["tasks"]["BASELINE"]["status"], "WAITING")

    def test_engine_verified_full_state_can_restart_whole_registered_trace(self):
        self.scheduler.tick()
        self.finish("COMMON_START")
        self.scheduler.tick()
        attempt = self.marker("ENGINE", checkpoint=True, engine_restart=True)
        self.backend.jobs[attempt["job_id"]]["state"] = "TIMEOUT"
        state = self.scheduler.tick()
        self.assertEqual(len(state["tasks"]["ENGINE"]["attempts"]), 2)
        self.assertEqual(state["tasks"]["BASELINE"]["status"], "WAITING")

    def test_unknown_submission_no_duplicate_across_restart(self):
        self.backend.ambiguous = True
        self.scheduler.tick()
        self.assertEqual(self.scheduler.state["tasks"]["COMMON_START"]["status"], "UNKNOWN")
        self.backend.outage = True
        self.scheduler = self.new_scheduler()
        for _ in range(3):
            self.scheduler.tick()
        self.assertEqual(len(self.backend.jobs), 1)
        self.backend.outage = False
        self.scheduler.tick()
        self.assertEqual(len(self.backend.jobs), 1)
        self.assertEqual(self.scheduler.state["tasks"]["COMMON_START"]["status"], "ACTIVE")

    def test_whole_project_five_gpu_cap_includes_old_jobs(self):
        self.through_baseline()
        self.backend.external_gpus = 3
        self.scheduler.tick()
        active = [job for job in self.backend.jobs.values() if job["state"] == "RUNNING"]
        self.assertEqual(len(active), 2)
        self.assertEqual(self.backend.maximum_live_gpus, 5)
        self.assertFalse(
            any(job["task_id"].startswith("EVAL_") for job in self.backend.jobs.values())
        )

    def test_capacity_query_outage_submits_nothing(self):
        self.backend.capacity_outage = True
        state = self.scheduler.tick()
        self.assertEqual(state["capacity"]["status"], "UNKNOWN")
        self.assertEqual(len(self.backend.jobs), 0)

    def test_timeout_requires_full_state_not_only_terminal(self):
        self.through_baseline()
        self.scheduler.tick()
        run_id = self.matrix[0]["run_id"]
        attempt = self.scheduler.state["tasks"][run_id]["attempts"][-1]
        self.backend.jobs[attempt["job_id"]]["state"] = "TIMEOUT"
        self.scheduler.tick()
        self.assertEqual(self.scheduler.state["tasks"][run_id]["status"], "TECHNICAL_FAILED")
        self.assertEqual(len(self.scheduler.state["tasks"][run_id]["attempts"]), 1)

    def test_verified_full_state_continues_identical_logical_path(self):
        self.through_baseline()
        self.scheduler.tick()
        run_id = self.matrix[0]["run_id"]
        first = self.finish(run_id, checkpoint=True, terminal="TIMEOUT")
        self.scheduler.tick()
        task = self.scheduler.state["tasks"][run_id]
        self.assertEqual(len(task["attempts"]), 2)
        self.assertEqual(task["attempts"][0]["job_id"], first["job_id"])
        self.assertEqual(task["attempts"][0]["accounting"]["gpu_seconds"], 10)
        script = self.root / "orchestration/attempts" / (task["attempts"][1]["attempt_id"] + ".sh")
        self.assertIn("SR_F1_RESUME_RECEIPT", script.read_text())

    def test_stop_checkpoint_requires_explicit_resume(self):
        self.through_baseline()
        self.scheduler.tick()
        run_id = self.matrix[0]["run_id"]
        attempt = self.marker(run_id, checkpoint=True, reason="STOP_REQUESTED")
        self.backend.jobs[attempt["job_id"]]["state"] = "COMPLETED"
        state = self.scheduler.tick()
        self.assertEqual(state["phase"], "STOPPED")
        self.assertEqual(state["tasks"][run_id]["status"], "STOPPED")
        self.scheduler.tick()
        self.assertEqual(len(self.scheduler.state["tasks"][run_id]["attempts"]), 1)
        self.assertEqual(self.scheduler.resume_stopped(), [run_id])
        self.scheduler.tick()
        self.assertEqual(len(self.scheduler.state["tasks"][run_id]["attempts"]), 2)

    def test_failed_arm_remains_and_test_waits_all_fifteen(self):
        self.through_baseline()
        self.scheduler.tick()
        failed_id = self.matrix[0]["run_id"]
        self.finish(failed_id, failure=True, terminal="FAILED")
        self.scheduler.tick()
        self.assertEqual(self.scheduler.state["tasks"][failed_id]["status"], "TECHNICAL_FAILED")
        self.assertFalse((self.root / "RUN_MATRIX_FINAL.json").exists())
        while not self.scheduler._science_terminal():
            for row in self.matrix:
                if self.scheduler.state["tasks"][row["run_id"]]["status"] in {
                    "REGISTERED",
                    "ACTIVE",
                }:
                    self.finish(row["run_id"])
            self.scheduler.tick()
        final = read_json(self.root / "RUN_MATRIX_FINAL.json")
        self.assertEqual(len(final["runs"]), 15)
        self.assertEqual(final["runs"][0]["status"], "TECHNICAL_FAILED")
        self.assertIn("preserved", final["runs"][0]["failure"])
        self.assertFalse(self.scheduler.state["test_sealed"])

    def test_stop_prevents_new_submissions_without_cancel(self):
        self.scheduler.tick()
        (self.root / "STOP").write_text("explicit stop\n")
        self.finish("COMMON_START")
        state = self.scheduler.tick()
        self.assertEqual(state["phase"], "STOPPED")
        self.assertEqual(len(self.backend.jobs), 1)
        self.assertEqual(state["tasks"]["ENGINE"]["status"], "WAITING")

    def test_journal_tampering_fails_closed(self):
        self.scheduler.tick()
        journal = self.root / "orchestration/journal.jsonl"
        journal.write_text(journal.read_text().replace('"S1_VERIFIED"', '"RELEASED"', 1))
        with self.assertRaisesRegex(ValueError, "JOURNAL_INTEGRITY"):
            self.new_scheduler().tick()

    def test_squeue_unknown_gpu_shape_fails_closed(self):
        backend = SlurmBackend()
        replies = [
            {"returncode": 0, "stdout": "1|owner|RUNNING\n2|owner|PENDING\n"},
            {
                "returncode": 0,
                "stdout": (
                    "JobId=1 UserId=owner(1) ReqTRES=cpu=4,gres/gpu=1,"
                    "gres/gpu:pro6000=1 AllocTRES=gres/gpu=1"
                ),
            },
            {
                "returncode": 0,
                "stdout": "JobId=2 UserId=owner(1) ReqTRES=cpu=8,gres/gpu:pro6000=2 AllocTRES=",
            },
        ]
        with patch.object(backend, "_run", side_effect=replies):
            observed = backend.project_capacity({"owner": "owner"})
        self.assertEqual([row["gpus"] for row in observed["jobs"]], [1, 2])
        with (
            patch.object(
                backend,
                "_run",
                side_effect=[
                    {"returncode": 0, "stdout": "1|owner|RUNNING\n"},
                    {"returncode": 0, "stdout": "JobId=1 UserId=owner(1) ReqTRES=gres/gpu=unknown"},
                ],
            ),
            self.assertRaisesRegex(ValueError, "INVALID_GPU_TRES"),
        ):
            backend.project_capacity({"owner": "owner"})

    def test_worker_dispatch_resolves_plan_and_common_step_zero(self):
        module_spec = importlib.util.spec_from_file_location(
            "srf1_worker_test", REPO / "scripts/sr_f1/run_worker.py"
        )
        worker = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(worker)
        calls = []

        def load(plan, root, model_id, step):
            self.assertIsInstance(plan, dict)
            self.assertEqual(plan["version"], PLAN_ID)
            calls.append((model_id, step))
            return object()

        def evaluate(runtime, root, model_id, stage, step):
            calls.append((stage, step))
            return {"status": "COMPLETE", "artifacts": ["raw.jsonl"]}

        with patch.dict(
            "sys.modules",
            {
                "sr_f1.runtime": types.SimpleNamespace(load_for_evaluation=load),
                "sr_f1.evaluation": types.SimpleNamespace(evaluate_model=evaluate),
            },
        ):
            worker.dispatch(
                self.plan,
                self.root,
                {"operation": "evaluate", "model_id": "SRF1_COMMON_START", "stage": "baseline"},
            )
        self.assertEqual(calls, [("SRF1_COMMON_START", 0), ("baseline", 0)])

    def test_controller_requeue_uncertainty_has_one_intent_and_positive_restart(self):
        backend = SlurmBackend()
        commands = []

        def run(command):
            commands.append(command)
            if command[1] == "requeue":
                raise TimeoutError("requeue response lost")
            return {
                "returncode": 0,
                "stdout": (
                    "JobId=123 UserId=owner(1) Account=rose JobState=RUNNING AllocTRES=cpu=1,mem=2G"
                ),
            }

        with (
            patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "0"}),
            patch.object(backend, "_run", side_effect=run),
        ):
            identity = controller_incarnation(self.root, backend=backend)
            result = request_controller_requeue(self.root, identity, backend=backend)
            self.assertEqual(result["status"], "UNKNOWN")
            with self.assertRaisesRegex(ValueError, "ALREADY_REQUESTED"):
                request_controller_requeue(self.root, identity, backend=backend)
        self.assertEqual(sum(command[1] == "requeue" for command in commands), 1)
        with (
            patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "1"}),
            patch.object(backend, "_run", side_effect=run),
        ):
            self.assertEqual(controller_incarnation(self.root, backend=backend)["restart"], 1)

    def test_worker_needs_positive_live_identity_before_model_work(self):
        self.scheduler.tick()
        module_spec = importlib.util.spec_from_file_location(
            "srf1_worker_identity_test", REPO / "scripts/sr_f1/run_worker.py"
        )
        worker = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(worker)
        attempt = self.scheduler.state["tasks"]["COMMON_START"]["attempts"][-1]
        environment = {
            "SR_F1_TASK_ID": "COMMON_START",
            "SR_F1_ATTEMPT_ID": attempt["attempt_id"],
            "SR_F1_REGISTRATION_HASH": digest(self.scheduler.registration),
            "SLURM_JOB_ID": attempt["job_id"],
        }
        details = (
            f"JobId={attempt['job_id']} JobName={attempt['job_name']} "
            f"Comment={attempt['comment']} UserId=owner(1) Account=rose "
            "QOS=teacher JobState=RUNNING AllocTRES=gres/gpu=1,gres/gpu:pro6000=1"
        )
        backend = SlurmBackend()
        with (
            patch.dict(os.environ, environment),
            patch.object(
                backend,
                "_run",
                return_value={
                    "returncode": 0,
                    "stdout": details.replace(attempt["comment"], "wrong-study"),
                },
            ),
            self.assertRaisesRegex(ValueError, "LIVE_WORKER_IDENTITY_MISMATCH"),
        ):
            worker.validate_allocation(self.root, "COMMON_START", backend=backend)
        with (
            patch.dict(os.environ, environment),
            patch.object(backend, "_run", return_value={"returncode": 0, "stdout": details}),
        ):
            self.assertEqual(
                worker.validate_allocation(self.root, "COMMON_START", backend=backend)["operation"],
                "common_start",
            )


if __name__ == "__main__":
    unittest.main()
