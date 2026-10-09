"""CPU-only simulation of durable scheduling; no Slurm command is executed."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from mm_dev.contract import PLAN_ID, run_matrix
from mm_dev.orchestration import (
    AUTO_RESUME,
    LIVE,
    Scheduler,
    SlurmBackend,
    atomic_json,
    checkpoint_task,
    complete_task,
    digest,
    fail_task,
    process_lease,
    read_json,
    summarize_accounting,
    tasks_for,
    worker_lease,
)

REPOSITORY = Path(__file__).resolve().parents[2]


class SimulatedSlurm:
    def __init__(self, root):
        self.root = root
        self.jobs = {}
        self.submissions = []
        self.ambiguous_next = False
        self.query_outage = False
        self.maximum_live_gpus = 0
        self.submit_limit = 5
        self.external_jobs = 0
        self.capacity_outage = False
        self.hide_queued_jobs = False

    def submission_capacity(self, permission):
        if self.capacity_outage:
            raise TimeoutError("capacity query unavailable")
        jobs = [str(9000 + index) for index in range(self.external_jobs)]
        if not self.hide_queued_jobs:
            jobs += [
                job["job_id"]
                for job in self.jobs.values()
                if job["state"] in {"PENDING", "RUNNING"}
            ]
        return {
            "owner": permission["owner"],
            "qos": permission["qos"],
            "max_submit_jobs_per_user": self.submit_limit,
            "queued_job_ids": jobs,
            "receipts": {"simulated": True},
        }

    def submit(self, command):
        row = json.loads((self.root / "orchestration/journal.jsonl").read_text().splitlines()[-1])
        assert row["event"] == "SUBMISSION_INTENT_DURABLE"
        task_id = row["details"]["task_id"]
        assert row["state"]["tasks"][task_id]["status"] == "SUBMITTING"
        job_id = str(1000 + len(self.jobs))
        spec = {
            "task_id": task_id,
            "job_id": job_id,
            "name": command[command.index("--job-name") + 1],
            "comment": command[command.index("--comment") + 1],
            "state": "PENDING",
            "held": True,
            "gpus": int("--gres" in command),
            "elapsed": 10,
            "exit_code": "0:0",
        }
        self.jobs[job_id] = spec
        self.submissions.append(task_id)
        live = sum(
            job["gpus"] for job in self.jobs.values() if job["state"] in {"PENDING", "RUNNING"}
        )
        self.maximum_live_gpus = max(live, self.maximum_live_gpus)
        assert live <= 5
        if self.ambiguous_next:
            self.ambiguous_next = False
            raise TimeoutError("sbatch accepted but response lost")
        return {"returncode": 0, "stdout": job_id + "\n", "stderr": ""}

    def release(self, job_id):
        allocations = read_json(self.root / "manifests/ALLOCATIONS.json")["allocations"]
        assert any(row["job_id"] == job_id for row in allocations)
        self.jobs[job_id]["held"] = False
        self.jobs[job_id]["state"] = "RUNNING"
        return {"returncode": 0, "stdout": "", "stderr": ""}

    def observe(self, attempt, permission):
        if self.query_outage:
            raise TimeoutError("SSH/scheduler unavailable")
        matching = [job for job in self.jobs.values() if job["name"] == attempt["job_name"]]
        queue, accounting = [], []
        for job in matching:
            if job["state"] in {"PENDING", "RUNNING"}:
                queue.append(
                    {
                        "job_id": job["job_id"],
                        "state": job["state"],
                        "reason": "JobHeldUser" if job["held"] else "None",
                    }
                )
            else:
                start = datetime(2026, 10, 9)
                end = start + timedelta(seconds=job["elapsed"])
                accounting.append(
                    {
                        "JobIDRaw": job["job_id"],
                        "JobName": job["name"],
                        "Comment": job["comment"],
                        "User": permission["owner"],
                        "Account": permission["account"],
                        "QOS": permission["qos"],
                        "State": job["state"],
                        "ElapsedRaw": str(job["elapsed"]),
                        "Start": start.isoformat(),
                        "End": end.isoformat(),
                        "ExitCode": job["exit_code"],
                        "AllocTRES": "gres/gpu=1,gres/gpu:pro6000=1" if job["gpus"] else "cpu=1",
                    }
                )
        return {"queue": queue, "accounting": accounting, "receipts": {"simulated": True}}


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root, self.code = self.base / "run", self.base / "code"
        self.plan = self.base / "config/MM_DEV_F2.json"
        self.plan.parent.mkdir()
        source = REPOSITORY / "docs/mm_dev_f2/design/config/MM_DEV_F2.json"
        self.plan.write_bytes(source.read_bytes())
        atomic_json(self.plan.parent / "run_matrix.json", run_matrix())
        atomic_json(self.root / "manifests/F2_FREEZE.json", {"plan_id": PLAN_ID, "fixture": True})
        atomic_json(
            self.root / "manifests/ALLOCATION_PERMISSION.json",
            {
                "plan_id": PLAN_ID,
                "authorized": True,
                "run_root": str(self.root),
                "owner": "testowner",
                "account": "explicit_account",
                "qos": "explicit_qos",
                "gres": "gpu:pro6000:1",
                "gpus_per_worker": 1,
                "max_gpus_concurrent": 5,
                "verified_at": "2026-10-09T00:00:00+00:00",
                "partition": None,
            },
        )
        for task in tasks_for(run_matrix()).values():
            script = self.code / "scripts/mm_dev" / task["script"]
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("# never executed by CPU simulator\n")
        self.backend = SimulatedSlurm(self.root)
        self.gates = []
        self.scheduler = self.new_scheduler()

    def verify(self, plan, root, require_engine=False):
        self.gates.append(require_engine)
        self.assertEqual(Path(plan), self.plan)
        self.assertEqual(Path(root), self.root)
        if require_engine:
            self.assertTrue((self.root / "orchestration/completions/ENGINE_F2.json").is_file())
        return {"test_only": True}

    def new_scheduler(self, shards=1):
        return Scheduler(
            self.plan,
            self.root,
            code_root=self.code,
            python=Path(sys.executable),
            lease_minutes=60,
            measurement_shards=shards,
            backend=self.backend,
            verify_execution=self.verify,
        )

    def environment(self, task_id):
        registration = read_json(self.root / "orchestration/REGISTRATION.json")
        state = read_json(self.root / "orchestration/STATE.json")
        attempt = state["tasks"][task_id]["attempts"][-1]
        return {
            "MM_DEV_TASK_ID": task_id,
            "MM_DEV_ATTEMPT_ID": attempt["attempt_id"],
            "MM_DEV_REGISTRATION_HASH": digest(registration),
        }

    def finish(self, task_id, terminal="COMPLETED", marker=True, checkpoint=False, elapsed=10):
        job = next(
            job for job in reversed(list(self.backend.jobs.values())) if job["task_id"] == task_id
        )
        job["state"], job["elapsed"] = terminal, elapsed
        if terminal != "COMPLETED":
            job["exit_code"] = "1:0"
        environment = self.environment(task_id)
        relative = "evidence/" + environment["MM_DEV_ATTEMPT_ID"] + ".json"
        atomic_json(
            self.root / relative,
            {
                "sample_seed": 123,
                "saved_logical_slots": [0, 1, 2],
                "completion_scope": "simulation_only",
            },
        )
        with patch.dict(os.environ, environment):
            if checkpoint:
                checkpoint_task(self.root, task_id, "TIME_LEASE_END", [relative])
            elif marker:
                complete_task(self.root, task_id, [relative], {"test_only": True})

    def finish_all_running(self):
        for job in list(self.backend.jobs.values()):
            if job["state"] == "RUNNING":
                self.finish(job["task_id"])

    def test_full_fixed_matrix_reaches_release_and_restart_has_no_duplicates(self):
        phases = []
        state = self.scheduler.tick()
        self.assertEqual(self.backend.submissions, ["ENGINE_F2"])
        self.finish("ENGINE_F2")
        state = self.new_scheduler().tick()
        self.assertEqual(
            self.backend.submissions[1:6],
            ["prep_A_0", "prep_AP_0", "prep_A_1", "prep_AP_1", "probe_S0"],
        )
        for _ in range(40):
            phases.append(state["phase"])
            self.finish_all_running()
            state = self.new_scheduler().tick()
            if state["phase"] == "RELEASED":
                break
        self.assertEqual(state["phase"], "RELEASED")
        self.assertEqual(len(self.backend.submissions), 77)
        self.assertEqual(len(set(self.backend.submissions)), 77)
        self.assertEqual(self.backend.maximum_live_gpus, 5)
        self.assertEqual(state["accounting"]["unresolved_attempt_ids"], [])
        self.assertEqual(state["accounting"]["known_actual_gpu_seconds"], 750)
        self.assertIsNone(state["accounting"]["max_allocated_gpu_hours"])
        self.assertIsNone(state["accounting"]["max_wallclock_hours_for_study"])
        gpu_receipt = read_json(self.root / "manifests/GPU_ACCOUNTING_FINAL.json")
        self.assertEqual(gpu_receipt["actual_gpu_seconds"], 750)
        self.assertEqual(gpu_receipt["gpu_task_count"], 75)
        self.assertEqual(len(gpu_receipt["allocations"]), 75)
        self.assertIn("PROBE_COMPLETE", phases)
        self.assertIn("RESPONSES_COMPLETE", phases)
        self.assertIn("EVAL_COMPLETE", phases)
        self.assertIn("ANALYZED", phases)
        for task_id, task in state["tasks"].items():
            with self.subTest(task=task_id):
                command = task["attempts"][0]["command"]
                self.assertIn("--cpus-per-task=4", command)
                self.assertIn("--mem=64G", command)
                script = Path(command[-1]).read_text()
                self.assertIn("export OMP_NUM_THREADS=4\n", script)
                self.assertIn("export MKL_NUM_THREADS=4\n", script)
                if task_id in {"ANALYZE", "RELEASE"}:
                    self.assertNotIn("--gres", command)
        self.new_scheduler().tick()
        self.assertEqual(len(self.backend.submissions), 77)

    def test_unknown_submission_is_reconciled_and_never_blindly_resubmitted(self):
        self.backend.ambiguous_next = True
        state = self.scheduler.tick()
        self.assertEqual(state["tasks"]["ENGINE_F2"]["status"], "UNKNOWN")
        self.backend.query_outage = True
        for _ in range(2):
            state = self.new_scheduler().tick()
        self.assertEqual(self.backend.submissions, ["ENGINE_F2"])
        self.assertEqual(state["tasks"]["ENGINE_F2"]["status"], "UNKNOWN")
        self.backend.query_outage = False
        state = self.new_scheduler().tick()
        self.assertEqual(state["tasks"]["ENGINE_F2"]["status"], "ACTIVE")
        self.assertFalse(next(iter(self.backend.jobs.values()))["held"])
        self.assertEqual(self.backend.submissions, ["ENGINE_F2"])

    def test_unknown_submission_without_positive_match_stays_reserved(self):
        self.backend.ambiguous_next = True
        self.scheduler.tick()
        self.backend.jobs.clear()
        for _ in range(3):
            state = self.new_scheduler().tick()
        self.assertEqual(state["tasks"]["ENGINE_F2"]["status"], "UNKNOWN")
        self.assertEqual(self.backend.submissions, ["ENGINE_F2"])

    def test_preemption_timelease_and_partial_saved_slots_resume_same_task(self):
        self.scheduler.tick()
        self.finish("ENGINE_F2")
        self.scheduler.tick()
        self.finish("prep_A_0", "PREEMPTED", marker=False)
        self.finish("prep_AP_0", "TIMEOUT", marker=False)
        self.finish("prep_A_1", marker=False, checkpoint=True)
        before = (self.root / "evidence/prep_A_1_attempt0000.json").read_bytes()
        state = self.new_scheduler().tick()
        for task_id in ("prep_A_0", "prep_AP_0", "prep_A_1"):
            self.assertEqual(len(state["tasks"][task_id]["attempts"]), 2)
            self.assertEqual(self.backend.submissions.count(task_id), 2)
            first, second = state["tasks"][task_id]["attempts"]
            self.assertEqual(first["accounting"]["gpu_seconds"], 10)
            self.assertNotEqual(first["job_id"], second["job_id"])
            manifest = read_json(self.root / second["manifest_path"])
            self.assertEqual(manifest["worker_command"][-2:], ["--run-id", task_id])
        self.assertEqual((self.root / "evidence/prep_A_1_attempt0000.json").read_bytes(), before)
        self.assertNotIn("cont_S0_a0_f0", self.backend.submissions)
        for _ in range(40):
            self.finish_all_running()
            state = self.new_scheduler().tick()
            if state["phase"] == "RELEASED":
                break
        self.assertEqual(state["phase"], "RELEASED")
        self.assertEqual(
            read_json(self.root / "manifests/GPU_ACCOUNTING_FINAL.json")["actual_gpu_seconds"], 780
        )

    def test_software_failure_and_missing_completion_block_only_affected_task(self):
        self.scheduler.tick()
        self.finish("ENGINE_F2")
        self.scheduler.tick()
        self.finish("prep_A_0", "FAILED", marker=False)
        self.finish("prep_AP_0", "COMPLETED", marker=False)
        self.finish("prep_A_1")
        state = self.scheduler.tick()
        self.assertEqual(state["tasks"]["prep_A_0"]["status"], "BLOCKED")
        self.assertEqual(state["tasks"]["prep_AP_0"]["status"], "BLOCKED")
        self.assertEqual(state["tasks"]["prep_A_1"]["status"], "COMPLETE")
        self.assertIn("probe_S_A_1", self.backend.submissions)
        self.assertNotIn("cont_S0_a0_f0", self.backend.submissions)
        self.assertEqual(self.backend.submissions.count("prep_A_0"), 1)

    def test_failure_marker_prevents_preemption_hiding_nan_or_gradient_bug(self):
        self.scheduler.tick()
        with patch.dict(os.environ, self.environment("ENGINE_F2")):
            fail_task(self.root, "ENGINE_F2", "NaN/GRADIENT_MISMATCH")
        self.finish("ENGINE_F2", "PREEMPTED", marker=False)
        state = self.scheduler.tick()
        self.assertEqual(state["tasks"]["ENGINE_F2"]["status"], "BLOCKED")
        self.assertEqual(self.backend.submissions, ["ENGINE_F2"])

    def test_worker_lease_blocks_resume_even_after_scheduler_reports_terminal(self):
        self.scheduler.tick()
        self.finish("ENGINE_F2")
        self.scheduler.tick()
        self.finish("prep_A_0", "TIMEOUT", marker=False)
        with worker_lease(self.root, "prep_A_0"):
            state = self.scheduler.tick()
        self.assertEqual(state["tasks"]["prep_A_0"]["status"], "UNKNOWN")
        self.assertEqual(self.backend.submissions.count("prep_A_0"), 1)
        self.scheduler.tick()
        self.assertEqual(self.backend.submissions.count("prep_A_0"), 2)

    def test_interrupted_engine_continuous_trace_needs_explicit_repair(self):
        self.scheduler.tick()
        self.finish("ENGINE_F2", "PREEMPTED", marker=False)
        state = self.scheduler.tick()
        self.assertEqual(state["tasks"]["ENGINE_F2"]["status"], "BLOCKED")
        self.assertIn("CONTINUOUS_TRACE", state["tasks"]["ENGINE_F2"]["blocker"])
        self.assertEqual(self.backend.submissions, ["ENGINE_F2"])

    def test_controller_lease_is_exclusive(self):
        with (
            process_lease(self.root / "orchestration/controller.lock"),
            self.assertRaisesRegex(RuntimeError, "LEASE_BUSY"),
        ):
            self.scheduler.tick()
        self.assertEqual(self.backend.submissions, [])

    def test_checkpoint_reason_and_marker_identity_are_strict(self):
        self.scheduler.tick()
        with (
            patch.dict(os.environ, self.environment("ENGINE_F2")),
            self.assertRaisesRegex(ValueError, "NONRESUMABLE_REASON"),
        ):
            checkpoint_task(self.root, "ENGINE_F2", "NaN", ["irrelevant"])
        self.finish("ENGINE_F2")
        artifact = self.root / "evidence/ENGINE_F2_attempt0000.json"
        artifact.write_text("tampered")
        state = self.scheduler.tick()
        self.assertEqual(state["tasks"]["ENGINE_F2"]["status"], "BLOCKED")

    def test_matrix_and_permission_drift_rejected_before_submission(self):
        matrix = run_matrix()
        matrix[4]["steps"] = 33
        with self.assertRaisesRegex(ValueError, "RUN_MATRIX_DIFFERS"):
            tasks_for(matrix)
        permission = read_json(self.root / "manifests/ALLOCATION_PERMISSION.json")
        permission["authorized"] = False
        atomic_json(self.root / "manifests/ALLOCATION_PERMISSION.json", permission)
        with self.assertRaisesRegex(ValueError, "ALLOCATION_PERMISSION_REQUIRED"):
            self.scheduler.tick()
        self.assertEqual(self.backend.submissions, [])

    def test_registered_lease_shards_and_source_cannot_change(self):
        self.scheduler.tick()
        with self.assertRaisesRegex(ValueError, "IMMUTABLE_REGISTRATION_CHANGED"):
            self.new_scheduler(shards=2).tick()
        script = self.code / "scripts/mm_dev/run_train_path.py"
        script.write_text("changed source")
        with self.assertRaisesRegex(ValueError, "IMMUTABLE_REGISTRATION_CHANGED"):
            self.scheduler.tick()

    def test_accounting_only_policy_never_uses_gpu_hours_as_a_threshold(self):
        self.scheduler.tick()
        self.finish("ENGINE_F2", elapsed=3600 * 1000)
        state = self.scheduler.tick()
        self.assertEqual(state["phase"], "ENGINE_VERIFIED")
        self.assertEqual(len(self.backend.submissions), 6)
        self.assertEqual(state["accounting"]["known_actual_gpu_seconds"], 3600 * 1000)

    def test_shards_are_fixed_and_every_registered_panel_shard_is_a_dependency(self):
        tasks = tasks_for(run_matrix(), 5)
        self.assertEqual(sum(t["phase"] == "PROBE" for t in tasks.values()), 25)
        self.assertEqual(sum(t["phase"] == "EVAL" for t in tasks.values()), 175)
        self.assertEqual(len(tasks["cont_S0_a0_f0"]["dependencies"]), 29)
        self.assertIn("probe_S_AP_1_shard4of5", tasks["ANALYZE"]["dependencies"])
        self.assertIn("eval_cont_S_AP_1_aC_f1_shard4of5", tasks["RELEASE"]["dependencies"])

    def test_journal_corruption_is_not_silently_reconstructed_from_projection(self):
        self.scheduler.tick()
        journal = self.root / "orchestration/journal.jsonl"
        with journal.open("ab") as stream:
            stream.write(b'{"partial":')
        with self.assertRaisesRegex(ValueError, "JOURNAL_TRUNCATED"):
            self.new_scheduler().tick()
        self.assertEqual(self.backend.submissions, ["ENGINE_F2"])

    def test_accounting_sums_distinct_epochs_and_rejects_wrong_identity(self):
        self.scheduler.tick()
        self.finish("ENGINE_F2", "PREEMPTED", marker=False)
        attempt = self.scheduler.state["tasks"]["ENGINE_F2"]["attempts"][-1]
        permission = self.scheduler.registration["permission"]
        rows = self.backend.observe(attempt, permission)["accounting"]
        extra = copy.deepcopy(rows[0])
        extra.update(
            {
                "Start": "Unknown",
                "End": "2026-10-09T00:00:20",
                "ElapsedRaw": "0",
                "AllocTRES": "",
                "State": "CANCELLED by 1000",
            }
        )
        result = summarize_accounting([*rows, extra], attempt, permission, 1)
        self.assertEqual(result["gpu_seconds"], 10)
        self.assertEqual(result["epochs"], 2)
        self.assertEqual(result["terminal_state"], "CANCELLED")
        with self.assertRaisesRegex(ValueError, "DUPLICATE_ACCOUNTING_EPOCH"):
            summarize_accounting([*rows, *rows], attempt, permission, 1)
        rows[0]["User"] = "other_owner"
        with self.assertRaisesRegex(ValueError, "ACCOUNTING_IDENTITY_MISMATCH"):
            summarize_accounting(rows, attempt, permission, 1)

    def test_no_unrelated_cancellation_or_hidden_resource_defaults(self):
        self.scheduler.tick()
        command = self.scheduler.state["tasks"]["ENGINE_F2"]["attempts"][0]["command"]
        self.assertIn("--hold", command)
        self.assertIn("--no-requeue", command)
        self.assertIn("--nodes=1", command)
        self.assertIn("--signal=B:USR1@600", command)
        self.assertEqual(command[command.index("--time") + 1], "720")
        self.assertIn("--mem=64G", command)
        self.assertIn("--cpus-per-task=4", command)
        self.assertFalse(hasattr(SlurmBackend(), "cancel"))
        self.assertEqual(AUTO_RESUME, {"PREEMPTED", "TIMEOUT"})
        self.assertIn("UNKNOWN", LIVE)

    def test_standalone_cli_help_without_pythonpath(self):
        environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        result = subprocess.run(
            [sys.executable, str(REPOSITORY / "scripts/mm_dev/submit_matrix.py"), "--help"],
            cwd=self.base,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--plan PLAN", result.stdout)
        self.assertIn("--run-root RUN_ROOT", result.stdout)
        self.assertIn("--engine-lease-minutes", result.stdout)
        self.assertEqual(self.backend.submissions, [])

    def test_submission_guard_does_not_reenter_registered_task(self):
        self.scheduler.tick()
        with self.assertRaisesRegex(ValueError, "DUPLICATE_SUBMISSION_GUARD"):
            self.scheduler._submit("ENGINE_F2")
        self.assertEqual(self.backend.submissions, ["ENGINE_F2"])

    def test_full_qos_capacity_defers_without_attempt_then_resumes_after_restart(self):
        # Two controllers plus three unrelated jobs consume all five submitted slots.
        self.backend.external_jobs = 5
        state = self.scheduler.tick()
        self.assertEqual(state["tasks"]["ENGINE_F2"], {"status": "WAITING", "attempts": []})
        self.assertEqual(state["technical_blockers"], [])
        self.assertEqual(state["submission_capacity"]["status"], "WAITING")
        self.assertEqual(state["submission_capacity"]["available_slots"], 0)
        self.assertEqual(self.backend.submissions, [])
        self.assertFalse((self.root / "orchestration/attempts").exists())
        receipt = read_json(self.root / state["submission_capacity"]["observation"]["path"])
        self.assertEqual(len(receipt["observation"]["queued_job_ids"]), 5)
        self.backend.external_jobs = 4
        state = self.new_scheduler().tick()
        self.assertEqual(state["tasks"]["ENGINE_F2"]["status"], "REGISTERED")
        self.assertEqual(self.backend.submissions, ["ENGINE_F2"])

    def test_controller_and_unrelated_jobs_reduce_same_tick_gpu_submissions(self):
        self.scheduler.tick()
        self.finish("ENGINE_F2")
        self.backend.external_jobs = 3
        state = self.scheduler.tick()
        self.assertEqual(len(self.backend.submissions), 3)  # Engine plus two new workers.
        self.assertEqual(state["submission_capacity"]["queued_jobs"], 5)
        self.assertEqual(state["submission_capacity"]["status"], "WAITING")
        self.assertEqual(state["technical_blockers"], [])
        self.assertEqual(state["tasks"]["prep_A_1"]["status"], "WAITING")

    def test_capacity_query_failure_preserves_waiting_and_recovers(self):
        self.backend.capacity_outage = True
        state = self.scheduler.tick()
        self.assertEqual(state["tasks"]["ENGINE_F2"]["status"], "WAITING")
        self.assertEqual(state["submission_capacity"]["status"], "UNKNOWN")
        self.assertEqual(state["technical_blockers"], [])
        self.assertEqual(self.backend.submissions, [])
        self.backend.capacity_outage = False
        self.scheduler.tick()
        self.assertEqual(self.backend.submissions, ["ENGINE_F2"])

    def test_missing_queue_rows_do_not_free_unaccounted_submission_slots(self):
        self.scheduler.tick()
        self.finish("ENGINE_F2")
        self.backend.external_jobs = 3
        self.backend.hide_queued_jobs = True
        state = self.scheduler.tick()
        self.assertEqual(len(self.backend.submissions), 3)
        capacity = state["submission_capacity"]
        self.assertEqual(capacity["queued_jobs"], 3)
        self.assertEqual(len(capacity["reserved_attempt_ids"]), 2)
        self.assertEqual(capacity["available_slots"], 0)

    def test_full_capacity_preserves_retryable_task_until_slot_frees(self):
        self.scheduler.tick()
        self.finish("ENGINE_F2")
        self.backend.external_jobs = 3
        self.scheduler.tick()
        self.finish("prep_A_0", terminal="TIMEOUT", marker=False, checkpoint=True)
        self.backend.external_jobs = 4
        state = self.scheduler.tick()
        self.assertEqual(state["tasks"]["prep_A_0"]["status"], "RETRYABLE")
        self.assertEqual(self.backend.submissions.count("prep_A_0"), 1)
        self.backend.external_jobs = 3
        self.scheduler.tick()
        self.assertEqual(self.backend.submissions.count("prep_A_0"), 2)

    def test_capacity_backend_counts_cpu_dependencies_and_expanded_arrays_across_accounts(self):
        permission = {"owner": "alice", "qos": "teacher"}
        backend = SlurmBackend()
        with patch.object(
            backend,
            "_run",
            side_effect=[
                {"returncode": 0, "stdout": "teacher|5\n", "stderr": ""},
                {
                    "returncode": 0,
                    "stdout": (
                        "1|alice|teacher|RUNNING\n2|alice|teacher|PENDING\n"
                        "3_1|alice|teacher|PENDING\n3_2|alice|teacher|PENDING\n"
                        "4|alice|other|RUNNING\n"
                        "5|alice|teacher|COMPLETED\n6|alice|teacher|FAILED\n"
                        "7|alice|teacher|CANCELLED\n8|alice|teacher|TIMEOUT\n"
                    ),
                    "stderr": "",
                },
            ],
        ) as commands:
            capacity = backend.submission_capacity(permission)
        self.assertEqual(capacity["queued_job_ids"], ["1", "2", "3_1", "3_2"])
        self.assertEqual(capacity["max_submit_jobs_per_user"], 5)
        queue_command = commands.call_args_list[1].args[0]
        self.assertIn("--array", queue_command)
        self.assertNotIn("--account", queue_command)

    def test_capacity_backend_zero_unlimited_and_invalid_limits(self):
        permission = {"owner": "alice", "qos": "teacher"}
        backend = SlurmBackend()
        for value, expected in [("0", 0), ("", None), ("UNLIMITED", None)]:
            with (
                self.subTest(limit=value),
                patch.object(
                    backend,
                    "_run",
                    side_effect=[
                        {"returncode": 0, "stdout": f"teacher|{value}|\n", "stderr": ""},
                        {"returncode": 0, "stdout": "", "stderr": ""},
                    ],
                ),
            ):
                self.assertEqual(
                    backend.submission_capacity(permission)["max_submit_jobs_per_user"], expected
                )
        for output in ["", "other|5\n", "teacher|bad\n", "teacher|5\nteacher|5\n"]:
            with (
                self.subTest(output=output),
                patch.object(
                    backend,
                    "_run",
                    side_effect=[
                        {"returncode": 0, "stdout": output, "stderr": ""},
                        {"returncode": 0, "stdout": "", "stderr": ""},
                    ],
                ),
                self.assertRaises(ValueError),
            ):
                backend.submission_capacity(permission)

    def test_capacity_backend_query_failure_and_malformed_queue_fail_closed(self):
        permission = {"owner": "alice", "qos": "teacher"}
        backend = SlurmBackend()
        for failed in [0, 1]:
            receipts = [
                {"returncode": 0, "stdout": "teacher|5\n", "stderr": ""},
                {"returncode": 0, "stdout": "", "stderr": ""},
            ]
            receipts[failed]["returncode"] = 1
            with (
                self.subTest(failed=failed),
                patch.object(backend, "_run", side_effect=receipts),
                self.assertRaisesRegex(RuntimeError, "SLURM_CAPACITY_QUERY_FAILED"),
            ):
                backend.submission_capacity(permission)
        for output in [
            "garbled",
            "1|bob|teacher|PENDING\n",
            "1|alice|teacher|PENDING\n1|alice|teacher|PENDING\n",
        ]:
            with (
                self.subTest(output=output),
                patch.object(
                    backend,
                    "_run",
                    side_effect=[
                        {"returncode": 0, "stdout": "teacher|5\n", "stderr": ""},
                        {"returncode": 0, "stdout": output, "stderr": ""},
                    ],
                ),
                self.assertRaises(ValueError),
            ):
                backend.submission_capacity(permission)

    def test_backend_parses_real_column_order_blank_comment_and_skips_job_steps(self):
        self.scheduler.tick()
        self.finish("ENGINE_F2")
        attempt = self.scheduler.state["tasks"]["ENGINE_F2"]["attempts"][-1]
        permission = self.scheduler.registration["permission"]
        row = self.backend.observe(attempt, permission)["accounting"][0]
        row["Comment"] = ""  # AccountingStoreFlags need not include job_comment.
        from mm_dev.orchestration import SACCT_FIELDS

        batch = {**row, "JobIDRaw": row["JobIDRaw"] + ".batch", "JobName": "batch"}
        stdout = "\n".join("|".join(item[key] for key in SACCT_FIELDS) for item in [row, batch])
        backend = SlurmBackend()
        with patch.object(
            backend,
            "_run",
            side_effect=[
                {"returncode": 0, "stdout": "", "stderr": ""},
                {"returncode": 0, "stdout": stdout + "\n", "stderr": ""},
            ],
        ) as commands:
            observation = backend.observe(attempt, permission)
        command = commands.call_args_list[1].args[0]
        self.assertRegex(
            command[command.index("--starttime") + 1], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
        )
        self.assertEqual(observation["accounting"], [row])
        cost = summarize_accounting(observation["accounting"], attempt, permission, 1)
        self.assertFalse(cost["comment_recorded_by_sacct"])
        self.assertEqual(cost["gpu_seconds"], 10)

    def test_sacct_lag_and_command_failure_remain_unknown_not_software_failure(self):
        self.scheduler.tick()
        self.finish("ENGINE_F2")
        attempt = self.scheduler.state["tasks"]["ENGINE_F2"]["attempts"][-1]
        permission = self.scheduler.registration["permission"]
        observation = self.backend.observe(attempt, permission)
        observation["accounting"][0]["State"] = "RUNNING"
        with patch.object(self.backend, "observe", return_value=observation):
            state = self.scheduler.tick()
        self.assertEqual(state["tasks"]["ENGINE_F2"]["status"], "UNKNOWN")
        self.assertEqual(self.backend.submissions, ["ENGINE_F2"])
        backend = SlurmBackend()
        with (
            patch.object(
                backend, "_run", return_value={"returncode": 1, "stdout": "", "stderr": "offline"}
            ),
            self.assertRaisesRegex(RuntimeError, "SLURM_OBSERVATION_FAILED"),
        ):
            backend.observe(attempt, permission)


if __name__ == "__main__":
    unittest.main()
