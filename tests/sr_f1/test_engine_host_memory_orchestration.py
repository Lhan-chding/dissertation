"""Authenticated recovery from an unstarted allocation with insufficient host RAM."""

import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sr_f1.orchestration import (
    ENGINE_STORAGE_REPAIR_ID,
    TEACHER_QOS,
    atomic_json,
    file_hash,
    read_json,
)

REPO = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = load_module(
    "host_memory_recovery_fixtures", Path(__file__).with_name("test_engine_memory_orchestration.py")
)


class EngineHostMemoryRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture = FIXTURES.EngineMemoryRecoveryTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        permission_path = self.root / "manifests/ALLOCATION_PERMISSION.json"
        permission = read_json(permission_path)
        atomic_json(permission_path, {**permission, "qos": TEACHER_QOS})
        self.scheduler = fixture.prepare_failure()
        self.simulation = fixture.fixture
        self.backend = self.simulation.backend
        self.scheduler.resume_repaired_engine()
        self.scheduler.tick()
        self.cancelled_id = self.scheduler.state["tasks"]["ENGINE"]["attempts"][1]["job_id"]
        self.backend.jobs[self.cancelled_id]["state"] = "CANCELLED"
        original_observe = self.backend.observe

        def observe(attempt, permission):
            evidence = original_observe(attempt, permission)
            for row in evidence["accounting"]:
                if row["JobIDRaw"] == self.cancelled_id:
                    row.update(ElapsedRaw="0", End=row["Start"], AllocTRES="")
            return evidence

        self.backend.observe = observe
        self.scheduler.tick()
        self.original_attempts = copy.deepcopy(self.scheduler.state["tasks"]["ENGINE"]["attempts"])
        self.memory_authorization = copy.deepcopy(
            self.scheduler.state["engine_memory_repair_resume"]
        )
        atomic_json(
            self.root / "ENGINE_STORAGE_REPAIR.json", {"repair_id": ENGINE_STORAGE_REPAIR_ID}
        )
        self.repair = {
            "repair_id": ENGINE_STORAGE_REPAIR_ID,
            "repair_sha256": file_hash(self.root / "ENGINE_STORAGE_REPAIR.json"),
            "source_commit": "b" * 40,
            "original_execution_freeze_sha256": file_hash(self.root / "EXECUTION_FREEZE.json"),
            "gpu_worker_constraint": "highmem",
            "minimum_gpu_host_memory_gb": 80,
        }
        stub = patch(
            "sr_f1.freeze.verify_engine_storage_repair", return_value=self.repair, create=True
        )
        self.verifier = stub.start()
        self.addCleanup(stub.stop)
        self.actual_memory = "90G"
        self.actual_features = "highmem"
        self.actual_qos = TEACHER_QOS
        self.actual_gpu_tres = "gres/gpu=1,gres/gpu:pro6000=1"

        def detail(command):
            self.assertEqual(command[:3], ["scontrol", "show", "job"])
            job_id = command[3]
            job = self.backend.jobs[job_id]
            return {
                "returncode": 0,
                "stdout": (
                    f"JobId={job_id} JobName={job['name']} Comment={job['comment']} "
                    f"UserId=owner(1) Account=rose QOS={self.actual_qos} "
                    f"JobState=PENDING Reason=JobHeldUser Features={self.actual_features} "
                    f"ReqTRES=cpu=4,mem={self.actual_memory},{self.actual_gpu_tres}"
                ),
            }

        self.backend._run = Mock(side_effect=detail)

    def authorize(self):
        return self.scheduler.resume_repaired_engine_storage()

    def test_resume_preserves_two_attempts_and_consumed_first_authorization(self):
        journal = self.root / "orchestration/journal.jsonl"
        before = journal.read_bytes()
        frozen = file_hash(self.root / "EXECUTION_FREEZE.json")
        registry = file_hash(self.root / "orchestration/REGISTRATION.json")
        authorization = self.authorize()
        self.assertEqual(authorization["permitted_next_attempt_id"], "ENGINE_attempt0002")
        self.assertEqual(
            self.scheduler.state["tasks"]["ENGINE"]["attempts"], self.original_attempts
        )
        self.assertEqual(
            self.scheduler.state["engine_memory_repair_resume"], self.memory_authorization
        )
        self.assertEqual(len(self.backend.jobs), 3)
        self.assertTrue(journal.read_bytes().startswith(before))
        self.assertEqual(file_hash(self.root / "EXECUTION_FREEZE.json"), frozen)
        self.assertEqual(file_hash(self.root / "orchestration/REGISTRATION.json"), registry)
        with self.assertRaisesRegex(ValueError, "ALREADY_AUTHORIZED"):
            self.authorize()

    def test_attempt_two_has_highmem_no_requested_mem_and_verified_release(self):
        self.authorize()
        self.scheduler.tick()
        task = self.scheduler.state["tasks"]["ENGINE"]
        attempt = task["attempts"][-1]
        self.assertEqual(attempt["attempt_id"], "ENGINE_attempt0002")
        self.assertTrue(attempt["released"])
        self.assertIn("--constraint=highmem", attempt["command"])
        self.assertFalse(any(argument.startswith("--mem") for argument in attempt["command"]))
        self.assertEqual(attempt["command"][attempt["command"].index("--qos") + 1], TEACHER_QOS)
        self.assertTrue((self.root / attempt["host_memory_observation"]["path"]).is_file())
        self.assertEqual(
            self.scheduler.state["engine_storage_repair_resume"]["consumed_by_attempt_id"],
            "ENGINE_attempt0002",
        )
        self.assertEqual(
            self.scheduler.state["engine_memory_repair_resume"], self.memory_authorization
        )

    def test_33g_override_leaves_job_held_and_unknown(self):
        self.actual_memory = "33G"
        self.authorize()
        self.scheduler.tick()
        task = self.scheduler.state["tasks"]["ENGINE"]
        attempt = task["attempts"][-1]
        self.assertEqual(task["status"], "UNKNOWN")
        self.assertTrue(self.backend.jobs[attempt["job_id"]]["held"])
        self.assertNotIn("released", attempt)
        job_count = len(self.backend.jobs)
        self.scheduler.tick()
        self.assertEqual(len(self.backend.jobs), job_count)
        self.assertEqual(task["status"], "UNKNOWN")
        self.assertTrue(self.backend.jobs[attempt["job_id"]]["held"])
        rows = [
            read_json(path)
            for path in (self.root / "orchestration/observations/host_memory").rglob("*.json")
        ]
        self.assertEqual(len(rows), 2)

    def test_mib_memory_and_generic_verified_minimum(self):
        self.actual_memory = "81920M"
        self.authorize()
        self.scheduler.tick()
        self.assertTrue(self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]["released"])

    def test_reheld_previously_released_job_is_reverified(self):
        self.authorize()
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.assertTrue(attempt["released"])
        self.backend.jobs[attempt["job_id"]].update(state="PENDING", held=True)
        self.actual_memory = "33G"
        self.scheduler.tick()
        self.assertEqual(self.scheduler.state["tasks"]["ENGINE"]["status"], "UNKNOWN")
        self.assertTrue(self.backend.jobs[attempt["job_id"]]["held"])

    def test_unavailable_scheduler_details_never_release_the_job(self):
        self.authorize()
        self.backend._run.side_effect = None
        self.backend._run.return_value = {"returncode": 1, "stdout": "", "stderr": "unavailable"}
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.assertEqual(self.scheduler.state["tasks"]["ENGINE"]["status"], "UNKNOWN")
        self.assertNotIn("released", attempt)
        self.assertTrue(self.backend.jobs[attempt["job_id"]]["held"])
        self.assertEqual(
            read_json(self.root / attempt["host_memory_observation"]["path"])["returncode"], 1
        )

    def test_missing_highmem_wrong_qos_or_extra_gpu_prevents_release(self):
        self.authorize()
        self.actual_features = "(null)"
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.assertNotIn("released", attempt)
        self.actual_features = "highmem"
        self.actual_qos = "rose"
        self.scheduler.tick()
        self.assertNotIn("released", attempt)
        self.actual_qos = TEACHER_QOS
        self.actual_gpu_tres = "gres/gpu=2,gres/gpu:pro6000=2"
        self.scheduler.tick()
        self.assertNotIn("released", attempt)
        self.assertTrue(self.backend.jobs[attempt["job_id"]]["held"])

    def test_unknown_submission_is_consumed_and_never_resubmitted(self):
        self.authorize()
        self.backend.ambiguous = True
        self.scheduler.tick()
        self.backend.outage = True
        restarted = self.simulation.new_scheduler()
        restarted.tick()
        restarted.tick()
        self.assertEqual(len(self.backend.jobs), 4)
        self.assertEqual(restarted.state["tasks"]["ENGINE"]["status"], "UNKNOWN")
        self.assertEqual(
            restarted.state["engine_storage_repair_resume"]["consumed_by_attempt_id"],
            "ENGINE_attempt0002",
        )

    def test_unverified_receipt_or_changed_freeze_prevents_authorization(self):
        self.repair["original_execution_freeze_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "ENGINE_STORAGE_REPAIR_IDENTITY_MISMATCH"):
            self.authorize()
        self.verifier.side_effect = PermissionError("PROCESS_ALREADY_ACTIVATED")
        with self.assertRaisesRegex(PermissionError, "PROCESS_ALREADY_ACTIVATED"):
            self.authorize()
        self.assertNotIn("engine_storage_repair_resume", self.scheduler.state)

    def test_still_queued_cancelled_attempt_prevents_authorization(self):
        self.backend.jobs[self.cancelled_id]["state"] = "PENDING"
        with self.assertRaisesRegex(ValueError, "STILL_ACTIVE"):
            self.authorize()

    def test_nonzero_cancelled_gpu_time_or_downstream_attempt_prevents_repair(self):
        task = self.scheduler.state["tasks"]["ENGINE"]
        task["attempts"][1]["accounting"]["gpu_seconds"] = 1
        with self.assertRaisesRegex(ValueError, "TERMINAL_STATES_NOT_AUTHENTICATED"):
            self.authorize()
        task["attempts"][1]["accounting"]["gpu_seconds"] = 0
        self.scheduler.state["tasks"]["BASELINE"]["attempts"].append({})
        with self.assertRaisesRegex(ValueError, "AFTER_DOWNSTREAM_WORK"):
            self.authorize()

    def test_future_attempt_artifact_prevents_repair(self):
        atomic_json(self.root / "orchestration/attempts/ENGINE_attempt0002.json", {})
        with self.assertRaisesRegex(ValueError, "FUTURE_ATTEMPT_EXISTS"):
            self.authorize()

    def test_cpu_worker_keeps64g_without_highmem_or_resource_query(self):
        self.simulation.engine_verified = True
        self.assertTrue(self.scheduler._submit("ANALYZE"))
        attempt = self.scheduler.state["tasks"]["ANALYZE"]["attempts"][0]
        self.assertIn("--mem=64G", attempt["command"])
        self.assertNotIn("--constraint=highmem", attempt["command"])
        self.backend._run.assert_not_called()

    def test_cli_storage_repair_is_one_shot(self):
        cli = load_module("host_memory_submit_cli", REPO / "scripts/sr_f1/submit_matrix.py")
        scheduler = Mock()
        scheduler.resume_repaired_engine_storage.return_value = {"status": "RETRYABLE"}
        arguments = [
            "submit_matrix.py",
            "--plan",
            str(self.simulation.plan),
            "--run-root",
            str(self.root),
            "--code-root",
            str(REPO),
            "--python",
            "/usr/bin/python3",
            "--resume-repaired-engine-storage",
        ]
        with patch("sys.argv", arguments), patch.object(cli, "Scheduler", return_value=scheduler):
            self.assertEqual(cli.main(), 0)
        scheduler.resume_repaired_engine_storage.assert_called_once_with()
        scheduler.tick.assert_not_called()
        with (
            patch("sys.argv", [*arguments, "--resume-repaired-engine"]),
            self.assertRaises(SystemExit),
        ):
            cli.main()


if __name__ == "__main__":
    unittest.main()
