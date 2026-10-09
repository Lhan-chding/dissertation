"""One-shot recovery of the original ENGINE OOM without scheduler side effects."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sr_f1.orchestration import ENGINE_MEMORY_REPAIR_ID, atomic_json, file_hash, read_json

REPO = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = load_module(
    "engine_memory_scheduler_fixtures", Path(__file__).with_name("test_sr_f1_orchestration.py")
)


class EngineMemoryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = FIXTURES.SchedulerTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def prepare_failure(self):
        fixture = self.fixture
        fixture.scheduler.tick()
        fixture.finish("COMMON_START")
        fixture.scheduler.tick()
        attempt = fixture.finish("ENGINE", failure=True, terminal="FAILED")
        fixture.backend.jobs[attempt["job_id"]]["exit_code"] = "1:0"
        fixture.scheduler.tick()
        self.original = copy.deepcopy(fixture.scheduler.state["tasks"]["ENGINE"]["attempts"][0])
        atomic_json(self.root / "ENGINE_MEMORY_REPAIR.json", {"repair_id": ENGINE_MEMORY_REPAIR_ID})
        self.repair = {
            "repair_id": ENGINE_MEMORY_REPAIR_ID,
            "repair_sha256": file_hash(self.root / "ENGINE_MEMORY_REPAIR.json"),
            "source_commit": "a" * 40,
            "original_execution_freeze_sha256": file_hash(self.root / "EXECUTION_FREEZE.json"),
            "gpu_worker_host_memory_gb": 384,
        }
        stub = patch(
            "sr_f1.freeze.verify_engine_memory_repair", return_value=self.repair, create=True
        )
        self.verify_repair = stub.start()
        self.addCleanup(stub.stop)
        return fixture.scheduler

    def test_explicit_resume_preserves_history_and_submits_no_job(self):
        scheduler = self.prepare_failure()
        registry_hash = file_hash(self.root / "orchestration/REGISTRATION.json")
        freeze_hash = file_hash(self.root / "EXECUTION_FREEZE.json")
        journal = self.root / "orchestration/journal.jsonl"
        original_journal = journal.read_bytes()
        original_blocker = scheduler.state["tasks"]["ENGINE"]["blocker"]
        authorization = scheduler.resume_repaired_engine()
        self.assertEqual(scheduler.state["tasks"]["ENGINE"]["status"], "RETRYABLE")
        self.assertEqual(scheduler.state["tasks"]["ENGINE"]["attempts"], [self.original])
        self.assertEqual(scheduler.state["tasks"]["ENGINE"]["blocker"], original_blocker)
        self.assertEqual(len(self.fixture.backend.jobs), 2)
        self.assertEqual(authorization["permitted_next_attempt_id"], "ENGINE_attempt0001")
        self.assertEqual(file_hash(self.root / "orchestration/REGISTRATION.json"), registry_hash)
        self.assertEqual(file_hash(self.root / "EXECUTION_FREEZE.json"), freeze_hash)
        self.assertTrue(journal.read_bytes().startswith(original_journal))
        self.assertTrue((self.root / authorization["failure_marker_path"]).is_file())
        with self.assertRaisesRegex(ValueError, "ALREADY_AUTHORIZED"):
            scheduler.resume_repaired_engine()

    def test_resume_consumed_once_and_unknown_submission_never_repeated(self):
        scheduler = self.prepare_failure()
        scheduler.resume_repaired_engine()
        self.fixture.backend.ambiguous = True
        scheduler.tick()
        self.fixture.backend.outage = True
        restarted = self.fixture.new_scheduler()
        restarted.tick()
        restarted.tick()
        task = restarted.state["tasks"]["ENGINE"]
        self.assertEqual(task["status"], "UNKNOWN")
        self.assertEqual(task["attempts"][0], self.original)
        self.assertEqual(len(task["attempts"]), 2)
        self.assertEqual(len(self.fixture.backend.jobs), 3)
        self.assertEqual(
            restarted.state["engine_memory_repair_resume"]["consumed_by_attempt_id"],
            "ENGINE_attempt0001",
        )
        self.assertIn("--mem=384G", task["attempts"][1]["command"])
        self.assertEqual(task["attempts"][1]["technical_repair_resume"]["source_commit"], "a" * 40)

    def test_live_old_job_prevents_repair(self):
        scheduler = self.prepare_failure()
        self.fixture.backend.jobs[self.original["job_id"]]["state"] = "RUNNING"
        with self.assertRaisesRegex(ValueError, "STILL_ACTIVE"):
            scheduler.resume_repaired_engine()
        self.assertNotIn("engine_memory_repair_resume", scheduler.state)

    def test_changed_terminal_accounting_prevents_repair(self):
        scheduler = self.prepare_failure()
        self.fixture.backend.jobs[self.original["job_id"]]["exit_code"] = "2:0"
        with self.assertRaisesRegex(ValueError, "TERMINAL_ACCOUNTING_CHANGED"):
            scheduler.resume_repaired_engine()

    def test_current_receipt_is_rechecked_before_submission(self):
        scheduler = self.prepare_failure()
        scheduler.resume_repaired_engine()
        path = self.root / "ENGINE_MEMORY_REPAIR.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "ENGINE_MEMORY_REPAIR_IDENTITY_MISMATCH"):
            scheduler.tick()
        self.assertEqual(len(self.fixture.backend.jobs), 2)
        self.assertIsNone(scheduler.state["engine_memory_repair_resume"]["consumed_by_attempt_id"])

    def test_wrong_freeze_or_unverified_repair_prevents_authorization(self):
        scheduler = self.prepare_failure()
        self.repair["original_execution_freeze_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "ENGINE_MEMORY_REPAIR_IDENTITY_MISMATCH"):
            scheduler.resume_repaired_engine()
        self.assertNotIn("engine_memory_repair_resume", scheduler.state)
        self.verify_repair.side_effect = PermissionError("UNAUTHENTICATED_SOURCE_REPAIR")
        with self.assertRaisesRegex(PermissionError, "UNAUTHENTICATED_SOURCE_REPAIR"):
            scheduler.resume_repaired_engine()

    def test_failure_marker_identity_must_match_registration(self):
        scheduler = self.prepare_failure()
        path = self.root / "orchestration/failures/ENGINE_attempt0000.json"
        marker = read_json(path)
        atomic_json(path, {**marker, "registration_hash": "c" * 64})
        with self.assertRaisesRegex(ValueError, "FAILURE_MARKER_CHANGED"):
            scheduler.resume_repaired_engine()

    def test_downstream_work_and_future_attempt_prevent_authorization(self):
        scheduler = self.prepare_failure()
        scheduler.state["tasks"]["BASELINE"]["attempts"].append({"attempt_id": "unregistered"})
        with self.assertRaisesRegex(ValueError, "AFTER_DOWNSTREAM_WORK"):
            scheduler.resume_repaired_engine()
        scheduler.state["tasks"]["BASELINE"]["attempts"].clear()
        atomic_json(self.root / "orchestration/attempts/ENGINE_attempt0001.json", {})
        with self.assertRaisesRegex(ValueError, "FUTURE_ATTEMPT_EXISTS"):
            scheduler.resume_repaired_engine()

    def test_optimizer_progress_or_existing_checkpoint_prevents_restart(self):
        scheduler = self.prepare_failure()
        ledger = self.root / "accounting/ENGINE.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps({"kind": "physical_optimizer_updates", "count": 1}) + "\n")
        with self.assertRaisesRegex(ValueError, "PRIOR_OPTIMIZER_UPDATE"):
            scheduler.resume_repaired_engine()
        ledger.unlink()
        atomic_json(self.root / "orchestration/checkpoints/ENGINE_attempt0000.json", {})
        with self.assertRaisesRegex(ValueError, "EXISTING_COMPLETION_OR_CONTINUATION"):
            scheduler.resume_repaired_engine()

    def test_repaired_gpu_host_memory_is_not_applied_to_cpu_workers(self):
        scheduler = self.prepare_failure()
        self.fixture.engine_verified = True
        self.assertTrue(scheduler._submit("ANALYZE"))
        command = scheduler.state["tasks"]["ANALYZE"]["attempts"][0]["command"]
        self.assertIn("--mem=64G", command)
        self.assertNotIn("--gres", command)
        self.verify_repair.assert_not_called()

    def test_original_gpu_worker_memory_stays_64g_without_receipt(self):
        self.fixture.scheduler.tick()
        command = self.fixture.scheduler.state["tasks"]["COMMON_START"]["attempts"][0]["command"]
        self.assertIn("--mem=64G", command)
        self.assertEqual(command[command.index("--gres") + 1], "gpu:pro6000:1")

    def test_later_science_gpu_workers_use_authenticated_memory(self):
        scheduler = self.prepare_failure()
        self.fixture.engine_verified = True
        run_id = self.fixture.matrix[0]["run_id"]
        self.assertTrue(scheduler._submit(run_id))
        command = scheduler.state["tasks"][run_id]["attempts"][0]["command"]
        self.assertIn("--mem=384G", command)
        self.assertEqual(command[command.index("--gres") + 1], "gpu:pro6000:1")
        self.assertEqual(command[command.index("--qos") + 1], "teacher")

    def test_cli_repair_is_one_shot_and_does_not_tick(self):
        cli = load_module("engine_memory_submit_cli", REPO / "scripts/sr_f1/submit_matrix.py")
        scheduler = Mock()
        scheduler.resume_repaired_engine.return_value = {"status": "RETRYABLE"}
        arguments = [
            "submit_matrix.py",
            "--plan",
            str(self.fixture.plan),
            "--run-root",
            str(self.root),
            "--code-root",
            str(REPO),
            "--python",
            "/usr/bin/python3",
            "--resume-repaired-engine",
        ]
        with patch("sys.argv", arguments), patch.object(cli, "Scheduler", return_value=scheduler):
            self.assertEqual(cli.main(), 0)
        scheduler.resume_repaired_engine.assert_called_once_with()
        scheduler.tick.assert_not_called()
        for extra in ("--watch", "--resume-stopped", "--resume-repaired-common-start"):
            with (
                self.subTest(extra=extra),
                patch("sys.argv", [*arguments, extra]),
                self.assertRaises(SystemExit) as stopped,
            ):
                cli.main()
            self.assertEqual(stopped.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
