"""One-shot zero-update I/O recovery preserves consumed markers and failed jobs."""

import copy
import json
import unittest
from importlib import util
from pathlib import Path
from unittest.mock import Mock, patch

from sr_f1.orchestration import ENGINE_IO_REPAIR_ID, atomic_json, file_hash


def load_module(name, path):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = load_module(
    "io_host_memory_fixtures", Path(__file__).with_name("test_engine_host_memory_orchestration.py")
)
REPO = Path(__file__).resolve().parents[2]


class EngineIORecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture = FIXTURES.EngineHostMemoryRecoveryTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.host_fixture = fixture
        self.root = fixture.root
        self.scheduler = fixture.scheduler
        self.backend = fixture.backend
        self.simulation = fixture.simulation
        fixture.authorize()
        self.scheduler.tick()
        self.failed_id = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]["job_id"]
        self.backend.jobs[self.failed_id].update(state="FAILED", exit_code="120:0")
        # The failed filesystem can prevent fail_task from writing its marker.
        self.scheduler.tick()
        self.assertEqual(self.scheduler.state["tasks"]["ENGINE"]["status"], "BLOCKED")
        self.original_attempts = copy.deepcopy(self.scheduler.state["tasks"]["ENGINE"]["attempts"])
        self.prior_authorizations = {
            key: copy.deepcopy(self.scheduler.state[key])
            for key in ("engine_memory_repair_resume", "engine_storage_repair_resume")
        }
        self.marker_paths = [
            self.root / "technical_incidents/engine_memory_20261010" / name
            for name in ("ZERO_UPDATE_REUSE_ACTIVATED.json", "ZERO_UPDATE_REUSE_PROCESS.json")
        ]
        for path in self.marker_paths:
            atomic_json(path, {"attempt_id": "ENGINE_attempt0002", "consumed": True})
        self.marker_hashes = [file_hash(path) for path in self.marker_paths]
        atomic_json(self.root / "ENGINE_IO_REPAIR.json", {"repair_id": ENGINE_IO_REPAIR_ID})
        self.repair = {
            "repair_id": ENGINE_IO_REPAIR_ID,
            "repair_sha256": file_hash(self.root / "ENGINE_IO_REPAIR.json"),
            "source_commit": "c" * 40,
            "original_execution_freeze_sha256": file_hash(self.root / "EXECUTION_FREEZE.json"),
            "gpu_worker_constraint": "highmem",
            "minimum_gpu_host_memory_gb": 80,
            "failed_attempt_id": "ENGINE_attempt0002",
            "failed_job_id": self.failed_id,
        }
        stub = patch("sr_f1.freeze.verify_engine_io_repair", return_value=self.repair, create=True)
        self.verifier = stub.start()
        self.addCleanup(stub.stop)

    def authorize(self):
        return self.scheduler.resume_repaired_engine_io()

    def test_authorization_preserves_three_attempts_and_consumed_markers(self):
        journal = self.root / "orchestration/journal.jsonl"
        journal_before = journal.read_bytes()
        immutable = {
            path: file_hash(self.root / path)
            for path in ("EXECUTION_FREEZE.json", "orchestration/REGISTRATION.json")
        }
        authorization = self.authorize()
        self.assertEqual(authorization["permitted_next_attempt_id"], "ENGINE_attempt0003")
        self.assertIsNone(authorization["failure_marker_sha256"])
        self.assertFalse((self.root / authorization["failure_marker_path"]).exists())
        self.assertEqual(
            self.scheduler.state["tasks"]["ENGINE"]["attempts"], self.original_attempts
        )
        self.assertEqual(len(self.backend.jobs), 4)
        self.assertTrue(journal.read_bytes().startswith(journal_before))
        for path, checksum in immutable.items():
            self.assertEqual(file_hash(self.root / path), checksum)
        for key, value in self.prior_authorizations.items():
            self.assertEqual(self.scheduler.state[key], value)
        self.assertEqual([file_hash(path) for path in self.marker_paths], self.marker_hashes)
        with self.assertRaisesRegex(ValueError, "ALREADY_AUTHORIZED"):
            self.authorize()

    def test_submit_prefers_new_authorization_and_keeps_host_memory_checks(self):
        self.authorize()
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.assertEqual(attempt["attempt_id"], "ENGINE_attempt0003")
        self.assertTrue(attempt["released"])
        self.assertEqual(attempt["technical_repair_resume"]["repair_id"], ENGINE_IO_REPAIR_ID)
        self.assertIn("--constraint=highmem", attempt["command"])
        self.assertFalse(any(argument.startswith("--mem") for argument in attempt["command"]))
        self.assertEqual(
            self.scheduler.state["engine_io_repair_resume"]["consumed_by_attempt_id"],
            "ENGINE_attempt0003",
        )
        for key, value in self.prior_authorizations.items():
            self.assertEqual(self.scheduler.state[key], value)
        self.assertEqual([file_hash(path) for path in self.marker_paths], self.marker_hashes)
        self.scheduler.tick()
        self.scheduler.tick()
        self.assertEqual(len(self.backend.jobs), 5)

    def test_ambiguous_submission_reserves_attempt_and_never_duplicates(self):
        self.authorize()
        self.backend.ambiguous = True
        self.scheduler.tick()
        self.backend.outage = True
        restarted = self.simulation.new_scheduler()
        restarted.tick()
        restarted.tick()
        self.assertEqual(len(self.backend.jobs), 5)
        self.assertEqual(restarted.state["tasks"]["ENGINE"]["status"], "UNKNOWN")
        self.assertEqual(
            restarted.state["engine_io_repair_resume"]["consumed_by_attempt_id"],
            "ENGINE_attempt0003",
        )

    def test_live_failed_allocation_prevents_authorization(self):
        self.backend.jobs[self.failed_id]["state"] = "RUNNING"
        with self.assertRaisesRegex(ValueError, "STILL_ACTIVE"):
            self.authorize()
        self.assertNotIn("engine_io_repair_resume", self.scheduler.state)

    def test_wrong_terminal_exit_or_unknown_accounting_prevents_authorization(self):
        self.backend.jobs[self.failed_id]["exit_code"] = "1:0"
        with self.assertRaisesRegex(ValueError, "TERMINAL_ACCOUNTING_CHANGED"):
            self.authorize()
        self.backend.jobs[self.failed_id]["exit_code"] = "120:0"
        self.backend.outage = True
        with self.assertRaises(TimeoutError):
            self.authorize()
        self.assertNotIn("engine_io_repair_resume", self.scheduler.state)

    def test_registered_wrong_exit_prevents_authorization(self):
        self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]["accounting"]["exit_code"] = "1:0"
        with self.assertRaisesRegex(ValueError, "PRIOR_TERMINAL_STATES_NOT_AUTHENTICATED"):
            self.authorize()

    def test_science_attempt_or_completed_science_prevents_authorization(self):
        task = self.scheduler.state["tasks"]["BASELINE"]
        task["attempts"].append({})
        with self.assertRaisesRegex(ValueError, "AFTER_DOWNSTREAM_WORK"):
            self.authorize()
        task["attempts"].clear()
        task["status"] = "COMPLETE"
        with self.assertRaisesRegex(ValueError, "AFTER_DOWNSTREAM_WORK"):
            self.authorize()

    def test_changed_consumed_authorization_prevents_restart(self):
        self.scheduler.state["engine_storage_repair_resume"]["consumed_by_attempt_id"] = None
        with self.assertRaisesRegex(ValueError, "PRIOR_AUTHORIZATION_CHANGED"):
            self.authorize()

    def test_unverified_cpu_or_activation_marker_state_prevents_authorization(self):
        self.verifier.side_effect = PermissionError("IO_REPAIR_ACTIVATION_MARKER_CHANGED")
        with self.assertRaisesRegex(PermissionError, "ACTIVATION_MARKER_CHANGED"):
            self.authorize()
        self.assertNotIn("engine_io_repair_resume", self.scheduler.state)

    def test_wrong_freeze_or_failed_job_receipt_prevents_authorization(self):
        self.repair["original_execution_freeze_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "ENGINE_IO_REPAIR_IDENTITY_MISMATCH"):
            self.authorize()
        self.repair["original_execution_freeze_sha256"] = file_hash(
            self.root / "EXECUTION_FREEZE.json"
        )
        self.repair["failed_job_id"] = "999999"
        with self.assertRaisesRegex(ValueError, "FAILED_ALLOCATION_MISMATCH"):
            self.authorize()

    def test_optimizer_update_or_full_state_continuation_prevents_restart(self):
        ledger = self.root / "accounting/ENGINE.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(json.dumps({"kind": "physical_optimizer_updates", "count": 1}) + "\n")
        with self.assertRaisesRegex(ValueError, "PRIOR_OPTIMIZER_UPDATE"):
            self.authorize()
        ledger.unlink()
        self.scheduler.state["tasks"]["ENGINE"]["resume_checkpoint"] = "existing.json"
        with self.assertRaisesRegex(ValueError, "REPAIR_CANNOT_REPLACE_FULL_STATE_CONTINUATION"):
            self.authorize()

    def test_acceptance_marker_or_future_attempt_prevents_authorization(self):
        marker = self.root / "ENGINE_PROBABILITY_GRADIENT_RESUME.json"
        atomic_json(marker, {})
        with self.assertRaisesRegex(ValueError, "FORBIDDEN_COMPLETION_OR_CONTINUATION"):
            self.authorize()
        marker.unlink()
        atomic_json(self.root / "orchestration/attempts/ENGINE_attempt0003.json", {})
        with self.assertRaisesRegex(ValueError, "FUTURE_ATTEMPT_EXISTS"):
            self.authorize()

    def test_receipt_changed_before_submission_never_submits(self):
        self.authorize()
        path = self.root / "ENGINE_IO_REPAIR.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "ENGINE_IO_REPAIR_IDENTITY_MISMATCH"):
            self.scheduler.tick()
        self.assertEqual(len(self.backend.jobs), 4)
        self.assertIsNone(self.scheduler.state["engine_io_repair_resume"]["consumed_by_attempt_id"])

    def test_existing_failure_marker_is_validated_not_overwritten(self):
        path = self.root / "orchestration/failures/ENGINE_attempt0002.json"
        atomic_json(path, {"registration_hash": "wrong"})
        checksum = file_hash(path)
        with self.assertRaisesRegex(ValueError, "FAILURE_MARKER_CHANGED"):
            self.authorize()
        self.assertEqual(file_hash(path), checksum)

    def test_insufficient_actual_memory_remains_held_without_duplicate(self):
        self.authorize()
        self.host_fixture.actual_memory = "33G"
        self.scheduler.tick()
        self.scheduler.tick()
        self.assertEqual(len(self.backend.jobs), 5)
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.assertEqual(self.scheduler.state["tasks"]["ENGINE"]["status"], "UNKNOWN")
        self.assertTrue(self.backend.jobs[attempt["job_id"]]["held"])
        self.assertNotIn("released", attempt)

    def test_cli_io_resume_is_one_shot_and_submits_no_jobs(self):
        cli = load_module("io_repair_submit_cli", REPO / "scripts/sr_f1/submit_matrix.py")
        scheduler = Mock()
        scheduler.resume_repaired_engine_io.return_value = {"status": "RETRYABLE"}
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
            "--resume-repaired-engine-io",
        ]
        with patch("sys.argv", arguments), patch.object(cli, "Scheduler", return_value=scheduler):
            self.assertEqual(cli.main(), 0)
        scheduler.resume_repaired_engine_io.assert_called_once_with()
        scheduler.tick.assert_not_called()
        for extra in ("--watch", "--resume-stopped", "--resume-repaired-engine-storage"):
            with patch("sys.argv", [*arguments, extra]), self.assertRaises(SystemExit):
                cli.main()


if __name__ == "__main__":
    unittest.main()
