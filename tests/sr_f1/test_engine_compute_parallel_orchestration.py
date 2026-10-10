"""New compute allocations preserve historical costs and use a fresh one-use gate."""

import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from sr_f1.orchestration import (
    ENGINE_COMPUTE_PARALLEL_REPAIR_ID,
    atomic_json,
    attempt_gpu_count,
    baseline_parallel_repair,
    file_hash,
    resource_override,
)

spec = importlib.util.spec_from_file_location(
    "compute_multigpu_fixture", Path(__file__).with_name("test_engine_multigpu_orchestration.py")
)
FIXTURES = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FIXTURES)


class ComputeRecoveryTests(unittest.TestCase):
    def setUp(self):
        f = FIXTURES.MultiGPURecoveryTests(methodName="runTest")
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        self.root, self.scheduler, self.backend = f.root, f.scheduler, f.backend
        f.authorize()
        self.scheduler.tick()
        self.maintained = copy.deepcopy(self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1])
        self.backend.jobs[self.maintained["job_id"]]["state"] = "CANCELLED"
        self.scheduler.tick()
        self.prior = copy.deepcopy(self.scheduler.state["tasks"]["ENGINE"]["attempts"])
        self.old_authorization = copy.deepcopy(
            self.scheduler.state["engine_multigpu_repair_resume"]
        )
        atomic_json(
            self.root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json",
            {"repair_id": ENGINE_COMPUTE_PARALLEL_REPAIR_ID},
        )
        self.repair = {
            **f.repair,
            "repair_id": ENGINE_COMPUTE_PARALLEL_REPAIR_ID,
            "repair_sha256": file_hash(self.root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json"),
            "source_commit": "e" * 40,
            "previous_worker_source_sha256": f.repair["worker_source_sha256"],
            "maintained_attempt_id": self.maintained["attempt_id"],
            "maintained_job_id": self.maintained["job_id"],
            "engine_multigpu_repair": f.repair,
        }
        stub = patch("sr_f1.freeze.verify_engine_compute_parallel_repair", return_value=self.repair)
        stub.start()
        self.addCleanup(stub.stop)

    def test_history_resources_survive_changed_current_compute_count(self):
        self.repair["gpu_count"] = 5
        count = attempt_gpu_count(self.root, self.scheduler.registration, "ENGINE", self.prior[-1])
        self.assertEqual(count, 4)
        self.assertEqual(
            resource_override(self.root, self.scheduler.registration, "ENGINE")["gpus"], 5
        )
        self.scheduler._load()
        self.assertEqual(self.scheduler.state["tasks"]["ENGINE"]["attempts"], self.prior)

    def test_old_resource_mutation_rejected(self):
        bad = copy.deepcopy(self.prior[-1])
        bad["resource_override"]["gpus"] = 5
        with self.assertRaisesRegex(Exception, "ATTEMPT_RESOURCES_CHANGED"):
            attempt_gpu_count(self.root, self.scheduler.registration, "ENGINE", bad)

    def test_compute_resume_does_not_reconsume_old_token_or_discard_updates(self):
        ledger = self.root / "accounting/ENGINE.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text('{"kind":"physical_optimizer_updates","count":2}\n')
        checksum = file_hash(ledger)
        result = self.scheduler.resume_repaired_engine_compute_parallel()
        self.assertEqual(result["permitted_next_attempt_id"], "ENGINE_attempt0005")
        self.assertEqual(
            self.scheduler.state["engine_multigpu_repair_resume"], self.old_authorization
        )
        self.assertEqual(self.scheduler.state["tasks"]["ENGINE"]["attempts"], self.prior)
        self.assertEqual(file_hash(ledger), checksum)
        self.scheduler.tick()
        attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.assertEqual(
            attempt["resource_override"]["repair_id"], ENGINE_COMPUTE_PARALLEL_REPAIR_ID
        )
        self.assertEqual(
            self.scheduler.state["engine_compute_parallel_repair_resume"]["consumed_by_attempt_id"],
            "ENGINE_attempt0005",
        )
        with self.assertRaisesRegex(Exception, "ALREADY_AUTHORIZED"):
            self.scheduler.resume_repaired_engine_compute_parallel()

    def test_active_previous_job_blocks_new_authorization(self):
        self.backend.jobs[self.maintained["job_id"]]["state"] = "RUNNING"
        with self.assertRaisesRegex(Exception, "STILL_ACTIVE"):
            self.scheduler.resume_repaired_engine_compute_parallel()

    def test_stop_blocks_new_authorization(self):
        (self.root / "STOP").write_text("stop")
        with self.assertRaisesRegex(Exception, "NOT_COMPUTE_REPAIR_ELIGIBLE"):
            self.scheduler.resume_repaired_engine_compute_parallel()

    def test_new_baseline_capability_uses_compute_receipt_and_original_gate(self):
        capability = baseline_parallel_repair(self.root)
        self.assertEqual(capability["certification"], "ENGINE_COMPUTE_PARALLEL_REPAIR")
        self.assertEqual(capability["repair_sha256"], self.repair["repair_sha256"])
        self.scheduler.resume_repaired_engine_compute_parallel()
        self.scheduler.tick()
        self.assertEqual(self.scheduler.state["tasks"]["BASELINE"]["attempts"], [])
        self.assertEqual(self.scheduler.state["tasks"]["BASELINE"]["status"], "WAITING")

    def test_capacity_does_not_cancel_other_teacher_job(self):
        self.f.other_teacher_gpus = 2
        self.scheduler.resume_repaired_engine_compute_parallel()
        count = len(self.backend.jobs)
        self.scheduler.tick()
        self.assertEqual(len(self.backend.jobs), count)
        self.assertEqual(self.scheduler.state["capacity"]["teacher_qos_reserved_gpus"], 2)

    def test_unknown_submission_reserved_not_retried(self):
        self.scheduler.resume_repaired_engine_compute_parallel()
        self.backend.ambiguous = True
        self.scheduler.tick()
        count = len(self.backend.jobs)
        self.scheduler.tick()
        self.assertEqual(len(self.backend.jobs), count)

    def test_compute_engine_pass_automatically_selects_parallel_baseline(self):
        self.scheduler.resume_repaired_engine_compute_parallel()
        self.scheduler.tick()
        self.f.simulation.finish("ENGINE")
        self.scheduler._observe("ENGINE")
        self.f.simulation.engine_verified = True
        self.scheduler.tick()
        baseline = self.scheduler.state["tasks"]["BASELINE"]["attempts"][-1]
        self.assertEqual(baseline["resource_override"]["gpus"], 4)
        self.assertEqual(
            baseline["resource_override"]["repair_sha256"], self.repair["repair_sha256"]
        )
        self.assertTrue(baseline["released"])
        self.assertFalse((self.root / "BASELINE_PARALLEL_REPAIR.json").exists())
        self.scheduler._load()
        old_count = attempt_gpu_count(
            self.root, self.scheduler.registration, "ENGINE", self.prior[-1]
        )
        self.assertEqual(old_count, 4)
