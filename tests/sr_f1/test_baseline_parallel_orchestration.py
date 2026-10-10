"""Baseline fan-out keeps the frozen task graph and historical ENGINE allocations."""

import copy
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from sr_f1.orchestration import (
    BASELINE_ALLOCATION_PATH,
    BASELINE_PARALLEL_REPAIR_ID,
    MULTIGPU_CAPACITY_MODE,
    TEACHER_QOS,
    atomic_json,
    baseline_parallel_allocation,
    digest,
    file_hash,
    read_json,
    resource_override,
)

REPO = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = load_module(
    "baseline_multigpu_fixture", Path(__file__).with_name("test_engine_multigpu_orchestration.py")
)


class BaselineParallelSchedulerTests(unittest.TestCase):
    def setUp(self):
        fixture = FIXTURES.MultiGPURecoveryTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root, self.scheduler, self.backend = fixture.root, fixture.scheduler, fixture.backend
        self.simulation = fixture.simulation
        fixture.authorize()
        self.scheduler.tick()
        self.engine_attempt = self.scheduler.state["tasks"]["ENGINE"]["attempts"][-1]
        self.engine_override = copy.deepcopy(self.engine_attempt["resource_override"])
        self.registration = copy.deepcopy(self.scheduler.registration)
        self.registry_hash = file_hash(self.root / "orchestration/REGISTRATION.json")
        self.simulation.finish("ENGINE")
        self.scheduler._observe("ENGINE")
        self.assertEqual(self.scheduler.state["tasks"]["ENGINE"]["status"], "COMPLETE")
        self.simulation.engine_verified = True
        atomic_json(
            self.root / "BASELINE_PARALLEL_REPAIR.json",
            {"repair_id": BASELINE_PARALLEL_REPAIR_ID},
        )
        self.repair = {
            "repair_id": BASELINE_PARALLEL_REPAIR_ID,
            "repair_sha256": file_hash(self.root / "BASELINE_PARALLEL_REPAIR.json"),
            "source_commit": "f" * 40,
            "worker_source_sha256": "e" * 64,
            "previous_worker_source_sha256": fixture.repair["worker_source_sha256"],
        }
        stub = patch("sr_f1.freeze.verify_baseline_parallel_repair", return_value=self.repair)
        stub.start()
        self.addCleanup(stub.stop)
        real_hash = file_hash

        def candidate_hash(path):
            if Path(path) == REPO / "scripts/sr_f1/run_worker.py":
                return self.repair["worker_source_sha256"]
            return real_hash(path)

        hashes = patch("sr_f1.orchestration.file_hash", side_effect=candidate_hash)
        hashes.start()
        self.addCleanup(hashes.stop)

    def baseline(self):
        return self.scheduler.state["tasks"]["BASELINE"]

    def test_four_gpus_with_one_external_and_exact_immutable_graph(self):
        self.scheduler.tick()
        attempt = self.baseline()["attempts"][-1]
        allocation = baseline_parallel_allocation(self.root, self.registration)
        self.assertEqual(allocation["gpus"], 4)
        self.assertEqual(attempt["resource_override"]["gpus"], 4)
        self.assertIn("gpu:pro6000:4", attempt["command"])
        self.assertIn("--cpus-per-task=16", attempt["command"])
        self.assertIn("--constraint=highmem", attempt["command"])
        self.assertTrue(attempt["released"])
        self.assertEqual(
            file_hash(self.root / "orchestration/REGISTRATION.json"), self.registry_hash
        )
        self.assertEqual(self.scheduler.registration, self.registration)
        self.assertEqual(self.engine_attempt["resource_override"], self.engine_override)
        self.assertEqual(
            resource_override(self.root, self.registration, "ENGINE"), self.engine_override
        )
        self.assertEqual(self.scheduler.state["capacity"]["teacher_qos_reserved_gpus"], 1)
        events = [read_json_line["event"] for read_json_line in self._journal()]
        self.assertLess(
            events.index("BASELINE_PARALLEL_ALLOCATION_FIXED"),
            len(events) - 1 - events[::-1].index("SUBMISSION_INTENT_DURABLE"),
        )

    def _journal(self):
        import json

        return [
            json.loads(line)
            for line in (self.root / "orchestration/journal.jsonl").read_text().splitlines()
        ]

    def test_five_gpus_if_all_available(self):
        self.fixture.other_teacher_gpus = 0
        self.scheduler.tick()
        self.assertEqual(self.baseline()["attempts"][0]["resource_override"]["gpus"], 5)
        self.assertTrue(self.baseline()["attempts"][0]["released"])

    def test_one_gpu_if_four_are_in_use(self):
        self.fixture.other_teacher_gpus = 4
        self.scheduler.tick()
        self.assertEqual(self.baseline()["attempts"][0]["resource_override"]["gpus"], 1)

    def test_no_free_gpu_or_unknown_capacity_never_reserves_or_submits(self):
        count = len(self.backend.jobs)
        self.fixture.other_teacher_gpus = 5
        self.scheduler.tick()
        self.assertFalse((self.root / BASELINE_ALLOCATION_PATH).exists())
        self.assertEqual(self.baseline()["attempts"], [])
        self.assertEqual(len(self.backend.jobs), count)
        self.fixture.other_teacher_gpus = 0
        self.backend.capacity_outage = True
        self.scheduler.tick()
        self.assertEqual(self.scheduler.state["capacity"]["status"], "UNKNOWN")
        self.assertFalse((self.root / BASELINE_ALLOCATION_PATH).exists())
        self.assertEqual(len(self.backend.jobs), count)

    def test_failed_engine_gate_or_unfinished_engine_cannot_choose_allocation(self):
        self.simulation.engine_verified = False
        with self.assertRaisesRegex(ValueError, "ENGINE_NOT_VERIFIED"):
            self.scheduler._submit("BASELINE")
        self.assertFalse((self.root / BASELINE_ALLOCATION_PATH).exists())
        self.simulation.engine_verified = True
        self.scheduler.state["tasks"]["ENGINE"]["status"] = "ACTIVE"
        with self.assertRaisesRegex(ValueError, "BASELINE_ENGINE_DEPENDENCY_NOT_COMPLETE"):
            self.scheduler._submit("BASELINE")
        self.assertFalse((self.root / BASELINE_ALLOCATION_PATH).exists())

    def test_parallel_baseline_rejects_unverified_capacity_modes(self):
        original = self.backend.project_capacity

        def unverified(permission):
            result = original(permission)
            result.pop("capacity_mode")
            return result

        self.backend.project_capacity = unverified
        self.scheduler.tick()
        self.assertEqual(self.scheduler.state["capacity"]["status"], "UNKNOWN")
        self.assertFalse((self.root / BASELINE_ALLOCATION_PATH).exists())

    def test_existing_unknown_submission_keeps_fixed_count_and_no_duplicate(self):
        self.backend.ambiguous = True
        self.scheduler.tick()
        self.assertEqual(self.baseline()["status"], "UNKNOWN")
        attempt = self.baseline()["attempts"][0]
        self.assertEqual(attempt["resource_override"]["gpus"], 4)
        self.backend.jobs.pop(str(1000 + len(self.backend.jobs) - 1))
        self.backend.outage = True
        self.fixture.other_teacher_gpus = 0
        checksum = file_hash(self.root / BASELINE_ALLOCATION_PATH)
        self.scheduler.tick()
        self.scheduler.tick()
        self.assertEqual(len(self.baseline()["attempts"]), 1)
        self.assertEqual(file_hash(self.root / BASELINE_ALLOCATION_PATH), checksum)
        self.assertFalse(self.scheduler._capacity_available("SRF1_A_s71001"))
        self.assertEqual(self.scheduler.state["capacity"]["teacher_qos_reserved_gpus"], 4)

    def test_valid_checkpoint_resume_keeps_four_when_five_become_available(self):
        self.scheduler.tick()
        checksum = file_hash(self.root / BASELINE_ALLOCATION_PATH)
        first = self.simulation.finish("BASELINE", checkpoint=True, terminal="PREEMPTED")
        self.fixture.other_teacher_gpus = 0
        self.scheduler.tick()
        second = self.baseline()["attempts"][-1]
        self.assertNotEqual(second["attempt_id"], first["attempt_id"])
        self.assertEqual(second["resource_override"]["gpus"], 4)
        self.assertEqual(file_hash(self.root / BASELINE_ALLOCATION_PATH), checksum)

    def test_other_job_arriving_before_release_leaves_same_attempt_held(self):
        original = self.backend.project_capacity
        calls = 0

        def capacity(permission):
            nonlocal calls
            calls += 1
            self.fixture.other_teacher_gpus = 1 if calls == 1 else 2
            return original(permission)

        self.backend.project_capacity = capacity
        self.scheduler.tick()
        attempt = self.baseline()["attempts"][0]
        self.assertNotIn("released", attempt)
        self.assertTrue(attempt["release_waiting_for_teacher_gpu_capacity"])
        self.assertEqual(attempt["resource_override"]["gpus"], 4)
        self.backend.project_capacity = original
        self.fixture.other_teacher_gpus = 1
        self.scheduler.tick()
        self.assertTrue(attempt["released"])
        self.assertEqual(len(self.baseline()["attempts"]), 1)

    def test_allocation_tamper_and_missing_receipt_block_historical_projection(self):
        self.scheduler.tick()
        path = self.root / BASELINE_ALLOCATION_PATH
        saved = read_json(path)
        for update in ({"gpus": True}, {"gpus": 0}, {"gpus": 6}, {"gpus": 3}, {"extra": 1}):
            with self.subTest(update=update):
                atomic_json(path, {**saved, **update})
                with self.assertRaises(ValueError):
                    self.scheduler._projections()
        atomic_json(path, saved)
        self.scheduler._projections()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "ATTEMPT_RESOURCES_CHANGED"):
            self.scheduler._projections()

    def test_latest_worker_is_authenticated_but_old_engine_override_remains_exact(self):
        self.scheduler._load()
        self.assertEqual(self.scheduler.registration, self.registration)
        self.assertNotEqual(
            self.repair["worker_source_sha256"], self.engine_override["worker_source_sha256"]
        )
        self.assertEqual(
            resource_override(self.root, self.registration, "ENGINE"), self.engine_override
        )
        self.scheduler.tick()
        attempt = self.baseline()["attempts"][0]
        worker = load_module("baseline_parallel_test_worker", REPO / "scripts/sr_f1/run_worker.py")
        self.fixture.resource_edits["JobState"] = "RUNNING"
        with patch.dict(
            os.environ,
            {
                "SR_F1_TASK_ID": "BASELINE",
                "SR_F1_ATTEMPT_ID": attempt["attempt_id"],
                "SR_F1_REGISTRATION_HASH": digest(self.registration),
                "SLURM_JOB_ID": attempt["job_id"],
            },
        ):
            result = worker.validate_allocation(self.root, "BASELINE", backend=self.backend)
        self.assertEqual(result["operation"], "evaluate")
        self.assertEqual(result["stage"], "baseline")

    def test_cpu_tasks_are_not_blocked_by_zero_gpu_request(self):
        self.assertTrue(self.scheduler._capacity_available("ANALYZE"))
        self.assertEqual(self.scheduler.state["capacity"]["requested_gpus"], 0)
        self.assertEqual(self.scheduler.state["capacity"]["capacity_mode"], MULTIGPU_CAPACITY_MODE)
        self.assertEqual(self.scheduler.state["capacity"]["qos"], TEACHER_QOS)


if __name__ == "__main__":
    unittest.main()
