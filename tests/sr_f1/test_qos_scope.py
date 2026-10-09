"""Verified QoS capacity amendment; no real scheduler commands are executed."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sr_f1.orchestration import QOS_CAPACITY_MODE, TEACHER_QOS, Scheduler, SlurmBackend


class QosScopeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "QOS_SCOPE_REPAIR.json").write_text("{}\n")
        self.permission = {
            "owner": "owner",
            "qos": TEACHER_QOS,
            "run_root": str(self.root),
        }
        self.repair = {
            "capacity_mode": QOS_CAPACITY_MODE,
            "qos": TEACHER_QOS,
            "repair_sha256": "a" * 64,
        }
        self.verifier = patch(
            "sr_f1.freeze.verify_qos_scope_repair", return_value=self.repair, create=True
        ).start()
        self.addCleanup(patch.stopall)

    def scheduler(self, backend, attempts=()):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.registration = {
            "permission": self.permission,
            "tasks": {"NEXT": {"gpus": 1}, "PRIOR": {"gpus": 1}},
        }
        scheduler.state = {
            "tasks": {"NEXT": {"attempts": []}, "PRIOR": {"attempts": list(attempts)}}
        }
        scheduler.backend = backend
        scheduler.folder = self.root / "orchestration"
        scheduler.sequence = 0
        scheduler._save = Mock()
        return scheduler

    def queue_backend(self, teacher_jobs, *, maximum="5", other_jobs=()):
        backend = SlurmBackend()
        rows = [f"{job}|owner|{TEACHER_QOS}|RUNNING" for job in teacher_jobs]
        rows.extend(f"{job}|owner|rose|PENDING" for job in other_jobs)

        def run(command):
            if command[0] == "sacctmgr":
                self.assertIn("name=" + TEACHER_QOS, command)
                return {"returncode": 0, "stdout": f"{TEACHER_QOS}|{maximum}\n"}
            self.assertEqual(command[0], "squeue")
            self.assertIn("--array", command)
            self.assertIn("--format=%i|%u|%q|%T", command)
            return {"returncode": 0, "stdout": "\n".join(rows) + "\n" if rows else ""}

        backend._run = Mock(side_effect=run)
        return backend

    def test_unrelated_rose_jobs_do_not_block_or_require_gpu_inventory(self):
        # The four other-QoS jobs may each request four GPUs. Their GPU shape is
        # irrelevant to the authorized teacher-QoS submission limit.
        backend = self.queue_backend(["10"], other_jobs=["901", "902", "903", "904"])
        scheduler = self.scheduler(backend)
        self.assertTrue(scheduler._capacity_available("NEXT"))
        self.assertEqual(scheduler.state["capacity"]["submission_slots_used"], 1)
        self.assertEqual(scheduler.state["capacity"]["capacity_mode"], QOS_CAPACITY_MODE)
        self.assertNotIn("project_reserved_gpus", scheduler.state["capacity"])
        self.assertEqual(backend._run.call_count, 2)
        self.verifier.assert_called_once_with(str(self.root))

    def test_teacher_qos_full_native_submit_limit_blocks(self):
        scheduler = self.scheduler(self.queue_backend(["10", "11", "12", "13", "14"]))
        self.assertFalse(scheduler._capacity_available("NEXT"))
        self.assertEqual(scheduler.state["capacity"]["status"], "WAITING")
        self.assertEqual(scheduler.state["capacity"]["submission_slots_used"], 5)

    def test_unknown_submission_reserves_a_native_submit_slot(self):
        scheduler = self.scheduler(
            self.queue_backend(["10", "11", "12", "13"]),
            [{"status": "UNKNOWN", "attempt_id": "PRIOR_attempt0000"}],
        )
        self.assertFalse(scheduler._capacity_available("NEXT"))
        self.assertEqual(scheduler.state["capacity"]["submission_slots_used"], 5)

    def test_visible_reservation_is_not_double_counted(self):
        scheduler = self.scheduler(
            self.queue_backend(["10", "11", "12", "13"]),
            [{"status": "UNKNOWN", "job_id": "10", "attempt_id": "PRIOR_attempt0000"}],
        )
        self.assertTrue(scheduler._capacity_available("NEXT"))
        self.assertEqual(scheduler.state["capacity"]["submission_slots_used"], 4)

    def test_expanded_teacher_array_ids_need_no_scontrol_gpu_inventory(self):
        scheduler = self.scheduler(self.queue_backend(["10_1", "10_2", "10_3"]))
        self.assertTrue(scheduler._capacity_available("NEXT"))
        self.assertEqual(scheduler.state["capacity"]["submission_slots_used"], 3)

    def test_unlimited_native_submit_limit_has_no_manual_five_gpu_guard(self):
        scheduler = self.scheduler(
            self.queue_backend([str(number) for number in range(12)], maximum="UNLIMITED")
        )
        self.assertTrue(scheduler._capacity_available("NEXT"))
        self.assertEqual(scheduler.state["capacity"]["submission_slots_used"], 12)
        self.assertIsNone(scheduler.state["capacity"]["max_submit_jobs_per_user"])

    def test_wrong_permission_qos_is_rejected(self):
        backend = SlurmBackend()
        backend._run = Mock()
        with self.assertRaisesRegex(ValueError, "QOS_SCOPE_REPAIR_PERMISSION_MISMATCH"):
            backend.project_capacity({**self.permission, "qos": "rose"})
        backend._run.assert_not_called()

    def test_invalid_repair_is_rejected_before_capacity_inventory(self):
        self.verifier.side_effect = ValueError("QOS_SCOPE_REPAIR_HASH_MISMATCH")
        backend = SlurmBackend()
        backend._run = Mock()
        with self.assertRaisesRegex(ValueError, "QOS_SCOPE_REPAIR_HASH_MISMATCH"):
            backend.project_capacity(self.permission)
        backend._run.assert_not_called()

    def test_invalid_repair_blocks_scheduler_with_unknown_receipt(self):
        self.verifier.side_effect = ValueError("QOS_SCOPE_REPAIR_HASH_MISMATCH")
        scheduler = self.scheduler(self.queue_backend([]))
        self.assertFalse(scheduler._capacity_available("NEXT"))
        self.assertEqual(scheduler.state["capacity"]["status"], "UNKNOWN")
        self.assertIn("QOS_SCOPE_REPAIR_HASH_MISMATCH", scheduler.state["capacity"]["error"])

    def test_legacy_root_preserves_owner_wide_five_gpu_guard(self):
        (self.root / "QOS_SCOPE_REPAIR.json").unlink()
        backend = SlurmBackend()
        replies = [
            {"returncode": 0, "stdout": f"{TEACHER_QOS}|5\n"},
            {"returncode": 0, "stdout": "901|owner|rose|RUNNING\n"},
            {"returncode": 0, "stdout": "901|owner|RUNNING\n"},
            {
                "returncode": 0,
                "stdout": (
                    "JobId=901 UserId=owner(1) QOS=rose "
                    "ReqTRES=cpu=4,gres/gpu=16 AllocTRES=gres/gpu=16"
                ),
            },
        ]
        backend._run = Mock(side_effect=replies)
        scheduler = self.scheduler(backend)
        self.assertFalse(scheduler._capacity_available("NEXT"))
        self.assertEqual(scheduler.state["capacity"]["project_reserved_gpus"], 16)
        self.verifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
