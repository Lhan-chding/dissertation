"""Tiny local operator fixtures; no SSH, scheduler or scientific execution.

Run: python -B test_recover_candidate06_phase.py
The helper is loaded beside this file so both files can be archived together.
"""

from __future__ import annotations

import errno
import importlib.util
import json
import os
import stat
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

HELPER = Path(__file__).with_name("recover_candidate06_phase.py")
SPEC = importlib.util.spec_from_file_location("recovery_helper_fixture", HELPER)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


class RecoveryAttempt03Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "unit"
        self.source.mkdir()
        (self.source / "data.npz").write_bytes(b"original finite bytes")
        (self.source / "part.pending-55").write_bytes(b"interrupted")
        (self.source / "zero").write_bytes(b"")
        (self.source / "nested").mkdir()
        (self.source / "nested" / "COMPLETE.json").write_bytes(b'{"status":"COMPLETE"}')
        self.inventory = recovery.tree_entries(self.source, hash_files=True)
        self.parent = self.root / "INTERRUPTED_ATTEMPT03"
        self.parent.mkdir(mode=0o700)
        self.destination = self.parent / self.source.name
        self.owner = {"fixture": True, "attempt": "03"}

    def move(self):
        return recovery.private_ceph_rename(
            self.source, self.destination, self.inventory, self.owner
        )

    @contextmanager
    def reported_parent_mode(self, mode):
        """Model Ceph's observed setgid stat; local macOS strips that fixture bit."""
        self.parent.chmod(mode & 0o777)
        parent = self.parent.stat()
        original = os.fstat

        def directory_mode(fd):
            info = original(fd)
            if (info.st_dev, info.st_ino) == (parent.st_dev, parent.st_ino):
                fields = list(info)
                fields[0] = stat.S_IFMT(info.st_mode) | mode
                return os.stat_result(fields)
            return info

        with patch.object(recovery.os, "fstat", side_effect=directory_mode):
            yield

    def previous_failure(self):
        phase, spec = "Q1", recovery.PHASES["Q1"]
        previous = self.root / "RECOVERY_Q1_01"
        previous.mkdir()
        recovery.write_new(
            previous / "INTENT.json",
            {
                "slurm_job_id": spec["previous_recovery_job"],
                "helper_sha256": recovery.PREVIOUS_HELPER_SHA256,
                "snapshot_sha256": recovery.SNAPSHOT_SHA256,
                "phase": phase,
            },
        )
        recovery.write_new(
            previous / "FAILED.json",
            {
                "exception_type": "OSError",
                "status": "STOPPED",
                "message": "[Errno 22] Invalid argument: renameat2 fixture",
            },
        )
        recovery.write_new(previous / "PARTIAL_INVENTORY.json", {"entries": self.inventory})
        recovery.write_new(previous / "VERIFIED_BEFORE_MOVE.json", {"fixture": True})
        old_backup = self.root / ("INTERRUPTED_ORIGINALS_" + spec["job"])
        old_backup.mkdir()
        first = recovery.verify_attempt01_failure(self.root, phase, spec)
        previous = self.root / "RECOVERY_Q1_02"
        previous.mkdir()
        recovery.write_new(
            previous / "INTENT.json",
            {
                "slurm_job_id": spec["attempt02_recovery_job"],
                "helper_sha256": recovery.ATTEMPT02_HELPER_SHA256,
                "snapshot_sha256": recovery.SNAPSHOT_SHA256,
                "phase": phase,
                "recovery_attempt": "02",
            },
        )
        recovery.write_new(
            previous / "FAILED.json",
            {
                "exception_type": "ValueError",
                "status": "STOPPED",
                "message": "backup parent must be owned by this operator with mode 0700",
            },
        )
        recovery.write_new(previous / "PARTIAL_INVENTORY.json", {"entries": self.inventory})
        recovery.write_new(previous / "VERIFIED_BEFORE_MOVE.json", {"fixture": True})
        recovery.write_new(previous / "PREVIOUS_ATTEMPT_PRESERVED.json", first)
        old_backup = self.root / ("INTERRUPTED_ORIGINALS_" + spec["job"] + "_ATTEMPT02")
        old_backup.mkdir()
        return previous, old_backup

    def test_explicit_attempt03_plain_rename_preserves_every_byte(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "not_read").write_bytes(b"outside")
        (self.source / "link").symlink_to(outside, target_is_directory=True)
        self.inventory = recovery.tree_entries(self.source, hash_files=True)
        self.assertNotIn("link/not_read", self.inventory)
        original_rename = recovery.os.rename
        with patch.object(recovery.os, "rename", wraps=original_rename) as operation:
            receipt = self.move()
        self.assertEqual(operation.call_count, 1)
        self.assertFalse(self.source.exists())
        self.assertEqual(recovery.tree_entries(self.destination, hash_files=True), self.inventory)
        self.assertEqual(receipt["rename_calls"], 1)
        self.assertFalse(receipt["automatic_retry"])
        self.assertEqual(receipt["mechanism"], "EXPLICIT_ATTEMPT03_CEPH_PLAIN_RENAME")
        self.assertEqual(
            recovery.sha256_file(self.parent / "OWNER_LOCK.json"), receipt["owner_lock_sha256"]
        )

    def test_move_failure_has_no_retry_and_keeps_source_and_owner(self):
        with patch.object(
            recovery.os, "rename", side_effect=OSError(errno.EINVAL, "Invalid argument")
        ) as operation:
            with self.assertRaises(OSError):
                self.move()
            self.assertEqual(operation.call_count, 1)
            with self.assertRaises(FileExistsError):
                self.move()
            self.assertEqual(operation.call_count, 1)
        self.assertEqual(recovery.tree_entries(self.source, hash_files=True), self.inventory)
        self.assertFalse(self.destination.exists())
        self.assertTrue((self.parent / "OWNER_LOCK.json").is_file())

    def test_existing_empty_destination_is_never_replaced(self):
        self.destination.mkdir()
        inode = self.destination.stat().st_ino
        with patch.object(recovery.os, "rename") as operation:
            with self.assertRaises(FileExistsError):
                self.move()
            operation.assert_not_called()
        self.assertEqual(self.destination.stat().st_ino, inode)
        self.assertTrue(self.source.is_dir())

    def test_dangling_destination_link_is_never_replaced(self):
        self.destination.symlink_to(self.root / "missing")
        with patch.object(recovery.os, "rename") as operation:
            with self.assertRaises(FileExistsError):
                self.move()
            operation.assert_not_called()
        self.assertTrue(self.destination.is_symlink())

    def test_existing_owner_cannot_be_claimed(self):
        path = self.parent / "OWNER_LOCK.json"
        path.write_bytes(b"another owner")
        with patch.object(recovery.os, "rename") as operation:
            with self.assertRaises(FileExistsError):
                self.move()
            operation.assert_not_called()
        self.assertEqual(path.read_bytes(), b"another owner")

    def test_world_accessible_parent_is_rejected(self):
        self.parent.chmod(0o755)
        with patch.object(recovery.os, "rename") as operation:
            with self.assertRaisesRegex(ValueError, "mode 0700"):
                self.move()
            operation.assert_not_called()

    def test_setgid_private_parent_is_preserved_and_can_rename(self):
        with self.reported_parent_mode(0o2700), patch.object(recovery.os, "chmod") as chmod:
            receipt = self.move()
            chmod.assert_not_called()
        self.assertEqual(receipt["destination_parent_mode"], "0o2700")
        self.assertEqual(receipt["destination_parent_permissions"], "0700")
        self.assertTrue(receipt["destination_parent_setgid"])
        self.assertEqual(self.parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(recovery.tree_entries(self.destination, hash_files=True), self.inventory)

    def test_setgid_does_not_allow_group_or_other_access(self):
        for mode in (0o2710, 0o2701):
            with self.subTest(mode=oct(mode)):
                with (
                    self.reported_parent_mode(mode),
                    patch.object(recovery.os, "rename") as operation,
                ):
                    with self.assertRaisesRegex(ValueError, "mode 0700"):
                        self.move()
                    operation.assert_not_called()
                self.assertEqual(self.parent.stat().st_mode & 0o777, mode & 0o777)
                self.assertTrue(self.source.is_dir())

    def test_changed_partial_is_rejected_after_owner_acquired(self):
        (self.source / "data.npz").write_bytes(b"changed")
        with patch.object(recovery.os, "rename") as operation:
            with self.assertRaisesRegex(ValueError, "partial changed"):
                self.move()
            operation.assert_not_called()
        self.assertTrue((self.parent / "OWNER_LOCK.json").is_file())
        self.assertEqual((self.source / "data.npz").read_bytes(), b"changed")

    def test_cross_filesystem_is_rejected_without_rename(self):
        original = os.fstat

        def different_device(fd):
            fields = list(original(fd))
            fields[2] += 1
            return os.stat_result(fields)

        with (
            patch.object(recovery.os, "fstat", side_effect=different_device),
            patch.object(recovery.os, "rename") as operation,
        ):
            with self.assertRaisesRegex(ValueError, "same filesystem"):
                self.move()
            operation.assert_not_called()
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_previous_failure_and_empty_backup_preserved(self):
        previous, old_backup = self.previous_failure()
        first_record = self.root / "RECOVERY_Q1_01"
        first_before = recovery.tree_entries(first_record, hash_files=True)
        before = recovery.tree_entries(previous, hash_files=True)
        receipt = recovery.verify_previous_failure(self.root, "Q1", recovery.PHASES["Q1"])
        self.assertEqual(receipt["entries"], before)
        self.move()
        self.assertEqual(recovery.tree_entries(previous, hash_files=True), before)
        self.assertEqual(recovery.tree_entries(first_record, hash_files=True), first_before)
        self.assertEqual(list(old_backup.iterdir()), [])
        self.assertEqual(list((self.root / "INTERRUPTED_ORIGINALS_155942").iterdir()), [])

    def test_previous_moved_receipt_or_nonempty_backup_blocks_attempt03(self):
        previous, old_backup = self.previous_failure()
        (old_backup / "unexpected").write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "empty directory"):
            recovery.verify_previous_failure(self.root, "Q1", recovery.PHASES["Q1"])
        (previous / "MOVED_AND_VERIFIED.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "move/execution"):
            recovery.verify_previous_failure(self.root, "Q1", recovery.PHASES["Q1"])
        self.assertEqual((old_backup / "unexpected").read_bytes(), b"preserve")

    def test_attempt02_must_bind_unchanged_attempt01(self):
        previous, _ = self.previous_failure()
        path = previous / "PREVIOUS_ATTEMPT_PRESERVED.json"
        data = json.loads(path.read_bytes())
        data["observed_errno"] = 999
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "attempt01 binding"):
            recovery.verify_previous_failure(self.root, "Q1", recovery.PHASES["Q1"])

    def test_original_request_only_gains_resume(self):
        self.assertEqual(recovery.ATTEMPT, "03")
        for spec in recovery.PHASES.values():
            argv = [
                spec["command"],
                "--collection",
                str(recovery.BASE / spec["collection"]),
                "--config",
                recovery.CONFIG,
                "--out",
                str(recovery.BASE / spec["out"]),
            ]
            self.assertEqual(
                recovery.validate_original_request({"argv": argv}, spec),
                {"argv": [*argv, "--resume"]},
            )
            with self.assertRaises(ValueError):
                recovery.validate_original_request({"argv": [*argv, "--pilot"]}, spec)

    def test_unsafe_paths_and_local_production_run_rejected(self):
        for value in ("../outside", "/absolute", "a//b", "a/../b", ".", ".git/config"):
            with self.assertRaises(ValueError):
                recovery.safe_relative(value)
        with (
            patch.object(recovery.sys, "platform", "fixture-not-linux"),
            self.assertRaises(RuntimeError),
        ):
            recovery.enforce_cpu_environment()


if __name__ == "__main__":
    unittest.main(verbosity=2)
