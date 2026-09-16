"""Tiny metadata-only fixtures. Run with python -B beside the helper; no SSH."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HELPER = Path(__file__).with_name("measure_completed_cpu_output.py")
SPEC = importlib.util.spec_from_file_location("cpu_output_measurement_fixture", HELPER)
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)


class CompletedOutputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()
        self.candidate = self.base / "modeling_v3_20260915/candidate06"
        self.root = self.candidate / "Q2_coverage_full"
        self.root.mkdir(parents=True)
        self.complete = self.root / "COMPLETE.json"
        self.complete.write_bytes(b'{"status":"COMPLETE","summary":{"stage":"Q2"}}\n')
        self.digest = hashlib.sha256(self.complete.read_bytes()).hexdigest()
        self.out = self.base / "resource_measurement"
        self.addCleanup(patch.stopall)
        patch.object(measure, "PHYSICAL_SSVC_ROOT", self.base).start()
        patch.object(measure, "LOGICAL_SSVC_ROOT", self.base / "unused_alias").start()
        patch.object(measure.sys, "platform", "linux").start()
        patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=True).start()
        patch.object(measure.os, "getxattr", create=True, side_effect=self.xattr).start()

    @staticmethod
    def xattr(_fd, name):
        return {"ceph.dir.rbytes": b"4096", "ceph.quota.max_bytes": b"1000000"}[name]

    def run_measure(self):
        return measure.measure_completed_output(self.root, self.digest, self.out)

    def test_pending_and_closing_files_counted_symlinks_never_followed(self):
        units = self.root / "units"
        units.mkdir()
        (units / "data.npz").write_bytes(b"do not read data values")
        (units / "still.pending").write_bytes(b"pending")
        (self.root / "COVERAGE_SUMMARY.json").write_bytes(b"closing stage summary")
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "not_counted").write_bytes(b"x" * 7000)
        (units / "linkdir").symlink_to(outside, target_is_directory=True)
        (units / "linkfile").symlink_to(outside / "not_counted")
        (units / "dangling").symlink_to(outside / "missing")
        expected = sum(
            p.stat().st_size
            for p in [
                self.complete,
                units / "data.npz",
                units / "still.pending",
                self.root / "COVERAGE_SUMMARY.json",
            ]
        )
        original_open = measure.os.open

        def metadata_only_open(path, *args, **kwargs):
            if str(path).endswith(
                ("data.npz", "still.pending", "COVERAGE_SUMMARY.json", "not_counted")
            ):
                self.fail("scientific or linked data opened")
            return original_open(path, *args, **kwargs)

        with patch.object(measure.os, "open", side_effect=metadata_only_open):
            report = self.run_measure()
        total = report["tree"]["total"]
        self.assertEqual(total["regular_apparent_bytes"], expected)
        self.assertEqual(total["regular_file_count"], 4)
        self.assertEqual(total["directory_count"], 2)
        self.assertEqual(total["symlink_count"], 3)
        self.assertEqual(total["entry_count"], 9)
        self.assertEqual(report["tree"]["pending_name_count"], 1)
        self.assertEqual(report["tree"]["partitions"]["units"]["regular_file_count"], 2)
        self.assertIn("COMPLETE.json", report["tree"]["root_regular_files"])
        self.assertFalse(report["summary_bytes_used"])
        self.assertTrue((self.out / "RESOURCE_MEASUREMENT.json").is_file())
        self.assertEqual(
            report["ceph_parent"]["before"]["headroom_at_this_directory_bytes"], 995904
        )

    def test_hardlinks_deduplicated_globally_and_partition_overlap_explained(self):
        role = self.root / "interval_calibration"
        (role / "collection").mkdir(parents=True)
        (role / "response").mkdir()
        data = role / "collection/data.npz"
        data.write_bytes(b"z" * 9000)
        os.link(data, role / "response/linked.npz")
        report = self.run_measure()
        tree, size = report["tree"], data.stat().st_size
        self.assertEqual(tree["total"]["regular_file_count"], 3)
        self.assertEqual(tree["total"]["unique_regular_inode_count"], 2)
        self.assertEqual(
            tree["total"]["regular_apparent_bytes"]
            - tree["total"]["unique_regular_apparent_bytes"],
            size,
        )
        self.assertEqual(tree["partition_unique_regular_apparent_overlap_bytes"], size)
        self.assertEqual(
            tree["total"]["regular_allocated_bytes"]
            - tree["total"]["unique_regular_allocated_bytes"],
            data.stat().st_blocks * 512,
        )
        self.assertEqual(
            tree["partition_unique_regular_allocated_overlap_bytes"], data.stat().st_blocks * 512
        )
        self.assertFalse(
            report["space_semantics"]["unique_allocated_bytes_are_exclusively_owned_physical_bytes"]
        )

    def test_output_inside_tree_or_existing_output_refused(self):
        self.out = self.root / "new_output"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.run_measure()
        self.assertFalse(self.out.exists())
        self.out = self.base / "already_exists"
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_measure()

    def test_digest_changes_during_scan_refused_and_no_success_receipt(self):
        original = measure.scan_tree

        def mutate(fd):
            result = original(fd)
            self.complete.write_bytes(b'{"status":"CHANGED"}')
            return result

        with (
            patch.object(measure, "scan_tree", side_effect=mutate),
            self.assertRaisesRegex(ValueError, "COMPLETE changed"),
        ):
            self.run_measure()
        self.assertFalse((self.out / "RESOURCE_MEASUREMENT.json").exists())
        failure = json.loads((self.out / "FAILED.json").read_text())
        self.assertEqual(failure["status"], "FAILED_NO_VALID_MEASUREMENT")

    def test_wrong_initial_digest_creates_no_output(self):
        self.digest = "0" * 64
        with self.assertRaisesRegex(ValueError, "COMPLETE SHA256"):
            self.run_measure()
        self.assertFalse(self.out.exists())

    def test_root_symlink_and_symlink_ancestor_refused(self):
        original = self.root
        alias = self.candidate / "Q3_campaign"
        alias.symlink_to(original, target_is_directory=True)
        self.root = alias / "interval_calibration_seed21001"
        with self.assertRaises((ValueError, OSError)):
            self.run_measure()
        original.rename(self.candidate / "original")
        original.symlink_to(self.candidate / "original", target_is_directory=True)
        self.root = original
        with self.assertRaises((ValueError, OSError)):
            self.run_measure()
        self.assertFalse(self.out.exists())

    def test_output_symlink_ancestor_or_outside_physical_root_refused(self):
        actual = self.base / "actual_receipts"
        actual.mkdir()
        alias = self.base / "receipt_alias"
        alias.symlink_to(actual, target_is_directory=True)
        self.out = alias / "never_created"
        with self.assertRaises((ValueError, OSError)):
            self.run_measure()
        self.assertFalse((actual / "never_created").exists())
        self.out = self.base.parent / "outside_ssvc"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.run_measure()

    def test_complete_symlink_refused_even_with_correct_referenced_digest(self):
        original = self.candidate / "outside_complete.json"
        self.complete.rename(original)
        self.complete.symlink_to(original)
        with self.assertRaises(OSError):
            self.run_measure()
        self.assertFalse(self.out.exists())

    def test_cpu_slurm_and_scope_required(self):
        with (
            patch.object(measure.sys, "platform", "darwin"),
            self.assertRaisesRegex(RuntimeError, "Linux"),
        ):
            self.run_measure()
        with (
            patch.dict(os.environ, {"SLURM_JOB_GPUS": "0"}),
            self.assertRaisesRegex(RuntimeError, "GPU"),
        ):
            self.run_measure()
        with (
            patch.dict(os.environ, {"SLURM_JOB_ID": ""}),
            self.assertRaisesRegex(RuntimeError, "Slurm"),
        ):
            self.run_measure()
        self.root = self.base / "not_a_candidate_output"
        self.root.mkdir()
        with self.assertRaisesRegex(ValueError, "candidate06"):
            self.run_measure()

    def test_q3_top_level_partitions_and_unavailable_ceph_are_explicit(self):
        self.root = self.candidate / "Q3_campaign/interval_calibration_seed21001"
        (self.root / "interval_calibration/collection").mkdir(parents=True)
        (self.root / "interval_calibration/response").mkdir()
        self.complete = self.root / "COMPLETE.json"
        self.complete.write_bytes(b'{"status":"COMPLETE"}')
        self.digest = hashlib.sha256(self.complete.read_bytes()).hexdigest()
        (self.root / "interval_calibration/collection/a").write_bytes(b"123")
        (self.root / "interval_calibration/response/b").write_bytes(b"12345")
        with patch.object(measure.os, "getxattr", side_effect=OSError(95, "unsupported")):
            report = self.run_measure()
        self.assertEqual(report["tree"]["partitions"]["collection"]["regular_apparent_bytes"], 3)
        self.assertEqual(report["tree"]["partitions"]["response"]["regular_apparent_bytes"], 5)
        self.assertEqual(
            report["ceph_parent"]["before"]["attributes"]["ceph.dir.rbytes"]["status"],
            "UNAVAILABLE",
        )
        self.assertIsNone(report["ceph_parent"]["before"]["headroom_at_this_directory_bytes"])
        self.assertFalse(report["ceph_parent"]["delta_is_attributable_to_measured_task"])

    def test_ssvc_zero_quota_keeps_actual_ancestor_quota_and_headroom(self):
        ancestor_inode = self.base.parent.stat().st_ino
        ancestor_reads = 0

        def quota_for_layer(fd, name):
            nonlocal ancestor_reads
            if os.fstat(fd).st_ino == ancestor_inode:
                if name == "ceph.quota.max_bytes":
                    return b"750000000000"
                ancestor_reads += 1
                return b"700000000000" if ancestor_reads == 1 else b"701000000000"
            return {"ceph.dir.rbytes": b"12000000000", "ceph.quota.max_bytes": b"0"}[name]

        with patch.object(measure.os, "getxattr", side_effect=quota_for_layer):
            report = self.run_measure()
        ssvc, ancestor = report["ceph_parent"], report["ceph_quota_ancestor"]
        self.assertEqual(ssvc["layer"], "SSVC_DIRECTORY")
        self.assertEqual(ancestor["layer"], "FIXED_PHYSICAL_SSVC_PARENT")
        self.assertEqual(ancestor["before"]["path"], str(self.base.parent))
        self.assertEqual(ssvc["before"]["attributes"]["ceph.quota.max_bytes"]["value"], 0)
        self.assertIsNone(ssvc["before"]["headroom_at_this_directory_bytes"])
        self.assertEqual(
            ancestor["before"]["attributes"]["ceph.quota.max_bytes"]["value"], 750000000000
        )
        self.assertEqual(report["shared_quota_headroom"]["source_layer"], "ceph_quota_ancestor")
        self.assertEqual(report["shared_quota_headroom"]["before_bytes"], 50000000000)
        self.assertEqual(report["shared_quota_headroom"]["after_bytes"], 49000000000)
        self.assertEqual(ancestor["shared_attribute_delta"]["ceph.dir.rbytes"], 1000000000)
        self.assertFalse(ancestor["delta_is_attributable_to_measured_task"])

    def test_unavailable_ancestor_does_not_inherit_ssvc_headroom(self):
        ancestor_inode = self.base.parent.stat().st_ino

        def missing_ancestor(fd, name):
            if os.fstat(fd).st_ino == ancestor_inode:
                raise OSError(95, "ancestor xattr unavailable")
            return self.xattr(fd, name)

        with patch.object(measure.os, "getxattr", side_effect=missing_ancestor):
            report = self.run_measure()
        self.assertEqual(
            report["ceph_parent"]["before"]["headroom_at_this_directory_bytes"], 995904
        )
        self.assertIsNone(report["shared_quota_headroom"]["before_bytes"])
        self.assertIsNone(report["shared_quota_headroom"]["after_bytes"])

    def test_fixed_quota_ancestor_symlink_refused(self):
        actual = self.base / "physical_parent"
        actual.mkdir()
        alias = self.base / "ancestor_alias"
        alias.symlink_to(actual, target_is_directory=True)
        with (
            patch.object(measure, "PHYSICAL_SSVC_ROOT", alias / "ssvc"),
            self.assertRaisesRegex(ValueError, "ancestor must not contain symlinks"),
        ):
            measure.open_quota_ancestor()


if __name__ == "__main__":
    unittest.main(verbosity=2)
