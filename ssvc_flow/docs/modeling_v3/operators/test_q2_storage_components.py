"""Tiny local stat-only fixtures; no experiment, SSH or scheduler calls."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HELPER = Path(__file__).with_name("q2_storage_components.py")
SPEC = importlib.util.spec_from_file_location("q2_components_fixture", HELPER)
components = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(components)


class Q2ComponentsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "modeling_v3_20260915/candidate06/Q2_coverage_full"
        self.root.mkdir(parents=True)
        self.complete = self.root / "COMPLETE.json"
        self.complete.write_bytes(b'{"status":"COMPLETE"}')
        self.digest = hashlib.sha256(self.complete.read_bytes()).hexdigest()
        self.out = self.base / "components"
        self.addCleanup(patch.stopall)
        patch.object(components.measurement, "PHYSICAL_SSVC_ROOT", self.base).start()
        patch.object(components.measurement, "LOGICAL_SSVC_ROOT", self.base / "alias").start()
        patch.object(components.measurement.sys, "platform", "linux").start()
        patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}, clear=True).start()

    def file(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def run_measure(self):
        return components.measure_q2_components(self.root, self.digest, self.out)

    def test_stat_only_categories_and_q3_matching_cache_n(self):
        self.file("units/o1/calibration_n64/contributions.npz", b"a" * 10)
        self.file("units/o1/direct_n64/contributions.npz", b"b" * 20)
        self.file("units/o1/direct_n64/COMPLETE.json", b"cache metadata")
        self.file("units/o1/calibration_n84/contributions.npz", b"c" * 100)
        self.file("units/o1/calibration_n1024/raw_scores.npz", b"d" * 30)
        self.file("units/o1/fits/abc/PREDICTIONS.npz", b"not actually a zip")
        self.file("units/o1/QUERY_REFERENCE.npz", b"reference")
        self.file("units/o1/error_decomposition/ORIGIN_JACOBIAN_AND_BASELINES.npz", b"jacobian")
        self.file("units/o1/error_decomposition/abc.npz", b"error component")
        self.file("units/o1/fits/abc/QUERY_LOG.json", b"query log")
        original_open = components.os.open

        def no_data_open(path, *args, **kwargs):
            if str(path).endswith(".npz") or str(path) == "QUERY_LOG.json":
                self.fail("data file opened; only stat is permitted")
            return original_open(path, *args, **kwargs)

        with patch.object(components.os, "open", side_effect=no_data_open):
            report = self.run_measure()
        categories = report["category_totals"]
        self.assertEqual(categories["measurement_contributions"]["regular_apparent_bytes"], 130)
        self.assertEqual(categories["measurement_raw_scores"]["regular_apparent_bytes"], 30)
        self.assertEqual(categories["fit_predictions"]["regular_file_count"], 1)
        self.assertEqual(categories["decomposition_origin"]["regular_file_count"], 1)
        self.assertEqual(categories["decomposition_design"]["regular_file_count"], 1)
        self.assertEqual(
            report["q3_matching_cache_n"]["per_n"]["64"]["total"]["regular_apparent_bytes"],
            30 + len(b"cache metadata"),
        )
        self.assertEqual(
            report["q3_matching_cache_n"]["per_n"]["1024"]["total"]["regular_apparent_bytes"], 30
        )
        self.assertEqual(report["q2_other_cache_n"]["total"]["regular_apparent_bytes"], 100)
        self.assertEqual(
            sum(row["regular_apparent_bytes"] for row in categories.values()),
            report["tree_total"]["regular_apparent_bytes"],
        )
        self.assertTrue(report["reconciliation"]["matches_independent_full_tree_scan"])
        self.assertFalse(report["zip_headers_read"])
        self.assertIsNone(report["Q3_storage_estimate_bytes"])

    def test_symlinks_not_followed_pending_retained_and_hardlinks_deduplicated(self):
        data = self.file("units/o1/calibration_n64/contributions.npz", b"x" * 8000)
        second = self.root / "units/o1/direct_n64/contributions.npz"
        second.parent.mkdir()
        os.link(data, second)
        self.file("units/o1/unfinished.pending-1", b"pending")
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "not_counted").write_bytes(b"z" * 10000)
        (self.root / "link").symlink_to(outside, target_is_directory=True)
        report = self.run_measure()
        total = report["tree_total"]
        self.assertEqual(total["regular_file_count"], 4)
        self.assertEqual(total["unique_regular_inode_count"], 3)
        self.assertEqual(
            total["regular_apparent_bytes"] - total["unique_regular_apparent_bytes"], 8000
        )
        self.assertEqual(report["symlink_count"], 1)
        self.assertEqual(report["pending_name_count"], 1)
        self.assertEqual(report["category_totals"]["other_regular"]["regular_file_count"], 1)

    def test_digest_change_and_no_valid_receipt(self):
        scan = components.scan_components

        def change(fd):
            result = scan(fd)
            self.complete.write_bytes(b"changed")
            return result

        with (
            patch.object(components, "scan_components", side_effect=change),
            self.assertRaisesRegex(ValueError, "COMPLETE changed"),
        ):
            self.run_measure()
        self.assertFalse((self.out / "COMPONENTS.json").exists())
        self.assertEqual(
            json.loads((self.out / "FAILED.json").read_text())["status"],
            "FAILED_NO_VALID_COMPONENT_MEASUREMENT",
        )

    def test_metadata_change_between_two_scans_rejected(self):
        self.file("units/o1/data.pending", b"a")
        scan = components.scan_components

        def change(fd):
            result = scan(fd)
            self.file("units/o1/data.pending", b"changed metadata")
            return result

        with (
            patch.object(components, "scan_components", side_effect=change),
            self.assertRaisesRegex(ValueError, "metadata changed"),
        ):
            self.run_measure()

    def test_outside_scope_output_inside_root_and_existing_output_refused(self):
        self.out = self.root / "not_allowed"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.run_measure()
        self.out = self.base / "already"
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_measure()
        self.root = self.base / "modeling_v3_20260915/candidate06/Q3_campaign"
        with self.assertRaisesRegex(ValueError, "Q2_coverage_full"):
            self.run_measure()

    def test_gpu_and_wrong_digest_rejected_before_output(self):
        with (
            patch.dict(os.environ, {"SLURM_JOB_GPUS": "0"}),
            self.assertRaisesRegex(RuntimeError, "GPU"),
        ):
            self.run_measure()
        self.digest = "0" * 64
        with self.assertRaisesRegex(ValueError, "COMPLETE SHA256"):
            self.run_measure()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
