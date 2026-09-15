"""Tiny scheduler mocks; never contact SSH, Slurm or a scientific dataset."""

import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

FLOW = next(p for p in Path(__file__).resolve().parents if (p / "src/modeling_v3").is_dir())
SCRIPT = FLOW / "docs/modeling_v3/operators/submit_candidate06_gate.py"
spec = importlib.util.spec_from_file_location("gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GateFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / "candidate06"
        self.checkout = self.base / "checkout"
        self.checkout.mkdir(parents=True)
        code = self.checkout / "ssvc_flow/src/minimal.py"
        code.parent.mkdir(parents=True)
        code.write_text("# fixture\n")
        manifest = self.checkout / "CODE_SNAPSHOT_MANIFEST.json"
        manifest.write_text(json.dumps({"ssvc_flow/src/minimal.py": gate.digest(code)}))
        self.cpu = self.base / "CPU_acceptance_02"
        originals = {}
        for rel in [
            "CPU_acceptance_02/result.json",
            "Q1_observation_full/COMPLETE.json",
            "Q2_coverage_full/COMPLETE.json",
            "Q1_analysis_full/COMPLETE.json",
            "Q2_analysis_full/COMPLETE.json",
        ]:
            p = self.base / rel
            p.parent.mkdir(exist_ok=True)
            p.write_text("{}\n")
            originals[str(p)] = gate.digest(p)
        (self.cpu / "pytest.log").write_text("fixture: 2 passed\n")
        (self.cpu / "junit.xml").write_text(
            '<testsuite><testcase classname="test_observation_geometry" name="fixture1"/>'
            '<testcase classname="test_coverage_models" name="fixture2"/></testsuite>'
        )
        result = {
            "execution_kind": "SERVER_CPU",
            "platform": "Linux-fixture",
            "slurm_job_id": "999999",
            "exit_code": 0,
            "sources_unchanged_during_run": True,
            "acceptance_scope": "FULL_REPOSITORY_EXCEPT_SEALED_DATA_TEST",
            "requested_targets": gate.TARGETS,
            "snapshot_manifest_sha256": gate.digest(manifest),
            "counts": {"passed": 2, "failed": 0, "errors": 0, "skipped": 0},
            "source_and_test_hashes_before": {"src/minimal.py": gate.digest(code)},
            "log_sha256": gate.digest(self.cpu / "pytest.log"),
            "junit_sha256": gate.digest(self.cpu / "junit.xml"),
        }
        (self.cpu / "result.json").write_text(json.dumps(result))
        originals[str(self.cpu / "result.json")] = gate.digest(self.cpu / "result.json")
        review = self.base / "CPU_DEVELOPMENT_REVIEW_01.json"
        review.write_text(
            json.dumps(
                {
                    "status": "REVIEWED_FOR_FREEZE_AND_Q4_PREPARATION",
                    "source_sha256": gate.SOURCE,
                    "originals": originals,
                }
            )
        )
        inputs = self.base / "Q3_Q4_OPERATOR_INPUTS_01"
        inputs.mkdir()
        shutil.copyfile(
            FLOW / "docs/modeling_v3/Q3_SELECTION_DRAFT.json", inputs / "Q3_SELECTION_DRAFT.json"
        )
        (inputs / "PRE_GPU_INPUT_BINDINGS.json").write_text('{"fixture": true}\n')
        fixture_binding = patch.object(
            gate, "INPUT_BINDINGS_SHA", gate.digest(inputs / "PRE_GPU_INPUT_BINDINGS.json")
        )
        fixture_binding.start()
        self.addCleanup(fixture_binding.stop)
        for key, value in [
            ("BASE", self.base),
            ("CHECKOUT", self.checkout),
            ("SNAPSHOT", gate.digest(manifest)),
        ]:
            context = patch.object(gate, key, value)
            context.start()
            self.addCleanup(context.stop)
        self.argv = [
            "gate",
            "--stage",
            "Q3_FREEZE",
            "--review-sha256",
            gate.digest(review),
            "--cpu-tests",
            str(self.cpu),
        ]

    def invoke(self, queued=0, **kwargs):
        calls = []

        def fake_run(argv, **options):
            calls.append(argv)
            output = (gate.QOS + "\n") * queued if argv[0] == "squeue" else "999999\n"
            return subprocess.CompletedProcess(argv, 0, output, "")

        with (
            patch.object(sys, "argv", self.argv),
            patch.object(gate.subprocess, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            gate.main()
        return calls

    def test_only_frozen_cpu_command_is_submitted_and_receipt_written(self):
        calls = self.invoke()
        self.assertEqual([a[0] for a in calls], ["squeue", "sbatch"])
        self.assertIn("--array", calls[0])
        self.assertFalse(any("--gres" in a for a in calls[1]))
        receipt = json.loads((self.base / "Q3_FREEZE_SUBMIT_RECEIPT.json").read_text())
        self.assertEqual(receipt["job_id"], "999999")
        request = json.loads((self.base / "Q3_FREEZE_REQUEST.json").read_text())
        self.assertEqual(request["argv"][0], "freeze")

    def test_full_qos_rejects_before_request_or_submission(self):
        with self.assertRaisesRegex(ValueError, "no submission capacity"):
            self.invoke(queued=5)
        self.assertFalse((self.base / "Q3_FREEZE_REQUEST.json").exists())

    def test_disk_quota_failure_prevents_sbatch(self):
        calls = []

        def fake_run(argv, **options):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, "", "")

        with (
            patch.object(sys, "argv", self.argv),
            patch.object(gate.subprocess, "run", side_effect=fake_run),
            patch.object(gate.os, "fsync", side_effect=OSError(122, "Disk quota exceeded")),
            self.assertRaises(OSError),
        ):
            gate.main()
        self.assertEqual([a[0] for a in calls], ["squeue"])
        self.assertFalse((self.base / "Q3_FREEZE_SUBMIT_RECEIPT.json").exists())

    def test_existing_intent_rejects_before_scheduler(self):
        (self.base / "Q3_FREEZE_SUBMIT_INTENT.json").write_text("{}")
        with (
            patch.object(sys, "argv", self.argv),
            patch.object(gate.subprocess, "run") as call,
            self.assertRaises(FileExistsError),
        ):
            gate.main()
        call.assert_not_called()

    def test_failed_cpu_with_updated_review_hash_cannot_submit(self):
        result = self.cpu / "result.json"
        result.write_text(json.dumps({"exit_code": 1, "counts": {"failed": 1}}))
        review_path = self.base / "CPU_DEVELOPMENT_REVIEW_01.json"
        review = json.loads(review_path.read_text())
        review["originals"][str(result)] = gate.digest(result)
        review_path.write_text(json.dumps(review))
        self.argv[self.argv.index("--review-sha256") + 1] = gate.digest(review_path)
        with (
            patch.object(sys, "argv", self.argv),
            patch.object(
                gate.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")
            ) as call,
            self.assertRaisesRegex(ValueError, "CPU acceptance"),
        ):
            gate.main()
        call.assert_not_called()

    def test_changed_original_rejects_before_scheduler(self):
        (self.cpu / "result.json").write_text('{"changed":true}')
        with (
            patch.object(sys, "argv", self.argv),
            patch.object(gate.subprocess, "run") as call,
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            gate.main()
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
