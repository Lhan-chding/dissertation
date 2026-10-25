"""Regression checks for public input isolation and cross-split audits."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from sr_f5.data import audit_banks, audit_historical_worlds, construct_banks, public_prompt


class DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(
            (Path(__file__).resolve().parents[2] / "configs/sr_f5/experiment.json").read_text()
        )
        # A compact complete split/family matrix exercises the real constructors.
        for key in cls.config["data"]:
            if key.endswith("_roots_per_family"):
                cls.config["data"][key] = 1
        cls.config["evaluation"]["dev_monitor_roots_per_family"] = 1
        cls.banks = construct_banks(cls.config)

    def test_complete_cpu_audit(self):
        report = audit_banks(self.banks, self.config)
        self.assertEqual(report["oracle_check"]["questions_compared"], 8 * 6 * 4)
        self.assertEqual(report["split_audit"]["cross_split_world_collisions"], 0)
        self.assertFalse(
            report["factor_coverage"]["train"]["registered_heldout_conjunctions"]
        )

    def test_prompt_does_not_read_hidden_facts_or_labels(self):
        task = copy.deepcopy(self.banks["dev"][0])
        expected = public_prompt(task)
        task["gold_answer"] = "do not expose this answer"
        task["gold_selected"] = ["private selected labels"]
        task["qid"] = "private task identifier"
        task["world"]["series"] = {"Alpha": [999] * 6, "Beta": [888] * 6}
        self.assertEqual(public_prompt(task), expected)
        self.assertNotIn("private", expected)

    def test_descendant_cannot_move_splits(self):
        banks = copy.deepcopy(self.banks)
        banks["pilot"][0] = copy.deepcopy(banks["dev"][0])
        banks["pilot"][0]["split"] = "pilot"
        with self.assertRaises((AssertionError, ValueError)):
            audit_banks(banks, self.config)

    def test_reordered_renamed_world_cannot_cross_splits(self):
        banks = copy.deepcopy(self.banks)
        world = copy.deepcopy(banks["dev"][0]["world"])
        world["categories"] = list(reversed(world["categories"]))
        world["series"] = {key: list(reversed(value)) for key, value in world["series"].items()}
        banks["pilot"][0]["world"] = world
        with self.assertRaises((AssertionError, ValueError)):
            audit_banks(banks, self.config)

    def test_relation_metadata_is_verified(self):
        banks = copy.deepcopy(self.banks)
        banks["train"][1]["relation_to_v0"]["answer_delta"] = 999
        with self.assertRaisesRegex(ValueError, "root relation"):
            audit_banks(banks, self.config)

    def _historical_fixture(self, root, old_task):
        data = root / "DATA"
        data.mkdir()
        (data / "BUILD_RECEIPT.json").write_text(json.dumps({"questions": 1, "status": "PASS"}))
        (data / "TASKS_PRIVATE.jsonl").write_text(json.dumps(self.banks["dev"][0]) + "\n")
        history = root / "old_tasks.jsonl"
        history.write_text(json.dumps(old_task) + "\n")
        return history

    def test_historical_exclusion_records_success_and_preserves_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self._historical_fixture(root, self.banks["pilot"][0])
            report = audit_historical_worlds(root, [history])
            self.assertEqual(report["status"], "PASS")
            receipt = json.loads((root / "DATA/BUILD_RECEIPT.json").read_text())
            self.assertEqual(receipt["questions"], 1)
            self.assertEqual(receipt["historical_world_exclusion"], "PASS")

    def test_historical_relabelled_reordered_world_fails_with_evidence(self):
        old = copy.deepcopy(self.banks["dev"][0])
        old["world"]["categories"] = [
            f"OldCategory{i}" for i in range(len(old["world"]["categories"]))
        ]
        old["world"]["series"] = {
            key: list(reversed(value)) for key, value in old["world"]["series"].items()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = self._historical_fixture(root, old)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                audit_historical_worlds(root, [history])
            report = json.loads((root / "DATA/HISTORICAL_WORLD_AUDIT.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["collisions"][0]["kind"], "renamed_reordered_world")

    def test_missing_historical_source_cannot_be_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._historical_fixture(root, self.banks["pilot"][0])
            with self.assertRaises(FileNotFoundError):
                audit_historical_worlds(root, [root / "absent.jsonl"])
            report = json.loads((root / "DATA/HISTORICAL_WORLD_AUDIT.json").read_text())
            self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
