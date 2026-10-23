"""Construction and information-boundary tests for the fixed TOPK probe."""

import copy
import json
import tempfile
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from sr_f3_ce.prompts import CONTRACT_V2
from sr_f4 import c_probe_data as probe
from sr_f4 import data
from sr_f4 import task_contract as oracle


class TopkProbeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = probe.generate_tasks()

    def test_fixed_budget_balances_every_cell_and_parent(self):
        self.assertEqual(
            probe.validate_tasks(self.tasks),
            {"worlds": 32, "queries": 64, "groups": 192, "responses": 1536},
        )
        self.assertEqual(len({t["qid"] for t in self.tasks}), 64)
        self.assertEqual(len({t["world_id"] for t in self.tasks}), 32)
        self.assertEqual(Counter(t["cell"] for t in self.tasks), dict.fromkeys(probe.CELLS, 16))
        groups = probe.generate_groups(self.tasks)
        self.assertEqual(probe.generate_groups(list(reversed(self.tasks))), groups)
        self.assertEqual(Counter(g["block"] for g in groups), {"b0": 64, "b1": 64, "b2": 64})
        self.assertEqual(sum(g["G"] for g in groups), 1536)
        self.assertEqual(len({g["group_id"] for g in groups}), 192)
        for cell in probe.CELLS:
            worlds = [t for t in self.tasks if t["cell"] == cell and t["branch"] == 0]
            self.assertEqual(
                Counter((t["query"]["aggregate"], t["query"]["direction"]) for t in worlds),
                {
                    ("sum", "largest"): 2,
                    ("sum", "smallest"): 2,
                    ("mean", "largest"): 2,
                    ("mean", "smallest"): 2,
                },
            )
            if cell.endswith("BOUNDARY"):
                for subtype in ("near", "tied"):
                    subset = [t for t in worlds if t["boundary_subtype"] == subtype]
                    self.assertEqual(len(subset), 4)
                    self.assertEqual(
                        Counter((t["query"]["aggregate"], t["query"]["direction"]) for t in subset),
                        {
                            ("sum", "largest"): 1,
                            ("sum", "smallest"): 1,
                            ("mean", "largest"): 1,
                            ("mean", "smallest"): 1,
                        },
                    )

    def test_deterministic_single_construction_and_no_shared_mutable_worlds(self):
        with patch.object(probe, "_column", wraps=probe._column) as build:
            again = probe.generate_tasks()
        self.assertEqual(build.call_count, 64)
        self.assertEqual(again, self.tasks)
        self.assertTrue(all(t["generation_attempt"] == 0 for t in again))
        again[0]["world"]["series"]["Alpha"][0] = 999
        self.assertEqual(again[1], self.tasks[1])
        again[0]["query"]["aggregate"] = "range"
        self.assertEqual(again[0]["gold_program"]["query"], self.tasks[0]["query"])
        self.assertEqual(probe.generate_tasks(), self.tasks)

    def test_unique_worlds_and_noncanonical_printed_category_order(self):
        worlds = [t["world"] for t in self.tasks if t["branch"] == 0]
        self.assertEqual(len({data.base_key(world) for world in worlds}), 32)
        for world in worlds:
            n = len(world["categories"])
            self.assertIn(n, (6, 8))
            self.assertEqual(set(world["categories"]), set(probe.CATEGORIES[:n]))
            self.assertNotEqual(world["categories"], list(probe.CATEGORIES[:n]))
            # The current SR-F4 generator always uses this canonical category order.
            self.assertNotEqual(world["categories"], list(data.CATEGORIES[:n]))

    def test_fixed_boundaries_hold_for_both_ranking_columns(self):
        for task in self.tasks:
            world, query = task["world"], task["query"]
            oracle.validate_query(world, query)
            self.assertEqual(world["domain"], [10, 90])
            self.assertEqual(len(world["categories"]), 2 * query["k"])
            expected = {"separated": 12, "near": 1, "tied": 0}[task["boundary_subtype"]]
            for values in world["series"].values():
                self.assertTrue(all(type(v) is int and 10 <= v <= 90 for v in values))
                ordered = sorted(values, reverse=query["direction"] == "largest")
                self.assertEqual(abs(ordered[query["k"] - 1] - ordered[query["k"]]), expected)
                if expected:
                    self.assertEqual(len(set(values)), len(values))
                else:
                    self.assertEqual(len(set(values)), len(values) - 1)

    def test_pairs_and_exact_oracles_and_printed_order_ties(self):
        for first, second in zip(self.tasks[::2], self.tasks[1::2], strict=True):
            self.assertEqual(first["world"], second["world"])
            self.assertEqual(first["world_id"], second["world_id"])
            self.assertEqual(first["query"]["selection_series"], "Alpha")
            self.assertEqual(second["query"]["selection_series"], "Beta")
            self.assertEqual(first["query"]["value"], "Beta")
            self.assertEqual(second["query"]["value"], "Alpha")
            for task in (first, second):
                world, query = task["world"], task["query"]
                self.assertEqual(
                    oracle.execute(world, query), oracle.independently_answer(world, query)
                )
                self.assertEqual(task["gold_answer"], oracle.encode(oracle.execute(world, query)))
                values = world["series"][query["selection_series"]]
                if task["boundary_subtype"] == "tied":
                    repeated = next(value for value, count in Counter(values).items() if count == 2)
                    tied = [i for i, value in enumerate(values) if value == repeated]
                    self.assertIn(tied[0], task["target_indices"])
                    self.assertNotIn(tied[1], task["target_indices"])

    def test_wrong_set_correct_answer_cases_are_retained_as_diagnostics(self):
        collisions = [
            t for t in self.tasks if t["answer_collision_diagnostic"]["wrong_set_same_answer_count"]
        ]
        self.assertTrue(collisions, "Frozen stream must preserve its observed answer collisions")
        for task in self.tasks:
            diagnostic = task["answer_collision_diagnostic"]
            self.assertEqual(diagnostic["policy"], "RECORD_ONLY_NO_FILTER")
            self.assertEqual(
                diagnostic["wrong_set_same_answer_count"],
                len(diagnostic["wrong_set_same_answer_sets"]),
            )
            for wrong_set in diagnostic["wrong_set_same_answer_sets"]:
                self.assertNotEqual(set(wrong_set), set(task["target_sets"]))
                world, query = task["world"], task["query"]
                indices = [world["categories"].index(category) for category in wrong_set]
                answer = sum(Fraction(world["series"][query["value"]][i]) for i in indices)
                if query["aggregate"] == "mean":
                    answer /= query["k"]
                self.assertEqual(oracle.encode(answer), task["gold_answer"])

    def test_manifest_input_does_not_expose_gold_or_boundary_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = probe.prepare_data(directory)
            root = Path(directory)
            self.assertEqual(receipt["model_calls"], 0)
            self.assertEqual(receipt["status"], "MANIFESTS_READY_IMAGES_NOT_RENDERED")
            self.assertEqual(receipt["counts"]["responses"], 1536)
            self.assertEqual(receipt["rejected_worlds"], 0)
            self.assertGreater(receipt["queries_with_answer_collisions"], 0)
            inputs = data.read_jsonl(root / "MODEL_INPUTS.jsonl")
            self.assertEqual(len(inputs), 64)
            for model_input, task in zip(inputs, self.tasks, strict=True):
                self.assertEqual(set(model_input), set(probe.MODEL_INPUT_FIELDS))
                self.assertEqual(model_input["prompt"], task["question"] + "\n" + CONTRACT_V2)
                altered = copy.deepcopy(task["world"])
                altered["series"] = {"Alpha": [10] * 8, "Beta": [90] * 8}
                self.assertEqual(data.question(altered, task["query"]), task["question"])
            plan = json.loads((root / "PLAN.json").read_text())
            self.assertEqual(plan["measurement_version"], probe.MEASUREMENT_VERSION)
            self.assertEqual(plan["training_updates"], 0)
            self.assertEqual(plan["primary_policy"], "F_PRIME_KNOWN_75")
            before = (root / "TASKS.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                probe.prepare_data(directory)
            self.assertEqual((root / "TASKS.jsonl").read_bytes(), before)

    def test_validation_rejects_changed_boundary_or_duplicate_without_replacement(self):
        changed = copy.deepcopy(self.tasks)
        changed[0]["world"]["series"]["Alpha"] = [50] * 6
        with self.assertRaisesRegex(ValueError, "Ranking boundary"):
            probe.validate_tasks(changed)
        changed = copy.deepcopy(self.tasks)
        changed[1]["qid"] = changed[0]["qid"]
        with self.assertRaisesRegex(ValueError, "duplicate qid"):
            probe.validate_tasks(changed)


if __name__ == "__main__":
    unittest.main()
