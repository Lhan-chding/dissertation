"""Focused query and split tests; no model loading or GPU smoke runs."""

import copy
import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from sr_f3_ce import contract as native
from sr_f4 import data
from sr_f4 import task_contract as contract

WORLD = {
    "categories": ["January", "February", "March", "April"],
    "series": {"Alpha": [80, 70, 30, 10], "Beta": [10, 20, 60, 90]},
    "domain": [10, 90],
}


def full_response(world, query):
    return json.dumps(
        {
            "evidence": [
                {"series": name, "category": category, "value": values[index]}
                for name, values in world["series"].items()
                for index, category in enumerate(world["categories"])
            ],
            "answer": contract.encode(contract.execute(world, query)),
        }
    )


def old_tasks():
    rng = random.Random(198)
    rows = []
    for family in data.FAMILIES:
        for root in range(8):
            world = data._world(rng, 4)
            for variant in range(8):
                query = data._representative_query(family, rng, variant)
                query.pop("selection_series", None)
                query.pop("include_end", None)
                text = data.question(world, query)
                rows.append(
                    dict(
                        qid=f"OLD_{family}_{root}_{variant}",
                        root_id=f"OLD_{family}_{root}",
                        world=world,
                        query=query,
                        pool="TRAIN_RL",
                        family=family,
                        variant=variant,
                        chart="grouped_bar",
                        render_style="main",
                        question=text,
                        prompt=text + "\n" + data.CONTRACT_V2,
                    )
                )
    return rows


class QueryTests(unittest.TestCase):
    def test_beta_topk_and_sufficiency_use_reported_beta(self):
        query = dict(
            family="TOPK",
            k=2,
            direction="largest",
            selection_series="Beta",
            value="Alpha",
            aggregate="sum",
        )
        self.assertEqual(contract.selected(WORLD, query), [3, 2])
        self.assertEqual(contract.execute(WORLD, query), 40)
        self.assertEqual(contract.independently_answer(WORLD, query), 40)
        self.assertEqual(contract.score(full_response(WORLD, query), WORLD, query)["J"], 1)
        facts = {("Beta", i): WORLD["series"]["Beta"][i] for i in range(4)}
        facts.update({("Alpha", 2): 30, ("Alpha", 3): 10})
        self.assertTrue(contract.sufficient_template(WORLD, query, facts))
        del facts[("Beta", 0)]
        self.assertFalse(contract.sufficient_template(WORLD, query, facts))
        self.assertEqual(WORLD["series"]["Alpha"], [80, 70, 30, 10])

    def test_interval_right_exclusion_and_empty_semantics(self):
        query = dict(
            family="INTERVAL",
            start=0,
            end=3,
            include_start=False,
            include_end=False,
            value="diff",
            aggregate="mean",
        )
        self.assertEqual(contract.selected(WORLD, query), [1, 2])
        self.assertEqual(contract.execute(WORLD, query), 10)
        self.assertEqual(contract.score(full_response(WORLD, query), WORLD, query)["J"], 1)
        empty = dict(query, start=1, end=1)
        self.assertIsNone(contract.execute(WORLD, empty))
        self.assertIsNone(contract.independently_answer(WORLD, empty))
        self.assertEqual(contract.execute(WORLD, dict(empty, aggregate="sum")), 0)

    def test_legacy_scorer_is_unchanged(self):
        for query in [
            dict(family="TOPK", k=2, value="Beta", aggregate="mean"),
            dict(
                family="INTERVAL",
                start=0,
                end=3,
                include_start=False,
                value="diff",
                aggregate="mean",
            ),
            dict(family="THRESHOLD", threshold=50, value="Alpha", aggregate="sum"),
            dict(family="CROSS", value="Alpha", aggregate="sum"),
        ]:
            for text in (
                full_response(WORLD, query),
                '{"answer":0}',
                '{"evidence":[],"answer":0}',
                '{"answer":0,"answer":1}',
            ):
                self.assertEqual(
                    contract.score(text, WORLD, query), native.score(text, WORLD, query)
                )

    def test_exact_programs_agree_all_supported_aggregates(self):
        for cell in data.CELLS:
            for query in data._query_pair(cell, 4, 0):
                for operation in ("sum", "mean", "range", "min", "max", "count"):
                    query = dict(query, aggregate=operation)
                    self.assertEqual(
                        contract.execute(WORLD, query), contract.independently_answer(WORLD, query)
                    )
                    self.assertEqual(
                        contract.score(full_response(WORLD, query), WORLD, query)["J"], 1
                    )


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = old_tasks()
        cls.rows = data.generate_new_tasks(cls.old)

    def test_registered_counts_isolation_and_gold(self):
        self.assertEqual(len(self.rows), 1696)
        self.assertEqual(len({row["world_id"] for row in self.rows}), 880)
        counts = data.validate_splits(self.rows, self.old)
        self.assertEqual(counts["B_DEV"], {"worlds": 128, "prompts": 200})
        self.assertEqual(counts["B_CONFIRM"], {"worlds": 32, "prompts": 56})
        for row in self.rows:
            self.assertEqual(row["gold_answer"], row["gold_independent"])
            self.assertEqual(
                contract.encode(contract.execute(row["world"], row["query"])), row["gold_answer"]
            )

    def test_determinism_and_no_input_mutation(self):
        before = copy.deepcopy(self.old)
        again = data.generate_new_tasks(self.old)
        self.assertEqual(again, self.rows)
        self.assertEqual(self.old, before)

    def test_controlled_pairs_change_only_registered_fields(self):
        pairs = {}
        for row in self.rows:
            if row["pair_id"] is not None:
                pairs.setdefault(row["pair_id"], []).append(row)
        for pair in pairs.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]["world"], pair[1]["world"])
            self.assertEqual(pair[0]["chart"], pair[1]["chart"])
            if pair[0]["purpose"] == "presentation":
                self.assertEqual(pair[0]["query"], pair[1]["query"])
                self.assertEqual(pair[0]["gold_answer"], pair[1]["gold_answer"])
                changed = sum(
                    pair[0][key] != pair[1][key] for key in ("render_style", "wording_template")
                )
                self.assertEqual(changed, 1)

    def test_exposure_and_opposite_branches(self):
        new, old = data.inventories(data._copy_old(self.old) + self.rows)
        schedules = [data.build_c_schedule(new, old, repeat) for repeat in (1, 2)]
        chosen = []
        for schedule in schedules:
            self.assertEqual(len(schedule), 32)
            exposed = {}
            for step in schedule:
                self.assertEqual(step["answers"], 128)
                self.assertEqual(len(step["assignments"]), 16)
                self.assertEqual(
                    Counter(x["source"] for x in step["assignments"]),
                    {"C_TRAIN": 12, "OLD_TRAIN": 4},
                )
                for item in step["assignments"]:
                    if item["source"] == "C_TRAIN":
                        self.assertNotIn(item["world_id"], exposed)
                        exposed[item["world_id"]] = item["qid"]
            self.assertEqual(len(exposed), 384)
            chosen.append(exposed)
        self.assertEqual(set(chosen[0]), set(chosen[1]))
        self.assertTrue(all(chosen[0][world] != chosen[1][world] for world in chosen[0]))

    def test_probe_and_manifest_output_counts(self):
        groups = data.b0_groups(self.old)
        self.assertEqual(sum(row["G"] for row in groups), 1280)
        self.assertEqual(len({row["world_id"] for row in groups if row["probe"] == "extra_v7"}), 8)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data.write_jsonl(root / "plan/manifests/TASKS_AUDIT_ONLY.jsonl", self.old)
            receipt = data.build_dataset(root / "plan", root / "DATA", render=False)
            self.assertEqual(receipt["confirm_model_calls"], 0)
            self.assertEqual(receipt["status"], "MANIFESTS_READY_IMAGES_NOT_RENDERED")
            self.assertEqual(len(data.read_jsonl(root / "DATA/C_TRAIN_PROBE_GROUPS.jsonl")), 48)
            self.assertEqual(len(data.read_jsonl(root / "DATA/B2_DEV_GROUPS.jsonl")), 200)


if __name__ == "__main__":
    unittest.main()
