import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "satdose_parallel_main_plan.py"
SPEC = importlib.util.spec_from_file_location("satdose_parallel_main_plan", SOURCE)
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


class ParallelMainCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # In-memory sample layouts exercise partitioning without accessing
        # cluster mounts. The production public entry point pins all bytes.
        cls.rows = []
        for start in (11031, 11032):
            for _, arm, step in plan.lane_endpoints(start):
                panels = plan.FULL_PANELS if step in (0, 24) else plan.PRIMARY_PANEL
                for panel, count in panels.items():
                    for i in range(count):
                        row = dict(split="MAIN", start_seed=start, arm=arm, step=step,
                                   panel=panel, root_id=f"fixture-{i}", view="view", draw=0)
                        row["seed"] = plan._historical_seed(row)
                        row["request_id"] = f"SAT1/MAIN/{start}/{arm}/s{step:02d}/{panel}/fixture-{i}/view/d00"
                        cls.rows.append(row)

    def test_lane_bounds_and_shared_draw_seeds(self):
        receipt = plan._partition(self.rows)
        self.assertEqual(receipt["unique_requests"], 13760)
        self.assertEqual(receipt["endpoint_count"], 14)
        self.assertEqual(receipt["omitted_requests"], 0)
        self.assertEqual(receipt["overlapping_requests"], 0)
        for seed in (11031, 11032):
            lane = receipt["lanes"][str(seed)]
            self.assertEqual(lane["responses"], 6880)
            self.assertEqual([v["responses"] for v in lane["endpoints"]],
                             [1952, 256, 256, 1952, 256, 256, 1952])
            ids = [r["request_id"] for r in self.rows if r["start_seed"] == seed]
            self.assertEqual(lane["request_ids_sha256"], plan.object_sha(ids))
        self.assertEqual(self.rows[0]["seed"], self.rows[6880]["seed"])
        self.assertNotEqual(self.rows[0]["request_id"], self.rows[6880]["request_id"])

    def test_duplicate_with_unchanged_total_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "repeated"):
            plan._partition([self.rows[1], *self.rows[1:]])

    def test_missing_request_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "allocation"):
            plan._partition(self.rows[1:])

    def test_cross_lane_assignment_and_seed_tampering_are_rejected(self):
        for fields, error in (({"start_seed": 11032}, "Misassigned"), ({"seed": -1}, "seed differs")):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, error):
                plan._partition([dict(self.rows[0], **fields), *self.rows[1:]])

    def test_within_endpoint_reordering_changes_binding(self):
        before = plan._partition(self.rows)
        after = plan._partition([self.rows[1], self.rows[0], *self.rows[2:]])
        self.assertNotEqual(before["lanes"]["11031"]["endpoints"][0]["request_ids_sha256"],
                            after["lanes"]["11031"]["endpoints"][0]["request_ids_sha256"])
        self.assertEqual(before["lanes"]["11032"], after["lanes"]["11032"])

    def test_unfrozen_manifest_or_symlink_is_rejected_before_planning(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "manifest"
            path.write_text(json.dumps(self.rows[0]))
            with self.assertRaisesRegex(ValueError, "frozen SHA"):
                plan.build_plan(path)
            link = path.parent / "link"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "Noncanonical"):
                plan.build_plan(link)


if __name__ == "__main__":
    unittest.main()
