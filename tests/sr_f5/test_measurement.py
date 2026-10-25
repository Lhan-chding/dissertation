import copy
import json
import unittest
from pathlib import Path

from sr_f4 import task_contract as legacy
from sr_f5.measurement import lossless_legacy_projection, parse_response, score_response
from srf5_ref.generator import make_bank
from srf5_ref.oracle import canonical_evidence, encode, evaluate_a


class NativeMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).parent / "collision_J1_SFAIL.json").read_text())

    def test_selected_semantics_cannot_change_j(self):
        f = self.fixture
        obj = json.loads(f["raw_response"])
        for selected in (["Mug", "Lamp", "Desk"], [], ["Mug", "Mug"], ["unknown"], ["Vase"]):
            obj["selected"] = selected
            row = score_response(json.dumps(obj), f["world"], f["query"])
            self.assertEqual((row["V"], row["J"], row["J_legacy_projection"]), (1, 1, 1))
            self.assertEqual(row["S"], "PASS" if selected == ["Mug", "Lamp", "Desk"] else "FAIL")

    def test_projection_preserves_literals_and_work_in_all_member_positions(self):
        members = [
            '"evidence" : []',
            '"work":"候选=[x]; selected=not-a-field; \\"selected\\": [x]"',
            '"answer":90.00000000000000000000000000000000000000001',
        ]
        for position in range(4):
            parts = [*members[:position], '"selected": ["Mug", "Lamp"]', *members[position:]]
            raw = "  {" + ",".join(parts) + "} \n"
            projection = lossless_legacy_projection(raw)
            self.assertEqual(projection, "  {" + ",".join(members) + "} \n")
            for member in members:
                self.assertIn(member, projection)

    def test_decimal_not_rounded_to_correct_answer(self):
        f = self.fixture
        raw = f["raw_response"].replace(
            '"answer":90', '"answer":90.00000000000000000000000000000000001'
        )
        score = score_response(raw, f["world"], f["query"])
        self.assertEqual(score["value_correct"], 0)
        self.assertEqual(score["J"], 0)

    def test_invalid_original_never_repaired(self):
        f = self.fixture
        malformed = [
            f["raw_response"][:-1],
            f["raw_response"] + " trailing",
            f["raw_response"] + "{}",
            f["raw_response"].replace('"answer":90', '"answer":1,"answer":90'),
            f["raw_response"].replace('"selected":', '"selected":[],"selected":'),
            f["raw_response"].replace('"value":84', '"value":84,"value":84'),
        ]
        for raw in malformed:
            with self.subTest(raw=raw[-100:]):
                with self.assertRaises(ValueError):
                    lossless_legacy_projection(raw)
                score = score_response(raw, f["world"], f["query"])
                self.assertEqual((score["V"], score["S"], score["J"]), (0, "UNKNOWN", 0))
                self.assertIsNone(score["J_legacy_projection"])

    def test_native_parity_all_families_and_extensions(self):
        for task in make_bank("dev", 1, 2026102502):
            world, query = task["world"], task["query"]
            gold = evaluate_a(world, query)
            obj = {
                "evidence": canonical_evidence(world, query),
                "selected": gold["selected"],
                "answer": encode(gold["answer"]),
            }
            for mutation in ("none", "wrong_answer", "wrong_fact", "missing_evidence"):
                changed = copy.deepcopy(obj)
                if mutation == "wrong_answer":
                    changed["answer"] = "99999"
                elif mutation == "wrong_fact":
                    changed["evidence"][0]["value"] = 11
                elif mutation == "missing_evidence":
                    changed["evidence"] = []
                raw = json.dumps(changed)
                score = score_response(raw, world, query)
                expected = legacy.score(lossless_legacy_projection(raw), world, query)
                for key in (
                    "value_correct",
                    "value_parseable",
                    "certificate",
                    "certificate_truth",
                    "certificate_sufficient",
                    "finish",
                    "J",
                ):
                    self.assertEqual(score[key], expected[key], (task["qid"], mutation, key))

    def test_extraction_has_no_gold_dependency(self):
        f = self.fixture
        before = parse_response(f["raw_response"])
        changed = copy.deepcopy(f["world"])
        changed["series"]["Alpha"].reverse()
        score_response(f["raw_response"], changed, f["query"])
        self.assertEqual(parse_response(f["raw_response"]), before)

    def test_natural_work_uses_native_scoring_without_selected(self):
        f = self.fixture
        raw = "Sum the values. <final>90</final>"
        score = score_response(raw, f["world"], f["query"], protocol="natural_work")
        expected = legacy.score(
            raw, f["world"], f["query"], protocol="natural_work", finish_reason="final_tag"
        )
        self.assertEqual(score["value_correct"], expected["value_correct"])
        self.assertEqual(score["S"], "NOT_APPLICABLE")
        self.assertIsNone(score["selected_span"])

    def test_truncation_and_unknown_query_field(self):
        f = self.fixture
        score = score_response(f["raw_response"], f["world"], f["query"], finished=False)
        self.assertEqual((score["finish"], score["J"]), (0, 0))
        with self.assertRaises(ValueError):
            score_response(f["raw_response"], f["world"], dict(f["query"], surprise=True))

    def test_natural_work_v2_alias_preserves_native_score_and_finish(self):
        f = self.fixture
        for raw, finished in (
            ("Sum the values. <final>90</final>", True),
            ("Sum the values. <final>89</final>", True),
            ("Sum the values. <final>90", False),
        ):
            with self.subTest(raw=raw, finished=finished):
                direct = score_response(
                    raw, f["world"], f["query"], finished=finished, protocol="natural_work"
                )
                alias = score_response(
                    raw, f["world"], f["query"], finished=finished, protocol="natural_work_v2"
                )
                self.assertEqual(alias.pop("protocol"), "natural_work_v2")
                direct.pop("protocol")
                self.assertEqual(alias, direct)
                self.assertEqual(alias["finish"], int(finished))
                self.assertEqual(alias["S"], "NOT_APPLICABLE")


if __name__ == "__main__":
    unittest.main()
