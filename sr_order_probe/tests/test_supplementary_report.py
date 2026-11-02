import importlib.util
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("supplementary_report", ROOT / "scripts/supplementary_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def scores(unknown=False):
    return [dict(request_id=f"{p}/{c}/{i}", parent=p, condition=c,
                 S=None if unknown else 1, A_parseable=1, parse_error=None,
                 rank_order_known=1, rank_values_true=1)
            for p in report.PARENTS for c in report.CONDITIONS for i in range(4)]


class SupplementaryReportTests(unittest.TestCase):
    def test_stratification_uniqueness_and_order_independence(self):
        rows = scores()
        selected = report.select_audit(rows)
        self.assertEqual(len(selected), 72)
        self.assertEqual(len({r["request_id"] for r in selected}), 72)
        self.assertEqual(selected, report.select_audit(list(reversed(rows))))
        strata = Counter("/".join(r["request_id"].split("/")[:2]) for r in selected[:54])
        self.assertEqual(set(strata.values()), {2})
        self.assertEqual(len(strata), 27)

    def test_exception_priority_uses_unknown_first(self):
        selected = report.select_audit(scores(unknown=True))
        self.assertEqual({r["selection_reason"] for r in selected[54:]}, {"SELECTED_UNKNOWN"})

    def test_unknown_denominator_not_dropped(self):
        result = report.ternary_counts([{"B": 1}, {"B": 0}, {"B": None}], "B")
        self.assertEqual(result["B_confirmed_rate"], 1 / 3)
        self.assertEqual(result["B_pass_given_known"], 0.5)
        self.assertEqual(result["B_unknown"], 1)

    def test_raw_tokens_are_counted_without_retokenizing(self):
        row = dict(prompt_input_ids=[11, 12, 13], suffix_token_ids=[14, 15], generated_tokens=2,
                   prefill_boundary=dict(base_prompt_token_count=1, prefill_token_ids=[12, 13]),
                   generation_seconds=1.5)
        self.assertEqual(report.token_cost(row), dict(input_tokens=3, forced_prefix_tokens=2,
                         context_tokens_before_prefix=1, sampled_tokens=2, generation_seconds=1.5))
        row["prefill_boundary"]["prefill_token_ids"] = [99, 13]
        with self.assertRaises(ValueError):
            report.token_cost(row)

    def test_missing_timing_remains_missing(self):
        result = report.number_summary([None, None])
        self.assertEqual(result["missing"], 2)
        self.assertIsNone(result["total"])

    def test_duplicate_request_is_rejected(self):
        rows = scores()
        with self.assertRaises(ValueError):
            report.select_audit(rows + [rows[0]])


if __name__ == "__main__":
    unittest.main()
