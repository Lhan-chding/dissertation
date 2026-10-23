import unittest

from sr_f4.c_diagnosis import summarize


def row(p="PASS", s="UNKNOWN", **overrides):
    normalization = dict(j=0, protocol_valid=True, finished=True, sufficient=True, defined=True)
    normalization.update(overrides)
    return dict(
        normalization=normalization,
        native_score=dict(value_correct=1),
        dimensions={
            k: dict(status=v, reason="test")
            for k, v in zip("PSOE", [p, s, "UNKNOWN", "UNKNOWN"], strict=True)
        },
    )


def group(rows, source="C_TRAIN_PROBE"):
    return dict(
        source=source,
        group_id="g",
        world_id="w",
        block="b0",
        labels=rows,
        select_vs_evid_changed=False,
    )


class DiagnosisTests(unittest.TestCase):
    def test_sufficiency_blocks_even_perfect_selection(self):
        result = summarize([group([row(s="PASS"), row(s="FAIL", sufficient=False)])])
        self.assertEqual(result["group_gate_reasons"], {"TOO_FEW_ELIGIBLE_FAILURES": 1})
        self.assertEqual(result["eligibility_only_world_upper_bound"], 0)
        self.assertEqual(result["counts"]["removed_by_sufficiency_answer_correct"], 1)

    def test_all_p_fail_has_no_select_increment_even_if_s_varies(self):
        result = summarize([group([row(p="FAIL"), row(p="FAIL")])])
        self.assertEqual(result["eligibility_only_world_upper_bound"], 1)
        self.assertEqual(result["fixed_p_arbitrary_s_world_upper_bound"], 0)

    def test_bound_is_not_observed_signal(self):
        result = summarize([group([row(), row()])])
        self.assertEqual(result["counts"]["fixed_p_arbitrary_s_group_upper_bound"], 1)
        self.assertEqual(result["counts"]["observed_changed_groups"], 0)
        self.assertEqual(result["group_gate_reasons"], {"ABSTAIN_UNKNOWN_IN_FAILURE_STRATUM": 1})

    def test_confirmation_source_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize([group([row()], source="B_CONFIRM")])
