"""Run: python -m unittest discover -s reference -p 'test_reference_math.py' -v"""
import json
import math
from pathlib import Path
import unittest

from reference_math import (design_budget, fixed_panel_radius, group_statistics,
                            vector_comparison)

ROOT = Path(__file__).resolve().parents[1]


class ReferenceMathTests(unittest.TestCase):
    def assert_vectors_close(self, a, b, places=10):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=places)

    def test_exact_reward_vectors(self):
        self.assertEqual(group_statistics(list("XSWI"), auxiliary_weight=0)["rewards"], [2,0,0,0])
        self.assertEqual(group_statistics(list("XSWI"), auxiliary_weight=1)["rewards"], [3,1,1,0])

    def test_all_valid_invariance(self):
        cats = list("XXSSWWXX")
        baseline = group_statistics(cats, auxiliary_weight=0)["advantages"]
        for lam in [1e-6,1e-4,0.01,1]:
            self.assert_vectors_close(baseline, group_statistics(cats, auxiliary_weight=lam)["advantages"])

    def test_constant_groups_zero(self):
        for category in "XSWI":
            for lam in [0,1e-6,1]:
                self.assertEqual(group_statistics([category]*8, auxiliary_weight=lam)["advantages"], [0]*8)

    def test_no_x_off_zero(self):
        for cats in [list("WWWWIIII"), list("SSSSIIII"),list("WWWSSSII")]:
            self.assertEqual(group_statistics(cats, auxiliary_weight=1, policy="no_x_off")["advantages"],[0]*8)

    def test_no_x_on_with_x_same_as_joint(self):
        cats = list("XXXXWIII")
        self.assert_vectors_close(group_statistics(cats,auxiliary_weight=1)["advantages"],
                                  group_statistics(cats,auxiliary_weight=1,policy="no_x_off")["advantages"])

    def test_balanced_no_x_formula(self):
        for lam in [0,1e-6,1e-5,1e-4,2e-4,1e-3,0.01,1]:
            actual = group_statistics(list("WWWWIIII"),auxiliary_weight=lam)["advantages"]
            expected = 0.5*lam / math.sqrt(.25*lam*lam+1e-8)
            self.assert_vectors_close(actual,[expected]*4+[-expected]*4)

    def test_tiny_weight_not_fully_saturated(self):
        small = group_statistics(list("WWWWIIII"),auxiliary_weight=1e-6)["advantages"][0]
        mid = group_statistics(list("WWWWIIII"),auxiliary_weight=2e-4)["advantages"][0]
        big = group_statistics(list("WWWWIIII"),auxiliary_weight=1)["advantages"][0]
        self.assertLess(small,0.01)
        self.assertAlmostEqual(mid,1/math.sqrt(2))
        self.assertGreater(big,.9999)

    def test_derivative(self):
        cats = list("XXXXWIII")
        for lam in [.01,.2,1.]:
            h=1e-6
            right=group_statistics(cats,auxiliary_weight=lam+h)["advantages"]
            left=group_statistics(cats,auxiliary_weight=lam-h)["advantages"]
            numerical=[(a-b)/(2*h) for a,b in zip(right,left)]
            self.assert_vectors_close(numerical,group_statistics(cats,auxiliary_weight=lam)["d_advantage_d_requested_lambda"],places=7)

    def test_invalid_inputs(self):
        for kwargs in [dict(auxiliary_weight=True),dict(auxiliary_weight=-1),dict(auxiliary_weight=float('nan')),
                       dict(auxiliary_weight=1,epsilon=0),dict(auxiliary_weight=1,policy='independent')]:
            with self.assertRaises(ValueError):group_statistics(list("XSWI"),**kwargs)
        with self.assertRaises(ValueError):group_statistics(list("XZWQ"),auxiliary_weight=1)

    def test_original_bank_equivalence(self):
        snap=json.loads((ROOT/'reference/REPOSITORY_AND_EVIDENCE_SNAPSHOT.json').read_text())
        for bank in snap['selected_banks']:
            for group in bank['groups']:
                cats=[c for c in "XSWI" for _ in range(group['counts'][c])]
                if bank['bank_index'] in (3,4,5):
                    self.assert_vectors_close(group_statistics(cats,auxiliary_weight=0)['advantages'],
                                              group_statistics(cats,auxiliary_weight=1,policy='no_x_off')['advantages'])
                if bank['bank_index']==11:
                    self.assert_vectors_close(group_statistics(cats,auxiliary_weight=1)['advantages'],
                                              group_statistics(cats,auxiliary_weight=1,policy='no_x_off')['advantages'])

    def test_composite_has_distinct_advantage_treatments(self):
        groups=[list("WWWWIIII"),list("XXXXWIII"),list("XXXXXXXX"),list("XXXXXXXX")]
        def bank(lam,policy='joint'):
            return [v for g in groups for v in group_statistics(g,auxiliary_weight=lam,policy=policy)['advantages']]
        base,valid,off=bank(0),bank(1),bank(1,'no_x_off')
        self.assertNotEqual(base,off)
        self.assertNotEqual(valid,off)
        self.assert_vectors_close(off[:8],base[:8])
        self.assert_vectors_close(off[8:],valid[8:])

    def test_equal_norm_not_equal_vector(self):
        result=vector_comparison([1,0],[0,1])
        self.assertEqual(result['left_l2'],result['right_l2'])
        self.assertAlmostEqual(result['difference_l2'],math.sqrt(2))
        self.assertEqual(result['cosine'],0)
        self.assertFalse(result['exact_equal'])

    def test_zero_vector_cosine_undefined(self):
        self.assertIsNone(vector_comparison([0,0],[0,0])['cosine'])

    def test_fixed_panel_bound(self):
        radius=fixed_panel_radius([1/48]*48,[16]*48)
        self.assertAlmostEqual(radius,math.sqrt(math.log(40)/(2*48*16)))
        self.assertGreater(fixed_panel_radius([1/48]*48,[16]*48,statements=20),radius)

    def test_budget(self):
        d=json.loads((ROOT/'configs/followup_design.json').read_text())
        got=design_budget(d)
        expected={'new_candidates':41,'baseline_replays':6,'candidate_and_replay_adam_calls':47,
                  'backward_sequence_calls':1504,'logical_direct_candidates':18,
                  'S1_direct_outputs_upper':13824,'S1_optional_fresh64_outputs_upper':55296,
                  'S2_new_runs':7,'S2_training_outputs':14336,
                  'S2_evaluation_outputs_per_new_run_upper':8896,'S2_evaluation_outputs_new_runs_upper':62272,
                  'historical_two_endpoint_interface_outputs_upper':1728}
        self.assertEqual(got,expected)
        self.assertFalse(d['execution']['allow_gpu'])
        self.assertFalse(d['execution']['allow_training'])
        self.assertFalse(d['execution']['automatic_sbatch_submission'])


if __name__=='__main__':
    unittest.main()
