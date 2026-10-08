from __future__ import annotations
from fractions import Fraction as F
import json
import random
import unittest
from reference.contracts import (
    rational, operate, parse_response, score, reconstruct_six, collision_audit,
    mixed_probability, beta_mixed_predictive, QUOTAS, absolute_utility,
    assert_split_integrity, check_permission, check_budget, planned_counts,
)


class ContractTests(unittest.TestCase):
    def test_01_exact_six_cells(self):
        cases = [("111",[30,20],10),("100",[30,20],11),
                 ("011",[35,25],10),("010",[35,20],15),
                 ("001",[35,20],10),("000",[35,20],11)]
        for cell, r, a in cases:
            s = score(json.dumps({"readings":r,"answer":a}), [30,20], "difference")
            self.assertEqual(s["cell_exact"], cell)
    def test_02_six_cell_reconstruction(self):
        keys=["111","100","011","010","001","000"]
        rng=random.Random(20261008)
        for _ in range(200):
            counts=[rng.randrange(1,50) for _ in keys]; total=sum(counts)
            t=dict(zip(keys,[F(x,total) for x in counts]))
            p=t["111"]+t["100"]; a=t["111"]+t["011"]+t["001"]
            c=t["111"]+t["011"]+t["010"]
            self.assertEqual(reconstruct_six(p,a,c,t["111"],t["001"]),t)
    def test_03_incompatible_marginals_rejected(self):
        with self.assertRaises(ValueError): reconstruct_six(F(0),F(0),F(0),F(1),F(0))
    def test_04_executor_C_is_true(self):
        for r in ([30,20],[35,25],[35,20]):
            y=operate(tuple(map(F,r)),"difference")
            s=score(json.dumps({"readings":r,"answer":int(y)}),[30,20],"difference")
            self.assertTrue(s["C"])
            self.assertFalse(s["P"] and not s["A"])
    def test_05_decimal_equality(self):
        s=score('{"readings":[30.0,20.00],"answer":10.000}',[30,20],"difference")
        self.assertEqual(s["cell_exact"],"111")
    def test_06_duplicate_keys_rejected(self):
        p=parse_response('{"readings":[30,20],"answer":11,"answer":10}',2)
        self.assertFalse(p.L)
    def test_07_boolean_not_number(self):
        p=parse_response('{"readings":[true,20],"answer":10}',2)
        self.assertFalse(p.L_P); self.assertTrue(p.L_A)
    def test_08_nonfinite_rejected(self):
        p=parse_response('{"readings":[NaN,20],"answer":10}',2)
        self.assertFalse(p.L)
    def test_09_missing_answer_is_not_filled(self):
        s=score('{"readings":[30,20]}',[30,20],"difference")
        self.assertTrue(s["P_full"]); self.assertFalse(s["A_full"])
        self.assertIsNone(s["C"]); self.assertIsNone(s["cell_exact"])
    def test_10_missing_readings_preserves_answer(self):
        s=score('{"answer":10}',[30,20],"difference")
        self.assertTrue(s["A_full"]); self.assertIsNone(s["P"])
    def test_11_wrong_arity(self):
        p=parse_response('{"readings":[30,20,10],"answer":10}',2)
        self.assertFalse(p.L_P)
    def test_12_reversed_order_scored_not_deleted(self):
        p=parse_response('{"answer":10,"readings":[30,20]}',2)
        self.assertTrue(p.L); self.assertFalse(p.field_order_ok)
    def test_13_extra_fields_scored_schema_false(self):
        p=parse_response('{"readings":[30,20],"answer":10,"note":"x"}',2)
        self.assertTrue(p.L); self.assertFalse(p.schema_ok)
    def test_14_code_block_not_salvaged(self):
        p=parse_response('```json\n{"readings":[30,20],"answer":10}\n```',2)
        self.assertFalse(p.L)
    def test_15_outside_axis_not_deleted(self):
        s=score('{"readings":[130,120],"answer":10}',[30,20],"difference",axis=(0,100))
        self.assertTrue(s["L"]); self.assertTrue(s["outside_axis"])
        self.assertEqual(s["cell_exact"],"011")
    def test_16_zero_error_shape_is_missing(self):
        s=score('{"readings":[30,20],"answer":10}',[30,20],"difference")
        self.assertIsNone(s["S"]); self.assertFalse(s["near_miss"])
    def test_17_common_error_shape_one(self):
        s=score('{"readings":[35,25],"answer":10}',[30,20],"difference")
        self.assertEqual(s["S"],1); self.assertEqual(s["eps_P"],0)
    def test_18_opposite_error_shape_zero(self):
        s=score('{"readings":[35,15],"answer":20}',[30,20],"difference")
        self.assertEqual(s["S"],0)
    def test_19_near_miss_whole_vector(self):
        s=score('{"readings":[30.5,25],"answer":5.5}',[30,20],"difference")
        self.assertFalse(s["near_miss"])
    def test_20_near_miss_boundary(self):
        s=score('{"readings":[31,19],"answer":12}',[30,20],"difference")
        self.assertTrue(s["near_miss"])
    def test_21_tolerance_allows_101(self):
        s=score('{"readings":[30.4,20.4],"answer":50}',[30,20],"sum")
        self.assertEqual(s["cell_tolerance"],"101")
    def test_22_tolerance_allows_110(self):
        s=score('{"readings":[30.4,20.4],"answer":50.8}',[30,20],"sum")
        self.assertEqual(s["cell_tolerance"],"110")
    def test_23_residual_cancellation(self):
        s=score('{"readings":[35,20],"answer":10}',[30,20],"difference")
        self.assertEqual((s["eps_P"],s["eps_C"],s["eps_O"]),(5,-5,0))
    def test_24_sum_translation(self):
        self.assertEqual(operate([F(35),F(25)],"sum")-operate([F(30),F(20)],"sum"),10)
    def test_25_range_translation_with_three_values(self):
        self.assertEqual(operate([F(35),F(25),F(30)],"range"),operate([F(30),F(20),F(25)],"range"))
    def test_26_range_changed_extrema(self):
        s=score('{"readings":[20,40,25],"answer":20}',[30,20,25],"range")
        self.assertEqual(s["eps_P"],10)
    def test_27_delta_positive(self):
        with self.assertRaises(ValueError): score('{"readings":[30,20],"answer":10}',[30,20],"difference",0)
    def test_28_different_required_counts_rejected(self):
        with self.assertRaises(ValueError): operate([F(1),F(2),F(3)],"difference")
    def test_29_collision_estimate_can_be_negative(self):
        x=[(F(0),F(0)),(F(1),F(1))]
        self.assertEqual(collision_audit(x,x)["collision_contrast"],F(-1,2))
    def test_30_stable_wrong_not_correct(self):
        wrong=[(F(35),F(25))]*4
        self.assertEqual(collision_audit(wrong,wrong)["C12"],1)
        self.assertNotEqual(wrong[0],(F(30),F(20)))
    def test_31_missing_stability_denominator(self):
        s=collision_audit([(F(1),F(2))],[(F(1),F(2))]*4)
        self.assertIsNone(s["C11"]); self.assertIsNone(s["collision_contrast"])
    def test_32_stability_different_arity(self):
        with self.assertRaises(ValueError): collision_audit([(F(1),F(2))],[(F(1),F(2),F(3))])
    def test_33_collision_population_identity(self):
        p=[F(1,4),F(3,4)]; q=[F(1,2),F(1,2)]
        left=(sum(x*x for x in p)+sum(x*x for x in q))/2-sum(x*y for x,y in zip(p,q))
        self.assertEqual(left,sum((x-y)**2 for x,y in zip(p,q))/2)
    def test_34_support_group_one_zero(self):
        self.assertEqual(mixed_probability(F(1,3),1),0)
        self.assertEqual(beta_mixed_predictive(0,4,1),0)
    def test_35_zero_success_not_zero_posterior_support(self):
        self.assertEqual(mixed_probability(F(0),8),0)
        self.assertGreater(beta_mixed_predictive(0,4,8),0)
    def test_36_support_symmetric(self):
        self.assertEqual(beta_mixed_predictive(1,4,8),beta_mixed_predictive(3,4,8))
    def test_37_quota_marginals(self):
        for a,q in QUOTAS.items():
            self.assertEqual(sum(q),1)
            self.assertEqual(q[1]+q[3],F(3,4) if a=="aP" else F(1,2))
            self.assertEqual(q[2]+q[3],F(3,4) if a=="aC" else F(1,2))
    def test_38_minimum_complete_quota_block(self):
        for q in QUOTAS.values():
            self.assertTrue(all((48*x/6).denominator==1 for x in q))
    def test_39_no_train_zero_absolute_utility(self):
        self.assertEqual(absolute_utility([0],[0],[1],[1],1,1,0),0)
        self.assertLess(absolute_utility([F(1,10)],[F(-1,5)],[1],[1],1,0,0),0)
    def test_40_a0_not_automatically_safe(self):
        d_a0=F(-1,5); d_other=F(-1,10)
        self.assertGreater(d_other-d_a0,0); self.assertLess(d_other,0)
    def test_41_split_family_leak(self):
        with self.assertRaises(ValueError): assert_split_integrity([
            {"split":"probe","root_family_id":"r"},{"split":"eval","root_family_id":"r"}])
    def test_42_split_image_clone_leak(self):
        with self.assertRaises(ValueError): assert_split_integrity([
            {"split":"probe","root_family_id":"r1","image_hash":"h"},
            {"split":"eval","root_family_id":"r2","image_hash":"h"}])
    def test_43_equal_answer_not_leak(self):
        assert_split_integrity([{"split":"probe","root_family_id":"r1","answer":"10"},
                                {"split":"eval","root_family_id":"r2","answer":"10"}])
    def test_44_forbidden_phases(self):
        for stage in ["MM-DEV","MM-LOCK","SER-J23","MM-CAL"]:
            with self.assertRaises(PermissionError): check_permission(stage,trigger=True,prefreeze=True)
    def test_45_conditional_training_requires_trigger(self):
        with self.assertRaises(PermissionError): check_permission("BRIDGE",prefreeze=True)
        check_permission("BRIDGE",trigger=True,prefreeze=True)
    def test_46_missing_prefreeze_blocks_inference(self):
        with self.assertRaises(PermissionError): check_permission("MM-AUDIT")
    def test_47_budget_cannot_be_exceeded(self):
        with self.assertRaises(PermissionError): check_budget({"updates":63},{"updates":2},{"updates":64})
        check_budget({"updates":63},{"updates":1},{"updates":64})
    def test_48_negative_resource_rejected(self):
        with self.assertRaises(ValueError): check_budget({"updates":2},{"updates":-1},{"updates":64})
    def test_49_unregistered_resource_rejected(self):
        with self.assertRaises(ValueError): check_budget({}, {"unknown":1},{"updates":64})
    def test_50_normal_budget_with_bridge(self):
        s=planned_counts(True)
        self.assertEqual(s["normal_total_with_technical_allowance"],2864)
        self.assertEqual(s["optimizer_physical_updates"],24)
        self.assertEqual(s["future_dev_updates_not_authorized"],1088)
    def test_51_normal_budget_without_bridge(self):
        s=planned_counts(False)
        self.assertEqual(s["normal_total_with_technical_allowance"],2480)
        self.assertEqual(s["optimizer_physical_updates"],8)
    def test_52_max_technical_updates_within_generation_cap(self):
        for bridge in (True,False):
            s=planned_counts(bridge)
            remaining=64-(16 if bridge else 0)
            total=s["format_completions"]+1536+48+remaining*2*8
            self.assertLessEqual(total,4096)
    def test_53_geometric_nonredundancy_construction(self):
        states=([[(35,25),(25,15)]],[[(35,15),(25,25)]])
        vectors_u=[(35,25),(25,15)]; vectors_v=[(35,15),(25,25)]
        scores=[]
        for vectors in (vectors_u,vectors_v):
            ss=[score(json.dumps({"readings":list(v),"answer":11}),[30,20],"difference") for v in vectors]
            self.assertTrue(all(s["cell_exact"]=="000" for s in ss))
            scores.append(ss)
        for k in (0,1):
            self.assertEqual(sorted(s["e"][k] for s in scores[0]),sorted(s["e"][k] for s in scores[1]))
        self.assertEqual([s["S"] for s in scores[0]],[1,1])
        self.assertEqual([s["S"] for s in scores[1]],[0,0])
    def test_54_numeric_resource_guard_not_axis_clip(self):
        p=parse_response('{"readings":[1e1000,20],"answer":10}',2)
        self.assertFalse(p.L_P); self.assertTrue(p.L_A)
    def test_55_actual_range_not_assumed_extrema(self):
        v=[F(10),F(20),F(15)]; r=[F(30),F(20),F(15)]
        self.assertEqual(operate(r,"range")-operate(v,"range"),5)
        self.assertNotEqual((r[1]-v[1])-(r[0]-v[0]),5)

if __name__ == "__main__":
    unittest.main(verbosity=2)
