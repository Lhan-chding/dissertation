import itertools
import json
import math
from pathlib import Path
import unittest
from contracts import *

ROOT = Path(__file__).resolve().parents[1]


def star():
    return PublicTask('cross_example', (3,46,47,32),
                      ((1,0,0,1),(0,1,0,1),(0,0,1,1)),(35,78,55))


def trend():
    return PublicTask('trend_example',(45,83,37,33),
                      ((1,-2,1,0),(0,1,-2,1)),(0,0))


class ParseAndVerification(unittest.TestCase):
    def test_strict_ints(self): self.assertEqual(parse_vector(' [3,46,23,32] '),(3,46,23,32))
    def test_rejects_bool(self):
        with self.assertRaises(ValueError): parse_vector('[true,1,2,3]')
    def test_rejects_float(self):
        with self.assertRaises(ValueError): parse_vector('[1.0,1,2,3]')
    def test_rejects_prose(self):
        with self.assertRaises(ValueError): parse_vector('Answer: [1,2,3,4]')
    def test_rejects_extra_json(self):
        with self.assertRaises(ValueError): parse_vector('[1,2,3,4] [5,6,7,8]')
    def test_out_of_range_is_parseable_but_invalid(self):
        y=parse_vector('[45,83,121,159]')
        self.assertTrue(relations_hold(trend(),y))
        self.assertFalse(public_verifier(trend(),y))
    def test_all_permutation_roundtrips(self):
        y=(3,46,23,32)
        for p in itertools.permutations(range(4)):
            self.assertEqual(canonicalize([y[j] for j in p],p),y)
    def test_duplicate_order_rejected(self):
        with self.assertRaises(ValueError): canonicalize([1,2,3,4],[0,1,1,3])
    def test_unique_star(self): self.assertEqual(solve_single_edit(star()),[(3,46,23,32)])
    def test_unique_trend(self): self.assertEqual(solve_single_edit(trend()),[(45,41,37,33)])
    def test_copy_not_validated(self): self.assertFalse(public_verifier(trend(),trend().observed))
    def test_self_consistent_rewrite_not_validated(self):
        y=(45,61,77,93)
        self.assertTrue(relations_hold(trend(),y)); self.assertFalse(public_verifier(trend(),y))
    def test_B1_solution_and_bidirectional_equivalence(self):
        a=star(); b=transform_b1(a)
        self.assertEqual(solve_single_edit(a),solve_single_edit(b))
        for center in range(100):
            y=(35-center,78-center,55-center,center)
            self.assertEqual(relations_hold(a,y),relations_hold(b,y))
        for j in range(4):
            for value in range(100):
                y=list(a.observed);y[j]=value
                self.assertEqual(public_verifier(a,y),public_verifier(b,y))
    def test_B1_partial_score_not_invariant(self):
        y=(4,47,24,32)
        self.assertEqual(relation_score(star(),y),0)
        self.assertAlmostEqual(relation_score(transform_b1(star()),y),2/3)
    def test_B1_no_wrong_family(self):
        with self.assertRaises(ValueError): transform_b1(trend())
    def test_alias_signatures_not_exclusive(self):
        y=(9,10,11,64)
        s=program_signature(y,y,(9,10,11,12),(-95,-42,11,64),y)
        self.assertTrue(s['copy'] and s['V2']); self.assertFalse(s['distinct_program_error'])
    def test_truth_program_match_is_not_error(self):
        y=(3,46,23,32)
        self.assertFalse(program_signature(y,star().observed,y,y,y)['distinct_program_error'])


class SamplingAndBudgets(unittest.TestCase):
    def test_pass1(self): self.assertAlmostEqual(pass_at_k(16,4,1),.25)
    def test_pass_all_draws(self): self.assertEqual(pass_at_k(16,1,16),1.)
    def test_zero_and_all(self):
        self.assertEqual(pass_at_k(16,0,8),0);self.assertEqual(pass_at_k(16,16,8),1)
    def test_repeated_protocol_unbiased_formula(self):
        self.assertAlmostEqual(pass_at_k(16,4,2),1-(12*11)/(16*15))
        self.assertNotAlmostEqual(pass_at_k(16,4,2),1-(12/16)**2)
    def test_mixed_independent_formula(self): self.assertAlmostEqual(mixed_two_calls(16,4,16,8),.625)
    def test_mixed_gain_can_be_negative(self): self.assertLess(replacement_gain(.8,.2),0)
    def test_gain_identity(self):
        for p,q in itertools.product([0,.1,.5,.9,1],repeat=2):
            self.assertAlmostEqual(replacement_gain(p,q),(1-(1-p)*(1-q))-(1-(1-p)**2))
    def test_same_means_complementarity(self):
        p=[1.,0.];q=[0.,1.]
        self.assertEqual(sum(p)/2,sum(q)/2)
        self.assertEqual(sum(1-(1-a)*(1-b) for a,b in zip(p,q))/2,1)
        self.assertEqual(sum(1-(1-a)**2 for a in p)/2,.5)
    def test_valid_mix_budget(self):
        streams={'O0':[False]*16,'L11':[True]+[False]*15}
        self.assertTrue(verified_budget_success(streams,{'O0':8,'L11':8},16))
    def test_sixteen_plus_eight_is_not_sixteen(self):
        with self.assertRaises(ValueError): verified_budget_success({'O0':[False]*16,'L11':[False]*16},{'O0':16,'L11':8},16)
    def test_mixed_and_original_not_nested(self):
        s={'O0':[False]*15+[True],'L11':[False]*16}
        self.assertTrue(verified_budget_success(s,{'O0':16},16))
        self.assertFalse(verified_budget_success(s,{'O0':8,'L11':8},16))
    def test_missing_draws_rejected(self):
        with self.assertRaises(ValueError): verified_budget_success({'O0':[True]*8},{'O0':16},16)
    def test_seed_role_protocol_separation(self):
        args=('S96','T_train','a','O0','discovery',0,0)
        s=request_seed(*args);self.assertEqual(s,request_seed(*args))
        for index,replacement in [(0,'REP96'),(1,'E_test'),(3,'L11'),(4,'evaluation'),(5,1),(6,1)]:
            vals=list(args);vals[index]=replacement
            self.assertNotEqual(s,request_seed(*vals))
    def test_zero_count_uncertainty(self):
        lo,hi=wilson_interval(0,16);self.assertAlmostEqual(lo,0);self.assertGreater(hi,.1)
    def test_all_count_uncertainty(self):
        lo,hi=wilson_interval(16,16);self.assertLess(lo,.9);self.assertAlmostEqual(hi,1)
    def test_wilson_count_validation(self):
        with self.assertRaises(ValueError): wilson_interval(1,0)


class DataAndLoss(unittest.TestCase):
    def test_finite_trend_capacity(self):
        for width,ordered,orbits in [(50,784,392),(100,3234,1617)]:
            worlds=trend_worlds(width)
            self.assertEqual(len(worlds),ordered)
            self.assertEqual(len({conservative_truth_orbit(y) for y in worlds}),orbits)
    def test_permuted_truth_has_same_orbit(self):
        self.assertEqual(conservative_truth_orbit([1,2,3,4]),conservative_truth_orbit([4,1,3,2]))
    def test_training_view_dedup_and_canonical(self):
        rec={'task_id':'i','O0_prompt':'prompt','canonical':[3,46,23,32],'role':'T_train','verified':True}
        self.assertEqual(training_view([rec,rec]),[('i','prompt','[3,46,23,32]')])
    def test_eval_rejected_even_verified(self):
        for role in ['E_test','V_selection','P','D','U22']:
            with self.assertRaises(ValueError):
                training_view([{'task_id':'i','O0_prompt':'p','canonical':[1,2,3,4],'role':role,'verified':True}])
    def test_unverified_self_target_rejected(self):
        with self.assertRaises(ValueError): training_view([{'task_id':'i','role':'T_train','verified':False}])
    def test_solver_self_identical_view(self):
        a={'task_id':'i','O0_prompt':'p','canonical':list(solve_single_edit(star())[0]),'role':'T_train','verified':True}
        y=canonicalize(parse_vector('[32,23,46,3]'),[3,2,1,0])
        b={**a,'canonical':list(y),'source':'L11'}
        self.assertTrue(public_verifier(star(),y));self.assertEqual(training_view([a]),training_view([b]))
    def test_alias_requires_same_parent_seed_and_settings(self):
        view=[('i','p','[1,2,3,4]')]; cfg={'lr':1e-5}; replay=[('r','p','[2,3,4,5]')]
        k=alias_key('S96',0,view,replay,cfg)
        self.assertEqual(k,alias_key('S96',0,list(view),replay,cfg))
        self.assertNotEqual(k,alias_key('S96',1,view,replay,cfg))
        self.assertNotEqual(k,alias_key('REP96',0,view,replay,cfg))
        self.assertNotEqual(k,alias_key('S96',0,view,replay,{'lr':2e-5}))
    def test_target_mask_and_shift(self):
        row=completion_batch_row([101,102,103],[11,12],99,0,8)
        self.assertEqual(row['labels'],[-100,-100,-100,11,12,99,-100,-100])
        self.assertEqual(row['prediction_positions'],[2,3,4])
        self.assertEqual(row['completion_length'],3)
    def test_eos_equal_pad_not_dropped(self):
        row=completion_batch_row([1],[2],9,9,5)
        self.assertEqual(row['labels'],[-100,2,9,-100,-100])
        self.assertEqual(row['attention_mask'],[1,1,1,0,0])
    def test_rejects_duplicate_eos(self):
        with self.assertRaises(ValueError): completion_batch_row([1],[2,9],9,0,5)
    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError): completion_batch_row([1,2,3],[4,5],9,0,4)
    def test_weighted_loss(self): self.assertAlmostEqual(weighted_sft_loss([2]*12,[4]*4),2.5)
    def test_replay_not_renormalized(self): self.assertAlmostEqual(weighted_sft_loss([],[4]*4,True),1)
    def test_microbatch_weight_additivity(self):
        f=list(range(1,13));r=[2,3,4,5]
        pieces=[sum(f[k:k+4])/16 for k in range(0,12,4)]+[sum(r)/16]
        self.assertAlmostEqual(sum(pieces),weighted_sft_loss(f,r))
    def test_step_indexed_warmup(self):
        self.assertAlmostEqual(lr_at_update(1),1e-5/8)
        self.assertEqual(lr_at_update(8),1e-5);self.assertEqual(lr_at_update(256),1e-5)


class Configuration(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.c=json.loads((ROOT/'protocol.json').read_text())
    def test_workload_counts(self):
        w=workload(self.c)
        self.assertEqual(w['total_generated_sequences'],205504)
        self.assertEqual(w['SFT_updates'],5120);self.assertEqual(w['RL_updates'],128)
        self.assertEqual(w['SFT_processed_target_sequences'],75776)
    def test_unique_jobs(self):
        jobs=planned_jobs(self.c)
        self.assertEqual(len(jobs),24)
        self.assertEqual(len({(j['parent'],j['repeat'],j['arm']) for j in jobs}),24)
    def test_balanced_cells_feasible_by_count(self):
        for split in ['T_train','V_selection','E_test']:
            per_family=self.c['data']['new_task_counts'][split]//2
            self.assertEqual(per_family%16,0)
        self.assertEqual(sum(self.c['data']['new_task_counts'][s]//2 for s in ['T_train','V_selection','E_test']),432)
    def test_main_design_boundaries(self):
        self.assertFalse(self.c['scope']['importance_weighted_off_policy_RL'])
        self.assertFalse(self.c['data']['historical_ID_claim'])
        self.assertFalse(self.c['model']['lora']['reinitialize_lora'])
        self.assertEqual(self.c['discovery']['MIX_allocation'],{'O0':8,'family_companion':8})
    def test_empty_return_parent_and_finite_run(self):
        self.assertIn('RETURN_PARENT',self.c['gates']['empty_discovered_set'])
        self.assertFalse(self.c['sft']['auto_extend_after_negative_result'])
        self.assertEqual(self.c['evaluation']['main_endpoint'] if 'main_endpoint' in self.c['evaluation'] else self.c['sft']['main_endpoint'],256)


if __name__ == '__main__':
    unittest.main(verbosity=2)
