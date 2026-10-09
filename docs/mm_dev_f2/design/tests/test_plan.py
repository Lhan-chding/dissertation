from __future__ import annotations
import json,sys,math,hashlib,unittest
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'reference'))
from plan_contract import *
from audited_scoring_contracts import score,parse_response,reconstruct_six

class TestScoringAndRewards(unittest.TestCase):
    def test_exact_both_correct(self): self.assertEqual(score('{"readings":[30,20],"answer":10}',[30,20],'difference')['cell_exact'],'111')
    def test_cancellation_consistent(self): self.assertEqual(score('{"readings":[35,25],"answer":10}',[30,20],'difference')['cell_exact'],'011')
    def test_answer_correct_but_inconsistent(self): self.assertEqual(score('{"readings":[35,20],"answer":10}',[30,20],'difference')['cell_exact'],'001')
    def test_readings_correct_answer_wrong(self): self.assertEqual(score('{"readings":[30,20],"answer":8}',[30,20],'difference')['cell_exact'],'100')
    def test_reading_missing_not_joint_zero(self): self.assertIsNone(score('{"answer":10}',[30,20],'difference')['cell_exact'])
    def test_reading_missing_keeps_answer(self): self.assertTrue(score('{"answer":10}',[30,20],'difference')['A_full'])
    def test_invalid_json_not_repaired(self): self.assertFalse(parse_response('```json\n{"readings":[30,20],"answer":10}\n```',2).L)
    def test_wrong_arity(self): self.assertFalse(parse_response('{"readings":[30],"answer":10}',2).L_P)
    def test_duplicate_keys(self): self.assertFalse(parse_response('{"readings":[30,20],"answer":10,"answer":20}',2).L)
    def test_nan(self): self.assertFalse(parse_response('{"readings":[30,NaN],"answer":10}',2).L)
    def test_reversed_order_not_deleted(self):
        p=parse_response('{"answer":10,"readings":[30,20]}',2);self.assertFalse(p.field_order_ok);self.assertTrue(p.L)
    def test_fractional_reading_reward(self): self.assertEqual(preparation_reward(False,[1,8,3],[1,2,3]),F(1,3))
    def test_missing_reading_reward(self): self.assertEqual(preparation_reward(True,None,[1,2]),F(1,2))
    def test_all_correct_reward(self): self.assertEqual(preparation_reward(True,[1,2],[1,2]),1)
    def test_missing_answer_still_read_reward(self): self.assertEqual(preparation_reward(False,[1,2],[1,2]),F(1,2))
    def test_zero_group(self): self.assertEqual(advantages([0]*8),[0.]*8)
    def test_all_one_group(self): self.assertEqual(advantages([1]*8),[0.]*8)
    def test_fractional_advantages(self): self.assertGreater(advantages([F(1,3)]*4+[F(2,3)]*4)[-1],0)
    def test_advantage_centering(self): self.assertAlmostEqual(sum(advantages([0,1,0,1,0,0,1,1])),0)
    def test_constant_reading_affine_reward_equivalence(self):
        a=[0,1,1,0,1,1,1,0];ap=[.5*x+.5 for x in a]
        self.assertLess(max(abs(x-y) for x,y in zip(advantages(a),advantages(ap))),1e-6)
    def test_variable_reading_can_change_advantages(self):
        a=[1]*8;ap=[.5]*4+[1.]*4
        self.assertTrue(any(x!=y for x,y in zip(advantages(a),advantages(ap))))
    def test_zero_shape_missing(self): self.assertIsNone(shape([0,0,0]))
    def test_single_coordinate_shape(self): self.assertEqual(shape([0,1,0]),F(1,3))
    def test_two_coordinate_shape(self): self.assertEqual(shape([0,1,1]),F(2,3))
    def test_shared_shape_one(self): self.assertEqual(shape([2,2]),1)
    def test_opposite_shape_zero(self): self.assertEqual(shape([2,-2]),0)
    def test_range_shift_invariance(self): self.assertEqual(max([30,20,5])-min([30,20,5]),max([37,27,12])-min([37,27,12]))
    def test_exact_error_sum(self):
        s=score('{"readings":[35,20],"answer":10}',[30,20],'difference');self.assertEqual(s['eps_P']+s['eps_C'],s['eps_O'])
    def test_tolerance_not_six_cell_algebra(self):
        s=score('{"readings":[30.4,20.4],"answer":51}',[30,20],'sum');self.assertNotEqual(s['cell_exact'],s['cell_tolerance'])

class TestScheduleAndArtifacts(unittest.TestCase):
    def setUp(self): self.cfg=json.loads((ROOT/'config/MM_DEV_F2.json').read_text())
    def test_no_gpu_time_cap(self): self.assertIsNone(self.cfg['resource']['max_allocated_gpu_hours']);self.assertFalse(self.cfg['resource']['gpu_hour_triggered_stop'])
    def test_synthetic_gate_cannot_bypass_missing_gradient(self):
        self.assertIn('all actual reward groups are constant',self.cfg['engineering_gate']['synthetic_stress_if'])
        self.assertIn('technical blocker',self.cfg['engineering_gate']['nonconstant_reward_but_no_policy_gradient'])
    def test_crossfit_split_is_complete_and_disjoint(self):
        halves=self.cfg['statistics']['crossfit_root_halves']
        a,b=halves['half0_indices'],halves['half1_indices']
        self.assertEqual((len(a),len(b)),(64,64));self.assertFalse(set(a)&set(b));self.assertEqual(set(a+b),set(range(128)))
    def test_model_fixed(self): self.assertEqual(self.cfg['model']['model_id'],'Qwen/Qwen3.5-9B')
    def test_train_distribution_raw(self):
        s=self.cfg['sampling']['train'];self.assertEqual((s['temperature'],s['top_p'],s['top_k']),(1,1,0))
    def test_eval_audit_profile(self):
        s=self.cfg['sampling']['probe_eval'];self.assertEqual((s['temperature'],s['top_p'],s['K']),(.7,.9,4))
    def test_run_count(self): self.assertEqual(len(run_matrix()),34)
    def test_preparation_count(self): self.assertEqual(sum(r['phase']=='PREP' for r in run_matrix()),4)
    def test_response_count(self): self.assertEqual(sum(r['phase']=='CONTINUE' for r in run_matrix()),30)
    def test_no_identity_as_predictor(self): self.assertIn('seed IDs',self.cfg['prediction_analysis']['prohibited_inputs'])
    def test_budget(self): self.assertEqual(allocation_counts()['scientific_completions'],669696)
    def test_updates(self): self.assertEqual(sum(r['steps'] for r in run_matrix()),1088)
    def test_microbatch(self): self.assertEqual(self.cfg['training']['gradient_accumulation_sequences'],24*8)
    def test_seed_reproducible(self): self.assertEqual(seed('x',1),seed('x',1));self.assertNotEqual(seed('x',1),seed('x',2))
    def test_seed_63bit(self): self.assertTrue(0<=seed('long')<2**63)
    def test_schedule_all_steps_balanced_strata(self):
        for a in CELL_COUNTS:
            x=slots('CONTINUE',a,0,64)
            for step in range(1,33): self.assertEqual(sorted(Counter((r['chart_type'],r['operation']) for r in x if r['logical_step']==step).values()),[4]*6)
    def test_schedule_exact_quotas(self):
        for a in CELL_COUNTS:
            x=slots('CONTINUE',a,0,64)
            for c,o in STRATA:
                n=Counter(r['cell'] for r in x if (r['chart_type'],r['operation'])==(c,o))
                self.assertEqual([n[q] for q in CELLS],[v*16 for v in CELL_COUNTS[a]])
    def test_schedule_two_step_blocks(self):
        for a in CELL_COUNTS:
            x=slots('CONTINUE',a,1,64)
            for b in range(16):
                for c,o in STRATA:
                    n=Counter(r['cell'] for r in x if (r['chart_type'],r['operation'])==(c,o) and r['logical_step'] in [2*b+1,2*b+2]);self.assertEqual([n[q] for q in CELLS],list(CELL_COUNTS[a]))
    def test_schedule_no_duplicate_questions(self):
        for a in CELL_COUNTS: self.assertEqual(len(set(r['question_id'] for r in slots('CONTINUE',a,1,64))),768)
    def test_prep_full_pool(self):
        x=slots('PREP','a0',0,32);self.assertEqual(set(r['root_index'] for r in x),set(range(32)))
    def test_engine_short_schedule(self): self.assertEqual(len(slots('ENGINE_F2','a0',0,4,steps=4)),96)
    def test_undersized_pool_rejected(self):
        with self.assertRaises(ValueError): slots('CONTINUE','aP',0,32)
    def test_saved_schedules_reproduce(self):
        for repeat in (0,1):
            for a in CELL_COUNTS:
                saved=[json.loads(x) for x in (ROOT/f'config/schedules/CONT_f{repeat}_{a}.jsonl').read_text().splitlines()];self.assertEqual(saved,slots('CONTINUE',a,repeat,64))
    def test_numeric_sources_full(self):
        self.assertEqual(len((ROOT/'config/numeric_sources.jsonl').read_text().splitlines()),584)
    def test_numeric_source_no_pair_duplicate(self):
        src=[json.loads(x) for x in (ROOT/'config/numeric_sources.jsonl').read_text().splitlines()];keys=[(s['numeric_level'],tuple(s['truth_targets'][:2])) for s in src];self.assertEqual(len(keys),len(set(keys)))
    def test_numeric_source_no_range_set_duplicate(self):
        src=[json.loads(x) for x in (ROOT/'config/numeric_sources.jsonl').read_text().splitlines()];keys=[tuple(sorted(s['truth_targets'])) for s in src];self.assertEqual(len(keys),len(set(keys)))
    def test_numeric_domain_and_gaps(self):
        for s in map(json.loads,(ROOT/'config/numeric_sources.jsonl').read_text().splitlines()):
            v=s['truth_targets'];self.assertTrue(all(10<=x<=89 for x in v));self.assertGreater(v[0],v[1]);self.assertGreaterEqual(min(abs(x-y) for i,x in enumerate(v) for y in v[i+1:]),5)
    def test_numeric_difficulty(self):
        for s in map(json.loads,(ROOT/'config/numeric_sources.jsonl').read_text().splitlines()):
            v=s['truth_targets'];flags=[v[0]%10+v[1]%10>=10,v[0]%10<v[1]%10,max(v)%10<min(v)%10];self.assertEqual(flags,[s['numeric_level']=='high']*3)
    def test_no_train_anchor(self): self.assertEqual(utility([0.]*6,[0.]*6)['U'],0)
    def test_penalty_positive_part_before_averaging(self): self.assertAlmostEqual(utility([0.]*6,[-.06,.06,0,0,0,0])['I'],.01)
    def test_feasible_uses_absolute_start(self): self.assertFalse(utility([.1]*6,[-.03]*6)['feasible_point_estimate'])
    def test_pareto_gain(self): self.assertAlmostEqual(utility([.02]*6,[-.01]*6)['U'],.01)

class TestToyGradient(unittest.TestCase):
    def test_zero_loss_not_zero_gradient(self):
        import torch
        z=torch.zeros(2,dtype=torch.float64,requires_grad=True);ids=torch.tensor([0]*4+[1]*4);old=z.detach().log_softmax(0)[ids];cur=z.log_softmax(0)[ids];a=torch.tensor(advantages([0]*4+[1]*4),dtype=torch.float64);r=(cur-old).exp();loss=-torch.minimum(r*a,r.clamp(.8,1.2)*a).mean();self.assertAlmostEqual(loss.item(),0);loss.backward();self.assertGreater(z.grad.abs().max().item(),0)
    def test_zero_group_policy_gradient_zero(self):
        import torch
        z=torch.zeros(2,dtype=torch.float64,requires_grad=True);p=z.log_softmax(0);loss=-(p.exp()*torch.zeros(2)).sum();loss.backward();self.assertEqual(z.grad.abs().max().item(),0)
    def test_kl_nonnegative(self):
        for d in [-3,-1,-.1,0,.1,1,3]: self.assertGreaterEqual(math.expm1(d)-d,-1e-14)
    def test_kl_can_update_zero_reward_group(self):
        import torch
        z=torch.tensor([.7,.3],dtype=torch.float64).log().requires_grad_();cur=z.log_softmax(0);ref=torch.tensor([.5,.5],dtype=torch.float64).log();d=ref-cur;loss=(torch.expm1(d)-d).mean();loss.backward();self.assertGreater(z.grad.abs().max().item(),0)
    def test_microbatch_equal_sequence_normalization(self):
        import torch
        x=torch.tensor(1.,dtype=torch.float64,requires_grad=True);ls=[(x*torch.arange(1,n+1,dtype=torch.float64)).mean() for n in [2,3,7]];sum(ls).div(3).backward();g=x.grad.item();x.grad=None
        for n in [2,3,7]: ((x*torch.arange(1,n+1,dtype=torch.float64)).mean()/3).backward()
        self.assertAlmostEqual(g,x.grad.item())

if __name__=='__main__': unittest.main()
