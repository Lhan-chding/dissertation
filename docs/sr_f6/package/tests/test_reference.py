import copy, itertools, json, math, random, tempfile, unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from srf6_ref.core import *
from srf6_ref.measurement import *
from srf6_ref.credit import *
from srf6_ref.design import *
from srf6_ref.planning import *
from srf6_ref.statistics import *
from srf6_ref.gates import *

W={'categories':['Mug','Lamp','Desk','Vase'],'series':{'Alpha':[12,24,48,72],'Beta':[18,30,30,60]}}
Q={'family':'TOPK','selection_series':'Alpha','value':'Beta','k':2,'direction':'smallest','aggregate':'sum'}
def rows(js,ss,bs=None):
    if bs is None:bs=ss
    return [dict(J=j,S=s,B=b,V=1,finished=1) for j,s,b in zip(js,ss,bs)]

class ExactAndOracle(unittest.TestCase):
    def test_exact_fraction(self):self.assertEqual(exact('215/3'),Fraction(215,3))
    def test_no_float_tolerance(self):self.assertNotEqual(exact('71.666667'),Fraction(215,3))
    def test_finite_decimal(self):self.assertTrue(finite_decimal(Fraction(3,8)));self.assertFalse(finite_decimal(Fraction(1,3)))
    def test_no_bool(self):
        with self.assertRaises(ValueError):exact(True)
    def test_nonnumeric(self):
        for bad in ['nan','inf','1/0','__import__("os")']:
            with self.subTest(bad=bad),self.assertRaises((ValueError,ZeroDivisionError)):exact(bad)
    def test_dual_oracle_random(self):
        rng=random.Random(7)
        for i in range(150):
            w={'categories':list(LABELS[:6]),'series':{c:rng.sample(GRID,6) for c in ('Alpha','Beta')}}
            q=copy.deepcopy(Q);q.update(k=1+i%4,direction='largest' if i%2 else 'smallest',aggregate='mean' if i%3 else 'sum',value=['Alpha','Beta','diff'][i%3])
            self.assertEqual(oracle_a(w,q),oracle_b(w,q))
    def test_ties(self):
        w=copy.deepcopy(W);w['series']['Alpha']=[24,24,24,72]
        self.assertEqual(oracle_a(w,Q)['selected'],['Mug','Lamp']);self.assertEqual(oracle_a(w,Q),oracle_b(w,Q))
    def test_interval_endpoints(self):
        for a,b,p in itertools.product([True,False],repeat=3):
            q=dict(family='INTERVAL',start=0,end=3,include_start=a,include_end=b,positive_only=p,value='diff',aggregate='mean')
            self.assertEqual(oracle_a(W,q),oracle_b(W,q))
    def test_negative_difference(self):
        q={**Q,'value':'diff'};self.assertEqual(oracle_a(W,q)['answer'],-12)

class Measurement(unittest.TestCase):
    def setUp(self):self.o=json.loads(full_truth_response(W,Q))
    def score(self):return labels(json.dumps(self.o),W,Q)
    def test_natural_gold_blind_exact(self):
        self.assertEqual(natural_answer_score('Work: 18+30=48.\nAnswer: 48',Fraction(48)),1)
        self.assertEqual(natural_answer_score('Answer: 71.666667',Fraction(215,3)),0)
    def test_natural_duplicate_answer_rejected(self):
        self.assertEqual(natural_answer_score('Answer: 0\nAnswer: 48',Fraction(48)),0)
    def test_full_pass(self):
        r=self.score();self.assertEqual([r[k] for k in ['S','B','A','E_anchored']],[1,1,1,1])
    def test_order_invariance(self):
        self.o['selected'].reverse();self.o['bindings'].reverse();self.assertEqual(self.score()['B'],1);self.assertEqual(self.score()['S'],1)
    def test_duplicate_not_unknown(self):
        self.o['selected']=['Mug','Mug'];self.assertEqual(self.score()['S'],0)
    def test_gold_blind_missing_fields(self):
        del self.o['bindings'];r=self.score();self.assertEqual(r['V'],0);self.assertIsNone(r['B']);self.assertEqual(r['A'],1)
    def test_null_operand_unknown(self):
        self.o['bindings'][0]['operand']=None;r=self.score();self.assertIsNone(r['B']);self.assertEqual(r['V'],0);self.assertEqual(r['A'],1)
    def test_correct_answer_wrong_binding(self):
        self.o['bindings'][0]['operand']='999';r=self.score();self.assertEqual((r['A'],r['S'],r['B']),(1,1,0))
    def test_truthful_wrong_scope_not_rewarded(self):
        self.o['selected']=['Mug','Desk'];self.o['bindings']=[{'category':'Mug','operand':'18'},{'category':'Desk','operand':'30'}]
        r=self.score();self.assertEqual((r['A'],r['S'],r['B'],r['E_raw'],r['E_anchored']),(1,0,0,1,0))
    def test_empty_not_vacuously_true(self):
        self.o['bindings']=[];self.assertEqual(self.score()['B'],0)
    def test_extra_bindings_fail(self):
        self.o['bindings'].append({'category':'Desk','operand':'30'});self.assertEqual(self.score()['B'],0)
    def test_unknown_selected_semantic_fail(self):
        self.o['selected']=['Banana'];self.assertEqual(self.score()['V'],1);self.assertEqual(self.score()['S'],0)
    def test_duplicate_json_rejected(self):
        self.assertFalse(parse('{"answer":1,"answer":2}')['valid'])
    def test_exec_overlap_exact(self):
        for a,b in [('sum','48'),('mean','24'),('sum','0')]:
            self.o['aggregate']=a;self.o['answer']=b;r=self.score()
            self.assertEqual(r['E_anchored'],int(r['B']==1 and a==Q['aggregate'] and r['A']==1))
    def test_projection_preserves_legacy_fields(self):
        o=json.loads(legacy_projection(json.dumps(self.o)));self.assertEqual(o['answer'],self.o['answer']);self.assertNotIn('bindings',o)
    def test_partial_truncation_unknown(self):
        r=labels(json.dumps(self.o),W,Q,finished=False);self.assertIsNone(r['S']);self.assertEqual(r['finished'],0)
    def test_gold_does_not_change_extraction(self):
        text=json.dumps(self.o);before=parse(text);q={**Q,'direction':'largest'}
        labels(text,W,q);self.assertEqual(parse(text),before)

class Credit(unittest.TestCase):
    def test_standardized_advantage_correct(self):
        a=native_advantage([1]*7+[0]);self.assertAlmostEqual(a[-1],-math.sqrt(7));self.assertAlmostEqual(a[0],1/math.sqrt(7))
    def test_constant_J_still_semantic(self):
        c=credit(rows([0]*8,[0]*4+[1]*4));self.assertTrue(c['nonzero']);self.assertEqual(c['native'],[0.]*8)
    def test_constant_success_has_signal(self):self.assertTrue(credit(rows([1]*8,[1]*7+[0]))['nonzero'])
    def test_stratum_centering(self):
        r=rows([1,1,1,1,0,0,0,0],[1,1,0,0,1,1,1,0]);c=credit(r)
        self.assertAlmostEqual(sum(c['applied'][:4]),0);self.assertAlmostEqual(sum(c['applied'][4:]),0)
    def test_singleton_zero(self):
        c=credit(rows([1]*7+[0],[1]*7+[0]));self.assertEqual(c['applied'],[0.]*8)
    def test_unknown_does_not_get_fail(self):
        c=credit(rows([0]*4,[None,None,1,0],[1,0,1,0]));self.assertEqual(c['component_unscaled'][0][:2],[0,0])
    def test_unknown_pattern_zero_sum(self):
        c=credit(rows([0]*6,[None,None,1,0,1,1],[1,0,1,0,None,None]))
        for ids in c['strata']:self.assertAlmostEqual(sum(c['applied'][i] for i in ids),0)
    def test_global_not_equivalent_to_conditional(self):
        r=rows([1]*4+[0]*4,[1]*4+[0]*4)
        self.assertFalse(credit(r)['nonzero']);self.assertTrue(credit(r,'GLOBAL')['nonzero'])
    def test_shuffle_preserves_exact_credit_multiset(self):
        r=rows([1]*4+[0]*4,[1,1,1,0,1,1,0,0]);x=credit(r);y=credit(r,'SHUF',seed=101)
        for ids in x['strata']:self.assertEqual(sorted(x['applied'][i] for i in ids),sorted(y['applied'][i] for i in ids))
    def test_shuffle_can_legitimately_not_change(self):
        r=rows([1,1,0,0],[1,1,0,0]);self.assertFalse(credit(r,'SHUF',seed=1)['permutation_changed'])
    def test_no_resampling_until_shuffle_changes(self):
        r=rows([0]*2,[1,0]);res=[credit(r,'SHUF',seed=i)['permutation_changed'] for i in range(30)]
        self.assertIn(True,res);self.assertIn(False,res)
    def test_zero_lambda(self):self.assertEqual(credit(rows([1,0],[1,0]),lam=0)['applied'],[0.,0.])
    def test_J_no_aux(self):self.assertEqual(credit(rows([0]*4,[1,0,1,0]),'J')['applied'],[0.]*4)
    def test_mean_score_can_cancel_not_bug(self):
        c=credit(rows([0]*2,[1,0],[0,1]));self.assertEqual(c['applied'],[0.,0.])

class Design(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bank=build_bank('DEV_CAL',2,2026102711)
    def test_all_families(self):self.assertEqual(len(self.bank),12)
    def test_dual_oracle_on_views(self):
        for r in self.bank:
            for side in ('P0','P1'):
                for t in r['tasks'][side]:self.assertEqual(oracle_a(t['world'],t['query']),oracle_b(t['world'],t['query']))
    def test_matching_bins(self):
        for r in self.bank:
            self.assertEqual([x['edit_descriptor'] for x in r['tasks']['P0']],[x['edit_descriptor'] for x in r['tasks']['P1']])
    def test_not_adversarial_P0(self):
        for r in self.bank:self.assertGreaterEqual(r['certificate']['coverage_P1']+1e-12,r['certificate']['coverage_P0'])
    def test_shared_anchor(self):
        for r in self.bank:self.assertEqual(r['tasks']['P0'][0]['world'],r['tasks']['P1'][0]['world'])
    def test_no_heldout_leak(self):
        for r in self.bank:
            for side in ('P0','P1'):
                for t in r['tasks'][side]:self.assertFalse(t['heldout_tags'])
    def test_no_gold_in_prompt(self):
        t=self.bank[0]['tasks']['P0'][0];p=visible_prompt(t);x=copy.deepcopy(t);x['gold_answer']='999999';x['gold_selected']=['NotAThing'];x['factor']='SECRET';x['root_id']='SECRET'
        self.assertEqual(p,visible_prompt(x));self.assertNotIn('SECRET',p)
    def test_collision_is_mathematical_not_model_rate(self):
        for r in self.bank:
            if r['certificate']['engineered_collision_anchor']:self.assertIn('boundary_swap',r['certificate']['base_alias_rules'])
    def test_deterministic(self):self.assertEqual(self.bank[0],make_root('DEV_CAL','BINDING',0,2026102711))

class PlanningStatsGates(unittest.TestCase):
    def test_schedule_one_pass(self):
        s=root_schedule(48,99,0);self.assertEqual(len(s),72);ids=sum(s,[]);self.assertEqual(len(ids),288);self.assertEqual(len(set(ids)),288)
    def test_window_balance(self):
        s=root_schedule(48,99,0)
        for k in range(0,72,12):
            roots=sum(s[k:k+12],[])
            for f in FAMILIES:self.assertEqual(sum(f'TRAIN_{f}_' in r for r in roots),8)
    def test_same_seed_no_arm(self):self.assertEqual(draw_seed(1,'b0',0,0,0,0,0),draw_seed(1,'b0',0,0,0,0,0))
    def test_independent_repeats(self):self.assertNotEqual(draw_seed(1,'b0',0,0,0,0,0),draw_seed(1,'b0',1,0,0,0,0))
    def test_factorial(self):
        c=factorial({'U00':.4,'U01':.5,'U10':.6,'U11':.65});self.assertAlmostEqual(c['interaction'],-.05)
    def test_holm(self):self.assertEqual(holm({'a':.01,'b':.04}),{'a':.02,'b':.04})
    def test_zero_variance_not_certainty(self):self.assertIsNone(fixed_parent_interval(np.ones((3,2))*.01)['ci'])
    def test_run_df(self):
        r=fixed_parent_interval(np.array([[.01,.02],[.03,.01],[.00,.02]]));self.assertEqual(r['nominal_df'],3);self.assertGreaterEqual(r['df'],1);self.assertLessEqual(r['df'],3)
    def test_bootstrap_identical_arms(self):
        x=np.ones((3,2,2,2,4))*.7;r=crossed_bootstrap(x,np.array([1,-1]),reps=100);self.assertEqual(r['ci95'],[0.,0.])
    def test_launch_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'config.json').write_text('{}');self.assertFalse(launch_check(p,p/'config.json')['allowed'])

class TorchObjective(unittest.TestCase):
    def test_zero_aux_matches_native_gradient(self):
        import torch
        from srf6_ref.objective import loss
        x=torch.tensor([-.7,-1.2,-.2],dtype=torch.float64,requires_grad=True);sam=x.detach().clone();ref=sam+.01
        l=loss(x,sam,ref,.378,0)['loss'];l.backward();g=x.grad.clone();x.grad.zero_()
        old=x.detach();ratio=torch.exp(x-old);d=ref-x
        z=(-.378*ratio+.02*(torch.expm1(d)-d)).mean();z.backward()
        torch.testing.assert_close(l,z);torch.testing.assert_close(g,x.grad)
    def test_fixed_sequence_normalization(self):
        import torch
        from srf6_ref.objective import loss
        x=torch.full((8,),-.5,dtype=torch.float64,requires_grad=True)
        r=loss(x,x.detach(),x.detach(),0,.25);r['loss'].backward()
        torch.testing.assert_close(x.grad,torch.full_like(x,-.25/8))
    def test_kl_only_once(self):
        import torch
        from srf6_ref.objective import loss
        x=torch.tensor([-1.],requires_grad=True);r=loss(x,x.detach(),x.detach()+.1,.3,.2)
        self.assertAlmostEqual(float(r['loss'].detach()),float((r['native']+r['aux']+r['kl']).detach()),places=7)


class ClassoutAndCensus(unittest.TestCase):
    def test_classout_two_cell_new_edit(self):
        for f in FAMILIES:
            r=make_classout_root(f,0,2026102718)
            for t in r['tasks']['P0'][1:]:
                self.assertEqual(len(t['edit_descriptor']),2)
                self.assertTrue(all(abs(x[2])==12 for x in t['edit_descriptor']))
                self.assertNotEqual(t['edit_descriptor'][0][1],t['edit_descriptor'][1][1])
                self.assertEqual(oracle_a(t['world'],t['query']),oracle_b(t['world'],t['query']))
    def test_compound_wrong_rule_defined(self):
        for k in range(3):self.assertIsInstance(compound_wrong_value(W,Q,k),Fraction)
    def fixture(self):
        return [dict(parent='b0',factor='P0',group_id='g0',qid='q0',root_id='r0',family='BINDING',draw=i,policy_hash='fixture',A=0,J=0,V=1,finished=1,S=i%2,B=i%2) for i in range(8)]
    def test_census_conditional_signal_with_all_fail(self):
        from srf6_ref.census import summarize
        o=summarize(self.fixture());self.assertEqual(o['factors']['P0']['nonzero_groups'],1)
        self.assertEqual(o['factors']['P0']['nonzero_by_J_class'],{'all_zero':1})
    def test_census_duplicate_rejected(self):
        from srf6_ref.census import summarize
        x=self.fixture();x[-1]['draw']=0
        with self.assertRaises(ValueError):summarize(x)
    def test_census_mixed_policy_rejected(self):
        from srf6_ref.census import summarize
        x=self.fixture();x[-1]['policy_hash']='other'
        with self.assertRaises(ValueError):summarize(x)

if __name__=='__main__':unittest.main(verbosity=2)
