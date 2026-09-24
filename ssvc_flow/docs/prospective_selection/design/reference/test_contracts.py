import json
from pathlib import Path
import unittest
import numpy as np
from contracts import *

ROOT=Path(__file__).resolve().parents[1]

class InformationTests(unittest.TestCase):
    def test_event_inversion(self):
        np.testing.assert_allclose(event_probs(.4,.55,.9),[.4,.15,.35,.1])
    def test_invalid_nesting(self):
        with self.assertRaises(ValueError): event_probs(.8,.6,.9)
    def test_event_mass(self):
        rng=np.random.default_rng(24)
        for p in rng.dirichlet(np.ones(4),size=40):
            np.testing.assert_allclose(event_probs(p[0],p[:2].sum(),p[:3].sum()),p,atol=1e-14)
    def test_trend_identity(self):
        c=np.array([0,.5,1.,0,.5,1]); x=np.array([0,0,1,0,0,0])
        self.assertAlmostEqual(np.mean((c==1)&(x==0)),2*np.mean(c*c)-c.mean()-x.mean())
    def test_same_reward_different_repair(self):
        a=repair([17,5,7,2],[17,12,7,2],[17,5,7,2])
        b=repair([17,5,7,2],[17,12,7,2],[17,5,9,3])
        self.assertEqual(a['B'],0); self.assertEqual(b['B'],2)
        self.assertEqual(a['coord_accuracy'],.75); self.assertEqual(b['coord_accuracy'],.25)
        np.testing.assert_equal(rewards('W',0),[0,0,1,0])
    def test_exact_repair_identity(self):
        x=[17,12,7,2]; f=repair([17,5,7,2],x,x)
        self.assertEqual((f['F'],f['B'],f['M']),(1,0,1)); self.assertEqual(f['coord_accuracy'],1.)
    def test_invalid_missingness(self):
        f=repair([1,9,3,4],[1,2,3,4],None)
        self.assertIsNone(f['B']); self.assertEqual(f['penalty'],1)
    def test_boolean_coordinate_not_integer(self):
        self.assertFalse(repair([1,9,3,4],[1,2,3,4],[True,2,3,4])['valid'])
    def test_wrong_corruption_count(self):
        with self.assertRaises(ValueError): repair([1,2,3,4],[1,2,3,4],[1,2,3,4])
    def test_information_can_help(self):
        u=[[1,0],[0,1]]
        self.assertEqual(information_value(u,[.5,.5],[0,0]),.5)
        self.assertEqual(information_value(u,[.5,.5],[0,1]),1.)
    def test_more_information_need_not_help(self):
        u=[[1,.2],[.7,.4]]
        self.assertAlmostEqual(information_value(u,[.5,.5],[0,0]),information_value(u,[.5,.5],[0,1]))
    def test_duplicate_information_not_new(self):
        u=[[1,0],[.2,.9]]
        self.assertEqual(information_value(u,[.4,.6],[0,1]),information_value(u,[.4,.6],['a','b']))

class TrainingTests(unittest.TestCase):
    def test_recipes_exact(self):
        for r,w in RECIPES.items(): self.assertEqual(training_reward(r,'X',1),sum(w))
    def test_direct_damage(self):
        self.assertEqual(training_reward('DIRECT_REPAIR_R4','W',0,1),0)
        self.assertEqual(training_reward('DIRECT_REPAIR_R4','I',0,1),-.5)
    def test_invalid_damage_required(self):
        with self.assertRaises(ValueError): training_reward('DIRECT_REPAIR_R4','I',0,0)
    def test_x_relation_consistency(self):
        with self.assertRaises(ValueError): rewards('X',.5)
    def test_zero_group(self): np.testing.assert_equal(advantages([3]*8),np.zeros(8))
    def test_population_normalization(self):
        np.testing.assert_allclose(advantages([1,1,0,0]),np.array([.5,.5,-.5,-.5])/np.sqrt(.25+1e-8))
    def test_normalization_shift(self):
        np.testing.assert_allclose(advantages([2,0,0,0]),advantages([3,1,1,1]))

class SelectionTests(unittest.TestCase):
    def test_no_future_outcome(self):
        with self.assertRaises(ValueError): predecision_packet({'origin_id':'x','current':{},'history':{},'step':32,'future_pX':.9})
    def test_no_nested_future(self):
        with self.assertRaises(ValueError): predecision_packet({'origin_id':'x','current':{'future_pX':.9},'history':{},'step':32})
    def test_packet_minimal(self):
        self.assertEqual(predecision_packet({'origin_id':'x','current':{},'history':{},'step':32})['step'],32)
    def test_same_lineage_same_role_allowed(self):
        self.assertEqual(len(check_lineage_splits([{'lineage_id':1,'role':'fit'},{'lineage_id':1,'role':'fit'}])),1)
    def test_lineage_leakage_rejected(self):
        with self.assertRaises(ValueError): check_lineage_splits([{'lineage_id':1,'role':'fit'},{'lineage_id':1,'role':'test'}])
    def test_choose_tie(self):
        self.assertEqual(choose({'R0':0,'R3':.004},'R0')['recipe'],'R0')
    def test_choose_non_tie(self):
        self.assertEqual(choose({'R0':0,'R3':.02},'R0')['recipe'],'R3')
    def test_choose_not_safety(self):
        self.assertIn('NOT_SAFETY',choose({'R0':0,'R1':.1},'R0')['status'])
    def test_action_dedup(self):
        self.assertEqual(len(unique_branches('o',['R3','R3'],['R0','R3'])),4)
    def test_repeats_not_new_seed(self):
        a=collapse_lineage([{'lineage_id':1,'difference':1},{'lineage_id':1,'difference':3},{'lineage_id':2,'difference':0}])
        self.assertEqual(a,{1:2.,2:0.})
    def test_histogram_kernel_psd(self):
        p=np.array([sqrt_histogram([1,2,3]),sqrt_histogram([0,3,0]),sqrt_histogram([4,0,1])])
        k=p@p.T; self.assertGreaterEqual(np.linalg.eigvalsh(k).min(),-1e-12)
        np.testing.assert_allclose(np.diag(k),1)
    def test_krr_fit(self):
        k=np.eye(3); y=np.array([-.1,0,.1]); fit=krr_fit(k,y,.1)
        np.testing.assert_allclose(fit['beta'],y/1.3)
    def test_krr_constant_targets(self):
        f=krr_fit(np.ones((3,3)),np.ones(3)*.4,.1)
        np.testing.assert_allclose(f['beta'],0,atol=1e-12)

class StratifiedTests(unittest.TestCase):
    def test_cell_mean(self):
        vals={0:[.01,.02,.03],1:[0,.01,.02],2:[-.01,0,.01],3:[0,.005,.01]}
        rows=[{'lineage_id':10*c+j,'cell':c,'difference':d} for c,v in vals.items() for j,d in enumerate(v)]
        r=stratified_summary(rows,[0,1,2,3]); self.assertAlmostEqual(r['mean_difference'],.00875)
        self.assertAlmostEqual(r['se']**2,.000325/48); self.assertGreater(r['df'],0)
    def test_zero_variance_not_certificate(self):
        rows=[{'lineage_id':j,'cell':j%4,'difference':0} for j in range(12)]
        r=stratified_summary(rows,[0,1,2,3]); self.assertIsNone(r['df'])
        self.assertEqual(r['status'],'NO_OBSERVED_VARIABILITY')
    def test_repeat_rejected(self):
        rows=[{'lineage_id':1,'cell':0,'difference':0}]*2
        with self.assertRaises(ValueError): stratified_summary(rows,[0])
    def test_selection_bound(self):
        rng=np.random.default_rng(41)
        for _ in range(50):
            utility=rng.random(8); eps=.05; tau=.01; prediction=utility+rng.uniform(-eps,eps,8)
            d=choose({str(i):v for i,v in enumerate(prediction)},'0',tau)
            self.assertLessEqual(utility.max()-utility[int(d['recipe'])],2*eps+tau+1e-12)

class PlanningTests(unittest.TestCase):
    def test_power_2pp_sd(self):
        p=plan_n(.02); self.assertEqual(p['selected'],32)
    def test_power_unresolved(self): self.assertTrue(plan_n(.03)['target_power_not_reached_by_plan'])
    def test_monte_carlo_scale(self):
        s=mc_se_upper(288,16,12,2); self.assertAlmostEqual(s,.0021262931794992866)
        self.assertLess(s,.0025)
    def test_more_seeds_not_larger_se(self):
        self.assertLess(mc_se_upper(288,16,24,2),mc_se_upper(288,16,12,2))
    def test_draw_plan(self): self.assertEqual(plan_draws(12)['draws'],16)
    def test_budget_json(self):
        p=json.loads((ROOT/'protocol.json').read_text())
        w=workload();
        for key,value in w.items(): self.assertEqual(value,p['workload_min_test'][key])
    def test_gpu_cap(self):
        p=json.loads((ROOT/'protocol.json').read_text()); self.assertEqual(p['execution']['max_concurrent_gpu_jobs'],5)
        self.assertEqual(p['execution']['gpus_per_job'],1)
    def test_final_cells_balanced(self):
        p=json.loads((ROOT/'protocol.json').read_text()); rows=p['origins']['test_pool'][:12]
        from collections import Counter
        c=Counter((r['source_recipe'],r['anchors'][0]) for r in rows)
        self.assertEqual(set(c.values()),{3})
    def test_registry_disjoint(self):
        p=json.loads((ROOT/'protocol.json').read_text()); ids=[]
        for name in ('development','tuning','test_pool'): ids += [r['seed'] for r in p['origins'][name]]
        self.assertEqual(len(ids),len(set(ids)))

if __name__=='__main__': unittest.main(verbosity=2)
