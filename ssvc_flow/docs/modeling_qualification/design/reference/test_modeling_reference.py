"""Independent algebra/contract tests; these do not train the CPU toy or Qwen."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from modeling_reference import (
    advantages, core_budget, error_bounds, event_gradient_affine, from_nested,
    geometry, helmert, mle, nested_fisher, nested_jacobian, orthogonal_span,
    probability, softmax, to_nested, truncate_response,
)

ROOT=Path(__file__).resolve().parents[1]


class ModelingReferenceTests(unittest.TestCase):
    def test_01_helmert_orthonormal(self):
        h=helmert()
        np.testing.assert_allclose(h.T@h,np.eye(3),atol=1e-14)
        np.testing.assert_allclose(h.T@np.ones(4),np.zeros(3),atol=1e-14)

    def test_02_helmert_roundtrip_and_distance(self):
        rng=np.random.default_rng(11)
        for _ in range(50):
            p,q=rng.dirichlet(np.ones(4),size=2)
            np.testing.assert_allclose(.25+helmert()@(helmert().T@p),p,atol=1e-14)
            self.assertAlmostEqual(np.linalg.norm(p-q),np.linalg.norm(helmert().T@(p-q)),places=13)

    def test_03_nested_interior_roundtrip(self):
        rng=np.random.default_rng(12)
        for v,q,s in rng.uniform(.03,.97,size=(100,3)):
            p=from_nested(v,q,s)
            nested=to_nested(p)
            np.testing.assert_allclose([nested['v'],nested['q'],nested['s']],[v,q,s],atol=1e-12)

    def test_04_boundaries_are_not_fake_zero_information(self):
        a=to_nested([0.,0.,0.,1.])
        self.assertIsNone(a['q']); self.assertIsNone(a['s'])
        np.testing.assert_array_equal(from_nested(0.,None,None),[0,0,0,1])
        b=to_nested([1.,0.,0.,0.])
        self.assertEqual(b['q'],1.); self.assertIsNone(b['s'])
        np.testing.assert_array_equal(from_nested(1.,1.,None),[1,0,0,0])
        with self.assertRaises(ValueError): from_nested(.5,None,None)

    def test_05_fisher_identity(self):
        rng=np.random.default_rng(13)
        for v,q,s in rng.uniform(.03,.97,size=(200,3)):
            p=from_nested(v,q,s); d=nested_jacobian(v,q,s)
            np.testing.assert_allclose(d.T@np.diag(1/p)@d,nested_fisher(v,q,s),atol=1e-10,rtol=1e-12)

    def test_06_jacobian_finite_difference(self):
        x=np.array([.7,.4,.2]); h=1e-6
        cols=[]
        for i in range(3):
            step=np.eye(3)[i]*h
            cols.append((from_nested(*(x+step))-from_nested(*(x-step)))/(2*h))
        np.testing.assert_allclose(np.column_stack(cols),nested_jacobian(*x),atol=1e-9)

    def test_07_mle_equivalence(self):
        for counts in ([3,2,4,7],[0,0,0,16],[16,0,0,0],[0,2,5,9],[1,0,15,0]):
            result=mle(counts)
            np.testing.assert_allclose(from_nested(result['v'],result['q'],result['s']),result['p'],atol=1e-14)

    def test_08_invalid_inputs(self):
        for p in ([.1,.2,.3,.5],[-.1,.4,.4,.3],[float('nan'),0,0,1]):
            with self.assertRaises(ValueError): probability(p)
        with self.assertRaises(ValueError): mle([0,0,0,0])
        with self.assertRaises(ValueError): mle([True,1,1,1])
        with self.assertRaises(ValueError): nested_fisher(0,.5,.5)

    def test_09_all_valid_shift_invariant(self):
        cats=['X','X','S','W','W','X','W','S']
        np.testing.assert_allclose(advantages(cats,0),advantages(cats,1),atol=1e-12)

    def test_10_small_positive_weight_saturation(self):
        cats=['W']*4+['I']*4
        small=advantages(cats,.01); large=advantages(cats,1)
        self.assertLess(np.max(np.abs(small-large)),.00021)
        self.assertGreater(np.linalg.norm(small),2.8)
        np.testing.assert_array_equal(advantages(cats,1,no_x_off=True),np.zeros(8))

    def test_11_no_x_rule_unchanged_when_x_present(self):
        cats=['X','W','S','I']*2
        np.testing.assert_array_equal(advantages(cats,1),advantages(cats,1,no_x_off=True))

    def test_12_duplicate_directions_do_not_increase_rank(self):
        d=np.array([[1.,2.,0.,0.],[0.,0.,1.,3.],[0.,0.,0.,0.]])
        q,_=orthogonal_span(d)
        self.assertEqual(q.shape,(3,2))
        np.testing.assert_allclose(q@q.T@d,d,atol=1e-12)
        q,_=orthogonal_span(np.zeros((3,4)))
        self.assertEqual(q.shape,(3,0))

    def test_13_truncated_svd_error(self):
        rng=np.random.default_rng(14)
        r=rng.normal(size=(12,7))
        for rank in range(8):
            approx,tail=truncate_response(r,rank)
            self.assertAlmostEqual(np.linalg.norm(r-approx,2),tail,places=12)

    def test_14_equal_norm_opposite_directions(self):
        value=geometry([1.,0.],[-1.,0.])
        self.assertEqual(value['left_norm'],value['right_norm'])
        self.assertEqual(value['cosine'],-1.)
        self.assertEqual(value['distance'],2.)

    def test_15_projection_residual_summary_is_not_identifying(self):
        u=np.array([[1.],[0.]])
        plus=np.array([0.,.2]); minus=-plus
        np.testing.assert_array_equal(u.T@plus,u.T@minus)
        self.assertEqual(np.linalg.norm(plus),np.linalg.norm(minus))
        pplus=softmax([plus.sum(),0,0,0])[0]
        pminus=softmax([minus.sum(),0,0,0])[0]
        self.assertGreater(pplus,.25); self.assertLess(pminus,.25)
        minimax=(pplus-pminus)/2
        midpoint=(pplus+pminus)/2
        self.assertAlmostEqual(abs(pplus-midpoint),minimax)

    def test_16_net_and_path_conditional_bounds(self):
        rng=np.random.default_rng(15)
        for _ in range(30):
            a=rng.normal(size=(4,5)); theta=rng.normal(size=5)
            bias=rng.normal(size=4); w=np.array([1.,0.,0.,0.])
            u=np.linalg.qr(rng.normal(size=(5,2)))[0]
            start,g=event_gradient_affine(a,theta,bias,w)
            coefficient=u.T@g+rng.normal(scale=.01,size=2)
            eps=np.linalg.norm(u.T@g-coefficient)
            kappa=np.linalg.norm(g-u@(u.T@g))
            curvature=3*np.linalg.norm(a,2)**2
            steps=rng.normal(scale=.04,size=(8,5))
            for t in range(1,9):
                d=steps[:t].sum(axis=0)
                truth,_=event_gradient_affine(a,theta+d,bias,w)
                estimate=start+coefficient@(u.T@d)
                bound=error_bounds(steps[:t],u,eps=eps,kappa=kappa,curvature=curvature)
                self.assertLessEqual(abs(truth-estimate),bound['net']+1e-12)
                self.assertLessEqual(bound['net'],bound['path']+1e-12)

    def test_17_intermediate_excursion_and_endpoint_return(self):
        u=np.array([[1.],[0.]])
        steps=np.array([[0.,.5],[0.,-.5]])
        first=error_bounds(steps[:1],u,kappa=1.,curvature=1.)
        final=error_bounds(steps,u,kappa=1.,curvature=1.)
        self.assertGreater(first['net'],0.)
        self.assertEqual(final['net'],0.)
        self.assertGreater(final['path'],0.)
        self.assertNotEqual(softmax([.5,0,0,0])[0],softmax([0,0,0,0])[0])

    def test_18_bad_basis_rejected(self):
        with self.assertRaises(ValueError): error_bounds([[1.,2.]],[[2.],[0.]])
        with self.assertRaises(ValueError): error_bounds([[1.,2.]],[[1.],[0.]],kappa=-1)

    def test_19_core_budget_and_model_size(self):
        config=json.loads((ROOT/'protocol.json').read_text())
        self.assertEqual(core_budget(config),config['budget_derived'])
        self.assertEqual(core_budget(config)['total_optimizer_steps'],11400)
        self.assertEqual(core_budget(config)['total_sampled_finite_actions'],203520)
        m=config['toy_model']
        self.assertEqual(m['input_dim']*m['hidden_dim']+m['hidden_dim']+m['hidden_dim']+1,m['parameter_count'])

    def test_20_split_overlap_rejected(self):
        config=json.loads((ROOT/'protocol.json').read_text())
        bad=copy.deepcopy(config)
        bad['profiles']['core']['seeds_by_role']['locked_test'][0]=101
        with self.assertRaises(ValueError): core_budget(bad)

    def test_21_calibration_not_automatically_cost_saving(self):
        full_panels=8*3
        tracking_horizon=16
        self.assertGreater(full_panels,tracking_horizon)
        self.assertLess(2*3,tracking_horizon)

    def test_22_conditional_fisher_has_valid_domain(self):
        p=[0.,.3,.6,.1]
        n=to_nested(p)
        self.assertEqual(n['q'],0.)
        self.assertTrue(n['s_defined'])
        with self.assertRaises(ValueError): nested_fisher(n['v'],n['q'],n['s'])


if __name__=='__main__':
    unittest.main(verbosity=2)
