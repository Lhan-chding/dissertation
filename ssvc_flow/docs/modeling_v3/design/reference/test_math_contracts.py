"""Deterministic algebra and interface tests; not research experiments."""
import json
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
import math_contracts as m

class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.b=np.array([.1,.2,.3,.4]); self.u=np.array([.11,.19,.32,.38])
        self.r=(self.b+self.u)/2
        self.z=m.exact_contributions(self.b,self.u,self.r,np.arange(4))
        self.mu,self.cov=m.moments(self.z,self.r)
    def test_helmert_orthonormal(self):
        h=m.helmert(); assert_allclose(h.T@h,np.eye(3),atol=1e-14)
    def test_helmert_null_mass(self):
        assert_allclose(m.helmert().T@m.ONES,0,atol=1e-14)
    def test_noisy_lossless_four_numbers(self):
        y=np.array([.1,.3,-.2,.5]); h=m.helmert()
        assert_allclose(h@(h.T@y)+y.sum()/4*m.ONES,y,atol=1e-14)
    def test_exact_signed_measure(self):
        assert_allclose(self.mu,self.u-self.b,atol=1e-14)
    def test_equal_projection_total_error(self):
        raw=np.array([0.,2.,0.,0.]); p=m.correction(raw,m.B_EQUAL)
        self.assertAlmostEqual(raw@raw-p@p,raw.sum()**2/4)
    def test_equal_projection_can_hurt_x(self):
        raw=np.array([0.,2.,0.,0.]); self.assertGreater(abs(m.correction(raw,m.B_EQUAL)[0]),abs(raw[0]))
    def test_preserve_xi(self):
        p=m.correction(self.z,m.B_PRESERVE_XI)
        assert_allclose(p[:,[0,3]],self.z[:,[0,3]],atol=1e-14)
    def test_preserve_xi_is_zero_sum(self):
        assert_allclose(m.correction(self.z,m.B_PRESERVE_XI).sum(axis=1),0,atol=1e-14)
    def test_drop_w_preserves_other_events(self):
        p=m.correction(self.z,m.B_DROP_W)
        assert_allclose(p[:,[0,1,3]],self.z[:,[0,1,3]],atol=1e-14)
    def test_mixture_formula(self):
        assert_allclose(m.stable_pair_weight(np.log(self.u),np.log(self.b)),(self.u-self.b)/self.r,atol=1e-14)
    def test_extreme_log_ratio(self):
        assert_allclose(m.stable_pair_weight([-1,-10000,-1],[-10000,-1,-1]),[2,-2,0],atol=1e-14)
    def test_small_log_ratio(self):
        w=m.stable_pair_weight([-20+1e-9],[-20]); self.assertTrue(0<float(w[0])<2e-9)
    def test_identical_policy_exact_zero(self):
        z=m.exact_contributions(self.b,self.b,self.b,np.arange(4)); assert_allclose(z,0,atol=0)
    def test_support_failure(self):
        with self.assertRaises(ValueError):
            m.exact_contributions(self.b,self.u,[0,.2,.3,.5],np.arange(4))
    def test_unknown_label(self):
        with self.assertRaises(ValueError):
            m.exact_contributions(self.b,self.u,self.r,[0,1,2,9])
    def test_probability_reject(self):
        with self.assertRaises(ValueError): m.probability([.1,.2])
    def test_oracle_coefficient_sum(self):
        self.assertAlmostEqual(float(m.oracle_b(self.cov).sum()),1.0)
    def test_oracle_covariance_formula(self):
        b=m.oracle_b(self.cov); c=self.cov@m.ONES; v=float(m.ONES@c)
        assert_allclose(m.covariance_after(self.cov,b),self.cov-np.outer(c,c)/v,atol=1e-14)
    def test_oracle_variance_never_increases(self):
        fixed=m.covariance_after(self.cov,m.oracle_b(self.cov))
        self.assertTrue(np.all(np.diag(fixed)<=np.diag(self.cov)+1e-14))
    def test_coefficient_error_penalty(self):
        optimum=m.oracle_b(self.cov); b=np.array([-.2,.5,.4,.3]); v=float(m.ONES@self.cov@m.ONES)
        difference=m.covariance_after(self.cov,b)-m.covariance_after(self.cov,optimum)
        assert_allclose(difference,v*np.outer(b-optimum,b-optimum),atol=1e-14)
    def test_mean_covariance_n(self):
        assert_allclose(m.covariance_after(self.cov,m.B_EQUAL,16)*16,m.covariance_after(self.cov,m.B_EQUAL),atol=1e-14)
    def test_pilot_uninformative(self):
        b,report=m.pilot_coefficient(np.zeros((8,4)))
        assert_allclose(b,m.B_PRESERVE_XI); self.assertEqual(report['status'],'PILOT_UNINFORMATIVE')
    def test_pilot_bound_and_sum(self):
        z=np.array([[1,-.999,0,0],[-1,.999,0,0],[2,-1.998,0,0]])
        b,_=m.pilot_coefficient(z,shrink=0,l1_cap=8)
        self.assertLessEqual(float(np.abs(b).sum()),8+1e-10); self.assertAlmostEqual(float(b.sum()),1,places=9)
    def test_fixed_b_unbiased(self):
        for b in [m.B_EQUAL,m.B_DROP_W,m.B_PRESERVE_XI,m.oracle_b(self.cov)]:
            assert_allclose(self.r@m.correction(self.z,b),self.mu,atol=1e-14)
    def test_random_independent_b_covariance(self):
        bs=[m.B_EQUAL,m.B_DROP_W,m.B_PRESERVE_XI]; prior=np.array([.2,.3,.5]); opt=m.oracle_b(self.cov)
        lhs=sum(p*m.covariance_after(self.cov,b) for p,b in zip(prior,bs))
        penalty=sum(p*np.outer(b-opt,b-opt) for p,b in zip(prior,bs))
        rhs=m.covariance_after(self.cov,opt)+float(m.ONES@self.cov@m.ONES)*penalty
        assert_allclose(lhs,rhs,atol=1e-14)
    def test_qx_unknown(self):
        self.assertIsNone(m.qx_interval(0,.1,0,.5))

class CoverageTests(unittest.TestCase):
    def test_span_reconstruction(self):
        g=m.span_geometry([[1,0,0],[0,1,0]],[1,2,3]); assert_allclose(g['parallel']+g['perp'],[1,2,3]); self.assertEqual(g['k'],2)
    def test_empty_excitation(self):
        g=m.span_geometry(np.zeros((2,3)),[1,0,0]); self.assertEqual(g['status'],'NO_CALIBRATION_EXCITATION'); self.assertEqual(g['rho'],1.)
    def test_zero_query(self):
        g=m.span_geometry([[1,0,0]],[0,0,0]); self.assertIsNone(g['rho'])
    def test_in_span(self):
        g=m.span_geometry([[1,0],[0,1]],[.2,.3]); self.assertLess(g['rho'],1e-14)
    def test_unidentifiable_pair(self):
        fit=np.array([[1.,0.],[2.,0.]]); a=np.array([.3,2.]); b=np.array([.3,-2.])
        assert_allclose(fit@a,fit@b); self.assertNotEqual(float(a@np.array([0,1.])),float(b@np.array([0,1.])))
    def test_equal_norm_unequal_vectors(self):
        a=np.array([1.,0.]); b=-a; self.assertEqual(np.linalg.norm(a),np.linalg.norm(b)); self.assertGreater(np.linalg.norm(a-b),0)
    def test_no_x_gate_before_normalization(self):
        labels=['W']*4+['I']*4
        assert_allclose(m.grouped_advantages(labels,1.,policy='no_x_off'),0)
        self.assertGreater(float(np.linalg.norm(m.grouped_advantages(labels,1.))),0)
    def test_small_lambda_is_not_small_advantage(self):
        labels=['W']*4+['I']*4
        a=m.grouped_advantages(labels,.01); b=m.grouped_advantages(labels,1.)
        self.assertLess(np.linalg.norm(a-b)/np.linalg.norm(b),.001)
    def test_full_valid_constant_cancels(self):
        labels=['X','W','S','W']; assert_allclose(m.grouped_advantages(labels,0),m.grouped_advantages(labels,1),atol=1e-14)
    def test_reference_noise_mse_identity(self):
        pred=.2; truth=.1; eps=np.array([-.04,.04]); raw=np.mean((pred-(truth+eps))**2)
        self.assertAlmostEqual(raw-np.mean(eps**2),(pred-truth)**2)

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=json.loads((Path(__file__).resolve().parents[1]/'protocol.json').read_text())
    def test_seeds_disjoint(self):
        groups=list(self.c['qwen']['seed_roles'].values()); all_seeds=sum(groups,[])
        self.assertEqual(len(all_seeds),len(set(all_seeds)))
    def test_source_counts(self):
        d=self.c['derived_workload']; self.assertEqual(d['qwen_source_optimizer_steps'],3072); self.assertEqual(d['qwen_source_training_outputs'],98304)
    def test_fork_counts(self):
        d=self.c['derived_workload']; self.assertEqual(d['qwen_base_fork_updates'],3240); self.assertEqual(d['qwen_total_response_origins'],30)
    def test_two_gpu_and_no_feedback(self):
        self.assertEqual(self.c['scope']['max_concurrent_project_gpus'],2); self.assertFalse(self.c['scope']['online_ssvc']); self.assertFalse(self.c['scope']['cost_success_gate'])
    def test_independent_pilot(self):
        self.assertTrue(self.c['observation']['pilot_and_main_rng_independent']); self.assertTrue(self.c['observation']['same_sample_pilot_for_main_forbidden'])
    def test_true_reduction_required(self):
        self.assertTrue(self.c['models']['primary_comparisons_require_r_less_k'])
    def test_raw_baseline_kept(self):
        self.assertIn('RAW4',self.c['observation']['methods']); self.assertIn('PRESERVE_XI',self.c['observation']['methods'])
    def test_no_predicted_reference_as_truth(self):
        self.assertTrue(self.c['qwen']['reference']['noisy_reference_not_exact_truth'])

if __name__=='__main__':
    unittest.main(verbosity=2)
