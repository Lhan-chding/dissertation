import unittest
import numpy as np
from reference_math import (
    helmert, conditional_coordinates, from_conditionals, independent_count_covariance,
    contrast_covariance, weighted_moments, lr_moments, inverse_cdf_joint, crn_moments,
    stable_exp_difference, gls_ridge, contrast_taylor_bound, resolution_label, fresh_budget,
)


class ReferenceContracts(unittest.TestCase):
    def setUp(self):
        self.p=np.array([.2,.25,.4,.15])
        self.q=np.array([.201,.25,.401,.148])

    def test_helmert_orthogonality(self):
        h=helmert()
        np.testing.assert_allclose(h.T@h,np.eye(3),atol=1e-15)
        np.testing.assert_allclose(h@h.T,np.eye(4)-np.ones((4,4))/4,atol=1e-15)

    def test_level_roundtrip(self):
        h=helmert()
        np.testing.assert_allclose(.25+(self.p@h)@h.T,self.p,atol=1e-15)

    def test_conditionals_roundtrip(self):
        c=conditional_coordinates(self.p)
        np.testing.assert_allclose(from_conditionals(**c),self.p)

    def test_missing_conditionals_unknown(self):
        c=conditional_coordinates([0,0,0,1])
        self.assertIsNone(c['q']); self.assertIsNone(c['s'])
        self.assertIsNone(conditional_coordinates([1,0,0,0])['s'])

    def test_shared_anchor_exact_cancellation(self):
        noise=np.array([.03,-.02,.01,-.02])
        origin=self.p+noise
        np.testing.assert_allclose((self.q-origin)-(self.p-origin),self.q-self.p,atol=1e-15)

    def test_common_baseline_covariance(self):
        # P contains independent origin, baseline, candidate1, candidate2.
        v=np.diag([.8,.3,.4,.5]); l=np.array([[0,-1,1,0],[0,-1,0,1]])
        c=contrast_covariance(v,l)
        np.testing.assert_allclose(c,[[.7,.3],[.3,.8]])

    def test_shared_origin_not_in_contrast(self):
        l=np.array([[0,-1,1]])
        a=contrast_covariance(np.diag([1.,.2,.3]),l)
        b=contrast_covariance(np.diag([100.,.2,.3]),l)
        np.testing.assert_allclose(a,b)

    def test_multinomial_covariance_singular(self):
        c=independent_count_covariance(self.p,64)
        np.testing.assert_allclose(c@np.ones(4),0,atol=1e-15)
        self.assertGreater(np.linalg.eigvalsh(helmert().T@c@helmert()).min(),0)

    def test_alias_contrast_exact_zero_covariance(self):
        c=contrast_covariance(np.array([[.4,.4],[.4,.4]]),[[-1,1]])
        np.testing.assert_allclose(c,0,atol=1e-15)

    def test_crn_marginals(self):
        j=inverse_cdf_joint(self.p,self.q)
        np.testing.assert_allclose(j.sum(1),self.p,atol=1e-15)
        np.testing.assert_allclose(j.sum(0),self.q,atol=1e-15)

    def test_crn_estimator_unbiased(self):
        m,c=crn_moments(self.p,self.q,np.eye(4))
        np.testing.assert_allclose(m,self.q-self.p,atol=1e-15)
        self.assertGreaterEqual(np.linalg.eigvalsh(c).min(),-1e-14)

    def test_crn_can_increase_variance(self):
        p=np.array([.49,.01,.01,.49]);q=np.array([.01,.49,.49,.01]);f=np.array([1,0,1,0])
        _,cov=crn_moments(p,q,f)
        independent=(p@f)*(1-p@f)+(q@f)*(1-q@f)
        self.assertGreater(cov[0,0],independent)

    def test_crn_alias_zero_variance(self):
        m,c=crn_moments(self.p,self.p,np.eye(4))
        np.testing.assert_allclose(m,0,atol=1e-15)
        np.testing.assert_allclose(c,0,atol=1e-15)

    def test_crn_bernoulli_discordance_formula(self):
        f=np.array([1,0,0,0]); j=inverse_cdf_joint(self.p,self.q)
        z=f[None,:]-f[:,None]
        mean,cov=crn_moments(self.p,self.q,f)
        eta=np.sum(j*z*z)
        self.assertAlmostEqual(cov[0,0],eta-mean[0]**2)

    def test_lr_unbiased(self):
        m,c,w=lr_moments(self.p,self.q,self.p,np.eye(4))
        np.testing.assert_allclose(m,self.q-self.p,atol=1e-15)
        self.assertGreaterEqual(np.linalg.eigvalsh(c).min(),-1e-14)

    def test_lr_mixture_weight_bound(self):
        p=np.array([.99,.01,0]);q=np.array([0,.01,.99]);rho=(p+q)/2
        m,_,w=lr_moments(p,q,rho,np.eye(3))
        self.assertLessEqual(np.abs(w).max(),2)
        np.testing.assert_allclose(m,q-p,atol=1e-15)

    def test_lr_missing_support_rejected(self):
        with self.assertRaises(ValueError):
            lr_moments([1,0],[.9,.1],[1,0],np.eye(2))

    def test_lr_alias_zero(self):
        m,c,w=lr_moments(self.p,self.p,self.p,np.eye(4))
        self.assertEqual(float(c.sum()),0);self.assertEqual(float(np.abs(w).sum()),0)

    def test_lr_quadratic_variance_scaling(self):
        h=np.array([.001,0,.001,-.002])
        _,v1,_=lr_moments(self.p,self.p+h,self.p,np.eye(4))
        _,v2,_=lr_moments(self.p,self.p+2*h,self.p,np.eye(4))
        np.testing.assert_allclose(v2,4*v1,rtol=1e-12,atol=1e-15)

    def test_lr_raw_vs_helmert_control_variate(self):
        h=helmert();sample_cat=2
        _,_,w=lr_moments(self.p,self.q,self.p,np.eye(4))
        raw=w[sample_cat]*np.eye(4)[sample_cat]
        projected=(w[sample_cat]*h[sample_cat])@h.T
        np.testing.assert_allclose(projected,raw-raw.sum()/4,atol=1e-15)
        self.assertAlmostEqual(projected.sum(),0)
        m,_,_=lr_moments(self.p,self.q,self.p,h)
        np.testing.assert_allclose(m@h.T,self.q-self.p,atol=1e-15)

    def test_log_difference_near_equal(self):
        a=np.array([0.,-10.,-1.]);b=a+1e-12
        d=stable_exp_difference(a,b)
        np.testing.assert_allclose(d,-np.exp(a)*np.expm1(b-a),rtol=1e-12,atol=1e-20)
        np.testing.assert_array_equal(stable_exp_difference(a,a),0)

    def test_log_zero_probability_is_supported(self):
        np.testing.assert_allclose(stable_exp_difference([-np.inf,0.,-np.inf],[0.,-np.inf,-np.inf]),[-1.,1.,0.])

    def test_overflow_not_clipped(self):
        with self.assertRaises(FloatingPointError):
            stable_exp_difference([1000.],[0.])

    def test_gls_equals_ridge_with_identity(self):
        x=np.array([[1.,0],[0,1],[1,1]])
        y=np.array([.2,.3,.6]); lam=.1
        fit=gls_ridge(x,y,np.eye(3),lam)
        np.testing.assert_allclose(fit,np.linalg.solve(x.T@x+lam*np.eye(2),x.T@y))

    def test_gls_whitening_matches_normal_equations(self):
        x=np.array([[1.,0],[0,1],[1,1]])
        y=np.array([.2,.3,.6]);cov=np.array([[2,.5,0],[.5,1,0],[0,0,3]])
        expected=np.linalg.solve(x.T@np.linalg.solve(cov,x)+.1*np.eye(2),x.T@np.linalg.solve(cov,y))
        np.testing.assert_allclose(gls_ridge(x,y,cov,.1),expected)

    def test_contrast_taylor_bound_quadratic(self):
        rng=np.random.default_rng(15)
        a=np.diag([1.,2.,4.]);theta=rng.normal(size=3);d=rng.normal(size=3);e=rng.normal(size=3)*.1
        f=lambda x:.5*x@a@x
        residual=abs(f(theta+d+e)-f(theta+d)-(a@theta)@e)
        self.assertLessEqual(residual,contrast_taylor_bound(4,d,e)+1e-14)
        baseline_residual=abs(f(theta+d+e)-f(theta+d)-(a@(theta+d))@e)
        self.assertLessEqual(baseline_residual,2*np.linalg.norm(e)**2+1e-14)

    def test_equal_norm_not_equal_vector(self):
        a=np.array([1.,0]);b=-a
        self.assertEqual(np.linalg.norm(a),np.linalg.norm(b))
        self.assertGreater(np.linalg.norm(a-b),0)

    def test_total_low_rank_does_not_preserve_contrast(self):
        # A common movement of magnitude 1000 dominates a decision contrast 0.01.
        y=np.array([[1000.,.01],[1000.,-.01]])
        u,s,vh=np.linalg.svd(y,full_matrices=False)
        r1=(u[:,:1]*s[:1])@vh[:1]
        self.assertLess(np.linalg.norm(y-r1)/np.linalg.norm(y),1e-3)
        self.assertAlmostEqual(np.linalg.norm((r1[1]-r1[0])-(y[1]-y[0]))/
                               np.linalg.norm(y[1]-y[0]),1.,places=10)

    def test_nonzero_diagnostic_not_replaced_by_zero(self):
        caps=[1,2,4]; k=2
        self.assertEqual([min(c,k) for c in caps],[1,2,2])
        selected=None
        self.assertIsNone(selected)

    def test_resolution_unknown_and_equivalence(self):
        self.assertEqual(resolution_label(None,.1),'UNKNOWN')
        self.assertEqual(resolution_label([-.2,.2],.1),'UNKNOWN')
        self.assertEqual(resolution_label([-.01,.02],.1),'WITHIN_TOLERANCE')
        self.assertEqual(resolution_label([.11,.2],.1),'MEANINGFUL_POSITIVE')
        self.assertEqual(resolution_label([-.2,-.11],.1),'MEANINGFUL_NEGATIVE')

    def test_known_event_scoring_baseline(self):
        # A finite support with one X and two I: only three specified probabilities needed.
        p=np.array([.2,.1,.1,.15,.15,.1,.1,.1]); category=np.array([0,1,1,2,2,2,3,3])
        px=p[np.where(category==0)[0][0]];v=1-p[category==3].sum()
        self.assertAlmostEqual(px,p[category==0].sum())
        self.assertAlmostEqual(v,p[category!=3].sum())

    def test_new_budget(self):
        b=fresh_budget()
        self.assertEqual(b,{'trajectories':32,'main_updates':2048,'fork_updates':4032,
                            'total_updates':6080,'training_and_bank_actions':108544})


if __name__ == '__main__':
    unittest.main(verbosity=2)
