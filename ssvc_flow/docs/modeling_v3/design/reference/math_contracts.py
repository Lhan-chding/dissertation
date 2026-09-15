"""Independent V3 mathematical contracts. No training, model API or GPU usage.

NumPy FP64 reference only; Codex must integrate and test production paths
separately. Covariances are per-draw unless explicitly divided by n.
"""
from __future__ import annotations
import math
import numpy as np

ONES = np.ones(4, dtype=np.float64)
B_EQUAL = ONES / 4
B_DROP_W = np.array([0., 0., 1., 0.])
B_PRESERVE_XI = np.array([0., .5, .5, 0.])


def matrix(a, *, columns=None):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Finite two-dimensional matrix required")
    if columns is not None and a.shape[1] != columns:
        raise ValueError("Wrong number of columns")
    return a


def probability(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Finite probability vector required")
    if not math.isclose(float(p.sum()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Probability must sum to one")
    return p


def helmert():
    return np.array([[1/math.sqrt(2), 1/math.sqrt(6), 1/math.sqrt(12)],
                     [-1/math.sqrt(2), 1/math.sqrt(6), 1/math.sqrt(12)],
                     [0., -2/math.sqrt(6), 1/math.sqrt(12)],
                     [0., 0., -3/math.sqrt(12)]])


def stable_pair_weight(log_u, log_b):
    u, b = np.broadcast_arrays(np.asarray(log_u, dtype=float), np.asarray(log_b, dtype=float))
    if not (np.isfinite(u).all() and np.isfinite(b).all()):
        raise ValueError("Finite log probabilities required in this reference")
    return 2*np.tanh((u-b)/2)


def exact_contributions(baseline, candidate, proposal, categories):
    b, u, rho = map(probability, (baseline, candidate, proposal))
    labels = np.asarray(categories)
    if b.shape != u.shape or b.shape != rho.shape or labels.shape != b.shape:
        raise ValueError("Aligned policy and label vectors required")
    if labels.dtype.kind not in 'iu' or np.any((labels < 0) | (labels > 3)):
        raise ValueError("Four-event integer labels required")
    delta = u-b
    if np.any((rho == 0) & (delta != 0)):
        raise ValueError("Proposal does not cover the contrast support")
    w = np.divide(delta, rho, out=np.zeros_like(delta), where=rho>0)
    z = np.eye(4)[labels]*w[:, None]
    return z


def moments(z, rho):
    z, p = matrix(z, columns=4), probability(rho)
    if len(z) != len(p):
        raise ValueError("Probability and contribution lengths differ")
    mu = p@z
    centered = z-mu
    cov = (centered.T*p)@centered
    return mu, cov


def correction(z, b):
    z = np.asarray(z, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if z.shape[-1] != 4 or b.shape != (4,) or not np.isfinite(z).all() or not np.isfinite(b).all():
        raise ValueError("Finite four-event values required")
    return z-z.sum(axis=-1, keepdims=True)*b


def oracle_b(cov):
    cov = matrix(cov, columns=4)
    if cov.shape != (4,4) or not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("Symmetric 4x4 covariance required")
    if np.linalg.eigvalsh(cov).min() < -1e-11*max(1., float(np.trace(cov))):
        raise ValueError("Covariance is not positive semidefinite")
    denom = float(ONES@cov@ONES)
    if denom <= 1e-14*max(float(np.trace(cov)), np.finfo(float).tiny):
        return None
    b = cov@ONES/denom
    return b


def covariance_after(cov, b, n=1):
    if type(n) is not int or n < 1:
        raise ValueError("Positive integer main sample count required")
    cov = matrix(cov, columns=4)
    if cov.shape != (4,4):
        raise ValueError("4x4 covariance required")
    b = np.asarray(b, dtype=np.float64)
    if b.shape != (4,) or not np.isfinite(b).all():
        raise ValueError("Four finite coefficients required")
    a = np.eye(4)-np.outer(b,ONES)
    return a@cov@a.T/n


def pilot_coefficient(samples, *, shrink=0.1, l1_cap=8.0):
    z = matrix(samples, columns=4)
    if len(z)<2 or not 0<=shrink<=1 or not np.isfinite(l1_cap) or l1_cap<1:
        raise ValueError("Pilot needs n>=2, shrink in [0,1], L1 cap>=1")
    cov = np.cov(z, rowvar=False, ddof=1)
    cov = (1-shrink)*cov+shrink*np.trace(cov)/4*np.eye(4)
    b = oracle_b(cov)
    if b is None:
        return B_PRESERVE_XI.copy(), {'status':'PILOT_UNINFORMATIVE','mix':0.0}
    eta = 1.0
    if np.abs(b).sum() > l1_cap:
        lo, hi = 0., 1.
        for _ in range(70):
            mid=(lo+hi)/2
            v=B_EQUAL+mid*(b-B_EQUAL)
            if np.abs(v).sum() <= l1_cap:
                lo=mid
            else:
                hi=mid
        eta=lo
        b=B_EQUAL+eta*(b-B_EQUAL)
    return b, {'status':'PILOT_ESTIMATED','mix':eta}


def span_geometry(fit_rows, query, *, atol=1e-14, rtol=1e-10):
    rows=matrix(fit_rows)
    e=np.asarray(query, dtype=np.float64)
    if e.shape!=(rows.shape[1],) or not np.isfinite(e).all():
        raise ValueError("Aligned finite query required")
    if len(rows):
        u,s,_=np.linalg.svd(rows.T,full_matrices=False)
        threshold=max(atol,rtol*s[0]) if len(s) else atol
        k=int(np.sum(s>threshold)); q=u[:,:k]
    else:
        k=0; q=np.zeros((rows.shape[1],0)); s=np.array([])
    par=q@(q.T@e)
    perp=e-par
    norm=float(np.linalg.norm(e))
    return {'k':k,'Q':q,'parallel':par,'perp':perp,
            'rho':float(np.linalg.norm(perp)/norm) if norm else None,
            'status':'ZERO_QUERY' if not norm else 'NO_CALIBRATION_EXCITATION' if not k else 'GEOMETRY_AVAILABLE'}


def grouped_advantages(categories, lam, *, policy='joint', epsilon=1e-4):
    labels=np.asarray(categories)
    if labels.ndim!=1 or len(labels)<2 or not np.isin(labels,['X','S','W','I']).all():
        raise ValueError("At least two known labels required")
    if policy not in ('joint','no_x_off') or not math.isfinite(lam) or lam<0 or epsilon<=0:
        raise ValueError("Invalid reward configuration")
    effective=0. if policy=='no_x_off' and not np.any(labels=='X') else lam
    rewards=2*(labels=='X')+effective*(labels!='I')
    centered=rewards-rewards.mean()
    return centered/np.sqrt(np.mean(centered**2)+epsilon**2)


def qx_interval(px_low, px_high, v_low, v_high):
    vals=np.array([px_low,px_high,v_low,v_high],float)
    if not np.isfinite(vals).all() or px_low>px_high or v_low>v_high:
        raise ValueError("Finite ordered intervals required")
    if v_low<=0:
        return None
    return (px_low/v_high,px_high/v_low)
