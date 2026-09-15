"""Independent NumPy formula witnesses, not the repository training implementation.

No model loading, training, network calls or policy-effect claims. Finite tables
are used only to test estimator algebra and covariance identities.
"""
from __future__ import annotations
import math
import numpy as np

EVENTS = ('X', 'S', 'W', 'I')


def probability(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('A finite nonnegative probability vector is required')
    if not np.isclose(p.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError('Probabilities must sum to one')
    return p


def helmert():
    h = np.zeros((4, 3), dtype=float)
    for j in range(1, 4):
        h[:j, j-1] = 1 / math.sqrt(j*(j+1))
        h[j, j-1] = -j / math.sqrt(j*(j+1))
    return h


def conditional_coordinates(p):
    p = probability(p)
    if len(p) != 4:
        raise ValueError('Four event probabilities required')
    v = float(p[:3].sum())
    return {'v': v, 'q': float(p[0]/v) if v else None,
            's': float(p[1]/p[1:3].sum()) if p[1:3].sum() else None}


def from_conditionals(v, q, s):
    vals = np.asarray([v, q, s], dtype=float)
    if not np.isfinite(vals).all() or np.any(vals < 0) or np.any(vals > 1):
        raise ValueError('Interior/boundary numbers in [0,1] required')
    return np.array([v*q, v*(1-q)*s, v*(1-q)*(1-s), 1-v])


def independent_count_covariance(p, n):
    p = probability(p)
    if type(n) is not int or n <= 0:
        raise ValueError('Positive integer n required')
    return (np.diag(p)-np.outer(p,p))/n


def contrast_covariance(level_cov, contrast_matrix, coordinates=1):
    cov = np.asarray(level_cov, dtype=float)
    l = np.asarray(contrast_matrix, dtype=float)
    if l.ndim != 2 or not np.allclose(l.sum(axis=1), 0):
        raise ValueError('Each contrast must sum to zero')
    t = np.kron(l, np.eye(coordinates))
    if cov.shape != (t.shape[1], t.shape[1]):
        raise ValueError('Covariance shape mismatch')
    return t @ cov @ t.T


def weighted_moments(values, p):
    p = probability(p)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(p) or not np.isfinite(values).all():
        raise ValueError('Finite values must match distribution')
    mean = p @ values
    centered = values - mean
    return mean, (centered.T*p) @ centered


def lr_moments(p0, p1, rho, features):
    """Exact one-draw moments for f(a)*(p1-p0)/rho, used as an oracle witness."""
    p0, p1, rho = map(probability, (p0,p1,rho))
    if p0.shape != p1.shape or p0.shape != rho.shape:
        raise ValueError('Policy support arrays differ')
    diff = p1-p0
    if np.any((rho == 0) & (diff != 0)):
        raise ValueError('Proposal does not cover the contrast support')
    w = np.divide(diff, rho, out=np.zeros_like(diff), where=rho > 0)
    f = np.asarray(features, dtype=float)
    if f.ndim == 1:
        f = f[:, None]
    mean, cov = weighted_moments(w[:,None]*f, rho)
    return mean, cov, w


def inverse_cdf_joint(p0, p1):
    """Joint probability table under shared U and fixed action ordering."""
    p0,p1=probability(p0),probability(p1)
    if p0.shape != p1.shape:
        raise ValueError('Common action space required')
    c0=np.r_[0,np.cumsum(p0)]; c1=np.r_[0,np.cumsum(p1)]
    n=len(p0)
    j=np.empty((n,n))
    for a in range(n):
        for b in range(n):
            j[a,b]=max(0,min(c0[a+1],c1[b+1])-max(c0[a],c1[b]))
    return j


def crn_moments(p0,p1,f):
    j=inverse_cdf_joint(p0,p1)
    f=np.asarray(f,dtype=float)
    if f.ndim == 1:
        f=f[:,None]
    z=f[None,:,:]-f[:,None,:]
    return weighted_moments(z.reshape(-1,f.shape[1]),j.ravel())


def stable_exp_difference(loga,logb):
    """exp(loga)-exp(logb); rejects overflow rather than clipping weights."""
    a,b=np.broadcast_arrays(np.asarray(loga,float),np.asarray(logb,float))
    if np.any(np.isnan(a) | np.isnan(b) | np.isposinf(a) | np.isposinf(b)):
        raise ValueError('No NaN or positive infinity in log weights')
    maximum=np.maximum(a,b)
    if np.any(maximum > math.log(np.finfo(float).max)):
        raise FloatingPointError('Weight overflow; no clipping allowed')
    with np.errstate(invalid='ignore'):
        difference=a-b
        result=np.sign(difference)*np.exp(maximum)*(-np.expm1(-np.abs(difference)))
    return np.where(np.isneginf(maximum),0.,result)


def gls_ridge(x,y,cov,penalty=0.):
    """Whiten via Cholesky; no inverse. Scalar/vector output for test witnesses."""
    x,y,cov=map(lambda v: np.asarray(v,dtype=float),(x,y,cov))
    if not all(np.isfinite(v).all() for v in (x,y,cov)) or penalty<0:
        raise ValueError('Finite data and nonnegative penalty required')
    if x.ndim != 2 or cov.shape!=(len(x),len(x)) or len(y)!=len(x):
        raise ValueError('Incompatible shapes')
    chol=np.linalg.cholesky(cov)
    xw=np.linalg.solve(chol,x); yw=np.linalg.solve(chol,y)
    return np.linalg.solve(xw.T@xw+penalty*np.eye(x.shape[1]),xw.T@yw)


def contrast_taylor_bound(lipschitz, main_update, auxiliary_increment):
    if not math.isfinite(lipschitz) or lipschitz<0:
        raise ValueError('Nonnegative finite L required')
    d,e=np.asarray(main_update,float),np.asarray(auxiliary_increment,float)
    return float(lipschitz*np.linalg.norm(d)*np.linalg.norm(e)+
                 .5*lipschitz*np.linalg.norm(e)**2)


def resolution_label(interval, delta):
    if interval is None:
        return 'UNKNOWN'
    lo,hi=interval
    if not all(math.isfinite(x) for x in [lo,hi,delta]) or lo>hi or delta<=0:
        raise ValueError('Valid finite interval and positive tolerance required')
    if lo > delta:
        return 'MEANINGFUL_POSITIVE'
    if hi < -delta:
        return 'MEANINGFUL_NEGATIVE'
    if lo >= -delta and hi <= delta:
        return 'WITHIN_TOLERANCE'
    return 'UNKNOWN'


def fresh_budget(calibration_seeds=6,test_seeds=10,arms=2,steps=64,
                 anchors=3,banks=14,candidates=3,b=4,k=8):
    trajectories=(calibration_seeds+test_seeds)*arms
    main=trajectories*steps
    bank_draws=trajectories*anchors*banks
    fork=bank_draws*candidates
    return {'trajectories':trajectories,'main_updates':main,'fork_updates':fork,
            'total_updates':main+fork,'training_and_bank_actions':(main+bank_draws)*b*k}
