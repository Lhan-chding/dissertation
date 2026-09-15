"""Small independent mathematical reference; not the repository experiment runner.

Requires NumPy. No model loading, network, CUDA, training, or repository writes.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

CATEGORIES = ("X", "S", "W", "I")


def helmert() -> np.ndarray:
    return np.array([
        [1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(12)],
        [-1 / math.sqrt(2), 1 / math.sqrt(6), 1 / math.sqrt(12)],
        [0, -2 / math.sqrt(6), 1 / math.sqrt(12)],
        [0, 0, -3 / math.sqrt(12)],
    ], dtype=np.float64)


def probability(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (4,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("A finite four-event probability vector is required")
    if not math.isclose(float(p.sum()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Probability mass must sum to one")
    return p


def to_nested(p: Sequence[float]) -> dict:
    p = probability(p)
    valid, errors = float(p[:3].sum()), float(p[1:3].sum())
    return {
        "v": valid,
        "q": float(p[0] / valid) if valid > 0 else None,
        "s": float(p[1] / errors) if errors > 0 else None,
        "q_defined": valid > 0,
        "s_defined": errors > 0,
    }


def from_nested(v: float, q: float | None, s: float | None) -> np.ndarray:
    for name, value in (("v", v), ("q", q), ("s", s)):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must lie in [0,1]")
    if v == 0:
        return np.array([0., 0., 0., 1.])
    if q is None:
        raise ValueError("q must be defined when v>0")
    b = v * (1 - q)
    if b == 0:
        return np.array([v, 0., 0., 1-v])
    if s is None:
        raise ValueError("s must be defined when valid errors have positive mass")
    return probability([v*q, b*s, b*(1-s), 1-v])


def nested_jacobian(v: float, q: float, s: float) -> np.ndarray:
    return np.array([
        [q, v, 0.],
        [(1-q)*s, -v*s, v*(1-q)],
        [(1-q)*(1-s), -v*(1-s), -v*(1-q)],
        [-1., 0., 0.],
    ])


def nested_fisher(v: float, q: float, s: float) -> np.ndarray:
    if any(not 0 < x < 1 for x in (v, q, s)):
        raise ValueError("The interior Fisher formula does not apply to boundaries")
    return np.diag([1/(v*(1-v)), v/(q*(1-q)), v*(1-q)/(s*(1-s))])


def mle(counts: Sequence[int]) -> dict:
    counts = list(counts)
    if len(counts) != 4 or any(type(n) is not int or n < 0 for n in counts) or sum(counts) == 0:
        raise ValueError("Four nonnegative integer counts with positive total required")
    p = np.array(counts, dtype=float)/sum(counts)
    return {"p": p, **to_nested(p)}


def advantages(categories: Sequence[str], lam: float, epsilon=1e-4, no_x_off=False):
    if len(categories) < 2 or any(c not in CATEGORIES for c in categories):
        raise ValueError("A nonempty known-category group with K>=2 is required")
    if not math.isfinite(lam) or lam < 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Finite nonnegative lambda and positive epsilon required")
    effective = 0. if no_x_off and "X" not in categories else lam
    rewards = np.array([2.*(c == "X")+effective*(c != "I") for c in categories])
    centered = rewards-rewards.mean()
    return centered / math.sqrt(float(np.mean(centered**2)) + epsilon**2)


def orthogonal_span(deltas: np.ndarray, rtol=1e-10, atol=1e-14):
    d = np.asarray(deltas, dtype=float)
    if d.ndim != 2 or not np.isfinite(d).all():
        raise ValueError("Finite matrix required")
    if d.shape[1] == 0:
        return np.zeros((d.shape[0],0)), np.array([])
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    cutoff = max(atol, rtol*(float(s[0]) if len(s) else 0.))
    return u[:, s > cutoff], s


def truncate_response(r: np.ndarray, rank: int):
    r = np.asarray(r, dtype=float)
    if r.ndim != 2 or not np.isfinite(r).all() or type(rank) is not int or not 0 <= rank <= min(r.shape):
        raise ValueError("Invalid matrix/rank")
    u, s, vt = np.linalg.svd(r, full_matrices=False)
    approx = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    tail = float(s[rank]) if rank < len(s) else 0.
    return approx, tail


def error_bounds(deltas, basis, *, e=0., eps=0., kappa=0., curvature=0.):
    d, u = np.asarray(deltas, float), np.asarray(basis, float)
    if d.ndim != 2 or u.ndim != 2 or d.shape[1] != u.shape[0]:
        raise ValueError("Expected steps-by-parameters and parameters-by-rank")
    if not np.isfinite(d).all() or not np.isfinite(u).all():
        raise ValueError("Finite arrays required")
    if not np.allclose(u.T@u, np.eye(u.shape[1]), atol=1e-12, rtol=1e-12):
        raise ValueError("Basis must be orthonormal")
    if any(not math.isfinite(v) or v < 0 for v in (e, eps, kappa, curvature)):
        raise ValueError("Nonnegative finite error components required")
    projected = d@u
    perpendicular = d-projected@u.T
    total = d.sum(axis=0)
    total_proj = total@u
    total_perp = total-total_proj@u.T
    net = e+eps*np.linalg.norm(total_proj)+kappa*np.linalg.norm(total_perp)+0.5*curvature*float(total@total)
    path = e+eps*np.linalg.norm(projected,axis=1).sum()+kappa*np.linalg.norm(perpendicular,axis=1).sum()+0.5*curvature*np.linalg.norm(d,axis=1).sum()**2
    return {"net":float(net),"path":float(path)}


def softmax(logits):
    z=np.asarray(logits,float)
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError("Finite logits required")
    v=np.exp(z-z.max())
    return v/v.sum()


def event_gradient_affine(matrix, theta, bias, event_weights):
    a=np.asarray(matrix,float)
    p=softmax(a@np.asarray(theta,float)+np.asarray(bias,float))
    w=np.asarray(event_weights,float)
    f=float(p@w)
    return f, a.T@(p*(w-f))


def geometry(left, right):
    a,b=np.asarray(left,float).reshape(-1),np.asarray(right,float).reshape(-1)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Comparable finite vectors required")
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return {"left_norm":float(na),"right_norm":float(nb),
            "distance":float(np.linalg.norm(a-b)),
            "cosine":float(np.clip(a@b/(na*nb),-1,1)) if na*nb>0 else None}


def core_budget(config):
    core=config["profiles"]["core"]
    seed_sets=list(core["seeds_by_role"].values())
    all_seeds=[s for values in seed_sets for s in values]
    if len(all_seeds)!=len(set(all_seeds)):
        raise ValueError("Global seed roles overlap")
    tr=len(all_seeds)*len(core["arms"])
    anchors=tr*len(core["anchors"])
    nbank=sum(len(core[k]) for k in ("fit_bank_indices","diagnostic_bank_indices","evaluation_bank_indices"))
    bks=[v for k in ("fit_bank_indices","diagnostic_bank_indices","evaluation_bank_indices") for v in core[k]]
    if len(bks)!=len(set(bks)):
        raise ValueError("Local bank roles overlap")
    steps=tr*core["steps"]
    forks=anchors*nbank*len(config["fork_interventions"])
    draw=core["B"]*core["K"]
    return {"seed_count":len(all_seeds),"trajectory_count":tr,
            "main_optimizer_steps":steps,"main_sampled_actions":steps*draw,
            "anchor_count":anchors,"fork_optimizer_steps":forks,
            "fork_base_sampled_actions":anchors*nbank*draw,
            "total_optimizer_steps":steps+forks,
            "total_sampled_finite_actions":(steps+anchors*nbank)*draw}
