"""Local realized-update models. Predictors receive paid observations only."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

METHODS = (
    "B0_PERSISTENCE",
    "B1_LOG_RIDGE",
    "B2_RANDOM_Q",
    "B3_UPDATE_PCA",
    "B4_RESPONSE_SVD",
    "B5_WORST_GROUP_RANK",
    "B6_FULL_Q_RIDGE",
    "O0_FULL_J_ORACLE",
    "O1_JQ_SVD_ORACLE",
)


def matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


def build_subspace(updates: np.ndarray, rtol: float = 1e-10, atol: float = 1e-14):
    d = matrix(updates, "updates")
    q, s, _ = np.linalg.svd(d.T, full_matrices=False)
    threshold = max(atol, rtol * float(s[0])) if len(s) else atol
    k = int(np.count_nonzero(s > threshold))
    return q[:, :k], {
        "k": k,
        "singular_values": s.tolist(),
        "rank_threshold": threshold,
        "condition_number": float(s[0] / s[k - 1]) if k else None,
        "alias_or_dependent_columns": len(d) - k,
        "centered": False,
    }


def ridge_map(a: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    a, y = matrix(a, "design"), matrix(y, "responses")
    if len(a) != len(y) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("ridge row mismatch or invalid alpha")
    if a.shape[1] == 0:
        return np.zeros((y.shape[1], 0))
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    if not len(s) or s[0] == 0:
        return np.zeros((y.shape[1], a.shape[1]))
    cutoff = max(1e-14, 1e-10 * s[0])
    weights = np.zeros_like(s)
    mask = s > cutoff
    weights[mask] = 1 / s[mask] if alpha == 0 else s[mask] / (s[mask] ** 2 + alpha * s[0] ** 2)
    return ((y.T @ u) * weights) @ vt


def effective_rank(cap: int | str, k: int) -> int:
    if cap == "FULL":
        return k
    if isinstance(cap, bool) or int(cap) != cap or cap < 0:
        raise ValueError("invalid rank cap")
    return min(int(cap), k)


def fit_local_model(
    updates: np.ndarray,
    responses: np.ndarray,
    method: str,
    rank_cap: int | str,
    alpha: float,
    *,
    random_seed: int = 44001,
    jacobian: np.ndarray | None = None,
    prepared: dict | None = None,
) -> dict:
    """Return fixed U/C and a pure predictor; no evaluator or paths are retained."""
    d, y = matrix(updates, "updates"), matrix(responses, "responses")
    if len(d) != len(y):
        raise ValueError("updates/responses identity length mismatch")
    if method not in METHODS and method not in (
        "B2_FULL_PARAMETER_SKETCH",
        "B4_ENERGY95",
        "B5_EXACT_SELECTED_DIAGNOSTIC",
    ):
        raise ValueError(f"unknown method {method}")
    if prepared is None:
        q, meta = build_subspace(d)
        b = ridge_map(d @ q, y, alpha)
    else:
        q, meta, b = prepared["Q"], prepared["meta"], prepared["B"]
    k = meta["k"]
    r = k if method in ("B6_FULL_Q_RIDGE", "O0_FULL_J_ORACLE") else effective_rank(rank_cap, k)
    spectrum = np.linalg.svd(b, compute_uv=False).tolist() if b.size else []
    if method == "B0_PERSISTENCE":
        r = 0
        u = q[:, :0]
        c = np.zeros((y.shape[1], 0))
    elif method == "O0_FULL_J_ORACLE":
        if jacobian is None:
            raise ValueError("oracle model requires explicit evaluator Jacobian")
        u, c = np.eye(d.shape[1]), matrix(jacobian, "jacobian")
        r = d.shape[1]
    elif method == "O1_JQ_SVD_ORACLE":
        if jacobian is None:
            raise ValueError("oracle model requires explicit evaluator Jacobian")
        jq = matrix(jacobian, "jacobian") @ q
        _, _, vt = np.linalg.svd(jq, full_matrices=False)
        r = min(r, vt.shape[0])
        u = q @ vt[:r].T
        c = jacobian @ u
    elif method == "B2_FULL_PARAMETER_SKETCH":
        sketch, _ = np.linalg.qr(
            np.random.default_rng(random_seed).normal(size=(d.shape[1], max(r, 1)))
        )
        u = sketch[:, :r]
        c = ridge_map(d @ u, y, alpha)
    elif method == "B2_RANDOM_Q":
        rotation, _ = np.linalg.qr(np.random.default_rng(random_seed).normal(size=(k, k)))
        u = q @ rotation[:, :r]
        c = ridge_map(d @ u, y, alpha)
    elif method in (
        "B4_RESPONSE_SVD",
        "B5_WORST_GROUP_RANK",
        "B4_ENERGY95",
        "B5_EXACT_SELECTED_DIAGNOSTIC",
    ):
        _, s, vt = np.linalg.svd(b, full_matrices=False)
        if method == "B4_ENERGY95":
            r = (
                int(np.searchsorted(np.cumsum(s * s), 0.95 * np.sum(s * s)) + 1)
                if np.sum(s * s)
                else 0
            )
        r = min(r, vt.shape[0])
        u = q @ vt[:r].T
        c = ridge_map(d @ u, y, alpha)
    else:
        u = q[:, :r]
        c = ridge_map(d @ u, y, alpha)
    if c.shape != (y.shape[1], u.shape[1]):
        raise ValueError("Jacobian response shape mismatch")

    def predict(new_updates):
        new = matrix(new_updates, "prediction updates")
        if new.shape[1] != d.shape[1]:
            raise ValueError("parameter coordinate mismatch")
        return new @ c.T if method == "O0_FULL_J_ORACLE" else (new @ u) @ c.T

    return {
        **meta,
        "method": method,
        "rank_cap": rank_cap,
        "r": r,
        "alpha": float(alpha),
        "U": u,
        "C": c,
        "Q": q,
        "response_singular_values": spectrum,
        "predict": predict,
        "status": "IDENTIFIED" if k else "NO_IDENTIFIED_UPDATE_SUBSPACE",
    }


def prepare_local(updates: np.ndarray, responses: np.ndarray, alpha: float) -> dict:
    q, meta = build_subspace(updates)
    return {"Q": q, "meta": meta, "B": ridge_map(updates @ q, responses, alpha)}


def log_features(
    p0: np.ndarray,
    logs: np.ndarray,
    updates: np.ndarray,
    operations: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """18 anchor group contrasts + observed logs, realized norm, operation class."""
    from .math_contracts import to_helmert

    group_features = np.concatenate(
        [to_helmert(p0[groups == g].mean(axis=0)) for g in sorted(set(groups.tolist()))]
    )
    logs = matrix(logs, "logs")
    if logs.shape[1] != 11:
        raise ValueError("expected the frozen 11-field training log schema")
    # Update norm and operation already have explicit features below.
    log_fields = logs[:, 1:10]
    op = np.asarray(operations, dtype=int)
    if len(op) != len(logs) or np.any((op < 0) | (op > 2)):
        raise ValueError("operation identity mismatch")
    return np.column_stack(
        (
            np.broadcast_to(group_features, (len(logs), len(group_features))),
            np.linalg.norm(updates, axis=1),
            log_fields,
            np.eye(3)[op],
        )
    )


def fit_log_predictor(features: np.ndarray, responses: np.ndarray, alpha: float) -> dict:
    x, y = matrix(features, "log features"), matrix(responses, "log responses")
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[scale < 1e-12] = 1.0
    return {"scale": scale, "coef": ridge_map(x / scale, y, alpha)}


def array_hash(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()
