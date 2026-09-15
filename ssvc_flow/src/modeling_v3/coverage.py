"""Uncentered calibration geometry and deliberately conservative identification.

Rows are realized parameter contrasts in one fixed parameter coordinate system.
Neither a norm ratio nor ridge leverage is a probability or confidence bound.
"""

from __future__ import annotations

import numpy as np


def _matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return value


def build_subspace(updates, *, rank_atol=1e-14, rank_rtol=1e-10):
    """Return the uncentered row-space basis Q[D,k] and rank diagnostics."""
    e = _matrix(updates, "updates")
    if e.shape[1] == 0:
        raise ValueError("updates require a parameter coordinate")
    if not np.isfinite([rank_atol, rank_rtol]).all() or min(rank_atol, rank_rtol) < 0:
        raise ValueError("rank tolerances must be finite and nonnegative")
    _, singular, vt = np.linalg.svd(e, full_matrices=False)
    threshold = max(rank_atol, rank_rtol * singular[0]) if singular.size else rank_atol
    k = int(np.sum(singular > threshold))
    q = vt[:k].T.copy()
    return q, {
        "k": k,
        "D": int(e.shape[1]),
        "m": int(e.shape[0]),
        "centered": False,
        "singular_values": singular.copy(),
        "rank_threshold": float(threshold),
        "min_nonzero_singular_value": float(singular[k - 1]) if k else None,
        "condition_number": float(singular[0] / singular[k - 1]) if k else None,
    }


def coverage_diagnostics(
    fit_updates, query_updates, *, row_covariance=None, alpha=0.0, rank_atol=1e-14, rank_rtol=1e-10
):
    """Report rho and a'(A' Omega^-1 A + alpha I)^-1 a.

    ``row_covariance`` is explicitly the m-by-m covariance/weight metric used
    for this geometric leverage diagnostic, not a silently diagonalized joint
    four-event covariance. Alpha is the absolute ridge in this formula.
    """
    e = _matrix(fit_updates, "fit_updates")
    query = _matrix(query_updates, "query_updates")
    if query.shape[1] != e.shape[1]:
        raise ValueError("fit and query parameter dimensions must match")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    q, meta = build_subspace(e, rank_atol=rank_atol, rank_rtol=rank_rtol)
    parallel = (query @ q) @ q.T
    perpendicular = query - parallel
    norm = np.linalg.norm(query, axis=1)
    perp_norm = np.linalg.norm(perpendicular, axis=1)
    rho = np.divide(perp_norm, norm, out=np.zeros_like(norm), where=norm > 0)
    design = e @ q
    omega_source = "IDENTITY"
    if row_covariance is not None:
        omega = _matrix(row_covariance, "row_covariance")
        if omega.shape != (len(e), len(e)) or not np.allclose(omega, omega.T, atol=1e-12):
            raise ValueError("row_covariance must be symmetric and match fit rows")
        if len(e) and np.linalg.eigvalsh(omega)[0] <= 0:
            raise ValueError("row_covariance must be positive definite; stabilize explicitly")
        normal = design.T @ np.linalg.solve(omega, design)
        omega_source = "SUPPLIED_ROW_COVARIANCE"
    else:
        normal = design.T @ design
    if meta["k"]:
        coordinates = query @ q
        leverage = np.einsum(
            "ni,in->n",
            coordinates,
            np.linalg.solve(normal + alpha * np.eye(meta["k"]), coordinates.T),
        )
    else:
        leverage = np.zeros(len(query), dtype=np.float64)
    return {
        **meta,
        "Q": q,
        "e_parallel": parallel,
        "e_perp": perpendicular,
        "e_norm": norm,
        "e_perp_norm": perp_norm,
        "rho": rho,
        "zero_update": norm == 0,
        "leverage": leverage,
        "leverage_ridge_lambda": float(alpha),
        "leverage_metric": omega_source,
        "leverage_is_calibrated_interval": False,
    }


def _flags(value, n, name):
    arr = np.asarray(value)
    if arr.dtype != bool:
        raise ValueError(f"{name} must contain booleans")
    try:
        return np.broadcast_to(arr, (n,))
    except ValueError as error:
        raise ValueError(f"{name} must be scalar or match query count") from error


def classify_coverage(
    report,
    *,
    rho_threshold,
    leverage_threshold,
    target_excitation=True,
    measurement_resolved=False,
    validated_tolerance=False,
):
    """Apply externally development-frozen thresholds; default to UNKNOWN.

    Target excitation, precision and empirical tolerance validation are distinct
    caller-supplied facts. Geometry alone never certifies a prediction.
    """
    if (
        not np.isfinite([rho_threshold, leverage_threshold]).all()
        or not 0 <= rho_threshold <= 1
        or leverage_threshold < 0
    ):
        raise ValueError("finite rho in [0,1] and nonnegative leverage thresholds required")
    rho = np.asarray(report["rho"])
    n = len(rho)
    excited = _flags(target_excitation, n, "target_excitation")
    resolved = _flags(measurement_resolved, n, "measurement_resolved")
    validated = _flags(validated_tolerance, n, "validated_tolerance")
    labels = []
    for i in range(n):
        if report["zero_update"][i]:
            label = "IDENTICAL_POLICY"
        elif report["k"] == 0 or not excited[i]:
            label = "NO_CALIBRATION_EXCITATION"
        elif rho[i] > rho_threshold:
            label = "OUT_OF_CALIBRATION_SPAN"
        elif report["leverage"][i] > leverage_threshold:
            label = "HIGH_LEVERAGE"
        elif not resolved[i] or not validated[i]:
            label = "MEASUREMENT_UNRESOLVED"
        else:
            label = "PREDICTABLE_AT_VALIDATED_TOLERANCE"
        labels.append(label)
    accepted = [
        label in {"IDENTICAL_POLICY", "PREDICTABLE_AT_VALIDATED_TOLERANCE"} for label in labels
    ]
    return {
        "labels": labels,
        "status": ["IDENTIFIED" if x else "UNKNOWN" for x in accepted],
        "accepted": np.asarray(accepted),
        "rho_threshold": float(rho_threshold),
        "leverage_threshold": float(leverage_threshold),
        "thresholds_source": "CALLER_FROZEN",
    }


def nonidentifiability_witness(fit_updates, query_update, *, kappa=1.0):
    """Construct two zero-sum linear maps indistinguishable on calibration.

    Their unobserved-direction derivative norms are kappa and their query gap
    is 2*kappa*||e_perp||, giving the stated minimax lower bound.
    """
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    query = np.asarray(query_update, dtype=np.float64)
    if query.ndim != 1:
        raise ValueError("query_update must be one parameter vector")
    report = coverage_diagnostics(fit_updates, query[None])
    norm = report["e_perp_norm"][0]
    w = report["e_perp"][0] / norm if norm > 0 else np.zeros_like(query)
    output_direction = np.array([1.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0)
    delta = kappa * np.outer(output_direction, w)
    return {
        "J_minus": -delta,
        "J_plus": delta,
        "w": w,
        "query_gap": 2 * delta @ query,
        "minimax_error_lower_bound": float(kappa * norm),
        "assumed_derivative_bound": float(kappa),
        "unbounded_without_kappa": bool(norm > 0),
        "status": "UNIDENTIFIABLE" if norm > 0 else "NO_UNCOVERED_DIRECTION",
    }
