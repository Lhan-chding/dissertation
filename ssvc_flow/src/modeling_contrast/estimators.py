"""Fit-only local contrast models and two-stage parameter projections.

All coefficients use row-major candidate-contrast / Helmert coordinates.  The
GLS solver whitens the complete within-prompt covariance before an SVD; it does
not replace joint covariance by marginal variances or form a matrix inverse.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..modeling_qualification.models import build_subspace, matrix, ridge_map

METHOD_ALIASES = {
    "C0_ZERO": "C0",
    "C2_CONTRAST_OLS_FULL": "C2",
    "C3_CONTRAST_GLS_FULL": "C3",
    "C4_UPDATE_PCA": "C4_PCA",
    "C4_PCA_RANDOM": "C4_PCA",
    "C5_UNWEIGHTED_RESPONSE": "C5",
    "C6_GROUP_AWARE_RESPONSE": "C6",
}
METHODS = ("C0", "C2", "C3", "C4_PCA", "C4_RANDOM", "C5", "C6")
GLS_METHODS = ("C3", "C5", "C6")


def _responses(value: Any, rows: int | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 3 and value.shape[-1] == 3:
        value = value.reshape(value.shape[0], -1)
    y = matrix(value, "responses")
    if y.shape[1] == 0 or y.shape[1] % 3 or (rows is not None and len(y) != rows):
        raise ValueError("responses require matching fit rows and three coordinates per prompt")
    return y


def effective_rank(cap: int | str, k: int) -> int:
    """Nonzero ablations retain min(requested, fit-update rank), including zero y."""
    if cap == "FULL":
        return k
    if isinstance(cap, (bool, np.bool_)) or not isinstance(cap, (int, np.integer)) or cap <= 0:
        raise ValueError("nonzero rank cap must be a positive integer or FULL")
    return min(int(cap), k)


def _alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("ridge alpha must be finite and nonnegative")
    return alpha


def _active_rows(active_rows, row_count):
    if active_rows is None:
        return np.arange(row_count)
    rows = np.asarray(active_rows)
    if rows.ndim != 1:
        raise ValueError("active rows must be a one-dimensional index list or mask")
    if rows.dtype == bool:
        if len(rows) != row_count:
            raise ValueError("active row mask must match fit row count")
        return np.flatnonzero(rows)
    if rows.size == 0:
        return np.array([], dtype=int)
    if not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0) or np.any(rows >= row_count):
        raise ValueError("active row indices must be integers within fit rows")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("active row indices cannot contain duplicates")
    return rows.astype(int)


def _svd_plan(design: np.ndarray) -> dict:
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    threshold = max(1e-14, 1e-10 * float(s[0])) if len(s) else 1e-14
    return {"u": u, "s": s, "vt": vt, "threshold": threshold}


def _solve_plan(plan: dict, target: np.ndarray, alpha: float) -> tuple[np.ndarray, dict]:
    s = plan["s"]
    lam = alpha * float(s[0] ** 2) if len(s) else 0.0
    mask = s > plan["threshold"]
    factors = np.zeros_like(s)
    factors[mask] = s[mask] / (s[mask] ** 2 + lam)
    beta = plan["vt"].T @ (factors * (plan["u"].T @ target))
    return beta, {
        "ridge_lambda": lam,
        "whitened_design_max_singular_value": float(s[0]) if len(s) else 0.0,
        "whitened_design_rank": int(mask.sum()),
        "rank_threshold": float(plan["threshold"]),
    }


def _covariance_plan(covariance, prompt_count, m, eta, observation_n, observation_scale):
    from .covariance import shrink_covariance

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (prompt_count, 3 * m, 3 * m) or not np.isfinite(covariance).all():
        raise ValueError("covariance must have shape (prompt_count, 3*m, 3*m)")
    transforms, reports = [], []
    for cov in covariance:
        stabilized, report = shrink_covariance(
            cov, eta=eta, observation_scale=observation_scale, n=observation_n
        )
        eigenvalues, eigenvectors = np.linalg.eigh(stabilized)
        if np.any(eigenvalues <= 0):
            raise ValueError("stabilized covariance must be positive definite")
        transforms.append((eigenvectors / np.sqrt(eigenvalues)).T)
        reports.append(report)
    return transforms, reports


def _gls_plans(a: np.ndarray, transforms: list[np.ndarray]) -> list[dict]:
    design = np.kron(a, np.eye(3))
    return [_svd_plan(transform @ design) for transform in transforms]


def _fit_gls(a, y, alpha, transforms, covariance_reports, plans=None):
    plans = _gls_plans(a, transforms) if plans is None else plans
    b = np.zeros((y.shape[1], a.shape[1]))
    reports = []
    for j, (transform, plan) in enumerate(zip(transforms, plans, strict=True)):
        target = y[:, 3 * j : 3 * j + 3].reshape(-1)
        beta, report = _solve_plan(plan, transform @ target, alpha)
        b[3 * j : 3 * j + 3] = beta.reshape(a.shape[1], 3).T
        reports.append({**report, "covariance": covariance_reports[j]})
    return b, {
        "solver": "WHITENED_SVD",
        "vectorization": "ROW_MAJOR_CONTRAST_THEN_HELMERT",
        "ridge_scale": "alpha_times_whitened_design_smax_squared",
        "covariance_source": "SUPPLIED_JOINT_WITHIN_PROMPT",
        "cross_prompt_sampling": "INDEPENDENT_PROMPT_BLOCKS",
        "prompt_fits": reports,
    }


def gls_ridge_map(
    design: np.ndarray,
    responses: np.ndarray,
    covariance: np.ndarray,
    alpha: float,
    *,
    eta: float = 0.01,
    observation_n: int | None = None,
    observation_scale: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit m-by-k design to m-by-(3p) responses using p joint covariance blocks."""
    a = matrix(design, "design")
    y = _responses(responses, len(a))
    alpha = _alpha(alpha)
    transforms, reports = _covariance_plan(
        covariance, y.shape[1] // 3, len(a), eta, observation_n, observation_scale
    )
    return _fit_gls(a, y, alpha, transforms, reports)


def prepare_contrast(
    updates: np.ndarray,
    responses: np.ndarray,
    *,
    covariance: np.ndarray | None = None,
    eta: float = 0.01,
    observation_n: int | None = None,
    observation_scale: float | None = None,
    active_rows: np.ndarray | list | None = None,
) -> dict:
    """Cache only fit-data Q, covariance whitening and full-design SVDs.

    A prepared object can be reused across alpha/r grids for this same fit data
    and eta.  It contains no evaluation updates, response truth, or callbacks.
    """
    original_e = matrix(updates, "updates").copy()
    original_y = _responses(responses, len(original_e)).copy()
    if len(original_e) == 0 or original_e.shape[1] == 0:
        raise ValueError("at least one fit update and parameter coordinate required")
    rows = _active_rows(active_rows, len(original_e))
    e, y = original_e[rows], original_y[rows]
    q, meta = build_subspace(e)
    meta.update(
        retained_fit_rows=rows.tolist(),
        excluded_fit_rows=np.setdiff1d(np.arange(len(original_e)), rows).tolist(),
        row_selection="CALLER_POLICY_IDENTITY_ONLY" if active_rows is not None else "ALL_FIT_ROWS",
    )
    a = e @ q
    result = {
        "input_updates": original_e,
        "input_responses": original_y,
        "active_rows": rows,
        "updates": e,
        "responses": y,
        "Q": q,
        "meta": meta,
        "A": a,
        "eta": float(eta),
        "covariance": None,
        "transforms": None,
        "covariance_reports": [],
        "full_gls_plans": None,
        "coefficient_cache": {},
    }
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=np.float64)
        order = 3 * len(original_e)
        if covariance.shape != (y.shape[1] // 3, order, order) or not np.isfinite(covariance).all():
            raise ValueError("covariance must match original prompt and fit contrast coordinates")
        result["covariance"] = covariance.copy()
        if not len(rows):
            return result
        coordinates = (3 * rows[:, None] + np.arange(3)).reshape(-1)
        selected_covariance = covariance[:, coordinates][:, :, coordinates]
        transforms, reports = _covariance_plan(
            selected_covariance, y.shape[1] // 3, len(e), eta, observation_n, observation_scale
        )
        result.update(
            transforms=transforms,
            covariance_reports=reports,
            full_gls_plans=_gls_plans(a, transforms),
        )
    return result


def _fit_coefficients(prepared, coordinates, y, alpha, use_gls, *, full=False):
    a = prepared["updates"] @ coordinates
    if use_gls and prepared["transforms"] is not None:
        return _fit_gls(
            a,
            y,
            alpha,
            prepared["transforms"],
            prepared["covariance_reports"],
            prepared["full_gls_plans"] if full else None,
        )
    return ridge_map(a, y, alpha), {
        "solver": "ORDINARY_SVD",
        "ridge_scale": "alpha_times_design_smax_squared",
        "covariance_source": "IDENTITY_NO_COVARIANCE" if use_gls else "OLS_UNWEIGHTED",
        "prompt_fits": [],
    }


def fit_contrast_model(
    updates: np.ndarray,
    responses: np.ndarray,
    method: str,
    rank_cap: int | str = "FULL",
    alpha: float = 1e-5,
    *,
    covariance: np.ndarray | None = None,
    eta: float = 0.01,
    group_map: np.ndarray | None = None,
    random_seed: int = 44001,
    observation_n: int | None = None,
    observation_scale: float | None = None,
    direction_responses: np.ndarray | None = None,
    coefficient_responses: np.ndarray | None = None,
    prepared: dict | None = None,
    active_rows: np.ndarray | list | None = None,
) -> dict:
    """Fit C0/C2-C6 without any test data or automatic rank-zero fallback.

    ``direction_responses`` and ``coefficient_responses`` explicitly supply
    exact *fit-bank* truth for development-only error isolation.  They must not
    contain evaluation banks.  ``covariance=None`` gives the identity objective
    for exact-target diagnostics and is recorded; finite GLS callers must supply
    the paid joint covariance. ``active_rows`` is an explicit caller decision
    based on endpoint policy identities; equality of updates alone never removes
    rows. Selection precedes covariance stabilization. C1 remains the unchanged
    parent implementation.
    """
    original_method = method
    method = METHOD_ALIASES.get(method, method)
    if method not in METHODS:
        raise ValueError(f"unknown contrast model {method}")
    alpha = _alpha(alpha)
    e = matrix(updates, "updates")
    y = _responses(responses, len(e))
    if prepared is None:
        prepared = prepare_contrast(
            e,
            y,
            covariance=covariance,
            eta=eta,
            observation_n=observation_n,
            observation_scale=observation_scale,
            active_rows=active_rows,
        )
    elif (
        not np.array_equal(e, prepared["input_updates"])
        or not np.array_equal(y, prepared["input_responses"])
        or float(eta) != prepared["eta"]
        or (covariance is not None and not np.array_equal(covariance, prepared["covariance"]))
        or (
            active_rows is not None
            and not np.array_equal(_active_rows(active_rows, len(e)), prepared["active_rows"])
        )
    ):
        raise ValueError("prepared object must match exact fit arrays, covariance and eta")
    direction_y = y if direction_responses is None else _responses(direction_responses, len(e))
    coefficient_y = (
        y if coefficient_responses is None else _responses(coefficient_responses, len(e))
    )
    if direction_y.shape != y.shape or coefficient_y.shape != y.shape:
        raise ValueError("oracle fit responses must preserve all prompt coordinate identities")
    rows = prepared["active_rows"]
    direction_y, coefficient_y = direction_y[rows], coefficient_y[rows]
    e, y = prepared["updates"], prepared["responses"]
    q, meta = prepared["Q"], prepared["meta"]
    k = meta["k"]
    requested_r = 0 if method == "C0" else effective_rank(rank_cap, k)
    r = k if method in ("C2", "C3") else requested_r
    if method == "C6":
        if group_map is None:
            raise ValueError("C6 requires the frozen group_XV linear map")
        group_map = matrix(group_map, "group map")
        if group_map.shape[1] != y.shape[1] or group_map.shape[0] == 0:
            raise ValueError("group map must match flattened prompt Helmert coordinates")
    use_gls = method in GLS_METHODS
    spectrum, weights, full_report = [], {}, {}
    if method == "C0" or not k:
        u, c = q[:, :0], np.zeros((y.shape[1], 0))
        r, fit_report = 0, {"solver": "NO_FIT", "prompt_fits": []}
    else:
        if method in ("C5", "C6"):
            cache_key = (alpha, use_gls, direction_y.tobytes())
            if cache_key not in prepared["coefficient_cache"]:
                prepared["coefficient_cache"][cache_key] = _fit_coefficients(
                    prepared, q, direction_y, alpha, use_gls, full=True
                )
            b, full_report = prepared["coefficient_cache"][cache_key]
            target = b
            if method == "C6":
                weights = {"full": 1 / np.sqrt(y.shape[1]), "group_XV": 1 / np.sqrt(len(group_map))}
                target = np.vstack((b * weights["full"], (group_map @ b) * weights["group_XV"]))
            # Complete right basis preserves requested nonzero ranks even when
            # all observed response coefficients vanish or response width < k.
            _, singular, vt = np.linalg.svd(target, full_matrices=target.shape[0] < k)
            spectrum = singular.tolist()
            u = q @ vt[:r].T
        elif method == "C4_RANDOM":
            rotation, _ = np.linalg.qr(np.random.default_rng(random_seed).normal(size=(k, k)))
            u = q @ rotation[:, :r]
        else:
            u = q[:, :r]
        if method in ("C5", "C6") and r == k:
            # An orthogonal full-basis change leaves residuals, ||beta|| and
            # the design singular values unchanged: this is the same refit.
            key = (alpha, use_gls, coefficient_y.tobytes())
            if key not in prepared["coefficient_cache"]:
                prepared["coefficient_cache"][key] = _fit_coefficients(
                    prepared, q, coefficient_y, alpha, use_gls, full=True
                )
            full_c, fit_report = prepared["coefficient_cache"][key]
            c = full_c @ (q.T @ u)
            fit_report = {**fit_report, "refit_equivalence": "ORTHOGONAL_FULL_BASIS_CHANGE"}
        else:
            c, fit_report = _fit_coefficients(
                prepared,
                u,
                coefficient_y,
                alpha,
                use_gls,
                full=method in ("C2", "C3"),
            )
    # Keep retained fitted state independent of caller-owned arrays/caches.
    q, u, c = q.copy(), u.copy(), c.copy()
    p = e.shape[1]

    def predict(new_updates):
        new = matrix(new_updates, "prediction updates")
        if new.shape[1] != p:
            raise ValueError("prediction parameter coordinate mismatch")
        return (new @ u) @ c.T

    diagnostic = "FINITE_U_FINITE_COEFFICIENTS"
    if direction_responses is not None and coefficient_responses is not None:
        diagnostic = "EXACT_U_EXACT_COEFFICIENTS"
    elif direction_responses is not None:
        diagnostic = "EXACT_U_FINITE_COEFFICIENTS"
    elif coefficient_responses is not None:
        diagnostic = "FINITE_U_EXACT_COEFFICIENTS"
    return {
        **meta,
        "method": original_method,
        "canonical_method": method,
        "rank_cap": rank_cap,
        "r": int(r),
        "effective_r": int(r),
        "alpha": alpha,
        "eta": float(eta),
        "U": u,
        "C": c,
        "Q": q,
        "predict": predict,
        "response_singular_values": spectrum,
        "spectrum_block_weights": weights,
        "coefficient_fit": fit_report,
        "direction_fit": full_report,
        "diagnostic": diagnostic,
        "development_oracle": direction_responses is not None or coefficient_responses is not None,
        "status": "NO_CONTRAST_EXCITATION"
        if not k
        else ("ZERO_BASELINE" if method == "C0" else "IDENTIFIED"),
        "fit_only_subspace": True,
        "parameter_count": p,
        "response_coordinates": y.shape[1],
    }


def out_of_subspace(updates: np.ndarray, q: np.ndarray) -> dict:
    e, q = matrix(updates, "updates"), matrix(q, "Q")
    if e.shape[1] != len(q) or not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-9):
        raise ValueError("Q must be an orthonormal basis matching update coordinates")
    norms = np.linalg.norm(e, axis=1)
    residual = np.linalg.norm(e - (e @ q) @ q.T, axis=1)
    fractions = np.divide(residual, norms, out=np.zeros_like(norms), where=norms > 0)
    return {
        "per_update_fraction": fractions.tolist(),
        "pooled_energy_fraction": float(residual @ residual / (norms @ norms))
        if norms @ norms
        else 0.0,
        "zero_update_count": int(np.count_nonzero(norms == 0)),
    }


def subspace_distance(u: np.ndarray, v: np.ndarray) -> dict:
    u, v = matrix(u, "first subspace"), matrix(v, "second subspace")
    if len(u) != len(v):
        raise ValueError("subspace ambient dimensions differ")
    for basis in (u, v):
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-9):
            raise ValueError("subspace diagnostic requires orthonormal columns")
    overlap = np.linalg.svd(u.T @ v, compute_uv=False)
    distance_sq = max(0.0, float(u.shape[1] + v.shape[1] - 2 * np.sum(overlap**2)))
    return {
        "first_rank": u.shape[1],
        "second_rank": v.shape[1],
        "projector_frobenius_distance": float(np.sqrt(distance_sq)),
        "principal_angles_radians": np.arccos(np.clip(overlap, 0, 1)).tolist(),
    }


def group_xv_map(groups: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Fixed within-group normalized pX/v map, rows interleaved group, metric."""
    from ..modeling_qualification.math_contracts import helmert

    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("one fixed group identity required per prompt")
    weights = np.ones(len(groups)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != groups.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("finite nonnegative fixed prompt weights required")
    labels = np.unique(groups)
    result = np.zeros((2 * len(labels), 3 * len(groups)))
    h = helmert()
    event_rows = np.stack((h[0], -h[3]))
    for j, group in enumerate(labels):
        indices = np.flatnonzero(groups == group)
        total = float(weights[indices].sum())
        if total <= 0:
            raise ValueError("each group requires positive total weight")
        for i in indices:
            result[2 * j : 2 * j + 2, 3 * i : 3 * i + 3] = event_rows * (weights[i] / total)
    return result
