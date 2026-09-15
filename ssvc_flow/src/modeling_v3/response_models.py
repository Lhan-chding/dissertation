"""Fit-only V3 response models with explicit four-event output policies.

Every fit uses realized parameter differences, no intercept and no query labels.
RAW4 remains four-output regression; an output constraint is an explicit option
and keeps the raw observations and the unconstrained prediction available.
"""

from __future__ import annotations

import itertools

import numpy as np

from .coverage import _matrix, build_subspace

METHODS = (
    "ZERO",
    "FULL_RIDGE",
    "FULL_GLS",
    "PCA",
    "RANDOM_Q",
    "RESPONSE_SVD",
    "GROUP_WEIGHTED_RESPONSE",
    "RBF_RIDGE",
)
OUTPUT_POLICIES = ("RAW4", "EXPLICIT_ZERO_SUM")


def _helmert():
    h = np.zeros((4, 3))
    for j in range(3):
        h[: j + 1, j] = 1 / np.sqrt((j + 1) * (j + 2))
        h[j + 1, j] = -(j + 1) / np.sqrt((j + 1) * (j + 2))
    return h


def effective_rank(rank_cap, k):
    if rank_cap == "FULL":
        return k
    if (
        isinstance(rank_cap, (bool, np.bool_))
        or not isinstance(rank_cap, (int, np.integer))
        or rank_cap <= 0
    ):
        raise ValueError("rank_cap must be a positive integer or FULL")
    return min(int(rank_cap), k)


def _response_matrix(responses, rows):
    y = np.asarray(responses, dtype=np.float64)
    if y.ndim not in (2, 3) or (y.ndim == 3 and y.shape[-1] != 4):
        raise ValueError("responses require [row,4*prompt] or [row,prompt,4]")
    shape = y.shape[1:]
    y = y.reshape(rows, -1)
    if not np.isfinite(y).all() or y.shape[1] == 0 or y.shape[1] % 4:
        raise ValueError("responses require finite complete X/S/W/I output blocks")
    if np.asarray(responses).shape[0] != rows:
        raise ValueError("responses must match fit update rows")
    return y.copy(), shape


def _ridge(design, target, alpha):
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    lam = alpha * float(s[0] ** 2) if len(s) else 0.0
    tolerance = max(1e-14, float(s[0]) * 1e-10) if len(s) else 1e-14
    keep = s > tolerance
    factors = np.zeros_like(s)
    factors[keep] = s[keep] / (np.square(s[keep]) + lam)
    projected = u.T @ target
    beta = vt.T @ (factors[:, None] * projected if target.ndim == 2 else factors * projected)
    return beta, {
        "ridge_lambda": lam,
        "design_rank": int(keep.sum()),
        "singular_values": s,
        "ridge_scale": "ALPHA_TIMES_DESIGN_SMAX_SQUARED",
    }


def _whiten(covariance, relative_floor, absolute_floor):
    cov = _matrix(covariance, "covariance")
    if cov.shape[0] != cov.shape[1] or not np.allclose(cov, cov.T, rtol=1e-9, atol=1e-12):
        raise ValueError("covariance must be symmetric and square")
    eigenvalues, vectors = np.linalg.eigh((cov + cov.T) / 2)
    scale = float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else 0.0
    if len(eigenvalues) and eigenvalues[0] < -max(1e-12, scale * 1e-9):
        raise ValueError("covariance must be positive semidefinite")
    floor = max(absolute_floor, relative_floor * scale)
    stable = np.maximum(eigenvalues, floor)
    return (vectors / np.sqrt(stable)).T, {
        "spectral_floor": float(floor),
        "eigenvalues_floored": int(np.sum(eigenvalues < floor)),
        "status": "EMPIRICAL_DEGENERATE" if scale == 0 else "SUPPLIED_COVARIANCE",
        "floor_is_measurement_information": False,
    }


def _covariance_blocks(covariance, rows, prompts):
    if covariance is None:
        return None, "IDENTITY_DIAGNOSTIC"
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape == (rows, rows):
        block = np.kron(cov, np.eye(4))
        return np.broadcast_to(
            block, (prompts, 4 * rows, 4 * rows)
        ), "ROW_COVARIANCE_X_EVENT_IDENTITY"
    if cov.shape != (prompts, 4 * rows, 4 * rows):
        raise ValueError("covariance requires [prompt,4*row,4*row] joint blocks or [row,row]")
    return cov.copy(), "JOINT_WITHIN_PROMPT_RAW4_ROW_MAJOR"


def _fit_coefficients(
    a,
    y,
    alpha,
    regression,
    blocks,
    output_policy,
    covariance_relative_floor,
    covariance_absolute_floor,
):
    raw, ordinary_meta = _ridge(a, y, alpha)
    prompts = y.shape[1] // 4
    h = _helmert()
    if regression == "RIDGE" or blocks is None:
        beta = raw.copy()
        if output_policy == "EXPLICIT_ZERO_SUM":
            beta = (raw.reshape(a.shape[1], prompts, 4) @ h @ h.T).reshape(raw.shape)
        return (
            beta,
            raw,
            [
                {
                    **ordinary_meta,
                    "solver": "ORDINARY_SVD",
                    "covariance": "NOT_USED" if regression == "RIDGE" else "IDENTITY_DIAGNOSTIC",
                }
            ],
        )
    beta = np.empty((a.shape[1], y.shape[1]))
    raw = np.empty_like(beta)
    fits = []
    for j in range(prompts):
        transform, report = _whiten(blocks[j], covariance_relative_floor, covariance_absolute_floor)
        target = y[:, 4 * j : 4 * j + 4].reshape(-1)
        raw_design = np.kron(a, np.eye(4))
        raw_coefficients, raw_meta = _ridge(transform @ raw_design, transform @ target, alpha)
        raw[:, 4 * j : 4 * j + 4] = raw_coefficients.reshape(a.shape[1], 4)
        if output_policy == "EXPLICIT_ZERO_SUM":
            # Fit constrained *raw likelihood*. Projecting noisy y first would
            # discard the mass residual and is not equivalent for general GLS.
            design = np.kron(a, h)
            coefficients, meta = _ridge(transform @ design, transform @ target, alpha)
            beta[:, 4 * j : 4 * j + 4] = coefficients.reshape(a.shape[1], 3) @ h.T
        else:
            meta = raw_meta
            beta[:, 4 * j : 4 * j + 4] = raw[:, 4 * j : 4 * j + 4]
        fits.append(
            {
                **meta,
                "solver": "JOINT_WHITENED_SVD",
                "covariance": report,
                "raw_fit": raw_meta,
                "prompt_index": j,
            }
        )
    return beta, raw, fits


def group_xv_map(groups, *, prompt_weights=None):
    """Fixed prompt-weighted group averages of delta pX and primary delta v=-I.

    The alternative noisy RAW4 valid-event sum X+S+W differs from -I by the
    retained mass residual. Only exact or explicitly zero-sum responses make
    these representations equal; the primary mapping preserves raw X and I.
    """
    groups = list(groups)
    weights = (
        np.ones(len(groups)) if prompt_weights is None else np.asarray(prompt_weights, dtype=float)
    )
    if weights.shape != (len(groups),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("prompt_weights must be finite nonnegative prompt weights")
    distinct = list(dict.fromkeys(groups))
    mapping = np.zeros((2 * len(distinct), 4 * len(groups)))
    for j, group in enumerate(distinct):
        indices = [i for i, g in enumerate(groups) if g == group]
        total = weights[indices].sum()
        if total <= 0:
            raise ValueError("each group must have positive total prompt weight")
        for i in indices:
            mapping[2 * j, 4 * i] = weights[i] / total
            mapping[2 * j + 1, 4 * i + 3] = -weights[i] / total
    return mapping


def fit_response_model(
    updates,
    responses,
    method,
    rank_cap="FULL",
    alpha=1e-5,
    *,
    covariance=None,
    output_policy="RAW4",
    regression="RIDGE",
    group_map=None,
    random_seed=0,
    rbf_gamma=None,
    covariance_relative_floor=1e-10,
    covariance_absolute_floor=1e-12,
):
    """Fit all V3 model baselines using calibration data only.

    Full rank direction methods are equivalent under the SAME regression and
    output policy. FULL_GLS always uses GLS; other direction methods can use
    regression='GLS' for clean matched comparisons. RBF_RIDGE uses an origin-
    anchored Gaussian kernel and is exclusively a nonlinearity diagnostic.
    A rank-zero non-ZERO model returns NaN predictions (UNKNOWN), not zeros.
    """
    if method not in METHODS:
        raise ValueError(f"unknown response model {method}")
    if output_policy not in OUTPUT_POLICIES:
        raise ValueError(f"unknown explicit output policy {output_policy}")
    if regression not in ("RIDGE", "GLS"):
        raise ValueError("regression must be RIDGE or GLS")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if (
        not np.isfinite([covariance_relative_floor, covariance_absolute_floor]).all()
        or covariance_relative_floor < 0
        or covariance_absolute_floor <= 0
    ):
        raise ValueError(
            "finite nonnegative relative and positive absolute covariance floors required"
        )
    e = _matrix(updates, "updates").copy()
    if len(e) == 0:
        raise ValueError("fit requires at least one calibration row")
    y, output_shape = _response_matrix(responses, len(e))
    q, geometry = build_subspace(e)
    k = geometry["k"]
    requested_r = effective_rank(rank_cap, k) if not (method == "ZERO" and rank_cap == 0) else 0
    r = (
        0
        if method == "ZERO"
        else k
        if method in ("FULL_RIDGE", "FULL_GLS", "RBF_RIDGE")
        else requested_r
    )
    regression = "GLS" if method == "FULL_GLS" else regression
    if method in ("FULL_RIDGE", "RBF_RIDGE") and regression != "RIDGE":
        raise ValueError(f"{method} requires its declared RIDGE objective")
    blocks, covariance_source = _covariance_blocks(covariance, len(e), y.shape[1] // 4)
    a = e @ q
    directions = q.copy()
    direction_meta = {}
    if method == "RANDOM_Q" and k:
        rotation, _ = np.linalg.qr(np.random.default_rng(random_seed).normal(size=(k, k)))
        directions = q @ rotation[:, :r]
    elif method in ("RESPONSE_SVD", "GROUP_WEIGHTED_RESPONSE") and k:
        full_beta, _, _ = _fit_coefficients(
            a,
            y,
            alpha,
            regression,
            blocks,
            output_policy,
            covariance_relative_floor,
            covariance_absolute_floor,
        )
        weighted = full_beta
        if method == "GROUP_WEIGHTED_RESPONSE":
            if group_map is None:
                mapping = group_xv_map(range(y.shape[1] // 4))
                direction_meta["group_weighting"] = "DEFAULT_EQUAL_PER_PROMPT_XV"
            else:
                mapping = _matrix(group_map, "group_map")
                direction_meta["group_weighting"] = "CALLER_FIXED_GROUP_MAP"
            if mapping.shape[1] != y.shape[1] or not len(mapping):
                raise ValueError("group_map must map all flattened event outputs")
            weighted = full_beta @ mapping.T
            direction_meta["group_map"] = mapping.copy()
        rotation, spectrum, _ = np.linalg.svd(weighted, full_matrices=True)
        directions = q @ rotation[:, :r]
        direction_meta["response_singular_values"] = spectrum
    elif method == "PCA":
        directions = q[:, :r]
    elif method == "ZERO":
        directions = q[:, :0]

    def query_coordinates(query):
        query = _matrix(query, "query_updates")
        if query.shape[1] != e.shape[1]:
            raise ValueError("query parameter dimension must match calibration")
        return query

    metadata = {
        "output_policy": output_policy,
        "observation_channels": "RAW_X_S_W_I",
        "regression": regression,
        "covariance_source": covariance_source,
        "cross_prompt_covariance": "INDEPENDENT_PROMPT_BLOCKS_ASSUMED",
        "alpha": float(alpha),
        "random_seed": int(random_seed),
        "intercept": False,
        "fit_only": True,
        "query_labels_read": False,
        "secondary_only": method == "RBF_RIDGE",
        "direction": direction_meta,
        "full_rank_equivalence_group": f"{regression}/{output_policy}" if r == k and k else None,
        "primary_direction_comparison_eligible": bool(0 < r < k),
        "fits": [],
    }
    status = (
        "STATISTICAL_BASELINE_ONLY"
        if method == "ZERO"
        else "NO_CALIBRATION_EXCITATION"
        if k == 0
        else "FIT_AVAILABLE_NOT_VALIDATED"
    )
    coefficients = np.empty((r, y.shape[1]))
    raw_coefficients = np.empty_like(coefficients)
    if method == "ZERO" or k == 0:
        fill = 0.0 if method == "ZERO" else np.nan

        def predict(query):
            query = query_coordinates(query)
            return np.full((len(query), *output_shape), fill)

        predict_raw = predict
    elif method == "RBF_RIDGE":
        squared_norm = np.square(a).sum(axis=1)
        squared_distances = np.maximum(squared_norm[:, None] + squared_norm[None] - 2 * a @ a.T, 0)
        positive = squared_distances[squared_distances > 0]
        gamma = (
            float(rbf_gamma)
            if rbf_gamma is not None
            else 1 / float(np.median(positive))
            if positive.size
            else 1.0
        )
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("rbf_gamma must be finite and positive")

        def kernel(query):
            coordinates = query @ q
            norms = np.square(coordinates).sum(axis=1)
            distance = np.maximum(norms[:, None] + squared_norm[None] - 2 * coordinates @ a.T, 0)
            return (
                np.exp(-gamma * distance)
                - np.exp(-gamma * norms[:, None])
                - np.exp(-gamma * squared_norm[None])
                + 1
            )

        gram = kernel(e)
        eig, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
        lam = float(alpha * max(float(eig[-1]), 0))
        keep = eig > max(1e-14, float(eig[-1]) * 1e-10)
        factors = np.zeros_like(eig)
        factors[keep] = 1 / (eig[keep] + lam)
        raw_coefficients = eigenvectors @ (factors[:, None] * (eigenvectors.T @ y))
        coefficients = raw_coefficients.copy()
        if output_policy == "EXPLICIT_ZERO_SUM":
            coefficients = (
                coefficients.reshape(len(e), -1, 4) @ _helmert() @ _helmert().T
            ).reshape(coefficients.shape)

        def predict(query):
            return (kernel(query_coordinates(query)) @ coefficients).reshape((-1, *output_shape))

        def predict_raw(query):
            return (kernel(query_coordinates(query)) @ raw_coefficients).reshape(
                (-1, *output_shape)
            )

        metadata.update(
            rbf_gamma=gamma,
            rbf_gamma_source="CALLER_FROZEN"
            if rbf_gamma is not None
            else "FIT_GEOMETRY_MEDIAN_DISTANCE",
            kernel="GAUSSIAN_ORIGIN_ANCHORED",
            fits=[{"ridge_lambda": lam, "solver": "KERNEL_EIGH"}],
            full_rank_equivalence_group=None,
            coefficient_space="FIT_KERNEL_CENTERS",
        )
    else:
        coefficients, raw_coefficients, fits = _fit_coefficients(
            e @ directions,
            y,
            alpha,
            regression,
            blocks,
            output_policy,
            covariance_relative_floor,
            covariance_absolute_floor,
        )
        metadata["fits"] = fits

        def predict(query):
            return ((query_coordinates(query) @ directions) @ coefficients).reshape(
                (-1, *output_shape)
            )

        def predict_raw(query):
            return ((query_coordinates(query) @ directions) @ raw_coefficients).reshape(
                (-1, *output_shape)
            )

    return {
        "method": method,
        "rank_cap": rank_cap,
        "k": k,
        "r": r,
        "D": e.shape[1],
        "status": status,
        "Q": q,
        "directions": directions,
        "coefficients": coefficients,
        "raw_coefficients": raw_coefficients,
        "predict": predict,
        "predict_raw": predict_raw,
        "metadata": metadata,
        "geometry": geometry,
        "raw_observations": y.reshape((len(e), *output_shape)),
        "raw_observation_mass": y.reshape(len(e), -1, 4).sum(axis=-1).squeeze(axis=1)
        if y.shape[1] == 4
        else y.reshape(len(e), -1, 4).sum(axis=-1),
    }


def error_decomposition(truth, full_linear, projected_linear, exact_fit, finite_fit):
    """Keep four telescoping vectors and all signed 2<a,b> cross terms.

    These algebraic components are diagnostics, not independent causal shares.
    Per-unit squared norms retain every leading observation axis.
    """
    values = [
        np.asarray(x, dtype=float)
        for x in (truth, full_linear, projected_linear, exact_fit, finite_fit)
    ]
    if (
        not values[0].ndim
        or any(x.shape != values[0].shape for x in values)
        or any(not np.isfinite(x).all() for x in values)
    ):
        raise ValueError("decomposition requires finite equally shaped response arrays")
    names = ("nonlinearity", "span_omission", "exact_calibration_fit", "finite_observation_fit")
    components = {name: values[i] - values[i + 1] for i, name in enumerate(names)}
    squared = {name: float(np.square(value).sum()) for name, value in components.items()}
    cross = {
        f"{left}__{right}": float(2 * np.sum(components[left] * components[right]))
        for left, right in itertools.combinations(names, 2)
    }
    per_unit = {name: np.square(value).sum(axis=-1) for name, value in components.items()}
    per_unit_cross = {
        f"{left}__{right}": 2 * np.sum(components[left] * components[right], axis=-1)
        for left, right in itertools.combinations(names, 2)
    }
    total = values[0] - values[-1]
    total_squared = float(np.square(total).sum())
    return {
        "components": components,
        "squared_norms": squared,
        "cross_terms": cross,
        "per_unit_squared_norms": per_unit,
        "per_unit_cross_terms": per_unit_cross,
        "total_error": total,
        "total_squared_error": total_squared,
        "reconstruction_residual": total_squared - sum(squared.values()) - sum(cross.values()),
        "causal_fraction_interpretation": False,
    }
