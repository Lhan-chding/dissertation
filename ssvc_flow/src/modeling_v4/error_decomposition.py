"""Exact evaluator diagnostics; these derivatives never supply fitted labels.

The three signed terms sum to reference minus prediction. Their squared norms
alone do not add to the error: all three doubled inner products are retained.
"""

from __future__ import annotations

import numpy as np


def exact_toy_geometry(theta, features, categories):
    """Actual 737-parameter tanh policy, without a surrogate linear softmax."""
    from src.modeling_qualification.toy import _numpy_forward, event_jacobian

    theta = np.asarray(theta, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    categories = np.asarray(categories)
    if (
        theta.shape != (737,)
        or features.ndim != 3
        or features.shape[1:] != (16, 44)
        or categories.shape != features.shape[:2]
        or not np.issubdtype(categories.dtype, np.integer)
        or np.any((categories < 0) | (categories > 3))
        or not np.isfinite(theta).all()
        or not np.isfinite(features).all()
    ):
        raise ValueError("Finite actual toy parameters/features and four-event labels required")
    hidden, action, _, mass = _numpy_forward(theta, features, categories)
    derivative = (1 - hidden**2) * theta[720:736]
    logits_jacobian = np.concatenate(
        (
            (derivative[..., :, None] * features[..., None, :]).reshape(len(features), 16, 704),
            derivative,
            hidden,
            np.ones((*hidden.shape[:-1], 1)),
        ),
        axis=-1,
    )
    scores = logits_jacobian - np.einsum("pa,pad->pd", action, logits_jacobian)[:, None, :]
    return {
        "mass": mass,
        "action_probabilities": action,
        "scores": scores,
        "jacobian": event_jacobian(theta, features, categories),
        "information_access": "EXACT_FINITE_ACTION_EVALUATOR",
    }


def fisher_direction_diagnostics(geometry, residual, *, actions=None):
    """Compute r'Fr directly from full scores, without allocating a D-by-D F.

    Empirical actions are independently sampled under this same policy. The
    empirical expression is an observed score second moment, not a bound on an
    unseen event or a calibrated guarantee.
    """
    p = np.asarray(geometry["action_probabilities"], dtype=float)
    scores = np.asarray(geometry["scores"], dtype=float)
    mass = np.asarray(geometry["mass"], dtype=float)
    residual = np.asarray(residual, dtype=float)
    if (
        p.ndim != 2
        or scores.shape[:2] != p.shape
        or scores.ndim != 3
        or mass.shape != (len(p), 4)
        or residual.shape != (scores.shape[-1],)
        or not all(np.isfinite(x).all() for x in (p, scores, mass, residual))
        or np.any(p < 0)
        or not np.allclose(p.sum(-1), 1)
    ):
        raise ValueError("Aligned finite probability, full score and residual arrays required")
    directional = np.einsum("pad,d->pa", scores, residual)
    exact = np.sum(p * directional**2, axis=-1)
    result = {
        "exact_energy": exact,
        "exact_residual": np.sqrt(np.maximum(exact, 0)),
        "exact_local_bound": np.sqrt(np.maximum(mass * (1 - mass), 0) * exact[:, None]),
        "empirical_energy": None,
        "empirical_sample_count": 0,
        "empirical_bound_is_certified": False,
    }
    if actions is not None:
        actions = np.asarray(actions)
        if (
            actions.ndim != 2
            or actions.shape[0] != len(p)
            or actions.shape[1] < 1
            or not np.issubdtype(actions.dtype, np.integer)
            or np.any((actions < 0) | (actions >= p.shape[1]))
        ):
            raise ValueError("Independent in-support sample actions required")
        observed = np.take_along_axis(directional, actions, axis=1)
        result.update(
            empirical_energy=np.mean(observed**2, axis=-1), empirical_sample_count=actions.shape[1]
        )
    return result


def decompose_response(truth, base_jacobian, updates, basis, prediction):
    """Inputs: C/pred[q,p,4], J_base[q,p,4,d], e[q,d], Q[d,k]."""
    truth, jac, e, q, prediction = [
        np.asarray(x, dtype=float) for x in (truth, base_jacobian, updates, basis, prediction)
    ]
    if (
        truth.ndim != 3
        or truth.shape[-1] != 4
        or prediction.shape != truth.shape
        or e.ndim != 2
        or e.shape[0] != len(truth)
        or jac.shape != (*truth.shape, e.shape[1])
        or q.ndim != 2
        or q.shape[0] != e.shape[1]
        or not all(np.isfinite(x).all() for x in (truth, jac, e, q))
        or np.isinf(prediction).any()
    ):
        raise ValueError(
            "Aligned finite truth/Jacobian/update arrays and finite-or-NaN prediction required"
        )
    if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-9, rtol=1e-9):
        raise ValueError("An orthonormal calibration basis is required")
    parallel = (e @ q) @ q.T
    perpendicular = e - parallel
    full = np.einsum("qped,qd->qpe", jac, e)
    covered = np.einsum("qped,qd->qpe", jac, parallel)
    omitted = np.einsum("qped,qd->qpe", jac, perpendicular)
    nonlinear, fit = truth - full, covered - prediction
    components = np.stack((nonlinear, omitted, fit), axis=-2)
    cross = np.stack(
        [
            2 * np.sum(components[..., i, :] * components[..., j, :], axis=-1)
            for i, j in ((0, 1), (0, 2), (1, 2))
        ],
        axis=-1,
    )
    return {
        "nonlinearity": nonlinear,
        "omitted_semantic": omitted,
        "fit_error": fit,
        "full_base_linear": full,
        "covered_base_linear": covered,
        "e_parallel": parallel,
        "e_perp": perpendicular,
        "component_squared_norms": np.sum(components**2, axis=-1),
        "cross_terms": cross,
        "cross_term_order": ["nonlinearity_omitted", "nonlinearity_fit", "omitted_fit"],
        "total_squared_error": np.sum((truth - prediction) ** 2, axis=-1),
        "vector_reconstruction_residual": truth - prediction - nonlinear - omitted - fit,
        "error_sign": "REFERENCE_MINUS_PREDICTION",
        "causal_fraction_interpretation": False,
    }
