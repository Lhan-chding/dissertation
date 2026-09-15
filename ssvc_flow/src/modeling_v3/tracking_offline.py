"""Q6 fixed-basis offline replay, with evaluator-only dense references.

The source optimizer is never imported or called. Every intermediate increment
is required; saving only horizon endpoints cannot establish window accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import file_hash
from .io import canonical_hash
from .vlm_campaign import _bound_json, _publish


def fixed_basis_tracking(
    anchor_probability,
    updates,
    basis,
    coefficients,
    *,
    qualification,
    interval_half_width=None,
    uncovered_path_coefficient=0.0,
    maximum_uncovered_path=None,
    novelty_tolerance=1e-12,
    basis_fingerprint=None,
):
    """Predict all intermediate states using one basis and one fitted response.

    Path information can widen intervals or cause abstention. It intentionally
    does not change a prediction obtained from projected net displacement.
    """
    if (
        qualification.get("pointwise_qualified") is not True
        or not qualification.get("fit_hash")
        or not qualification.get("source_hash")
    ):
        return {
            "status": "NOT_QUALIFIED_FOR_TRACKING",
            "online_feedback": False,
            "reason": "Pointwise qualification with frozen fit/source identity is required",
        }
    anchor, updates, q, coef = [
        np.asarray(value, dtype=float)
        for value in (anchor_probability, updates, basis, coefficients)
    ]
    if (
        anchor.ndim not in (1, 2)
        or anchor.shape[-1] != 4
        or updates.ndim != 2
        or q.ndim != 2
        or q.shape[0] != updates.shape[1]
        or coef.shape != (q.shape[1], anchor.size)
        or len(updates) < 1
        or not all(np.isfinite(x).all() for x in (anchor, updates, q, coef))
    ):
        raise ValueError(
            "Finite aligned raw4 anchors, every update, fixed basis and coefficients required"
        )
    if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10, rtol=1e-10):
        raise ValueError("The fixed tracking basis must be orthonormal")
    fingerprint = canonical_hash(
        {"basis": q.tolist(), "coefficients": coef.tolist(), "fit_hash": qualification["fit_hash"]}
    )
    if basis_fingerprint is not None and basis_fingerprint != fingerprint:
        raise ValueError("Tracking basis/coefficients changed inside a frozen window")
    if (
        (
            interval_half_width is not None
            and (not np.isfinite(interval_half_width) or interval_half_width < 0)
        )
        or not np.isfinite(uncovered_path_coefficient)
        or uncovered_path_coefficient < 0
        or not np.isfinite(novelty_tolerance)
        or novelty_tolerance < 0
        or (
            maximum_uncovered_path is not None
            and (not np.isfinite(maximum_uncovered_path) or maximum_uncovered_path < 0)
        )
    ):
        raise ValueError("Frozen tracking widths/path thresholds must be finite and nonnegative")
    projected = updates @ q
    residuals = updates - projected @ q.T
    net = np.cumsum(updates, axis=0)
    projected_net = np.cumsum(projected, axis=0)
    delta = (projected_net @ coef).reshape((len(updates), *anchor.shape))
    predictions = anchor + delta
    residual_norm = np.linalg.norm(residuals, axis=1)
    uncovered_path = np.cumsum(residual_norm)
    novelty = np.flatnonzero(residual_norm > novelty_tolerance)
    k = q.shape[1]
    rows = []
    for index in range(len(updates)):
        identified = k > 0 and (
            maximum_uncovered_path is None or uncovered_path[index] <= maximum_uncovered_path
        )
        width = None if interval_half_width is None else float(interval_half_width)
        path_width = (
            None
            if width is None
            else width + uncovered_path_coefficient * float(uncovered_path[index])
        )
        rows.append(
            {
                "horizon": index + 1,
                "status": "PREDICTED" if identified else "UNKNOWN",
                "delta_prediction": delta[index].tolist() if identified else None,
                "probability_prediction": predictions[index].tolist() if identified else None,
                "projected_net_prediction": predictions[index].tolist() if k else None,
                "projection_plus_path_point_prediction": predictions[index].tolist() if k else None,
                "persistence_prediction": anchor.tolist(),
                "interval_half_width": width,
                "path_interval_half_width": path_width,
                "range_width": None if width is None else 2 * width,
                "path_range_width": None if path_width is None else 2 * path_width,
                "update_norm": float(np.linalg.norm(updates[index])),
                "update_path_length": float(np.linalg.norm(updates[: index + 1], axis=1).sum()),
                "projected_net_norm": float(np.linalg.norm(projected_net[index])),
                "full_net_norm": float(np.linalg.norm(net[index])),
                "uncovered_step_norm": float(residual_norm[index]),
                "uncovered_path_length": float(uncovered_path[index]),
                "uncovered_net_norm": float(
                    np.linalg.norm(net[index] - projected_net[index] @ q.T)
                ),
            }
        )
    return {
        "status": "OFFLINE_PREDICTIONS_FROZEN",
        "rows": rows,
        "anchor_probability": anchor.tolist(),
        "rank": k,
        "basis_fingerprint": fingerprint,
        "qualification": qualification,
        "source_hash": qualification["source_hash"],
        "first_novel_direction_horizon": int(novelty[0] + 1) if len(novelty) else None,
        "basis_frozen_within_window": True,
        "fit_refreshed_inside_window": False,
        "path_changes_point_prediction": False,
        "online_feedback": False,
        "unclipped_probability_predictions": True,
        "calibrated_95_percent_claim": False,
    }


def evaluate_tracking(
    frozen_predictions,
    reference,
    *,
    horizons=(1, 2, 4, 8, 16),
    meaningful_change=0.001,
    reference_half_width=None,
):
    """Only this function reads dense intermediate probabilities (or estimates)."""
    if frozen_predictions.get("status") != "OFFLINE_PREDICTIONS_FROZEN":
        return {"status": "NOT_QUALIFIED_FOR_TRACKING", "online_feedback": False}
    values = np.asarray(reference, dtype=float)
    rows = frozen_predictions["rows"]
    anchor = np.asarray(frozen_predictions["anchor_probability"])
    if values.shape != (len(rows), *anchor.shape) or not np.isfinite(values).all():
        raise ValueError(
            "Dense reference must cover every intermediate state and every raw4 channel"
        )
    if not np.isfinite(meaningful_change) or meaningful_change <= 0:
        raise ValueError("Meaningful-change threshold must be frozen and positive")
    widths = (
        np.zeros_like(values)
        if reference_half_width is None
        else np.broadcast_to(np.asarray(reference_half_width, dtype=float), values.shape)
    )
    if not np.isfinite(widths).all() or (widths < 0).any():
        raise ValueError("Reference uncertainty must be finite and nonnegative")
    reports = []
    for horizon in horizons:
        if type(horizon) is not int or not 1 <= horizon <= len(rows):
            raise ValueError("Every requested horizon needs all intervening states")
        # This baseline is evaluated only after frozen predictions. It cannot be
        # deployed at any intermediate time because it observes the future endpoint.
        interpolated = anchor + (np.arange(1, horizon + 1) / horizon).reshape(
            (-1,) + (1,) * anchor.ndim
        ) * (values[horizon - 1] - anchor)
        point_errors, persistence_errors, interpolation_errors, intermediate = [], [], [], []
        missed = ambiguous = rejected = 0
        for index, row in enumerate(rows[:horizon]):
            actual_change = float(np.max(np.abs(values[index] - anchor)))
            lower_change = float(
                np.max(np.maximum(0, np.abs(values[index] - anchor) - widths[index]))
            )
            upper_change = float(np.max(np.abs(values[index] - anchor) + widths[index]))
            resolution = (
                "CHANGE_RESOLVED"
                if lower_change >= meaningful_change
                else (
                    "NO_CHANGE_RESOLVED"
                    if upper_change < meaningful_change
                    else "REFERENCE_UNRESOLVED"
                )
            )
            ambiguous += int(resolution == "REFERENCE_UNRESOLVED")
            is_unknown = row["status"] == "UNKNOWN"
            rejected += int(is_unknown)
            error = None
            interval_covered = path_interval_covered = None
            if not is_unknown:
                estimate = np.asarray(row["probability_prediction"])
                residual = np.abs(estimate - values[index])
                error = float(residual.max())
                point_errors.append(error)
                predicted_change = float(np.max(np.abs(estimate - anchor)))
                missed += int(
                    resolution == "CHANGE_RESOLVED" and predicted_change < meaningful_change
                )
                if row["interval_half_width"] is not None:
                    interval_covered = bool(
                        (residual + widths[index] <= row["interval_half_width"]).all()
                    )
                    path_interval_covered = bool(
                        (residual + widths[index] <= row["path_interval_half_width"]).all()
                    )
            persistence_errors.append(float(np.abs(anchor - values[index]).max()))
            interpolation_errors.append(float(np.abs(interpolated[index] - values[index]).max()))
            intermediate.append(
                {
                    "horizon": index + 1,
                    "status": row["status"],
                    "maximum_event_error": error,
                    "reference_maximum_half_width": float(widths[index].max()),
                    "actual_change_estimate": actual_change,
                    "reference_resolution": resolution,
                    "interval_covers_reference_interval": interval_covered,
                    "path_interval_covers_reference_interval": path_interval_covered,
                    "range_width": row["range_width"],
                    "path_range_width": row["path_range_width"],
                }
            )
        reports.append(
            {
                "horizon": horizon,
                "intermediate": intermediate,
                "window_maximum_prediction_error": max(point_errors) if point_errors else None,
                "window_maximum_persistence_error": max(persistence_errors),
                "window_maximum_hindsight_interpolation_error": max(interpolation_errors),
                "hindsight_interpolation_online_usable": False,
                "missed_resolved_changes": missed,
                "reference_unresolved_steps": ambiguous,
                "rejected_steps": rejected,
                "all_steps": horizon,
                "identified_coverage": (horizon - rejected) / horizon,
                "unknown_is_safe": False,
            }
        )
    return {
        "status": "OFFLINE_ASSESSED",
        "windows": reports,
        "prediction_hash": canonical_hash(frozen_predictions),
        "reference_hash": canonical_hash(values.tolist()),
        "online_feedback": False,
        "reference_is_exact": reference_half_width is None,
        "reference_uncertainty_retained": reference_half_width is not None,
        "scientific_status": "POINTWISE_AND_WINDOW_RESULTS_REQUIRED_FOR_ANY_DECISION",
    }


def track_offline(payload, out=None):
    """CLI entry: freeze predictions before opening any dense reference file."""
    payload = dict(payload)
    required = {"anchor_probability", "updates", "basis", "coefficients", "qualification"}
    if not required <= set(payload):
        raise ValueError(
            "Offline replay needs the frozen anchor, every update, basis, fit and qualification"
        )
    config = payload.get("tracking_config", {})
    if (
        config.get("feedback_into_source_training", False) is not False
        or config.get("basis_frozen_within_window", True) is not True
    ):
        raise ValueError("Q6 cannot feed back into training or refresh the basis inside a window")
    predictions = fixed_basis_tracking(
        **{key: payload[key] for key in required},
        interval_half_width=payload.get("interval_half_width"),
        uncovered_path_coefficient=payload.get("uncovered_path_coefficient", 0),
        maximum_uncovered_path=payload.get("maximum_uncovered_path"),
        novelty_tolerance=payload.get("novelty_tolerance", 1e-12),
        basis_fingerprint=payload.get("basis_fingerprint"),
    )
    if out is not None:
        root = Path(out)
        _publish(root / "predictions.json", predictions)
        _publish(
            root / "prediction_lock.json",
            {
                "sha256": file_hash(root / "predictions.json"),
                "canonical_hash": canonical_hash(predictions),
            },
        )
    if predictions["status"] != "OFFLINE_PREDICTIONS_FROZEN":
        return predictions
    reference_binding = payload.get("reference_binding")
    if reference_binding is None:
        return predictions
    reference = _bound_json(reference_binding)
    # Zero uncertainty is allowed only when the evaluator explicitly declares a
    # finite toy exact reference. A Qwen reference is always an estimate.
    if reference.get("kind") not in {"FINITE_TOY_EXACT", "REAL_VLM_ESTIMATE"}:
        raise ValueError("Reference must explicitly identify finite-toy exact or real VLM estimate")
    if reference.get("kind") == "REAL_VLM_ESTIMATE" and "half_width" not in reference:
        raise ValueError("VLM tracking reference must retain measured uncertainty")
    assessment = evaluate_tracking(
        predictions,
        reference["probabilities"],
        reference_half_width=reference.get("half_width"),
        horizons=config.get("horizons", [1, 2, 4, 8, 16]),
        meaningful_change=payload.get("meaningful_change", 0.001),
    )
    if out is not None:
        _publish(Path(out) / "assessment.json", assessment)
    return {"predictions": predictions, "assessment": assessment}
