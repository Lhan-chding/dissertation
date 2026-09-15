"""Q6 fixed-basis offline replay, with evaluator-only dense references.

The source optimizer is never imported or called. Every intermediate increment
is required; saving only horizon endpoints cannot establish window accuracy.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np

from ..core import file_hash
from .io import canonical_hash, finalize_run, source_identity, verify_manifest
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
    anchor_half_width=None,
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
    anchor_width = (
        np.zeros_like(anchor)
        if anchor_half_width is None
        else np.broadcast_to(np.asarray(anchor_half_width, dtype=float), anchor.shape)
    )
    if not np.isfinite(anchor_width).all() or (anchor_width < 0).any():
        raise ValueError("Anchor uncertainty must be finite and nonnegative")
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
        missed = ambiguous = rejected = path_rejected = 0
        all_case_errors = []
        for index, row in enumerate(rows[:horizon]):
            change_width = widths[index] + anchor_width
            actual_change = float(np.max(np.abs(values[index] - anchor)))
            lower_change = float(
                np.max(np.maximum(0, np.abs(values[index] - anchor) - change_width))
            )
            upper_change = float(np.max(np.abs(values[index] - anchor) + change_width))
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
            path_rejected += int(row.get("projection_plus_path_status", row["status"]) == "UNKNOWN")
            if row.get("projected_net_prediction") is not None:
                all_case_errors.append(
                    float(np.max(abs(np.asarray(row["projected_net_prediction"]) - values[index])))
                )
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
                    if row["path_interval_half_width"] is not None:
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
                    "anchor_maximum_half_width": float(anchor_width.max()),
                    "change_maximum_half_width": float(change_width.max()),
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
                "all_case_projection_maximum_error_vs_noisy_reference": max(all_case_errors)
                if all_case_errors
                else None,
                "window_maximum_persistence_error": max(persistence_errors),
                "window_maximum_hindsight_interpolation_error": max(interpolation_errors),
                "hindsight_interpolation_online_usable": False,
                "missed_resolved_changes": missed,
                "reference_unresolved_steps": ambiguous,
                "rejected_steps": rejected,
                "projection_plus_path_rejected_steps": path_rejected,
                "projection_plus_path_identified_coverage": (horizon - path_rejected) / horizon,
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
        "anchor_uncertainty_retained": anchor_half_width is not None,
        "scientific_status": "POINTWISE_AND_WINDOW_RESULTS_REQUIRED_FOR_ANY_DECISION",
    }


def _verify_qualification(config, binding, *, fit_binding):
    from .vlm_results import verify_vlm_qualification

    return verify_vlm_qualification(config, binding, fit_binding=fit_binding)


def _tracking_model(config, fit_binding, model_binding):
    """Load the actual linear response map, never a caller-supplied coefficient."""
    fit, model = _bound_json(fit_binding), _bound_json(model_binding)
    root = Path(model_binding["path"]).resolve().parent
    if not (root / "COMPLETE.json").is_file():
        raise ValueError("Q6 requires the completed fitted model original")
    source = source_identity()["sha256"]
    expected_input = {
        "spec_sha256": fit_binding["sha256"],
        "arrays_sha256": fit["arrays"]["sha256"],
    }
    verify_manifest(root, {**expected_input, "source": source, "config": canonical_hash(config)})
    receipt = _bound_json(
        {"path": str(root / "RECEIPT.json"), "sha256": file_hash(root / "RECEIPT.json")}
    )
    if (
        model.get("input_binding") != expected_input
        or model.get("fit_specification") != fit
        or receipt.get("input_binding") != expected_input
        or receipt.get("command") != "fit"
        or receipt.get("config_sha256") != canonical_hash(config)
        or fit.get("config_hash") != canonical_hash(config)
        or fit.get("source_hash") != source
        or fit.get("fixture") is not False
        or fit.get("kind") != "V3_Q5_RESPONSE_FIT"
    ):
        raise ValueError("Tracking model, fit, and current source/config identities differ")
    if model.get("method") in {"ZERO", "RBF_RIDGE"} or model.get("r", 0) < 1:
        raise ValueError("Q6 fixed-basis replay requires an identified linear response map")
    arrays_path = (root / model["array_payload"]["path"]).resolve()
    if (
        not arrays_path.is_relative_to(root)
        or file_hash(arrays_path) != model["array_payload"]["sha256"]
    ):
        raise ValueError(
            "Tracking model arrays are missing, changed, or outside their completed fit"
        )
    with np.load(arrays_path, allow_pickle=False) as arrays:
        q, basis, coefficients = [arrays[key].copy() for key in ("Q", "directions", "coefficients")]
    dimension = fit["vector_identity"]["dimension"]
    if (
        q.shape != (dimension, model["k"])
        or basis.shape != (dimension, model["r"])
        or coefficients.shape != (model["r"], 4 * len(fit["probe_ids"]))
        or not all(np.isfinite(value).all() for value in (q, basis, coefficients))
        or not np.allclose(q.T @ q, np.eye(model["k"]), atol=1e-10, rtol=1e-10)
        or not np.allclose(basis.T @ basis, np.eye(model["r"]), atol=1e-10, rtol=1e-10)
        or not np.allclose(q @ (q.T @ basis), basis, atol=1e-10, rtol=1e-10)
    ):
        raise ValueError("Actual fixed basis/coefficients do not align with the qualified fit")
    return fit, basis, coefficients


def _source_window(config, source_binding, fit, horizons):
    """Read real complete CPU checkpoint tensors for every intervening source step."""
    from ..followup_updates import load_checkpoint, state_hash
    from .vlm_campaign import _verified_execution

    source = _bound_json(source_binding)
    verified = _verified_execution(Path(source_binding["path"]).parent, config, unit="V3_SOURCE")
    if source != verified:
        raise ValueError("Bound source is not the verified original source result")
    origin_seed, arm, anchor = source["identity"]["seed"], source["identity"]["arm"], 64
    if (
        arm != "X_BASE"
        or origin_seed not in config["qwen"]["seed_roles"]["locked_test"]
        or fit["origin_id"] != f"{origin_seed}_{arm}_{anchor}"
        or source.get("retained_all_intermediate_states") is not True
        or source.get("online_feedback") is not False
        or source.get("steps") != 128
        or source.get("source_optimizer_steps") != 128
        or [record["step"] for record in source["checkpoints"]] != list(range(1, 129))
    ):
        raise ValueError("Q6 needs a complete locked-test X_BASE source with calibration at step64")
    order = fit["vector_identity"]["parameter_order"]
    if not order or len(order) != len(set(order)):
        raise ValueError("Qualified fit requires one explicit unique parameter order")
    records = {record["step"]: record for record in source["checkpoints"]}
    # The fork origin must be the exact source checkpoint, not merely the same step label.
    fork = _bound_json(fit["forks"])
    bank_plan_path = Path(fit["forks"]["path"]).parent / "bank_plan.json"
    bank_plan = _bound_json({"path": str(bank_plan_path), "sha256": file_hash(bank_plan_path)})
    if bank_plan["origin_identity"].get("state_hash") != records[anchor]["state_hash"] or fork[
        "identity"
    ].get("config_hash") != canonical_hash(config):
        raise ValueError("Step64 calibration fork and replay source have different original states")
    updates, ledger = [], []
    previous = None
    for step in range(anchor, anchor + max(horizons) + 1):
        record = records[step]
        state = load_checkpoint(
            record["path"],
            record["checkpoint_identity"],
            expected_file_sha256=record["checkpoint_sha256"],
        )
        if (
            state_hash(state) != record["state_hash"]
            or state_hash(state["parameters"]) != record["parameter_hash"]
            or any(
                state["metadata"].get(key) != value
                for key, value in (("seed", origin_seed), ("arm", arm), ("checkpoint_step", step))
            )
            or set(state["parameters"]) != set(order)
        ):
            raise ValueError("Dense source checkpoint identity/parameter coordinate mismatch")
        vector = np.concatenate(
            [
                state["parameters"][name].detach().cpu().double().numpy().reshape(-1)
                for name in order
            ]
        )
        if vector.shape != (fit["vector_identity"]["dimension"],) or not np.isfinite(vector).all():
            raise ValueError("Dense source checkpoint has invalid actual parameter values")
        if previous is not None:
            update = vector - previous
            if not np.isclose(np.linalg.norm(update), record["update_norm"], rtol=1e-9, atol=1e-12):
                raise ValueError("Actual source increment differs from its retained update norm")
            updates.append(update)
        previous = vector
        ledger.append(
            {
                key: record[key]
                for key in (
                    "step",
                    "path",
                    "checkpoint_sha256",
                    "state_hash",
                    "parameter_hash",
                    "inference_fingerprint",
                )
            }
        )
    return np.asarray(updates), ledger


def _real_prediction_values(config, inputs, anchor_binding, *, fixture=False, prepared=None):
    from .q6_observation import _complete, _qualified_inputs, verify_q6_absolute
    from .vlm_results import _geometry_from_originals

    qualified, fit, basis, coefficients, updates, checkpoints = (
        prepared if prepared is not None else _qualified_inputs(config, inputs, fixture)
    )
    anchor = verify_q6_absolute(config, anchor_binding, fixture=fixture)
    plan = _complete(anchor["plan_binding"])
    if (
        anchor["phase"] != "anchor"
        or anchor["states"] != [64]
        or plan["inputs"] != inputs
        or anchor["origin_id"] != fit["origin_id"]
        or anchor["probe_ids"] != fit["probe_ids"]
    ):
        raise ValueError(
            "Q6 anchor is not an independent absolute observation at this fitted origin"
        )
    selection = _bound_json(qualified["selection_lock"])["selected"]
    tolerance = selection["pointwise_criteria"]["primary_tolerance"]
    anchor_probability = np.asarray(anchor["probabilities"])[0]
    anchor_width = np.asarray(anchor["half_width"])[0]
    model = _bound_json(inputs["model_binding"])
    model_arrays = Path(inputs["model_binding"]["path"]).parent / model["array_payload"]["path"]
    with np.load(model_arrays, allow_pickle=False) as data:
        q = data["Q"].copy()
    from .vlm_observation import _read_binding

    with np.load(_read_binding(fit["arrays"]), allow_pickle=False) as data:
        fit_updates = data["updates"].copy()
    geometry = _geometry_from_originals(
        fit_updates, np.cumsum(updates, axis=0), q, config, fit["alpha"]
    )
    prediction = fixed_basis_tracking(
        anchor_probability,
        updates,
        basis,
        coefficients,
        qualification=qualified,
        interval_half_width=float(tolerance + anchor_width.max()),
    )
    for index, row in enumerate(prediction["rows"]):
        reasons = []
        if not anchor["empirical_precision_met"]:
            reasons.append("ANCHOR_REFERENCE_UNRESOLVED")
        if geometry["rho"][index] > selection["rho_threshold"]:
            reasons.append("OUTSIDE_FROZEN_CALIBRATION_SPAN_TOLERANCE")
        if geometry["leverage"][index] > selection["leverage_threshold"]:
            reasons.append("OUTSIDE_FROZEN_CALIBRATION_LEVERAGE")
        row.update(
            source_step=65 + index,
            geometry_rho=float(geometry["rho"][index]),
            geometry_leverage=float(geometry["leverage"][index]),
            abstention_reasons=reasons,
        )
        if reasons:
            row.update(status="UNKNOWN", delta_prediction=None, probability_prediction=None)
        row["projection_plus_path_status"] = (
            "UNKNOWN" if row["uncovered_path_length"] > 1e-12 else row["status"]
        )
        row["path_interval_half_width"] = None
        row["path_range_width"] = None
    prediction.update(
        fixture=fixture,
        origin_id=fit["origin_id"],
        probe_ids=fit["probe_ids"],
        anchor_binding=anchor_binding,
        anchor_half_width=anchor_width.tolist(),
        operational_error_budget=tolerance,
        interval_scope="FROZEN_OPERATIONAL_TOLERANCE_PLUS_ANCHOR_MEASUREMENT_WIDTH_NOT_95_CERTIFICATION",
        path_scope=(
            "NET_POINT_PREDICTION_UNCHANGED; ABSTAIN_AFTER_UNCOVERED_PATH; "
            "PATH_ERROR_WIDTH_UNCALIBRATED"
        ),
        checkpoints=checkpoints,
        cross_seed_95="NOT_CERTIFIED",
        online_ssvc="NOT_CERTIFIED",
        meaningful_change=config["statistics"]["qwen_primary_meaningful_change"],
        scientific_status="OFFLINE_PREDICTIONS_AWAITING_INDEPENDENT_DENSE_REFERENCE",
    )
    return prediction


def verify_tracking_prediction_lock(config, binding, inputs, *, fixture=False):
    from .q6_observation import _complete, _identity

    lock = _complete(binding)
    if (
        lock.get("kind") != "V3_Q6_FROZEN_PREDICTIONS"
        or any(lock.get(k) != v for k, v in _identity(config, fixture).items())
        or lock.get("inputs") != inputs
        or lock.get("reference_labels_read") is not False
    ):
        raise ValueError(
            "Q6 reference requires a source/config-bound pre-reference prediction lock"
        )
    actual = _bound_json(lock["predictions"])
    expected = _real_prediction_values(config, inputs, lock["anchor_binding"], fixture=fixture)
    if actual != expected:
        raise ValueError(
            "Frozen Q6 predictions differ from actual model, source increments, and anchor"
        )
    return lock


def _finish_real_tracking(config, payload, out, prepared, *, fixture=False):
    from .q6_observation import _binding, _complete, _identity, verify_q6_absolute

    inputs = {
        key: payload[key]
        for key in ("qualification_binding", "fit_binding", "model_binding", "source_binding")
    }
    prediction_binding = payload.get("prediction_binding")
    if prediction_binding is None:
        if payload.get("reference_binding") is not None:
            raise ValueError(
                "Freeze the Q6 prediction artifact before supplying any dense reference"
            )
        if payload.get("anchor_binding") is None:
            return {
                "status": "Q6_ANCHOR_OBSERVATION_REQUIRED",
                "tracking_executed": False,
                "next_command": "prepare-q6 --phase anchor",
                "online_feedback": False,
                "pointwise_qualification_verified": True,
                "inputs": inputs,
            }
        if out is None:
            raise ValueError("Real Q6 requires an immutable prediction output directory")
        prediction = _real_prediction_values(
            config, inputs, payload["anchor_binding"], prepared=prepared, fixture=fixture
        )
        if (
            payload.get("basis_fingerprint", prediction["basis_fingerprint"])
            != prediction["basis_fingerprint"]
        ):
            raise ValueError("Tracking basis/coefficients changed inside the frozen step64 window")
        root = Path(out).resolve()
        for value in [*inputs.values(), payload["anchor_binding"]]:
            original = Path(value["path"]).resolve().parent
            if root.is_relative_to(original) or original.is_relative_to(root):
                raise ValueError("Q6 prediction output must not overlap immutable originals")
        root.mkdir(parents=True, exist_ok=False)
        _publish(root / "predictions.json", prediction)
        lock = {
            "kind": "V3_Q6_FROZEN_PREDICTIONS",
            **_identity(config, fixture),
            "inputs": inputs,
            "anchor_binding": payload["anchor_binding"],
            "predictions": _binding(root / "predictions.json"),
            "reference_labels_read": False,
            "online_feedback": False,
        }
        _publish(root / "PREDICTION_LOCK.json", lock)
        finalize_run(
            root,
            {**_identity(config, fixture), "inputs": inputs, "anchor": payload["anchor_binding"]},
        )
        return {
            "status": "Q6_PREDICTIONS_FROZEN",
            "prediction_binding": _binding(root / "PREDICTION_LOCK.json"),
            "online_feedback": False,
            "next_command": "prepare-q6 --phase reference",
        }
    lock = verify_tracking_prediction_lock(config, prediction_binding, inputs, fixture=fixture)
    if payload.get("anchor_binding", lock["anchor_binding"]) != lock["anchor_binding"]:
        raise ValueError("Q6 assessment cannot replace the frozen independent anchor")
    prediction = _bound_json(lock["predictions"])
    if payload.get("reference_binding") is None:
        return {
            "status": "Q6_PREDICTIONS_FROZEN",
            "prediction_binding": prediction_binding,
            "online_feedback": False,
        }
    reference = verify_q6_absolute(config, payload["reference_binding"], fixture=fixture)
    plan = _complete(reference["plan_binding"])
    if (
        reference["phase"] != "reference"
        or reference["states"] != list(range(65, 81))
        or plan["prediction_binding"] != prediction_binding
        or plan["inputs"] != inputs
        or reference["probe_ids"] != prediction["probe_ids"]
    ):
        raise ValueError("Q6 dense references do not belong to the frozen prediction window")
    assessment = evaluate_tracking(
        prediction,
        reference["probabilities"],
        horizons=config["tracking_offline"]["horizons"],
        meaningful_change=config["statistics"]["qwen_primary_meaningful_change"],
        reference_half_width=reference["half_width"],
        anchor_half_width=prediction["anchor_half_width"],
    )
    assessment.update(
        kind="V3_Q6_OFFLINE_ASSESSMENT",
        **_identity(config, fixture),
        prediction_binding=prediction_binding,
        reference_binding=payload["reference_binding"],
        reference_precision_status=reference["status"],
        cross_seed_95="NOT_CERTIFIED",
        online_ssvc="NOT_CERTIFIED",
    )
    if out is not None:
        root = Path(out).resolve()
        for value in [prediction_binding, payload["reference_binding"], *inputs.values()]:
            original = Path(value["path"]).resolve().parent
            if root.is_relative_to(original) or original.is_relative_to(root):
                raise ValueError("Q6 assessment output must not overlap immutable originals")
        root.mkdir(parents=True, exist_ok=False)
        _publish(root / "ASSESSMENT.json", assessment)
        finalize_run(
            root,
            {
                **_identity(config, fixture),
                "prediction": prediction_binding,
                "reference": payload["reference_binding"],
            },
        )
    return assessment


def _real_tracking(payload, out, config):
    if not isinstance(config, dict):
        raise ValueError("Real Q6 tracking requires the frozen protocol config")
    if platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Real Q6 tracking must run on server CPU under Slurm")
    if any(
        key in payload
        for key in ("qualification", "anchor_probability", "updates", "basis", "coefficients")
    ):
        raise ValueError(
            "Real Q6 cannot use handwritten qualification, states, basis, or coefficients"
        )
    settings = config["tracking_offline"]
    if (
        settings.get("window_anchor") != 64
        or settings.get("primary_arm") != "X_BASE"
        or settings.get("feedback_into_source_training") is not False
        or settings.get("basis_frozen_within_window") is not True
        or settings.get("dense_reference_for_assessment_only") is not True
        or settings.get("horizons") != [1, 2, 4, 8, 16]
        or payload.get("tracking_config", settings) != settings
    ):
        raise ValueError(
            "Q6 must retain the frozen step64 window, horizons, and offline-only protocol"
        )
    if "qualification_binding" not in payload or "fit_binding" not in payload:
        raise ValueError("Real Q6 needs bound pointwise qualification and the actual step64 fit")
    for key in ("qualification_binding", "fit_binding", "model_binding", "source_binding"):
        if key in payload and (
            not isinstance(payload[key], dict) or not {"path", "sha256"} <= payload[key].keys()
        ):
            raise ValueError("Real Q6 original bindings require explicit path and SHA-256")
    qualification = _verify_qualification(
        config, payload["qualification_binding"], fit_binding=payload["fit_binding"]
    )
    if (
        qualification.get("pointwise_qualified") is not True
        or qualification.get("tracking_qualified") is False
    ):
        return {
            "status": "NOT_QUALIFIED_FOR_TRACKING",
            "qualification": qualification,
            "online_feedback": False,
            "reference_inputs_consumed": False,
        }
    if (
        qualification.get("kind") != "V3_VLM_POINTWISE_QUALIFICATION"
        or qualification.get("status") != "QUALIFIED_EMPIRICALLY"
        or qualification.get("fit_hash") != payload["fit_binding"].get("sha256")
        or qualification.get("design_id") not in qualification.get("qualified_design_ids", [])
        or qualification.get("tracking_qualified") is not True
        or qualification.get("design_id")
        not in qualification.get("tracking_eligible_design_ids", [])
        or qualification.get("cross_seed_95") != "NOT_CERTIFIED"
        or qualification.get("online_ssvc") != "NOT_CERTIFIED"
    ):
        raise ValueError(
            "Verified Q6 qualification has inconsistent fit/design/certification scope"
        )
    if not {"model_binding", "source_binding"} <= payload.keys():
        raise ValueError("Real Q6 needs the completed fitted model and full source originals")
    fit, basis, coefficients = _tracking_model(
        config, payload["fit_binding"], payload["model_binding"]
    )
    updates, checkpoints = _source_window(
        config, payload["source_binding"], fit, settings["horizons"]
    )
    return _finish_real_tracking(
        config, payload, out, (qualification, fit, basis, coefficients, updates, checkpoints)
    )


def track_offline(payload, out=None, *, config=None, fixture=False):
    """Real Q6 verifies original evidence; explicit fixtures only exercise arithmetic."""
    payload = dict(payload)
    if not fixture:
        return _real_tracking(payload, out, config)
    if any(
        key in payload
        for key in ("qualification_binding", "fit_binding", "model_binding", "source_binding")
    ):
        raise ValueError(
            "A tracking fixture cannot consume production qualification or model bindings"
        )
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
    predictions.update(fixture=True, scientific_status="FIXTURE_NOT_AUTHORIZATION")
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
    if reference.get("kind") == "REAL_VLM_ESTIMATE" and reference.get("half_width") is None:
        raise ValueError("VLM tracking reference must retain measured uncertainty")
    if (
        reference.get("kind") == "REAL_VLM_ESTIMATE"
        and (np.asarray(reference["half_width"], dtype=float) <= 0).any()
    ):
        raise ValueError("Real VLM reference uncertainty cannot be zero or masquerade as exact")
    assessment = evaluate_tracking(
        predictions,
        reference["probabilities"],
        reference_half_width=reference.get("half_width"),
        horizons=config.get("horizons", [1, 2, 4, 8, 16]),
        meaningful_change=payload.get("meaningful_change", 0.001),
    )
    assessment.update(fixture=True, scientific_status="FIXTURE_NOT_AUTHORIZATION")
    if out is not None:
        _publish(Path(out) / "assessment.json", assessment)
    return {"predictions": predictions, "assessment": assessment}
