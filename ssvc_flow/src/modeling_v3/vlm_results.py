"""Original-backed VLM calibration and independent-seed point prediction analysis.

Three calibration seeds support an explicitly frozen *empirical* operating rule;
they cannot supply a finite distribution-free 95% cross-seed radius. Test errors
never choose query acceptance. The independent reference remains an estimate,
including its raw samples, uncertainty, unresolved cells and aggregate noise
correction. Production analysis belongs on server CPU, not the workstation.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .io import (
    atomic_bytes,
    atomic_json,
    canonical_hash,
    finalize_run,
    sha256_file,
    verify_manifest,
)
from .statistics import error_metrics, seed_bootstrap
from .vlm_response import _bound_spec, _check_identity, _cpu_scope, _identity

TARGETS = (
    "joint_1_minus_joint_0",
    "no_x_off_1_minus_joint_0",
    "joint_1_minus_no_x_off_1",
)
DESIGN_KEYS = (
    "method",
    "rank_cap",
    "alpha",
    "output_policy",
    "regression",
    "observation_method",
    "selector",
    "n_banks",
    "selection_seed",
)
NOT_CERTIFIED = "NOT_CERTIFIED"


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _file(value):
    if isinstance(value, (str, Path)):
        value = _binding(value)
    if not isinstance(value, dict) or not {"path", "sha256"} <= set(value):
        raise ValueError("Original file path or hash binding required")
    path = Path(value["path"]).resolve()
    if not path.is_file() or sha256_file(path) != value["sha256"]:
        raise ValueError("Original file binding hash changed")
    return path


def _complete(path):
    root = Path(path).parent
    if not (root / "COMPLETE.json").is_file():
        raise ValueError("A complete original artifact is required")
    if verify_manifest(root).get("status") != "COMPLETE":
        raise ValueError("Original manifest is not COMPLETE")


def _document(value, config, *, kind=None, fixture=False, complete=True):
    doc, binding = _bound_spec(value)
    if complete:
        _complete(binding["path"])
    _check_identity(doc, config)
    if doc.get("fixture", False) is not fixture:
        raise ValueError("VLM fixture and production receipt scopes differ")
    if kind is not None and doc.get("kind") != kind:
        raise ValueError("Unexpected original receipt kind: " + str(doc.get("kind")))
    return doc, binding


def _criteria(config, selected):
    rule = selected.get("pointwise_criteria", {})
    keys = {
        "primary_tolerance",
        "reference_half_width_max",
        "minimum_geometry_coverage",
        "maximum_q95_absolute_residual",
        "maximum_nrmse",
    }
    if not keys <= set(rule):
        raise ValueError("Explicit development-frozen pointwise criteria required")
    if rule["primary_tolerance"] != config["statistics"]["qwen_primary_meaningful_change"]:
        raise ValueError("VLM primary scale differs from the protocol")
    if rule["reference_half_width_max"] != config["qwen"]["reference"]["primary_half_width_goal"]:
        raise ValueError("Reference precision differs from the frozen protocol")
    if not 0 <= rule["minimum_geometry_coverage"] <= 1:
        raise ValueError("A minimum geometry coverage in [0,1] must be frozen")
    limits = [rule[key] for key in ("maximum_q95_absolute_residual", "maximum_nrmse")]
    if all(value is None for value in limits) or any(
        value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0)
        for value in limits
    ):
        raise ValueError("At least one nonnegative pointwise error limit must be frozen")
    rho, lev = selected.get("rho_threshold"), selected.get("leverage_threshold")
    if not isinstance(rho, (float, int)) or not math.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("Finite frozen rho threshold required")
    if not isinstance(lev, (float, int)) or not math.isfinite(lev) or lev < 0:
        raise ValueError("Finite frozen leverage threshold required")
    return rule


def _selection(config, value, fixture):
    lock, binding = _document(value, config, kind="V3_VLM_SELECTION_LOCK", fixture=fixture)
    if lock.get("selection_hash") != canonical_hash(lock["selected"]):
        raise ValueError("Frozen VLM selection hash changed")
    if not fixture:
        from .vlm_response import verify_vlm_selection_lock

        verify_vlm_selection_lock(config, lock, fixture=False)
    _criteria(config, lock["selected"])
    designs = lock["selected"].get("designs", [])
    ids = [d.get("design_id") for d in designs]
    if not ids or None in ids or len(set(ids)) != len(ids):
        raise ValueError("Distinct frozen VLM designs required")
    for design in designs:
        if not set(DESIGN_KEYS) <= set(design) or design["output_policy"] != "RAW4":
            raise ValueError("Frozen VLM designs require all settings and RAW4 output")
    return lock, binding


def _origin(config, origin_id, role, *, tracking=False):
    expected = {}
    for seed in config["qwen"]["seed_roles"][role]:
        pairs = [("X_BASE", 64)] if tracking else [("X_BASE", 32), ("X_BASE", 96)]
        if role == "locked_test" and not tracking:
            pairs.append(("X_VALID", 96))
        for arm, step in pairs:
            expected[f"{seed}_{arm}_{step}"] = (seed, arm, step)
    if origin_id not in expected:
        raise ValueError("Origin is outside the exact frozen role/seed/arm/step matrix")
    return expected[origin_id]


def _design(config, lock, selection_binding, fit, fixture):
    if fit.get("selection_lock") != selection_binding:
        raise ValueError("Response fit uses a different VLM selection lock")
    geometry, _ = _document(
        fit["geometry"], config, kind="V3_Q5_CALIBRATION_GEOMETRY", fixture=fixture
    )
    if any(geometry.get(k) != fit.get(k) for k in ("origin_id", "role")):
        raise ValueError("Fit geometry origin/role changed")
    actual = {key: fit.get(key) for key in DESIGN_KEYS}
    actual.update(
        selector=geometry["method"], n_banks=geometry["n_banks"], selection_seed=geometry["seed"]
    )
    for key in ("selector", "n_banks", "selection_seed"):
        if key in fit and fit[key] != actual[key]:
            raise ValueError("Fit settings disagree with original geometry selection")
    matches = [
        d for d in lock["selected"]["designs"] if all(d[key] == actual[key] for key in DESIGN_KEYS)
    ]
    if (
        len(matches) != 1
        or fit.get("design_id", matches[0]["design_id"] if matches else None)
        != matches[0]["design_id"]
    ):
        raise ValueError("Fit does not match exactly one development-frozen VLM design")
    return matches[0]


def _geometry_from_originals(updates, query, q, config, alpha):
    """Validate the saved uncentered span and accumulate residuals in D blocks.

    Actual 9B LoRA vectors are large. Never retain per-origin D-dimensional
    projections/residuals in an all-seed analysis or allocate Q @ Q.T.
    """
    if (
        updates.ndim != 2
        or query.ndim != 2
        or q.ndim != 2
        or updates.shape[1] != query.shape[1]
        or q.shape[0] != query.shape[1]
    ):
        raise ValueError("Original vector/basis coordinate dimensions differ")
    if not all(np.isfinite(x).all() for x in (updates, query, q)):
        raise ValueError("Original vector geometry must be finite")
    k = q.shape[1]
    if not np.allclose(q.T @ q, np.eye(k), atol=1e-9):
        raise ValueError("Saved model basis is not orthonormal")
    a, coordinates = updates @ q, query @ q
    singular = np.linalg.svd(a, compute_uv=False)
    threshold = max(
        config["coverage"]["rank_atol"],
        config["coverage"]["rank_rtol"] * (singular[0] if len(singular) else 0),
    )
    if int(np.count_nonzero(singular > threshold)) != k:
        raise ValueError("Saved model basis rank differs from actual calibration excitation")
    perp_ss, query_ss, fit_perp_ss = np.zeros(len(query)), np.zeros(len(query)), 0.0
    for start in range(0, query.shape[1], 65536):
        section = slice(start, start + 65536)
        residual = query[:, section] - coordinates @ q[section].T
        perp_ss += np.einsum("ij,ij->i", residual, residual)
        query_ss += np.einsum("ij,ij->i", query[:, section], query[:, section])
        fit_residual = updates[:, section] - a @ q[section].T
        fit_perp_ss += float(np.sum(fit_residual**2))
    if math.sqrt(fit_perp_ss) > max(threshold * math.sqrt(len(updates)) * 2, 1e-12):
        raise ValueError("Saved model basis does not span its actual calibration updates")
    norm = np.sqrt(query_ss)
    rho = np.divide(np.sqrt(perp_ss), norm, out=np.zeros_like(norm), where=norm > 0)
    leverage = (
        np.einsum(
            "ni,in->n", coordinates, np.linalg.solve(a.T @ a + alpha * np.eye(k), coordinates.T)
        )
        if k
        else np.zeros(len(query))
    )
    return {
        "k": k,
        "rho": rho,
        "leverage": leverage,
        "e_norm": norm,
        "leverage_metric": "IDENTITY",
        "leverage_ridge_lambda": alpha,
    }


def _fit_geometry(config, lock, selection_binding, fit_spec, model_arrays, fixture):
    fit, fit_binding = _document(fit_spec, config, kind="V3_Q5_RESPONSE_FIT", fixture=fixture)
    if fit.get("reference_labels_read") is not False or fit.get("heldout_labels_read") is not False:
        raise ValueError("Fit must be frozen before heldout/reference labels are read")
    design = _design(config, lock, selection_binding, fit, fixture)
    model_path = _file(model_arrays)
    _complete(model_path)
    receipt = json.loads((model_path.parent / "RECEIPT.json").read_text())
    if (
        receipt.get("command") != "fit"
        or receipt.get("input_binding", {}).get("spec_sha256") != fit_binding["sha256"]
    ):
        raise ValueError("Model arrays do not come from the exact frozen fit input")
    with np.load(model_path, allow_pickle=False) as data:
        predictions = data["predictions"].copy()
        q = data["Q"]
    with np.load(_file(fit["arrays"]), allow_pickle=False) as data:
        updates, query = data["updates"], data["query_updates"]
    geometry = _geometry_from_originals(updates, query, q, config, design["alpha"])
    shape = (len(fit["query_units"]), len(fit["probe_ids"]), 4)
    if predictions.shape != shape or len(query) != shape[0]:
        raise ValueError("Frozen predictions/query vectors have incompatible axes")
    selected = lock["selected"]
    mask = (geometry["rho"] <= selected["rho_threshold"]) & (
        geometry["leverage"] <= selected["leverage_threshold"]
    )
    # k=0 never licenses a learned response; zero vector is not a forward-fingerprint proof.
    mask &= geometry["k"] > 0
    mask = np.broadcast_to(mask[:, None], shape[:-1]).copy() & np.isfinite(predictions).all(-1)
    model = json.loads((model_path.parent / "MODEL.json").read_text())
    if model.get("k") != geometry["k"] or not 0 <= model.get("r", -1) <= geometry["k"]:
        raise ValueError("Saved model rank differs from calibration geometry")
    return {
        "fit": fit,
        "fit_binding": fit_binding,
        "design": design,
        "geometry": geometry,
        "geometry_mask": mask,
        "predictions": predictions,
        "k": model["k"],
        "r": model["r"],
        "model_binding": _binding(model_path),
    }


def _stage(config, value, role, fixture):
    if not fixture:
        from .vlm_campaign import _verify_q5_completion

        _, binding = _bound_spec(value)
        result = _verify_q5_completion(config, binding, role)
        # The completed task aggregator already audited complete checkpoints and
        # shards. Bind that immutable audit and its original completion markers;
        # rehash every artifact actually read below. Reentering each worker's
        # authorization gate here would recursively reanalyze all calibration
        # origins once per test observation task.
    else:
        result, binding = _document(value, config, kind="V3_Q5_STAGE_COMPLETION", fixture=True)
    if (
        result.get("role") != role
        or result.get("status") != "COMPLETE"
        or result.get("technical_failures")
    ):
        raise ValueError("Complete, successful technical stage for exact role required")
    return result, binding


def _reference_uncertainty(config, spec, receipt, arrays, fixture):
    precision, _ = _document(
        receipt["reference_precision"], config, kind="V3_REFERENCE_PRECISION", fixture=fixture
    )
    if (
        any(precision.get(key) != spec.get(key) for key in ("origin_id", "role"))
        or precision.get("prediction_binding") != spec["prediction_lock"]
        or precision.get("precision_only") is not True
        or precision.get("predictor_rankings_used") is not False
    ):
        raise ValueError("Reference precision must be independent of predictor rankings")
    evidence, _ = _bound_spec(precision["decision_evidence"])
    if (
        evidence.get("precision_only") is not True
        or evidence.get("predictor_rankings_used") is not False
    ):
        raise ValueError("Reference diagnostics lack frozen precision-only provenance")
    units, prompts = spec["query_units"], spec["probe_ids"]
    shape = (len(units), len(prompts), 4)
    required = ("reference", "reference_estimate_unmasked", "reference_variance")
    if any(name not in arrays or arrays[name].shape != shape for name in required):
        raise ValueError(
            "Aligned reference estimate, variance and precision mask originals required"
        )
    if np.any(np.isfinite(arrays["reference_variance"]) & (arrays["reference_variance"] < 0)):
        raise ValueError("Negative reference variance")
    reports = evidence.get("unit_reports", [])
    keys = [(row["bank_id"], row["contrast_id"], row["prompt_id"]) for row in reports]
    expected = {(u["bank_id"], u["contrast_id"], p) for u in units for p in prompts}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Complete distinct reference uncertainty cell matrix required")
    by_key = dict(zip(keys, reports, strict=True))
    half_width = np.full(shape, np.nan)
    if not fixture:
        from .vlm_response import _reference_protocol

        protocol = _reference_protocol(config)
        if evidence.get("protocol") != protocol:
            raise ValueError(
                "Reference precision protocol/multiplicity differs from preregistration"
            )
    for i, unit in enumerate(units):
        for p, prompt in enumerate(prompts):
            report = by_key[unit["bank_id"], unit["contrast_id"], prompt]
            proposal = report.get("selected_proposal")
            if report.get("reference_is_exact_truth") is not False:
                raise ValueError("Noisy VLM reference cannot be called exact truth")
            selected = proposal.lower() if proposal in {"ORIGIN", "MIX"} else "origin"
            diag = report.get(selected)
            raw_key = f"reference_{selected}_{i}_{p}"
            if not diag or raw_key not in arrays:
                raise ValueError(
                    "Reference uncertainty is missing its original contribution samples"
                )
            raw = arrays[raw_key]
            if raw.ndim != 2 or raw.shape[1] != 4 or len(raw) != diag["n"] or len(raw) < 2:
                raise ValueError("Reference draw ledger shape changed")
            finite = np.isfinite(raw).all()
            if not finite:
                if np.isfinite(arrays["reference"][i, p]).any():
                    raise ValueError("Nonfinite reference raw moments cannot be resolved")
                continue
            mean, variance = raw.mean(0), raw.var(0, ddof=1) / len(raw)
            if not np.allclose(
                mean, arrays["reference_estimate_unmasked"][i, p], atol=1e-13, rtol=1e-9
            ):
                raise ValueError("Reference mean disagrees with raw contribution originals")
            if not np.allclose(variance, arrays["reference_variance"][i, p], atol=1e-22, rtol=1e-8):
                raise ValueError("Reference variance disagrees with raw contribution originals")
            covariance = arrays.get(raw_key + "_covariance")
            if covariance is None or not np.allclose(
                covariance, np.cov(raw, rowvar=False, ddof=1) / len(raw), atol=1e-22, rtol=1e-8
            ):
                raise ValueError("Reference covariance originals are incomplete or inconsistent")
            alpha = diag.get("alpha_per_interval")
            if fixture and alpha is None and proposal not in {"ORIGIN", "MIX"}:
                # The actual Q4-sized fixture report deliberately supplies no
                # Q5 normal interval. Keep it unresolved, never invent a width.
                if np.isfinite(arrays["reference"][i, p]).any():
                    raise ValueError("Below-look reference fixture cannot be resolved")
                continue
            if not isinstance(alpha, (float, int)) or not 0 < alpha < 1:
                raise ValueError("Frozen reference interval multiplicity required")
            if not fixture and (
                diag["n"] not in protocol["looks"]
                or alpha
                != protocol["alpha"] / (len(protocol["looks"]) * protocol["family_cells"] * 4)
                or diag.get("protocol_hash") != canonical_hash(protocol)
            ):
                raise ValueError(
                    "Reference precision changed its frozen cell/look/event multiplicity"
                )
            if proposal == "MIX" and np.max(np.abs(raw)) > 2 + 1e-12:
                raise ValueError("Mixture reference contributions exceed the finite bound")
            se = np.sqrt(variance)
            width = NormalDist().inv_cdf(1 - alpha / 2) * se
            if not np.allclose(
                diag.get("standard_error"), se, atol=1e-14, rtol=1e-8
            ) or not np.allclose(
                diag.get("empirical_normal_half_width"), width, atol=1e-14, rtol=1e-8
            ):
                raise ValueError("Reference uncertainty does not match raw samples")
            zero = raw.var(0, ddof=1) == 0
            if np.any(zero):
                formal = diag.get("formal_hoeffding_half_width")
                if proposal != "MIX" or formal is None:
                    width[zero] = np.inf
                else:
                    expected_formal = 4 * math.sqrt(math.log(2 / alpha) / (2 * len(raw)))
                    if not math.isclose(formal, expected_formal):
                        raise ValueError("Finite-range reference width disagrees with protocol")
                    width[zero] = formal
            resolved = (
                proposal in {"ORIGIN", "MIX"}
                and diag.get("empirical_precision_met") is True
                and (proposal != "ORIGIN" or report.get("crosscheck_consistent") is True)
            )
            reference = arrays["reference"][i, p]
            if resolved:
                if not np.isfinite(reference).all() or not np.allclose(
                    reference, mean, atol=1e-13, rtol=1e-9
                ):
                    raise ValueError("Resolved reference mask differs from precision receipt")
                if (
                    not np.isfinite(width).all()
                    or width.max() > config["qwen"]["reference"]["primary_half_width_goal"]
                ):
                    raise ValueError(
                        "Reference claims precision not supported by original uncertainty"
                    )
                half_width[i, p] = width
            elif np.isfinite(reference).any():
                raise ValueError(
                    "Unresolved reference cannot silently become a finite primary label"
                )
    return half_width


def _collect(config, lock, selection_binding, stage_completion, evaluation_inputs, role, fixture):
    stage, stage_binding = _stage(config, stage_completion, role, fixture)
    records, bindings, seen = [], [], set()
    shared = {}
    fixed_panel = None
    stage_evaluations = {}
    for task_id, task in stage.get("verified_tasks", {}).items():
        if not task_id.startswith("evaluation_"):
            continue
        body, body_binding = _document(
            task["binding"], config, kind="V3_RESPONSE_EVALUATION", fixture=fixture
        )
        body_fit, _ = _bound_spec(body["fit_spec"])
        design = _design(config, lock, selection_binding, body_fit, fixture)
        key = (body["origin_id"], design["design_id"])
        if key in stage_evaluations:
            raise ValueError("Duplicate completed evaluation origin/design")
        stage_evaluations[key] = (body, body_binding)
    for value in evaluation_inputs:
        spec, binding = _bound_spec(value)
        if spec.get("kind") == "V3_RESPONSE_EVALUATION":
            spec, binding = _bound_spec(spec["evaluation_input"])
        spec, binding = _document(
            binding, config, kind="V3_Q5_REFERENCE_EVALUATION_INPUT", fixture=fixture
        )
        if spec.get("role") != role:
            raise ValueError("Evaluation original has the wrong independent seed role")
        seed, arm, step = _origin(config, spec["origin_id"], role)
        if (
            spec.get("reference_kind") != "INDEPENDENT_NOISY_PRECISION_MASKED"
            or spec.get("primary_delta_v") != "-delta_pI"
        ):
            raise ValueError("VLM references must retain uncertainty and primary v=-I")
        root = Path(binding["path"]).parent
        receipt, receipt_binding = _document(
            root / "RESPONSE_EVALUATION_RECEIPT.json",
            config,
            kind="V3_RESPONSE_EVALUATION",
            fixture=fixture,
        )
        if (
            receipt.get("evaluation_input") != binding
            or receipt.get("prediction_lock") != spec["prediction_lock"]
        ):
            raise ValueError("Evaluation receipt is not bound to these exact originals")
        pred, _ = _document(
            spec["prediction_lock"], config, kind="V3_FROZEN_PREDICTIONS", fixture=fixture
        )
        if any(
            pred.get(k) != spec.get(k) for k in ("origin_id", "role", "query_units", "probe_ids")
        ):
            raise ValueError("Prediction lock unit identity differs from evaluation")
        if (
            pred.get("reference_labels_read") is not False
            or pred.get("heldout_labels_read") is not False
        ):
            raise ValueError("Predictions must be frozen before opening test labels")
        access = json.loads((root / "LABEL_ACCESS.json").read_text())
        if (
            access.get("prediction_lock") != spec["prediction_lock"]
            or access.get("prediction_validated_before_any_heldout_reference_labels") is not True
        ):
            raise ValueError("Original label access chronology is missing")
        fit = _fit_geometry(
            config, lock, selection_binding, pred["fit_spec"], pred["model_arrays"], fixture
        )
        stage_entry = stage_evaluations.get((spec["origin_id"], fit["design"]["design_id"]))
        if stage_entry is None or stage_entry[1] != receipt_binding:
            raise ValueError("Exact design evaluation original missing from completed stage matrix")
        if any(
            fit["fit"].get(k) != spec.get(k)
            for k in ("origin_id", "role", "query_units", "probe_ids")
        ):
            raise ValueError("Evaluation query identity differs from original fit")
        units, prompts = spec["query_units"], spec["probe_ids"]
        if fixed_panel is not None and prompts != fixed_panel:
            raise ValueError("All origins must use the same ordered fixed probe panel")
        fixed_panel = prompts
        unit_keys = [(u["bank_id"], u["contrast_id"]) for u in units]
        banks = {u["bank_id"] for u in units}
        if (
            len(prompts) != len(set(prompts))
            or len(prompts) != config["qwen"]["probe_panel"]["prompts"]
        ):
            raise ValueError("Complete unique fixed probe panel required")
        if (
            len(banks) != config["qwen"]["training_bank_partition"]["heldout_banks"]
            or len(unit_keys) != len(set(unit_keys))
            or set(unit_keys) != {(bank, target) for bank in banks for target in TARGETS}
        ):
            raise ValueError("Complete heldout bank and all-three-target matrix required")
        with np.load(_file(spec["arrays"]), allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        with np.load(_file(pred["arrays"]), allow_pickle=False) as data:
            if not np.array_equal(
                data["predictions"], arrays["predictions"], equal_nan=True
            ) or not np.array_equal(data["accepted"], arrays["accepted"]):
                raise ValueError("Evaluation changed immutable prediction/acceptance arrays")
        if not np.array_equal(fit["predictions"], arrays["predictions"], equal_nan=True):
            raise ValueError("Frozen predictions differ from actual model output")
        if (
            arrays["accepted"].dtype.kind != "b"
            or arrays["accepted"].shape != arrays["predictions"].shape[:-1]
        ):
            raise ValueError("Acceptance requires the original aligned boolean mask")
        if pred.get("accepted_count") != int(arrays["accepted"].sum()):
            raise ValueError("Frozen accepted count differs from actual mask")
        widths = _reference_uncertainty(config, spec, receipt, arrays, fixture)
        design_id = fit["design"]["design_id"]
        key = (design_id, seed, arm, step)
        if key in seen:
            raise ValueError("Duplicate independent origin/design in analysis")
        seen.add(key)
        # A shared independent reference is mandatory for paired design comparisons.
        ref_key = spec["origin_id"]
        reference_binding_hash = canonical_hash(receipt["reference_bundle"])
        ref_identity = (
            unit_keys,
            prompts,
            reference_binding_hash,
            arrays["reference_estimate_unmasked"],
            arrays["reference_variance"],
        )
        if ref_key in shared:
            old = shared[ref_key]
            if old[:3] != ref_identity[:3] or any(
                not np.array_equal(a, b, equal_nan=True)
                for a, b in zip(old[3:], ref_identity[3:], strict=True)
            ):
                raise ValueError("Paired designs must share the identical independent reference")
        shared[ref_key] = ref_identity
        records.append(
            {
                **fit,
                "spec": spec,
                "input_binding": binding,
                "receipt_binding": receipt_binding,
                "prediction": pred,
                "seed": seed,
                "arm": arm,
                "step": step,
                "arrays": arrays,
                "reference_half_width": widths,
            }
        )
        bindings.append(binding)
    expected = {
        (d["design_id"], seed, arm, step)
        for d in lock["selected"]["designs"]
        for seed in config["qwen"]["seed_roles"][role]
        for arm, step in (
            [("X_BASE", 32), ("X_BASE", 96)] + ([("X_VALID", 96)] if role == "locked_test" else [])
        )
    }
    if seen != expected:
        raise ValueError("The complete frozen design/seed/arm/origin matrix is required")
    return records, bindings, stage_binding


def _metric(prediction, reference, variance, width, accepted):
    finite = np.isfinite(prediction).all(-1)
    ref_ok = (
        np.isfinite(reference).all(-1) & np.isfinite(variance).all(-1) & np.isfinite(width).all(-1)
    )
    paired = finite & ref_ok
    scoped = paired & accepted
    result = {
        "all_case_count": len(prediction),
        "finite_prediction_count": int(finite.sum()),
        "reference_resolved_count": int(ref_ok.sum()),
        "paired_case_count": int(paired.sum()),
        "accepted_case_count": int(accepted.sum()),
        "accepted_reference_resolved_count": int(scoped.sum()),
        "unknown_case_count": int((~accepted).sum()),
        "geometry_coverage": float(accepted.mean()),
        "reference_is_exact_truth": False,
        "quantiles_are_confidence_intervals": False,
    }
    if paired.any():
        result.update(error_metrics(prediction[paired], reference[paired]))
        result["reference_variance_mean"] = float(variance[paired].mean())
        result["aggregate_reference_noise_corrected_mse"] = (
            result["mse"] - result["reference_variance_mean"]
        )
        result["noise_correction_clipped"] = False
        pxv = error_metrics(
            prediction[paired][:, [0, 3]] * [1, -1], reference[paired][:, [0, 3]] * [1, -1]
        )
        result["pX_v"] = {**pxv, "v_definition": "-delta_pI"}
    else:
        result.update(
            mse=None,
            mae=None,
            nrmse=None,
            q95_absolute_residual=None,
            aggregate_reference_noise_corrected_mse=None,
            pX_v={"v_definition": "-delta_pI"},
        )
    accepted_metrics = None
    if scoped.any():
        accepted_metrics = error_metrics(prediction[scoped], reference[scoped])
        upper = np.abs(prediction[scoped] - reference[scoped]) + width[scoped]
        lower_energy = float(np.sum(np.maximum(0, np.abs(reference[scoped]) - width[scoped]) ** 2))
        accepted_metrics.update(
            q95_absolute_residual_plus_reference_half_width=float(np.quantile(upper, 0.95)),
            nrmse_reference_envelope=math.sqrt(float(np.sum(upper**2)) / lower_energy)
            if lower_energy > 0
            else None,
            reference_half_width_max=float(width[scoped].max()),
            empirical_reference_envelope_is_95_cross_seed_certificate=False,
        )
    result["accepted_metrics"] = accepted_metrics
    return result


def _target_summary(records, target, *, bootstrap, config):
    parts = defaultdict(list)
    seeds = []
    for row in records:
        selected = np.array([u["contrast_id"] == target for u in row["spec"]["query_units"]])
        arrays = row["arrays"]
        shape = (-1, 4)
        parts["reference"].append(arrays["reference_estimate_unmasked"][selected].reshape(shape))
        parts["variance"].append(arrays["reference_variance"][selected].reshape(shape))
        parts["width"].append(row["reference_half_width"][selected].reshape(shape))
        parts["accepted"].append(row["analysis_mask"][selected].reshape(-1))
        seeds.extend([row["seed"]] * row["analysis_mask"][selected].size)
        methods = {
            "MODEL": "predictions",
            "DIRECT_MEASURE": "query_direct_measurement",
            "DIRECT_COUNTS": "direct_count_measurement",
        }
        methods.update(
            {
                "DIRECT_MEASURE_" + k.removeprefix("direct_measure_"): k
                for k in arrays
                if k.startswith("direct_measure_")
            }
        )
        for name, key in methods.items():
            if key not in arrays or arrays[key].shape != arrays["predictions"].shape:
                raise ValueError("Required all-case strong direct baseline originals missing")
            parts[name].append(arrays[key][selected].reshape(shape))
    cat = {key: np.concatenate(value) for key, value in parts.items()}
    seed_array = np.asarray(seeds)
    result = {}
    for name in sorted(set(cat) - {"reference", "variance", "width", "accepted"}):
        mask = cat["accepted"] if name == "MODEL" else np.ones(len(seed_array), dtype=bool)
        result[name] = _metric(cat[name], cat["reference"], cat["variance"], cat["width"], mask)
        raw_finite = (
            np.isfinite(cat[name]) & np.isfinite(cat["reference"]) & np.isfinite(cat["variance"])
        )
        result[name]["all_noisy_reference_diagnostic"] = {
            "finite_event_count": int(raw_finite.sum()),
            "mse": float(np.mean((cat[name][raw_finite] - cat["reference"][raw_finite]) ** 2))
            if raw_finite.any()
            else None,
            "aggregate_reference_noise_corrected_mse": float(
                np.mean(
                    (cat[name][raw_finite] - cat["reference"][raw_finite]) ** 2
                    - cat["variance"][raw_finite]
                )
            )
            if raw_finite.any()
            else None,
            "ranking_allowed": False,
        }
    seed_rows = {}
    for seed in sorted(set(seeds)):
        take = seed_array == seed
        seed_rows[str(seed)] = _metric(
            cat["MODEL"][take],
            cat["reference"][take],
            cat["variance"][take],
            cat["width"][take],
            cat["accepted"][take],
        )
    result["seed_metrics"] = seed_rows
    if bootstrap:
        valid = np.isfinite(cat["MODEL"]).all(-1) & np.isfinite(cat["width"]).all(-1)
        result["MODEL"]["seed_cluster_intervals"] = _cluster_error_intervals(
            seed_array, cat["MODEL"], cat["reference"], valid, config
        )
        result["MODEL"]["accepted_seed_cluster_intervals"] = _cluster_error_intervals(
            seed_array, cat["MODEL"], cat["reference"], valid & cat["accepted"], config
        )
        paired = (
            np.isfinite(cat["MODEL"]).all(-1)
            & np.isfinite(cat["DIRECT_MEASURE"]).all(-1)
            & np.isfinite(cat["width"]).all(-1)
        )
        # Preserve every seed and its full reference-resolved pair; incomplete data cannot rank.
        if paired.all():
            delta = np.mean(
                (cat["MODEL"] - cat["reference"]) ** 2
                - (cat["DIRECT_MEASURE"] - cat["reference"]) ** 2,
                axis=-1,
            )
            result["paired_model_minus_direct"] = seed_bootstrap(
                seed_array, delta, reps=5000, seed=config["statistics"]["bootstrap_seed"]
            )
            result["paired_model_minus_direct"].update(
                metric="paired raw MSE difference on shared independent reference",
                reference_noise_cancels_in_paired_difference=True,
                ranking_status="EMPIRICAL_REFERENCE_PRECISION_RESOLVED",
            )
        else:
            result["paired_model_minus_direct"] = {
                "status": "UNKNOWN_REFERENCE_OR_PREDICTION_UNRESOLVED",
                "reps": 0,
                "independent_seeds": len(set(seeds)),
                "ranking_allowed": False,
            }
    return result


def _cluster_error_intervals(seeds, prediction, reference, valid, config):
    """Resample whole seeds; ratios use summed SSE/energy, not mean ratios."""
    totals = []
    for seed in sorted(set(seeds)):
        take = (seeds == seed) & valid
        residual = prediction[take] - reference[take]
        totals.append(
            [
                np.abs(residual).sum(),
                np.sum(residual**2),
                np.sum(reference[take] ** 2),
                residual.size,
            ]
        )
    totals = np.asarray(totals)
    count = totals[:, 3].sum()
    if count == 0:
        return {"status": "UNKNOWN_NO_RESOLVED_EVENTS", "reps": 0, "independent_seeds": len(totals)}
    rng = np.random.default_rng(config["statistics"]["bootstrap_seed"])
    draws = totals[rng.integers(0, len(totals), (5000, len(totals)))].sum(axis=1)
    result = {
        "status": "EMPIRICAL_SEED_CLUSTER_BOOTSTRAP",
        "reps": 5000,
        "independent_seeds": len(totals),
        "unit": "training_seed",
        "undefined_resamples": int((draws[:, 3] == 0).sum()),
        "reference_is_exact_truth": False,
    }
    for name, numerator, denominator, power in (
        ("mae", 0, 3, 1),
        ("mse", 1, 3, 1),
        ("nrmse", 1, 2, 0.5),
    ):
        summed = totals.sum(axis=0)
        values = np.full(5000, np.nan)
        np.divide(
            draws[:, numerator], draws[:, denominator], out=values, where=draws[:, denominator] > 0
        )
        values **= power
        result[name] = {
            "estimate": float((summed[numerator] / summed[denominator]) ** power)
            if summed[denominator] > 0
            else None,
            "interval": np.quantile(values, [0.025, 0.975]).tolist()
            if np.isfinite(values).all()
            else None,
            "undefined_resamples": int((~np.isfinite(values)).sum()),
        }
    return result


def _seed_envelopes(rows):
    scores = {}
    for seed in sorted({row["seed"] for row in rows}):
        blocks = []
        unresolved = False
        for row in rows:
            if row["seed"] != seed:
                continue
            mask = row["analysis_mask"]
            if not mask.any():
                continue
            values = (
                np.abs(row["arrays"]["predictions"] - row["arrays"]["reference_estimate_unmasked"])
                + row["reference_half_width"]
            )[mask]
            if not np.isfinite(values).all():
                unresolved = True
            blocks.append(values)
        scores[str(seed)] = (
            float(np.concatenate(blocks).max()) if blocks and not unresolved else None
        )
    return scores


def _failures(summary, criteria):
    failures = []
    if summary["geometry_coverage"] < criteria["minimum_geometry_coverage"]:
        failures.append("GEOMETRY_COVERAGE_BELOW_FROZEN_MINIMUM")
    if summary["accepted_case_count"] == 0:
        failures.append("NO_ACCEPTED_QUERIES")
    if summary["accepted_reference_resolved_count"] != summary["accepted_case_count"]:
        failures.append("ACCEPTED_REFERENCE_UNRESOLVED")
    metrics = summary["accepted_metrics"]
    if metrics is None:
        failures.append("POINTWISE_ERROR_UNRESOLVED")
    else:
        if metrics["reference_half_width_max"] > criteria["reference_half_width_max"]:
            failures.append("REFERENCE_PRECISION_INSUFFICIENT")
        for limit, metric, failure in (
            (
                "maximum_q95_absolute_residual",
                "q95_absolute_residual_plus_reference_half_width",
                "Q95_EXCEEDS_FROZEN_LIMIT",
            ),
            ("maximum_nrmse", "nrmse_reference_envelope", "NRMSE_EXCEEDS_FROZEN_LIMIT"),
        ):
            if criteria[limit] is not None and (
                metrics[metric] is None or metrics[metric] > criteria[limit]
            ):
                failures.append(failure)
    return failures


def _summarize(records, config, lock, *, calibration):
    result = {}
    criteria = lock["selected"]["pointwise_criteria"]
    for design in lock["selected"]["designs"]:
        rows = [r for r in records if r["design"]["design_id"] == design["design_id"]]
        entry = {
            "design": design,
            "dimensions": [
                {
                    "origin_id": r["spec"]["origin_id"],
                    "k": r["k"],
                    "r": r["r"],
                    "r_less_k": 0 < r["r"] < r["k"],
                }
                for r in rows
            ],
        }
        failures = []
        for name, arm in (("primary", "X_BASE"), ("shift", "X_VALID")):
            current = [row for row in rows if row["arm"] == arm]
            if not current:
                continue
            targets = {
                target: _target_summary(current, target, bootstrap=not calibration, config=config)
                for target in TARGETS
            }
            entry[name] = {
                "origins": len(current),
                "seeds": sorted({r["seed"] for r in current}),
                "targets": targets,
                "distribution_shift_covered": False,
            }
            if name == "primary":
                for target, value in targets.items():
                    failures.extend(
                        target + ":" + item for item in _failures(value["MODEL"], criteria)
                    )
                    # Pooling must not conceal a bad independent calibration seed.
                    if calibration:
                        for seed, metrics in value["seed_metrics"].items():
                            failures.extend(
                                f"{target}:seed{seed}:{item}"
                                for item in _failures(metrics, criteria)
                            )
        entry["failures"] = sorted(set(failures))
        seed_scores = _seed_envelopes([row for row in rows if row["arm"] == "X_BASE"])
        entry["primary_seed_maximum_error_envelopes"] = seed_scores
        entry["empirical_seed_maximum_envelope"] = (
            max(seed_scores.values())
            if all(value is not None for value in seed_scores.values())
            else None
        )
        entry[
            "pointwise_calibration_eligible" if calibration else "pointwise_qualified"
        ] = not failures
        entry["compression_claim_status"] = (
            "REQUIRES_MATCHED_FULL_SPACE_INDEPENDENT_TEST_COMPARISON"
        )
        curve = []
        for threshold in sorted(
            set(config["coverage"]["coverage_bins"] + [lock["selected"]["rho_threshold"]])
        ):
            scoped = []
            for row in rows:
                if row["arm"] != "X_BASE":
                    continue
                mask = row["analysis_mask"] & (row["geometry"]["rho"] <= threshold)[:, None]
                scoped.append({**row, "analysis_mask": mask})
            curve.append(
                {
                    "rho_threshold": threshold,
                    "threshold_source": "FROZEN_PROTOCOL_BINS",
                    "used_to_select_test_acceptance": False,
                    "targets": {
                        target: _target_summary(scoped, target, bootstrap=False, config=config)[
                            "MODEL"
                        ]
                        for target in TARGETS
                    },
                }
            )
        entry["risk_coverage_curve"] = curve
        result[design["design_id"]] = entry
    return result


def _csv_tables(root, designs):
    for filename, by_seed in (("TARGET_METRICS.csv", False), ("SEED_METRICS.csv", True)):
        rows = []
        for design_id, design in designs.items():
            for scope in ("primary", "shift"):
                for target, summary in design.get(scope, {}).get("targets", {}).items():
                    items = (
                        summary["seed_metrics"].items() if by_seed else (("ALL", summary["MODEL"]),)
                    )
                    for seed, metric in items:
                        rows.append(
                            {
                                "design_id": design_id,
                                "scope": scope,
                                "target": target,
                                "seed": seed,
                                **{
                                    k: metric.get(k)
                                    for k in (
                                        "all_case_count",
                                        "accepted_case_count",
                                        "reference_resolved_count",
                                        "geometry_coverage",
                                        "mse",
                                        "mae",
                                        "nrmse",
                                        "q95_absolute_residual",
                                        "aggregate_reference_noise_corrected_mse",
                                    )
                                },
                            }
                        )
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_bytes(root / filename, stream.getvalue().encode())


def _base_report(config, lock, selection_binding, stage_binding, input_bindings, fixture):
    return {
        **_identity(config),
        "fixture": fixture,
        "requires_server_cpu": not fixture,
        "selection_lock": selection_binding,
        "selection_hash": lock["selection_hash"],
        "stage_completion": stage_binding,
        "input_bindings": input_bindings,
        "pointwise_criteria": lock["selected"]["pointwise_criteria"],
        "criteria_scope": (
            "Explicit development-selected operational criteria, not a 95% safety certificate"
        ),
        "error_limit_application": (
            "q95 of absolute residual plus empirical reference half-width; NRMSE uses "
            "reference error envelope and lower reference signal energy"
        ),
        "geometry_metric": (
            "uncentered actual e span; identity row metric ridge leverage; frozen alpha"
        ),
        "cross_seed_95": NOT_CERTIFIED,
        "online_ssvc": NOT_CERTIFIED,
        "unseen_prompt_generalization": NOT_CERTIFIED,
        "distribution_shift_covered": False,
    }


def _empirical_interval_coverage(designs, calibration):
    for design_id, result in designs.items():
        radius = calibration["designs"][design_id]["empirical_seed_maximum_envelope"]
        scores = result["primary_seed_maximum_error_envelopes"]
        covered = sum(
            radius is not None and value is not None and value <= radius
            for value in scores.values()
        )
        result["calibration_envelope_test_coverage"] = {
            "radius": radius,
            "interval_width": 2 * radius if radius is not None else None,
            "covered_seeds": covered if radius is not None else None,
            "unresolved_seeds": sum(value is None for value in scores.values()),
            "all_test_seeds_denominator": len(scores),
            "empirical_coverage_lower_fraction": covered / len(scores)
            if radius is not None
            else None,
            "nominal_95_certified": False,
            "scope": "seed maximum over all accepted primary origins/targets/probes",
            "reference_uncertainty_included": True,
        }


def _dimension_comparisons(records, config):
    by_design = defaultdict(list)
    for row in records:
        if row["arm"] == "X_BASE":
            by_design[row["design"]["design_id"]].append(row)
    result = []
    for name, low in by_design.items():
        design = low[0]["design"]
        if design["method"] in {"ZERO", "FULL_RIDGE", "FULL_GLS", "RBF_RIDGE"}:
            continue
        for base_id, full in by_design.items():
            base = full[0]["design"]
            objective = "GLS" if design["regression"] == "GLS" else "RIDGE"
            full_objective = "GLS" if base["method"] == "FULL_GLS" else base["regression"]
            matching = (
                base["method"] in {"FULL_RIDGE", "FULL_GLS"}
                and base["rank_cap"] == "FULL"
                and full_objective == objective
                and all(
                    design[key] == base[key]
                    for key in DESIGN_KEYS
                    if key not in {"method", "rank_cap", "regression"}
                )
            )
            if not matching:
                continue
            full_by_origin = {row["spec"]["origin_id"]: row for row in full}
            eligible = all(
                0 < row["r"] < row["k"]
                and full_by_origin[row["spec"]["origin_id"]]["r"] == row["k"]
                for row in low
            )
            comparison = {
                "reduced_design": name,
                "full_design": base_id,
                "r_less_k_every_origin": eligible,
                "same_frozen_regression_objective": True,
                "targets": {},
            }
            for target in TARGETS:
                seeds, values, unresolved = [], [], False
                for row in low:
                    other = full_by_origin[row["spec"]["origin_id"]]
                    target_mask = np.array(
                        [unit["contrast_id"] == target for unit in row["spec"]["query_units"]]
                    )
                    pred = row["arrays"]["predictions"][target_mask]
                    baseline = other["arrays"]["predictions"][target_mask]
                    reference = row["arrays"]["reference_estimate_unmasked"][target_mask]
                    precise = np.isfinite(row["reference_half_width"][target_mask]).all()
                    if (
                        not precise
                        or not np.isfinite(pred).all()
                        or not np.isfinite(baseline).all()
                    ):
                        unresolved = True
                        continue
                    values.append(
                        float(np.mean((pred - reference) ** 2 - (baseline - reference) ** 2))
                    )
                    seeds.append(row["seed"])
                comparison["targets"][target] = (
                    {"status": "NOT_ELIGIBLE_R_EQUALS_K_OR_UNRESOLVED", "ranking_allowed": False}
                    if not eligible or unresolved
                    else seed_bootstrap(
                        seeds, values, reps=5000, seed=config["statistics"]["bootstrap_seed"]
                    )
                )
            result.append(comparison)
    return result


def analyze_vlm_calibration(
    config, selection_lock, *, stage_completion, evaluation_inputs, out, fixture=False
):
    _cpu_scope(fixture=fixture)
    lock, selection_binding = _selection(config, selection_lock, fixture)
    seeds = config["qwen"]["seed_roles"]["interval_calibration"]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Exactly three distinct frozen VLM calibration seeds required")
    records, inputs, completion = _collect(
        config,
        lock,
        selection_binding,
        stage_completion,
        evaluation_inputs,
        "interval_calibration",
        fixture,
    )
    for row in records:
        if row["arrays"]["accepted"].any():
            raise ValueError("Calibration predictions cannot already claim test acceptance")
        row["analysis_mask"] = row["geometry_mask"]
    designs = _summarize(records, config, lock, calibration=True)
    report = {
        "kind": "V3_VLM_CALIBRATION",
        "status": "CALIBRATION_ANALYZED",
        **_base_report(config, lock, selection_binding, completion, inputs, fixture),
        "calibration_seeds": sorted(seeds),
        "designs": designs,
        "conformal_95": {
            "status": "INSUFFICIENT_CALIBRATION_SEEDS",
            "radius": None,
            "n_calibration_seeds": len(seeds),
            "order_statistic": math.ceil((len(seeds) + 1) * 0.95),
            "nominal": 0.95,
            "distribution_shift_covered": False,
            "coverage_object": "one future exchangeable training-seed maximum error",
        },
        "accepted_mask_rule": (
            "geometry and finite prediction, gated by all-three-target empirical "
            "calibration eligibility; never by test errors"
        ),
    }
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "CALIBRATION_ANALYSIS.json", report)
    _csv_tables(root, designs)
    finalize_run(root, _identity(config))
    return {**report, "receipt": _binding(root / "CALIBRATION_ANALYSIS.json")}


def verify_vlm_calibration(config, selection_lock, calibration_receipt, *, fixture=False):
    _cpu_scope(fixture=fixture)
    lock, selection_binding = _selection(config, selection_lock, fixture)
    report, binding = _document(
        calibration_receipt, config, kind="V3_VLM_CALIBRATION", fixture=fixture
    )
    if (
        report.get("status") != "CALIBRATION_ANALYZED"
        or report.get("selection_lock") != selection_binding
        or report.get("selection_hash") != lock["selection_hash"]
        or report.get("calibration_seeds")
        != sorted(config["qwen"]["seed_roles"]["interval_calibration"])
        or len(report["calibration_seeds"]) != 3
        or report.get("cross_seed_95") != NOT_CERTIFIED
        or report.get("pointwise_criteria") != lock["selected"]["pointwise_criteria"]
    ):
        raise ValueError("Calibration identity, seed role or frozen criteria changed")
    records, _, _ = _collect(
        config,
        lock,
        selection_binding,
        report["stage_completion"],
        report["input_bindings"],
        "interval_calibration",
        fixture,
    )
    for row in records:
        if row["arrays"]["accepted"].any():
            raise ValueError("Calibration mask claims unfrozen acceptance")
        row["analysis_mask"] = row["geometry_mask"]
    expected = _summarize(records, config, lock, calibration=True)
    if expected != report.get("designs"):
        raise ValueError("Calibration eligibility/report differs from original measured errors")
    return {**report, "receipt": binding}


def derive_vlm_acceptance(
    config, selection_lock, calibration_receipt, fit_spec, model_arrays, *, fixture=False
):
    _cpu_scope(fixture=fixture)
    lock, selection_binding = _selection(config, selection_lock, fixture)
    calibration = verify_vlm_calibration(
        config, selection_lock, calibration_receipt, fixture=fixture
    )
    fit = _fit_geometry(config, lock, selection_binding, fit_spec, model_arrays, fixture)
    if fit["fit"]["role"] != "locked_test":
        raise ValueError("Empirical acceptance is only available after calibration for locked_test")
    # Window anchor 64 can use this same frozen rule when a real Q6 fit exists.
    _origin(
        config,
        fit["fit"]["origin_id"],
        "locked_test",
        tracking=fit["fit"]["origin_id"].endswith("_64"),
    )
    design_id = fit["design"]["design_id"]
    eligible = calibration["designs"][design_id]["pointwise_calibration_eligible"]
    geometry = fit["geometry"]
    return {
        "accepted": fit["geometry_mask"] & eligible,
        "rho": geometry["rho"],
        "leverage": geometry["leverage"],
        "e_norm": geometry["e_norm"],
        "k": fit["k"],
        "r": fit["r"],
        "design_id": design_id,
        "calibration_binding": calibration["receipt"],
        "selection_binding": selection_binding,
        "acceptance_kind": "V3_VLM_EMPIRICAL_GEOMETRY_ACCEPTANCE",
        "calibration_eligible": eligible,
        "cross_seed_95": NOT_CERTIFIED,
        "test_reference_labels_read": False,
    }


def _test_records(
    config, lock, selection_binding, calibration, stage_completion, evaluation_inputs, fixture
):
    records, inputs, completion = _collect(
        config, lock, selection_binding, stage_completion, evaluation_inputs, "locked_test", fixture
    )
    for row in records:
        design_id = row["design"]["design_id"]
        expected = (
            row["geometry_mask"]
            & calibration["designs"][design_id]["pointwise_calibration_eligible"]
        )
        if row["prediction"].get("calibration_receipt") != calibration["receipt"]:
            raise ValueError("Test prediction acceptance was not bound to preceding calibration")
        if not np.array_equal(row["arrays"]["accepted"], expected):
            raise ValueError("Test acceptance mask differs from frozen calibration/geometry rule")
        row["analysis_mask"] = expected
    return records, inputs, completion


def analyze_vlm_test(
    config,
    selection_lock,
    *,
    calibration_receipt,
    stage_completion,
    evaluation_inputs,
    out,
    fixture=False,
):
    _cpu_scope(fixture=fixture)
    lock, selection_binding = _selection(config, selection_lock, fixture)
    calibration = verify_vlm_calibration(
        config, selection_lock, calibration_receipt, fixture=fixture
    )
    seeds = config["qwen"]["seed_roles"]["locked_test"]
    if (
        len(seeds) != 6
        or len(set(seeds)) != 6
        or set(seeds) & set(calibration["calibration_seeds"])
    ):
        raise ValueError("Six distinct independent VLM test seeds required")
    records, inputs, completion = _test_records(
        config, lock, selection_binding, calibration, stage_completion, evaluation_inputs, fixture
    )
    designs = _summarize(records, config, lock, calibration=False)
    _empirical_interval_coverage(designs, calibration)
    report = {
        "kind": "V3_VLM_TEST_ANALYSIS",
        "status": "TEST_ANALYZED",
        **_base_report(config, lock, selection_binding, completion, inputs, fixture),
        "calibration_receipt": calibration["receipt"],
        "test_seeds": sorted(seeds),
        "designs": designs,
        "statistical_unit": (
            "training seed; keep arms, anchors, banks, prompts and all targets together"
        ),
        "bootstrap_reps": 5000,
        "formal_hypotheses": {
            "status": "NOT_PERFORMED",
            "holm": None,
            "reason": (
                "No complete predeclared four-hypothesis VLM test family is manufactured "
                "from exploratory error comparisons"
            ),
        },
        "dimension_comparisons": _dimension_comparisons(records, config),
    }
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "TEST_ANALYSIS.json", report)
    qualification = _qualification_body(
        config, lock, report, _binding(root / "TEST_ANALYSIS.json"), fixture
    )
    atomic_json(root / "POINTWISE_QUALIFICATION.json", qualification)
    _csv_tables(root, designs)
    finalize_run(root, _identity(config))
    return {
        **report,
        "receipt": _binding(root / "TEST_ANALYSIS.json"),
        "qualification_receipt": _binding(root / "POINTWISE_QUALIFICATION.json"),
    }


def _qualification_body(config, lock, report, test_binding, fixture):
    qualified = sorted(
        key for key, value in report["designs"].items() if value["pointwise_qualified"]
    )
    tracking = [
        key
        for key in qualified
        if report["designs"][key]["design"]["method"] not in {"ZERO", "RBF_RIDGE"}
        and all(
            dimension["r"] > 0
            for dimension in report["designs"][key]["dimensions"]
            if "_X_BASE_" in dimension["origin_id"]
        )
    ]
    return {
        "kind": "V3_VLM_POINTWISE_QUALIFICATION",
        **_identity(config),
        "fixture": fixture,
        "status": "QUALIFIED_EMPIRICALLY" if qualified else "NOT_QUALIFIED",
        "pointwise_qualified": bool(qualified),
        "qualified_design_ids": qualified,
        "tracking_eligible_design_ids": tracking,
        "tracking_qualified": bool(tracking),
        "selection_lock": report["selection_lock"],
        "selection_hash": lock["selection_hash"],
        "calibration_receipt": report["calibration_receipt"],
        "test_analysis": test_binding,
        "test_seeds": report["test_seeds"],
        "cross_seed_95": NOT_CERTIFIED,
        "online_ssvc": NOT_CERTIFIED,
        "distribution_shift_covered": False,
        "primary_arm": "X_BASE",
        "window_anchor": 64,
        "horizons": config["tracking_offline"]["horizons"],
        "failure_conditions": {key: value["failures"] for key, value in report["designs"].items()},
        "scope_limits": [
            "fixed known probe panel",
            "same frozen design and source/config identity",
            "geometry acceptance from calibration before reference access",
            "offline frozen basis only; no source training feedback",
            "Q6 still requires its own actual step-64 calibration and tracking originals",
        ],
    }


def verify_vlm_qualification(config, binding, *, fit_binding=None, fixture=False):
    _cpu_scope(fixture=fixture)
    qualification, _ = _document(
        binding, config, kind="V3_VLM_POINTWISE_QUALIFICATION", fixture=fixture
    )
    report, test_binding = _document(
        qualification["test_analysis"], config, kind="V3_VLM_TEST_ANALYSIS", fixture=fixture
    )
    lock, selection_binding = _selection(config, report["selection_lock"], fixture)
    calibration = verify_vlm_calibration(
        config, selection_binding, report["calibration_receipt"], fixture=fixture
    )
    if report.get("status") != "TEST_ANALYZED" or report.get("test_seeds") != sorted(
        config["qwen"]["seed_roles"]["locked_test"]
    ):
        raise ValueError("Qualification requires the completed independent six-seed test analysis")
    records, _, _ = _test_records(
        config,
        lock,
        selection_binding,
        calibration,
        report["stage_completion"],
        report["input_bindings"],
        fixture,
    )
    expected_designs = _summarize(records, config, lock, calibration=False)
    _empirical_interval_coverage(expected_designs, calibration)
    if expected_designs != report.get("designs") or _dimension_comparisons(
        records, config
    ) != report.get("dimension_comparisons"):
        raise ValueError("Test qualification differs from actual original independent-test errors")
    expected = _qualification_body(config, lock, report, test_binding, fixture)
    if qualification != expected:
        raise ValueError(
            "Qualification receipt is not derived from the frozen complete test report"
        )
    if fit_binding is not None:
        fit, actual_binding = _document(
            fit_binding, config, kind="V3_Q5_RESPONSE_FIT", fixture=fixture
        )
        if fit.get("role") != "locked_test":
            raise ValueError("Tracking fit requires independent locked-test role")
        _origin(config, fit["origin_id"], "locked_test", tracking=True)
        design = _design(config, lock, selection_binding, fit, fixture)
        qualification = {
            **qualification,
            "fit_binding": actual_binding,
            "fit_hash": actual_binding["sha256"],
            "design_id": design["design_id"],
            "pointwise_qualified": design["design_id"] in qualification["qualified_design_ids"],
            "tracking_qualified": design["design_id"]
            in qualification["tracking_eligible_design_ids"],
        }
    return qualification
