"""Finite anchor-score diagnostics without a dense parameter Fisher matrix.

Inputs are actual per-draw autograd score/direction products. A finite sequence
log-ratio signature is not a score derivative and is deliberately rejected. The
Euclidean calibration projection is label-free; its residual may have MORE
score energy than the original direction, so energy ratios are never clipped.
"""

from __future__ import annotations

import copy
import re

import numpy as np

from .full_kernels import _spectrum

EVENTS = ("X", "S", "W", "I")


def _finite(value, name, ndim):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != ndim or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} needs nonempty finite declared axes")
    return value


def _identities(values, name, *, nonempty=True):
    if (
        not isinstance(values, (list, tuple))
        or (nonempty and not values)
        or any(not isinstance(v, str) or not v for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} needs unique explicit identities")
    return list(values)


def unavailable_empirical_diagnostics(reason, query_direction_ids):
    """Missing derivatives are unavailable, never zero energy or signature Fisher."""
    if not isinstance(reason, str) or not reason:
        raise ValueError("An actual reason for unavailable derivatives is required")
    return {
        "status": "NOT_AVAILABLE",
        "reason": reason,
        "query_direction_ids": _identities(query_direction_ids, "query directions"),
        "fisher_residual": None,
        "score_residual": None,
        "metadata": {
            "exact_fisher": False,
            "full_fisher_matrix_materialized": False,
            "finite_sample_energy_is_a_bound": False,
            "signature_substituted_for_derivative": False,
        },
    }


def _provenance(provenance, sample_records, samples, calibration_count, query_count):
    if not isinstance(provenance, dict):
        raise ValueError("Actual derivative provenance required")
    required = {
        "derivative_kind": "AUTOGRAD_FULL_PARAMETER_SCORE",
        "expansion_point": "ORIGIN",
        "query_reference_labels_read": False,
        "cost_scope": "SHARED_ORIGIN_GRADIENT_COLLECTION",
    }
    if any(provenance.get(k) != v for k, v in required.items()):
        raise ValueError("Only actual full-parameter origin autograd scores are Fisher inputs")
    if (
        not isinstance(provenance.get("origin_id"), str)
        or not provenance["origin_id"]
        or not isinstance(provenance.get("proposal_policy_fingerprint"), str)
        or not provenance["proposal_policy_fingerprint"]
        or provenance.get("expansion_policy_fingerprint")
        != provenance["proposal_policy_fingerprint"]
        or not isinstance(provenance.get("parameter_layout_hash"), str)
        or re.fullmatch(r"[a-f0-9]{64}", provenance["parameter_layout_hash"]) is None
    ):
        raise ValueError("Origin policy, full parameter layout and same-policy draws required")
    calibration_ids = _identities(
        provenance.get("calibration_direction_ids"), "calibration directions"
    )
    query_ids = _identities(provenance.get("query_direction_ids"), "query directions")
    if (
        len(calibration_ids) != calibration_count
        or len(query_ids) != query_count
        or set(calibration_ids) & set(query_ids)
    ):
        raise ValueError("Declared calibration and query direction axes differ")
    reference_ids = set(
        _identities(provenance.get("reference_sample_ids"), "reference samples", nonempty=False)
    )
    reference_rng = set(_identities(provenance.get("reference_rng_stream_ids"), "reference RNG"))
    if not isinstance(sample_records, (tuple, list)) or len(sample_records) != samples:
        raise ValueError("One actual identity/event record per independent anchor draw required")
    seen, prompts, groups, labels = set(), {}, [], []
    for index, row in enumerate(sample_records):
        if not isinstance(row, dict):
            raise ValueError("Actual sample records required")
        for name in ("sample_id", "prompt_id", "rng_namespace"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError("Missing anchor draw identity")
        if (
            row.get("proposal") != "ORIGIN"
            or row.get("proposal_policy_fingerprint") != provenance["expansion_policy_fingerprint"]
        ):
            raise ValueError("Empirical Fisher requires actual same-origin proposal samples")
        if (
            row["sample_id"] in seen
            or row["sample_id"] in reference_ids
            or row["rng_namespace"] in reference_rng
        ):
            raise ValueError("Duplicate anchor or independent-reference sample/RNG reuse")
        seen.add(row["sample_id"])
        group = row.get("group")
        if (
            not isinstance(group, (list, tuple))
            or len(group) != 2
            or any(not isinstance(v, str) or not v for v in group)
        ):
            raise ValueError("Actual family/interface group required")
        group = tuple(group)
        prompt = prompts.setdefault(row["prompt_id"], {"group": group, "indices": []})
        if prompt["group"] != group or row.get("category") not in EVENTS:
            raise ValueError("Prompt group and exhaustive event identities must remain fixed")
        prompt["indices"].append(index)
        if group not in groups:
            groups.append(group)
        labels.append(EVENTS.index(row["category"]))
    if any(len(p["indices"]) < 2 for p in prompts.values()):
        raise ValueError("At least two actual draws per fixed prompt required")
    cost = provenance.get("cost")
    if not isinstance(cost, dict) or any(
        type(cost.get(k)) is not int or cost[k] < samples
        for k in ("backward_calls", "gradient_score_calls", "scored_sequences", "scored_tokens")
    ):
        raise ValueError("Actual shared derivative collection cost must account for every draw")
    return calibration_ids, query_ids, prompts, groups, np.asarray(labels)


def diagnostics_from_collection(
    collection,
    calibration_gram,
    query_cross_gram,
    query_norms,
    *,
    origin_id,
    calibration_direction_ids,
    query_direction_ids,
    reference_sample_ids,
    reference_rng_stream_ids,
):
    """Consume ``collect_score_response`` raw per-draw products, without refitting.

    The caller verifies the collection artifact's immutable binding. This adapter
    checks its actual direction/sample ordering, ORIGIN policy and unweighted
    derivative provenance. BASE/MIDPOINT, importance-weighted products and finite
    log-ratio signatures do not have this empirical-Fisher interpretation.
    """
    if not isinstance(collection, dict) or (
        collection.get("expansion_point") != "ORIGIN"
        or collection.get("reference_labels_read") is not False
        or collection.get("finite_difference_is_derivative") is not False
        or collection.get("raw_score_products_importance_weighted") is not False
        or collection.get("derivative_kind") != "AUTOGRAD_FULL_PARAMETER_SCORE"
    ):
        raise ValueError("Actual unweighted ORIGIN autograd collection required")
    direction_ids = _identities(collection.get("direction_ids"), "actual collected directions")
    calibration_ids = _identities(calibration_direction_ids, "calibration directions")
    query_ids = _identities(query_direction_ids, "query directions")
    if set(calibration_ids + query_ids) - set(direction_ids):
        raise ValueError("Required directions were not actually differentiated")
    products = _finite(collection.get("raw_directional_score_products"), "actual score products", 2)
    records = collection.get("sample_records")
    if (
        not isinstance(records, (tuple, list))
        or len(records) != len(products)
        or products.shape[1] != len(direction_ids)
        or any(not isinstance(r, dict) for r in records)
        or collection.get("sample_ids") != [r.get("sample_id") for r in records]
        or sorted(collection.get("rng_stream_ids", []))
        != sorted({r.get("rng_namespace") for r in records})
    ):
        raise ValueError("Actual collected direction/sample/RNG order differs")
    policy = collection.get("expansion_policy_identity")
    if not isinstance(policy, dict) or not policy.get("inference_fingerprint"):
        raise ValueError("Actual loaded expansion-policy identity required")
    provenance = {
        "origin_id": origin_id,
        "derivative_kind": collection["derivative_kind"],
        "expansion_point": "ORIGIN",
        "proposal_policy_fingerprint": policy["inference_fingerprint"],
        "expansion_policy_fingerprint": policy["inference_fingerprint"],
        "parameter_layout_hash": collection.get("parameter_layout_hash"),
        "calibration_direction_ids": calibration_ids,
        "query_direction_ids": query_ids,
        "reference_sample_ids": reference_sample_ids,
        "reference_rng_stream_ids": reference_rng_stream_ids,
        "query_reference_labels_read": False,
        "cost": collection.get("cost"),
        "cost_scope": "SHARED_ORIGIN_GRADIENT_COLLECTION",
    }
    return empirical_score_diagnostics(
        calibration_gram,
        query_cross_gram,
        query_norms,
        products[:, [direction_ids.index(d) for d in calibration_ids]],
        products[:, [direction_ids.index(d) for d in query_ids]],
        sample_records=records,
        provenance=provenance,
    )


def empirical_score_diagnostics(
    calibration_gram,
    query_cross_gram,
    query_norms,
    calibration_score_products,
    query_score_products,
    *,
    sample_records,
    provenance,
):
    """Evaluate S(e-Proj_cal e), retaining full derivative and semantic diagnostics.

    Gram axes are M by M, Q by M and Q. Score products are S by M and S by Q,
    where S contains actual iid draws within each frozen prompt. Every score is
    grad(log pi_origin(action)) in the complete trainable parameter coordinates.
    Prompts have equal weight, and groups average their own prompts equally.
    Reference outcomes are never accepted. This function performs no model call.
    """
    gram = _finite(calibration_gram, "calibration Gram", 2)
    cross = _finite(query_cross_gram, "query cross Gram", 2)
    norms = _finite(query_norms, "query squared norms", 1)
    cal_scores = _finite(calibration_score_products, "calibration score products", 2)
    query_scores = _finite(query_score_products, "query score products", 2)
    if (
        gram.shape != (len(gram), len(gram))
        or cross.shape != (len(norms), len(gram))
        or cal_scores.shape[1] != len(gram)
        or query_scores.shape != (len(cal_scores), len(norms))
        or np.any(norms < 0)
    ):
        raise ValueError("Gram, norm and per-draw score direction axes differ")
    cal_ids, query_ids, prompts, groups, labels = _provenance(
        provenance, sample_records, len(cal_scores), len(gram), len(norms)
    )
    values, vectors, keep, threshold = _spectrum(gram)
    zero_calibration = np.diag(gram) == 0
    if np.any(cross[:, zero_calibration] != 0) or np.any(cal_scores[:, zero_calibration] != 0):
        raise ValueError("Zero calibration directions must have zero inner/score products")
    cauchy_bound = np.sqrt(norms[:, None] * np.maximum(np.diag(gram), 0)[None])
    if np.any(np.abs(cross) > cauchy_bound * (1 + 1e-8)):
        raise ValueError("Query/calibration inner products violate parameter norms")
    basis = vectors[:, keep]
    coefficients = (cross @ basis / values[keep]) @ basis.T
    covered_norms = np.sum(coefficients * cross, axis=1)
    tolerance = 1e-8 * np.maximum(norms, np.finfo(float).tiny)
    if np.any(covered_norms - norms > tolerance):
        raise ValueError("Query norms/cross Gram cannot describe these parameter directions")
    zero = norms == 0
    if np.any(cross[zero] != 0) or np.any(query_scores[:, zero] != 0):
        raise ValueError("A zero parameter direction must have exactly zero score products")
    residual_norms = np.maximum(norms - covered_norms, 0)
    rho = np.sqrt(np.divide(residual_norms, norms, out=np.zeros_like(norms), where=norms > 0))
    score_residuals = query_scores - cal_scores @ coefficients.T
    indicator = np.eye(4)[labels]
    prompt_full, prompt_energy, prompt_event = [], [], []
    for prompt in prompts.values():
        indices = prompt["indices"]
        prompt_full.append(np.mean(query_scores[indices] ** 2, axis=0))
        prompt_energy.append(np.mean(score_residuals[indices] ** 2, axis=0))
        prompt_event.append(
            np.einsum("sq,se->qe", score_residuals[indices], indicator[indices]) / len(indices)
        )
    prompt_full, prompt_energy, prompt_event = map(
        np.asarray, (prompt_full, prompt_energy, prompt_event)
    )
    if not all(np.isfinite(v).all() for v in (prompt_full, prompt_energy, prompt_event)):
        raise FloatingPointError("Nonfinite empirical score moment; no clipping permitted")
    energy, full_energy = prompt_energy.mean(axis=0), prompt_full.mean(axis=0)
    ratios = np.sqrt(
        np.divide(energy, full_energy, out=np.full_like(energy, np.nan), where=full_energy > 0)
    )
    group_event = np.stack(
        [
            prompt_event[[i for i, p in enumerate(prompts.values()) if p["group"] == g]].mean(
                axis=0
            )
            for g in groups
        ],
        axis=1,
    )
    group_xv = np.stack((group_event[..., 0], -group_event[..., 3]), axis=-1)
    return {
        "status": "MEASURED_EMPIRICAL_NOT_CERTIFIED",
        "calibration_direction_ids": cal_ids,
        "query_direction_ids": query_ids,
        "prompt_ids": list(prompts),
        "groups": [list(g) for g in groups],
        "geometry_k": int(keep.sum()),
        "projection_coefficients": coefficients,
        "rho": rho,
        "empirical_fisher_residual_energy": energy,
        "empirical_fisher_full_energy": full_energy,
        "fisher_residual": np.sqrt(energy),
        "fisher_residual_ratio": ratios,
        "fisher_ratio_defined": full_energy > 0,
        "prompt_fisher_residual_energy": prompt_energy.T,
        "prompt_fisher_full_energy": prompt_full.T,
        "prompt_event_residual": prompt_event.transpose(1, 0, 2),
        "group_event_residual": group_event,
        "group_Xv_residual": group_xv,
        "score_residual": np.sqrt(np.mean(group_event**2, axis=(1, 2))),
        "score_Xv_residual": np.sqrt(np.mean(group_xv**2, axis=(1, 2))),
        "metadata": {
            "origin_id": provenance["origin_id"],
            "derivative_kind": provenance["derivative_kind"],
            "parameter_layout_hash": provenance["parameter_layout_hash"],
            "projection": "UNCENTERED_RAW_PARAMETER_CALIBRATION_SPAN",
            "rank_eigenvalue_threshold": threshold,
            "rank_threshold_scope": "GEOMETRY_DIAGNOSTIC_NOT_FULL_MODEL_TRUNCATION",
            "score_energy_definition": "equal_prompt_mean_of_iid_draw_mean_squared_score_direction",
            "score_residual_definition": (
                "RMS_of_equal_prompt_group_RAW4_omitted_direction_derivative"
            ),
            "energy_ratio_may_exceed_one": True,
            "zero_empirical_energy_proves_no_semantic_effect": False,
            "exact_fisher": False,
            "full_fisher_matrix_materialized": False,
            "finite_sample_energy_is_a_bound": False,
            "independent_source_seed_count": None,
            "actual_anchor_draw_count": len(sample_records),
            "actual_shared_collection_cost": copy.deepcopy(provenance["cost"]),
            "collection_cost_counted_per_query": False,
            "new_model_calls": 0,
            "new_backward_calls": 0,
            "query_reference_labels_read": False,
            "signature_substituted_for_derivative": False,
            "event_order": list(EVENTS),
            "primary_channels": ["delta_pX", "delta_v=-delta_pI"],
            "output_constraint_applied": False,
        },
    }
