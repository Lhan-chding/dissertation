import copy
import runpy
from pathlib import Path

import numpy as np
import pytest

from src.modeling_v4.empirical_diagnostics import (
    diagnostics_from_collection,
    empirical_score_diagnostics,
    unavailable_empirical_diagnostics,
)


def inputs():
    # Unequal prompt draw counts must not change equal-prompt weighting.
    scores = np.array(
        [
            [1.0, 2.0, 0.0],
            [-1.0, -2.0, 0.0],
            [0.0, 1.0, 3.0],
            [0.0, -1.0, -3.0],
            [0.0, 1.0, 3.0],
            [0.0, -1.0, -3.0],
        ]
    )
    fit = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    query = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    records = [
        dict(
            sample_id=f"s{i}",
            prompt_id="p0" if i < 2 else "p1",
            group=["family", "interface"],
            category="X" if i % 2 == 0 else "I",
            rng_namespace="anchor",
            proposal="ORIGIN",
            proposal_policy_fingerprint="policy",
        )
        for i in range(len(scores))
    ]
    provenance = dict(
        origin_id="origin",
        derivative_kind="AUTOGRAD_FULL_PARAMETER_SCORE",
        expansion_point="ORIGIN",
        proposal_policy_fingerprint="policy",
        expansion_policy_fingerprint="policy",
        parameter_layout_hash="a" * 64,
        calibration_direction_ids=["c0", "c1"],
        query_direction_ids=["q0", "q1", "alias"],
        reference_sample_ids=["r0"],
        reference_rng_stream_ids=["reference"],
        query_reference_labels_read=False,
        cost=dict(backward_calls=6, gradient_score_calls=6, scored_sequences=12, scored_tokens=24),
        cost_scope="SHARED_ORIGIN_GRADIENT_COLLECTION",
    )
    args = (fit @ fit.T, query @ fit.T, np.sum(query**2, axis=1), scores @ fit.T, scores @ query.T)
    return args, records, provenance, scores, fit, query


def test_empirical_energy_matches_explicit_small_fisher_and_keeps_signed_semantics():
    args, records, provenance, scores, fit, query = inputs()
    before = [a.copy() for a in args]
    result = empirical_score_diagnostics(*args, sample_records=records, provenance=provenance)
    residual = query - query @ np.linalg.pinv(fit) @ fit
    fisher = (scores[:2].T @ scores[:2] / 2 + scores[2:].T @ scores[2:] / 4) / 2
    expected = np.einsum("qd,de,qe->q", residual, fisher, residual)
    np.testing.assert_allclose(result["empirical_fisher_residual_energy"], expected)
    np.testing.assert_allclose(result["fisher_residual"], np.sqrt(expected))
    np.testing.assert_allclose(result["rho"], [1 / np.sqrt(2), 1, 0])
    # Signed I derivative is retained; reporting v reverses I explicitly.
    np.testing.assert_allclose(result["group_event_residual"][0, 0], [0.75, 0, 0, -0.75])
    np.testing.assert_allclose(result["group_Xv_residual"][0, 0], [0.75, 0.75])
    assert result["geometry_k"] == 1
    assert result["metadata"]["full_fisher_matrix_materialized"] is False
    assert result["metadata"]["exact_fisher"] is False
    assert result["metadata"]["finite_sample_energy_is_a_bound"] is False
    assert result["metadata"]["independent_source_seed_count"] is None
    assert result["metadata"]["actual_shared_collection_cost"] == provenance["cost"]
    for original, value in zip(before, args, strict=True):
        np.testing.assert_array_equal(original, value)


def test_euclidean_projection_can_increase_empirical_score_energy_no_ratio_clipping():
    args, records, provenance, *_ = inputs()
    cal = np.array([[1.0, 2.0]] * 6)
    query = np.zeros((6, 3))
    query[:, 0] = 0.1
    result = empirical_score_diagnostics(
        *args[:3], cal, query, sample_records=records, provenance=provenance
    )
    assert result["fisher_residual_ratio"][0] == pytest.approx(9)
    assert np.isnan(result["fisher_residual_ratio"][1])
    assert result["empirical_fisher_residual_energy"][2] == 0


@pytest.mark.parametrize(
    "change", ["signature", "reference", "proposal", "duplicate", "cost", "constant_prompt_group"]
)
def test_identity_and_information_guards(change):
    args, records, provenance, *_ = inputs()
    records, provenance = copy.deepcopy(records), copy.deepcopy(provenance)
    if change == "signature":
        provenance["derivative_kind"] = "FINITE_SEQUENCE_LOG_RATIO_SIGNATURE"
    elif change == "reference":
        provenance["reference_rng_stream_ids"] = ["anchor"]
    elif change == "proposal":
        provenance["proposal_policy_fingerprint"] = "another"
    elif change == "duplicate":
        records[1]["sample_id"] = records[0]["sample_id"]
    elif change == "cost":
        provenance["cost"]["backward_calls"] = 0
    else:
        records[1]["group"] = ["another", "interface"]
    with pytest.raises(ValueError):
        empirical_score_diagnostics(*args, sample_records=records, provenance=provenance)


def test_no_derivatives_remains_explicitly_unavailable_and_does_not_invent_zero():
    result = unavailable_empirical_diagnostics("Actual per-draw gradients not collected", ["q0"])
    assert result["status"] == "NOT_AVAILABLE"
    assert result["fisher_residual"] is None
    assert result["score_residual"] is None
    assert result["metadata"]["exact_fisher"] is False


def collection_inputs():
    args, records, provenance, *_ = inputs()
    # The producer may interleave actual direction columns; use explicit IDs.
    all_products = np.concatenate((args[3], args[4]), axis=1)
    order = [3, 0, 4, 1, 2]
    ids = provenance["calibration_direction_ids"] + provenance["query_direction_ids"]
    collection = dict(
        expansion_point="ORIGIN",
        derivative_kind=provenance["derivative_kind"],
        reference_labels_read=False,
        finite_difference_is_derivative=False,
        raw_score_products_importance_weighted=False,
        expansion_policy_identity={"inference_fingerprint": "policy"},
        parameter_layout_hash="a" * 64,
        direction_ids=[ids[i] for i in order],
        raw_directional_score_products=all_products[:, order],
        sample_records=records,
        sample_ids=[r["sample_id"] for r in records],
        rng_stream_ids=["anchor"],
        cost=provenance["cost"],
    )
    kwargs = dict(
        origin_id="origin",
        calibration_direction_ids=provenance["calibration_direction_ids"],
        query_direction_ids=provenance["query_direction_ids"],
        reference_sample_ids=["r0"],
        reference_rng_stream_ids=["reference"],
    )
    return args, collection, kwargs


def test_actual_collection_adapter_keeps_direction_and_sample_identities():
    args, collection, kwargs = collection_inputs()
    result = diagnostics_from_collection(collection, *args[:3], **kwargs)
    direct = empirical_score_diagnostics(*args, sample_records=inputs()[1], provenance=inputs()[2])
    np.testing.assert_allclose(result["fisher_residual"], direct["fisher_residual"])
    np.testing.assert_allclose(result["group_event_residual"], direct["group_event_residual"])


@pytest.mark.parametrize(
    "change", ["weighted", "sample_order", "missing_direction", "wrong_policy", "reference_read"]
)
def test_collection_adapter_rejects_proxies_and_identity_drift(change):
    args, collection, kwargs = collection_inputs()
    if change == "weighted":
        collection["raw_score_products_importance_weighted"] = True
    elif change == "sample_order":
        collection["sample_ids"] = list(reversed(collection["sample_ids"]))
    elif change == "missing_direction":
        collection["direction_ids"][0] = "unrelated"
    elif change == "wrong_policy":
        collection["sample_records"][0]["proposal_policy_fingerprint"] = "other"
    else:
        collection["reference_labels_read"] = True
    with pytest.raises(ValueError):
        diagnostics_from_collection(collection, *args[:3], **kwargs)


def test_unexcited_calibration_retains_full_empirical_query_and_nonzero_raw_mass():
    args, records, provenance, *_ = inputs()
    gram, cross, cal = np.zeros((2, 2)), np.zeros((3, 2)), np.zeros((6, 2))
    query = args[4].copy()
    query[:, 0] = 1
    result = empirical_score_diagnostics(
        gram, cross, args[2], cal, query, sample_records=records, provenance=provenance
    )
    assert result["geometry_k"] == 0
    np.testing.assert_array_equal(result["rho"], [1, 1, 0])
    np.testing.assert_array_equal(result["group_event_residual"][0, 0], [0.5, 0, 0, 0.5])
    assert result["group_event_residual"][0, 0].sum() == 1


@pytest.mark.parametrize("change", ["cross", "cal_score", "zero_query_score"])
def test_impossible_zero_parameter_geometry_is_rejected(change):
    args, records, provenance, *_ = inputs()
    gram, cross, cal, query = np.zeros((2, 2)), np.zeros((3, 2)), np.zeros((6, 2)), args[4].copy()
    if change == "cross":
        cross[0, 0] = 1
    elif change == "cal_score":
        cal[0, 0] = 1
    else:
        query[0, 2] = 1
    with pytest.raises(ValueError):
        empirical_score_diagnostics(
            gram, cross, args[2], cal, query, sample_records=records, provenance=provenance
        )


def test_real_small_autograd_collector_to_empirical_adapter_without_extra_model_calls():
    from src.modeling_v4.score_response import collect_score_response

    fixture = runpy.run_path(str(Path(__file__).with_name("test_score_response.py")))
    runtime = fixture["runtime"]()
    runtime["active_policy_identity"] = {"inference_fingerprint": "tiny-fixture-policy"}
    draws = [fixture["sample"](runtime, f"anchor-{i}") for i in range(2)]
    for draw in draws:
        draw["proposal_fingerprint"] = "tiny-fixture-policy"
    directions = {
        "cal": {"lora_A": np.array([1.0, 0.0]), "lora_B": np.zeros(2)},
        "query": {"lora_A": np.array([1.0, 1.0]), "lora_B": np.zeros(2)},
    }
    collection = collect_score_response(
        runtime,
        [fixture["prompt"]()],
        draws,
        expected_groups=[("f0", "SYMBOLIC_FRESH")],
        directions=directions,
    )
    calls = runtime["adapter"].forward_calls
    result = diagnostics_from_collection(
        collection,
        np.ones((1, 1)),
        np.ones((1, 1)),
        np.array([2.0]),
        origin_id="tiny-origin",
        calibration_direction_ids=["cal"],
        query_direction_ids=["query"],
        reference_sample_ids=[],
        reference_rng_stream_ids=["independent-reference"],
    )
    raw = collection["raw_directional_score_products"]
    np.testing.assert_allclose(
        result["empirical_fisher_residual_energy"], np.mean((raw[:, 1] - raw[:, 0]) ** 2)
    )
    assert runtime["adapter"].forward_calls == calls
    assert result["metadata"]["actual_shared_collection_cost"]["backward_calls"] == 2
