"""Small exact witnesses for V3 calibration coverage and response models."""

import inspect

import numpy as np
import pytest

from src.modeling_v3.bank_selection import select_banks
from src.modeling_v3.coverage import (
    build_subspace,
    classify_coverage,
    coverage_diagnostics,
    nonidentifiability_witness,
)
from src.modeling_v3.response_models import error_decomposition, fit_response_model, group_xv_map


def test_uncentered_geometry_zero_query_and_high_leverage():
    fit = np.array([[1e-3, 0, 0], [1e-3, 0, 0]])
    q, meta = build_subspace(fit)
    assert meta["k"] == 1 and meta["centered"] is False
    assert np.allclose(q.T @ q, np.eye(1))
    report = coverage_diagnostics(fit, [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert np.allclose(report["rho"], [0, 1, 0])
    assert report["leverage"][0] > 1e5
    assert np.allclose(report["e_parallel"] + report["e_perp"], [[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    labels = classify_coverage(report, rho_threshold=0.05, leverage_threshold=10)
    assert labels["labels"] == ["HIGH_LEVERAGE", "OUT_OF_CALIBRATION_SPAN", "IDENTICAL_POLICY"]
    assert labels["status"] == ["UNKNOWN", "UNKNOWN", "IDENTIFIED"]


def test_small_rho_does_not_certify_without_validation_and_target_excitation():
    report = coverage_diagnostics(np.eye(2), np.eye(2))
    labels = classify_coverage(report, rho_threshold=0.05, leverage_threshold=10)
    assert labels["labels"] == ["MEASUREMENT_UNRESOLVED"] * 2
    labels = classify_coverage(
        report,
        rho_threshold=0.05,
        leverage_threshold=10,
        target_excitation=[False, True],
        measurement_resolved=True,
        validated_tolerance=True,
    )
    assert labels["labels"] == ["NO_CALIBRATION_EXCITATION", "PREDICTABLE_AT_VALIDATED_TOLERANCE"]


def test_nonidentifiability_witness_matches_fit_and_differs_at_query():
    fit = np.array([[1, 0, 0], [-1, 0, 0]])
    witness = nonidentifiability_witness(fit, [0, 2, 0], kappa=3)
    assert np.allclose(fit @ witness["J_minus"].T, fit @ witness["J_plus"].T)
    assert np.isclose(witness["minimax_error_lower_bound"], 6)
    assert np.isclose(np.linalg.norm(witness["query_gap"]), 12)


def test_block_selectors_are_geometry_only_and_retain_whole_bank():
    pool = np.array(
        [
            [[8.0, 0, 0], [9, 0, 0]],
            [[10, 0, 0], [11, 0, 0]],
            [[0, 1, 0], [0, 2, 0]],
            [[0, 0, 1], [0, 0, 2]],
            [[1, 1, 1], [2, 2, 2]],
        ]
    )
    selections = {}
    for method in ("FIRST", "STRATIFIED_RANDOM", "LARGEST_NORM", "BLOCK_PIVOT_QR", "BLOCK_LOGDET"):
        result = select_banks(pool, 2, method, strata=["a", "a", "b", "b", "c"], seed=14)
        selections[method] = result["selected_indices"]
        assert result["selected_updates"].shape == (2, 2, 3)
        assert np.array_equal(result["selected_updates"], pool[result["selected_indices"]])
        assert result["audit"]["heldout_responses_read"] is False
    assert selections["FIRST"] != selections["LARGEST_NORM"]
    assert len({tuple(v) for v in selections.values()}) >= 3
    assert not any(
        "response" in name or "query" in name for name in inspect.signature(select_banks).parameters
    )


@pytest.mark.parametrize("method", ["PCA", "RANDOM_Q", "RESPONSE_SVD", "GROUP_WEIGHTED_RESPONSE"])
def test_full_rank_rotation_equivalence_and_genuine_truncation(method):
    rng = np.random.default_rng(40)
    fit = rng.normal(size=(9, 5))
    response = rng.normal(size=(9, 2, 4))
    query = rng.normal(size=(3, 5))
    full = fit_response_model(fit, response, "FULL_RIDGE", alpha=0.01)
    model = fit_response_model(
        fit, response, method, "FULL", alpha=0.01, group_map=rng.normal(size=(3, 8))
    )
    assert model["r"] == model["k"] == 5
    assert np.allclose(full["predict"](query), model["predict"](query), atol=1e-11)
    low = fit_response_model(
        fit, response, method, 2, alpha=0.01, group_map=rng.normal(size=(3, 8))
    )
    assert low["r"] == 2 < low["k"]
    assert not np.allclose(full["predict"](query), low["predict"](query))


def test_raw4_not_silently_projected_and_explicit_constraint_audited():
    fit = np.eye(3)
    response = np.array([[1.0, 2, 3, 4], [2, 3, 4, 5], [1, -1, 0, 0]])
    raw = fit_response_model(fit, response, "FULL_RIDGE", alpha=0)
    constrained = fit_response_model(
        fit, response, "FULL_RIDGE", alpha=0, output_policy="EXPLICIT_ZERO_SUM"
    )
    assert np.allclose(raw["predict"](fit), response)
    assert np.allclose(constrained["predict"](fit).sum(axis=-1), 0)
    assert np.allclose(constrained["predict_raw"](fit), response)
    assert np.array_equal(constrained["raw_observation_mass"], response.sum(axis=-1))


def test_group_primary_v_uses_negative_i_and_preserves_raw_mass_difference():
    raw = np.array([[1.0, 2, 3, -2], [2, 1, -3, -1]])
    mapping = group_xv_map(["one", "one"], prompt_weights=[1, 3])
    primary = mapping @ raw.reshape(-1)
    weights = np.array([0.25, 0.75])
    assert np.allclose(primary, [weights @ raw[:, 0], weights @ -raw[:, 3]])
    raw_valid_sum = weights @ raw[:, :3].sum(axis=-1)
    mass = weights @ raw.sum(axis=-1)
    assert raw_valid_sum - primary[1] == pytest.approx(mass)
    assert raw_valid_sum != primary[1]


def test_joint_gls_matches_full_kronecker_objective_and_rotation():
    rng = np.random.default_rng(91)
    fit = rng.normal(size=(5, 2))
    y = rng.normal(size=(5, 4))
    z = rng.normal(size=(20, 20))
    covariance = z @ z.T + np.eye(20)
    model = fit_response_model(fit, y, "FULL_GLS", alpha=0.02, covariance=covariance[None])
    a = fit @ model["directions"]
    design = np.kron(a, np.eye(4))
    precision_x = np.linalg.solve(covariance, design)
    lam = model["metadata"]["fits"][0]["ridge_lambda"]
    beta = np.linalg.solve(
        design.T @ precision_x + lam * np.eye(8),
        design.T @ np.linalg.solve(covariance, y.reshape(-1)),
    )
    assert np.allclose(model["predict"](fit), a @ beta.reshape(2, 4), atol=1e-10)
    rotated = fit_response_model(
        fit,
        y,
        "RANDOM_Q",
        "FULL",
        alpha=0.02,
        covariance=covariance[None],
        regression="GLS",
        random_seed=71,
    )
    assert np.allclose(model["predict"](fit), rotated["predict"](fit), atol=1e-10)


def test_constrained_gls_fits_raw_likelihood_and_keeps_unconstrained_result():
    rng = np.random.default_rng(841)
    fit = rng.normal(size=(4, 2))
    y = rng.normal(size=(4, 4))
    factor = rng.normal(size=(16, 16))
    covariance = factor @ factor.T + np.eye(16)
    raw = fit_response_model(fit, y, "FULL_GLS", alpha=0.0, covariance=covariance[None])
    constrained = fit_response_model(
        fit,
        y,
        "FULL_GLS",
        alpha=0.0,
        covariance=covariance[None],
        output_policy="EXPLICIT_ZERO_SUM",
    )
    h = np.array([[1, 1, 1], [-1, 1, 1], [0, -2, 1], [0, 0, -3]], dtype=float)
    h /= np.sqrt([2, 6, 12])
    a = fit @ constrained["directions"]
    design = np.kron(a, h)
    beta = np.linalg.solve(
        design.T @ np.linalg.solve(covariance, design),
        design.T @ np.linalg.solve(covariance, y.reshape(-1)),
    )
    prediction = constrained["predict"](fit)
    assert np.allclose(prediction, a @ beta.reshape(2, 3) @ h.T)
    assert np.allclose(prediction.sum(axis=-1), 0)
    assert np.allclose(constrained["predict_raw"](fit), raw["predict"](fit))
    assert not np.allclose(prediction, raw["predict"](fit) @ h @ h.T)


def test_covariance_degeneracy_is_explicit_and_indefinite_covariance_rejected():
    fit = np.array([[1.0], [2.0]])
    y = np.zeros((2, 4))
    model = fit_response_model(fit, y, "FULL_GLS", covariance=np.zeros((1, 8, 8)))
    report = model["metadata"]["fits"][0]["covariance"]
    assert report["status"] == "EMPIRICAL_DEGENERATE"
    assert report["floor_is_measurement_information"] is False
    with pytest.raises(ValueError, match="positive semidefinite"):
        fit_response_model(fit, y, "FULL_GLS", covariance=-np.eye(8)[None])


def test_same_norm_opposite_direction_is_not_same_response():
    fit = np.array([[1.0, 0], [-1.0, 0]])
    response = np.array([[1.0, -1, 0, 0], [-1.0, 1, 0, 0]])
    model = fit_response_model(fit, response, "FULL_RIDGE", alpha=0)
    assert np.linalg.norm(fit[0]) == np.linalg.norm(fit[1])
    assert np.allclose(model["predict"](fit), response)
    assert not np.array_equal(response[0], response[1])


def test_selectors_zero_geometry_and_seed_reproducibility():
    pool = np.zeros((4, 2, 3))
    for method in ("FIRST", "STRATIFIED_RANDOM", "LARGEST_NORM", "BLOCK_PIVOT_QR", "BLOCK_LOGDET"):
        first = select_banks(pool, 3, method, seed=21)
        again = select_banks(pool, 3, method, seed=21)
        assert first["selected_indices"] == again["selected_indices"]
        assert first["geometry"]["k"] == 0
        assert len(set(first["selected_indices"])) == 3
        assert select_banks(pool, 0, method)["selected_updates"].shape == (0, 2, 3)
    assert np.all(pool == 0)


def test_fit_copies_caller_data_and_rejects_bad_shapes():
    x = np.eye(2)
    y = np.array([[1.0, 0, 0, -1], [0, 1, 0, -1]])
    model = fit_response_model(x, y, "FULL_RIDGE", alpha=0)
    prediction = model["predict"](np.eye(2)).copy()
    x[:] = 0
    y[:] = 0
    assert np.array_equal(model["predict"](np.eye(2)), prediction)
    with pytest.raises(ValueError):
        fit_response_model(np.eye(2), np.ones((3, 4)), "FULL_RIDGE")
    with pytest.raises(ValueError):
        fit_response_model(np.eye(2), np.ones((2, 3)), "FULL_RIDGE")


def test_zero_rank_models_unknown_while_zero_baseline_remains_scored():
    fit = np.zeros((3, 5))
    y = np.ones((3, 4))
    model = fit_response_model(fit, y, "FULL_RIDGE")
    baseline = fit_response_model(fit, y, "ZERO")
    assert model["status"] == "NO_CALIBRATION_EXCITATION"
    assert np.isnan(model["predict"](np.ones((2, 5)))).all()
    assert np.all(baseline["predict"](np.ones((2, 5))) == 0)
    assert baseline["status"] == "STATISTICAL_BASELINE_ONLY"


def test_rbf_is_fitted_nonlinear_diagnostic_with_zero_origin():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.column_stack((x[:, 0] ** 2, -(x[:, 0] ** 2), np.zeros((4, 2))))
    model = fit_response_model(x, y, "RBF_RIDGE", alpha=1e-8)
    linear = fit_response_model(x, y, "FULL_RIDGE", alpha=1e-8)
    assert np.linalg.norm(model["predict"](x) - y) < np.linalg.norm(linear["predict"](x) - y)
    assert np.allclose(model["predict"]([[0]]), 0)
    assert model["metadata"]["secondary_only"] is True


def test_error_vectors_and_cross_terms_reconstruct_total_squared_error():
    rng = np.random.default_rng(63)
    values = [rng.normal(size=(3, 4)) for _ in range(5)]
    report = error_decomposition(*values)
    assert np.allclose(sum(report["components"].values()), values[0] - values[4])
    assert np.isclose(
        sum(report["squared_norms"].values()) + sum(report["cross_terms"].values()),
        np.square(values[0] - values[4]).sum(),
    )
