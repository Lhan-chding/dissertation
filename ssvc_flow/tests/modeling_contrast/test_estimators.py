"""Regression contracts for fit-only contrast direction and coefficient estimation."""

import numpy as np
import pytest

from src.modeling_contrast.estimators import (
    build_subspace,
    effective_rank,
    fit_contrast_model,
    gls_ridge_map,
    group_xv_map,
    out_of_subspace,
    prepare_contrast,
    ridge_map,
    subspace_distance,
)


def example(seed=9):
    rng = np.random.default_rng(seed)
    return rng, rng.normal(size=(7, 9)), rng.normal(size=(7, 6))


def test_fit_only_uncentered_subspace_and_opposite_updates():
    updates = np.array([[1.0, 0, 0], [-1, 0, 0], [2, 0, 0]])
    q, meta = build_subspace(updates)
    assert meta["k"] == 1
    assert meta["centered"] is False
    assert np.allclose(q @ q.T, np.diag([1.0, 0, 0]))
    report = out_of_subspace(np.array([[0.0, 1, 0], [0, 0, 0]]), q)
    assert report["per_update_fraction"] == [1.0, 0.0]
    assert report["zero_update_count"] == 1


@pytest.mark.parametrize("scale", [1e-5, 1.0, 1e5])
def test_spherical_gls_is_scaled_ordinary_ridge(scale):
    _rng, a, y = example()
    a = a[:, :4]
    covariance = np.broadcast_to(np.eye(21) * scale, (2, 21, 21)).copy()
    ordinary = ridge_map(a, y, 0.01)
    generalized, meta = gls_ridge_map(a, y, covariance, 0.01, eta=0.1)
    assert np.allclose(ordinary, generalized, rtol=2e-10, atol=2e-10)
    assert meta["solver"] == "WHITENED_SVD"
    assert meta["vectorization"] == "ROW_MAJOR_CONTRAST_THEN_HELMERT"


def test_joint_gls_matches_row_major_kronecker_objective():
    rng = np.random.default_rng(20260915)
    a = rng.normal(size=(4, 2))
    y = rng.normal(size=(4, 3))
    factor = rng.normal(size=(12, 12))
    covariance = factor @ factor.T + np.eye(12)
    alpha = 0.01
    actual, meta = gls_ridge_map(a, y, covariance[None], alpha, eta=0.01)
    covariance = 0.99 * covariance + 0.01 * np.diag(np.diag(covariance))
    x = np.kron(a, np.eye(3))
    eig, basis = np.linalg.eigh(covariance)
    whitened = (basis / np.sqrt(eig)).T @ x
    lam = alpha * np.linalg.svd(whitened, compute_uv=False)[0] ** 2
    beta = np.linalg.solve(
        x.T @ np.linalg.solve(covariance, x) + lam * np.eye(6),
        x.T @ np.linalg.solve(covariance, y.reshape(-1)),
    )
    assert np.allclose(actual, beta.reshape(2, 3).T, atol=1e-11)
    assert np.isclose(meta["prompt_fits"][0]["ridge_lambda"], lam)


def test_nonzero_empirical_floor_does_not_create_information():
    a = np.array([[1.0, 0], [0, 1], [1, 1]])
    y = np.zeros((3, 3))
    fitted, meta = gls_ridge_map(a, y, np.zeros((1, 9, 9)), 0.01, observation_n=16)
    assert np.all(fitted == 0)
    assert meta["prompt_fits"][0]["covariance"]["status"] == "EMPIRICAL_DEGENERATE"
    assert meta["prompt_fits"][0]["covariance"]["spectral_floor"] > 0


@pytest.mark.parametrize("method", ["C2", "C3", "C4_PCA", "C4_RANDOM", "C5", "C6"])
def test_all_full_models_match_and_accept_prompt_axes(method):
    rng, updates, y = example()
    kwargs = {"group_map": rng.normal(size=(4, 6))}
    fitted = fit_contrast_model(updates, y.reshape(7, 2, 3), method, "FULL", 0.01, **kwargs)
    expected = fit_contrast_model(updates, y, "C2", "FULL", 0.01)
    assert fitted["r"] == fitted["k"] == 7
    assert np.allclose(fitted["predict"](updates), expected["predict"](updates), atol=1e-9)
    assert fitted["predict"](updates).shape == (7, 6)
    assert np.all(fitted["predict"](np.zeros((2, 9))) == 0)


@pytest.mark.parametrize("method", ["C4_PCA", "C4_RANDOM", "C5", "C6"])
def test_zero_response_retains_nonzero_requested_direction(method):
    updates = np.eye(5)
    y = np.zeros((5, 3))
    fitted = fit_contrast_model(updates, y, method, 4, 0.01, group_map=np.ones((2, 3)))
    assert fitted["r"] == 4
    assert fitted["status"] == "IDENTIFIED"
    assert np.all(fitted["predict"](updates) == 0)


def test_unexcited_fit_is_explicit_and_never_uses_test_updates():
    fitted = fit_contrast_model(np.zeros((3, 4)), np.ones((3, 3)), "C5", 2, 0.01)
    assert fitted["status"] == "NO_CONTRAST_EXCITATION"
    assert fitted["r"] == fitted["k"] == 0
    assert np.all(fitted["predict"](np.ones((2, 4))) == 0)
    assert fitted["rank_cap"] == 2


def test_c0_only_allows_zero_rank_and_nonzero_models_reject_it():
    _rng, updates, y = example()
    c0 = fit_contrast_model(updates, y, "C0", 0, 0.01)
    assert c0["r"] == 0
    assert np.all(c0["predict"](updates) == 0)
    for rank in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            fit_contrast_model(updates, y, "C5", rank, 0.01)
    assert effective_rank(4, 2) == 2
    assert effective_rank("FULL", 0) == 0


def test_group_aware_spectrum_preserves_fixed_target_block_weights():
    updates = np.eye(2)
    y = np.zeros((2, 6))
    y[0, 0] = 2.0
    y[1, 1] = 1.0
    group_map = np.zeros((2, 6))
    group_map[0, 1] = 10.0
    c5 = fit_contrast_model(updates, y, "C5", 1, 0.0)
    c6 = fit_contrast_model(updates, y, "C6", 1, 0.0, group_map=group_map)
    assert np.allclose(c5["U"] @ c5["U"].T, np.diag([1.0, 0.0]))
    assert np.allclose(c6["U"] @ c6["U"].T, np.diag([0.0, 1.0]))
    assert c6["spectrum_block_weights"] == {"full": 1 / np.sqrt(6), "group_XV": 1 / np.sqrt(2)}


def test_direction_and_coefficient_oracles_are_separate_and_labeled():
    updates = np.eye(2)
    exact = np.array([[2.0, 0, 0], [0, 1, 0]])
    finite = np.array([[0.0, 0, 0], [0, 3, 0]])
    exact_u = fit_contrast_model(updates, finite, "C5", 1, 0.0, direction_responses=exact)
    exact_coef = fit_contrast_model(updates, finite, "C5", 1, 0.0, coefficient_responses=exact)
    assert exact_u["diagnostic"] == "EXACT_U_FINITE_COEFFICIENTS"
    assert exact_coef["diagnostic"] == "FINITE_U_EXACT_COEFFICIENTS"
    assert np.allclose(exact_u["U"] @ exact_u["U"].T, np.diag([1.0, 0.0]))
    assert np.allclose(exact_coef["U"] @ exact_coef["U"].T, np.diag([0.0, 1.0]))
    assert np.all(exact_u["predict"](updates) == 0)
    assert np.allclose(exact_coef["predict"](updates), [[0, 0, 0], [0, 1, 0]])


def test_projector_diagnostics_ignore_sign_and_basis_rotations():
    u = np.eye(4)[:, :2]
    same = u @ np.array([[0.0, -1], [1, 0]])
    report = subspace_distance(u, same)
    assert report["projector_frobenius_distance"] < 1e-12
    assert np.allclose(report["principal_angles_radians"], 0, atol=1e-12)
    other = np.eye(4)[:, 2:]
    assert np.isclose(subspace_distance(u, other)["projector_frobenius_distance"], 2)


def test_group_xv_map_matches_event_differences_with_fixed_weights():
    from src.modeling_qualification.math_contracts import helmert

    groups = np.array([0, 0, 1])
    weights = np.array([1.0, 3.0, 2.0])
    mapping = group_xv_map(groups, weights)
    event_diff = np.array([[0.1, -0.1, 0, 0], [0.2, -0.1, -0.05, -0.05], [0.0, 0.1, 0, -0.1]])
    contrasts = event_diff @ helmert()
    expected = np.array([0.175, 0.0375, 0.0, 0.1])
    assert mapping.shape == (4, 9)
    assert np.allclose(mapping @ contrasts.reshape(-1), expected)


def test_boundary_validation_and_input_arrays_are_not_modified():
    _rng, updates, y = example()
    updates_before, y_before = updates.copy(), y.copy()
    fitted = fit_contrast_model(updates, y, "C4_RANDOM", 2, 0.01)
    assert np.array_equal(updates, updates_before)
    assert np.array_equal(y, y_before)
    with pytest.raises(ValueError):
        fitted["predict"](np.ones((2, 8)))
    with pytest.raises(ValueError):
        fit_contrast_model(updates, np.ones((7, 5)), "C2")
    with pytest.raises(ValueError):
        fit_contrast_model(updates, y, "C6", 1)
    with pytest.raises(ValueError):
        gls_ridge_map(updates, y, np.eye(21)[None], 0.01)


def test_explicit_alias_row_selection_precedes_q_and_covariance_shrinkage():
    rng = np.random.default_rng(77)
    e = rng.normal(size=(4, 5))
    y = rng.normal(size=(4, 6))
    f = rng.normal(size=(2, 12, 12))
    cov = f @ f.transpose(0, 2, 1) + np.eye(12)
    rows = [0, 2]
    coordinates = [0, 1, 2, 6, 7, 8]
    prepared = prepare_contrast(e, y, covariance=cov, active_rows=rows)
    actual = fit_contrast_model(e, y, "C3", prepared=prepared)
    expected = fit_contrast_model(
        e[rows], y[rows], "C3", covariance=cov[:, coordinates][:, :, coordinates]
    )
    assert actual["retained_fit_rows"] == rows
    assert actual["excluded_fit_rows"] == [1, 3]
    assert actual["k"] == 2
    assert np.allclose(actual["predict"](e), expected["predict"](e))
    with pytest.raises(ValueError):
        fit_contrast_model(e, y, "C3", prepared=prepared, active_rows=[0, 1])


def test_explicit_all_alias_rows_remain_unexcited():
    e = np.ones((2, 4))
    y = np.zeros((2, 3))
    model = fit_contrast_model(e, y, "C3", covariance=np.zeros((1, 6, 6)), active_rows=[])
    assert model["status"] == "NO_CONTRAST_EXCITATION"
    assert model["r"] == 0
    assert model["retained_fit_rows"] == []


def test_same_updates_are_not_automatically_inferred_to_be_policy_aliases():
    e = np.array([[1.0, 0], [1.0, 0]])
    y = np.array([[1.0, 0, 0], [2.0, 0, 0]])
    fitted = fit_contrast_model(e, y, "C2", alpha=0.0)
    assert fitted["retained_fit_rows"] == [0, 1]
    assert np.allclose(fitted["predict"](e), [[1.5, 0, 0], [1.5, 0, 0]])


def test_cached_full_gls_rotated_models_match_same_anisotropic_objective():
    rng, e, y = example()
    f = rng.normal(size=(2, 21, 21))
    cov = f @ f.transpose(0, 2, 1) + np.eye(21)
    prepared = prepare_contrast(e, y, covariance=cov, eta=0.1)
    direct = fit_contrast_model(e, y, "C3", alpha=0.01, eta=0.1, prepared=prepared)
    for method in ["C5", "C6"]:
        actual = fit_contrast_model(
            e,
            y,
            method,
            "FULL",
            0.01,
            eta=0.1,
            prepared=prepared,
            group_map=rng.normal(size=(4, 6)),
        )
        assert np.allclose(actual["predict"](e), direct["predict"](e), atol=1e-10)
    with pytest.raises(ValueError):
        fit_contrast_model(e, y + 1, "C3", prepared=prepared, eta=0.1)
