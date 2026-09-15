"""Executable M0 contracts; written before the implementation."""

import json

import numpy as np
import pytest
from scipy.stats import binom

from src.modeling_qualification.math_contracts import (
    CATEGORIES,
    advantages,
    clopper_pearson,
    conditional_intervals,
    error_bounds,
    event_gradient_affine,
    from_helmert,
    from_nested,
    geometry,
    helmert,
    mle,
    nested_fisher,
    nested_jacobian,
    orthogonal_span,
    prediction_metrics,
    probability,
    project_simplex,
    run_math,
    to_helmert,
    to_nested,
    truncate_response,
    weighted_group_q,
)


def test_helmert_orthogonality_roundtrip_and_distance():
    h = helmert()
    assert h.dtype == np.float64
    np.testing.assert_allclose(h.T @ h, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(h.T @ np.ones(4), 0, atol=1e-15)
    p = np.vstack([np.eye(4), np.random.default_rng(12).dirichlet(np.ones(4), 20)])
    np.testing.assert_allclose(from_helmert(to_helmert(p)), p, atol=2e-16)
    np.testing.assert_allclose(
        np.linalg.norm(p - p[0], axis=1),
        np.linalg.norm(to_helmert(p) - to_helmert(p[0]), axis=1),
        atol=1e-15,
    )


def test_category_and_weight_identity_errors_fail():
    assert CATEGORIES == ("X", "S", "W", "I")
    with pytest.raises(ValueError, match="order"):
        probability([0.25] * 4, categories=("S", "X", "W", "I"))
    with pytest.raises(ValueError, match="order"):
        weighted_group_q(
            [[0.2, 0.1, 0.1, 0.6], [0.2, 0.3, 0.3, 0.2]], [1, 2], ["a", "b"], ["b", "a"]
        )


@pytest.mark.parametrize(
    "p",
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0.3, 0.6, 0.1],
        [0.3, 0, 0, 0.7],
        [0.21, 0.13, 0.26, 0.4],
    ],
)
def test_nested_boundaries_roundtrip(p):
    item = to_nested(p)
    np.testing.assert_allclose(from_nested(item["v"], item["q"], item["s"]), p, atol=1e-15)
    if p == [0, 0, 0, 1]:
        assert item["q"] is None and item["q_status"] == "UNDEFINED_ZERO_MASS"
    if sum(p[1:3]) == 0:
        assert item["s"] is None


def test_mle_does_not_infer_population_zero_from_counts():
    for counts in ([3, 2, 4, 7], [0, 0, 0, 16], [16, 0, 0, 0], [0, 2, 5, 9]):
        item = mle(counts)
        np.testing.assert_allclose(from_nested(item["v"], item["q"], item["s"]), item["p"])
    item = mle([0, 0, 0, 16])
    assert item["q"] is None and item["q_status"] == "UNOBSERVED_CONDITION"
    assert item["s_status"] == "UNOBSERVED_CONDITION"
    assert item["category_status"]["X"] == "UNOBSERVED_EVENT"
    for bad in ([0, 0, 0, 0], [True, 1, 1, 1], [1.5, 1, 1, 1], [-1, 1, 1, 1]):
        with pytest.raises(ValueError):
            mle(bad)


def test_fisher_and_finite_difference_fp64():
    for x in ([0.7, 0.4, 0.2], [0.08, 0.9, 0.1], [0.91, 0.85, 0.95]):
        x = np.asarray(x, np.float64)
        p = from_nested(*x)
        jac = nested_jacobian(*x)
        np.testing.assert_allclose(
            jac.T @ np.diag(1 / p) @ jac, nested_fisher(*x), atol=1e-9, rtol=0
        )
        h = 1e-6
        fd = np.column_stack(
            [
                (from_nested(*(x + step)) - from_nested(*(x - step))) / (2 * h)
                for step in h * np.eye(3)
            ]
        )
        assert np.max(np.abs(fd - jac)) < 1e-9
    for x in ([0, 0.5, 0.5], [1, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 1]):
        with pytest.raises(ValueError):
            nested_fisher(*x)


def test_clopper_pearson_actual_denominators_and_zero():
    unknown = clopper_pearson(0, 0)
    assert unknown["estimate"] is None and unknown["interval"] == [0.0, 1.0]
    item = conditional_intervals([2, 3, 5, 6])
    assert item["q"]["n"] == 10 and item["s"]["n"] == 8
    assert item["q"]["estimate"] == 0.2 and item["s"]["estimate"] == 3 / 8
    assert conditional_intervals([0, 0, 0, 16])["q"]["status"] == "UNOBSERVED_CONDITION"
    assert clopper_pearson(0, 16)["interval"][1] > 0
    panel = conditional_intervals([2, 3, 5, 6], simultaneous=True)
    assert panel["coverage_scope"] == "BONFERRONI_WITHIN_ONE_COUNT_VECTOR"
    assert panel["q"]["alpha"] == pytest.approx(0.05 / 7)
    # Exact finite-binomial coverage, including endpoints; no Monte Carlo claim.
    for n in (1, 8, 32):
        bounds = [clopper_pearson(k, n)["interval"] for k in range(n + 1)]
        for p in np.linspace(0, 1, 41):
            coverage = sum(
                binom.pmf(k, n, p)
                for k, (lo, hi) in enumerate(bounds)
                if lo - 1e-14 <= p <= hi + 1e-14
            )
            assert coverage >= 0.95 - 1e-12


def test_weighted_q_is_ratio_of_weighted_probabilities():
    result = weighted_group_q(
        [[0.2, 0.1, 0.1, 0.6], [0.2, 0.3, 0.3, 0.2]], [1.0, 2.0], ["a", "b"], ["a", "b"]
    )
    assert result == pytest.approx(0.6 / 2.0)
    assert result != pytest.approx((0.5 + 2 * 0.25) / 3)
    assert weighted_group_q([[0, 0, 0, 1]], [1.0], ["a"], ["a"]) is None


def test_advantage_shift_saturation_and_epsilon_contract():
    cats = ["X", "X", "S", "W", "W", "X", "W", "S"]
    np.testing.assert_allclose(advantages(cats, 0), advantages(cats, 1), atol=1e-12)
    cats = ["W"] * 4 + ["I"] * 4
    assert np.max(np.abs(advantages(cats, 0.01) - advantages(cats, 1))) < 0.00021
    np.testing.assert_array_equal(advantages(cats, 1, no_x_off=True), np.zeros(8))
    observed = advantages(["W", "I"], 0.0002, epsilon=0.0001)
    np.testing.assert_allclose(observed, [1 / np.sqrt(2), -1 / np.sqrt(2)], atol=1e-15)
    assert not np.allclose(observed, [0.5, -0.5])


def test_rank_duplicate_columns_and_spectral_error():
    delta = np.array([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
    q, _ = orthogonal_span(delta)
    assert q.shape == (3, 2)
    np.testing.assert_allclose(q @ q.T @ delta, delta, atol=1e-12)
    assert orthogonal_span(np.zeros((3, 4)))[0].shape == (3, 0)
    r = np.random.default_rng(14).normal(size=(12, 7))
    for rank in range(8):
        approx, tail = truncate_response(r, rank)
        assert np.linalg.norm(r - approx, 2) == pytest.approx(tail, abs=1e-12)


def test_raw_simplex_prediction_preserved():
    raw = np.array([[-0.1, 0.2, 0.3, 0.6], [0.5, 0.5, 0.5, 0.5]])
    before = raw.copy()
    projected = project_simplex(raw)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_allclose(projected.sum(axis=-1), 1, atol=1e-15)
    assert np.min(projected) >= 0
    metrics = prediction_metrics(raw, np.full((2, 4), 0.25))
    assert metrics["raw_invalid_simplex_fraction"] == 1
    assert metrics["raw_negative_probability_max"] == 0.1
    assert metrics["raw_event_mae"] != metrics["projected_event_mae"]


def test_geometry_does_not_merge_equal_norm_updates():
    item = geometry([1, 0], [-1, 0])
    assert item["left_norm"] == item["right_norm"]
    assert item["distance"] == 2 and item["cosine"] == -1


def test_two_bounds_cover_each_intermediate_step_with_known_L():
    rng = np.random.default_rng(15)
    for _ in range(20):
        a, theta, bias = rng.normal(size=(4, 5)), rng.normal(size=5), rng.normal(size=4)
        weights = np.array([1.0, -0.5, 0.5, 0.0])
        u = np.linalg.qr(rng.normal(size=(5, 2)))[0]
        start, g = event_gradient_affine(a, theta, bias, weights)
        coef = u.T @ g + rng.normal(scale=0.01, size=2)
        eps, kappa = np.linalg.norm(u.T @ g - coef), np.linalg.norm(g - u @ (u.T @ g))
        steps = rng.normal(scale=0.04, size=(8, 5))
        for t in range(1, 9):
            d = steps[:t].sum(axis=0)
            truth, _ = event_gradient_affine(a, theta + d, bias, weights)
            estimate = start + 0.001 + coef @ (u.T @ d)
            bound = error_bounds(
                steps[:t], u, e=0.001, eps=eps, kappa=kappa, curvature=3 * np.linalg.norm(a, 2) ** 2
            )
            assert abs(truth - estimate) <= bound["net"] + 1e-12
            assert bound["net"] <= bound["path"] + 1e-12
    with pytest.raises(ValueError):
        error_bounds([[1, 2]], [[2], [0]])


def test_linear_regression_probability_helmert_equivalence():
    rng = np.random.default_rng(46)
    updates = rng.normal(size=(30, 5))
    p = rng.dirichlet(np.ones(4), size=(31, 6))
    delta = p[1:] - p[0]
    prob_coef = np.linalg.lstsq(updates, delta.reshape(30, -1), rcond=None)[0]
    helm_coef = np.linalg.lstsq(
        updates, to_helmert(delta, validate=False).reshape(30, -1), rcond=None
    )[0]
    query = rng.normal(size=(4, 5))
    prob_pred = (query @ prob_coef).reshape(4, 6, 4)
    helm_pred = (query @ helm_coef).reshape(4, 6, 3) @ helmert().T
    np.testing.assert_allclose(prob_pred, helm_pred, atol=1e-14)


def test_math_runner_writes_real_results_without_clobber(tmp_path):
    result = run_math({}, tmp_path / "M0")
    assert result["status"] == "PASS" and result["dtype"] == "float64"
    assert (tmp_path / "M0" / "coordinate_reconstruction.csv").exists()
    assert json.loads((tmp_path / "M0" / "math_results.json").read_text())["status"] == "PASS"
    with pytest.raises(FileExistsError):
        run_math({}, tmp_path / "M0")
