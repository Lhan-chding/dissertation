"""Oracle-only differentiation and fit-error isolation contracts."""

import numpy as np

from src.modeling_contrast.oracle_diagnostics import (
    autograd_helmert_jacobian,
    common_response_rank_one_witness,
    isolate_fit_errors,
)


def test_autograd_jacobian_matches_parent_analytic_and_preserves_parameters():
    from src.modeling_qualification.math_contracts import helmert
    from src.modeling_qualification.toy import (
        event_jacobian,
        event_probabilities_from_theta,
        flatten_parameters,
        make_model,
    )

    rng = np.random.default_rng(119)
    model = make_model(7001)
    theta = flatten_parameters(model)
    features = rng.normal(size=(2, 16, 44))
    labels = np.tile(np.arange(16) % 4, (2, 1))
    result = autograd_helmert_jacobian(model, theta, features, labels)
    expected = np.einsum("pew,ec->pcw", event_jacobian(theta, features, labels), helmert()).reshape(
        6, 737
    )
    assert np.allclose(result["jacobian"], expected, atol=1e-12, rtol=1e-10)
    assert np.array_equal(flatten_parameters(model), theta)
    assert all(p.grad is None for p in model.parameters())
    assert np.allclose(
        result["events"], event_probabilities_from_theta(theta, features, labels), atol=1e-15
    )
    assert result["cost"]["forward_model_calls"] == 1
    assert result["cost"]["batched_vjp_calls"] == 1
    assert result["cost"]["logical_vjp_directions"] == 6
    assert result["cost"]["optimizer_updates"] == 0


def test_common_response_rank_one_witness_does_not_imply_contrast_fit():
    result = common_response_rank_one_witness()
    assert result["overall_r1_energy_fraction"] > 0.99
    assert result["contrast_nrmse"] == 1.0
    assert result["contrast_truth_energy"] > 0
    assert result["diagnostic_only"] is True


def test_error_isolation_returns_all_nonzero_ranks_and_three_distinct_sources():
    e = np.eye(3)
    exact = np.diag([3.0, 2.0, 1.0])
    finite = np.diag([1.0, 2.0, 3.0])
    cov = np.eye(9)[None]
    group = np.array([[1.0, 0, 0], [0, 1, 0]])
    result = isolate_fit_errors(e, finite, exact, cov, e, group, 64)
    assert len(result["records"]) == 18
    assert result["predictions"].shape == (18, 3, 3)
    assert result["main_ranking_eligible"] is False
    assert {r["diagnostic"] for r in result["records"]} == {
        "EXACT_U_FINITE_COEFFICIENTS",
        "FINITE_U_EXACT_COEFFICIENTS",
        "FINITE_U_FINITE_COEFFICIENTS",
    }
    assert {r["rank_cap"] for r in result["records"]} == {1, 2, "FULL"}
    assert all(r["actual_rank"] > 0 for r in result["records"])
    assert all("projector_frobenius_distance" in r["subspace_to_exact"] for r in result["records"])


def test_oracle_covariance_remains_separately_labeled_and_alias_rows_supported():
    e = np.eye(3)
    finite = np.eye(3)
    group = np.array([[1.0, 0, 0], [0, 1, 0]])
    result = isolate_fit_errors(
        e,
        finite,
        finite,
        np.eye(9)[None],
        e,
        group,
        64,
        oracle_covariance=2 * np.eye(9)[None],
        active_rows=[0, 1],
    )
    assert len(result["records"]) == 36
    assert {r["covariance_source"] for r in result["records"]} == {"FINITE_PAID", "EXACT_ORACLE"}
    assert all(r["actual_rank"] <= 2 for r in result["records"])
    finite_only = result["predictions"][:18]
    oracle_only = result["predictions"][18:]
    assert np.allclose(finite_only, oracle_only, atol=1e-12)
