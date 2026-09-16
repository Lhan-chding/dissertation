import numpy as np
import pytest

from src.modeling_v4.error_decomposition import (
    decompose_response,
    exact_toy_geometry,
    fisher_direction_diagnostics,
)


def test_signed_vector_terms_and_all_cross_terms_reconstruct_actual_error():
    rng = np.random.default_rng(44)
    e = rng.normal(size=(3, 5))
    q, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    jacobian = rng.normal(size=(3, 2, 4, 5))
    truth = rng.normal(size=(3, 2, 4))
    prediction = rng.normal(size=truth.shape)
    result = decompose_response(truth, jacobian, e, q, prediction)
    np.testing.assert_allclose(
        result["nonlinearity"] + result["omitted_semantic"] + result["fit_error"],
        truth - prediction,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        result["component_squared_norms"].sum(-1) + result["cross_terms"].sum(-1),
        result["total_squared_error"],
        atol=3e-13,
    )
    assert result["cross_terms"].shape == (3, 2, 3)
    assert np.any(result["cross_terms"] < 0)
    np.testing.assert_allclose(result["e_parallel"] + result["e_perp"], e)
    assert result["error_sign"] == "REFERENCE_MINUS_PREDICTION"


def test_zero_rank_is_a_defined_geometry_but_nan_prediction_remains_unknown():
    truth = np.zeros((1, 2, 4))
    result = decompose_response(
        truth,
        np.zeros((1, 2, 4, 5)),
        np.ones((1, 5)),
        np.empty((5, 0)),
        np.full_like(truth, np.nan),
    )
    np.testing.assert_array_equal(result["e_perp"], np.ones((1, 5)))
    assert np.isnan(result["fit_error"]).all()
    assert np.isnan(result["total_squared_error"]).all()


def test_nonorthogonal_basis_is_not_silently_used_as_a_projector():
    with pytest.raises(ValueError, match="orthonormal"):
        decompose_response(
            np.zeros((1, 1, 4)),
            np.zeros((1, 1, 4, 2)),
            np.ones((1, 2)),
            np.ones((2, 1)),
            np.zeros((1, 1, 4)),
        )


def test_actual_737_parameter_toy_scores_jacobian_and_fisher_bound():
    from src.modeling_qualification.toy import (
        event_probabilities_from_theta,
        flatten_parameters,
        make_model,
    )

    rng = np.random.default_rng(2)
    theta = flatten_parameters(make_model(91))
    features = rng.normal(size=(2, 16, 44))
    categories = np.tile(np.arange(16) % 4, (2, 1))
    direction = rng.normal(size=737)
    direction /= np.linalg.norm(direction)
    geometry = exact_toy_geometry(theta, features, categories)
    finite_difference = (
        event_probabilities_from_theta(theta + 1e-5 * direction, features, categories)
        - event_probabilities_from_theta(theta - 1e-5 * direction, features, categories)
    ) / 2e-5
    np.testing.assert_allclose(geometry["jacobian"] @ direction, finite_difference, atol=2e-10)
    np.testing.assert_allclose(
        np.einsum("pa,pad->pd", geometry["action_probabilities"], geometry["scores"]), 0, atol=1e-15
    )
    actions = np.stack([rng.choice(16, 64, p=p) for p in geometry["action_probabilities"]])
    fisher = fisher_direction_diagnostics(geometry, direction, actions=actions)
    assert np.all(np.abs(geometry["jacobian"] @ direction) <= fisher["exact_local_bound"] + 1e-14)
    manual = np.einsum("pad,d->pa", geometry["scores"], direction)
    np.testing.assert_allclose(
        fisher["exact_energy"], (geometry["action_probabilities"] * manual**2).sum(-1)
    )
    assert fisher["empirical_sample_count"] == 64
    assert fisher["empirical_bound_is_certified"] is False


def test_shape_and_bad_fisher_actions_fail_explicitly():
    geometry = {
        "action_probabilities": np.full((1, 2), 0.5),
        "scores": np.zeros((1, 2, 3)),
        "mass": np.full((1, 4), 0.25),
    }
    with pytest.raises(ValueError, match="actions"):
        fisher_direction_diagnostics(geometry, np.ones(3), actions=np.array([[2]]))
