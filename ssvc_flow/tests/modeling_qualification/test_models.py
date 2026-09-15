import numpy as np
import pytest

from src.modeling_qualification.models import build_subspace, fit_local_model, ridge_map


def test_uncentered_rank_and_alias_invariance():
    d = np.array([[1.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    q, meta = build_subspace(d)
    assert q.shape == (3, 1)
    assert meta["k"] == 1
    assert meta["alias_or_dependent_columns"] == 2
    assert np.allclose(q @ q.T @ d.T, d.T)


def test_exact_ridge_and_zero_response_contract():
    a = np.eye(2)
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = ridge_map(a, y, 0.0)
    assert np.allclose(a @ b.T, y)
    assert np.allclose(np.zeros((1, 2)) @ b.T, 0)


def test_all_direction_models_use_fit_only_and_full_matches():
    rng = np.random.default_rng(3)
    d = rng.normal(size=(5, 7))
    y = rng.normal(size=(5, 6))
    fitted = [
        fit_local_model(d, y, method, "FULL", 1e-5)
        for method in (
            "B2_RANDOM_Q",
            "B3_UPDATE_PCA",
            "B4_RESPONSE_SVD",
            "B5_WORST_GROUP_RANK",
            "B6_FULL_Q_RIDGE",
        )
    ]
    assert all(x["k"] == 5 for x in fitted)
    pred = [x["predict"](d) for x in fitted]
    assert all(np.allclose(p, pred[-1], atol=1e-9) for p in pred)
    assert all(np.allclose(x["predict"](np.zeros((1, 7))), 0) for x in fitted)


def test_empty_subspace_is_explicit_and_inputs_rejected():
    fitted = fit_local_model(np.zeros((3, 4)), np.ones((3, 6)), "B4_RESPONSE_SVD", 2, 0.0)
    assert fitted["status"] == "NO_IDENTIFIED_UPDATE_SUBSPACE"
    assert np.all(fitted["predict"](np.ones((2, 4))) == 0)
    with pytest.raises(ValueError):
        fit_local_model(np.full((2, 4), np.nan), np.ones((2, 3)), "B6_FULL_Q_RIDGE", "FULL", 0.0)
