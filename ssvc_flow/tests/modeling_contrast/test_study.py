"""Runner boundaries and bounded staged experiment design."""

import numpy as np
import pytest

from src.modeling_contrast.study import (
    assemble_covariance,
    contrast_inputs,
    experiment_grid,
    prediction_first,
)


def test_bank_covariance_keeps_within_bank_cross_target_blocks():
    blocks = np.zeros((2, 3, 6, 6))
    blocks[0] = np.eye(6)
    blocks[1] = 2 * np.eye(6)
    blocks[:, :, 0, 3] = 0.1
    blocks[:, :, 3, 0] = 0.1
    result = assemble_covariance(blocks)
    assert result.shape == (3, 12, 12)
    assert np.all(result[:, 0, 3] == 0.1)
    assert np.all(result[:, :6, 6:] == 0)


def test_contrast_inputs_use_same_bank_main_endpoint():
    theta = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    e = contrast_inputs(theta)
    assert np.array_equal(e, np.stack((theta[:, 1] - theta[:, 0], theta[:, 2] - theta[:, 0]), 1))


def test_grid_has_nonzero_ablations_and_no_full_cartesian_stage():
    grid = experiment_grid("N2C", ["O_LR_ORIGIN"])
    assert all(x["rank"] != 0 for x in grid)
    assert {x["method"] for x in grid} >= {"C2", "C3", "C4_PCA", "C4_RANDOM", "C5", "C6"}
    assert {x["n"] for x in grid} == {64}
    with pytest.raises(ValueError):
        experiment_grid("N2C", ["O_IND", "O_CRN", "O_LR_MIX"])


def test_prediction_is_saved_before_oracle_is_accessed(tmp_path):
    history = []

    def predict():
        history.append("predict")
        return np.ones((2, 3))

    def score(pred):
        assert (tmp_path / "predictions.npz").exists()
        assert (tmp_path / "freeze.json").exists()
        history.append("score")
        return float(pred.sum())

    assert prediction_first(tmp_path, predict, score, {"fit_packet": "paid"}) == 6
    assert history == ["predict", "score"]
    with pytest.raises(FileExistsError):
        prediction_first(tmp_path, predict, score, {})
