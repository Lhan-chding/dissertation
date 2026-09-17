"""Sampling-error regressions for post-hoc bridge analysis; tiny CPU fixtures."""

import numpy as np
import pytest

from src.modeling_v4.bridge_analysis import (
    aligned_scores,
    direct_intervals,
    independent_packets,
    moments,
)


def test_no_observed_event_is_not_zero_uncertainty():
    x = np.tile([1, 0, 0, 0], (64, 1))
    assert np.all(moments(x)["se"] == 0)
    ci = direct_intervals(x, x)
    # Each endpoint is a two-sided 97.5% CP interval; union bound >=95%.
    width = 1 - 0.0125 ** (1 / 64)
    np.testing.assert_allclose(ci, np.tile([-width, width], (4, 1)))
    family = direct_intervals(x, x, alpha=0.05 / 96)
    assert np.all(family[:, 0] < ci[:, 0])
    assert np.all(family[:, 1] > ci[:, 1])


def test_direct_interval_reverses_with_candidate_direction():
    a = np.eye(4)[[0, 0, 0, 1, 2, 3, 3, 3]]
    b = np.eye(4)[[0, 1, 1, 1, 2, 2, 2, 3]]
    np.testing.assert_allclose(direct_intervals(a, b), -direct_intervals(b, a)[:, ::-1])


def test_mass_uncertainty_preserves_cross_event_covariance():
    z = np.array([[1, -1, 0, 0], [-1, 1, 0, 0]], float)
    m = moments(z)
    assert m["mass_mean"] == m["mass_se"] == 0
    assert m["se"][0] > 0
    assert m["covariance_of_mean"][0, 1] < 0
    with pytest.raises(ValueError, match="Nonfinite"):
        moments([[np.nan, 0, 0, 0], [0, 0, 0, 0]])


def test_identical_decimal_contributions_do_not_gain_fake_precision():
    values = np.tile([0.0000786280101340414, 0.0, 0.0, 0.0], (64, 1))
    result = moments(values)
    assert np.array_equal(result["covariance_of_mean"], np.zeros((4, 4)))
    assert result["mass_se"] == 0


@pytest.mark.parametrize("field", ["sample_key", "sample_seed", "rng_namespace"])
def test_independence_rejects_each_kind_of_reuse(field):
    a = [{"sample_key": "a", "sample_seed": 1, "rng_namespace": "a"}]
    b = [{"sample_key": "b", "sample_seed": 2, "rng_namespace": "b"}]
    assert independent_packets([a, b])["overlap"] == 0
    b[0][field] = a[0][field]
    with pytest.raises(ValueError, match="reuse"):
        independent_packets([a, b])


def test_score_alignment_uses_keys_and_rejects_changed_tokens():
    rows = [
        {"sample_key": key, "prompt_id": "p", "shared_token_identity": key, "input_hash": "i"}
        for key in ("a", "b")
    ]
    scores = [
        {
            **r,
            "sequence_logp": -i - 1,
            "inference_fingerprint": "fp",
            "probability_execution": "uncached_prefix_recompute",
        }
        for i, r in enumerate(rows)
    ]

    class Reader:
        def rows(self, receipt, *, score):
            return list(reversed(scores))

    receipt = {"identity": {"policy": {"inference_fingerprint": "fp"}}}
    np.testing.assert_array_equal(aligned_scores(rows, receipt, Reader()), [-1, -2])
    scores[0]["shared_token_identity"] = "changed"
    with pytest.raises(ValueError, match="token"):
        aligned_scores(rows, receipt, Reader())
