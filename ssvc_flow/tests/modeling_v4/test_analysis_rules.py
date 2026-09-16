import numpy as np
import pytest

from src.modeling_v4.analysis_rules import endpoint_overlap, reference_precision


def test_reference_with_large_error_is_unresolved_without_erasing_finite_values():
    se = np.array([0.0, 1e-5, 0.001, 188982.24, np.nan])
    result = reference_precision(se)
    np.testing.assert_array_equal(result["reference_resolved"], [False, True, True, False, False])
    assert not result["resolved_at_scale"]["0.00025"][2]
    assert result["all_finite_residuals_must_also_be_reported"]
    assert not result["coverage_guarantee"]
    assert reference_precision([0.0], exact_alias=True)["reference_resolved"][0]
    with pytest.raises(ValueError):
        reference_precision([0.001], exact_alias=True)


def test_extreme_origin_weight_requests_independent_mixture_without_clipping():
    uniform = np.zeros((64, 2))
    uniform[0, 1] = 10000
    before = uniform.copy()
    result = endpoint_overlap(uniform)
    np.testing.assert_allclose(result["ess"], [64, 1])
    np.testing.assert_array_equal(result["usable"], [True, False])
    np.testing.assert_array_equal(uniform, before)
    assert not result["raw_weights_changed"]
    assert not result["accuracy_certified"]
