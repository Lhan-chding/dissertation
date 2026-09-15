import numpy as np
import pytest

from src.modeling_contrast.targets import (
    contrast_decomposition,
    contrast_incidence,
    contrast_taylor_bound,
    group_xv_matrix,
    helmert,
    inference_fingerprint,
)


def test_targets_cancel_origin_and_preserve_consistency():
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(4), size=(4, 6))
    result = contrast_decomposition(p[0], p[1], p[2:])
    np.testing.assert_allclose(result["T"], result["B"] + result["C"])
    changed = contrast_decomposition(p[3], p[1], p[2:])
    np.testing.assert_array_equal(result["C"], changed["C"])
    np.testing.assert_allclose(result["C"][0] - result["C"][1], p[2] - p[3])


def test_alias_incidence_is_exact_zero():
    policies, incidence = contrast_incidence([("b", "u"), ("b", "a")], {"a": "b"})
    assert policies == ("b", "u")
    np.testing.assert_array_equal(incidence, [[-1, 1], [0, 0]])


def test_same_norm_is_not_same_inference():
    assert inference_fingerprint([1.0, 0.0]) != inference_fingerprint([-1.0, 0.0])


def test_group_xv_matches_probability_contrasts():
    metadata = [{"prompt_id": str(i), "group": i // 2, "weight": i + 1} for i in range(4)]
    mapping = group_xv_matrix(metadata)
    rng = np.random.default_rng(2)
    z = rng.normal(size=(4, 3))
    p = z @ helmert().T
    expected = []
    for group in range(2):
        ids = [i for i in range(4) if i // 2 == group]
        mean = np.average(p[ids], axis=0, weights=np.array(ids) + 1)
        expected.extend([mean[0], mean[:3].sum()])
    np.testing.assert_allclose(mapping @ z.ravel(), expected, atol=1e-15)
    with pytest.raises(ValueError):
        group_xv_matrix([metadata[0], metadata[0]])


def test_contrast_taylor_quadratic_and_zero_increment():
    a = np.diag([1.0, 2.0, 4.0])
    x = np.array([0.1, 0.2, 0.3])
    d = np.array([0.4, -0.2, 0.5])
    e = np.array([0.01, 0.03, -0.02])

    def f(q):
        return 0.5 * q @ a @ q

    residual = abs(f(x + d + e) - f(x + d) - (a @ x) @ e)
    assert residual <= contrast_taylor_bound(4, d, e)
    assert contrast_taylor_bound(4, d, np.zeros(3)) == 0
