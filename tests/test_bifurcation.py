"""RED contracts for the symmetric KL coordination bifurcation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from compbias.theory.coordination import (
    symmetric_bifurcation_branch,
    symmetric_bifurcation_root,
)


@pytest.mark.parametrize("beta_over_a", [0.5, 0.6, 1.0])
def test_at_or_above_critical_beta_only_the_center_root_exists(beta_over_a: float) -> None:
    assert symmetric_bifurcation_root(beta_over_a) == pytest.approx(0.0, abs=1e-14)


def test_below_critical_beta_returns_the_positive_nonzero_root() -> None:
    beta_over_a = 0.3

    root = symmetric_bifurcation_root(beta_over_a)

    assert 0.0 < root < 1.0
    assert root == pytest.approx(0.9073323166453315, abs=1e-10, rel=0.0)
    assert 2.0 * beta_over_a * np.arctanh(root) == pytest.approx(root, abs=1e-10)
    assert 2.0 * beta_over_a * np.arctanh(-root) == pytest.approx(-root, abs=1e-10)


def test_nonzero_branch_continuously_emerges_below_critical_beta() -> None:
    near_critical_root = symmetric_bifurcation_root(0.499999)
    farther_root = symmetric_bifurcation_root(0.49)

    assert 0.0 < near_critical_root < 0.01
    assert near_critical_root < farther_root < 1.0


@pytest.mark.parametrize("beta_over_a", [-1.0, -1e-12, np.nan, np.inf])
def test_bifurcation_root_rejects_invalid_ratio(beta_over_a: float) -> None:
    with pytest.raises(ValueError):
        symmetric_bifurcation_root(beta_over_a)


def test_bifurcation_branch_has_center_and_symmetric_nonzero_roots() -> None:
    ratios = np.array([0.6, 0.5, 0.3])
    original = ratios.copy()

    branch = symmetric_bifurcation_branch(ratios)

    np.testing.assert_array_equal(branch.beta_over_a, ratios)
    np.testing.assert_array_equal(branch.center, np.zeros_like(ratios))
    np.testing.assert_array_equal(branch.positive[:2], np.zeros(2))
    np.testing.assert_array_equal(branch.negative[:2], np.zeros(2))
    assert branch.positive[2] == pytest.approx(0.9073323166453315, abs=1e-10)
    assert branch.negative[2] == pytest.approx(-branch.positive[2], abs=1e-14)
    np.testing.assert_array_equal(ratios, original)
    assert not np.shares_memory(branch.beta_over_a, ratios)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        branch.center = np.ones_like(ratios)  # type: ignore[misc]


def test_bifurcation_branch_rejects_non_vector_input() -> None:
    with pytest.raises(ValueError):
        symmetric_bifurcation_branch(np.ones((2, 2)))
