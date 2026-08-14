"""RED contracts for immutable categorical perceiver/reasoner policies."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from compbias.models.tabular_policy import CategoricalPolicy


def test_policy_update_is_exponentiated_and_does_not_mutate_either_input() -> None:
    probabilities = np.array([0.45, 0.25, 0.20, 0.10], dtype=np.float64)
    logit_delta = np.array([-0.4, -0.1, 0.2, 0.8], dtype=np.float64)
    probabilities_before = probabilities.copy()
    delta_before = logit_delta.copy()
    policy = CategoricalPolicy.from_probabilities(
        probabilities,
        labels=("truth", "e1", "e2", "e3"),
    )

    updated = policy.updated(logit_delta)
    expected = probabilities * np.exp(logit_delta - np.max(logit_delta))
    expected /= expected.sum()

    np.testing.assert_allclose(updated.probabilities, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(policy.probabilities, probabilities_before)
    np.testing.assert_array_equal(probabilities, probabilities_before)
    np.testing.assert_array_equal(logit_delta, delta_before)
    assert updated is not policy
    assert tuple(updated.labels) == ("truth", "e1", "e2", "e3")


def test_probability_views_are_defensive_copies_and_policy_object_is_frozen() -> None:
    source = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    policy = CategoricalPolicy.from_probabilities(source)

    source[0] = 1.0
    exposed = policy.probabilities
    exposed[1] = 1.0

    np.testing.assert_allclose(policy.probabilities, [0.2, 0.3, 0.5], atol=0.0)
    assert not np.shares_memory(policy.probabilities, source)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        policy.probabilities = np.array([1.0, 0.0, 0.0])  # type: ignore[misc]


def test_large_finite_logit_update_is_stable_and_keeps_zero_support_zero() -> None:
    policy = CategoricalPolicy.from_probabilities([0.7, 0.0, 0.3])

    updated = policy.updated(np.array([-1_000.0, 1_000.0, 1_000.0]))

    assert np.all(np.isfinite(updated.probabilities))
    assert updated.probabilities.sum() == pytest.approx(1.0, abs=1e-15)
    assert updated.probabilities[0] == pytest.approx(0.0, abs=1e-15)
    assert updated.probabilities[1] == pytest.approx(0.0, abs=0.0)
    assert updated.probabilities[2] == pytest.approx(1.0, abs=1e-15)


def test_sampling_uses_the_supplied_generator_and_is_reproducible() -> None:
    policy = CategoricalPolicy.from_probabilities([0.1, 0.3, 0.6])

    first = policy.sample(size=20_000, rng=np.random.default_rng(20260814))
    second = policy.sample(size=20_000, rng=np.random.default_rng(20260814))

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(
        np.bincount(first, minlength=3) / first.size,
        policy.probabilities,
        atol=0.012,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([]),
        np.array([[0.5, 0.5]]),
        np.array([0.2, 0.2]),
        np.array([0.5, -0.1, 0.6]),
        np.array([0.5, np.nan, 0.5]),
        np.array([0.0, 0.0]),
    ],
)
def test_policy_rejects_invalid_probability_vectors(probabilities: np.ndarray) -> None:
    with pytest.raises(ValueError):
        CategoricalPolicy.from_probabilities(probabilities)


def test_policy_rejects_bad_labels_updates_and_sample_sizes() -> None:
    with pytest.raises(ValueError):
        CategoricalPolicy.from_probabilities([0.5, 0.5], labels=("truth",))
    with pytest.raises(ValueError):
        CategoricalPolicy.from_probabilities([0.5, 0.5], labels=("same", "same"))

    policy = CategoricalPolicy.from_probabilities([0.5, 0.5])
    with pytest.raises(ValueError):
        policy.updated(np.array([1.0]))
    with pytest.raises(ValueError):
        policy.updated(np.array([0.0, np.nan]))
    with pytest.raises(ValueError):
        policy.sample(size=0, rng=np.random.default_rng(0))
