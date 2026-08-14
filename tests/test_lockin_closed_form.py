"""RED tests for fixed-landscape compensability lock-in."""

from __future__ import annotations

import numpy as np
import pytest

from compbias.theory.lockin import (
    repeated_selection,
    repeated_selection_closed_form,
    selection_update,
)

PROPERTY_CASES = 1_000


def test_selection_update_matches_one_step_reweighting_without_mutation() -> None:
    mu = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    multiplier = np.array([0.5, 2.0, 1.2], dtype=np.float64)
    mu_before = mu.copy()
    multiplier_before = multiplier.copy()

    actual = selection_update(mu, multiplier)
    expected = mu * multiplier / np.dot(mu, multiplier)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(mu, mu_before)
    np.testing.assert_array_equal(multiplier, multiplier_before)


def test_iterated_update_matches_closed_form_and_ratio_law() -> None:
    mu0 = np.array([0.15, 0.35, 0.50], dtype=np.float64)
    multiplier = np.array([0.4, 1.7, 0.9], dtype=np.float64)
    steps = 12

    iterated = repeated_selection(mu0, multiplier, steps)
    closed_form = repeated_selection_closed_form(mu0, multiplier, steps)
    direct = mu0 * multiplier**steps
    direct = direct / direct.sum()

    np.testing.assert_allclose(iterated, direct, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(closed_form, direct, rtol=0.0, atol=1e-13)
    for error_index in (0, 2):
        actual_ratio = closed_form[error_index] / closed_form[1]
        expected_ratio = (mu0[error_index] / mu0[1]) * (
            multiplier[error_index] / multiplier[1]
        ) ** steps
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-12, abs=0.0)


def test_zero_steps_is_identity_and_unique_maximum_locks_in() -> None:
    mu0 = np.array([0.4, 0.35, 0.25], dtype=np.float64)
    multiplier = np.array([0.8, 1.2, 0.9], dtype=np.float64)

    np.testing.assert_array_equal(repeated_selection(mu0, multiplier, 0), mu0)
    np.testing.assert_array_equal(repeated_selection_closed_form(mu0, multiplier, 0), mu0)

    late = repeated_selection_closed_form(mu0, multiplier, 200)
    assert late[1] > 1.0 - 1e-12
    assert np.argmax(late) == 1
    assert late.sum() == pytest.approx(1.0, abs=1e-15)


def test_tied_maxima_preserve_their_initial_odds() -> None:
    mu0 = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    multiplier = np.array([2.0, 2.0, 0.5], dtype=np.float64)

    selected = repeated_selection_closed_form(mu0, multiplier, 100)

    assert selected[0] / selected[1] == pytest.approx(mu0[0] / mu0[1], rel=1e-12)
    assert selected[2] < 1e-50


def test_closed_form_is_stable_when_direct_powers_overflow_or_underflow() -> None:
    mu0 = np.array([0.4, 0.35, 0.25], dtype=np.float64)

    huge = repeated_selection_closed_form(mu0, np.array([1e200, 1e199, 1e198]), 10)
    huge_rescaled = repeated_selection_closed_form(mu0, np.array([1.0, 0.1, 0.01]), 10)
    np.testing.assert_allclose(huge, huge_rescaled, rtol=1e-12, atol=1e-15)

    tiny = repeated_selection_closed_form(mu0[:2] / mu0[:2].sum(), np.array([1e-200, 2e-200]), 10)
    tiny_rescaled = repeated_selection_closed_form(
        mu0[:2] / mu0[:2].sum(), np.array([0.5, 1.0]), 10
    )
    np.testing.assert_allclose(tiny, tiny_rescaled, rtol=1e-12, atol=1e-15)

    assert np.all(np.isfinite(huge))
    assert np.all(np.isfinite(tiny))
    assert huge.sum() == pytest.approx(1.0, abs=1e-15)
    assert tiny.sum() == pytest.approx(1.0, abs=1e-15)


def test_closed_form_matches_stepping_for_1000_seeded_random_cases() -> None:
    rng = np.random.default_rng(271828)
    max_iterated_error = 0.0
    max_closed_form_error = 0.0

    for _ in range(PROPERTY_CASES):
        size = int(rng.integers(2, 10))
        raw = rng.uniform(0.05, 1.0, size=size)
        mu0 = raw / raw.sum()
        multiplier = np.exp(rng.uniform(-2.0, 2.0, size=size))
        steps = int(rng.integers(0, 20))

        manual = mu0.copy()
        for _step in range(steps):
            manual = selection_update(manual, multiplier)

        iterated = repeated_selection(mu0, multiplier, steps)
        closed_form = repeated_selection_closed_form(mu0, multiplier, steps)
        max_iterated_error = max(max_iterated_error, float(np.max(np.abs(iterated - manual))))
        max_closed_form_error = max(
            max_closed_form_error, float(np.max(np.abs(closed_form - manual)))
        )

        assert np.all(np.isfinite(iterated))
        assert np.all(np.isfinite(closed_form))
        assert abs(float(iterated.sum()) - 1.0) < 1e-12
        assert abs(float(closed_form.sum()) - 1.0) < 1e-12

    assert max_iterated_error < 1e-12
    assert max_closed_form_error < 1e-12


@pytest.mark.parametrize(
    ("mu", "multiplier"),
    [
        (np.array([0.3, 0.3]), np.array([1.0, 2.0])),
        (np.array([0.5, -0.5, 1.0]), np.ones(3)),
        (np.array([0.5, np.nan, 0.5]), np.ones(3)),
        (np.array([0.5, 0.5]), np.array([1.0])),
        (np.array([0.5, 0.5]), np.array([1.0, 0.0])),
        (np.array([0.5, 0.5]), np.array([1.0, -1.0])),
        (np.array([0.5, 0.5]), np.array([1.0, np.inf])),
    ],
)
def test_lockin_functions_reject_invalid_distributions_and_multipliers(
    mu: np.ndarray, multiplier: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        selection_update(mu, multiplier)
    with pytest.raises(ValueError):
        repeated_selection(mu, multiplier, 2)
    with pytest.raises(ValueError):
        repeated_selection_closed_form(mu, multiplier, 2)


@pytest.mark.parametrize("steps", [-1, 1.5, np.nan, True, "2"])
def test_lockin_functions_reject_invalid_step_counts(steps: object) -> None:
    mu0 = np.array([0.5, 0.5])
    multiplier = np.array([1.0, 2.0])

    with pytest.raises((TypeError, ValueError)):
        repeated_selection(mu0, multiplier, steps)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        repeated_selection_closed_form(mu0, multiplier, steps)  # type: ignore[arg-type]


def test_torch_closed_form_matches_iterated_update_and_preserves_autograd() -> None:
    torch = pytest.importorskip("torch")
    dtype = torch.float64
    mu0 = torch.tensor([0.2, 0.3, 0.5], dtype=dtype)
    multiplier = torch.tensor([0.8, 1.3, 0.9], dtype=dtype, requires_grad=True)

    one_step = selection_update(mu0, multiplier)
    iterated = repeated_selection(mu0, multiplier, 17)
    closed_form = repeated_selection_closed_form(mu0, multiplier, 17)

    assert isinstance(one_step, torch.Tensor)
    assert isinstance(iterated, torch.Tensor)
    assert isinstance(closed_form, torch.Tensor)
    torch.testing.assert_close(iterated, closed_form, rtol=0.0, atol=1e-12)
    assert closed_form.sum().item() == pytest.approx(1.0, abs=1e-15)

    closed_form[0].backward()
    assert multiplier.grad is not None
    assert torch.all(torch.isfinite(multiplier.grad))
