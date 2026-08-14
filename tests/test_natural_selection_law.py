from __future__ import annotations

import numpy as np

from compbias.theory.natural_selection import (
    natural_binary_selected_distribution,
    natural_selection_shift,
)


def test_natural_selection_identity_holds_for_random_trajectory_tables() -> None:
    rng = np.random.default_rng(20260814)
    for _ in range(10_000):
        mu0 = rng.dirichlet(np.ones(5))
        c_sel = rng.uniform(0.0, 1.0, size=5)
        severity = rng.uniform(0.0, 3.0, size=5)
        beta = float(rng.uniform(0.1, 4.0))

        selected = natural_binary_selected_distribution(mu0, c_sel, beta)
        result = natural_selection_shift(mu0, severity, c_sel, beta)

        direct = float(selected @ severity - mu0 @ severity)
        assert np.isclose(result.direct_shift, direct, rtol=0.0, atol=1e-12)
        assert np.isclose(result.covariance_shift, direct, rtol=0.0, atol=1e-12)
        assert result.identity_residual < 1e-12


def test_selection_uses_natural_reward_not_forked_or_synthetic_reward() -> None:
    mu0 = np.array([0.5, 0.5])
    c_sel = np.array([0.9, 0.1])
    c_fork = np.array([0.1, 0.9])
    severity = np.array([0.0, 1.0])

    selected = natural_binary_selected_distribution(mu0, c_sel, beta=1.0)
    counterfactual_prediction = natural_binary_selected_distribution(mu0, c_fork, beta=1.0)

    assert selected[0] > mu0[0]
    assert selected[1] < mu0[1]
    assert counterfactual_prediction[1] > mu0[1]
    assert natural_selection_shift(mu0, severity, c_sel, beta=1.0).direct_shift < 0.0
