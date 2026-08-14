from __future__ import annotations

import numpy as np

from compbias.theory.transport_bounds import total_variation, transport_diagnostic


def test_transport_gap_is_bounded_by_total_variation() -> None:
    rng = np.random.default_rng(81)
    for _ in range(2_000):
        natural = rng.dirichlet(np.ones(7))
        synthetic = rng.dirichlet(np.ones(7))
        continuation_value = rng.uniform(0.0, 1.0, size=7)
        result = transport_diagnostic(natural, synthetic, continuation_value)

        assert result.reward_gap <= result.tv_bound + 1e-12
        assert np.isclose(
            result.tv_bound,
            total_variation(natural, synthetic),
            rtol=0.0,
            atol=1e-15,
        )
        assert result.bound_satisfied


def test_identical_state_distributions_have_zero_transport_gap() -> None:
    distribution = np.array([0.2, 0.3, 0.5])
    result = transport_diagnostic(distribution, distribution, np.array([0.0, 0.5, 1.0]))
    assert result.reward_gap == 0.0
    assert result.tv_bound == 0.0
