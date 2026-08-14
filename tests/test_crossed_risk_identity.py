from __future__ import annotations

import numpy as np

from compbias.theory.crossed_risk import crossed_risk_decomposition


def test_crossed_risk_identity_holds_for_random_bounded_losses() -> None:
    rng = np.random.default_rng(918)
    for _ in range(10_000):
        l_mm, l_om, l_mo, l_oo = rng.uniform(0.0, 1.0, size=4)
        result = crossed_risk_decomposition(
            l_mm=float(l_mm),
            l_om=float(l_om),
            l_mo=float(l_mo),
            l_oo=float(l_oo),
        )
        assert abs(result.identity_residual) < 1e-15
        assert np.isclose(
            result.l_mm - result.l_oo,
            result.perception_deficit + result.reasoning_deficit + result.interaction,
            rtol=0.0,
            atol=1e-15,
        )


def test_negative_interaction_is_labeled_compensatory() -> None:
    result = crossed_risk_decomposition(l_mm=0.2, l_om=0.3, l_mo=0.4, l_oo=0.0)
    assert result.interaction < 0.0
    assert result.interaction_label == "compensation"
