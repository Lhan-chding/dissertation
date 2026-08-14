from __future__ import annotations

import numpy as np

from compbias.theory.density_ratio_identity import checkpoint_selection_ratio


def test_checkpoint_selection_identity_with_common_support() -> None:
    before = {"truth": 0.7, "offset": 0.3}
    after = {"truth": 0.4, "offset": 0.6}
    severity = {"truth": 0.0, "offset": 2.0}
    result = checkpoint_selection_ratio(before, after, severity)

    assert result.new_support_mass == 0.0
    assert np.isclose(result.ratios["offset"], 2.0, rtol=0.0, atol=1e-15)
    assert np.isclose(result.ratios["truth"], 4.0 / 7.0, rtol=0.0, atol=1e-15)
    assert np.isclose(result.actual_shift, 0.6, atol=1e-15)
    assert result.identity_residual < 1e-15


def test_checkpoint_selection_identity_reports_new_support() -> None:
    before = {"truth": 1.0, "novel": 0.0}
    after = {"truth": 0.8, "novel": 0.2}
    severity = {"truth": 0.0, "novel": 3.0}
    result = checkpoint_selection_ratio(before, after, severity)

    assert np.isclose(result.new_support_mass, 0.2)
    assert result.new_support == ("novel",)
    assert np.isclose(result.actual_shift, 0.6)
    assert result.identity_residual < 1e-15
