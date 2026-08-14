from __future__ import annotations

import numpy as np
import pytest

from compbias.theory.frozen_regimes import audit_frozen_regime


def test_f0_requires_acquisition_mediator_and_perception_deficit_invariance() -> None:
    representation = np.array([[0.1, 0.2], [0.3, 0.4]])
    mediator = np.array([[0.8, 0.2], [0.1, 0.9]])
    report = audit_frozen_regime(
        regime="F0",
        acquisition_before=representation,
        acquisition_after=representation.copy(),
        mediator_before=mediator,
        mediator_after=mediator.copy(),
        perception_deficit_before=0.25,
        perception_deficit_after=0.25,
    )
    assert report.passed
    assert report.allowed_claims == ("reasoning", "interaction")


def test_f1_does_not_mislabel_frozen_vision_as_frozen_perception() -> None:
    report = audit_frozen_regime(
        regime="F1",
        acquisition_before=np.array([1.0, 2.0]),
        acquisition_after=np.array([1.0, 2.0]),
        mediator_before=np.array([0.8, 0.2]),
        mediator_after=np.array([0.6, 0.4]),
        perception_deficit_before=0.2,
        perception_deficit_after=0.1,
    )
    assert report.passed
    assert "readout" in report.allowed_claims
    assert report.mediator_drift > 0.0


def test_f0_rejects_changed_mediator() -> None:
    with pytest.raises(ValueError, match="F0"):
        audit_frozen_regime(
            regime="F0",
            acquisition_before=np.array([1.0]),
            acquisition_after=np.array([1.0]),
            mediator_before=np.array([0.2, 0.8]),
            mediator_after=np.array([0.3, 0.7]),
            perception_deficit_before=0.1,
            perception_deficit_after=0.1,
        )
