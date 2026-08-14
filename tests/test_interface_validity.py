from __future__ import annotations

import numpy as np

from compbias.identification.interface_spec import InterfaceSpec
from compbias.identification.partial_identification import (
    InterfaceGammaEstimate,
    robust_compensation_interval,
)
from compbias.identification.validity_gates import evaluate_interface_validity


def _valid_report(interface_id: str):
    return evaluate_interface_validity(
        interface_id=interface_id,
        oracle_loss=0.0,
        replay_js=0.01,
        replay_accuracy_gap=0.01,
        image_exclusion_gap=0.0,
        parser_reliability=0.99,
        natural_state_count_by_error={"offset:+2": 240, "truth": 300},
        natural_input_count_by_error={"offset:+2": 60, "truth": 100},
    )


def test_interface_spec_is_operational_not_anatomically_privileged() -> None:
    interface = InterfaceSpec(
        interface_id="evidence-prefix-v1",
        mode="behavioral",
        boundary="first_visual_fact_block",
        image_cut_mode="remove_image_context",
        parser_id="scene-parser-v2",
        oracle_serializer_id="cva-json-v2",
    )
    assert interface.mode == "behavioral"
    assert "unique" not in interface.to_mapping()


def test_validity_gate_rejects_low_support_even_if_gamma_is_favorable() -> None:
    invalid = evaluate_interface_validity(
        interface_id="thin-interface",
        oracle_loss=0.0,
        replay_js=0.01,
        replay_accuracy_gap=0.01,
        image_exclusion_gap=0.0,
        parser_reliability=0.99,
        natural_state_count_by_error={"offset:+2": 12},
        natural_input_count_by_error={"offset:+2": 4},
    )
    assert not invalid.passed
    assert "natural_support" in invalid.failed_gates

    result = robust_compensation_interval(
        (
            InterfaceGammaEstimate("valid-a", (-0.30, -0.25, -0.20, -0.28)),
            InterfaceGammaEstimate("thin-interface", (-0.99, -0.99, -0.99, -0.99)),
        ),
        (_valid_report("valid-a"), invalid),
        bootstrap_draws=2_000,
        confidence=0.95,
        seed=42,
    )
    assert result.valid_interfaces == ("valid-a",)
    assert result.invalid_interfaces == ("thin-interface",)
    assert result.point_upper < 0.0


def test_partial_identification_reports_uncertain_when_interfaces_cross_zero() -> None:
    result = robust_compensation_interval(
        (
            InterfaceGammaEstimate("negative", (-0.2, -0.1, -0.3, -0.2)),
            InterfaceGammaEstimate("positive", (0.1, 0.2, 0.3, 0.2)),
        ),
        (_valid_report("negative"), _valid_report("positive")),
        bootstrap_draws=2_000,
        confidence=0.95,
        seed=19,
    )
    assert result.point_lower < 0.0 < result.point_upper
    assert result.conclusion == "not_identified"
    assert np.isfinite(result.simultaneous_lower)
    assert np.isfinite(result.simultaneous_upper)
