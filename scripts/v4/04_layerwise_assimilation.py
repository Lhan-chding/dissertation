"""Validate layerwise-assimilation inputs and objective parity gates."""

from __future__ import annotations

from _guards import LEGACY_SCREEN_RECORDS_SHA256
from _phase_cli import run_phase_preflight


def main() -> int:
    return run_phase_preflight(
        phase="phase_2_layerwise_assimilation",
        description=__doc__ or "Phase 2 layerwise-assimilation preflight",
        default_output_name="04_layerwise_assimilation.json",
        intended_artifacts=("artifacts/v4/layerwise_assimilation/per_scene.jsonl",),
        integrity_gates=(
            "runtime_layer_count",
            "runtime_norm_and_lm_head",
            "final_layer_standard_forward_logit_parity",
        ),
        expected_input_sha256=(LEGACY_SCREEN_RECORDS_SHA256,),
    )


if __name__ == "__main__":
    raise SystemExit(main())
