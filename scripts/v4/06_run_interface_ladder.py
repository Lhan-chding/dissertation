"""Validate I0-I4 interface-ladder inputs before server execution."""

from __future__ import annotations

from _guards import LEGACY_SCREEN_RECORDS_SHA256
from _phase_cli import run_phase_preflight


def main() -> int:
    return run_phase_preflight(
        phase="phase_3_interface_ladder",
        description=__doc__ or "Phase 3 interface-ladder preflight",
        default_output_name="06_interface_ladder.json",
        intended_artifacts=(
            "artifacts/v4/interface_ladder/per_scene.jsonl",
            "artifacts/v4/interface_ladder/summary.json",
        ),
        integrity_gates=(
            "I0_I4_names_frozen",
            "four_cue_conditions_paired_by_scene",
            "counterfactual_world_legality",
            "cache_parity_required_for_I4_primary_use",
        ),
        expected_input_sha256=(LEGACY_SCREEN_RECORDS_SHA256,),
    )


if __name__ == "__main__":
    raise SystemExit(main())
