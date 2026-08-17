"""Validate the T1-T6 capability-chain server execution surface."""

from __future__ import annotations

from _guards import LEGACY_SCREEN_RECORDS_SHA256
from _phase_cli import run_phase_preflight


def main() -> int:
    return run_phase_preflight(
        phase="phase_1_capability_chain",
        description=__doc__ or "Phase 1 capability-chain preflight",
        default_output_name="02_capability_chain.json",
        intended_artifacts=(
            "artifacts/v4/capability_chain/per_scene.csv",
            "artifacts/v4/capability_chain/summary_by_family.csv",
            "artifacts/v4/capability_chain/paired_gaps.json",
        ),
        integrity_gates=("hash_bound_legacy_scenes", "minimal_T1_T6_output_contract"),
        expected_input_sha256=(LEGACY_SCREEN_RECORDS_SHA256,),
    )


if __name__ == "__main__":
    raise SystemExit(main())
