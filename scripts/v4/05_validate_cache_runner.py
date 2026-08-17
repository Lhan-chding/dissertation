"""Validate exact-cache continuation inputs before parity execution."""

from __future__ import annotations

from _guards import LEGACY_SCREEN_RECORDS_SHA256
from _phase_cli import run_phase_preflight


def main() -> int:
    return run_phase_preflight(
        phase="phase_3_cache_parity",
        description=__doc__ or "Phase 3 cache-parity preflight",
        default_output_name="05_cache_parity.json",
        intended_artifacts=("artifacts/v4/cache/cache_parity.json",),
        integrity_gates=(
            "saved_image_token_positions",
            "saved_image_grid_thw",
            "exact_suffix_tokens",
            "cache_sample_isolation",
            "greedy_token_parity_with_full_reencode",
        ),
        expected_input_sha256=(LEGACY_SCREEN_RECORDS_SHA256,),
    )


if __name__ == "__main__":
    raise SystemExit(main())
