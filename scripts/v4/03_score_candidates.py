"""Validate teacher-forced candidate-scoring inputs before model work."""

from __future__ import annotations

from _guards import LEGACY_SCREEN_RECORDS_SHA256
from _phase_cli import run_phase_preflight


def main() -> int:
    return run_phase_preflight(
        phase="phase_2_candidate_scoring",
        description=__doc__ or "Phase 2 candidate-scoring preflight",
        default_output_name="03_candidate_scoring.json",
        intended_artifacts=(
            "artifacts/v4/tokenizer/candidate_labels.json",
            "artifacts/v4/candidate_scoring/per_scene.jsonl",
        ),
        integrity_gates=(
            "single_token_labels",
            "unique_candidates",
            "balanced_random_candidate_order",
            "no_cue_valid_cue_facts_only_difference",
        ),
        expected_input_sha256=(LEGACY_SCREEN_RECORDS_SHA256,),
    )


if __name__ == "__main__":
    raise SystemExit(main())
