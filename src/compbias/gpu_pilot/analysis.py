"""Conservative CPU analysis gate for completed GPU pilot artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import load_pilot_paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = load_pilot_paths(args.paths)
        required = {
            "calibration": paths.trajectories / "natural" / "calibration_records.summary.json",
            "pilot_a": paths.outputs / "pilot_a" / "final_adapter",
            "pilot_b": paths.outputs / "pilot_b_lm_only" / "final_adapter",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        report = {
            "schema_version": 1,
            "artifact_type": "gpu_pilot_analysis_readiness",
            "ready": not missing,
            "missing": missing,
            "claims_permitted": [] if missing else ["operational_pilot_comparison_only"],
            "claims_forbidden": [
                "unique_perception_reasoning_boundary",
                "visual_acquisition_improved",
                "synthetic_equals_natural",
            ],
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if not missing else 2
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
