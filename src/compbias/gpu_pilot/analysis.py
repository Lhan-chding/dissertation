"""Conservative post-GPU claim gate (authentication intentionally pending)."""

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
        load_pilot_paths(args.paths)
        report = {
            "schema_version": 1,
            "artifact_type": "gpu_pilot_analysis_readiness",
            "ready": False,
            "missing": ["authenticated_post_gpu_analysis_gate_not_implemented"],
            "claims_permitted": [],
            "claims_forbidden": [
                "operational_pilot_comparison_only",
                "unique_perception_reasoning_boundary",
                "visual_acquisition_improved",
                "synthetic_equals_natural",
            ],
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
