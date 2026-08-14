#!/usr/bin/env python3
"""Hash and validate the already-completed v0.3 calibration evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from compbias.recoverability.evidence_capture import capture_v03_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-pilot", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pilot-data-log", type=Path, required=True)
    parser.add_argument("--calibration-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = capture_v03_evidence(
        negative_pilot_path=args.negative_pilot,
        records_path=args.records,
        summary_path=args.summary,
        pilot_data_log_path=args.pilot_data_log,
        calibration_log_path=args.calibration_log,
        output_path=args.output,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
