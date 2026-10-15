#!/usr/bin/env python3
"""Export released SR-F2 raw evidence and image identities on a CPU Slurm worker."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sr_f3_ce.prior_export import export_prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Historical bulk copy and image hashing require a CPU Slurm allocation")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = export_prior(args.parent_root, args.output_root)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "operation_completed",
                    "counts",
                    "copied_source_files",
                    "indexed_raw_records",
                    "model_calls",
                    "blocks_new_science",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
