#!/usr/bin/env python3
"""Check real native processor/masks in a CPU allocation before model outputs."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f3_ce.native_audit import audit_native_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        parser.error("Native full-corpus audit requires a CPU allocation")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    receipt = audit_native_inputs(args.root)
    print(
        json.dumps(
            {k: receipt[k] for k in ("status", "model_calls", "checked_images", "protocol_routes")}
        )
    )
