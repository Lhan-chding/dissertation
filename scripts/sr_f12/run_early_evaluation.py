#!/usr/bin/env python3
"""Generate registered endpoint samples without exposing scores before all training ends."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f12.endpoint import require_single_teacher_gpu, run_endpoint
from sr_f12.protocol import BASELINE


def main():
    p = argparse.ArgumentParser()
    for name in ("run-root", "model-path", "source-manifest"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--model-id", required=True)
    a = p.parse_args()
    require_single_teacher_gpu()
    result = run_endpoint(
        a.run_root,
        a.model_id,
        source_manifest=a.source_manifest,
        code_root=Path(__file__).resolve().parents[2],
        model_path=a.model_path,
        early_sealed=a.model_id != BASELINE,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
