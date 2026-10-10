#!/usr/bin/env python3
"""One SR-F1.2 model's complete endpoint evaluation, after the twelve-path gate."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f12.endpoint import require_single_teacher_gpu, run_endpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source-manifest", required=True, type=Path)
    args = parser.parse_args()
    require_single_teacher_gpu()
    result = run_endpoint(
        args.run_root,
        args.model_id,
        source_manifest=args.source_manifest,
        code_root=Path(__file__).resolve().parents[2],
        model_path=args.model_path,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
