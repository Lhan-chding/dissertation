#!/usr/bin/env python3
"""Run the independently registered training-priority endpoint controller."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sr_f12.early_endpoint_orchestration import EarlyEndpointController
from sr_f12.endpoint_orchestration import run_endpoint_controller


def main():
    p = argparse.ArgumentParser()
    for name in ("run-root", "model-path", "source-manifest"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    c = EarlyEndpointController(
        a.run_root,
        Path(__file__).resolve().parents[2],
        a.model_path,
        a.source_manifest,
        a.source_commit,
        python=a.python,
    )
    result = run_endpoint_controller(c, once=a.once)
    if result["stage"].startswith("BLOCKED_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
