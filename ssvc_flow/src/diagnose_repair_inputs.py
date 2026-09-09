"""Build the fixed R2 diagnostic panel and prompt variants without generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import load_split
from .next_stage_common import dry_run_plan, load_yaml, stage_status
from .r2_inputs import (
    CONDITIONS,
    build_requests,
    select_long_panel,
    select_panel,
    variant_prompt,
    write_panel_artifacts,
)

__all__ = [
    "CONDITIONS",
    "build_panel",
    "build_requests",
    "select_long_panel",
    "select_panel",
    "variant_prompt",
]


def build_panel(data_root, out, base_scenes=72):
    if base_scenes != 72:
        raise ValueError("R2 fixed panel requires exactly 72 base scenes")
    scenes = select_panel(load_split(data_root, "calibration", purpose="R2 fixed diagnostic panel"))
    manifest = write_panel_artifacts(scenes, out)
    stage_status(
        out,
        "R2",
        "BLOCKED",
        "CPU_AUDIT",
        {
            "panel_rows": len(manifest["rows"]),
            "request_count": manifest["request_count"],
            "reason": "REAL_CUDA_INFERENCE required for rollouts",
        },
    )
    return len(manifest["rows"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["R2"], default="R2")
    parser.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--out", type=Path, default=Path("runs/NEXT_20260909/R2"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "phase": "R2",
                    **dry_run_plan(config),
                    "conditions": list(CONDITIONS),
                    "execution_kind": "CPU_AUDIT",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    build_panel(args.data_root, args.out, config.get("R2", {}).get("base_scenes", 72))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
