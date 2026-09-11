"""Compatibility entry point for the complete R4 two-arm training protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .next_stage_common import dry_run_plan, load_yaml


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    p.add_argument(
        "--arm",
        choices=["X_BASE", "X_VALID"],
        help="Legacy dry-run selection; actual R4 runs both arms",
    )
    p.add_argument("--allow-training", action="store_true")
    p.add_argument("--out", type=Path, default=Path("runs/NEXT_20260909/R4"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--continuation-parent", type=Path)
    p.add_argument("--continuation-decision", type=Path)
    for name in ("data-root", "r0-dir", "r1-run", "supplement-dir", "r2-dir", "r3-dir"):
        p.add_argument("--" + name, type=Path)
    a = p.parse_args(argv)
    if (a.continuation_parent is None) != (a.continuation_decision is None):
        p.error("--continuation-parent and --continuation-decision must be supplied together")
    continuation = {}
    if a.continuation_parent is not None:
        if not a.allow_training and not a.dry_run:
            p.error("R4 continuation requires explicit --allow-training")
        continuation = {
            "continuation_parent": a.continuation_parent,
            "continuation_decision": a.continuation_decision,
        }
    c = load_yaml(a.config)
    if a.dry_run:
        print(
            json.dumps(
                {
                    "phase": "R4",
                    "arm": a.arm,
                    **dry_run_plan(c),
                    "execution_kind": "CPU_AUDIT",
                    "training_started": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    from .r4_runtime import _finish, run_r4

    if a.arm:
        if a.out.exists() and any(a.out.iterdir()):
            p.error("Single-arm actual execution is unsupported; existing evidence preserved")
        result = _finish(
            a.out,
            "BLOCKED",
            "CPU_AUDIT",
            {
                "training_started": False,
                "reason": (
                    "The complete R4 protocol requires both arms; omit --arm. "
                    "The --arm option remains available for dry-run."
                ),
            },
        )
    elif not a.allow_training:
        result = run_r4(
            c, a.data_root, a.out, a.r0_dir, a.r1_run, a.supplement_dir, a.r2_dir, a.r3_dir
        )
    else:
        required = ("data_root", "r0_dir", "r1_run", "supplement_dir", "r2_dir", "r3_dir")
        missing = ["--" + name.replace("_", "-") for name in required if getattr(a, name) is None]
        if missing:
            p.error("Real R4 requires " + ", ".join(missing))
        result = run_r4(
            c,
            a.data_root,
            a.out,
            a.r0_dir,
            a.r1_run,
            a.supplement_dir,
            a.r2_dir,
            a.r3_dir,
            allow_training=True,
            resume=a.resume,
            **continuation,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("PASS", "COMPLETED_WITH_DIAGNOSTIC_WARNINGS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
