"""R4 two-arm training entry point with an explicit server authorization gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .next_stage_common import dry_run_plan, load_yaml, stage_status


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/next_stage.yaml"))
    p.add_argument("--arm", choices=["X_BASE", "X_VALID"], required=True)
    p.add_argument("--allow-training", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
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
    out = a.out or Path("runs/NEXT_20260909/R4") / a.arm
    out.mkdir(parents=True, exist_ok=True)
    config_allow = bool(c.get("R4", {}).get("allow_training", False))
    details = {
        "arm": a.arm,
        "allow_training_flag": a.allow_training,
        "config_allow_training": config_allow,
        "training_started": False,
    }
    if not (a.allow_training and config_allow):
        details["reason"] = "requires explicit --allow-training and config R4.allow_training=true"
    for name, payload in {
        "two_arm_training_config.json": details,
        "checkpoint_manifest.json": {"status": "NOT_STARTED"},
        "learning_curves.csv": {"status": "NOT_STARTED"},
        "endpoint_metrics.json": {"status": "NOT_STARTED"},
        "N_L_OOD_effects.csv": {"status": "NOT_STARTED"},
        "pilot_report.md": "# R4\n\nTraining is disabled by default.\n",
    }.items():
        path = out / name
        path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    stage_status(out, "R4", "BLOCKED", "CPU_AUDIT", details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
