#!/usr/bin/env python3
"""A single-GPU SR-F1.2 science worker or bounded technical segment."""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from sr_f12.protocol import validate_config
from sr_f12.runner import PILOT_SEED, execute_path, preflight, train_scientific, write_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--phase", choices=("preflight", "pilot", "train"), default="train")
    parser.add_argument("--run-id")
    parser.add_argument("--stop-step", type=int, choices=(4, 8))
    parser.add_argument("--kill-after-commit", action="store_true")
    args = parser.parse_args()
    if args.kill_after_commit and (args.phase != "pilot" or args.stop_step != 4):
        parser.error("Forced termination is restricted to technical committed step four")
    plan = validate_config(args.plan)
    if args.phase == "preflight":
        result = preflight(plan, args.run_root)
    elif args.phase == "pilot":
        if (
            args.run_id not in ("SRF1_2_TECHNICAL_R0", "SRF1_2_TECHNICAL_R1")
            or args.stop_step is None
        ):
            parser.error("Pilot requires registered round and stop step")
        run = dict(
            model_id=args.run_id, arm="GATE", seed=PILOT_SEED, updates=8, technical_only=True
        )
        result = execute_path(plan, args.run_root, run, technical=True, stop_step=args.stop_step)
    else:
        if not args.run_id or args.stop_step is not None:
            parser.error("Science requires model id and all 96 registered steps")
        result = train_scientific(plan, args.run_root, args.run_id)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if args.kill_after_commit:
        write_once(
            args.run_root / "technical" / args.run_id / "KILL_AFTER_STEP4.json",
            dict(step=4, pid=os.getpid(), signal="SIGKILL", checkpoint=result["checkpoint"]),
        )
        sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    main()
