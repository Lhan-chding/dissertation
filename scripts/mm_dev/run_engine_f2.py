#!/usr/bin/env python3
"""Native F2 engine wrapper and fresh-process segment entrypoint."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mm_dev.engine import run_engine, run_segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--segment", choices=("continuous", "first", "resume"))
    parser.add_argument("--mode", choices=("natural", "stress"), default="natural")
    args = parser.parse_args()
    try:
        result = (
            run_segment(args.plan, args.run_root, mode=args.mode, segment=args.segment)
            if args.segment
            else run_engine(args.plan, args.run_root)
        )
    except Exception as error:
        from mm_dev.orchestration import checkpoint_task, fail_task
        from mm_dev.training import LeaseEnding, pin_checkpoint

        if isinstance(error, LeaseEnding):
            if args.segment:
                track = "continuous" if args.segment == "continuous" else "split"
                checkpoint_task(
                    args.run_root,
                    "ENGINE_F2",
                    "PREEMPTION",
                    pin_checkpoint(
                        args.run_root,
                        args.run_root / "engineering" / args.mode / track,
                        "ENGINE_F2",
                    ),
                )
                raise SystemExit(75) from error
            print(json.dumps(dict(status="CHECKPOINTED", reason="PREEMPTION")))
            return
        import os

        failure = (
            args.run_root
            / "orchestration/failures"
            / (os.environ.get("MM_DEV_ATTEMPT_ID", "missing") + ".json")
        )
        if not failure.exists():
            fail_task(args.run_root, "ENGINE_F2", repr(error))
        raise
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "mode",
                    "segment",
                    "final_step",
                    "process_id",
                    "freeze_sha256",
                )
                if key in result
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
