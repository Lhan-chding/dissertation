"""Run one durable scheduling tick or keep the registered F2 matrix progressing."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_dev.orchestration import LIVE, Scheduler, process_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--lease-minutes", type=int, default=4320)
    parser.add_argument("--engine-lease-minutes", type=int, default=720)
    parser.add_argument("--measurement-shards", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 300:
        parser.error("poll-seconds must be between 5 and 300")
    scheduler = Scheduler(
        args.plan,
        args.run_root,
        code_root=args.code_root,
        python=args.python,
        lease_minutes=args.lease_minutes,
        engine_lease_minutes=args.engine_lease_minutes,
        measurement_shards=args.measurement_shards,
    )
    with process_lease(args.run_root / "orchestration/controller_process.lock"):
        return run_controller(scheduler, watch=args.watch, poll_seconds=args.poll_seconds)


def run_controller(scheduler, *, watch, poll_seconds):
    while True:
        state = scheduler.tick()
        print(
            json.dumps(
                {
                    "phase": state["phase"],
                    "accounting": state["accounting"],
                    "technical_blockers": state["technical_blockers"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if state["phase"] == "RELEASED" or not watch:
            return 0
        if state["technical_blockers"] and not any(
            t["status"] in LIVE for t in state["tasks"].values()
        ):
            return 2
        time.sleep(poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
