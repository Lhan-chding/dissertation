#!/usr/bin/env python3
"""Bounded replacement admission or persistent replay pool in its bound allocation."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("verify", "status", "submit", "release", "pool", "dispatch")
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pool-id")
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()
    definition = json.loads(args.plan.read_text())
    parent = json.loads(Path(definition["parent_plan_path"]).read_text())
    extension = Path(__file__).resolve().parents[2]
    sys.path[:0] = [
        str(extension / "src"),
        str(Path(parent["extension_root"]) / "src"),
        str(Path(parent["frozen_code_root"]) / "src"),
    ]
    from sr_f2_replay.orchestration import Controller, ReplayQueue, run_dispatcher, run_pool
    from sr_f2_replay.replay import ReplayPlan

    plan = ReplayPlan(args.plan)
    if args.operation == "verify":
        result = dict(verification=plan.verify())
    elif args.operation == "status":
        result = dict(queue=ReplayQueue(plan).snapshot(), allocations=Controller(plan).context())
    elif args.operation == "submit":
        result = Controller(plan).submit_once()
    elif args.operation == "dispatch":
        result = run_dispatcher(plan, poll_seconds=args.interval)
    elif args.operation == "release":
        if not args.pool_id:
            parser.error("release requires --pool-id")
        Controller(plan).release(args.pool_id)
        result = dict(released_pool=args.pool_id)
    else:
        if not args.pool_id:
            parser.error("pool requires --pool-id")
        result = run_pool(plan, args.pool_id)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
