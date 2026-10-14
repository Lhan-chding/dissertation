#!/usr/bin/env python3
"""Run separately bound recovery control; original scientific worker is unchanged."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("verify", "status", "submit", "release", "pool"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pool-id")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    extension = Path(__file__).resolve().parents[2]
    # Load scientific helpers only from their original registered source root.
    sys.path[:0] = [str(extension / "src"), str(Path(plan["frozen_code_root"]) / "src")]
    from sr_f2_recovery.orchestration import Controller, Recovery, RecoveryQueue, run_pool

    recovery = Recovery(args.plan)
    if args.operation == "verify":
        result = {"plan_sha256": recovery.verify()}
    elif args.operation == "status":
        result = RecoveryQueue(recovery).snapshot()
    elif args.operation == "submit":
        result = Controller(recovery).submit_once()
    elif args.operation == "release":
        if not args.pool_id:
            parser.error("release requires --pool-id")
        Controller(recovery).release(args.pool_id)
        result = {"released_pool": args.pool_id}
    else:
        if not args.pool_id:
            parser.error("pool requires --pool-id")
        result = run_pool(recovery, args.pool_id)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
