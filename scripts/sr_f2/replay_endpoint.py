#!/usr/bin/env python3
"""Run one authorized endpoint replay, then the original frozen endpoint worker."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--model-id")
    identity.add_argument("--task-id")
    args = parser.parse_args()
    raw = json.loads(args.plan.read_text())
    parent = json.loads(Path(raw["parent_plan_path"]).read_text())
    extension = Path(__file__).resolve().parents[2]
    sys.path[:0] = [
        str(extension / "src"),
        str(Path(parent["extension_root"]) / "src"),
        str(Path(parent["frozen_code_root"]) / "src"),
    ]
    from sr_f2_replay.replay import ReplayPlan, run_endpoint

    plan = ReplayPlan(args.plan)
    key = args.task_id or "endpoint--" + args.model_id
    if key not in plan.tasks:
        parser.error("Task is outside the five authorized interrupted endpoints")
    result = run_endpoint(plan, plan.tasks[key]["task"])
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
