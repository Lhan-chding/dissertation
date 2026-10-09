#!/usr/bin/env python3
"""Release the fixed CPU analysis only after the complete F2 execution gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mm_dev.analysis import (
    STATES,
    build_artifacts,
    load_cost_accounting,
    load_training_records,
    score_panel,
    write_artifacts,
)
from mm_dev.contract import run_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    # Intentionally no fallback status-string gate and no partial-summary mode.
    from mm_dev.common import load_plan, verify_analysis_readiness
    from mm_dev.evaluation import load_panel
    from mm_dev.orchestration import complete_task, digest, fail_task, worker_lease

    registration = json.loads((args.run_root / "orchestration/REGISTRATION.json").read_text())
    if (
        os.environ.get("MM_DEV_TASK_ID") != "ANALYZE"
        or not os.environ.get("MM_DEV_ATTEMPT_ID")
        or os.environ.get("MM_DEV_REGISTRATION_HASH") != digest(registration)
        or "ANALYZE" not in registration["tasks"]
    ):
        raise PermissionError("ANALYZE worker identity is not registered")
    with worker_lease(args.run_root, "ANALYZE"):
        try:
            receipt = analyze(args, load_plan, verify_analysis_readiness, load_panel)
            paths = [r["path"] for r in receipt["files"]] + ["summary/ANALYSIS_MANIFEST.json"]
            complete_task(args.run_root, "ANALYZE", paths)
        except Exception as exc:
            fail_task(args.run_root, "ANALYZE", str(exc))
            raise
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


def analyze(args, load_plan, verify_analysis_readiness, load_panel):
    """Separately testable pipeline, called only within the registered lease."""

    completeness = verify_analysis_readiness(args.plan, args.run_root)
    plan = load_plan(args.plan)
    probes, evaluations = {}, {}
    for state in STATES:
        probes[state] = score_panel(
            load_panel(args.plan, args.run_root, "PROBE", state), state_id=state, panel="PROBE"
        )
    models = list(STATES) + [r["run_id"] for r in run_matrix() if r["phase"] == "CONTINUE"]
    for state in models:
        evaluations[state] = score_panel(
            load_panel(args.plan, args.run_root, "DEV_EVAL", state),
            state_id=state,
            panel="DEV_EVAL",
        )
    rollouts, updates = load_training_records(args.run_root)
    files = build_artifacts(
        plan=plan,
        evaluations=evaluations,
        probes=probes,
        completeness=completeness,
        training_rollouts=rollouts,
        training_updates=updates,
        cost_accounting=load_cost_accounting(args.run_root),
    )
    # Recheck before publication: inputs and sources may not change mid-analysis.
    if verify_analysis_readiness(args.plan, args.run_root) != completeness:
        raise PermissionError("F2 readiness evidence changed during analysis")
    receipt = write_artifacts(args.run_root, files)
    return receipt


if __name__ == "__main__":
    main()
