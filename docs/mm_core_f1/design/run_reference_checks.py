#!/usr/bin/env python3
"""Run offline tests and write an explicitly scoped validation receipt."""
from __future__ import annotations
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from reference.contracts import planned_counts

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    receipt = {
        "scope": "CPU constructed examples and contract/budget checks only",
        "model_calls": 0, "optimizer_updates": 0, "server_jobs_modified": 0,
        "historical_rollouts_rescored": 0, "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "passed": result.wasSuccessful(),
        "plan_budget_with_bridge": planned_counts(True),
        "plan_budget_without_bridge": planned_counts(False),
    }
    path = ROOT / "validation"
    path.mkdir(exist_ok=True)
    (path / "CPU_TEST_LOG.txt").write_text(output.getvalue(), encoding="utf-8")
    (path / "CPU_VALIDATION.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    if not result.wasSuccessful():
        print(output.getvalue(), file=sys.stderr)
        sys.exit(1)
