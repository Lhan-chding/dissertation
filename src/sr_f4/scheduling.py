"""Migrate only untouched evaluation work to a pipelined execution schedule."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from . import pool

AMENDMENT = "SCHEDULING_AMENDMENT_PIPELINED_EVAL.json"
STATE_BACKUP = "pool/STATE.before-pipelined-eval.json"
TASKS_BACKUP = "TASKS.before-pipelined-eval.json"


def _legacy_tasks(tasks):
    """Recover the one supported predecessor, without changing scientific tasks."""
    train = [task["task_id"] for task in tasks if task["kind"] == "train"]
    regression = [
        task["task_id"]
        for task in tasks
        if task["kind"] == "eval" and task.get("phase") == "regression"
    ]
    confirmation = [
        task for task in tasks if task["kind"] == "eval" and task.get("phase") == "confirm"
    ]
    if len(tasks) != 54 or len(train) != 6 or len(regression) != 18 or len(confirmation) != 18:
        raise ValueError("Expected the registered 54-task pipelined evaluation schedule")
    legacy = []
    for original in tasks:
        if original in confirmation:
            continue
        task = copy.deepcopy(original)
        if task["kind"] == "eval":
            task.pop("phase")
            task.pop("evaluation_task_id", None)
            task["dependencies"] = [*train, "B3-replay"]
        elif task["kind"] == "report":
            task["dependencies"] = regression
        legacy.append(task)
    return legacy


def _specs(state):
    return {key: row["spec"] for key, row in state["tasks"].items()}


def migrate_evaluation_schedule(root):
    """Preserve active work; publish the registry before unlocking new claims.

    The prepared receipt and immutable backups allow recovery if publication
    stops after TASKS.json is replaced but before STATE.json is replaced.
    Existing model processes keep their original, unchanged training spec.
    """
    from .prepare import registered_tasks

    root = Path(root).resolve()
    tasks = registered_tasks()
    new_specs = pool._validate_tasks(tasks)
    legacy = _legacy_tasks(tasks)
    legacy_specs = pool._validate_tasks(legacy)
    receipt_path = root / AMENDMENT
    with pool._locked(root) as state:
        if state is None:
            raise FileNotFoundError("Existing pool state is required")
        current = _specs(state)
        registry = pool._read(root / "TASKS.json")
        if current == new_specs:
            if registry != tasks:
                raise ValueError("Pipelined state and TASKS.json disagree")
            receipt = pool._read(receipt_path) if receipt_path.exists() else {}
            if receipt.get("status") == "PREPARED":
                receipt.update(status="APPLIED", applied_at=pool.now())
                pool._write(receipt_path, receipt)
            return dict(status="ALREADY_PIPELINED", task_count=len(tasks), amendment=receipt)
        if current != legacy_specs:
            raise ValueError("Pool differs from the supported original evaluation schedule")
        for key, row in state["tasks"].items():
            if row["spec"]["kind"] in ("eval", "report", "package"):
                if row["status"] != "PENDING" or row["attempts"] or row.get("owner"):
                    raise ValueError(f"Cannot migrate started task {key}")
                if (root / "tasks" / key / "COMPLETE.json").exists():
                    raise ValueError(f"Cannot migrate task with output receipt {key}")
        if (root / "CONFIRM_OPENED.json").exists():
            raise ValueError("Confirmation has already been opened")
        if any((root / "EVAL").rglob("groups/*")):
            raise ValueError("Evaluation group outputs already exist")
        recovering = registry == tasks
        if registry != legacy and not recovering:
            raise ValueError("TASKS.json differs from both supported schedules")
        if recovering:
            if not all((root / name).exists() for name in (AMENDMENT, STATE_BACKUP, TASKS_BACKUP)):
                raise ValueError("New registry without migration backups cannot be recovered")
            if pool._read(receipt_path).get("status") != "PREPARED":
                raise ValueError("Interrupted registry publication lacks a prepared amendment")
        for name, expected in ((TASKS_BACKUP, legacy), (STATE_BACKUP, state)):
            path = root / name
            if path.exists():
                saved = pool._read(path)
                valid = saved == legacy if name == TASKS_BACKUP else _specs(saved) == legacy_specs
                if not valid:
                    raise ValueError(f"Existing migration backup differs: {name}")
            else:
                pool._write(path, expected)
        receipt = (
            pool._read(receipt_path)
            if receipt_path.exists()
            else dict(
                status="PREPARED",
                prepared_at=pool.now(),
                amendment="PIPELINED_REGRESSION_EVALUATION",
                execution_order_only=True,
                training_specs_changed=False,
                seeds_changed=False,
                answer_budgets_changed=False,
                confirmation_gate_changed=False,
                old_task_count=len(legacy),
                new_task_count=len(tasks),
                backups=[STATE_BACKUP, TASKS_BACKUP],
            )
        )
        if receipt.get("status") != "PREPARED":
            raise ValueError("Unexpected existing scheduling amendment")
        pool._write(receipt_path, receipt)
        for key, spec in new_specs.items():
            if key not in state["tasks"]:
                state["tasks"][key] = dict(spec=spec, status="PENDING", attempts=[])
            elif state["tasks"][key]["spec"] != spec:
                state["tasks"][key]["spec"] = spec
        # Worker claims use this same lock; their fresh child reads TASKS.json.
        pool._write(root / "TASKS.json", tasks)
        pool._save(root, state)
        receipt.update(status="APPLIED", applied_at=pool.now())
        pool._write(receipt_path, receipt)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate_evaluation_schedule(args.root), ensure_ascii=False))
