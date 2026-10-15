#!/usr/bin/env python3
"""Execute one registered worker; scientific evaluation stays sealed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sr_f3_ce.orchestration import TASKS, worker_binding
from sr_f3_ce.protocol import record_incident, verify_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-id", choices=TASKS, required=True)
    parser.add_argument("--engine-segment", choices=("continuous", "split"))
    args = parser.parse_args(argv)
    root, task = args.root.resolve(strict=True), TASKS[args.task_id]
    try:
        verify_run(root, science=task["kind"] in {"train", "eval", "release"})
        worker_binding(root, args.task_id)
        if args.engine_segment:
            if args.task_id != "engine--resume":
                raise PermissionError("Engineering subprocess is restricted to recovery gate")
            from sr_f3_ce.runner import execute_training_segment

            result = execute_training_segment(
                root, engine=True, segment=args.engine_segment, stop_step=4
            )
        elif args.task_id == "p0":
            from sr_f3_ce.bootstrap import execute_p0

            result = execute_p0(root)
        elif args.task_id == "engine--resume":
            from sr_f3_ce.engine_recovery import execute_engine_recovery

            result = execute_engine_recovery(root)
        elif task["kind"] == "engine":
            from sr_f3_ce.runner import execute_engine_panel

            result = execute_engine_panel(root, int(args.task_id[-1]))
        elif task["kind"] == "train":
            from sr_f3_ce.runner import execute_train

            result = execute_train(root, task["model_id"])
        elif task["kind"] == "eval":
            from sr_f3_ce.endpoint_runner import execute_endpoint

            result = execute_endpoint(root, task["model_id"])
        elif task["kind"] == "release":
            from sr_f3_ce.endpoint_runner import execute_release

            result = execute_release(root)
        else:
            raise PermissionError("Unknown registered stage")
        print(
            json.dumps(
                dict(
                    task_id=args.task_id,
                    status=result.get("status"),
                    receipt_written=not bool(args.engine_segment),
                )
            )
        )
    except BaseException as error:
        record_incident(root, args.task_id, error, extra=dict(engine_segment=args.engine_segment))
        raise


if __name__ == "__main__":
    main()
