"""Prepare, audit, execute and summarize the registered frozen probe matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import PHASES, atomic_json, prepare, read_json


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    command = sub.add_parser(
        "prepare", help="Validate precompiled prompts against original prepared data; no model load"
    )
    command.add_argument("--protocol", required=True)
    command.add_argument("--prepared", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--runtime")
    command = sub.add_parser(
        "check-checkpoints",
        help="Read original COMMIT pointers; never train or instantiate a model",
    )
    command.add_argument("--run", required=True)
    command.add_argument("--path-mapping", action="append", default=[], metavar="OLD=NEW")
    command.add_argument(
        "--path-mappings", help="JSON object with explicit original-root to mounted-root mappings"
    )
    for name in ("smoke", "worker"):
        command = sub.add_parser(name)
        command.add_argument("--run", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--runtime")
        command.add_argument("--allow-gpu", action="store_true")
        command.add_argument("--resume", action="store_true")
        command.add_argument(
            "--lane-id",
            help="Stable allocated physical worker lane; at most five lanes receive smoke",
        )
        if name == "worker":
            command.add_argument("--phase", choices=PHASES)
            command.add_argument(
                "--with-smoke",
                action="store_true",
                help="Run first-lane smoke and main matrix with one checkpoint load",
            )
    command = sub.add_parser(
        "summarize", help="Analyze committed observations without loading a model"
    )
    command.add_argument("--run", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.protocol, args.prepared, args.out, args.runtime)
    elif args.command == "check-checkpoints":
        from .checkpoint_catalog import check_checkpoints

        mappings = read_json(args.path_mappings) if args.path_mappings else {}
        if not isinstance(mappings, dict):
            raise ValueError("--path-mappings must contain a JSON object")
        for value in args.path_mapping:
            if "=" not in value:
                raise ValueError("Path mappings require OLD=NEW")
            original, mounted = value.split("=", 1)
            if not original or not mounted:
                raise ValueError("Empty path mapping is forbidden")
            mappings[original] = mounted
        result = check_checkpoints(Path(args.run) / "checkpoints.json", path_mappings=mappings)
        atomic_json(Path(args.run) / "CHECKPOINT_AVAILABILITY.json", result)
    elif args.command in {"smoke", "worker"}:
        from .run import run_worker

        result = run_worker(
            args.run,
            args.checkpoint,
            runtime_path=args.runtime,
            allow_gpu=args.allow_gpu,
            resume=args.resume,
            phase=getattr(args, "phase", None),
            smoke=args.command == "smoke",
            with_smoke=getattr(args, "with_smoke", False),
            lane_id=args.lane_id,
        )
    elif args.command == "summarize":
        # pandas/scipy/pyarrow are reporting dependencies, never worker startup dependencies.
        from .statistics import summarize

        result = summarize(args.run)
    else:
        raise AssertionError(args.command)
    compact = {
        key: result[key]
        for key in ("status", "checkpoint", "role", "phase", "run", "identity", "planned_outputs")
        if key in result
    }
    print(json.dumps(compact or {"command": args.command, "completed": True}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
