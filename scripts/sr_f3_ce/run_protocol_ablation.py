#!/usr/bin/env python3
"""Freeze, execute or select the fixed zero-training ENGINE protocol ablation."""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE_ROOT = Path(__file__).resolve().parents[2]
if any(path.is_file() and path.suffix in (".pyc", ".pyo") for path in CODE_ROOT.rglob("*")):
    raise PermissionError("Unmanifested bytecode exists in the deployed source tree")
sys.path.insert(0, str(CODE_ROOT / "src"))

from sr_f3_ce.ablation import execute_parent_variant, freeze_ablation, select_protocol  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "run", "select"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--instruction")
    parser.add_argument("--stop-decision")
    parser.add_argument("--block", type=int, choices=(0, 1, 2))
    parser.add_argument("--variant", choices=("P1", "P2", "P3"))
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    if Path(__file__).resolve() != root / "code/scripts/sr_f3_ce/run_protocol_ablation.py":
        raise PermissionError("Use the run-bound deployed ablation entrypoint")
    if args.mode == "freeze":
        if not all((args.previous_root, args.instruction, args.stop_decision)):
            parser.error(
                "freeze requires previous-root, in-root instruction and predecessor stop-decision"
            )
        result = freeze_ablation(
            root,
            previous_root=args.previous_root,
            instruction=args.instruction,
            stop_decision=args.stop_decision,
        )
    elif args.mode == "run":
        if args.block is None or args.variant is None:
            parser.error("run requires block and variant")
        result = execute_parent_variant(root, args.block, args.variant)
    else:
        result = select_protocol(root)
    print(
        json.dumps(
            dict(status=result["status"], selected_revision=result.get("selected_revision")),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
