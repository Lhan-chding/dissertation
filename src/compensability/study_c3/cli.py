"""Fail-closed command registry for Study C3 stages 00--09."""

from __future__ import annotations

import argparse


def run_stage(stage: int) -> int:
    if stage not in range(10):
        raise ValueError(f"unregistered Study C3 stage: {stage}")
    parser = argparse.ArgumentParser(description=f"Study C3 registered stage {stage:02d}")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute and not arguments.preflight_only:
        print(f"BLOCKED: Study C3 stage {stage:02d} requires --execute or --preflight-only")
        return 2
    print(f"BLOCKED: Study C3 stage {stage:02d} runtime is not yet implemented")
    return 2


__all__ = ["run_stage"]
