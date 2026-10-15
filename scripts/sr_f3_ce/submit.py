#!/usr/bin/env python3
"""Submit/observe SR-F3-CE held jobs; never retry an uncertain submission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sr_f3_ce.orchestration import KINDS, Controller, run_job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--kind", required=True,
                        choices=(*KINDS, "controller", "tick", "status", "resume", "run-job"))
    parser.add_argument("--job-key", help="Fixed allocation identity for run-job or resume")
    parser.add_argument("--pool-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--cpu-partition", default="cluster02")
    parser.add_argument("--cpu-qos", help="Explicit currently authorized CPU QoS; no default")
    parser.add_argument("--cpu-node", default="cpu-2")
    args = parser.parse_args(argv)
    if args.kind == "run-job":
        if not args.job_key:
            parser.error("run-job requires --job-key")
        result = run_job(args.root, args.job_key)
    else:
        if args.kind == "resume" and not args.job_key:
            parser.error("resume requires the previous --job-key")
        if args.job_key and args.kind != "resume":
            parser.error("--job-key is only valid with run-job or resume")
        controller = Controller(
            args.root, args.code_root, python=args.python, pool_size=args.pool_size,
            cpu_partition=args.cpu_partition, cpu_qos=args.cpu_qos, cpu_node=args.cpu_node,
        )
        result = controller.resume(args.job_key) if args.kind == "resume" else (
            controller.status() if args.kind == "status" else (
            controller.tick() if args.kind == "tick" else controller.submit(args.kind)
        ))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
