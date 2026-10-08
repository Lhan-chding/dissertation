"""Explicit SER-J23 execution gates. No command silently enables model execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-legacy")
    audit.add_argument("--design", required=True)
    audit.add_argument("--legacy-run")
    audit.add_argument("--run", required=True)
    audit.add_argument("--metadata-only", action="store_true")
    build = sub.add_parser("build-training")
    build.add_argument("--design", required=True)
    build.add_argument("--legacy-run", required=True)
    build.add_argument("--run", required=True)
    for name in ("cpu-check", "register", "integrity", "release-and-analyze", "status"):
        item = sub.add_parser(name)
        item.add_argument("--run", required=True)
    bridge = sub.add_parser("bridge")
    bridge.add_argument("--run", required=True)
    bridge.add_argument("--allow-gpu", action="store_true")
    validation = sub.add_parser("validate")
    validation.add_argument("--run", required=True)
    validation.add_argument("--source-package", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--run", required=True)
    freeze.add_argument("--exclusion-audit", required=True)
    freeze.add_argument("--validation-receipt", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--run", required=True)
    worker.add_argument("--job-id")
    worker.add_argument("--worker")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args(argv)
    from . import control

    if args.command == "audit-legacy":
        from .legacy import audit_legacy

        design = control.bind_design(args.design, args.run)
        result = audit_legacy(
            args.design,
            args.legacy_run or design["historical_paths"]["run"],
            args.run,
            verify_weights=not args.metadata_only,
        )
    elif args.command == "build-training":
        result = control.build_training(args.design, args.legacy_run, args.run)
    elif args.command == "cpu-check":
        result = control.cpu_check(args.run)
        result = {
            k: v for k, v in result.items() if k not in ("code_sha256", "training", "development")
        }
    elif args.command == "bridge":
        from .bridge import run_bridge

        result = run_bridge(args.run, allow_gpu=args.allow_gpu)
    elif args.command == "validate":
        from .validation import validate

        result = validate(args.run, args.source_package)
        result = {
            k: v for k, v in result.items() if k not in ("code_sha256", "tests_sha256", "commands")
        }
    elif args.command == "freeze":
        result = control.freeze(args.run, args.exclusion_audit, args.validation_receipt)
    elif args.command == "register":
        from .queue import build_queue

        result = build_queue(args.run)
    elif args.command == "status":
        from collections import Counter

        from .queue import registered_queue

        queue = registered_queue(args.run)
        try:
            result = {
                "counts": dict(Counter(r["status"] for r in queue.rows())),
                "released": (Path(args.run) / "RELEASE_RECEIPT.json").exists(),
            }
        finally:
            queue.db.close()
    elif args.command == "worker":
        from .worker import worker

        result = worker(
            args.run, args.worker, allow_gpu=args.allow_gpu, once=args.once, job_id=args.job_id
        )
    elif args.command == "integrity":
        from .evaluate import integrity

        result = integrity(args.run)[0]
    else:
        from .reports import release_and_analyze

        result = release_and_analyze(args.run)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False))
    if isinstance(result, dict) and str(result.get("status", "")).startswith(("BLOCKED", "FAIL")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
