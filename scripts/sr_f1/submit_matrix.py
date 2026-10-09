#!/usr/bin/env python3
"""Advance the frozen SR-F1 stages, preserving UNKNOWN and all failed attempts."""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sr_f1.orchestration import (
    LIVE,
    Scheduler,
    controller_incarnation,
    process_lease,
    request_controller_requeue,
)


def run_controller(
    scheduler, *, watch=False, poll_seconds=30, lease_boundary=None, controller_identity=None
):
    while True:
        state = scheduler.tick()
        print(
            json.dumps(
                {
                    key: state[key]
                    for key in (
                        "phase",
                        "test_sealed",
                        "technical_blockers",
                        "technical_failed_runs",
                        "capacity",
                    )
                    if key in state
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if state["phase"] in {"RELEASED", "STOPPED"} or not watch:
            return 0
        if lease_boundary is not None and lease_boundary["requested"]:
            receipt = request_controller_requeue(scheduler.root, controller_identity)
            print(json.dumps(receipt, sort_keys=True), flush=True)
            return 0 if receipt["status"] == "REQUEST_ACCEPTED" else 3
        if state["technical_blockers"] and not any(
            task["status"] in LIVE for task in state["tasks"].values()
        ):
            return 2
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--lease-minutes", type=int, default=4320)
    parser.add_argument("--engine-lease-minutes", type=int, default=4320)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--resume-stopped",
        action="store_true",
        help="Explicitly resume STOP_REQUESTED paths after the STOP file is removed",
    )
    parser.add_argument(
        "--resume-repaired-common-start",
        action="store_true",
        help="One-shot authenticated pre-bridge repair transition; submits no jobs",
    )
    parser.add_argument(
        "--resume-repaired-engine",
        action="store_true",
        help="One-shot authenticated ENGINE memory repair transition; submits no jobs",
    )
    parser.add_argument(
        "--resume-repaired-engine-storage",
        action="store_true",
        help="One-shot authenticated ENGINE host-memory/storage repair; submits no jobs",
    )
    parser.add_argument(
        "--resume-repaired-engine-io",
        action="store_true",
        help="One-shot authenticated ENGINE activation I/O repair; submits no jobs",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--controller-requeue-on-lease",
        action="store_true",
        help="Continue this CPU controller JobID on USR1/TERM; requires Slurm --requeue",
    )
    args = parser.parse_args()
    repaired_resumes = sum(
        (
            args.resume_repaired_common_start,
            args.resume_repaired_engine,
            args.resume_repaired_engine_storage,
            args.resume_repaired_engine_io,
        )
    )
    if repaired_resumes and (
        args.watch
        or args.resume_stopped
        or args.controller_requeue_on_lease
        or repaired_resumes > 1
    ):
        parser.error("A repaired resume is one-shot and cannot combine with watch/another resume")
    if not 5 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be between 5 and 60")
    scheduler = Scheduler(
        args.plan,
        args.run_root,
        code_root=args.code_root,
        python=args.python,
        lease_minutes=args.lease_minutes,
        engine_lease_minutes=args.engine_lease_minutes,
    )
    boundary = {"requested": False}
    identity = None
    previous_handlers = {}
    if args.controller_requeue_on_lease:
        if not args.watch:
            parser.error("controller-requeue-on-lease requires --watch")
        identity = controller_incarnation(args.run_root)

        def mark_boundary(signum, frame):
            boundary["requested"] = True

        for signum in (signal.SIGUSR1, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, mark_boundary)
    with process_lease(args.run_root / "orchestration/controller_process.lock"):
        try:
            if args.resume_repaired_common_start:
                print(
                    json.dumps(scheduler.resume_repaired_common_start(), sort_keys=True), flush=True
                )
                return 0
            if args.resume_repaired_engine:
                print(json.dumps(scheduler.resume_repaired_engine(), sort_keys=True), flush=True)
                return 0
            if args.resume_repaired_engine_storage:
                print(
                    json.dumps(scheduler.resume_repaired_engine_storage(), sort_keys=True),
                    flush=True,
                )
                return 0
            if args.resume_repaired_engine_io:
                print(json.dumps(scheduler.resume_repaired_engine_io(), sort_keys=True), flush=True)
                return 0
            if args.resume_stopped:
                scheduler.resume_stopped()
            return run_controller(
                scheduler,
                watch=args.watch,
                poll_seconds=args.poll_seconds,
                lease_boundary=boundary if identity else None,
                controller_identity=identity,
            )
        finally:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)


if __name__ == "__main__":
    sys.exit(main())
