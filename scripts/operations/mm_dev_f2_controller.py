#!/usr/bin/env python3
"""CPU Slurm lease supervisor for the already frozen F2 scheduler.

This operational wrapper does not change scientific source, inputs or settings.
Its own source SHA is checked against a separately deployed receipt. One dependent
successor protects against controller-node failure and lease expiry; a successor
seeing release or an inactive technical block exits without creating another.
"""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/mm_dev_f2_qwen35_20261009_teacherqos")
CODE = ROOT / "code"
PYTHON = "/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python"


def should_finish(state, live):
    return state.get("phase") == "RELEASED" or (
        bool(state.get("technical_blockers"))
        and not any(t["status"] in live for t in state["tasks"].values())
    )


def wait_for_successor_slot(backend, permission, stop, report):
    """Reserve the next CPU lease before science can consume its submission slot."""
    while not stop.is_set():
        try:
            capacity = backend.submission_capacity(permission)
            maximum = capacity["max_submit_jobs_per_user"]
            if maximum is None or len(capacity["queued_job_ids"]) < maximum:
                report({"status": "SUCCESSOR_SLOT_AVAILABLE", "capacity": capacity})
                return True
            report({"status": "WAITING_FOR_SUCCESSOR_SLOT", "capacity": capacity})
        except Exception as error:
            report({"status": "SUCCESSOR_CAPACITY_UNKNOWN", "error": repr(error)})
        stop.wait(30)
    return False


def main():
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        raise PermissionError("Controller requires its own CPU Slurm allocation")
    config = json.loads((ROOT / "ops/CONTROLLER_CODE.json").read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != config["python_sha256"]:
        raise PermissionError("Controller source identity differs")
    wrapper = ROOT / "ops/controller.sh"
    if hashlib.sha256(wrapper.read_bytes()).hexdigest() != config["shell_sha256"]:
        raise PermissionError("Controller wrapper identity differs")
    sys.path.insert(0, str(CODE / "src"))
    from mm_dev.orchestration import LIVE, Scheduler, atomic_json, process_lease

    stop = threading.Event()
    for number in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(number, lambda _signum, _frame: stop.set())
    scheduler = Scheduler(
        CODE / "docs/mm_dev_f2/design/config/MM_DEV_F2.json",
        ROOT,
        code_root=CODE,
        python=Path(PYTHON),
        lease_minutes=4320,
        engine_lease_minutes=720,
        measurement_shards=1,
    )
    with process_lease(ROOT / "orchestration/controller_process.lock"):
        # Loading verifies the journal and immutable registration without submitting
        # science. A fresh successor must take its slot before the first tick.
        with process_lease(scheduler.folder / "controller.lock"):
            scheduler._load()
        state = scheduler.state
        if should_finish(state, LIVE):
            print(
                json.dumps(
                    {
                        "controller_job": job_id,
                        "phase": state["phase"],
                        "technical_blockers": state["technical_blockers"],
                    }
                ),
                flush=True,
            )
            return 0 if state["phase"] == "RELEASED" else 2
        if not wait_for_successor_slot(
            scheduler.backend,
            scheduler.registration["permission"],
            stop,
            lambda receipt: print(json.dumps(receipt), flush=True),
        ):
            return 0
        command = [
            "sbatch",
            "--parsable",
            "--no-requeue",
            "--account=rose",
            "--qos=soujanya-poria-startfund-2026-03",
            "--partition=cluster02",
            "--cpus-per-task=2",
            "--time=4320",
            "--signal=B:USR1@900",
            "--job-name=mmdev-f2-controller",
            "--dependency=afterany:" + job_id,
            "--chdir=" + str(ROOT),
            "--output=" + str(ROOT / "logs/controller_%j.log"),
            str(wrapper),
        ]
        receipt = ROOT / "ops/leases" / (job_id + ".json")
        atomic_json(
            receipt,
            {"status": "SUBMITTING_SUCCESSOR", "command": command, "parent_job_id": job_id},
            exclusive=True,
        )
        result = subprocess.run(command, text=True, capture_output=True, timeout=45)
        atomic_json(
            receipt.with_name(job_id + "_result.json"),
            {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
            },
            exclusive=True,
        )
        result.check_returncode()
        successor = result.stdout.strip().split(";")[0]
        if not re.fullmatch(r"[0-9]+", successor):
            raise RuntimeError("Unknown successor submission; do not resubmit blindly")
        print(json.dumps({"controller_job": job_id, "successor_job": successor}), flush=True)
        while not stop.is_set():
            state = scheduler.tick()
            print(
                json.dumps(
                    {
                        "controller_job": job_id,
                        "phase": state["phase"],
                        "accounting": state["accounting"],
                        "technical_blockers": state["technical_blockers"],
                    }
                ),
                flush=True,
            )
            if should_finish(state, LIVE):
                return 0 if state["phase"] == "RELEASED" else 2
            stop.wait(30)
        print(
            json.dumps(
                {"controller_job": job_id, "status": "LEASE_HANDOFF", "successor_job": successor}
            ),
            flush=True,
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
