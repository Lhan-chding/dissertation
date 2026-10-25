"""Two isolated census workers in one allocation; scientific code stays frozen."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, suppress
from pathlib import Path

from sr_f4.common import now, write
from sr_f4.pool import ACCOUNT, QOS, Slurm, gpu_count


def worker_environments(pair, visible_devices, environ):
    if type(pair) is not int or pair not in (0, 1):
        raise ValueError("Registered shard pair must be 0 or 1")
    devices = visible_devices.split(",")
    if (
        len(devices) != 2
        or len(set(devices)) != 2
        or any(not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", value) for value in devices)
    ):
        raise ValueError("Exactly two distinct allocated GPUs must be visible")
    return [
        dict(
            environ, CUDA_VISIBLE_DEVICES=device, SRF5_SHARD=str(2 * pair + offset), SRF5_SHARDS="4"
        )
        for offset, device in enumerate(devices)
    ]


def _stop_groups(children, grace_seconds):
    """Always address process groups: an exited worker may leave a descendant."""
    for process in children:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and any(p.poll() is None for p in children):
        time.sleep(0.05)
    for process in children:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    for process in children:
        process.wait()


def supervise(commands, environments, logs, *, grace_seconds=10):
    """Wait for both workers, terminating/reaping both groups on any failure."""
    if len(commands) != 2 or len(environments) != 2 or len(logs) != 2:
        raise ValueError("Exactly two worker commands, environments and logs required")
    children, streams = [], ExitStack()
    previous_handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f"Supervisor received signal {signum}")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        for command, environment, path in zip(commands, environments, logs, strict=True):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            stream = streams.enter_context(Path(path).open("a"))  # noqa: SIM115
            children.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        while True:
            statuses = [process.poll() for process in children]
            if any(code is not None and code != 0 for code in statuses):
                raise RuntimeError(f"Census worker failed: {statuses}")
            if all(code == 0 for code in statuses):
                return statuses
            time.sleep(0.2)
    finally:
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            _stop_groups(children, grace_seconds)
        finally:
            streams.close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def run_worker(code, root, python):
    """Same shard and seeds, with a new process per parent to release VRAM."""
    shard = int(os.environ["SRF5_SHARD"])
    if shard not in range(4) or os.environ.get("SRF5_SHARDS") != "4":
        raise ValueError("Unregistered census shard")
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 1:
        raise ValueError("Worker must see exactly one GPU")
    environment = dict(os.environ, PYTHONPATH=f"{code}/src:{code}")
    for parent in ("b0", "b1", "b2"):
        command = [
            str(python),
            "-u",
            "-m",
            "sr_f5.calibration",
            "--root",
            str(root),
            "--parent",
            parent,
            "--shard",
            str(shard),
            "--shards",
            "4",
        ]
        result = subprocess.run(command, cwd=code, env=environment, check=False)
        if result.returncode:
            return result.returncode
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=int, choices=(0, 1))
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(run_worker(args.code, args.root, args.python))
    environments = worker_environments(
        args.pair, os.environ.get("CUDA_VISIBLE_DEVICES", ""), os.environ
    )
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise PermissionError("A real Slurm allocation is required")
    actual = Slurm().show(job_id)
    if (
        actual.get("JobName") != f"LH-SRF5-G2-P{args.pair}"
        or actual.get("QOS") != QOS
        or actual.get("Account") != ACCOUNT
        or max(gpu_count(actual.get("ReqTRES")), gpu_count(actual.get("AllocTRES"))) != 2
    ):
        raise PermissionError("Dual-GPU allocation identity mismatch")
    command = [
        str(args.python),
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--code",
        str(args.code),
        "--root",
        str(args.root),
        "--python",
        str(args.python),
    ]
    logs = [args.root / "slurm" / f"G2-W{e['SRF5_SHARD']}-{job_id}.log" for e in environments]
    path = args.root / "G2/dual_jobs" / f"{job_id}.json"
    receipt = dict(
        at=now(),
        status="RUNNING",
        job_id=job_id,
        pair=args.pair,
        science_code=str(args.code),
        launcher=str(Path(__file__).resolve()),
        shards=[int(e["SRF5_SHARD"]) for e in environments],
        devices=[e["CUDA_VISIBLE_DEVICES"] for e in environments],
        logs=list(map(str, logs)),
        allocation=actual,
    )
    write(path, receipt)
    try:
        supervise([command, command], environments, logs)
    except BaseException as exc:
        receipt.update(status="FAILED", finished_at=now(), error=repr(exc))
        write(path, receipt)
        raise
    receipt.update(status="COMPLETE", finished_at=now())
    write(path, receipt)


if __name__ == "__main__":
    main()
