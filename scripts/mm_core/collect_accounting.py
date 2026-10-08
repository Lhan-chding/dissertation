"""Read exact registered Slurm IDs; release concurrency only on verified terminal state."""

import argparse
import getpass
import re
import subprocess
from pathlib import Path

from mm_core.allocations import mark_terminal
from mm_core.execution import atomic_json, read_json, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    owner = getpass.getuser()
    queue = subprocess.run(
        ["squeue", "-h", "-u", owner, "-o", "%i"], text=True, capture_output=True, timeout=30
    )
    queue.check_returncode()
    current = set(queue.stdout.split())
    summary = []
    for path in sorted((root / "accounting/allocations").glob("*.json")):
        record = read_json(path)
        if record["status"] == "TERMINAL_VERIFIED":
            summary.append({"job_id": record["job_id"], "status": "TERMINAL_VERIFIED"})
            continue
        job_id = record["job_id"]
        if job_id is None:
            summary.append(
                {"allocation_key": record["allocation_key"], "status": "UNBOUND_UNKNOWN"}
            )
            continue
        command = [
            "sacct",
            "-n",
            "-P",
            "-X",
            "-j",
            job_id,
            "-o",
            "JobIDRaw,State,ElapsedRaw,Start,End,AllocTRES%200,User,JobName%100",
        ]
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
        lines = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
        exact = [line for line in lines if line[0] == job_id]
        state = "UNKNOWN"
        if result.returncode == 0 and len(exact) == 1 and job_id not in current:
            fields = exact[0]
            state = fields[1].split()[0]
            terminal = state in {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "PREEMPTED",
            }
            gpu = re.search(r"gres/gpu:pro6000=(\d+)", fields[5])
            if terminal and fields[2].isdigit() and fields[4] != "Unknown":
                if fields[6] != owner or fields[7] != "mmcore-" + record["allocation_key"]:
                    raise PermissionError("Scheduler owner or registered job name changed")
                if not gpu or int(gpu.group(1)) != record["gpus"]:
                    raise PermissionError("Allocated GPU type/count does not match budget")
                receipt = root / f"accounting/scheduler/{job_id}.json"
                atomic_json(
                    receipt,
                    {
                        "time": utc_now(),
                        "job_id": job_id,
                        "state": state,
                        "squeue_absent": True,
                        "elapsed_seconds": int(fields[2]),
                        "owner": owner,
                        "gpus": int(gpu.group(1)),
                        "start": fields[3],
                        "end": fields[4],
                        "sacct_stdout": result.stdout,
                        "sacct_stderr": result.stderr,
                        "squeue_stdout": queue.stdout,
                        "command": command,
                    },
                )
                mark_terminal(root, record["allocation_key"], receipt)
                state = "TERMINAL_VERIFIED"
        summary.append({"job_id": job_id, "status": state})
    atomic_json(root / "accounting/SCHEDULER_STATUS.json", {"time": utc_now(), "jobs": summary})
    print(summary)


if __name__ == "__main__":
    main()
