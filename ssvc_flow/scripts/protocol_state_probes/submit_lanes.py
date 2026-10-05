"""Bounded, once-only submission of five fixed frozen-inference lanes.

A durable intent precedes sbatch. Ambiguous submissions require reconciliation;
this script never retries them and never submits training or changes the grid.
"""

import argparse
import fcntl
import json
import subprocess
from pathlib import Path

LANES = [["S96"], ["S32"], ["R4_128"], ["DIRECT_128"], ["REP96", "REP32"]]


def main():
    p = argparse.ArgumentParser()
    for key in ["run", "project", "python", "runtime", "cache", "partition", "account", "qos"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--lanes", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    a = p.parse_args()
    if len(set(a.lanes)) != len(a.lanes) or not set(a.lanes) <= set(range(1, 6)):
        raise ValueError("Choose unique registered lane numbers 1 through 5")
    run = Path(a.run).resolve()
    control = run / "scheduler"
    control.mkdir(parents=True, exist_ok=True)
    log = run / "logs"
    log.mkdir(exist_ok=True)
    script = Path(a.project).resolve() / "scripts/protocol_state_probes/launch_lane.sbatch"
    with (control / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        commands = []
        for index, checkpoints in enumerate(LANES, 1):
            if index not in a.lanes:
                continue
            lane = f"lane-{index}"
            command = [
                "sbatch",
                "--parsable",
                "--partition=" + a.partition,
                "--account=" + a.account,
                "--qos=" + a.qos,
                "--gres=gpu:pro6000:1",
                "--cpus-per-task=4",
                "--mem=33G",
                "--ntasks=1",
                "--nodes=1",
                "--time=3-00:00:00",
                "--no-requeue",
                "--job-name=psp-v1-" + lane,
                "--output=" + str(log / (lane + "-%j.out")),
                "--error=" + str(log / (lane + "-%j.err")),
                "--chdir=" + str(Path(a.project).resolve()),
                str(script),
                a.python,
                a.project,
                str(run),
                a.runtime,
                a.cache,
                lane,
                *checkpoints,
            ]
            commands.append({"lane": lane, "checkpoints": checkpoints, "command": command})
        if not a.execute:
            print(json.dumps(commands, indent=2))
            return
        prior = list(control.glob("lane-*.intent.json"))
        known_jobs = set()
        for intent in prior:
            receipt_file = intent.with_name(intent.name.replace(".intent.json", ".submission.json"))
            if not receipt_file.exists():
                raise RuntimeError("Ambiguous prior intent must be reconciled before submission")
            receipt = json.loads(receipt_file.read_text())
            job_id = receipt.get("stdout", "").strip().split(";")[0]
            if receipt.get("returncode") != 0 or not job_id.isdigit():
                raise RuntimeError("Prior failed/ambiguous submission requires reconciliation")
            known_jobs.add(job_id)
        if any((control / (row["lane"] + ".intent.json")).exists() for row in commands):
            raise RuntimeError("Selected lane has an intent; never blindly resubmit")
        if len(prior) + len(commands) > 5:
            raise RuntimeError("At most five fixed lane allocations are allowed")
        active = subprocess.check_output(
            ["squeue", "--me", "--noheader", "--format=%i %j"], text=True
        )
        for line in active.splitlines():
            job_id, name = line.split(maxsplit=1)
            if name.startswith("psp-v1-") and job_id not in known_jobs:
                raise RuntimeError("Unregistered protocol-probe allocation needs reconciliation")
        for row in commands:
            intent = control / (row["lane"] + ".intent.json")
            with intent.open("x") as f:
                json.dump(row, f, indent=2)
                f.flush()
                import os

                os.fsync(f.fileno())
            result = subprocess.run(row["command"], text=True, capture_output=True)
            receipt = {
                **row,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            (control / (row["lane"] + ".submission.json")).write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            print(json.dumps(receipt), flush=True)
            if result.returncode != 0:
                raise RuntimeError(
                    "sbatch failed. Previous submissions retained; "
                    "inspect receipts before continuation."
                )


if __name__ == "__main__":
    main()
