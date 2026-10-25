"""Submit exactly four frozen G2 shards after server CPU preparation.

Unknown submission outcomes require diagnosis; no automated duplicate retry.
This is a one-shot dispatcher, not a recurring monitor or training launcher.
"""

import fcntl
import getpass
import json
import os
import subprocess
from pathlib import Path

from sr_f4.common import now, read, write
from sr_f4.pool import QOS, Slurm, gpu_count

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025")
CODE = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f5_code_26e560b")


def main():
    if read(ROOT / "CPU_READY.json")["status"] != "CPU_READY_FOR_G2_PREFLIGHT":
        raise PermissionError("CPU preparation is not complete")
    if read(ROOT / "CPU_TESTS.json")["status"] != "PASS":
        raise PermissionError("Server CPU tests failed")
    if read(ROOT / "DATA/HISTORICAL_WORLD_AUDIT.json")["status"] != "PASS":
        raise PermissionError("Historical world exclusion failed")
    if read(ROOT / "RETENTION/FROZEN.json")["status"] != "FROZEN_BEFORE_NEW_MODEL_OUTPUTS":
        raise PermissionError("Retention subsets are not frozen")
    slurm = Slurm()
    with (ROOT / "G2_SUBMIT.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = ROOT / "G2_SUBMISSION.json"
        if receipt_path.exists():
            raise RuntimeError("Submission receipt already exists; inspect rather than duplicate")
        receipt = dict(
            at=now(),
            status="SUBMITTING",
            source_code=str(CODE),
            jobs=[],
            all_qos_job_limit=5,
            effective_total_gpu_limit=5,
            registered_workers=4,
        )
        write(receipt_path, receipt)
        for shard in range(4):
            user_ids = slurm.user_job_ids(getpass.getuser())
            queue = slurm.queue()
            for job_id in user_ids - set(queue):
                queue[job_id] = slurm.show(job_id)
            reserved = sum(
                max(gpu_count(row.get("ReqTRES")), gpu_count(row.get("AllocTRES")))
                for row in queue.values()
            )
            if len(user_ids) >= 5 or reserved >= 5:
                receipt.update(
                    status="CAPACITY_BLOCKED",
                    observed_jobs=len(user_ids),
                    observed_reserved_gpus=reserved,
                )
                write(receipt_path, receipt)
                raise RuntimeError("Capacity unavailable; no unrelated jobs changed")
            if any(not row.get("ReqTRES") for row in queue.values()):
                raise RuntimeError("Unknown reservation; refuse submission")
            name = f"SRF5-G2-W{shard}-LH"
            attempt = dict(shard=shard, name=name, status="SUBMISSION_UNKNOWN", before=queue)
            receipt["jobs"].append(attempt)
            write(receipt_path, receipt)
            command = [
                "sbatch",
                "--parsable",
                f"--job-name={name}",
                f"--output={ROOT}/slurm/G2-W{shard}-%j.log",
                str(CODE / "scripts/sr_f5/calibration.sbatch"),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=dict(os.environ, SRF5_CODE=str(CODE), SRF5_SHARD=str(shard), SRF5_SHARDS="4"),
            )
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdecimal():
                raise RuntimeError("Ambiguous submission; preserve UNKNOWN and diagnose")
            attempt.update(job_id=job_id, status="SUBMITTED", scheduler_message=result.stderr)
            write(receipt_path, receipt)
            actual = slurm.show(job_id)
            if actual.get("JobName") != name or actual.get("QOS") != QOS:
                raise RuntimeError("Submitted job identity mismatch")
            if gpu_count(actual.get("ReqTRES")) != 1:
                raise RuntimeError("Unexpected GPU allocation")
            attempt["verified"] = actual
            write(receipt_path, receipt)
        receipt["status"] = "FOUR_G2_WORKERS_SUBMITTED_AND_VERIFIED"
        write(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "jobs": [
                        {k: j[k] for k in ("job_id", "name", "shard")} for j in receipt["jobs"]
                    ],
                }
            )
        )


if __name__ == "__main__":
    main()
