"""Submit one registered v2 G2 shard pair when real CPU gates and capacity permit."""

import argparse
import fcntl
import getpass
import os
import runpy
import subprocess
from pathlib import Path

from sr_f4.common import now, read, write
from sr_f4.pool import ACCOUNT, QOS, Slurm, gpu_count

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_recipe_v2_20261026")
V1 = ROOT.parent / "sr_f5_semantic_j_20261025"


def submit(code, revision, pair):
    code = code.resolve(strict=True)
    if not code.is_relative_to(ROOT.parent) or ROOT.resolve(strict=True) != ROOT:
        raise PermissionError("Deployment and run must be canonical in our project")
    registration = read(V1 / "CONTINUATION/RECIPE_V2.json")
    config = read(ROOT / "CONFIG.json")
    if config.get("numerics", {}).get("recipe_version") != 2:
        raise PermissionError("Not the registered second recipe")
    ready = read(ROOT / "CPU_READY.json")
    if ready["status"] != "CPU_READY_FOR_G2_PREFLIGHT" or ready["code_revision"] != revision:
        raise PermissionError("V2 CPU readiness/revision does not match")
    for path in ("CPU_TESTS.json", "DATA/HISTORICAL_WORLD_AUDIT.json"):
        if read(ROOT / path)["status"] != "PASS":
            raise PermissionError("V2 CPU gate failed: " + path)
    if read(ROOT / "G2_SPEC_FROZEN.json")["code_revision"] != revision:
        raise PermissionError("Census revision differs")
    if read(V1 / "CONTINUATION/RECIPE_V2_CONFIG.json") != config:
        raise PermissionError("Configuration differs from prospective registration")
    # Invoke the same identity/counting helper used for the first census.
    capacity = runpy.run_path(str(code / "scripts/sr_f5/submit_g2.py"))["capacity"]
    slurm, user = Slurm(), getpass.getuser()
    with (ROOT / "G2_SUBMIT.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = ROOT / f"G2_SUBMISSION_PAIR{pair}.json"
        if receipt.exists():
            raise RuntimeError("Pair submission exists; inspect before retry")
        jobs, gpus, queue = capacity(slurm, user, set())
        if jobs + 1 > 5 or gpus + 2 > 5:
            write(
                ROOT / f"G2_PAIR{pair}_CAPACITY.json",
                {
                    "at": now(),
                    "status": "WAITING_FOR_CAPACITY",
                    "jobs": jobs,
                    "reserved_gpus": gpus,
                    "submitted": False,
                },
            )
            return {"status": "WAITING_FOR_CAPACITY", "pair": pair}
        record = dict(
            at=now(),
            status="SUBMISSION_UNKNOWN",
            pair=pair,
            science_revision=revision,
            source_code=str(code),
            registered_recipe=registration,
            before=queue,
            shards=[2 * pair, 2 * pair + 1],
            expected_answers=2304,
        )
        write(receipt, record)
        name = f"LH-SRF5-G2-P{pair}"
        env = dict(
            os.environ,
            SRF5_CODE=str(code),
            SRF5_LAUNCHER_CODE=str(code),
            SRF5_ROOT=str(ROOT),
            SRF5_PAIR=str(pair),
        )
        result = subprocess.run(
            [
                "sbatch",
                "--hold",
                "--parsable",
                f"--job-name={name}",
                "--exclude=gpu-pro6000-12",
                f"--output={ROOT}/slurm/G2-P{pair}-%j.log",
                str(code / "scripts/sr_f5/calibration.sbatch"),
            ],
            env=env,
            cwd=code,
            capture_output=True,
            text=True,
        )
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        write(receipt, record)
        if result.returncode:
            raise RuntimeError("Submission failed; inspect preserved receipt")
        job = result.stdout.strip().split(";")[0]
        if not job.isdecimal():
            raise RuntimeError("Unknown submission identity")
        record.update(job_id=job, status="HELD")
        write(receipt, record)
        actual = slurm.show(job)
        if (
            actual.get("JobName") != name
            or actual.get("Account") != ACCOUNT
            or actual.get("QOS") != QOS
            or gpu_count(actual.get("ReqTRES")) != 2
            or actual.get("WorkDir") != str(code)
        ):
            raise PermissionError("Submitted allocation identity differs; remains held")
        record["allocation"] = actual
        write(receipt, record)
        subprocess.run(["scontrol", "release", job], check=True)
        record["status"] = "RELEASED_VERIFIED"
        write(receipt, record)
        return {"status": record["status"], "pair": pair, "job_id": job}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--pair", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    print(submit(args.code, args.revision, args.pair))
