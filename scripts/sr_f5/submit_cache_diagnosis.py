"""One-shot held submission for the bounded cache diagnosis after 226846 completes."""

import fcntl
import getpass
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

from sr_f4.common import now, read, write
from sr_f4.pool import ACCOUNT, QOS, Slurm, gpu_count


def main():
    root = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f5_semantic_j_20261025")
    code = Path("/projects/_ssd/varunssd/louis-ssvc/sr_f5_development_ea50c51")
    revision = "ea50c51f0f145a183df360ab82025216d6303aca"
    ops = Path(sys.argv[1]).resolve(strict=True)
    opsrev = sys.argv[2]
    assert root.resolve() == root and ops.is_relative_to(root.parent)
    assert (ops / "scripts/sr_f5/diagnose_cache.py").is_file()
    slurm = Slurm()
    user = getpass.getuser()
    capacity = runpy.run_path(str(code / "scripts/sr_f5/submit_g2.py"))["capacity"]
    with (root / "CONTINUATION/CACHE_SUBMIT.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = root / "CONTINUATION/PROBABILITY_CACHE_226283"
        receipt = directory / "SUBMISSION.json"
        if receipt.exists():
            raise RuntimeError("Submission exists; inspect before retry")
        jobs, gpus, queue = capacity(slurm, user, set())
        if jobs + 1 > 5 or gpus + 2 > 5:
            raise RuntimeError("Capacity unavailable")
        assert "226846" not in slurm.user_job_ids(user)
        prior = slurm.terminal("226846")
        assert prior["state"] == "COMPLETED" and prior["exit_code"] == "0:0"
        for mode in ("bf16", "fp32"):
            assert (
                read(root / f"CONTINUATION/PROBABILITY_226283/{mode}.json")["status"]
                == "DIAGNOSTIC_COMPLETE"
            )
        assert read(root / "G3/FROZEN.json")["code_revision"] == revision
        directory.mkdir(exist_ok=True)
        script = ops / "scripts/sr_f5/diagnose_cache.py"
        d = dict(
            status="SUBMISSION_UNKNOWN",
            at=now(),
            science_revision=revision,
            operations_revision=opsrev,
            diagnostic_script=str(script),
            prior_jobs=jobs,
            prior_gpus=gpus,
            prior_queue=queue,
            prior_terminal=prior,
            new_samples=0,
            optimizer_updates=0,
            scope=(
                "Original G8 and row3 single forced-token cached replay through token162; "
                "bf16/fp32; no scientific restart"
            ),
        )
        write(receipt, d)
        env = dict(
            os.environ,
            SRF5_ROOT=str(root),
            SRF5_CODE=str(code),
            SRF5_REVISION=revision,
            SRF5_DIAG_SCRIPT=str(script),
        )
        p = subprocess.run(
            [
                "sbatch",
                "--hold",
                "--parsable",
                f"--output={root}/slurm/G3-CACHE-%j.log",
                str(ops / "scripts/sr_f5/cache_diagnosis.sbatch"),
            ],
            env=env,
            cwd=code,
            capture_output=True,
            text=True,
        )
        d.update(submit_returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        write(receipt, d)
        if p.returncode:
            raise RuntimeError("Submission failed; retained evidence")
        job = p.stdout.strip().split(";")[0]
        assert job.isdecimal()
        d.update(job_id=job, status="HELD")
        write(receipt, d)
        actual = slurm.show(job)
        assert (
            actual["JobName"] == "LH-SRF5-G3-CACHE"
            and actual["QOS"] == QOS
            and actual["Account"] == ACCOUNT
            and gpu_count(actual["ReqTRES"]) == 2
        )
        d["verified"] = actual
        write(receipt, d)
        current = read(root / "CONTINUATION/CURRENT.json")
        current["probability_diagnosis"]["status"] = "DIAGNOSTIC_COMPLETE_CACHE_REPLAY_REQUIRED"
        current["cache_diagnosis"] = dict(
            job_id=job,
            receipt="CONTINUATION/PROBABILITY_CACHE_226283/SUBMISSION.json",
            outputs=[f"CONTINUATION/PROBABILITY_CACHE_226283/{m}.json" for m in ("bf16", "fp32")],
            restart_blocked_until_diagnosed=True,
        )
        write(root / "CONTINUATION/CURRENT.json", current)
        subprocess.run(["scontrol", "release", job], check=True)
        d.update(status="RELEASED_VERIFIED")
        write(receipt, d)
        print(json.dumps(d))


if __name__ == "__main__":
    main()
