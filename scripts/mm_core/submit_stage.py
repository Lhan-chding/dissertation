"""Submit held, budgeted Slurm workers and release only after binding exact IDs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from mm_core.allocations import bind_allocation, reserve_allocation
from mm_core.execution import atomic_json, register_stage_plan, verify_gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--code-root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument(
        "--stage",
        required=True,
        choices=[
            "FORMAT_BASE_TEST",
            "FORMAT_TUNE_POST_BRIDGE",
            "FORMAT_CHECK",
            "MEASUREMENT_AUDIT",
            "BRIDGE",
            "ENGINE",
        ],
    )
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--minutes", type=int, required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--qos", required=True)
    p.add_argument("--partition")
    args = p.parse_args()
    root, code = args.run_root.resolve(), args.code_root.resolve()
    verify_gate(root, args.stage)
    if not 1 <= args.shards <= 5 or args.minutes < 1:
        raise ValueError("Invalid resource envelope")
    if args.stage in {"BRIDGE", "ENGINE"}:
        if args.shards != 1:
            raise ValueError("Training interfaces require a single GPU")
    else:
        register_stage_plan(root, args.stage, args.shards)
    folder = root / "accounting/submissions" / args.stage
    folder.mkdir(parents=True, exist_ok=False)
    for shard in range(args.shards):
        key = args.stage.lower().replace("_", "-") + f"-{shard}"
        reserve_allocation(root, key, args.stage, seconds=args.minutes * 60)
        if args.stage == "BRIDGE":
            script_name = "run_common_bridge.py"
            extra = []
        elif args.stage == "ENGINE":
            script_name = "run_engine_resume_test.py"
            extra = []
        else:
            script_name = (
                "run_measurement_audit.py"
                if args.stage == "MEASUREMENT_AUDIT"
                else "run_format_check.py"
            )
            extra = ["--stage", args.stage, "--shard", str(shard), "--shards", str(args.shards)]
        command = [
            str(args.python),
            str(code / "scripts/mm_core" / script_name),
            "--run-root",
            str(root),
            *extra,
        ]
        script = folder / f"worker_{shard}.sh"
        script.write_text(
            "#!/bin/bash\nset -euo pipefail\n"
            + "export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4\n"
            + "export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0\n"
            + "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
            + "export PYTHONPATH="
            + shlex.quote(str(code / "src"))
            + "\n"
            + "cd "
            + shlex.quote(str(code))
            + "\n"
            + "nvidia-smi --query-gpu=name,uuid,memory.total --format=csv\n"
            + "exec "
            + shlex.join(command)
            + "\n"
        )
        submit = [
            "sbatch",
            "--hold",
            "--parsable",
            "--account",
            args.account,
            "--qos",
            args.qos,
            "--job-name",
            "mmcore-" + key,
            "--gres=gpu:pro6000:1",
            "--cpus-per-task=4",
            "--mem=32G",
            "--time",
            str(args.minutes),
            "--output",
            str(folder / f"worker_{shard}_%j.log"),
        ]
        if args.partition:
            submit.extend(["--partition", args.partition])
        submit.append(str(script))
        result = subprocess.run(submit, text=True, capture_output=True, timeout=30)
        atomic_json(
            folder / f"submit_{shard}.json",
            {
                "command": submit,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            },
        )
        result.check_returncode()
        job_id = result.stdout.strip().split(";")[0]
        bind_allocation(root, key, job_id)
        release = subprocess.run(
            ["scontrol", "release", job_id], text=True, capture_output=True, timeout=30
        )
        atomic_json(
            folder / f"release_{shard}.json",
            {
                "job_id": job_id,
                "stdout": release.stdout,
                "stderr": release.stderr,
                "returncode": release.returncode,
            },
        )
        release.check_returncode()
        print(f"Registered {args.stage} shard {shard}/{args.shards}: {job_id}", flush=True)


if __name__ == "__main__":
    main()
