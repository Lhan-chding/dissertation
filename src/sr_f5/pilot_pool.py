"""Frozen four-worker G3 allocation; no formal-training transition."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from sr_f4.common import now, read, write
from sr_f4.pool import ACCOUNT, QOS, Slurm, gpu_count
from srf5_ref.generator import seed_for

from .calibration import digest
from .dual_gpu import _stop_groups, supervise, telemetry_command, worker_environments
from .pilot import registered_write, require_frozen


def pool_source_identity():
    code = Path(__file__).resolve().parents[2]
    names = (
        "src/sr_f5/pilot_pool.py",
        "src/sr_f5/dual_gpu.py",
        "scripts/sr_f5/pilot.sbatch",
        "scripts/sr_f5/submit_g3.py",
    )
    return {name: hashlib.sha256((code / name).read_bytes()).hexdigest() for name in names}


def build_plan(config, frozen, revision):
    if frozen.get("code_revision") != revision:
        raise PermissionError("Pool revision must equal the frozen scientific revision")
    paths = frozen["paths"]
    expected = [dict(parent=p, arm=a) for p in config["parents"] for a in config["arms"]]
    if paths != expected or len(paths) != 15:
        raise PermissionError("The frozen pilot is not the registered fifteen paths")
    workers = [dict(worker=i, weighted_steps=0, tasks=[]) for i in range(4)]
    arm_orders = {}
    for parent in config["parents"]:
        order_seed = seed_for(config["master_seed"], "G3_pool_arm_order", parent)
        arms = list(config["arms"])
        random.Random(order_seed).shuffle(arms)
        arm_orders[parent] = dict(seed=order_seed, arms=arms)
        for arm in arms:
            worker = min(workers, key=lambda w: (w["weighted_steps"], w["worker"]))
            recovery = parent == "b0"
            worker["tasks"].append(dict(parent=parent, arm=arm, recovery=recovery))
            worker["weighted_steps"] += 12 + (8 if recovery else 0)
    return dict(
        status="FROZEN_BEFORE_PILOT_OUTPUTS",
        code_revision=revision,
        g3_frozen_digest=digest(frozen),
        source_identity=pool_source_identity(),
        order_seed_namespace="seed_for(master_seed, G3_pool_arm_order, parent)",
        allocation="deterministic least-weighted-steps, worker-index tie break",
        arm_orders=arm_orders,
        workers=workers,
        jobs=2,
        gpus_per_job=2,
        primary_paths=15,
        recovery_paths=5,
        automatic_formal_training=False,
    )


def freeze_pool(root, revision):
    root = Path(root).resolve()
    frozen = require_frozen(root)
    plan = build_plan(read(root / "CONFIG.json"), frozen, revision)
    path = root / "G3/POOL_PLAN.json"
    if not path.exists() and any((root / "G3/RUNS").glob("*/*/*/rollouts/*")):
        raise PermissionError("Cannot register allocation after pilot outputs")
    registered_write(path, plan)
    return plan


def require_plan(root, revision):
    root = Path(root)
    if not (root / "G3/POOL_PLAN.json").is_file():
        raise PermissionError("Pilot allocation must be frozen before workers start")
    plan = read(root / "G3/POOL_PLAN.json")
    if freeze_pool(root, revision) != plan:
        raise PermissionError("Frozen pool identity differs")
    return plan


def verify_code(code):
    code = Path(code).resolve(strict=True)
    if code != Path(__file__).resolve().parents[2]:
        raise PermissionError("Worker is not importing the declared scientific deployment")
    return code


def verify_completion(root, task, lane):
    path = root / "G3/RUNS" / task["parent"] / task["arm"] / lane / "COMPLETE.json"
    result = read(path)
    if (
        result.get("status") != "COMPLETE"
        or result.get("committed_step") != 12
        or result.get("lane") != lane
        or result.get("identity", {}).get("parent") != task["parent"]
        or result.get("identity", {}).get("arm") != task["arm"]
    ):
        raise PermissionError("Pilot subprocess did not publish its registered endpoint")


def run_worker(code, root, revision, python):
    code, root = verify_code(code), Path(root)
    plan = require_plan(root, revision)
    worker = int(os.environ["SRF5_SHARD"])
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if (
        worker not in range(4)
        or os.environ.get("SRF5_SHARDS") != "4"
        or len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 1
        or not os.environ.get("CUDA_VISIBLE_DEVICES")
        or not job_id.isdecimal()
    ):
        raise PermissionError("Worker requires its real one-GPU allocation")
    directory = root / "G3/workers"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"W{worker}-{job_id}.json"
    with (directory / f"W{worker}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise PermissionError("Worker attempt receipt exists; diagnose before recovery")
        receipt = dict(
            at=now(),
            status="RUNNING",
            worker=worker,
            job_id=job_id,
            code_revision=revision,
            pool_plan_digest=digest(plan),
            completed=[],
            calls=[],
        )
        write(output, receipt)
        environment = dict(os.environ, PYTHONPATH=f"{code}/src:{code}")
        try:
            for task in plan["workers"][worker]["tasks"]:
                base = [
                    str(python),
                    "-u",
                    "-m",
                    "sr_f5.pilot",
                    "--root",
                    str(root),
                    "--parent",
                    task["parent"],
                    "--arm",
                    task["arm"],
                ]
                phases = [("primary", ["--lane", "primary"])]
                if task["recovery"]:
                    phases.extend(
                        [
                            ("recovery", ["--lane", "recovery"]),
                            ("compare_recovery", ["--compare-recovery"]),
                        ]
                    )
                for phase, arguments in phases:
                    call = dict(
                        parent=task["parent"],
                        arm=task["arm"],
                        phase=phase,
                        status="STARTED",
                        started_at=now(),
                    )
                    receipt["calls"].append(call)
                    write(output, receipt)
                    result = subprocess.run(
                        base + arguments, cwd=code, env=environment, check=False
                    )
                    call.update(exit_code=result.returncode, finished_at=now())
                    if result.returncode:
                        call["status"] = "FAILED"
                        raise RuntimeError(f"Pilot {task} {phase} failed: {result.returncode}")
                    if phase == "compare_recovery":
                        recovery = read(
                            root / "G3/RUNS" / task["parent"] / task["arm"] / "RECOVERY.json"
                        )
                        if recovery.get("status") != "PASS":
                            raise RuntimeError("Recovery comparison did not pass")
                    else:
                        verify_completion(root, task, phase)
                    call["status"] = "COMPLETE"
                    write(output, receipt)
                receipt["completed"].append(task)
                write(output, receipt)
        except BaseException as exc:
            receipt.update(status="FAILED", finished_at=now(), error=repr(exc))
            write(output, receipt)
            raise
        receipt.update(status="COMPLETE", finished_at=now(), scientific_gate="NOT_YET_AGGREGATED")
        write(output, receipt)
        return receipt


def run_pair(args):
    code = verify_code(args.code)
    plan = require_plan(args.root, args.revision)
    environments = worker_environments(
        args.pair, os.environ.get("CUDA_VISIBLE_DEVICES", ""), os.environ
    )
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal():
        raise PermissionError("A real Slurm allocation is required")
    actual = Slurm().show(job_id)
    if (
        actual.get("JobName") != f"LH-SRF5-G3-P{args.pair}"
        or actual.get("QOS") != QOS
        or actual.get("Account") != ACCOUNT
        or max(gpu_count(actual.get("ReqTRES")), gpu_count(actual.get("AllocTRES"))) != 2
    ):
        raise PermissionError("G3 dual-GPU allocation identity mismatch")
    command = [
        str(args.python),
        "-u",
        "-m",
        "sr_f5.pilot_pool",
        "--worker",
        "--code",
        str(code),
        "--root",
        str(args.root),
        "--revision",
        args.revision,
        "--python",
        str(args.python),
    ]
    logs = [args.root / "slurm" / f"G3-W{e['SRF5_SHARD']}-{job_id}.log" for e in environments]
    path = args.root / "G3/dual_jobs" / f"{job_id}.json"
    if path.exists():
        raise PermissionError("Allocation attempt already recorded; no blind retry")
    telemetry_path = args.root / "slurm" / f"GPU-{job_id}.csv"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        at=now(),
        status="RUNNING",
        job_id=job_id,
        pair=args.pair,
        code=str(code),
        code_revision=args.revision,
        pool_plan_digest=digest(plan),
        workers=[int(e["SRF5_SHARD"]) for e in environments],
        devices=[e["CUDA_VISIBLE_DEVICES"] for e in environments],
        allocation=actual,
        logs=list(map(str, logs)),
        gpu_telemetry=str(telemetry_path),
    )
    write(path, receipt)
    telemetry = None
    try:
        with telemetry_path.open("a") as stream:
            telemetry = subprocess.Popen(
                telemetry_command(receipt["devices"]),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            supervise([command, command], environments, logs)
    except BaseException as exc:
        receipt.update(status="FAILED", finished_at=now(), error=repr(exc))
        write(path, receipt)
        raise
    finally:
        if telemetry is not None:
            _stop_groups([telemetry], 2)
    receipt.update(status="COMPLETE", finished_at=now(), scientific_gate="NOT_YET_AGGREGATED")
    write(path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--pair", type=int, choices=(0, 1))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        verify_code(args.code)
        result = freeze_pool(args.root, args.revision)
    elif args.worker:
        result = run_worker(args.code, args.root, args.revision, args.python)
    elif args.pair is not None:
        result = run_pair(args)
    else:
        parser.error("--freeze, --worker or --pair is required")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
