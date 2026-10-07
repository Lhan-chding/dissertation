"""SER-J2 execution interface; every GPU operation requires explicit opt-in."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

from ..verified_discovery_transfer.queue import read_json, write_json
from .queue import build_queue, claim_registered, registered_queue, train_job_id


def _gpu_required(allow_gpu):
    if not allow_gpu:
        raise PermissionError("This command requires --allow-gpu")


def first_group_report(run):
    """Inspect the complete initial triad before subsequent schedule blocks."""
    import fcntl

    from .evaluate import analyze_diagnostic_job
    from .reports import summarize_cells
    from .training import student_directory

    run = Path(run)
    with (run / "first_group.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = run / "FIRST_GROUP_TECHNICAL.json"
        if path.exists():
            return read_json(path)
        queue = registered_queue(run)
        group = [
            r
            for r in queue.rows()
            if r["payload"].get("parent") == "S96"
            and r["payload"].get("block") == 0
            and r["payload"].get("panel") != "E_CONFIRM"
        ]
        if len(group) != 15 or any(
            r["status"] not in ("COMPLETE", "BLOCKED_TECHNICAL") for r in group
        ):
            raise RuntimeError("Initial three-arm training and diagnostics are not terminal")
        training, scored = [], []
        for row in group:
            if row["status"] != "COMPLETE":
                continue
            job = row["payload"]
            if job["kind"] == "train":
                directory = student_directory(run, job["parent"], job["block"], job["arm"])
                updates = [
                    read_json(directory / f"update{step:03d}.json") for step in range(1, 257)
                ]
                training.append(
                    {
                        "job": job["id"],
                        "updates": len(updates),
                        "gpu_seconds_per_update": sum(x["update_gpu_seconds"] for x in updates)
                        / 256,
                        "wall_seconds_per_update": sum(x["update_wall_seconds"] for x in updates)
                        / 256,
                        "peak_memory_allocated_bytes": max(
                            x["peak_memory_allocated_bytes"] for x in updates
                        ),
                        "mean_prompt_tokens": sum(sum(x["sequence_prompt_tokens"]) for x in updates)
                        / 4096,
                        "mean_target_tokens": sum(sum(x["sequence_target_tokens"]) for x in updates)
                        / 4096,
                        "first16_donor_nll": sum(x["sequence_nll"][11] for x in updates[:16]) / 16,
                        "last16_donor_nll": sum(x["sequence_nll"][11] for x in updates[-16:]) / 16,
                    }
                )
            else:
                scored.extend(analyze_diagnostic_job(run, job["id"])["records"])
        result = {
            "status": "FIRST_GROUP_REVIEWED_NO_SELECTION",
            "training": training,
            "diagnostics": summarize_cells(scored) if scored else {},
            "technical_missing": [r["id"] for r in group if r["status"] != "COMPLETE"],
            "generation_records": len(scored),
            "generation_seconds_per_1000_tokens": (
                1000
                * sum(r["elapsed_seconds"] for r in scored)
                / sum(r["generated_length"] for r in scored)
            )
            if scored and sum(r["generated_length"] for r in scored)
            else None,
            "confirmation_accessed": False,
            "conditions_removed": [],
            "configuration_changed": False,
        }
        write_json(path, result)
        queue.db.close()
        return result


def execute_job(run, job, *, allow_gpu):
    _gpu_required(allow_gpu)
    if job["kind"] == "train":
        from .training import train

        if job["block"] > 0:
            first_group_report(run)
        return train(run, parent=job["parent"], block=job["block"], arm=job["arm"], allow_gpu=True)
    if job["kind"] == "eval":
        from .evaluate import evaluate_job

        return evaluate_job(run, job["id"], allow_gpu=True)
    raise ValueError("Unregistered GPU job kind")


def execute_registered(run, job_id, *, allow_gpu):
    _gpu_required(allow_gpu)
    from .training import require_bridge

    require_bridge(run)
    queue = registered_queue(run)
    worker_id = f"direct-{os.getpid()}-{time.time_ns()}"
    job = claim_registered(queue, job_id, worker_id)
    if job is None:
        return next(row["result"] for row in queue.rows() if row["id"] == job_id)
    try:
        result = execute_job(run, job, allow_gpu=True)
        queue.finish(job_id, worker_id, "COMPLETE", result)
        return result
    except BaseException as exc:
        failure = {
            "status": "BLOCKED_TECHNICAL",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "job_id": job_id,
        }
        write_json(Path(run) / "failures" / f"{job_id}.{time.time_ns()}.json", failure)
        queue.finish(job_id, worker_id, "BLOCKED_TECHNICAL", failure)
        raise
    finally:
        queue.db.close()


def worker(run, worker_id, *, allow_gpu, once=False):
    _gpu_required(allow_gpu)
    from .training import require_bridge

    run = Path(run)
    require_bridge(run)
    queue = registered_queue(run)
    results = []
    while True:
        if (run / "STOP").exists():
            return {"status": "STOP_PRESENT", "completed_this_worker": results}
        job = queue.claim(worker_id)
        if job is None:
            rows = queue.rows()
            if queue.all_terminal() or once:
                return {
                    "status": "TERMINAL" if queue.all_terminal() else "NO_READY_JOB",
                    "counts": dict(Counter(r["status"] for r in rows)),
                    "completed_this_worker": results,
                }
            time.sleep(15)
            continue
        print(json.dumps({"event": "START", "job": job["id"], "worker": worker_id}), flush=True)
        started = time.time()
        try:
            result = execute_job(run, job, allow_gpu=True)
            queue.finish(job["id"], worker_id, "COMPLETE", result)
            results.append(job["id"])
        except Exception as exc:
            failure = {
                "status": "BLOCKED_TECHNICAL",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
                "job_id": job["id"],
                "worker": worker_id,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
            write_json(run / "failures" / f"{job['id']}.{time.time_ns()}.json", failure)
            queue.finish(job["id"], worker_id, "BLOCKED_TECHNICAL", failure)
            print(json.dumps(failure), flush=True)
            # A technical failure needs diagnosis before the same process loads
            # another model. Other independent workers may complete their jobs.
            raise
        finally:
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
        print(
            json.dumps(
                {"event": "COMPLETE", "job": job["id"], "elapsed_seconds": time.time() - started}
            ),
            flush=True,
        )
        if once:
            return {"status": "ONE_JOB_COMPLETE", "completed_this_worker": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--legacy-run-root", required=True)
    prep.add_argument("--plan-dir", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--later-history", action="append", default=[])
    prep.add_argument("--later-audit-receipt")
    prep.add_argument("--historical-index-path")
    prep.add_argument("--machine")
    for name in ("cpu-check", "build-queue", "status", "release-and-analyze"):
        command = sub.add_parser(name)
        command.add_argument("--run", required=True)
    bridge = sub.add_parser("bridge")
    bridge.add_argument("--run", required=True)
    bridge.add_argument("--parent", choices=("S96", "REP96"), default="S96")
    bridge.add_argument("--allow-gpu", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--run", required=True)
    train.add_argument("--parent", choices=("S96", "REP96"), required=True)
    train.add_argument("--block", type=int, choices=(0, 1, 2), required=True)
    train.add_argument(
        "--arm", choices=("A_LOCAL_C1", "B_FORWARD_C4", "C_FORWARD_C3"), required=True
    )
    train.add_argument("--allow-gpu", action="store_true")
    for name in ("eval-diagnostic", "eval-confirm-sealed"):
        command = sub.add_parser(name)
        command.add_argument("--run", required=True)
        command.add_argument("--job", required=True)
        command.add_argument("--step", type=int, choices=(0, 128, 256), required=True)
        command.add_argument("--allow-gpu", action="store_true")
    work = sub.add_parser("worker")
    work.add_argument("--run", required=True)
    work.add_argument("--worker", required=True)
    work.add_argument("--allow-gpu", action="store_true")
    work.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .import_sources import prepare

        result = prepare(
            args.legacy_run_root,
            args.plan_dir,
            args.out,
            later_history_paths=args.later_history,
            later_audit_receipt=args.later_audit_receipt,
            historical_index_path=args.historical_index_path,
            machine=args.machine,
        )
    elif args.command == "cpu-check":
        from .import_sources import cpu_check

        result = cpu_check(args.run)
    elif args.command == "build-queue":
        result = build_queue(args.run)
    elif args.command == "status":
        queue = registered_queue(args.run)
        rows = queue.rows()
        result = {
            "counts": dict(Counter(r["status"] for r in rows)),
            "all_terminal": queue.all_terminal(),
            "active": [
                {k: r[k] for k in ("id", "status", "worker")}
                for r in rows
                if r["status"] in ("RUNNING", "BLOCKED_TECHNICAL")
            ],
        }
    elif args.command == "bridge":
        _gpu_required(args.allow_gpu)
        from .bridge import run_bridge

        result = run_bridge(args.run, parent=args.parent, allow_gpu=True)
    elif args.command == "train":
        _gpu_required(args.allow_gpu)
        result = execute_registered(
            args.run, train_job_id(args.parent, args.block, args.arm), allow_gpu=True
        )
    elif args.command.startswith("eval-"):
        _gpu_required(args.allow_gpu)
        queue = registered_queue(args.run)
        jobs = {r["id"]: r["payload"] for r in queue.rows()}
        job = jobs[args.job]
        if job["step"] != args.step or (
            (job["panel"] == "E_CONFIRM") != (args.command == "eval-confirm-sealed")
        ):
            raise ValueError("Requested endpoint does not match registered job")
        result = execute_registered(args.run, args.job, allow_gpu=True)
    elif args.command == "worker":
        result = worker(args.run, args.worker, allow_gpu=args.allow_gpu, once=args.once)
    else:
        from .evaluate import release_and_analyze

        result = release_and_analyze(args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
