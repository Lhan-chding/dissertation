"""One registered task and one model per physical GPU, with durable attempts."""

from __future__ import annotations

import gc
import json
import os
import socket
import time
import traceback
from pathlib import Path


def physical_gpu_key():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker requires exactly one visible CUDA GPU")
    prop = torch.cuda.get_device_properties(0)
    gpu_uuid = getattr(prop, "uuid", None)
    if not gpu_uuid:
        raise RuntimeError("Physical GPU UUID required for exclusive task ownership")
    return socket.gethostname() + ":" + str(gpu_uuid)


def worker(run, worker_id=None, *, allow_gpu=False, once=False, job_id=None):
    if allow_gpu is not True:
        raise PermissionError("Registered worker requires --allow-gpu")
    from .control import assert_code_bindings
    from .evaluate import evaluate_job, write_once
    from .queue import claim_registered, registered_queue
    from .schema import verify_frozen
    from .training import require_bridge, train

    run = Path(run).resolve()
    assert_code_bindings(verify_frozen(run, role="control"))
    require_bridge(run, role="control")
    gpu_key = physical_gpu_key()
    worker_id = (
        worker_id
        or f"{socket.gethostname()}:{os.environ.get('SLURM_JOB_ID', 'local')}:{os.getpid()}"
    )
    queue = registered_queue(run)
    completed = []
    try:
        while True:
            if (run / "STOP").exists():
                return {"status": "STOP_PRESENT", "completed": completed}
            job = (
                claim_registered(queue, job_id, worker_id, gpu_key=gpu_key)
                if job_id
                else queue.claim(worker_id, gpu_key=gpu_key)
            )
            if job is None:
                rows = queue.rows()
                if all(r["status"] == "COMPLETE" for r in rows):
                    return {"status": "ALL_REGISTERED_TASKS_COMPLETE", "completed": completed}
                if (
                    once
                    or job_id
                    or all(r["status"] in ("COMPLETE", "BLOCKED_TECHNICAL") for r in rows)
                ):
                    return {"status": "NO_READY_JOB", "completed": completed}
                time.sleep(15)
                continue
            started = time.time()
            record = {
                "job_id": job["id"],
                "worker": worker_id,
                "physical_gpu": gpu_key,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "started_unix": started,
            }
            attempt_dir = run / "worker_attempts" / f"{time.time_ns()}_{os.getpid()}"
            write_once(attempt_dir / "STARTED.json", record)
            print(json.dumps({"event": "START", **record}), flush=True)
            try:
                if job["kind"] == "train":
                    result = train(
                        run, job["parent"], job["block"], job["logical_arm_id"], allow_gpu=True
                    )
                elif job["kind"] == "eval":
                    result = evaluate_job(run, job["id"], allow_gpu=True)
                else:
                    raise ValueError("Unknown registered job kind")
                queue.finish(job["id"], worker_id, "COMPLETE", result)
                write_once(
                    attempt_dir / "COMPLETE.json", {**record, "wall_seconds": time.time() - started}
                )
                completed.append(job["id"])
                print(json.dumps({"event": "COMPLETE", "job_id": job["id"]}), flush=True)
            except BaseException as exc:
                failure = {
                    **record,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "wall_seconds": time.time() - started,
                }
                write_once(attempt_dir / "FAILURE.json", failure)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    queue.mark_unknown(
                        job["id"],
                        worker_id,
                        reason="Interrupted process; scheduler and lock verification required",
                    )
                else:
                    queue.finish(job["id"], worker_id, "BLOCKED_TECHNICAL", failure)
                raise
            finally:
                gc.collect()
                import torch

                torch.cuda.empty_cache()
            if once or job_id:
                return {"status": "ONE_JOB_COMPLETE", "completed": completed}
    finally:
        queue.db.close()
