"""Frozen SER-J23 matrix and durable GPU/student ownership.

UNKNOWN is an occupied lease, never a licence for an automatic resubmission.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..verified_discovery_transfer.queue import Queue as DurableQueue
from ..verified_discovery_transfer.queue import digest, read_json
from .schema import ARMS, PARENTS, PHASE_ID, read_bound


def train_job_id(parent, block, arm):
    return f"train.{parent}.{block}.{arm}"


def analysis_identity():
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("semantics.py", "statistics.py", "reports.py")
    }


def code_identity():
    root = Path(__file__).resolve().parents[1]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def validate_training_jobs(jobs, mode):
    arms = tuple(ARMS) if mode == "RERUN_24" else tuple(ARMS)[2:] if mode == "REUSE_12" else ()
    expected = {(p, b, a) for p in PARENTS for b in range(3) for a in arms}
    actual = [(j["parent"], j["block"], j.get("logical_arm_id", j.get("arm"))) for j in jobs]
    if not expected or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Training must be the complete frozen REUSE_12 or RERUN_24 matrix")
    for j in jobs:
        if type(j["block"]) is not int:
            raise ValueError("Training block requires an exact integer")
        if (
            j.get("updates", 256) != 256
            or j.get("block_seed", 108701 + j["block"]) != 108701 + j["block"]
        ):
            raise ValueError("Training dose or schedule changed")
    return jobs


class Queue(DurableQueue):
    def __init__(self, path, *, maximum=5):
        super().__init__(path, maximum=maximum)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS leases(resource TEXT PRIMARY KEY,job TEXT NOT NULL)"
        )

    def claim(self, worker, *, gpu_key=None, blocked_kinds=()):
        for row in sorted(self.rows(), key=lambda r: r["payload"].get("priority", 10)):
            if row["status"] == "PENDING" and row["payload"]["kind"] not in blocked_kinds:
                try:
                    result = claim_registered(self, row["id"], worker, gpu_key=gpu_key)
                except ResourceUnavailable:
                    continue
                if result is not None:
                    return result
        return None

    def finish(self, job_id, worker, status, result):
        if status not in {"COMPLETE", "BLOCKED_TECHNICAL"}:
            raise ValueError("J23 only accepts complete or explicitly blocked results")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._assert_unreleased()
            super().finish(job_id, worker, status, result)
            self.db.execute("DELETE FROM leases WHERE job=?", (job_id,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def mark_unknown(self, job_id, worker, *, reason):
        if not reason:
            raise ValueError("Unknown state requires a retained reason")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._assert_unreleased()
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["worker"] != worker or row["status"] not in {"RUNNING", "UNKNOWN"}:
                raise ValueError("Only the owning worker can mark its active task UNKNOWN")
            self.db.execute(
                "INSERT INTO events(job,event,at,details) VALUES(?,?,?,?)",
                (
                    job_id,
                    "UNKNOWN",
                    time.time(),
                    json.dumps({"reason": reason, "previous": dict(row)}),
                ),
            )
            self.db.execute("UPDATE jobs SET status='UNKNOWN' WHERE id=?", (job_id,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def retry(self, job_id, *, reason, old_process_stopped=False, scheduler_state=None):
        if (
            not reason
            or old_process_stopped is not True
            or scheduler_state
            not in {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "LOCAL_PROCESS_EXITED",
            }
        ):
            raise PermissionError("Retry requires explicit scheduler/process termination evidence")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._assert_unreleased()
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["status"] not in {"UNKNOWN", "RUNNING", "BLOCKED_TECHNICAL"}:
                raise ValueError("Only a diagnosed interrupted task may retry")
            self.db.execute(
                "INSERT INTO events(job,event,at,details) VALUES(?,?,?,?)",
                (
                    job_id,
                    "EXPLICIT_RETRY",
                    time.time(),
                    json.dumps(
                        {
                            "reason": reason,
                            "scheduler_state": scheduler_state,
                            "old_process_stopped": True,
                            "previous": dict(row),
                        }
                    ),
                ),
            )
            self.db.execute(
                "UPDATE jobs SET status='PENDING',worker=NULL,started=NULL,"
                "finished=NULL,result=NULL WHERE id=?",
                (job_id,),
            )
            self.db.execute("DELETE FROM leases WHERE job=?", (job_id,))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def seal_release(self):
        if any(row["status"] != "COMPLETE" for row in self.rows()):
            raise PermissionError("J23 release requires every registered job COMPLETE")
        return super().seal_release()


class ResourceUnavailable(ValueError):
    pass


def claim_registered(queue, job_id, worker, *, gpu_key=None):
    """Idempotent named claim. gpu_key must identify the physical GPU, not a PID."""
    if not isinstance(worker, str) or not worker:
        raise ValueError("Worker identity is required")
    if not isinstance(gpu_key, str) or not gpu_key:
        raise ValueError("Physical GPU identity is required for one-task-per-card ownership")
    queue.db.execute("BEGIN IMMEDIATE")
    try:
        queue._assert_unreleased()
        rows = queue.rows()
        by_id = {r["id"]: r for r in rows}
        if job_id not in by_id:
            raise ValueError("Unregistered job")
        row = by_id[job_id]
        if row["status"] == "COMPLETE" or (row["status"] == "RUNNING" and row["worker"] == worker):
            queue.db.execute("COMMIT")
            return None
        if row["status"] != "PENDING":
            raise ResourceUnavailable(
                "Active/UNKNOWN tasks need diagnosed recovery, never double submission"
            )
        if sum(r["status"] in {"RUNNING", "UNKNOWN"} for r in rows) >= queue.maximum:
            raise ResourceUnavailable("Frozen GPU concurrency exhausted")
        payload = row["payload"]
        if payload.get("gpu_count", 1) != 1:
            raise ValueError("Each registered task must use one GPU")
        if any(
            by_id.get(dep, {}).get("status") != "COMPLETE"
            for dep in payload.get("dependencies", [])
        ):
            raise ResourceUnavailable("Required training paths are incomplete")
        resources = ["gpu:" + gpu_key]
        if payload["kind"] == "train":
            resources.append(
                "student:"
                + train_job_id(payload["parent"], payload["block"], payload["logical_arm_id"])
            )
        for resource in resources:
            if queue.db.execute("SELECT job FROM leases WHERE resource=?", (resource,)).fetchone():
                raise ResourceUnavailable("Physical GPU or student already owned")
            queue.db.execute("INSERT INTO leases VALUES(?,?)", (resource, job_id))
        queue.db.execute(
            "UPDATE jobs SET status='RUNNING',worker=?,started=? WHERE id=?",
            (worker, time.time(), job_id),
        )
        queue.db.execute(
            "INSERT INTO events(job,event,at,details) VALUES(?,?,?,?)",
            (job_id, "CLAIM", time.time(), json.dumps({"worker": worker, "gpu_key": gpu_key})),
        )
        queue.db.execute("COMMIT")
        return payload
    except BaseException:
        queue.db.execute("ROLLBACK")
        raise


def build_queue(run):
    from .evaluate import evaluation_job_id, validate_evaluation_jobs, write_once
    from .schema import verify_frozen

    run = Path(run)
    frozen = verify_frozen(run, role="control")
    mode = read_bound(run, "EXECUTION_MODE.json")["mode"]
    training = validate_training_jobs(read_bound(run, "manifests/training_jobs.jsonl"), mode)
    availability = read_bound(run, "manifests/diagnostic_availability.json")
    evaluation = validate_evaluation_jobs(
        read_bound(run, "manifests/evaluation_jobs.jsonl"),
        mode=mode,
        unavailable=availability.get("unavailable", []),
    )
    jobs = []
    for j in training:
        arm = j.get("logical_arm_id", j.get("arm"))
        jobs.append(
            {
                **j,
                "logical_arm_id": arm,
                "arm": arm,
                "id": train_job_id(j["parent"], j["block"], arm),
                "kind": "train",
                "dependencies": [],
                "priority": 0,
                "gpu_count": 1,
            }
        )
    training_ids = {j["id"] for j in jobs}
    for j in evaluation:
        arm = j["logical_arm_id"]
        dep = train_job_id(j["parent"], j["block"], arm)
        # Historical diagnostic students never acquire a dependency on a new run.
        dependencies = [dep] if j["source"] == "NEW_TRAINING" and dep in training_ids else []
        if j["panel"] == "E_CONFIRM2":
            dependencies = sorted(training_ids)
        jobs.append(
            {
                **j,
                "id": evaluation_job_id(j),
                "evaluation_kind": j["kind"],
                "kind": "eval",
                "dependencies": dependencies,
                "priority": 1 if j["panel"] == "E_CONFIRM2" else 2,
                "gpu_count": 1,
            }
        )
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate registered job identity")
    record = {
        "schema": "ser-j23-registered-matrix-v1",
        "phase_id": PHASE_ID,
        "mode": mode,
        "jobs": jobs,
        "plan_hash": digest(frozen),
        "analysis_identity": analysis_identity(),
        "code_identity": code_identity(),
        "formal_training_jobs": len(training),
        "formal_evaluation_jobs": len(evaluation),
        "formal_updates": 256 * len(training),
        "formal_generations": sum(j["tasks"] * j["draws"] for j in evaluation),
        "full_diagnostic_generation_budget": 259656,
        "unavailable": availability.get("unavailable", []),
    }
    record["identity"] = digest(record)
    write_once(run / "REGISTERED_MATRIX.json", record)
    queue = Queue(run / "queue.sqlite", maximum=5)
    try:
        queue.register(jobs, record["identity"])
        if [r["payload"] for r in queue.rows()] != jobs:
            raise ValueError("Queue contains unregistered work")
    finally:
        queue.db.close()
    return {k: v for k, v in record.items() if k != "jobs"}


def registered_queue(run):
    run = Path(run)
    registry = read_json(run / "REGISTERED_MATRIX.json")
    if (
        digest(read_json(run / "FROZEN_PLAN.json")) != registry["plan_hash"]
        or analysis_identity() != registry["analysis_identity"]
        or code_identity() != registry["code_identity"]
    ):
        raise ValueError("Frozen plan or scientific source changed after registration")
    queue = Queue(run / "queue.sqlite", maximum=5)
    if [r["payload"] for r in queue.rows()] != registry["jobs"] or queue.db.execute(
        "SELECT value FROM metadata WHERE key='identity'"
    ).fetchone()[0] != registry["identity"]:
        queue.db.close()
        raise ValueError("Queue differs from frozen matrix")
    return queue
