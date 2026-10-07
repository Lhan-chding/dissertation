"""Registered SER-J2 queue: one independent student per GPU, no adaptive arms."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from ..verified_discovery_transfer.queue import Queue, digest, read_json, write_json


def _lines(path):
    import json

    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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


def build_queue(run):
    from .evaluate import evaluation_job_id

    run = Path(run)
    frozen = read_json(run / "FROZEN_PLAN.json")
    training = _lines(run / "manifests/training_jobs.jsonl")
    evaluation = _lines(run / "manifests/evaluation_jobs.jsonl")
    if len(training) != 18 or len(evaluation) != 95:
        raise ValueError("Expected exactly 18 training and 95 evaluation jobs")
    if sum(j["tasks"] * j["draws"] for j in evaluation) != 145280:
        raise ValueError("Formal response budget differs")
    initial = [train_job_id("S96", 0, a) for a in ("A_LOCAL_C1", "B_FORWARD_C4", "C_FORWARD_C3")]
    initial_diag = [
        evaluation_job_id(j)
        for j in evaluation
        if j.get("parent") == "S96"
        and j.get("block") == 0
        and j["panel"] in ("E_DIAG", "TRAIN_FIT")
    ]
    # Fixed shuffle prevents tying each arm to a physical lane. Priority only
    # implements the preregistered first complete group and diagnostic check.
    shuffled = list(training)
    random.Random(107075).shuffle(shuffled)
    jobs = []
    for j in shuffled:
        first = j["block"] == 0
        jobs.append(
            {
                **j,
                "id": train_job_id(j["parent"], j["block"], j["arm"]),
                "kind": "train",
                "dependencies": [],
                "after": [] if first else initial + initial_diag,
                "priority": 0 if j["parent"] == "S96" and first else 2 if first else 5,
                "gpu_count": 1,
            }
        )
    for j in evaluation:
        student = j.get("block") is not None
        dep = [train_job_id(j["parent"], j["block"], j["arm"])] if student else []
        jobs.append(
            {
                **j,
                "id": evaluation_job_id(j),
                "evaluation_kind": j["kind"],
                "kind": "eval",
                "dependencies": dep,
                "gpu_count": 1,
                "priority": 1
                if student
                and j["parent"] == "S96"
                and j["block"] == 0
                and j["panel"] != "E_CONFIRM"
                else 7
                if j["panel"] == "E_CONFIRM"
                else 3,
            }
        )
    if len({j["id"] for j in jobs}) != 113:
        raise ValueError("Duplicate registered job identity")
    record = {
        "schema": "ser-j2-registered-matrix-v1",
        "jobs": jobs,
        "plan_hash": digest(frozen),
        "analysis_identity": analysis_identity(),
        "code_identity": code_identity(),
        "formal_training_jobs": 18,
        "formal_evaluation_jobs": 95,
        "formal_updates": 4608,
        "formal_generations": 145280,
    }
    record["identity"] = digest(record)
    path = run / "REGISTERED_MATRIX.json"
    if path.exists() and read_json(path) != record:
        raise ValueError("Registered matrix or analysis code changed")
    queue = Queue(run / "queue.sqlite", maximum=5)
    queue.register(jobs, record["identity"])
    write_json(path, record)
    return {k: v for k, v in record.items() if k != "jobs"}


def registered_queue(run):
    run = Path(run)
    registry = read_json(run / "REGISTERED_MATRIX.json")
    if digest(read_json(run / "FROZEN_PLAN.json")) != registry["plan_hash"]:
        raise ValueError("Frozen plan changed after queue registration")
    if analysis_identity() != registry["analysis_identity"]:
        raise ValueError("Analysis source changed after registration")
    if code_identity() != registry["code_identity"]:
        raise ValueError("Scientific source changed after registration")
    queue = Queue(run / "queue.sqlite", maximum=5)
    if [r["payload"] for r in queue.rows()] != registry["jobs"]:
        raise ValueError("Queue differs from registered matrix")
    return queue


def claim_registered(queue, job_id, worker):
    """Claim a named CLI job with the same dependency/concurrency contracts."""
    import time

    queue.db.execute("BEGIN IMMEDIATE")
    try:
        queue._assert_unreleased()
        rows = queue.rows()
        by_id = {row["id"]: row for row in rows}
        row = by_id[job_id]
        if row["status"] == "COMPLETE":
            queue.db.execute("COMMIT")
            return None
        if row["status"] != "PENDING":
            raise ValueError("Job is active or needs an explicit diagnosed retry")
        if sum(r["status"] == "RUNNING" for r in rows) >= queue.maximum:
            raise ValueError("Registered GPU concurrency exhausted")
        payload = row["payload"]
        if any(by_id[d]["status"] != "COMPLETE" for d in payload.get("dependencies", [])):
            raise ValueError("Required checkpoints are not complete")
        if any(
            by_id[d]["status"] not in ("COMPLETE", "BLOCKED_TECHNICAL")
            for d in payload.get("after", [])
        ):
            raise ValueError("Initial registered three-arm group is not terminal")
        queue.db.execute(
            "UPDATE jobs SET status='RUNNING',worker=?,started=? WHERE id=?",
            (worker, time.time(), job_id),
        )
        queue.db.execute("COMMIT")
        return payload
    except BaseException:
        queue.db.execute("ROLLBACK")
        raise
