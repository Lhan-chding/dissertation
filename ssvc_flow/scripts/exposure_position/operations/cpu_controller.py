#!/usr/bin/env python3
"""Metadata-only, one-run CPU controller. Never releases, cancels or retries work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

TERMINAL = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
)
SCIENCE_STATES = frozenset({"PENDING", "RUNNING", "UNKNOWN", "COMPLETE", "BLOCKED_TECHNICAL"})
SACCT_FIELDS = ("JobIDRaw", "State", "ExitCode", "ElapsedRaw", "AllocTRES", "Start", "End")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_json(value):
    return digest_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def file_sha(path):
    return digest_bytes(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once_bytes(path, data):
    """Atomic no-clobber publication, even when duplicate content is identical."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_once_json(path, value):
    write_once_bytes(
        path,
        (
            json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode(),
    )


def allocation_ids(values):
    values = [str(value) for value in values]
    if (
        not values
        or len(set(values)) != len(values)
        or any(not re.fullmatch(r"[1-9][0-9]*", value) for value in values)
    ):
        raise ValueError(
            "Provide unique positive numeric allocation IDs; no ranges, arrays or steps"
        )
    return values


def require_cpu_allocation(environ=None):
    """Reject explicit GPU allocations; do not import any GPU runtime."""
    env = os.environ if environ is None else environ
    for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS"):
        value = env.get(key, "").strip()
        if value and value not in (
            {"(null)", "N/A"} if key == "SLURM_JOB_GPUS" else {"0", "(null)", "N/A"}
        ):
            raise PermissionError("Controller must have zero allocated GPUs: " + key)


def read_science_snapshot(run, bindings):
    run = Path(run)
    for relative, sha in bindings.items():
        if file_sha(run / relative) != sha:
            raise ValueError("Registered metadata changed during controller session: " + relative)
    registry = read_json(run / "REGISTERED_MATRIX.json")
    frozen = read_json(run / "FROZEN_PLAN.json")
    if registry.get("plan_hash") != digest_json(frozen):
        raise ValueError("Registered matrix does not bind the frozen plan")
    database = run / "queue.sqlite"
    connection = sqlite3.connect(
        "file:" + quote(str(database.resolve())) + "?mode=ro",
        uri=True,
        timeout=10,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT id,payload,status,worker,started,finished FROM jobs ORDER BY rowid"
            )
        ]
        connection.execute("COMMIT")
    finally:
        connection.close()
    if metadata.get("identity") != registry.get("identity"):
        raise ValueError("SQLite registration identity differs")
    for row in rows:
        row["payload"] = json.loads(row["payload"])
    expected = registry.get("jobs", [])
    if (
        not expected
        or [row["payload"] for row in rows] != expected
        or len({row["id"] for row in rows}) != len(rows)
        or any(row["id"] != row["payload"].get("id") for row in rows)
        or any(row["status"] not in SCIENCE_STATES for row in rows)
    ):
        raise ValueError("SQLite scientific matrix/status differs from registration")
    return {
        "observed_at_utc": utc_now(),
        "counts": dict(Counter(row["status"] for row in rows)),
        "jobs": [{key: value for key, value in row.items() if key != "payload"} for row in rows],
        "all_complete": all(row["status"] == "COMPLETE" for row in rows),
        "registered_jobs": len(rows),
        "registry_identity": registry["identity"],
        "frozen_plan_hash": frozen.get("plan_hash"),
        "queue_released": "released" in metadata,
    }


def subprocess_query(command, timeout):
    began = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "error_type": None,
            "elapsed_seconds": time.monotonic() - began,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "returncode": None,
            "stdout": getattr(exc, "stdout", None) or b"",
            "stderr": getattr(exc, "stderr", None) or str(exc).encode(),
            "error_type": type(exc).__name__,
            "elapsed_seconds": time.monotonic() - began,
        }


def query_scheduler(ids, directory, *, timeout=20, runner=subprocess_query):
    ids = allocation_ids(ids)
    commands = {
        "sacct": [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs=" + ",".join(ids),
            "--format=JobIDRaw,State%64,ExitCode,ElapsedRaw,AllocTRES%256,Start,End",
        ],
        "squeue": ["squeue", "--noheader", "--jobs=" + ",".join(ids), "--format=%i|%T"],
    }
    results = {}
    for name, command in commands.items():
        record = runner(command, timeout)
        stdout, stderr = record.pop("stdout"), record.pop("stderr")
        if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
            raise TypeError("Scheduler runner must preserve raw stdout/stderr bytes")
        write_once_bytes(directory / (name + ".stdout.txt"), stdout)
        write_once_bytes(directory / (name + ".stderr.txt"), stderr)
        metadata = {
            **record,
            "command": command,
            "observed_at_utc": utc_now(),
            "stdout_sha256": digest_bytes(stdout),
            "stderr_sha256": digest_bytes(stderr),
        }
        write_once_json(directory / (name + ".json"), metadata)
        results[name] = {**metadata, "stdout": stdout.decode("utf-8", errors="replace")}
    return scheduler_observations(ids, results), results


def scheduler_observations(ids, results):
    """A failed query or ambiguous/missing record never proves a worker dead."""
    failures = [
        name
        for name, row in results.items()
        if row.get("returncode") != 0 or row.get("error_type") is not None
    ]
    if failures:
        return {
            identifier: {
                "observation": "UNKNOWN",
                "terminal_verified": False,
                "reason": "scheduler_query_failed",
                "failed_queries": failures,
            }
            for identifier in ids
        }
    sacct_rows, squeue_rows, malformed = {}, {}, []
    for line in results["sacct"]["stdout"].splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) == len(SACCT_FIELDS) + 1 and fields[-1] == "":
            fields.pop()
        if len(fields) != len(SACCT_FIELDS):
            malformed.append({"query": "sacct", "line": line})
            continue
        if fields[0] in ids:
            sacct_rows.setdefault(fields[0], []).append(
                dict(zip(SACCT_FIELDS, fields, strict=True))
            )
        # Job steps remain in raw evidence but never become duplicate allocations.
    for line in results["squeue"]["stdout"].splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 2:
            malformed.append({"query": "squeue", "line": line})
        elif fields[0] in ids:
            squeue_rows.setdefault(fields[0], []).append(fields[1])
    observations = {}
    for identifier in ids:
        records = sacct_rows.get(identifier, [])
        queued = squeue_rows.get(identifier, [])
        observation = {
            "observation": "UNKNOWN",
            "terminal_verified": False,
            "sacct_records": records,
            "squeue_states": queued,
        }
        if malformed:
            observation.update(reason="scheduler_output_malformed", malformed=malformed)
        elif queued:
            observation.update(observation="PRESENT_IN_SQUEUE", reason="allocation_still_present")
        elif len(records) != 1:
            observation.update(reason="missing_or_ambiguous_sacct_allocation")
        else:
            record = records[0]
            state = record["State"].split()[0].removesuffix("+") if record["State"] else ""
            try:
                datetime.fromisoformat(record["End"])
                valid_end = True
            except (TypeError, ValueError):
                valid_end = False
            valid_receipt = (
                re.fullmatch(r"[0-9]+:[0-9]+", record["ExitCode"]) is not None
                and record["ElapsedRaw"].isdigit()
                and valid_end
            )
            if state in TERMINAL and valid_receipt:
                observation.update(
                    observation="TERMINAL_VERIFIED",
                    terminal_verified=True,
                    state=state,
                    reason="terminal_sacct_and_absent_from_squeue",
                )
            else:
                observation.update(reason="sacct_not_verified_terminal", state=state)
        observations[identifier] = observation
    return observations


def task_allocations(run, snapshot, ids):
    mappings = {}
    attempts = []
    for path in sorted((Path(run) / "worker_attempts").glob("*/STARTED.json")):
        try:
            record = read_json(path)
        except (OSError, ValueError):
            continue
        attempts.append(record)
    for row in snapshot["jobs"]:
        if row["status"] != "RUNNING":
            continue
        found = {
            str(attempt.get("slurm_job_id"))
            for attempt in attempts
            if attempt.get("job_id") == row["id"]
            and attempt.get("worker") == row["worker"]
            and str(attempt.get("slurm_job_id")) in ids
        }
        worker_parts = str(row["worker"]).rsplit(":", 2)
        if len(worker_parts) == 3 and worker_parts[1] in ids and worker_parts[2].isdigit():
            found.add(worker_parts[1])
        mappings[row["id"]] = next(iter(found)) if len(found) == 1 else None
    return mappings


def stop_reasons(snapshot, scheduler, mappings):
    reasons = []
    for row in snapshot["jobs"]:
        if row["status"] in {"BLOCKED_TECHNICAL", "UNKNOWN"}:
            reasons.append(
                {
                    "kind": "SCIENTIFIC_TASK_REQUIRES_OPERATOR",
                    "job_id": row["id"],
                    "scientific_status": row["status"],
                    "worker": row["worker"],
                }
            )
        elif row["status"] == "RUNNING":
            identifier = mappings.get(row["id"])
            if identifier is not None and scheduler[identifier]["terminal_verified"]:
                reasons.append(
                    {
                        "kind": "TERMINAL_ALLOCATION_WITH_RUNNING_SCIENTIFIC_TASK",
                        "job_id": row["id"],
                        "worker": row["worker"],
                        "slurm_job_id": identifier,
                        "scheduler": scheduler[identifier],
                    }
                )
    return reasons


def publish_stop(run, event):
    path = Path(run) / "STOP"
    try:
        write_once_json(path, event)
        created = True
    except FileExistsError:
        created = False
    return {
        "created": created,
        "path": str(path),
        "sha256": file_sha(path),
        "existing_stop_preserved": not created,
    }


def start_session(run, receipts_root, session_id, ids, poll_seconds):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", session_id):
        raise ValueError("Unsafe session ID")
    if (
        type(poll_seconds) not in (int, float)
        or not math.isfinite(poll_seconds)
        or poll_seconds < 1
    ):
        raise ValueError("Polling interval must be finite and at least one second")
    run = Path(run).resolve()
    if not run.is_dir() or not (run / "queue.sqlite").is_file():
        raise ValueError("An existing registered run/queue is required")
    ids = allocation_ids(ids)
    bindings = {
        relative: file_sha(run / relative)
        for relative in ("FROZEN_PLAN.json", "REGISTERED_MATRIX.json")
    }
    snapshot = read_science_snapshot(run, bindings)
    directory = Path(receipts_root).resolve() / session_id
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()  # Existing sessions are never resumed or overwritten.
    config = {
        "schema": "ser-j23-cpu-operations-controller-v1",
        "session_id": session_id,
        "created_at_utc": utc_now(),
        "run": str(run),
        "worker_allocation_ids": ids,
        "poll_seconds": poll_seconds,
        "metadata_bindings": bindings,
        "frozen_plan_hash": snapshot["frozen_plan_hash"],
        "controller_sha256": file_sha(__file__),
        "hostname": socket.gethostname(),
        "controller_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_model_calls": 0,
        "new_scientific_budget": 0,
        "may_create_STOP": True,
        "may_release": False,
        "may_cancel": False,
        "may_retry": False,
        "may_mutate_queue": False,
        "scheduler_accounting_certified_by_controller": False,
    }
    write_once_json(directory / "CONFIG.json", config)
    return config, directory


def observe(config, directory, poll_index, *, runner=subprocess_query):
    directory = Path(directory)
    poll = directory / "polls" / f"{poll_index:06d}"
    poll.mkdir(parents=True)  # Duplicate polling IDs fail before overwriting evidence.
    snapshot = read_science_snapshot(config["run"], config["metadata_bindings"])
    write_once_json(poll / "SQLITE_SNAPSHOT_BEFORE_QUERIES.json", snapshot)
    scheduler, _ = query_scheduler(config["worker_allocation_ids"], poll, runner=runner)
    # A worker can commit its final task while sacct/squeue are queried. Decide
    # from a fresh snapshot so normal completion never creates a stale STOP.
    snapshot = read_science_snapshot(config["run"], config["metadata_bindings"])
    write_once_json(poll / "SQLITE_SNAPSHOT.json", snapshot)
    mappings = task_allocations(config["run"], snapshot, config["worker_allocation_ids"])
    reasons = stop_reasons(snapshot, scheduler, mappings)
    event = {
        "schema": "ser-j23-cpu-controller-observation-v1",
        "observed_at_utc": utc_now(),
        "poll_index": poll_index,
        "poll_directory": str(poll),
        "scientific_counts": snapshot["counts"],
        "all_scientific_complete": snapshot["all_complete"],
        "scheduler": scheduler,
        "running_task_allocation_mapping": mappings,
        "stop_reasons": reasons,
        "release_performed": False,
        "queue_modified": False,
        "scheduler_accounting_certified": False,
    }
    if reasons:
        stop_record = {
            "status": "STOP_REQUESTED_AT_TASK_BOUNDARY",
            "created_at_utc": utc_now(),
            "controller_session": config["session_id"],
            "reasons": reasons,
            "evidence_directory": str(poll),
            "plan_hash": config["frozen_plan_hash"],
            "inflight_tasks_may_finish": True,
            "automatic_retry_permitted": False,
        }
        event["stop"] = publish_stop(config["run"], stop_record)
    elif (Path(config["run"]) / "STOP").exists():
        event["stop"] = {
            "created": False,
            "existing_stop_preserved": True,
            "path": str(Path(config["run"]) / "STOP"),
            "sha256": file_sha(Path(config["run"]) / "STOP"),
        }
    terminal = all(value["terminal_verified"] for value in scheduler.values())
    if snapshot["all_complete"]:
        event["status"] = "SCIENTIFIC_MATRIX_COMPLETE_CANDIDATE"
    elif event.get("stop") and terminal:
        event["status"] = "STOPPED_WORKER_ALLOCATIONS_TERMINAL"
    elif terminal:
        event["status"] = "INCOMPLETE_NO_LIVE_WORKER_ALLOCATIONS"
    elif event.get("stop"):
        event["status"] = "STOP_REQUESTED_DRAINING"
    else:
        event["status"] = "OBSERVING"
    write_once_json(poll / "OBSERVATION.json", event)
    return event


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--worker-job-ids", nargs="+", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--receipts-root", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument(
        "--once", action="store_true", help="One metadata observation; never implies completion"
    )
    args = parser.parse_args(argv)
    require_cpu_allocation()
    config, directory = start_session(
        args.run, args.receipts_root, args.session_id, args.worker_job_ids, args.poll_seconds
    )
    poll_index = 0
    try:
        while True:
            event = observe(config, directory, poll_index)
            print(
                json.dumps(
                    {key: event[key] for key in ("poll_index", "status", "scientific_counts")}
                ),
                flush=True,
            )
            if (
                event["status"]
                in {
                    "SCIENTIFIC_MATRIX_COMPLETE_CANDIDATE",
                    "STOPPED_WORKER_ALLOCATIONS_TERMINAL",
                    "INCOMPLETE_NO_LIVE_WORKER_ALLOCATIONS",
                }
                or args.once
            ):
                final = {
                    "status": event["status"],
                    "finished_at_utc": utc_now(),
                    "last_poll": poll_index,
                    "scientific_release_performed": False,
                    "STAGE_COMPLETE_asserted": False,
                    "scheduler_accounting_certified": False,
                    "last_observation_sha256": file_sha(
                        directory / "polls" / f"{poll_index:06d}" / "OBSERVATION.json"
                    ),
                }
                write_once_json(directory / "FINAL.json", final)
                return 0 if event["status"] == "SCIENTIFIC_MATRIX_COMPLETE_CANDIDATE" else 2
            time.sleep(args.poll_seconds)
            poll_index += 1
    except BaseException as exc:
        write_once_json(
            directory / "CONTROLLER_INTERRUPTED.json",
            {
                "status": "CONTROLLER_OBSERVATION_INCOMPLETE",
                "observed_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "poll_index": poll_index,
                "scientific_outcome": "UNKNOWN_FROM_CONTROLLER",
                "release_performed": False,
                "automatic_resubmission": False,
            },
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
