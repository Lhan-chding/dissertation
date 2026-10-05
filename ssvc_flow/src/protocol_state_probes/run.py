"""Raw-only frozen inference with durable observations and eight-draw commits."""

from __future__ import annotations

import fcntl
import os
import shutil
import socket
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from .protocol import (
    PHASES,
    atomic_json,
    digest,
    file_hash,
    json_bytes,
    load_run,
    read_json,
    safe_component,
    sample_identity,
    write_once_json,
)


class TechnicalDrawFailure(RuntimeError):
    """A draw exhausted its three durable technical attempts."""


def _append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json_bytes(event))
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def lease(path: Path):
    """Kernel leases release on process death and never require unsafe lock deletion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"LEASE_BUSY: {path.name}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def worker_slot(root: Path, maximum: int = 5):
    if not 1 <= maximum <= 5:
        raise ValueError("At most five concurrent GPU workers are permitted")
    streams = []
    selected = None
    try:
        for index in range(maximum):
            path = root / "leases" / f"gpu-slot-{index}.lock"
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a+")
            streams.append(stream)
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            selected = stream
            yield index
            return
        raise RuntimeError("GPU_WORKER_LIMIT: all five experiment leases are active")
    finally:
        if selected is not None:
            fcntl.flock(selected.fileno(), fcntl.LOCK_UN)
        for stream in streams:
            stream.close()


def _commit_path(path: Path) -> Path:
    return path.with_name(path.stem + ".COMMIT.json")


def _validate_rows(rows: list[dict], experiment: str | None = None) -> None:
    seen = {}
    for row in rows:
        key = row["sample_key"]
        if key in seen and seen[key] != row:
            raise ValueError(f"DUPLICATE_SAMPLE_CONFLICT: {key}")
        seen[key] = row
        if experiment is not None:
            expected, seed = sample_identity(
                experiment,
                row["checkpoint"],
                row["canonical_case_owner"],
                row["role"],
                row["draw_index"],
            )
            if key != expected or row["seed"] != seed:
                raise ValueError(f"SAMPLE_IDENTITY_MISMATCH: {key}")


def read_committed_chunk(commit_path: Path, experiment: str | None = None) -> list[dict]:
    commit = read_json(commit_path)
    filename = commit["file"]
    if Path(filename).name != filename:
        raise ValueError(f"Unsafe chunk commit path: {filename}")
    path = commit_path.parent / filename
    if file_hash(path) != commit["sha256"]:
        raise ValueError(f"COMMITTED_CHUNK_HASH_MISMATCH: {path}")
    document = read_json(path)
    rows = document["rows"]
    if (
        len(rows) != commit["n_rows"]
        or [row["sample_key"] for row in rows] != commit["sample_keys"]
    ):
        raise ValueError(f"COMMITTED_CHUNK_INDEX_MISMATCH: {path}")
    _validate_rows(rows, experiment)
    return rows


def committed_rows(root: str | Path, checkpoint: str | None = None) -> list[dict]:
    """Only COMMIT-attested rows are scientific observations; smoke keeps its role."""
    raw = Path(root) / "raw"
    if checkpoint:
        raw /= checkpoint
    unique = {}
    for commit in sorted(raw.glob("**/*.COMMIT.json")):
        for row in read_committed_chunk(commit):
            previous = unique.get(row["sample_key"])
            if previous is not None and previous != row:
                raise ValueError(f"DUPLICATE_SAMPLE_CONFLICT: {row['sample_key']}")
            unique[row["sample_key"]] = row
    return list(unique.values())


def commit_chunk(path: Path, rows: list[dict], identity: str) -> None:
    _validate_rows(rows)
    if len({row["sample_key"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample key within one atomic chunk")
    document = {"schema": "protocol-probe-raw-chunk-v1", "identity": identity, "rows": rows}
    if path.exists() and read_json(path) != document:
        conflict = path.with_name(path.stem + f".CONFLICT-{time.time_ns()}.json")
        atomic_json(conflict, document)
        raise ValueError(f"DUPLICATE_SAMPLE_CONFLICT: retained proposed content in {conflict}")
    write_once_json(path, document)
    commit = {
        "schema": "protocol-probe-chunk-commit-v1",
        "file": path.name,
        "sha256": file_hash(path),
        "n_rows": len(rows),
        "sample_keys": [row["sample_key"] for row in rows],
    }
    write_once_json(_commit_path(path), commit)


def _error_event(root: Path, sample: dict, attempt: int, error: BaseException, status: str) -> dict:
    event = {
        **sample,
        "attempt": attempt,
        "status": status,
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp_ns": time.time_ns(),
    }
    if hasattr(error, "raw_result"):
        from ..modeling_v3.vlm_observation import _fault_json

        event["raw_result"] = _fault_json(error.raw_result)
    _append_event(root / "technical_failures.jsonl", event)
    return event


def _retryable_technical(error: Exception) -> bool:
    """Identity/contract/weight mutations are fatal, not retryable bad samples."""
    from ..modeling_v3.vlm_observation import ObservationFault

    if isinstance(error, ObservationFault):
        return "frozen" not in str(error).lower() and "mutat" not in str(error).lower()
    if isinstance(error, (OSError, MemoryError)) or type(error).__name__ == "OutOfMemoryError":
        return True
    return isinstance(error, RuntimeError) and "cuda out of memory" in str(error).lower()


def _materialize_row(
    context: dict, case: dict, job: dict, sample: dict, raw: dict, scorer: Callable
) -> dict:
    if not isinstance(raw, dict) or not isinstance(raw.get("raw_text"), str):
        raise ValueError("Backend must return a raw_text string")
    for key in ("token_ids", "stop_reason", "elapsed_seconds", "generated_length"):
        if key not in raw:
            raise ValueError(f"Missing raw generation evidence: {key}")
    features = scorer(
        raw["raw_text"],
        case,
        context["audits"][case["case_id"]],
        context["predictions"].get(case["case_id"]),
    )
    exposure = context.get("u22_exposure_audit", {}) if case["panel"] == "U22" else {}
    downgraded = (
        exposure.get("status") == "CONTAMINATED_DOWNGRADED"
        or exposure.get("downgraded_to_development") is True
    )
    return {
        **raw,
        **sample,
        "checkpoint_id": sample["checkpoint"],
        "case_id": case["case_id"],
        "panel": case["panel"],
        "original_split": case["original_split"],
        "split_role": case["split_role"],
        "effective_split_role": "development_diagnostic" if downgraded else case["split_role"],
        "exposure_audit_status": exposure.get("status"),
        "phase": job["phase"],
        "kind": case["kind"],
        "protocol": case["protocol"],
        "base_scene_id": case["base_scene_id"],
        "prompt_identity": case["prompt_identity"],
        "prompt": case["prompt"],
        "output_order": case["output_order"],
        "truncated": raw["stop_reason"] == "length",
        "features": features,
    }


def _draw(
    context: dict,
    backend,
    case: dict,
    job: dict,
    role: str,
    draw: int,
    pending: Path,
    scorer: Callable,
) -> dict:
    key, seed = sample_identity(
        context["protocol"]["protocol_id"], job["checkpoint_id"], case["case_id"], role, draw
    )
    sample = {
        "checkpoint": job["checkpoint_id"],
        "canonical_case_owner": case["case_id"],
        "role": role,
        "draw_index": draw,
        "seed": seed,
        "sample_key": key,
    }
    failure_sample = {
        **sample,
        "case_id": case["case_id"],
        "panel": case["panel"],
        "prompt": case["prompt"],
        "output_order": case["output_order"],
    }
    root = context["root"]
    row_path = pending / f"draw-{draw:06d}.json"
    if row_path.exists():
        row = read_json(row_path)
        _validate_rows([row], context["protocol"]["protocol_id"])
        if any(row[name] != value for name, value in sample.items()):
            raise ValueError(f"PENDING_SAMPLE_IDENTITY_MISMATCH: {row_path}")
        return row
    for attempt in range(3):
        attempt_path = pending / f"draw-{draw:06d}-attempt-{attempt}.json"
        if attempt_path.exists():
            record = read_json(attempt_path)
            if record["sample_key"] != key or record["seed"] != seed:
                raise ValueError("Technical retry changed sample identity")
            if record["status"] == "GENERATED":
                # A crash after generation or a scoring failure never causes another model call.
                row = _materialize_row(context, case, job, sample, record["raw"], scorer)
                write_once_json(row_path, row)
                return row
            if record["status"] == "STARTED":
                event = _error_event(
                    root,
                    failure_sample,
                    attempt,
                    RuntimeError("Process ended before durable generation receipt"),
                    "INTERRUPTED",
                )
                atomic_json(attempt_path, {**record, **event})
            if record["status"] == "FATAL_CONTRACT_FAILURE":
                raise RuntimeError(
                    f"Stored fatal contract failure requires a reviewed new run: {record['error']}"
                )
            continue
        record = {
            **failure_sample,
            "attempt": attempt,
            "status": "STARTED",
            "started_ns": time.time_ns(),
        }
        atomic_json(attempt_path, record)
        try:
            # Only these two strings cross the model boundary; audit/prediction objects do not.
            raw = backend.generate_public(
                {"system": case["prompt"]["system"], "user": case["prompt"]["user"]},
                seed=seed,
                max_new_tokens=64,
            )
        except Exception as error:
            retryable = _retryable_technical(error)
            event = _error_event(
                root,
                failure_sample,
                attempt,
                error,
                "TECHNICAL_FAILURE" if retryable else "FATAL_CONTRACT_FAILURE",
            )
            atomic_json(attempt_path, {**record, **event})
            if not retryable:
                raise
            continue
        atomic_json(attempt_path, {**record, "status": "GENERATED", "raw": raw})
        try:
            row = _materialize_row(context, case, job, sample, raw, scorer)
        except Exception as error:
            _error_event(root, failure_sample, attempt, error, "SCORING_FAILURE_RAW_RETAINED")
            raise
        write_once_json(row_path, row)
        return row
    raise TechnicalDrawFailure(f"Three technical attempts exhausted for {key}; no output imputed")


def _u22_allowed(root: Path) -> tuple[bool, dict]:
    path = root / "U22_EXPOSURE_AUDIT.json"
    if not path.is_file():
        return False, {"status": "U22_EXPOSURE_AUDIT_MISSING"}
    audit = read_json(path)
    # An explicit audit must attest completed record inspection, even when U22 is downgraded.
    verified = audit.get("verified") is True or audit.get("audit_complete") is True
    safe = audit.get("safe_to_generate") is True or audit.get("status") in {
        "VERIFIED_UNTOUCHED",
        "UNTOUCHED_VERIFIED",
        "SAFE",
    }
    downgraded = (
        audit.get("downgraded_to_development") is True
        or audit.get("status") == "CONTAMINATED_DOWNGRADED"
    )
    return verified and (safe or downgraded), audit


def _smoke_jobs(context: dict, checkpoint: str) -> list[dict]:
    selected = []
    for panel in ("CONTROL_COPY", "CONTROL_DUPLICATE"):
        cases = [case for case in context["cases"].values() if case["panel"] == panel]
        scenes = list(dict.fromkeys(case["base_scene_id"] for case in cases))[:2]
        for scene in scenes:
            for protocol in ("O0", "A1"):
                case = next(
                    case
                    for case in cases
                    if case["base_scene_id"] == scene and case["protocol"] == protocol
                )
                selected.append(
                    {
                        "checkpoint_id": checkpoint,
                        "case_id": case["case_id"],
                        "panel": panel,
                        "phase": "technical_smoke",
                        "protocol": protocol,
                        "draws": 2,
                    }
                )
    if len(selected) != 8:
        raise ValueError(
            "Smoke requires two fixed copy and two duplicate scenes in both output orders"
        )
    return selected


def _lane_record(context: dict, checkpoint: str, lane_id: str) -> tuple[Path, dict]:
    directory = context["root"] / "smoke_lanes"
    with lease(context["root"] / "leases" / "smoke-lane-registry.lock"):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{safe_component(lane_id)}.json"
        if path.exists():
            return path, read_json(path)
        if len(list(directory.glob("*.json"))) >= 5:
            raise ValueError("SMOKE_LIMIT: five physical lanes already registered")
        record = {
            "lane_id": lane_id,
            "first_checkpoint": checkpoint,
            "max_outputs": 16,
            "status": "STARTED",
        }
        write_once_json(path, record)
        return path, record


def _checkpoint_record(context: dict, checkpoint: str) -> dict:
    receipt = read_json(context["root"] / "CHECKPOINT_AVAILABILITY.json")
    records = receipt.get("checkpoints", receipt.get("records", []))
    matches = [
        record for record in records if record.get("checkpoint_id", record.get("id")) == checkpoint
    ]
    if len(matches) != 1 or matches[0].get("status") != "AVAILABLE":
        raise ValueError(f"CHECKPOINT_UNAVAILABLE: {checkpoint}; new training is prohibited")
    return matches[0]


def _assert_no_training(backend) -> dict:
    counters = dict(backend.counters)
    for name in ("backward_calls", "optimizer_updates"):
        if counters.get(name) != 0:
            raise RuntimeError(f"FROZEN_INFERENCE_VIOLATION: {name}={counters.get(name)}")
    return counters


def _resource_metrics(rows: list[dict]) -> dict:
    """Summarize measured generation only; memory peaks include the model load."""
    lengths = sorted(row["generated_length"] for row in rows)

    def percentile(fraction):
        index = (len(lengths) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(lengths) - 1)
        return lengths[lower] + (lengths[upper] - lengths[lower]) * (index - lower)

    seconds = sum(row["elapsed_seconds"] for row in rows)
    result = {
        "generation_seconds": seconds,
        "generation_seconds_per_output": seconds / len(rows),
        "generation_timing_scope": "backend_generation_only_excludes_loading_and_commit_io",
        "generated_tokens": sum(lengths),
        "generated_length_p50": percentile(0.5),
        "generated_length_p90": percentile(0.9),
        "memory_peak_scope": "process_since_historical_loader_reset_before_model_load",
    }
    for key in ("peak_memory_allocated_bytes", "peak_memory_reserved_bytes"):
        observed = [row[key] for row in rows if row.get(key) is not None]
        result[key] = max(observed) if observed else None
        result[key + "_observed_outputs"] = len(observed)
    return result


def _first_block_receipt(root: Path, checkpoint: str, role: str, identity: str, rows: list[dict]):
    if len(rows) != 8:
        return
    path = root / "first_block_receipts" / checkpoint / f"{role}.json"
    if path.exists():
        if read_json(path)["identity"] != identity:
            raise ValueError(f"First resource block identity changed: {path}")
        return
    write_once_json(
        path,
        {
            "schema": "protocol-probe-first-eight-resources-v1",
            "identity": identity,
            "checkpoint": checkpoint,
            "role": role,
            "committed_outputs": 8,
            "sample_keys": [row["sample_key"] for row in rows],
            "generated_lengths": [row["generated_length"] for row in rows],
            **_resource_metrics(rows),
        },
    )


def run_worker(
    run_path: str | Path,
    checkpoint: str,
    *,
    runtime_path: str | Path | None = None,
    allow_gpu: bool = False,
    resume: bool = False,
    phase: str | None = None,
    smoke: bool = False,
    with_smoke: bool = False,
    lane_id: str | None = None,
    backend_factory=None,
    scorer=None,
) -> dict:
    """Execute only registered cells, preserving wrong/invalid answers without retries."""
    if not allow_gpu:
        raise ValueError("Frozen GPU execution requires --allow-gpu")
    context = load_run(run_path, runtime_path)
    root = context["root"]
    if phase and phase not in PHASES:
        raise ValueError(f"Unknown registered phase: {phase}")
    record = _checkpoint_record(context, checkpoint)
    if context["runtime_path"] is None:
        raise ValueError("Supply the historical --runtime configuration before GPU execution")
    if backend_factory is None:
        from .inference import FrozenBackend

        backend_factory = FrozenBackend
    if scorer is None:
        from .semantics import score_raw

        scorer = score_raw
    if smoke and with_smoke:
        raise ValueError("Select standalone smoke or worker with smoke")
    smoke_jobs = []
    if smoke or with_smoke:
        lane_id = (
            lane_id
            or f"{socket.gethostname()}:{os.environ.get('CUDA_VISIBLE_DEVICES', 'UNSPECIFIED')}"
        )
        if lane_id.endswith(":UNSPECIFIED"):
            raise ValueError(
                "Supply --lane-id or the allocated CUDA_VISIBLE_DEVICES for bounded smoke"
            )
        lane_path, lane = _lane_record(context, checkpoint, lane_id)
        if lane["first_checkpoint"] != checkpoint:
            if smoke:
                return {
                    "status": "SMOKE_ALREADY_ASSIGNED_TO_LANE_FIRST_CHECKPOINT",
                    "lane_id": lane_id,
                    "first_checkpoint": lane["first_checkpoint"],
                    "new_outputs": 0,
                }
            lane_path = lane = None
        else:
            smoke_jobs = [{**job, "_role": "smoke"} for job in _smoke_jobs(context, checkpoint)]
    else:
        lane_path = lane = None
    jobs = (
        smoke_jobs
        if smoke
        else smoke_jobs
        + [
            {**job, "_role": "frozen_probe"}
            for job in context["manifest"]["jobs"]
            if job["checkpoint_id"] == checkpoint and (phase is None or job["phase"] == phase)
        ]
    )
    if not jobs:
        raise ValueError("No preregistered jobs selected")
    role = "smoke" if smoke else "frozen_probe"
    jobs.sort(
        key=lambda job: (
            PHASES.index(job["phase"]) if job["phase"] in PHASES else -1,
            job["panel"],
            job["case_id"],
        )
    )
    u22_ok, u22_audit = _u22_allowed(root)
    context["u22_exposure_audit"] = u22_audit
    blocked = [
        {**job, "status": "BLOCKED_U22_EXPOSURE_AUDIT", "reason": u22_audit.get("status")}
        for job in jobs
        if job["panel"] == "U22" and not u22_ok
    ]
    jobs = [job for job in jobs if job["panel"] != "U22" or u22_ok]
    runtime_hash = file_hash(context["runtime_path"])
    identity = digest(
        {
            "run_identity": context["manifest"]["identity"],
            "checkpoint": record,
            "runtime_sha256": runtime_hash,
        }
    )
    binding_path = root / "checkpoint_bindings" / f"{checkpoint}.json"
    binding = {
        "identity": identity,
        "checkpoint": checkpoint,
        "checkpoint_record": record,
        "runtime_sha256": runtime_hash,
        "run_identity": context["manifest"]["identity"],
    }
    results = []
    first_rows = {"smoke": [], "frozen_probe": []}
    started = time.monotonic()
    backend = None
    with (
        worker_slot(root),
        lease(root / "leases" / f"checkpoint-{safe_component(checkpoint)}.lock"),
    ):
        write_once_json(binding_path, binding)
        existing_rows = committed_rows(root, checkpoint)
        selected_roles = {job["_role"] for job in jobs}
        selected_existing = [row for row in existing_rows if row["role"] in selected_roles]
        if selected_existing and not resume:
            raise ValueError("Existing observations require --resume; no output is overwritten")
        if (smoke or with_smoke) and sum(row["role"] == "smoke" for row in existing_rows) > 16:
            raise ValueError("SMOKE_LIMIT: checkpoint already exceeds 16 technical outputs")
        # Completed or entirely blocked selections do not instantiate a model.
        pending_work = False
        for job in jobs:
            chunks = (
                root / "raw" / checkpoint / job["panel"] / safe_component(job["case_id"]) / "chunks"
            )
            if any(
                not _commit_path(chunks / f"{job['_role']}-{start:06d}.json").exists()
                for start in range(0, job["draws"], 8)
            ):
                pending_work = True
        try:
            if pending_work:
                usage = shutil.disk_usage(root)
                if usage.free < 256 * 1024 * 1024:
                    raise OSError(
                        "Less than 256 MiB free before phase; raw output preservation is at risk"
                    )
                backend = backend_factory(
                    context["runtime_path"], record, context["protocol"], allow_gpu=True
                )
                receipt = dict(backend.receipt)
                receipt_path = (
                    root / "backend_receipts" / checkpoint / f"{role}-{time.time_ns()}.json"
                )
                atomic_json(receipt_path, receipt)
                _assert_no_training(backend)
            for job_index, job in enumerate(jobs):
                row_role = job["_role"]
                owner = context["aliases"][job["case_id"]]
                if owner != job["case_id"]:
                    raise ValueError("Logical job unexpectedly schedules a reused alias")
                case = context["cases"][owner]
                chunks = root / "raw" / checkpoint / job["panel"] / safe_component(owner) / "chunks"
                failed = False
                completed = 0
                for start in range(0, job["draws"], 8):
                    path = chunks / f"{row_role}-{start:06d}.json"
                    commit_path = _commit_path(path)
                    expected_draws = list(range(start, min(start + 8, job["draws"])))
                    if commit_path.exists():
                        rows = read_committed_chunk(commit_path, context["protocol"]["protocol_id"])
                        if (
                            read_json(path).get("identity") != identity
                            or [row["draw_index"] for row in rows] != expected_draws
                        ):
                            raise ValueError(f"Committed chunk run identity/range changed: {path}")
                        completed += len(rows)
                        if len(first_rows[row_role]) < 8:
                            first_rows[row_role].extend(rows[: 8 - len(first_rows[row_role])])
                            _first_block_receipt(
                                root, checkpoint, row_role, identity, first_rows[row_role]
                            )
                        continue
                    pending = chunks / f"{row_role}-{start:06d}.pending"
                    try:
                        rows = [
                            _draw(context, backend, case, job, row_role, draw, pending, scorer)
                            for draw in expected_draws
                        ]
                    except TechnicalDrawFailure as error:
                        results.append(
                            {
                                **job,
                                "role": row_role,
                                "status": "TECHNICAL_FAILURE",
                                "error": str(error),
                                "completed_draws": completed,
                            }
                        )
                        failed = True
                        break
                    commit_chunk(path, rows, identity)
                    completed += len(rows)
                    _assert_no_training(backend)
                    if len(first_rows[row_role]) < 8:
                        first_rows[row_role].extend(rows[: 8 - len(first_rows[row_role])])
                        _first_block_receipt(
                            root, checkpoint, row_role, identity, first_rows[row_role]
                        )
                    _append_event(
                        root / "progress.jsonl",
                        {
                            "checkpoint": checkpoint,
                            "role": row_role,
                            "phase": job["phase"],
                            "case_id": owner,
                            "draws_committed": completed,
                            "chunk_outputs": len(rows),
                            "elapsed_seconds": time.monotonic() - started,
                            **_resource_metrics(rows),
                            "generated_lengths": [row["generated_length"] for row in rows],
                            "timestamp_ns": time.time_ns(),
                        },
                    )
                if not failed:
                    results.append(
                        {
                            **job,
                            "role": row_role,
                            "status": "COMPLETE",
                            "completed_draws": completed,
                        }
                    )
                if job_index == len(jobs) - 1 or jobs[job_index + 1]["phase"] != job["phase"]:
                    phase_results = [
                        result for result in results if result["phase"] == job["phase"]
                    ]
                    phase_status = (
                        "COMPLETE"
                        if all(result["status"] == "COMPLETE" for result in phase_results)
                        else "PARTIAL_WITH_EXPLICIT_BLOCKERS"
                    )
                    phase_receipt = {
                        "status": phase_status,
                        "checkpoint": checkpoint,
                        "phase": job["phase"],
                        "role": row_role,
                        "identity": identity,
                        "jobs": phase_results,
                        "committed_outputs": sum(
                            result["completed_draws"] for result in phase_results
                        ),
                    }
                    atomic_json(
                        root / "phase_receipts" / checkpoint / f"{job['phase']}.json", phase_receipt
                    )
                    if job["phase"] == "technical_smoke" and lane_path is not None:
                        atomic_json(
                            lane_path,
                            {
                                **lane,
                                "status": phase_status,
                                "committed_outputs": phase_receipt["committed_outputs"],
                            },
                        )
                    if job["phase"] == "technical_smoke" and phase_status != "COMPLETE":
                        blocked.extend(
                            {**later, "status": "BLOCKED_SMOKE_TECHNICAL_FAILURE"}
                            for later in jobs[job_index + 1 :]
                        )
                        raise RuntimeError(
                            "TECHNICAL_SMOKE_FAILED: main cells remain pending; "
                            "wrong or invalid mathematical answers never trigger this gate"
                        )
            counters = (
                _assert_no_training(backend)
                if backend is not None
                else {"backward_calls": 0, "optimizer_updates": 0, "model_not_instantiated": True}
            )
            status = (
                "COMPLETE"
                if not blocked and all(row["status"] == "COMPLETE" for row in results)
                else "PARTIAL_WITH_EXPLICIT_BLOCKERS"
            )
            receipt = {
                "status": status,
                "checkpoint": checkpoint,
                "role": role,
                "phase": phase,
                "identity": identity,
                "jobs": results + blocked,
                "counters": counters,
                "elapsed_seconds": time.monotonic() - started,
                "u22_exposure_audit": u22_audit,
            }
            atomic_json(
                root / "worker_receipts" / f"{checkpoint}-{role}-{phase or 'all'}.json", receipt
            )
            if smoke and lane_path is not None:
                atomic_json(
                    lane_path,
                    {
                        **lane,
                        "status": status,
                        "committed_outputs": sum(row.get("completed_draws", 0) for row in results),
                    },
                )
            return receipt
        except BaseException as error:
            emergency = {
                "status": "WORKER_INTERRUPTED",
                "checkpoint": checkpoint,
                "role": role,
                "phase": phase,
                "identity": identity,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "jobs": results + blocked,
                "counters": dict(backend.counters)
                if backend is not None
                else {"model_not_instantiated": True},
            }
            atomic_json(
                root / "worker_receipts" / f"{checkpoint}-{role}-interrupted-{time.time_ns()}.json",
                emergency,
            )
            raise
