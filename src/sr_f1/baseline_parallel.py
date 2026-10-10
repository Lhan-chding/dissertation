"""Identity-bound baseline shards in independent, single-visible-GPU processes.

Only the registered BASELINE parent uses this module. It never loads a model;
children retain the original per-slot sampling and append-only evaluation path.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from mm_dev.orchestration import atomic_json

from .evaluation import (
    BASELINE,
    encoded,
    evaluate_slots,
    evaluation_model_identity,
    iter_model_slots,
    read_jsonl,
    sha_file,
    validate_raw,
)

TOTAL_SLOTS = 6784
DIRECTORY = "evaluation/baseline_parallel"
INPUT_FILES = (
    "manifests/TASKS_GOLD_AUDIT_ONLY.jsonl",
    "manifests/MODEL_INPUTS.jsonl",
    "manifests/DIAGNOSTIC_INPUTS_AUDIT_ONLY.jsonl",
    "manifests/TRAIN_FIT_QIDS.json",
    "COMMON_START.json",
    "EXECUTION_FREEZE.json",
)


def _hash(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _count(count):
    if type(count) is not int or not 1 <= count <= 5:
        raise ValueError("Baseline requires one to five GPUs")


def _bounded(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise PermissionError("Baseline artifact escaped run root")
    return path


def baseline_slots(root):
    root = Path(root)
    tasks = list(read_jsonl(root / INPUT_FILES[0]))
    fit = _read(root / INPUT_FILES[3])
    slots = list(iter_model_slots(tasks, fit, BASELINE, stage="baseline", step=0))
    if len(slots) != TOTAL_SLOTS or len({s["slot_id"] for s in slots}) != TOTAL_SLOTS:
        raise ValueError("Baseline must contain exactly 6784 unique registered slots")
    if any(s["pool"] not in {"MONITOR", "MONITOR_DIAGNOSTIC_ONLY", "TRAIN_FIT"} for s in slots):
        raise PermissionError("TEST is sealed during baseline")
    return slots


def shard_slots(slots, rank, count):
    _count(count)
    if type(rank) is not int or not 0 <= rank < count:
        raise ValueError("Invalid baseline shard rank")
    return list(slots)[rank::count]


def raw_path(root, rank, count):
    shard_slots([], rank, count)
    return _bounded(
        root,
        f"raw/evaluation/{BASELINE}_step0_baseline_shard{rank:02d}_of{count:02d}.jsonl",
    )


def prepare_manifest(plan, root, count):
    """Freeze shard ownership once; later attempts must retain the same layout."""
    _count(count)
    root = Path(root).resolve(strict=True)
    slots = baseline_slots(root)
    canonical = root / "raw/evaluation" / f"{BASELINE}_step0_baseline.jsonl"
    if canonical.exists():
        raise PermissionError("Existing serial baseline cannot be resampled as parallel shards")
    expected = {
        "schema_version": 1,
        "run_root": str(root),
        "model_id": BASELINE,
        "stage": "baseline",
        "step": 0,
        "gpu_count": count,
        "partition": "original_slot_index_mod_gpu_count",
        "total_slots": TOTAL_SLOTS,
        "slots_sha256": _hash(slots),
        "plan_sha256": _hash(plan),
        "input_file_hashes": {name: sha_file(_bounded(root, name)) for name in INPUT_FILES},
        "shards": [
            {
                "rank": rank,
                "slots_sha256": _hash(shard_slots(slots, rank, count)),
                "count": len(shard_slots(slots, rank, count)),
                "raw_path": str(raw_path(root, rank, count).relative_to(root)),
            }
            for rank in range(count)
        ],
        "scores_released": False,
    }
    directory = _bounded(root, DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "PLAN.json"
    if manifest.exists():
        if _read(manifest) != expected:
            raise PermissionError("Immutable baseline shard/input/plan identity changed")
    else:
        # A raw file without its original ownership receipt is never adopted.
        if any((root / "raw/evaluation").glob(f"{BASELINE}_step0_baseline_shard*.jsonl")):
            raise PermissionError("Baseline shard outputs exist without their ownership receipt")
        atomic_json(manifest, expected, exclusive=True)
    plan_path = directory / "RUNTIME_PLAN.json"
    if plan_path.exists():
        if _read(plan_path) != plan:
            raise PermissionError("Frozen baseline runtime plan changed")
    else:
        atomic_json(plan_path, plan, exclusive=True)
    return expected, slots


def _input_maps(root):
    inputs = {row["qid"]: row for row in read_jsonl(root / INPUT_FILES[1])}
    diagnostics = {(row["qid"], row["view"]): row for row in read_jsonl(root / INPUT_FILES[2])}
    return inputs, diagnostics


def _validate_input(root, row, slot, inputs, diagnostics, image_hashes):
    request = diagnostics[(slot["qid"], slot["view"])] if slot["view"] else inputs[slot["qid"]]
    if row.get("image_file") != request["image_file"]:
        raise ValueError("Baseline image request changed")
    if slot["view"] == "neutral_hint":
        prefix = request["text"] + "\n\n"
        text = row.get("text", "")
        suffix = text[len(prefix) :]
        candidate = request["runtime_neutral_padding"]["candidate"]
        if not text.startswith(prefix) or not (
            candidate * (len(suffix) // len(candidate) + 1)
        ).startswith(suffix):
            raise ValueError("Baseline neutral text differs from frozen template")
        if abs(row.get("token_matching", {}).get("token_difference", 100)) > 1:
            raise ValueError("Baseline neutral text is not tokenizer matched")
    elif row.get("text") != request["text"]:
        raise ValueError("Baseline input text changed")
    if slot["view"] and row.get("information_granted") != request["information_granted"]:
        raise ValueError("Baseline diagnostic information changed")
    image = request["image_file"]
    if image not in image_hashes:
        image_hashes[image] = sha_file(_bounded(root, image)) if image is not None else None
    if row.get("image_hash") != image_hashes[image] or row.get("input_hash") != _hash(
        dict(text=row["text"], image_hash=image_hashes[image])
    ):
        raise ValueError("Baseline input/image hash mismatch")


def validate_baseline_outputs(root, slots, count, *, require_complete=True):
    """Recount exact raw ownership and provenance without scoring or TEST access."""
    root = Path(root).resolve()
    _count(count)
    inputs, diagnostics = _input_maps(root)
    amendment = _read(root / "AMENDMENT.json") if (root / "AMENDMENT.json").exists() else None
    amendment_id = amendment.get("amendment_id", amendment.get("id")) if amendment else None
    common = _read(root / "COMMON_START.json")
    seen, image_hashes, identity, adapter, summaries = set(), {}, None, None, []
    expected_paths = {raw_path(root, rank, count) for rank in range(count)}
    for path in (root / "raw/evaluation").glob(f"{BASELINE}_step0_baseline*.jsonl"):
        if path.resolve() not in expected_paths:
            raise ValueError("Unexpected or duplicate baseline raw file")
    for rank in range(count):
        path = raw_path(root, rank, count)
        expected = shard_slots(slots, rank, count)
        rows = list(read_jsonl(path)) if path.exists() else []
        if len(rows) > len(expected) or (require_complete and len(rows) != len(expected)):
            raise ValueError("Missing or excessive baseline shard coverage")
        for row, slot in zip(rows, expected, strict=False):
            if row.get("slot_id") in seen:
                raise ValueError("Duplicate baseline slot")
            validate_raw(row, slot, protocol_amendment_id=amendment_id)
            _validate_input(root, row, slot, inputs, diagnostics, image_hashes)
            observed = evaluation_model_identity(row["model_identity"])
            if identity is None:
                identity, adapter = observed, row["adapter_identity"]
            if observed != identity or row["adapter_identity"] != adapter:
                raise ValueError("Baseline shard model/adapter identities differ")
            for key in (
                "model_id",
                "step",
                "freeze_sha256",
                "adapter_hash",
                "trainable_state_hash",
            ):
                if key in common and row["model_identity"].get(key) != common[key]:
                    raise ValueError("Baseline model differs from verified common start")
            seen.add(row["slot_id"])
        summaries.append(
            {
                "rank": rank,
                "raw_path": str(path.relative_to(root)),
                "raw_sha256": sha_file(path) if path.exists() else None,
                "generated_slots": len(rows),
                "expected_slots": len(expected),
            }
        )
    if require_complete and seen != {slot["slot_id"] for slot in slots}:
        raise ValueError("Baseline union does not match the complete frozen slot set")
    return {"generated_slots": len(seen), "shards": summaries, "model_identity": identity}


def _attempt(root):
    attempt = os.environ.get("SR_F1_ATTEMPT_ID", "")
    job = os.environ.get("SLURM_JOB_ID", "")
    if (
        os.environ.get("SR_F1_TASK_ID") != "BASELINE"
        or not re.fullmatch(r"BASELINE_attempt[0-9]{4}", attempt)
        or not re.fullmatch(r"[0-9]+", job)
    ):
        raise PermissionError("Parallel baseline requires the registered Slurm parent identity")
    return attempt, job, _bounded(root, f"{DIRECTORY}/attempts/{attempt}")


@contextlib.contextmanager
def _stop_boundary(stop_path=None):
    flags = {"requested": False, "signal": None}

    class Boundary(dict):
        def __bool__(self):
            return True

        def __getitem__(self, key):
            if key == "requested":
                return flags["requested"] or (stop_path is not None and stop_path.exists())
            return flags[key]

    def request(number, _frame):
        flags.update(requested=True, signal=number)

    old = {number: signal.signal(number, request) for number in (signal.SIGTERM, signal.SIGUSR1)}
    try:
        yield Boundary()
    finally:
        for number, handler in old.items():
            signal.signal(number, handler)


def _request_stop(directory, reason):
    path = directory / "STOP_REQUEST.json"
    if not path.exists():
        atomic_json(path, {"reason": reason, "time_ns": time.time_ns()}, exclusive=True)


def _wait_children(children, directory, boundary, *, parent_exception=False):
    failure_since = time.monotonic() if parent_exception else None
    terminated = False
    while True:
        statuses = [child.poll() for child in children]
        if boundary["requested"] or any(code not in (None, 0) for code in statuses):
            _request_stop(directory, "PREEMPTION" if boundary["requested"] else "CHILD_FAILED")
        failed = any(code not in (None, 0) for code in statuses)
        if failed and failure_since is None:
            failure_since = time.monotonic()
        if all(code is not None for code in statuses):
            return statuses
        # This bound applies only after an actual exception/failed sibling, never
        # to healthy slow generation or normal checkpoint/preemption requests.
        if failure_since is not None and time.monotonic() - failure_since >= 300:
            if not terminated:
                atomic_json(
                    directory / "FAILED_CHILD_CLEANUP.json",
                    {
                        "reason": "PARENT_EXCEPTION" if parent_exception else "CHILD_FAILED",
                        "child_pids": [
                            child.pid
                            for child, code in zip(children, statuses, strict=True)
                            if code is None
                        ],
                        "cooperative_grace_seconds": 300,
                        "time_ns": time.time_ns(),
                    },
                    exclusive=True,
                )
                for child, code in zip(children, statuses, strict=True):
                    if code is None:
                        child.terminate()
                terminated = True
            if time.monotonic() - failure_since >= 360:
                for child, code in zip(children, statuses, strict=True):
                    if code is None:
                        child.kill()
                        child.wait()
        time.sleep(0.2)


def run_parallel_baseline(plan, root, gpu_count):
    """Worker-compatible parent; a failed child is never a successful checkpoint."""
    root = Path(root).resolve(strict=True)
    _count(gpu_count)
    attempt, job, directory = _attempt(root)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if (
        len(devices) != gpu_count
        or len(set(devices)) != gpu_count
        or any(not re.fullmatch(r"[A-Za-z0-9_-]+", device) for device in devices)
    ):
        raise PermissionError("Baseline parent CUDA visibility does not match allocation")
    _, slots = prepare_manifest(plan, root, gpu_count)
    directory.mkdir(parents=True, exist_ok=True)
    launch = {
        "attempt_id": attempt,
        "job_id": job,
        "gpu_count": gpu_count,
        "manifest_sha256": sha_file(root / DIRECTORY / "PLAN.json"),
        "devices": devices,
        "parent_pid": os.getpid(),
        "hostname": socket.gethostname(),
        "module_sha256": sha_file(__file__),
        "time_ns": time.time_ns(),
    }
    atomic_json(directory / "LAUNCH.json", launch, exclusive=True)
    children, logs = [], []
    with _stop_boundary() as boundary:
        try:
            for rank, device in enumerate(devices):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, SRF1_BASELINE_RANK=str(rank))
                source = str(Path(__file__).resolve().parents[1])
                env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
                log = (directory / f"rank{rank:02d}.log").open("x")
                logs.append(log)
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "sr_f1.baseline_parallel",
                            "--run-root",
                            str(root),
                            "--rank",
                            str(rank),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            statuses = _wait_children(children, directory, boundary)
        except BaseException:
            _request_stop(directory, "PARENT_EXCEPTION")
            _wait_children(children, directory, {"requested": True}, parent_exception=True)
            raise
        finally:
            for log in logs:
                log.close()
    if any(status != 0 for status in statuses):
        raise RuntimeError(f"Parallel baseline child failed: {statuses}; raw evidence retained")
    gpu_uuids = set()
    receipts, artifacts = [], [f"{DIRECTORY}/PLAN.json", f"{DIRECTORY}/RUNTIME_PLAN.json"]
    for rank in range(gpu_count):
        receipt_path, process_path = (
            directory / f"rank{rank:02d}_{suffix}.json" for suffix in ("RESULT", "PROCESS")
        )
        receipt, process = _read(receipt_path), _read(process_path)
        runtime_path = directory / f"rank{rank:02d}_RUNTIME.json"
        runtime = _read(runtime_path)
        hardware = runtime["hardware"]
        uuid = hardware.get("cuda_uuid")
        if (
            receipt.get("runtime_sha256") != sha_file(runtime_path)
            or runtime.get("process_sha256") != sha_file(process_path)
            or hardware.get("cuda_visible_device_count") != 1
            or hardware.get("cuda_visible_devices") != devices[rank]
            or not uuid
            or uuid in gpu_uuids
        ):
            raise ValueError("Baseline child actual GPU identity mismatch")
        gpu_uuids.add(uuid)
        if (
            receipt["attempt_id"] != attempt
            or receipt["rank"] != rank
            or receipt["job_id"] != job
            or receipt["status"] not in {"COMPLETE", "CHECKPOINTED"}
            or receipt["process_sha256"] != sha_file(process_path)
            or process["launch_sha256"] != sha_file(directory / "LAUNCH.json")
            or process["rank"] != rank
            or process["cuda_visible_devices"] != devices[rank]
            or process["attempt_id"] != attempt
            or process["job_id"] != job
            or process["pid"] != children[rank].pid
            or process["parent_pid"] != launch["parent_pid"]
            or process["hostname"] != launch["hostname"]
            or receipt["result"]["status"] != receipt["status"]
        ):
            raise ValueError("Baseline child receipt/process identity mismatch")
        child_result = receipt["result"]
        for relative in child_result["artifacts"]:
            if not _bounded(root, relative).is_file():
                raise ValueError("Baseline child artifact missing")
        if child_result["metadata"].get("generated_slots") != receipt["generated_slots"]:
            raise ValueError("Baseline child result count mismatch")
        if receipt["status"] == "CHECKPOINTED":
            metadata = child_result["metadata"]
            if (
                metadata.get("full_state") is not True
                or metadata.get("identity_verified") is not True
                or metadata.get("raw_prefix_sha256") != receipt["raw_sha256"]
                or metadata.get("resume_raw_path")
                != str(raw_path(root, rank, gpu_count).relative_to(root))
                or not any(
                    sha_file(_bounded(root, item)) == receipt["raw_sha256"]
                    for item in child_result["artifacts"]
                )
            ):
                raise ValueError("Baseline child checkpoint is not its full immutable raw state")
        elif child_result["metadata"].get("raw_sha256") != receipt["raw_sha256"]:
            raise ValueError("Baseline child complete hash mismatch")
        receipts.append(receipt)
        artifacts.extend(
            str(path.relative_to(root)) for path in (receipt_path, process_path, runtime_path)
        )
        artifacts.extend(receipt["result"]["artifacts"])
    complete = all(receipt["status"] == "COMPLETE" for receipt in receipts)
    audit = validate_baseline_outputs(root, slots, gpu_count, require_complete=complete)
    for receipt, shard in zip(receipts, audit["shards"], strict=True):
        if receipt["raw_sha256"] != shard["raw_sha256"]:
            raise ValueError("Baseline raw shard changed after child receipt")
        if receipt["generated_slots"] != shard["generated_slots"]:
            raise ValueError("Baseline child count differs from actual raw records")
    audit.update(
        status="COMPLETE" if complete else "CHECKPOINTED",
        attempt_id=attempt,
        manifest_sha256=launch["manifest_sha256"],
        scores_released=False,
    )
    audit_path = directory / "COVERAGE.json"
    atomic_json(audit_path, audit, exclusive=True)
    artifacts.extend(
        str(path.relative_to(root)) for path in (directory / "LAUNCH.json", audit_path)
    )
    result = {
        "status": audit["status"],
        "artifacts": sorted(set(artifacts)),
        "metadata": {
            "generated_slots": audit["generated_slots"],
            "scores_released": False,
            "gpu_count": gpu_count,
            "full_state": True,
            "identity_verified": True,
            "baseline_manifest_sha256": launch["manifest_sha256"],
            "coverage_sha256": sha_file(audit_path),
        },
    }
    if not complete:
        result["reason"] = "STOP_REQUESTED" if (root / "STOP").exists() else "PREEMPTION"
    return result


def run_child(root, rank):
    root = Path(root).resolve(strict=True)
    attempt, job, directory = _attempt(root)
    launch = _read(directory / "LAUNCH.json")
    manifest = _read(root / DIRECTORY / "PLAN.json")
    plan = _read(root / DIRECTORY / "RUNTIME_PLAN.json")
    count = manifest["gpu_count"]
    manifest, slots = prepare_manifest(plan, root, count)
    shard = shard_slots(slots, rank, count)
    if (
        launch["attempt_id"] != attempt
        or launch["job_id"] != job
        or launch["manifest_sha256"] != sha_file(root / DIRECTORY / "PLAN.json")
        or launch["module_sha256"] != sha_file(__file__)
        or os.environ.get("SRF1_BASELINE_RANK") != str(rank)
        or os.environ.get("CUDA_VISIBLE_DEVICES") != launch["devices"][rank]
    ):
        raise PermissionError("Baseline child allocation/source identity mismatch")
    process_path = directory / f"rank{rank:02d}_PROCESS.json"
    process = {
        "attempt_id": attempt,
        "job_id": job,
        "rank": rank,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "launch_sha256": sha_file(directory / "LAUNCH.json"),
        "time_ns": time.time_ns(),
    }
    atomic_json(process_path, process, exclusive=True)
    result_path = directory / f"rank{rank:02d}_RESULT.json"
    path = raw_path(root, rank, count)
    with (root / DIRECTORY / f"rank{rank:02d}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with _stop_boundary(directory / "STOP_REQUEST.json") as boundary:
                from .runtime import load_for_evaluation

                runtime = load_for_evaluation(plan, root, BASELINE, step=0)
                runtime_path = directory / f"rank{rank:02d}_RUNTIME.json"
                hardware = runtime.identity.get("hardware", {})
                if (
                    hardware.get("cuda_visible_device_count") != 1
                    or hardware.get("cuda_visible_devices") != launch["devices"][rank]
                    or not hardware.get("cuda_uuid")
                ):
                    raise PermissionError("Baseline child does not own one verified visible GPU")
                atomic_json(
                    runtime_path,
                    {
                        "process_sha256": sha_file(process_path),
                        "hardware": hardware,
                    },
                    exclusive=True,
                )
                inputs, diagnostics = _input_maps(root)
                result = evaluate_slots(runtime, shard, inputs, diagnostics, root, path, boundary)
            record = {
                "attempt_id": attempt,
                "job_id": job,
                "rank": rank,
                "status": result["status"],
                "result": result,
                "generated_slots": sum(1 for _ in read_jsonl(path)),
                "raw_sha256": sha_file(path),
                "process_sha256": sha_file(process_path),
                "runtime_sha256": sha_file(runtime_path),
            }
            atomic_json(result_path, record, exclusive=True)
            return result
        except BaseException as exc:
            if not result_path.exists():
                atomic_json(
                    result_path,
                    {
                        "attempt_id": attempt,
                        "job_id": job,
                        "rank": rank,
                        "status": "FAILED",
                        "error": repr(exc),
                        "process_sha256": sha_file(process_path),
                    },
                    exclusive=True,
                )
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int)
    args = parser.parse_args()
    run_child(args.run_root, args.rank)


if __name__ == "__main__":
    main()
