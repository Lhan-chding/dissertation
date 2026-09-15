#!/usr/bin/env python3
"""Run the authorized Q4 workers inside one two-GPU Slurm allocation.

No scheduler calls or authorization creation occur here. Allocation tokens
only isolate workers; the existing bridge finalizer verifies actual GPU UUIDs.
Imports and plan construction never load torch or a model.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


def gpu_tokens(raw):
    tokens = raw.split(",") if isinstance(raw, str) else []
    tokens = [token.strip() for token in tokens]
    if len(tokens) != 2 or any(
        re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", token) is None for token in tokens
    ):
        raise ValueError("exactly two whole GPU allocation tokens required; empty/MIG forbidden")
    canonical = [str(int(token)) if token.isdecimal() else token.lower() for token in tokens]
    if canonical[0] == canonical[1]:
        raise ValueError("GPU allocation tokens must be distinct")
    return tokens


def _uuid(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value) is None
    ):
        raise ValueError("a complete physical GPU UUID is required")
    return value.lower()


def query_allocated_gpus(tokens, *, run=None):
    """Query original Slurm tokens, without importing CUDA or a model runtime."""
    run = subprocess.run if run is None else run
    values = []
    for token in tokens:
        result = run(
            [
                "nvidia-smi",
                "--id",
                token,
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        rows = list(csv.reader(result.stdout.strip().splitlines()))
        if len(rows) != 1 or len(rows[0]) != 3:
            raise ValueError("GPU UUID query must return exactly one index/UUID/name row")
        index, gpu, name = [value.strip() for value in rows[0]]
        _uuid(gpu)
        if (
            re.fullmatch(r"[0-9]+", index) is None
            or (token.isdecimal() and int(index) != int(token))
            or (token.startswith("GPU-") and not gpu.lower().startswith(token.lower()))
            or "PRO6000" not in name.upper().replace(" ", "")
        ):
            raise ValueError("GPU query does not identify the requested whole PRO6000 token")
        values.append({"token": token, "index": index, "uuid": gpu, "name": name})
    return values


def _completed_worker(root, authorization, worker_index):
    """Verify completed original files locally without importing r3/model code."""
    from src.modeling_v3.io import sha256_file

    marker_path = root / "completed.json"
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text())
    manifest_path = root / "manifest.json"
    if marker.get("status") != "COMPLETED" or marker.get("manifest_sha256") != sha256_file(
        manifest_path
    ):
        raise ValueError("completed Q4 worker manifest hash/status changed")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest.get("files"), list):
        raise ValueError("completed Q4 worker manifest file list required")
    registered = {}
    for row in manifest["files"]:
        if not isinstance(row, dict) or not {"path", "sha256", "bytes"} <= set(row):
            raise ValueError("malformed completed worker manifest entry")
        relative = Path(row["path"])
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or not path.is_relative_to(root.resolve())
            or str(relative) in registered
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError("completed worker original manifest path/hash mismatch")
        registered[str(relative)] = row["sha256"]
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
        and p.name != ".writer.lock"
        and not p.name.endswith(".tmp")
        and p not in (marker_path, manifest_path)
    }
    if actual != set(registered) or "result.json" not in registered:
        raise ValueError("completed worker manifest inventory mismatch")

    def local_binding(binding):
        if not isinstance(binding, dict) or not {"path", "sha256"} <= set(binding):
            raise ValueError("completed worker requires a bound null original")
        path = Path(binding["path"]).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or registered.get(str(path.relative_to(root.resolve()))) != binding["sha256"]
        ):
            raise ValueError("completed worker bound null original hash/path mismatch")
        return json.loads(path.read_text())

    result = json.loads((root / "result.json").read_text())
    if (
        result.get("status") != "Q4_WORKER_COMPLETED_PENDING_TWO_GPU_NULL"
        or result.get("execution_kind") != "REAL_CUDA_MODEL"
        or result.get("worker_index") != worker_index
        or result.get("source_hash") != authorization["source_hash"]
        or result.get("config_hash") != authorization["config_hash"]
        or result.get("Q4_plan_hash") != authorization["q4_plan_hash"]
    ):
        raise ValueError("completed Q4 worker result identity/status mismatch")
    receipt = local_binding(result.get("null_receipt"))
    local_binding(result.get("null_anchor"))
    comparison = local_binding(result.get("null_comparison"))
    invocation = local_binding(result.get("null_invocation"))
    hardware = local_binding(invocation.get("hardware"))
    if (
        comparison.get("status") != "PASS"
        or comparison.get("current") != result["null_receipt"]
        or comparison.get("anchor") != result["null_anchor"]
        or invocation.get("null_receipt") != result["null_receipt"]
        or _uuid(receipt.get("gpu_identity"))
        != _uuid(hardware.get("allocated_gpu", {}).get("uuid"))
    ):
        raise ValueError("completed null invocation UUID or original binding mismatch")
    return {
        "worker_index": worker_index,
        "null_receipt_uuid": receipt["gpu_identity"],
        "result": {"path": str(root / "result.json"), "sha256": registered["result.json"]},
        "completed": {"path": str(marker_path), "sha256": sha256_file(marker_path)},
        "null_receipt": result["null_receipt"],
    }


def _assign_allocations(tokens, allocation, completed):
    if not isinstance(allocation, list) or len(allocation) != 2:
        raise ValueError("exactly two allocated GPU UUID query records required")
    if any(row.get("token") != token for row, token in zip(allocation, tokens, strict=True)):
        raise ValueError("GPU UUID query changed original allocation token order")
    if _uuid(allocation[0].get("uuid")) == _uuid(allocation[1].get("uuid")):
        raise ValueError("different Slurm tokens resolved to the same GPU UUID")
    for assigned in (allocation, allocation[::-1]):
        effective = [
            completed[i]["null_receipt_uuid"] if completed[i] else assigned[i]["uuid"]
            for i in range(2)
        ]
        if _uuid(effective[0]) != _uuid(effective[1]):
            return assigned, effective
    raise ValueError("completed Q4 worker UUIDs cannot form a distinct resumed null pair")


def _authorization_files(config_path, bindings_path):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.modeling_v3.io import canonical_hash, sha256_file, source_identity
    from src.modeling_v3.schema import load_config
    from src.modeling_v3.vlm_campaign import _bound_json, _technical_gate

    config_path, bindings_path = Path(config_path).resolve(), Path(bindings_path).resolve()
    config = load_config(config_path)
    bindings = json.loads(bindings_path.read_text())
    stage = _bound_json(bindings["v3_stage_lock"])
    confirmation = stage.get("first_gpu_confirmation", {})
    if (
        stage.get("operator_authorization") != "CONFIRMED_FIRST_GPU_SUBMISSION"
        or confirmation.get("scope") != "Q4"
        or not isinstance(confirmation.get("confirmation_text"), str)
        or not confirmation["confirmation_text"].strip()
    ):
        raise PermissionError("recorded first Q4 GPU authorization is required before launch")
    current = {"config_hash": canonical_hash(config), "source_hash": source_identity()["sha256"]}
    if any(stage.get(key) != value for key, value in current.items()):
        raise ValueError("authorized stage configuration/source changed")
    if (
        stage.get("phase") != "Q4"
        or "vlm-smoke" not in stage.get("operations", [])
        or stage.get("workers") != [0, 1]
        or stage.get("max_concurrent_project_gpus") != 2
    ):
        raise PermissionError("authorized Q4 stage must include exactly workers 0 and 1")
    handoff = confirmation["reviewed_handoff"]
    if sha256_file(handoff["path"]) != handoff["sha256"]:
        raise ValueError("first GPU authorization command/timing handoff changed")
    prepared = _bound_json(confirmation["prepared_stage"])
    expected = {
        **prepared,
        "operator_authorization": "CONFIRMED_FIRST_GPU_SUBMISSION",
        "first_gpu_confirmation": confirmation,
    }
    if (
        prepared.get("operator_authorization") != "PENDING_FIRST_GPU_CONFIRMATION"
        or stage != expected
    ):
        raise ValueError("Q4 authorization differs from its immutable prepared stage")
    _technical_gate(config, stage)
    bridge = _bound_json(bindings["q4_bridge_plan"])
    if (
        any(bridge.get(key) != value for key, value in current.items())
        or bridge.get("plan_hash")
        != canonical_hash({k: v for k, v in bridge.items() if k != "plan_hash"})
        or stage.get("q4_plan_hash") != bridge["plan_hash"]
    ):
        raise ValueError("Q4 bridge plan differs from its frozen configuration/source/stage")
    return {
        **current,
        "launcher_file": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        "config_file": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "bindings_file": {"path": str(bindings_path), "sha256": sha256_file(bindings_path)},
        "stage_binding": bindings["v3_stage_lock"],
        "bridge_binding": bindings["q4_bridge_plan"],
        "q4_plan_hash": bridge["plan_hash"],
    }


def build_plan(config, bindings, out, visible_devices, *, resume=False, gpu_query=None):
    tokens = gpu_tokens(visible_devices)
    authorization = _authorization_files(config, bindings)
    out = Path(out).resolve()
    if out.exists() and not resume:
        raise FileExistsError(
            "existing Q4 output requires explicit --resume; originals are retained"
        )
    allocation = (query_allocated_gpus if gpu_query is None else gpu_query)(tokens)
    completed = [_completed_worker(out / f"worker_{i}", authorization, i) for i in range(2)]
    assigned, effective = _assign_allocations(tokens, allocation, completed)
    workers = []
    for index, allocated in enumerate(assigned):
        token = allocated["token"]
        worker_out = out / f"worker_{index}"
        command = [
            sys.executable,
            "-m",
            "src.modeling_v3.cli",
            "vlm-smoke",
            "--config",
            authorization["config_file"]["path"],
            "--bindings",
            authorization["bindings_file"]["path"],
            "--device",
            "cuda:0",
            "--worker-index",
            str(index),
            "--out",
            str(worker_out),
        ]
        if resume:
            command.append("--resume")
        command.extend(["--allow-gpu", "--acknowledge-new-experiment"])
        workers.append(
            {
                "worker_index": index,
                "cuda_visible_devices": token,
                "allocated_gpu_uuid": allocated["uuid"],
                "out": str(worker_out),
                "command": command,
            }
        )
    identity = {**authorization, "out": str(out), "worker_outputs": [w["out"] for w in workers]}
    identity_path = out / "PAIR_IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Q4 pair resume authorization/configuration/source identity changed")
    return {
        "schema": "ssvc-v3-q4-pair-plan-1",
        "identity": identity,
        "out": str(out),
        "allocation_tokens": tokens,
        "allocated_gpus": allocation,
        "completed_workers": completed,
        "expected_effective_null_uuids": effective,
        "allocation_uuid_distinctness_verified": True,
        "resume_tokens_remapped": [row["token"] for row in assigned] != tokens,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "workers": workers,
        "resume": bool(resume),
        "distinct_gpu_uuids_verified": False,
        "actual_gpu_validation": "EXISTING_RUNTIME_PRO6000_CHECK_AND_Q4_BRIDGE_UUID_PARITY",
        "authorization_created": False,
        "scheduler_submission": False,
    }


def execute_pair(plan):
    """Launch and reap only these two child sessions; preserve every attempt."""
    from src.core import frozen_writer
    from src.modeling_v3.io import atomic_json

    if Path(plan["out"]).exists() and not plan["resume"]:
        raise FileExistsError("existing Q4 output requires explicit --resume")
    children, worker_results, previous_handlers = [], [], {}
    received = {"signal": None, "time": None}
    error = None

    def on_signal(signum, _frame):
        if received["signal"] is None:
            received.update(signal=signum, time=time.monotonic())

    def signal_children(signum):
        for child in children:
            if child.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signum)

    with frozen_writer(plan["out"]), contextlib.ExitStack() as files:
        root = Path(plan["out"])
        identity_path = root / "PAIR_IDENTITY.json"
        if identity_path.exists():
            if not plan["resume"]:
                raise FileExistsError("existing Q4 pair identity requires explicit --resume")
            if json.loads(identity_path.read_text()) != plan["identity"]:
                raise ValueError("Q4 pair identity changed before launch")
        else:
            atomic_json(identity_path, plan["identity"])
        attempt = root / "pair_attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        atomic_json(attempt / "PLAN.json", plan)
        print(json.dumps({"attempt": str(attempt), "plan": plan}), flush=True)
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, on_signal)
            for worker in plan["workers"]:
                if received["signal"] is not None:
                    break
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = worker["cuda_visible_devices"]
                environment["SSVC_V3_EXPECTED_GPU_UUID"] = worker["allocated_gpu_uuid"]
                environment["HIP_VISIBLE_DEVICES"] = ""
                index = worker["worker_index"]
                stdout = files.enter_context((attempt / f"worker_{index}.stdout.log").open("xb"))
                stderr = files.enter_context((attempt / f"worker_{index}.stderr.log").open("xb"))
                child = subprocess.Popen(
                    worker["command"],
                    env=environment,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                children.append(child)
            stopped = False
            while any(child.poll() is None for child in children):
                if received["signal"] is not None:
                    if not stopped:
                        signal_children(signal.SIGTERM)
                        stopped = True
                    if time.monotonic() - received["time"] >= 20:
                        signal_children(signal.SIGKILL)
                time.sleep(0.1)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            signal_children(signal.SIGTERM)
        finally:
            for child in children:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        for index, worker in enumerate(plan["workers"]):
            worker_results.append(
                {
                    **worker,
                    "pid": children[index].pid if index < len(children) else None,
                    "returncode": children[index].returncode if index < len(children) else None,
                    "status": "EXITED" if index < len(children) else "NOT_STARTED",
                }
            )
        exit_code = (
            128 + received["signal"]
            if received["signal"] is not None
            else 1
            if error or len(children) != 2 or any(c.returncode != 0 for c in children)
            else 0
        )
        result = {
            "schema": "ssvc-v3-q4-pair-result-1",
            "exit_code": exit_code,
            "workers": worker_results,
            "allocation_tokens": plan["allocation_tokens"],
            "slurm_job_id": plan["slurm_job_id"],
            "signal": received["signal"],
            "launcher_error": error,
            "distinct_gpu_uuids_verified": False,
            "bridge_finalization_required": True,
            "attempt": str(attempt),
        }
        atomic_json(attempt / "RESULT.json", result)
        print(json.dumps(result), flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (
        sys.platform != "linux"
        or re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")) is None
    ):
        raise ValueError("Q4 pair execution requires Linux and a numeric Slurm job ID")
    plan = build_plan(
        args.config,
        args.bindings,
        args.out,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        resume=args.resume,
    )
    return execute_pair(plan)["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
