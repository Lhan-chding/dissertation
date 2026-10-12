"""Registered historical, diagnostic and sealed-endpoint GPU work units."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from sr_f1.contract import file_hash
from sr_f1.data import load_inputs, load_tasks
from sr_f12.evaluation import durable_json, group_slots, shard_lease
from sr_f12.protocol import BASELINE, iter_model_slots, object_hash
from sr_f12.protocol import scientific_matrix as old_matrix

from .protocol import PARENT_ROOT, scientific_matrix, technical_matrix, verify_registration


def read_json(path):
    return json.loads(Path(path).read_text())


def load_policy(runtime, directory, step, model_id, *, expected_files=None):
    """Authenticate serialized checkpoint bytes and all state hashes before using policy."""
    import torch

    from mm_core.training import state_hash
    from sr_f1.runtime import copy_parameters

    checkpoint = Path(directory).resolve(strict=True)
    commit = read_json(checkpoint / f"commit-{step:02d}.json")
    relative = Path(commit["path"])
    path = (checkpoint / relative).resolve(strict=True)
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(checkpoint):
        raise PermissionError("Checkpoint payload escapes registered directory")
    if commit["step"] != step or file_hash(path) != commit["sha256"]:
        raise PermissionError("Checkpoint bytes or step differ")
    if expected_files is not None:
        commit_path = (checkpoint / f"commit-{step:02d}.json").resolve(strict=True)
        if expected_files.get(str(path)) != commit["sha256"] or expected_files.get(
            str(commit_path)
        ) != file_hash(commit_path):
            raise PermissionError("Historical checkpoint differs from frozen prior audit")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        state_hash(state) != commit["state_hash"]
        or {k: state_hash(v) for k, v in state.items()} != commit["field_hashes"]
        or state["committed_logical_step"] != step
    ):
        raise PermissionError("Checkpoint full state identity differs")
    source_id = state["run_identity"]["model_id"]
    if source_id != model_id and not (model_id == BASELINE and step == 0):
        raise PermissionError("Checkpoint belongs to a different model")
    copy_parameters(runtime.model, state["parameters"])
    runtime.identity.update(
        model_id=model_id, step=step, adapter_hash=state_hash(state["parameters"])
    )
    runtime.adapter_path = str(path)
    policy_hash = state_hash(state["parameters"])
    del state
    return dict(path=str(path), checkpoint_sha256=commit["sha256"], policy_hash=policy_hash)


def prior_file_hashes(registration):
    manifest = read_json(registration["prior_sources"])
    return {entry["canonical_path"]: entry["sha256"] for entry in manifest["files"]}


def task_complete(root, kind, model_id, artifacts, **extra):
    root = Path(root).resolve(strict=True)
    registration = verify_registration(root)
    refs = []
    for artifact in artifacts:
        path = Path(artifact).resolve(strict=True)
        if not path.is_relative_to(root):
            raise PermissionError("Task artifact must be in this experiment")
        refs.append(dict(path=str(path.relative_to(root)), sha256=file_hash(path)))
    receipt = dict(
        status="PASS" if kind == "pilot" else "COMPLETE",
        kind=kind,
        model_id=model_id,
        registration_sha256=object_hash(registration),
        artifacts=refs,
        **extra,
    )
    path = root / "tasks" / kind / model_id / "COMPLETE.json"
    durable_json(path, receipt)
    return receipt


def historical(root, model_id):
    from .evaluation import evaluate_monitor
    from .runner import load_real_runtime
    from .training import apply_monitor, initial_adaptive_state

    root = Path(root).resolve(strict=True)
    registration = verify_registration(root)
    registered = {row["model_id"] for row in old_matrix()}
    if model_id not in registered:
        raise PermissionError("Unknown historical model")
    plan = read_json(root / "config/SR_F2.json")
    runtime, _, _, _ = load_real_runtime(plan, root, task_id="history-" + model_id)
    tasks, inputs = load_tasks(root), load_inputs(root)
    prior = Path(PARENT_ROOT) / "runs" / model_id / "checkpoints"
    directory = root / "history" / model_id
    state, summaries, policies = initial_adaptive_state(), [], []
    for step in range(0, 97, 8):
        policies.append(
            load_policy(
                runtime, prior, step, model_id, expected_files=prior_file_hashes(registration)
            )
        )
        summary = evaluate_monitor(
            runtime,
            tasks,
            inputs,
            root,
            directory / f"step-{step:02d}",
            model_id=model_id,
            step=step,
            source_sha256=registration["source_manifest_sha256"],
        )
        state = apply_monitor(state, summary)
        summaries.append(summary)
    result = dict(
        status="COMPLETE",
        model_id=model_id,
        steps=list(range(0, 97, 8)),
        records=sum(s["count"] for s in summaries),
        summaries=summaries,
        trigger_state=state,
        policies=policies,
        thresholds_unchanged=True,
    )
    durable_json(directory / "COMPLETE.json", result)
    return task_complete(
        root, "history", model_id, [directory / "COMPLETE.json"], records=result["records"]
    )


def diagnostic(root, model_id):
    from .diagnostics import interval_slots, protocol_slots
    from .evaluation import run_group
    from .runner import load_real_runtime

    root = Path(root).resolve(strict=True)
    registration = verify_registration(root)
    if model_id not in {BASELINE} | {r["model_id"] for r in old_matrix()}:
        raise PermissionError("Unknown diagnostic model")
    plan = read_json(root / "config/SR_F2.json")
    runtime, _, _, _ = load_real_runtime(plan, root, task_id="diagnostic-" + model_id)
    prior_id = "SRF1_2_A_s71001" if model_id == BASELINE else model_id
    policy = load_policy(
        runtime,
        Path(PARENT_ROOT) / "runs" / prior_id / "checkpoints",
        0 if model_id == BASELINE else 96,
        model_id,
        expected_files=prior_file_hashes(registration),
    )
    tasks, inputs = load_tasks(root), load_inputs(root)
    manifest = read_json(root / "DIAGNOSTIC_MANIFEST.json")
    directory = root / "diagnostics" / model_id
    slots = protocol_slots(tasks, model_id) + interval_slots(tasks, model_id)
    groups = group_slots(slots)
    with shard_lease(directory):
        durable_json(
            directory / "PLAN.json",
            dict(
                slots_sha256=object_hash(slots),
                expected_records=2688,
                policy=policy,
                source_sha256=registration["source_manifest_sha256"],
            ),
        )
        refs = []
        for group in groups:
            result = run_group(
                runtime,
                group,
                inputs,
                manifest,
                tasks,
                root,
                directory,
                source_sha256=registration["source_manifest_sha256"],
                score=True,
                test_released=True,
            )
            refs.append({k: result[k] for k in ("group_id", "raw_sha256", "records_sha256")})
        if len(slots) != 2688:
            raise PermissionError("Frozen diagnostic count differs")
        durable_json(
            directory / "COMPLETE.json",
            dict(
                status="COMPLETE",
                model_id=model_id,
                records=len(slots),
                groups=refs,
                old_models_only=True,
                new_scientific_test_access=False,
            ),
        )
    return task_complete(root, "diagnostic", model_id, [directory / "COMPLETE.json"], records=2688)


def endpoint_slots(tasks, model_id):
    """Keep F1.2 evaluation seeds and prompts, replace only model slot identity."""
    allowed = {row["model_id"] for row in scientific_matrix()}
    if model_id not in allowed:
        raise PermissionError("Unregistered new endpoint")
    old_id = "SRF1_2_J_s71001"
    slots = []
    for original in iter_model_slots(tasks, [], old_id, stage="final", test_released=True):
        if original["pool"] not in ("TEST_ID", "TEST_COMPOSITION"):
            continue
        slot = dict(
            original, model_id=model_id, slot_id=original["slot_id"].replace(old_id, model_id, 1)
        )
        slots.append(slot)
    if len(slots) != 12800:
        raise PermissionError("Endpoint synthetic coverage differs")
    return slots


def endpoint(root, model_id):
    from sr_f12.endpoint import chartqa_panel

    from .evaluation import run_group
    from .runner import load_real_runtime

    root = Path(root).resolve(strict=True)
    registration = verify_registration(root, require_technical=True)
    from .orchestration import verify_task

    verify_task(root, dict(kind="train", model_id=model_id), object_hash(registration))
    run = root / "runs" / model_id
    done = read_json(run / "COMPLETE.json")
    if done.get("status") != "COMPLETE" or done.get("step") != 96:
        raise PermissionError("Endpoint requires its completed step96 policy")
    runtime, _, _, _ = load_real_runtime(
        read_json(root / "config/SR_F2.json"), root, task_id="endpoint-" + model_id
    )
    policy = load_policy(runtime, run / "checkpoints", 96, model_id)
    tasks, inputs = load_tasks(root), load_inputs(root)
    slots = endpoint_slots(tasks, model_id)
    chart_slots, chart_inputs, chart_tasks = chartqa_panel(root, model_id)
    inputs.update(chart_inputs)
    tasks.update(chart_tasks)
    slots.extend(chart_slots)
    if len(slots) != 15300:
        raise PermissionError("Full endpoint sample count differs")
    directory = root / "sealed" / "evaluation" / model_id
    with shard_lease(directory):
        durable_json(
            directory / "PLAN.json",
            dict(
                slots_sha256=object_hash(slots),
                expected_records=15300,
                policy=policy,
                scores_sealed=True,
                source_sha256=registration["source_manifest_sha256"],
            ),
        )
        refs = []
        for group in group_slots(slots):
            result = run_group(
                runtime,
                group,
                inputs,
                {},
                tasks,
                root,
                directory,
                source_sha256=registration["source_manifest_sha256"],
                score=False,
            )
            refs.append({k: result[k] for k in ("group_id", "raw_sha256", "records_sha256")})
        durable_json(
            directory / "COMPLETE.json",
            dict(
                status="COMPLETE", model_id=model_id, records=15300, groups=refs, scores_sealed=True
            ),
        )
    return task_complete(root, "endpoint", model_id, [directory / "COMPLETE.json"], records=15300)


def pilot(root, model_id):
    """One GPU handles one arm, with an orderly fresh-process restore at step four."""
    root = Path(root).resolve(strict=True)
    verify_registration(root)
    if model_id not in {r["model_id"] for r in technical_matrix()}:
        raise PermissionError("Unregistered pilot")
    worker = Path(__file__).resolve().parents[2] / "scripts/sr_f2/run_worker.py"
    logs = root / "technical" / model_id / "process_logs"
    logs.mkdir(parents=True, exist_ok=True)
    for stop in (4, 8):
        command = [
            sys.executable,
            str(worker),
            "--root",
            str(root),
            "--kind",
            "pilot-segment",
            "--model-id",
            model_id,
            "--stop-step",
            str(stop),
        ]
        log = logs / f"segment-{stop}-{time.time_ns()}.log"
        with log.open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        durable_json(
            log.with_suffix(".json"),
            dict(command=command, exit_code=result.returncode, log_sha256=file_hash(log)),
        )
        if result.returncode:
            raise RuntimeError("Technical segment failed; retained in " + str(log))
    path = root / "technical" / model_id
    return task_complete(
        root,
        "pilot",
        model_id,
        [path / "COMPLETE.json", path / "checkpoints/commit-08.json"],
        step=8,
    )


def train(root, model_id):
    from .runner import train_scientific

    root = Path(root).resolve(strict=True)
    result = train_scientific(read_json(root / "config/SR_F2.json"), root, model_id)
    directory = root / "runs" / model_id
    return task_complete(
        root,
        "train",
        model_id,
        [
            directory / "COMPLETE.json",
            directory / "checkpoints/commit-96.json",
            directory / "monitor_state/step-96.json",
        ],
        step=result["step"],
    )


def run(kind, root, model_id, *, stop_step=None):
    root = Path(root).resolve(strict=True)
    verify_registration(root)
    if kind == "pilot-segment":
        from .runner import execute_path

        candidates = {r["model_id"]: r for r in technical_matrix()}
        if stop_step not in (4, 8) or model_id not in candidates:
            raise PermissionError("Unregistered technical segment")
        return execute_path(
            read_json(root / "config/SR_F2.json"),
            root,
            candidates[model_id],
            technical=True,
            stop_step=stop_step,
        )
    operations = dict(
        history=historical, diagnostic=diagnostic, endpoint=endpoint, pilot=pilot, train=train
    )
    if kind not in operations:
        raise ValueError("Unknown worker kind")
    try:
        return operations[kind](root, model_id)
    except BaseException as error:
        durable_json(
            root / "technical_incidents" / f"{kind}-{model_id}-{time.time_ns()}.json",
            dict(
                kind=kind,
                model_id=model_id,
                status="TECHNICAL_FAILED",
                error=repr(error),
                job_id=os.environ.get("SLURM_JOB_ID"),
            ),
        )
        raise
