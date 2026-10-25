"""Gated, resumable 15-path SR-F5 development pilot, never formal training.

Primary paths generate 23,040 answers. Recovery lanes regenerate steps 5--12
from the same committed step 4 in a fresh process and are diagnostic calls.
Checkpoint directories become visible atomically with a COMMITTED marker.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import socket
import time
import uuid
from collections import Counter
from pathlib import Path

from sr_f4.common import append, now, read, rows, write, write_rows
from sr_f4.training import load_checkpoint, save_checkpoint
from srf5_ref.credit import ARMS
from srf5_ref.gates import check_calibration, check_pilot
from srf5_ref.schedule import build_schedule

from .calibration import annotate_group, digest, independent_review
from .runtime import PROTOCOL_VERSION, load
from .training import build_credit_group, update

STEPS = 12
DIAGNOSTIC_ROWS = (0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120)


def registered_write(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise PermissionError(f"Registered identity differs: {path}")
    else:
        write(path, value)


def source_identity():
    source = Path(__file__).resolve().parents[1]
    names = (
        "sr_f5/pilot.py",
        "sr_f5/runtime.py",
        "sr_f5/training.py",
        "sr_f5/calibration.py",
        "sr_f5/measurement.py",
        "sr_f4/training.py",
        "sr_f4/runtime.py",
        "sr_f3_ce/runtime.py",
        "sr_f3_ce/training.py",
        "sr_f12/training.py",
        "srf5_ref/credit.py",
        "srf5_ref/objective.py",
        "srf5_ref/schedule.py",
    )
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}


def pilot_inputs(root, config):
    private = list(rows(root / "DATA/pilot/TASKS_PRIVATE.jsonl"))
    public = list(rows(root / "DATA/pilot/MODEL_INPUTS.jsonl"))
    tasks, inputs = ({r["qid"]: r for r in values} for values in (private, public))
    if len(private) != 192 or len(tasks) != 192 or len(public) != 192 or set(tasks) != set(inputs):
        raise ValueError("Pilot requires its independent 192-question library")
    for qid, item in inputs.items():
        if (
            item["protocol"] != "structured_v3"
            or item["root_id"] != tasks[qid]["root_id"]
            or set(item["model_input"]) != {"prompt", "image_file", "image_sha256"}
        ):
            raise PermissionError("Pilot public input boundary differs")
    schedules = {}
    for parent in config["parents"]:
        expected = build_schedule(private, config["master_seed"], parent, 0, 8)
        actual = read(root / f"SCHEDULES/pilot-{parent}.json")
        if actual != expected or len(actual) != STEPS:
            raise PermissionError("Pilot schedule differs from registered plan")
        schedules[parent] = actual
    return tasks, inputs, schedules


def require_g2(root):
    gate, metrics = read(root / "GATES/G2.json"), read(root / "G2/METRICS.json")
    if (
        gate.get("status") != "PASS"
        or gate.get("actual_response_count") != 4608
        or check_calibration(metrics)["status"] != "PASS"
        or metrics.get("missing_groups")
        or metrics.get("failures")
    ):
        raise PermissionError("G3 requires actual complete G2 PASS evidence")
    if read(root / "CPU_TESTS.json").get("status") != "PASS":
        raise PermissionError("Server CPU checks have not passed")
    config_digest = hashlib.sha256(
        json.dumps(read(root / "CONFIG.json"), sort_keys=True).encode()
    ).hexdigest()
    if read(root / "G2_SPEC_FROZEN.json").get("config_digest") != config_digest:
        raise PermissionError("Configuration differs from the completed census")
    return dict(gate=digest(gate), metrics=digest(metrics))


def freeze(root, revision):
    root = Path(root).resolve()
    if not revision or len(revision) < 7:
        raise ValueError("Committed source revision is required")
    config = read(root / "CONFIG.json")
    if config["arms"] != list(ARMS) or config["parents"] != ["b0", "b1", "b2"]:
        raise PermissionError("Pilot arm/parent matrix changed")
    if config["pilot"]["updates"] != STEPS or config["pilot"]["repeats"] != 1:
        raise PermissionError("Pilot budget changed")
    tasks, inputs, schedules = pilot_inputs(root, config)
    plan = dict(
        status="FROZEN_BEFORE_PILOT_OUTPUTS",
        code_revision=revision,
        source_identity=source_identity(),
        config_digest=digest(config),
        g2=require_g2(root),
        private_digest=digest(tasks),
        public_digest=digest(inputs),
        schedule_digests={parent: digest(s) for parent, s in schedules.items()},
        protocol_version=PROTOCOL_VERSION,
        updates=STEPS,
        group_size=8,
        responses_per_step=128,
        primary_responses=23040,
        recovery_responses=5120,
        endpoint_probe_responses=0,
        diagnostic_steps=[1, 12],
        diagnostic_rows=list(DIAGNOSTIC_ROWS),
        step12_review=dict(
            unique_per_path=24,
            total=360,
            selection="sorted_row_id_unique_qid_text",
            reviewer="program",
        ),
        recovery=dict(
            from_step=4,
            to_step=12,
            paths="b0_all_five_arms",
            parameter_atol=0.0,
            parameter_rtol=0.0,
            optimizer_rng_comparison="exact",
            raw_token_probability_credit_update_comparison="exact",
        ),
        paths=[dict(parent=p, arm=a) for p in config["parents"] for a in config["arms"]],
        formal_outputs_allowed=False,
    )
    path = root / "G3/FROZEN.json"
    if not path.exists() and any((root / "G3/RUNS").glob("*/*/*/rollouts/*")):
        raise PermissionError("Cannot freeze after pilot generation")
    registered_write(path, plan)
    return plan


def require_frozen(root):
    plan = read(root / "G3/FROZEN.json")
    if freeze(root, plan["code_revision"]) != plan:
        raise PermissionError("G3 frozen identity changed")
    return plan


def committed(directory, identity):
    result = {}
    for candidate in (directory / "checkpoints").glob("step-*"):
        marker = candidate / "COMMITTED.json"
        if not marker.exists():
            continue
        receipt = read(marker)
        step = receipt.get("step")
        if (
            type(step) is not int
            or not 0 <= step <= STEPS
            or receipt.get("identity") != identity
            or candidate.name != f"step-{step:03d}"
            or not (candidate / "state.pt").is_file()
        ):
            raise PermissionError("Checkpoint commit identity differs")
        result[step] = candidate
    return result


def checkpoint_header(path, identity, step):
    """Read serialized metadata via mmap, without scanning weight payloads."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (
        state.get("identity") != identity
        or state.get("step") != step
        or state.get("sampler_state") != dict(next_step=step + 1, committed_log_step=step)
    ):
        raise PermissionError("Serialized checkpoint identity/cursor differs from its marker")


def file_metadata(path):
    value = Path(path).stat()
    return dict(
        size=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino, device=value.st_dev
    )


def commit(directory, runtime, optimizer, scheduler, identity, step, receipt=None):
    checkpoint_root = directory / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    target = checkpoint_root / f"step-{step:03d}"
    if target.exists():
        raise PermissionError("Never replace an existing committed checkpoint")
    staging = checkpoint_root / f".staging-{step:03d}-{uuid.uuid4().hex}"
    staging.mkdir()
    cursor = dict(next_step=step + 1, committed_log_step=step)
    save_checkpoint(
        staging / "state.pt",
        runtime,
        optimizer,
        scheduler,
        identity=identity,
        step=step,
        sampler_state=cursor,
    )
    if receipt is not None:
        write(staging / "UPDATE.json", receipt)
    write(staging / "COMMITTED.json", dict(identity=identity, step=step, cursor=cursor))
    directory_fd = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.rename(staging, target)
    parent_fd = os.open(checkpoint_root, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    # A small immutable update receipt survives compacting superseded state.
    if receipt is not None:
        registered_write(directory / "updates" / f"{step:03d}.json", receipt)
    for previous_step, previous in committed(directory, identity).items():
        if previous_step not in (4, STEPS, step):
            old_receipt = previous / "UPDATE.json"
            if old_receipt.exists():
                registered_write(
                    directory / "updates" / f"{previous_step:03d}.json", read(old_receipt)
                )
            shutil.rmtree(previous)
    return target


def validate_rollout(records, assignment, identity, step, model_input):
    if len(records) != 8 or [r["seed"] for r in records] != assignment["draw_seeds"]:
        raise ValueError("Pilot group must retain all eight registered draws")
    for row in records:
        actual = row["adapter_identity"]
        if any(
            actual.get(k) != v
            for k, v in dict(
                parent=identity["parent"], run_id=identity["run_id"], step=step - 1
            ).items()
        ):
            raise PermissionError("Rollout is not from this arm's committed pre-update policy")
        if row["model_input"] != model_input or row.get("technical_validation_errors"):
            raise PermissionError("Rollout input changed or contains technical failures")
        if row.get("generation_status") != "COMPLETE":
            raise ValueError("Unresolved generation cannot enter training")


def rollout(root, directory, runtime, task, public, assignment, identity, step, lane):
    index = assignment["group_index"]
    basename = directory / "rollouts" / f"{step:03d}" / f"{index:02d}"
    attempt, raw_path, final = (
        basename.with_suffix(ext) for ext in (".attempt.json", ".raw.jsonl", ".json")
    )
    spec = dict(
        identity=identity,
        step=step,
        assignment=assignment,
        input_digest=digest(public),
        task_digest=digest(task),
        lane=lane,
    )
    if final.exists():
        saved = read(final)
        if saved["spec"] != spec:
            raise PermissionError("Pilot cached rollout identity changed")
        records = saved["records"]
    else:
        if attempt.exists():
            if read(attempt)["spec"] != spec:
                raise PermissionError("Interrupted pilot specification changed")
            raw = list(rows(raw_path)) if raw_path.exists() else []
            if len(raw) != 8:
                raise RuntimeError("Partial pilot generation: preserve records and diagnose")
        else:
            write(attempt, dict(at=now(), spec=spec, status="STARTED"))
            append(
                root / "G3/CALLS.jsonl",
                dict(
                    at=now(),
                    run_id=identity["run_id"],
                    lane=lane,
                    purpose="primary_pilot" if lane == "primary" else "recovery_diagnostic",
                    count=8,
                    step=step,
                    qid=task["qid"],
                    seeds=assignment["draw_seeds"],
                ),
            )
            runtime.generate_group(
                public["model_input"],
                root,
                assignment["draw_seeds"],
                protocol=public["protocol"],
                on_completion=lambda row: append(raw_path, row),
            )
            raw = list(rows(raw_path))
        validate_rollout(raw, assignment, identity, step, public["model_input"])
        records = annotate_group(
            raw,
            task,
            dict(parent=identity["parent"], seeds=assignment["draw_seeds"]),
            runtime.processor.tokenizer,
        )
        group_id = f"{identity['run_id']}/{step:03d}/{index:02d}"
        for i, record in enumerate(records):
            record.update(
                row_id=f"{group_id}/{i}",
                group_id=group_id,
                run_root=str(root),
                protocol=public["protocol"],
                step=step,
                lane=lane,
            )
        write(final, dict(spec=spec, records=records))
    validate_rollout(records, assignment, identity, step, public["model_input"])
    credit = build_credit_group(records, identity["arm"], assignment["shuffle_seed"])
    return records, credit


def run(root, parent, arm, lane="primary", stop_after=STEPS):
    from mm_core.training import trainable_state

    root = Path(root).resolve()
    if parent not in ("b0", "b1", "b2") or arm not in ARMS or lane not in ("primary", "recovery"):
        raise ValueError("Unregistered pilot path")
    if lane == "recovery" and parent != "b0":
        raise PermissionError("Only the five b0 paths have registered recovery resampling")
    if stop_after not in (4, STEPS):
        raise ValueError("Only registered 4-step interruption or 12-step endpoint allowed")
    frozen = require_frozen(root)
    config = read(root / "CONFIG.json")
    tasks, inputs, schedules = pilot_inputs(root, config)
    run_id = f"G3-{parent}-{arm}"
    identity = dict(
        run_id=run_id,
        parent=parent,
        arm=arm,
        repeat=0,
        frozen_digest=digest(frozen),
        reference=config["model"]["parents"][parent],
        protocol_version=PROTOCOL_VERSION,
    )
    directory = root / "G3/RUNS" / parent / arm / lane
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registered_write(directory / "IDENTITY.json", identity)
        checkpoints = committed(directory, identity)
        if STEPS in checkpoints:
            checkpoint_header(checkpoints[STEPS] / "state.pt", identity, STEPS)
            endpoint = read(checkpoints[STEPS] / "UPDATE.json")
            if endpoint.get("run_id") != run_id or endpoint.get("step") != STEPS:
                raise PermissionError("Final committed update belongs to another run")
            registered_write(directory / "updates" / f"{STEPS:03d}.json", endpoint)
            registered_write(
                directory / "COMPLETE.json",
                dict(
                    status="COMPLETE",
                    identity=identity,
                    lane=lane,
                    committed_step=STEPS,
                    responses=1536 if lane == "primary" else 1024,
                    scientific_gate="NOT_YET_AGGREGATED",
                    execution=endpoint["execution"],
                ),
            )
            return dict(run_id=run_id, lane=lane, committed_step=STEPS, model_calls=0)
        execution_id = uuid.uuid4().hex
        load_task_id = f"{run_id}-{lane}-{execution_id}"
        runtime, optimizer, scheduler = load(root, parent, load_task_id, training=True)
        runtime.identity.update(
            run_id=run_id,
            pilot_lane=lane,
            pilot_execution_id=execution_id,
        )
        reference = trainable_state(runtime.model)
        initial = max(checkpoints) if checkpoints else (4 if lane == "recovery" else 0)
        if checkpoints:
            source = checkpoints[initial]
        elif lane == "recovery":
            source = directory.parent / "primary/checkpoints/step-004"
            receipt = read(source / "COMMITTED.json")
            if receipt["identity"] != identity or receipt["step"] != 4:
                raise PermissionError("Recovery source must be the matching primary step 4")
        else:
            source = commit(directory, runtime, optimizer, scheduler, identity, 0)
        expected_initial = initial
        initial, cursor = load_checkpoint(
            source / "state.pt", runtime, optimizer, scheduler, identity=identity
        )
        if initial != expected_initial or cursor != dict(
            next_step=initial + 1, committed_log_step=initial
        ):
            raise PermissionError("Checkpoint cursor differs")
        runtime.identity["step"] = initial
        process = dict(host=socket.gethostname(), pid=os.getpid())
        proc_stat = Path("/proc/self/stat")
        process["start_ticks"] = (
            proc_stat.read_text().rsplit(")", 1)[1].split()[19] if proc_stat.exists() else None
        )
        execution = dict(id=execution_id, process=process, restored_step=initial)
        append(directory / "EXECUTIONS.jsonl", dict(at=now(), **execution))
        write(
            directory / "environment" / f"{execution['id']}.json",
            dict(
                runtime_identity=runtime.identity,
                actual_hardware_identity=read(root / "runtime_identity" / f"{load_task_id}.json"),
                native_accounting_file=f"accounting/{load_task_id}.jsonl",
                trainable_parameters=[
                    dict(name=name, shape=list(value.shape), count=value.numel())
                    for name, value in runtime.model.named_parameters()
                    if value.requires_grad
                ],
            ),
        )
        for checkpoint_step, checkpoint_path in checkpoints.items():
            if (checkpoint_path / "UPDATE.json").exists():
                registered_write(
                    directory / "updates" / f"{checkpoint_step:03d}.json",
                    read(checkpoint_path / "UPDATE.json"),
                )
        step = initial
        try:
            for entry in schedules[parent][initial:stop_after]:
                step = entry["step"]
                started = time.monotonic()
                records, credit_groups = [], []
                for assignment in entry["groups"]:
                    qid = assignment["qid"]
                    group, credit = rollout(
                        root,
                        directory,
                        runtime,
                        tasks[qid],
                        inputs[qid],
                        assignment,
                        identity,
                        step,
                        lane,
                    )
                    records.extend(group)
                    credit_groups.append(credit)
                registered_write(directory / "credit" / f"{step:03d}.json", credit_groups)
                result = update(
                    runtime,
                    optimizer,
                    scheduler,
                    records,
                    [8] * 16,
                    reference,
                    credit_groups,
                    diagnostic_row_indices=DIAGNOSTIC_ROWS if step in (1, 12) else (),
                )
                result.update(
                    step=step,
                    run_id=run_id,
                    lane=lane,
                    parent=parent,
                    arm=arm,
                    primary_answers=128 if lane == "primary" else 0,
                    wall_seconds=time.monotonic() - started,
                    qids=[r["qid"] for r in records],
                    generated_tokens=sum(len(r["tokens"]) for r in records),
                    roots=sorted({tasks[r["qid"]]["root_id"] for r in records}),
                    execution=execution,
                )
                commit(directory, runtime, optimizer, scheduler, identity, step, result)
                runtime.identity["step"] = step
                write(
                    directory / "PROGRESS.json",
                    dict(
                        committed_step=step,
                        target_step=STEPS,
                        lane=lane,
                        answers=step * 128 if lane == "primary" else max(0, step - 4) * 128,
                    ),
                )
                print(json.dumps(dict(run_id=run_id, lane=lane, committed_step=step)), flush=True)
            if step == STEPS:
                write(
                    directory / "COMPLETE.json",
                    dict(
                        status="COMPLETE",
                        identity=identity,
                        lane=lane,
                        committed_step=STEPS,
                        responses=1536 if lane == "primary" else 1024,
                        scientific_gate="NOT_YET_AGGREGATED",
                        execution=execution,
                    ),
                )
        except BaseException as exc:
            append(
                root / "G3/FAILURES.jsonl",
                dict(
                    at=now(),
                    run_id=run_id,
                    lane=lane,
                    attempted_step=step,
                    error=repr(exc),
                    action="preserve_failed_attempt_and_restore_committed_checkpoint",
                ),
            )
            raise
    return dict(run_id=run_id, lane=lane, committed_step=step)


def exact_tree(left, right):
    """One actual checkpoint comparison; tensors are never compared via summaries."""
    import torch

    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            exact_tree(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def compare_recovery(root, parent, arm):
    import torch

    root = Path(root)
    require_frozen(root)
    directory = root / "G3/RUNS" / parent / arm
    primary, recovery = (directory / name for name in ("primary", "recovery"))
    if any(not (p / "COMPLETE.json").exists() for p in (primary, recovery)):
        raise PermissionError("Both 12-step paths must complete")
    if parent != "b0" or arm not in ARMS:
        raise PermissionError("Unregistered recovery comparison")
    left, right = [
        torch.load(p / "checkpoints/step-012/state.pt", map_location="cpu", weights_only=True)
        for p in (primary, recovery)
    ]
    checks = {key: exact_tree(value, right.get(key)) for key, value in left.items()}
    checks["state_keys"] = left.keys() == right.keys()
    completions = [read(p / "COMPLETE.json") for p in (primary, recovery)]
    processes = [c["execution"]["process"] for c in completions]
    checks["fresh_process"] = processes[0] != processes[1]
    checks["primary_continuous_from_zero"] = completions[0]["execution"]["restored_step"] == 0
    checks["recovery_restored_step_four"] = completions[1]["execution"]["restored_step"] == 4
    checks["terminal_step_twelve"] = left.get("step") == right.get("step") == STEPS
    update_matches = 0
    omitted = {"training_seconds", "wall_seconds", "lane", "primary_answers", "execution"}
    for step in range(1, STEPS + 1):
        primary_update = read(primary / "updates" / f"{step:03d}.json")
        if primary_update["execution"] != completions[0]["execution"]:
            checks["primary_continuous_from_zero"] = False
        if step > 4:
            recovery_update = read(recovery / "updates" / f"{step:03d}.json")
            if recovery_update["execution"] != completions[1]["execution"]:
                checks["recovery_restored_step_four"] = False
            a = {k: v for k, v in primary_update.items() if k not in omitted}
            b = {k: v for k, v in recovery_update.items() if k not in omitted}
            update_matches += exact_tree(a, b)
    checks["numerical_credit_update_receipts"] = update_matches == 8
    token_matches = 0
    for step in range(5, STEPS + 1):
        for group in range(16):
            values = [
                read(p / "rollouts" / f"{step:03d}" / f"{group:02d}.json")["records"]
                for p in (primary, recovery)
            ]
            fields = ("tokens", "full_text", "sampler_logprobs", "score", "selected_mask")
            matched = all(all(a[k] == b[k] for k in fields) for a, b in zip(*values, strict=True))
            token_matches += matched
    checks["regenerated_groups"] = token_matches == 128
    result = dict(
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        parent=parent,
        arm=arm,
        regenerated_groups=128,
        actual_matching_groups=token_matches,
        criterion="frozen exact; no tolerance widened after observing outputs",
        identity=left["identity"],
        checkpoint_metadata={
            name: file_metadata(directory / name / "checkpoints/step-012/state.pt")
            for name in ("primary", "recovery")
        },
        completion_digests={
            name: digest(read(directory / name / "COMPLETE.json"))
            for name in ("primary", "recovery")
        },
    )
    write(directory / "RECOVERY.json", result)
    return result


def aggregate(root):
    root = Path(root)
    frozen = require_frozen(root)
    config = read(root / "CONFIG.json")
    tasks, inputs, schedules = pilot_inputs(root, config)
    missing, errors, blocks, all_rows, reviews = [], [], {}, [], []
    for parent in ("b0", "b1", "b2"):
        parent_rows, arm_steps, arm_known, channel_evidence = [], {}, {}, {}
        hard_difference = dict(
            reward_changed_rows=0, reward_changed_groups=0, advantage_changed_groups=0
        )
        for arm in ARMS:
            directory = root / "G3/RUNS" / parent / arm
            lane = directory / "primary"
            if not (lane / "COMPLETE.json").exists():
                missing.append(f"{parent}/{arm}/primary")
                continue
            identity = dict(
                run_id=f"G3-{parent}-{arm}",
                parent=parent,
                arm=arm,
                repeat=0,
                frozen_digest=digest(frozen),
                reference=config["model"]["parents"][parent],
                protocol_version=PROTOCOL_VERSION,
            )
            completion = read(lane / "COMPLETE.json")
            checkpoints = committed(lane, identity)
            if (
                completion.get("identity") != identity
                or completion.get("status") != "COMPLETE"
                or completion.get("lane") != "primary"
                or completion.get("committed_step") != STEPS
                or completion.get("responses") != 1536
                or 12 not in checkpoints
                or 4 not in checkpoints
            ):
                raise PermissionError("Pilot completion is not bound to final committed path")
            for committed_step in (4, 12):
                checkpoint_header(
                    checkpoints[committed_step] / "state.pt", identity, committed_step
                )
            recovery_path = directory / "RECOVERY.json"
            if parent == "b0" and (
                not recovery_path.exists() or read(recovery_path).get("status") != "PASS"
            ):
                errors.append(f"{parent}/{arm}:recovery_not_passed")
            elif parent == "b0":
                recovery = read(recovery_path)
                if (
                    recovery.get("identity") != identity
                    or not recovery.get("checks")
                    or not all(recovery["checks"].values())
                    or any(
                        recovery.get("checkpoint_metadata", {}).get(name)
                        != file_metadata(directory / name / "checkpoints/step-012/state.pt")
                        or recovery.get("completion_digests", {}).get(name)
                        != digest(read(directory / name / "COMPLETE.json"))
                        for name in ("primary", "recovery")
                    )
                ):
                    raise PermissionError("Recovery evidence is stale or belongs to another path")
            steps, arm_rows = [], []
            measured_native, measured_auxiliary, actual_gradients = [], [], []
            for step in range(1, STEPS + 1):
                groups, generated = [], []
                for group in range(16):
                    row = read(lane / "rollouts" / f"{step:03d}" / f"{group:02d}.json")
                    assignment = schedules[parent][step - 1]["groups"][group]
                    qid = assignment["qid"]
                    expected_spec = dict(
                        identity=identity,
                        step=step,
                        assignment=assignment,
                        input_digest=digest(inputs[qid]),
                        task_digest=digest(tasks[qid]),
                        lane="primary",
                    )
                    if row.get("spec") != expected_spec:
                        raise PermissionError("Rollout specification differs from frozen schedule")
                    validate_rollout(
                        row["records"], assignment, identity, step, inputs[qid]["model_input"]
                    )
                    group_id = f"{identity['run_id']}/{step:03d}/{group:02d}"
                    if any(
                        r.get("qid") != qid
                        or r.get("group_id") != group_id
                        or r.get("row_id") != f"{group_id}/{draw}"
                        or r.get("run_root") != str(root)
                        or r.get("protocol") != "structured_v3"
                        for draw, r in enumerate(row["records"])
                    ):
                        raise PermissionError("Rollout row identity differs from frozen group")
                    generated.extend(row["records"])
                receipt = read(lane / "updates" / f"{step:03d}.json")
                groups = read(lane / "credit" / f"{step:03d}.json")
                for offset, credit in zip(range(0, 128, 8), groups, strict=True):
                    if (
                        build_credit_group(generated[offset : offset + 8], arm, credit["seed"])
                        != credit
                        or credit["seed"]
                        != schedules[parent][step - 1]["groups"][offset // 8]["shuffle_seed"]
                    ):
                        raise PermissionError("Saved pilot credit differs from applied group")
                    if arm == "HARD":
                        current = generated[offset : offset + 8]
                        changed_rows = sum(
                            r["score"]["J"] != r["score"]["J"] * int(r["score"]["S"] == "PASS")
                            for r in current
                        )
                        original = build_credit_group(current, "J", credit["seed"])
                        hard_difference["reward_changed_rows"] += changed_rows
                        hard_difference["reward_changed_groups"] += bool(changed_rows)
                        hard_difference["advantage_changed_groups"] += (
                            original["native"] != credit["native"]
                        )
                if (
                    receipt.get("run_id") != identity["run_id"]
                    or receipt.get("step") != step
                    or receipt.get("parent") != parent
                    or receipt.get("arm") != arm
                    or receipt.get("lane") != "primary"
                    or receipt.get("actual_credit_groups") != groups
                ):
                    raise PermissionError(
                        "Update receipt differs from actual path and coefficients"
                    )
                if step == STEPS and read(checkpoints[12] / "UPDATE.json") != receipt:
                    raise PermissionError("Final update differs from committed checkpoint receipt")
                if step == STEPS:
                    unique = {}
                    for record in sorted(generated, key=lambda r: r["row_id"]):
                        unique.setdefault((record["qid"], record["full_text"]), record)
                    selected = list(unique.values())[:24]
                    if len(selected) != 24:
                        errors.append(f"{parent}/{arm}:fewer_than_24_unique_step12_reviews")
                    for record in selected:
                        review = independent_review(record, tasks[record["qid"]])
                        reviews.append(dict(review, arm=arm, step=STEPS))
                        if review["status"] != "PASS":
                            errors.append(
                                f"{record['row_id']}:independent_measurement_review_failed"
                            )
                if (
                    receipt.get("sequence_count") != 128
                    or receipt.get("optimizer_updates") != 1
                    or receipt.get("status") != "FINITE"
                ):
                    errors.append(f"{parent}/{arm}/{step}:invalid_numerical_update")
                probabilities = (
                    receipt.get("sampler_train_mean_absolute_difference"),
                    receipt.get("tis_truncated_fraction"),
                    receipt.get("sampler_train_max_absolute_difference"),
                )
                if any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in probabilities
                ) or any(
                    x > threshold
                    for x, threshold in zip(probabilities, (0.01, 0.01, 5.0), strict=True)
                ):
                    errors.append(f"{parent}/{arm}/{step}:probability_threshold_not_passed")
                gradient = receipt.get("gradient_norm_before_clip")
                if type(gradient) not in (int, float) or not math.isfinite(gradient):
                    errors.append(f"{parent}/{arm}/{step}:gradient_not_finite")
                else:
                    actual_gradients.append(gradient)
                if (
                    step in (1, 12)
                    and receipt.get("channel_gradients", {}).get("status") != "MEASURED"
                ):
                    errors.append(f"{parent}/{arm}/{step}:channel_diagnostics_missing")
                elif step in (1, 12):
                    channels = receipt["channel_gradients"]
                    if channels.get("row_indices") != list(DIAGNOSTIC_ROWS):
                        raise PermissionError("Channel diagnostic subset changed")
                    values = [
                        channels.get(k) for k in ("native_gradient_norm", "auxiliary_gradient_norm")
                    ]
                    if any(
                        type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values
                    ):
                        errors.append(f"{parent}/{arm}/{step}:channel_gradient_not_measured")
                    else:
                        measured_native.append(values[0])
                        measured_auxiliary.append(values[1])
                summaries = [
                    {k: r[k] for k in ("S", "J", "root_id", "cell", "group_id")} for r in generated
                ]
                arm_rows.extend(summaries)
                parent_rows.extend(summaries)
                groups = [
                    {k: g[k] for k in ("group_id", "nonzero_group", "changed_group")}
                    for g in groups
                ]
                steps.append(
                    dict(
                        step=step,
                        groups=groups,
                        rows=summaries,
                        update=dict(parameter_update_norm=receipt.get("parameter_update_norm", 0)),
                    )
                )
            arm_known[arm] = sum(r["S"] != "UNKNOWN" for r in arm_rows) / len(arm_rows)
            if arm_known[arm] < 0.9:
                errors.append(f"{parent}/{arm}:S_observation_below_90_percent")
            if not any(s["update"]["parameter_update_norm"] > 0 for s in steps):
                errors.append(f"{parent}/{arm}:no_actual_parameter_update")
            channel_evidence[arm] = dict(
                maximum_native_gradient_norm=max(measured_native, default=0),
                maximum_auxiliary_gradient_norm=max(measured_auxiliary, default=0),
                maximum_actual_gradient_norm=max(actual_gradients, default=0),
            )
            if max(actual_gradients, default=0) <= 0:
                errors.append(f"{parent}/{arm}:no_actual_nonzero_gradient")
            if arm in ("DENSE", "LOCAL", "SHUF") and max(measured_auxiliary, default=0) <= 0:
                errors.append(f"{parent}/{arm}:nonzero_auxiliary_gradient_not_verified")
            arm_steps[arm] = steps
        if not hard_difference["reward_changed_rows"]:
            errors.append(f"{parent}/HARD:no_actual_reward_difference")
        all_rows.extend(parent_rows)
        windows = []
        for start in (0, 4, 8):
            windows.append(
                dict(
                    LOCAL_nonzero_groups=sum(
                        g["nonzero_group"]
                        for s in arm_steps.get("LOCAL", [])[start : start + 4]
                        for g in s["groups"]
                    ),
                    SHUF_changed_groups=sum(
                        g["changed_group"]
                        for s in arm_steps.get("SHUF", [])[start : start + 4]
                        for g in s["groups"]
                    ),
                )
            )
        signals = [
            r
            for s in arm_steps.get("LOCAL", [])
            for g in s["groups"]
            if g["nonzero_group"]
            for r in s["rows"]
            if r["group_id"] == g["group_id"]
        ]
        feedback = True
        for arm, field in (("LOCAL", "nonzero_group"), ("SHUF", "changed_group")):
            signal_steps = [
                s["step"]
                for s in arm_steps.get(arm, [])
                if any(g[field] for g in s["groups"])
                and s["update"].get("parameter_update_norm", 0) > 0
            ]
            feedback &= len(signal_steps) >= 2 and signal_steps[0] < signal_steps[-1]
        blocks[parent] = dict(
            windows=windows,
            signal_roots=len({r["root_id"] for r in signals}),
            signal_families=len({r["cell"] for r in signals}),
            S_known_fraction=sum(r["S"] != "UNKNOWN" for r in parent_rows) / len(parent_rows)
            if parent_rows
            else 0,
            verified_intervention_resampling_intervention=feedback,
            S_known_fraction_by_arm=arm_known,
            hard_reward_difference=hard_difference,
            channel_gradient_evidence_by_arm=channel_evidence,
            state_counts=dict(Counter(f"{r['S']}/J{r['J']}" for r in parent_rows)),
        )
    report = dict(
        actual_response_count=len(all_rows),
        blocks=blocks,
        missing=missing,
        errors=errors,
        numerical_and_runtime_checks="PASS" if not missing and not errors else "NOT_COMPLETE",
        frozen_digest=digest(frozen),
        scope="G3 feasibility; no arm ranking or formal outputs",
        unique_step12_reviews=len(reviews),
    )
    gate = check_pilot(report)
    if missing:
        gate["status"] = "NOT_COMPLETE"
    gate["errors"].extend(errors)
    report["gate"] = gate
    write(root / "G3/METRICS.json", report)
    write_rows(root / "G3/PROGRAM_REVIEWS.jsonl", reviews)
    write(
        root / "GATES/G3.json",
        dict(gate, at=now(), evidence="G3/METRICS.json", frozen_digest=digest(frozen)),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", choices=("b0", "b1", "b2"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--lane", choices=("primary", "recovery"), default="primary")
    parser.add_argument("--stop-after", type=int, choices=(4, STEPS), default=STEPS)
    parser.add_argument("--freeze-revision")
    parser.add_argument("--compare-recovery", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.freeze_revision:
        result = freeze(args.root, args.freeze_revision)
    elif args.aggregate:
        result = aggregate(args.root)
    elif args.parent is None or args.arm is None:
        parser.error("--parent and --arm are required for a path")
    elif args.compare_recovery:
        result = compare_recovery(args.root, args.parent, args.arm)
    else:
        result = run(args.root, args.parent, args.arm, args.lane, args.stop_after)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
