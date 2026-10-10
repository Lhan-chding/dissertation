"""Actual GPU four-step versus fresh-process two-plus-two SR-F1 qualification."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from mm_core.training import state_hash
from mm_dev.engine import forwarding_lease_signals, wait_engine_child

from .contract import PLAN_ID, digest, reward_advantages, stable_seed
from .data import load_inputs, load_tasks
from .json_protocol import score_record, validate_record
from .runtime import (
    atomic_json,
    bounded_path,
    file_hash,
    load_runtime,
    read_json,
    runtime_account,
    verified_adapter,
)
from .training import (
    CHECKPOINT_FIELDS,
    amendment_record_identity,
    execute_path,
    format_failure_accounting,
    stop_at_committed_boundary,
    token_path_record,
)


def engine_schedule(root):
    """Fixed balanced ENGINE-only stream, established without reading responses."""
    tasks = load_tasks(root)
    strata = {}
    for qid, task in tasks.items():
        if task["pool"] == "ENGINE":
            strata.setdefault((task["family"], task["chart"]), []).append(qid)
    if len(strata) != 8 or any(len(rows) != 16 for rows in strata.values()):
        raise PermissionError("ENGINE pool identity/quota differs")
    for key, rows in strata.items():
        rows.sort()
        random.Random(stable_seed(PLAN_ID, "ENGINE_ORDER", *key)).shuffle(rows)
    schedule = []
    for step in range(1, 5):
        chosen = [qid for key in sorted(strata) for qid in strata[key][(step - 1) * 2 : step * 2]]
        for slot, qid in enumerate(chosen):
            schedule.append(
                dict(
                    step=step,
                    slot=slot,
                    qid=qid,
                    rollout_seeds=[
                        stable_seed(PLAN_ID, "ENGINE_SAMPLE", step, qid, i) for i in range(8)
                    ],
                )
            )
    path = Path(root) / "manifests/ENGINE_SCHEDULE.json"
    if path.exists():
        if read_json(path) != schedule:
            raise PermissionError("Immutable ENGINE schedule changed")
    else:
        atomic_json(path, schedule, exclusive=True)
    return schedule


def engine_run(mode):
    if mode not in ("natural", "stress"):
        raise ValueError("Unregistered ENGINE mode")
    return dict(
        run_id="SRF1_ENGINE_" + mode.upper(),
        paired_seed=2026100902,
        H=4,
        B=16,
        G=8,
        arms=["DEC", "GATE", "DEC", "GATE"],
        common_start="SRF1_COMMON_START",
        technical_only=True,
    )


def run_segment(plan, root, *, mode, segment):
    with stop_at_committed_boundary() as boundary:
        return _run_segment(plan, root, mode=mode, segment=segment, boundary=boundary)


def _run_segment(plan, root, *, mode, segment, boundary):
    if segment not in ("continuous", "first", "resume"):
        raise PermissionError("Only registered continuous or 2+2 segments are allowed")
    root = Path(root)
    track = "continuous" if segment == "continuous" else "split"
    directory = root / "engineering/engine" / mode / track
    expected_start, end = (
        (0, 4) if segment == "continuous" else (0, 2) if segment == "first" else (2, 4)
    )
    latest = directory / "checkpoints/LATEST.json"
    actual_start = read_json(latest)["step"] if latest.exists() else 0
    if actual_start != expected_start:
        raise PermissionError("ENGINE segment must begin at its registered fresh-process boundary")
    recovery = _claim_zero_update_recovery_process(root, mode, segment)
    if mode == "stress":
        natural = read_json(root / "engineering/engine/natural/COMPARISON.json")
        if not natural.get("stress_allowed"):
            raise PermissionError(
                "Surrogate ENGINE is allowed only after entirely constant natural groups"
            )
    account = runtime_account(root, "ENGINE")
    runtime = load_runtime(
        plan,
        root,
        state_id="SRF1_COMMON_START",
        account=lambda kind, count, metadata: account(
            kind, count, {**metadata, "engine_mode": mode, "engine_segment": segment}
        ),
    )
    for config in getattr(runtime, "compute_parallel_config", {}).get("worker_configs", []):
        config["account"].update(engine_mode=mode, engine_segment=segment)
    result = execute_path(
        runtime,
        root,
        directory,
        engine_run(mode),
        engine_schedule(root),
        load_inputs(root),
        load_tasks(root),
        stop_step=end,
        engine=True,
        stress=mode == "stress",
        boundary=boundary,
    )
    if result["status"] != "COMPLETE":
        atomic_json(directory / "SEGMENT_CHECKPOINTED.json", result)
        return result
    receipt = dict(
        segment=segment,
        mode=mode,
        track=track,
        pid=os.getpid(),
        start=expected_start,
        stop=end,
        result=result,
        hardware=runtime.identity["hardware"],
        determinism=runtime.identity["determinism"],
        common_start_hash=verified_adapter(root, "SRF1_COMMON_START")["trainable_state_hash"],
    )
    if recovery is not None:
        receipt["zero_update_recovery"] = recovery
    atomic_json(directory / (segment.upper() + "_COMPLETE.json"), receipt, exclusive=True)
    return result


def _read_state(directory, step):
    import torch

    receipt = read_json(directory / "checkpoints" / f"commit-{step:02d}.json")
    path = bounded_path(directory / "checkpoints", receipt["path"])
    if file_hash(path) != receipt["sha256"]:
        raise PermissionError("ENGINE checkpoint bytes changed")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if set(state) != CHECKPOINT_FIELDS or state_hash(state) != receipt["state_hash"]:
        raise PermissionError("ENGINE checkpoint contents changed")
    if receipt["field_hashes"] != {k: state_hash(v) for k, v in state.items()}:
        raise PermissionError("ENGINE checkpoint field identities changed")
    return state


def compare_engine(root, mode):
    import math

    root = Path(root)
    directory = root / "engineering/engine" / mode
    tracks = [directory / name for name in ("continuous", "split")]
    segments = [
        read_json(tracks[0] / "CONTINUOUS_COMPLETE.json"),
        read_json(tracks[1] / "FIRST_COMPLETE.json"),
        read_json(tracks[1] / "RESUME_COMPLETE.json"),
    ]
    if len({s["pid"] for s in segments}) != 3:
        raise PermissionError("ENGINE requires three distinct actual Python processes")
    if any(
        s["hardware"] != segments[0]["hardware"] or s["determinism"] != segments[0]["determinism"]
        for s in segments
    ):
        raise PermissionError("ENGINE hardware/deterministic backend differs across processes")
    common = verified_adapter(root, "SRF1_COMMON_START")
    schedule = engine_schedule(root)
    initial = _read_state(tracks[0], 0)
    amendment = initial["run_identity"]["training_identity"].get("protocol_amendment")
    amendment_id = amendment["id"] if amendment else None
    comparisons = []
    for step in range(5):
        states = [_read_state(track, step) for track in tracks]
        fields = {
            k: state_hash(states[0][k]) == state_hash(states[1][k]) for k in CHECKPOINT_FIELDS
        }
        if not all(fields.values()):
            raise RuntimeError("ENGINE_EXACT_RESUME_MISMATCH:" + json.dumps(fields))
        if (
            states[0]["input_stream_hash"] != digest(schedule)
            or states[0]["plan_id"] != PLAN_ID
            or states[0]["committed_logical_step"] != step
        ):
            raise PermissionError("ENGINE checkpoint stream/plan/step differs")
        if step == 0 and state_hash(states[0]["parameters"]) != common["trainable_state_hash"]:
            raise PermissionError("ENGINE did not start from the scientific common adapter")
        if step and not states[0]["optimizer"]["state"]:
            raise PermissionError("ENGINE did not populate Adam moments")
        comparisons.append(dict(step=step, fields=fields))
    tasks = load_tasks(root)
    metrics = []
    keys = (
        "qid",
        "logical_step",
        "slot",
        "sample_index",
        "seed",
        "tokens",
        "raw_text",
        "old_logprobs",
        "sampler_logprobs",
        "policy_hash",
        "sampling_hash",
        "image_routing",
    )
    for track in tracks:
        processes = [read_json(p) for p in track.glob("PROCESS-*.json")]
        if len(processes) != (1 if track.name == "continuous" else 2):
            raise PermissionError("ENGINE has an unexpected number of process attempts")
        if any(p["restored_rng_hash"] != p["checkpoint_rng_hash"] for p in processes):
            raise RuntimeError("ENGINE RNG restoration mismatch")
        if len(list((track / "rollouts").glob("*.json"))) != 512:
            raise PermissionError("ENGINE must retain 512 generated sequences per track")
    for step in range(1, 5):
        pair = [read_json(t / "steps" / f"{step:02d}.json") for t in tracks]
        if pair[0] != pair[1]:
            raise RuntimeError("ENGINE loss/advantage/gradient diagnostics differ")
        for track, row in zip(tracks, pair, strict=True):
            state = _read_state(track, step)
            if digest(row) != state["diagnostics_hash"] or row[
                "parameter_hash_after"
            ] != state_hash(state["parameters"]):
                raise PermissionError("ENGINE metrics are not bound to checkpoint state")
        raw_tracks = []
        for track in tracks:
            rows, rescored = [], []
            for slot in range(16):
                schedule_row = schedule[(step - 1) * 16 + slot]
                group_scores = []
                for index in range(8):
                    record = read_json(track / "rollouts" / f"{step:02d}-{slot:02d}-{index}.json")
                    if (
                        record["record_hash"]
                        != digest({k: v for k, v in record.items() if k != "record_hash"})
                        or record["generation_status"] != "COMPLETE"
                        or record["qid"] != schedule_row["qid"]
                        or record["seed"] != schedule_row["rollout_seeds"][index]
                        or record["old_logprobs"] != record["sampler_logprobs"]
                        or len(record["tokens"]) != len(record["old_logprobs"])
                        or record["image_routing"]["generation_vision_forward_calls"] <= 0
                    ):
                        raise PermissionError("ENGINE raw generation provenance differs")
                    if (
                        record["logical_step"] != step
                        or record["slot"] != slot
                        or record["sample_index"] != index
                        or record["policy_hash"] != pair[0]["parameter_hash_before"]
                        or not all(math.isfinite(v) for v in record["old_logprobs"])
                    ):
                        raise PermissionError("ENGINE raw slot or sampling policy identity differs")
                    validate_record(record, amendment_id)
                    task = tasks[record["qid"]]
                    group_scores.append(score_record(record, task))
                    rows.append(
                        {**{k: record[k] for k in keys}, **amendment_record_identity(record)}
                    )
                rescored.append(group_scores)
            advantage, audit = reward_advantages(("DEC", "GATE", "DEC", "GATE")[step - 1], rescored)
            expected_coefficients = (
                [1.0 if i % 2 == 0 else -1.0 for i in range(128)]
                if mode == "stress"
                else advantage.reshape(-1).tolist()
            )
            token_path = [token_path_record(r) for r in rows]
            if "format_failures" in pair[0] and pair[0][
                "format_failures"
            ] != format_failure_accounting(rescored):
                raise PermissionError("ENGINE per-round format failure accounting differs")
            if (
                rescored != pair[0]["scores"]
                or audit != pair[0]["reward_advantage_audit"]
                or expected_coefficients != pair[0]["final_advantages"]
                or digest(token_path) != pair[0]["token_path_hash"]
                or pair[0]["natural_nonconstant_groups"] != sum(bool(any(a)) for a in advantage)
                or pair[0]["all_natural_advantages_zero"] != (not bool(advantage.any()))
            ):
                raise PermissionError("ENGINE raw scores/final advantage/token-path audit differs")
            raw_tracks.append(rows)
        if raw_tracks[0] != raw_tracks[1]:
            raise RuntimeError("ENGINE generated tokens, raw probabilities, slots or seeds differ")
        diffs = pair[0]["sampler_teacher_forcing"]["absolute_differences"]
        if (
            not diffs
            or len(diffs) != pair[0]["completion_tokens"]
            or not all(math.isfinite(d) and d >= 0 for d in diffs)
            or sum(diffs) / len(diffs) > 0.005
            or max(diffs) > 0.05
        ):
            raise RuntimeError("ENGINE native sampler probability tolerance failed")
        if pair[0]["parameter_hash_before"] != state_hash(
            _read_state(tracks[0], step - 1)["parameters"]
        ) or pair[0]["reference_hash"] != state_hash(_read_state(tracks[0], step)["reference"]):
            raise PermissionError("ENGINE parameter/reference state chain differs")
        metrics.append(pair[0])
    nonconstant = sum(m["natural_nonconstant_groups"] for m in metrics)
    if mode == "natural" and any(
        m["natural_nonconstant_groups"] and m["policy_gradient_norm"] <= 0 for m in metrics
    ):
        raise RuntimeError("GRADIENT_MISMATCH")
    stress_allowed = (
        mode == "natural"
        and nonconstant == 0
        and all(
            m["all_natural_advantages_zero"] and m["policy_gradient_norm"] == 0 for m in metrics
        )
    )
    if mode == "stress" and not all(
        m["policy_gradient_norm"] > 0 and m["parameter_changed"] for m in metrics
    ):
        raise RuntimeError("SURROGATE_GRADIENT_MISMATCH")
    cost_rows = [
        json.loads(line) for line in (root / "accounting/ENGINE.jsonl").read_text().splitlines()
    ]
    cost_rows = [r for r in cost_rows if r["metadata"].get("engine_mode") == mode]
    attempts = sum(r["count"] for r in cost_rows if r["kind"] == "completion_attempts")
    updates = sum(r["count"] for r in cost_rows if r["kind"] == "physical_optimizer_updates")
    if attempts < 1024 or updates < 8:
        raise PermissionError("ENGINE cost ledger does not cover actual physical attempts")
    artifacts = {
        str(p.relative_to(root)): file_hash(p)
        for track in tracks
        for p in track.rglob("*")
        if p.is_file()
    }
    recovery = segments[0].get("zero_update_recovery")
    if recovery is not None:
        expected = _zero_update_recovery_identity(root)[0]
        activation_path, process_path = _zero_update_recovery_markers(root)
        activation, process = read_json(activation_path), read_json(process_path)
        if (
            mode != "natural"
            or any(activation.get(k) != v for k, v in expected.items())
            or process != recovery
            or process.get("activation_sha256") != file_hash(activation_path)
            or process.get("repair_sha256") != expected["repair_sha256"]
            or process.get("pid") != segments[0]["pid"]
            or process.get("resume_step") != 0
        ):
            raise PermissionError("ENGINE zero-update process identity differs")
        physical = read_json(next(tracks[0].glob("PROCESS-*.json")))
        if physical["pid"] != process["pid"] or physical["initial_step"] != 0:
            raise PermissionError("ENGINE zero-update recovery was not its one actual process")
        artifacts.update(
            {
                str(p.relative_to(root)): file_hash(p)
                for p in (activation_path, process_path, root / "ENGINE_MEMORY_REPAIR.json")
            }
        )
    result = dict(
        plan_id=PLAN_ID,
        mode=mode,
        status="EXACT_CONSTANT_REWARD_TRACE" if stress_allowed else "EXACT_NONZERO_TRACE",
        stress_allowed=stress_allowed,
        natural_nonconstant_groups=nonconstant,
        logical_steps=4,
        physical_updates=updates,
        completion_attempts=attempts,
        valid_trace_updates=8,
        valid_trace_completions=1024,
        exact_state_and_rng=True,
        exact_token_and_slot_paths=True,
        all_probability_gates_passed=True,
        comparisons=comparisons,
        process_ids=[s["pid"] for s in segments],
        artifact_hashes=artifacts,
    )
    atomic_json(directory / "COMPARISON.json", result, exclusive=True)
    return result


def run_engine(plan, root):
    with forwarding_lease_signals() as lease:
        return _run_engine(plan, root, lease)


def _zero_update_recovery_markers(root):
    incident = Path(root) / "technical_incidents/engine_memory_20261010"
    return (
        incident / "ZERO_UPDATE_REUSE_ACTIVATED.json",
        incident / "ZERO_UPDATE_REUSE_PROCESS.json",
    )


def _zero_update_recovery_identity(root):
    """Authenticate permanent history; never bind a later preflight to live LATEST."""
    from .freeze import verify_engine_memory_repair

    verified = verify_engine_memory_repair(root)
    recovery = read_json(Path(root) / "ENGINE_MEMORY_REPAIR.json")["zero_update_recovery"]
    expected = {
        "staged_track_relative_path": "engineering/engine/natural/continuous",
        "archived_track_relative_path": (
            "technical_incidents/engine_memory_20261010/evidence/"
            "engineering/engine/natural/continuous"
        ),
        "original_raw_count": 128,
        "engine_mode": "natural",
        "resume_step": 0,
    }
    if any(recovery.get(k) != v for k, v in expected.items()):
        raise PermissionError("ENGINE zero-update recovery boundary differs")
    return (
        dict(
            **expected,
            plan_id=PLAN_ID,
            repair_sha256=verified["repair_sha256"],
            original_checkpoint_state_hash=recovery["original_checkpoint_state_hash"],
            staged_inventory_sha256=digest(recovery["artifact_hashes"]),
        ),
        recovery,
    )


def _validate_zero_update_stage(root, recovery):
    """Permit only the authenticated initial checkpoint and its first 16x8 raw batch."""
    root = Path(root).resolve()
    relative = recovery["staged_track_relative_path"]
    directory = bounded_path(root, relative)
    latest = read_json(directory / "checkpoints/LATEST.json")
    committed = read_json(directory / "checkpoints/commit-00.json")
    if (
        latest != committed
        or latest.get("step") != 0
        or latest.get("state_hash") != recovery["original_checkpoint_state_hash"]
    ):
        raise PermissionError("ENGINE recovery requires the original committed step zero")
    checkpoint = bounded_path(directory / "checkpoints", latest["path"])
    expected = {
        relative + "/RUN_MANIFEST.json",
        relative + "/checkpoints/LATEST.json",
        relative + "/checkpoints/commit-00.json",
        str(checkpoint.relative_to(root)),
        *{
            relative + f"/rollouts/01-{slot:02d}-{index}.json"
            for slot in range(16)
            for index in range(8)
        },
    }
    if set(recovery["artifact_hashes"]) != expected:
        raise PermissionError("ENGINE zero-update recovery inventory differs")
    paths = list((root / relative).rglob("*"))
    if (root / relative).is_symlink() or any(
        p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths
    ):
        raise PermissionError("ENGINE recovery staging contains non-regular artifacts")
    actual = {str(p.relative_to(root)): file_hash(p) for p in paths if p.is_file()}
    if actual != recovery["artifact_hashes"] or file_hash(checkpoint) != latest["sha256"]:
        raise PermissionError("ENGINE recovery staging bytes or complete inventory differ")


def _activate_zero_update_recovery(root, mode):
    root = Path(root)
    activation_path, _ = _zero_update_recovery_markers(root)
    if (
        mode != "natural"
        or not (root / "ENGINE_MEMORY_REPAIR.json").exists()
        or activation_path.exists()
    ):
        return False
    identity, recovery = _zero_update_recovery_identity(root)
    directory = root / recovery["staged_track_relative_path"]
    if set(directory.parent.iterdir()) != {directory}:
        raise PermissionError("ENGINE recovery staging contains another trial or process")
    _validate_zero_update_stage(root, recovery)
    # An exclusive, fsynced marker consumes the exception even if the child never
    # starts. A later interrupted attempt follows the normal whole-mode archive.
    atomic_json(
        activation_path,
        {**identity, "activating_pid": os.getpid()},
        exclusive=True,
    )
    return True


def _claim_zero_update_recovery_process(root, mode, segment):
    root = Path(root)
    if mode != "natural" or segment != "continuous":
        return None
    if not (root / "ENGINE_MEMORY_REPAIR.json").exists():
        return None
    staged = root / "engineering/engine/natural/continuous"
    if not (staged / "checkpoints/LATEST.json").exists():
        return None  # An ordinary fresh trial after archiving an interrupted retry.
    activation_path, process_path = _zero_update_recovery_markers(root)
    if not activation_path.exists() or process_path.exists():
        raise PermissionError("ENGINE zero-update reuse is absent or already consumed")
    identity, recovery = _zero_update_recovery_identity(root)
    activation = read_json(activation_path)
    if (
        any(activation.get(k) != v for k, v in identity.items())
        or activation.get("activating_pid") != os.getppid()
    ):
        raise PermissionError("ENGINE recovery activation does not belong to this parent")
    _validate_zero_update_stage(root, recovery)
    claim = dict(
        repair_sha256=identity["repair_sha256"],
        activation_sha256=file_hash(activation_path),
        pid=os.getpid(),
        resume_step=0,
    )
    atomic_json(process_path, claim, exclusive=True)
    return claim


def _previous_compute_mode(root, mode):
    """Identify traces from an earlier compute revision, even if fully complete."""
    root = Path(root)
    repair = root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json"
    directory = root / "engineering/engine" / mode
    if not repair.exists() or not directory.exists() or not any(directory.iterdir()):
        return None
    current = file_hash(repair)
    observed = {}
    for track in ("continuous", "split"):
        path = directory / track / "RUN_MANIFEST.json"
        if not (directory / track).exists():
            continue
        observed[track] = (
            read_json(path)
            .get("run_identity", {})
            .get("training_identity", {})
            .get("compute_parallel", {})
            .get("repair_sha256")
            if path.exists()
            else None
        )
    complete_requires_both = (directory / "COMPARISON.json").exists()
    if (
        not observed
        or any(value != current for value in observed.values())
        or (complete_requires_both and set(observed) != {"continuous", "split"})
    ):
        return dict(current_compute_repair_sha256=current, prior_track_compute_revisions=observed)
    return None


def _preserve_previous_compute_mode(root, mode):
    """A new compute implementation must repeat the entire independent 4 vs 2+2."""
    import time

    evidence = _previous_compute_mode(root, mode)
    if evidence is None:
        return None
    root = Path(root)
    directory = root / "engineering/engine" / mode
    archived = root / "engineering/engine_interrupted" / f"{mode}-{time.time_ns()}"
    archived.parent.mkdir(parents=True, exist_ok=True)
    os.rename(directory, archived)
    atomic_json(
        archived / "COMPUTE_REPAIR_RESTART_PRESERVED.json",
        dict(
            **evidence,
            reason="Previous compute revision cannot certify the current compute implementation",
            repeat_scope="entire_engine_mode",
            no_scientific_adapter_inherited=True,
        ),
        exclusive=True,
    )
    return archived


def _preserve_interrupted_mode(root, mode):
    """Retain an interrupted physical trace before repeating its registered trial."""
    import time

    previous_compute = _preserve_previous_compute_mode(root, mode)
    if previous_compute is not None:
        return previous_compute
    if _activate_zero_update_recovery(root, mode):
        return None
    directory = Path(root) / "engineering/engine" / mode
    continuous, split = directory / "continuous", directory / "split"
    partial_continuous = (
        continuous.exists() and not (continuous / "CONTINUOUS_COMPLETE.json").exists()
    )
    split_markers = (split / "FIRST_COMPLETE.json", split / "RESUME_COMPLETE.json")
    partial_split = (
        split.exists()
        and not split_markers[1].exists()
        and (not split_markers[0].exists() or len(list(split.glob("PROCESS-*.json"))) > 1)
    )
    if partial_continuous or partial_split:
        archived = Path(root) / "engineering/engine_interrupted" / f"{mode}-{time.time_ns()}"
        archived.parent.mkdir(parents=True, exist_ok=True)
        os.rename(directory, archived)
        atomic_json(
            archived / "INTERRUPTION_PRESERVED.json",
            dict(
                reason="Interrupted trace cannot replace registered 4 vs 2+2 boundaries",
                repeat_scope="entire_engine_mode",
                no_scientific_adapter_inherited=True,
            ),
            exclusive=True,
        )
        return archived
    return None


def _run_engine(plan, root, lease):
    root = Path(root)
    verified_adapter(root, "SRF1_COMMON_START")
    gate = root / "ENGINE_PROBABILITY_GRADIENT_RESUME.json"
    if gate.exists():
        result = read_json(gate)
        if (root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json").exists():
            for mode in ("natural", "stress"):
                if mode == "stress" and result.get("stress") is None:
                    continue
                directory = root / "engineering/engine" / mode
                if (
                    not all(
                        (directory / track / "RUN_MANIFEST.json").exists()
                        for track in ("continuous", "split")
                    )
                    or _previous_compute_mode(root, mode) is not None
                ):
                    raise PermissionError(
                        "Existing ENGINE gate belongs to a previous compute revision"
                    )
        for relative, expected in result["artifact_hashes"].items():
            if file_hash(bounded_path(root, relative)) != expected:
                raise PermissionError("ENGINE gate evidence changed")
        return dict(status="COMPLETE", artifacts=[str(gate.relative_to(root))], metadata=result)
    # This check precedes COMPARISON reuse: a fully completed old implementation
    # is still not acceptance evidence for the new compute implementation.
    for mode in ("natural", "stress"):
        _preserve_previous_compute_mode(root, mode)
    engine_schedule(root)
    config_path = root / "engineering/ENGINE_CONFIG.json"
    if config_path.exists():
        if read_json(config_path) != plan:
            raise PermissionError("ENGINE configuration changed")
    else:
        atomic_json(config_path, plan, exclusive=True)
    results = []
    for mode in ("natural", "stress"):
        if mode == "stress" and not results[0]["stress_allowed"]:
            break
        comparison = root / "engineering/engine" / mode / "COMPARISON.json"
        if comparison.exists():
            existing = read_json(comparison)
            for relative, expected in existing["artifact_hashes"].items():
                if file_hash(bounded_path(root, relative)) != expected:
                    raise PermissionError("Existing ENGINE trace changed")
            results.append(existing)
            continue
        _preserve_interrupted_mode(root, mode)
        for segment in ("continuous", "first", "resume"):
            track = "continuous" if segment == "continuous" else "split"
            directory = root / "engineering/engine" / mode
            marker = directory / track / (segment.upper() + "_COMPLETE.json")
            if marker.exists():
                continue
            # An interrupted non-boundary trace cannot be silently called an exact
            # fresh-process trial. Preserve and explicitly rerun its track later.
            logs = directory / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / f"{segment}-{os.getpid()}.log").open("x") as output:
                result = wait_engine_child(
                    [
                        sys.executable,
                        "-m",
                        "sr_f1.engine",
                        "--root",
                        str(root),
                        "--plan",
                        str(config_path),
                        "--mode",
                        mode,
                        "--segment",
                        segment,
                    ],
                    output,
                    lease,
                )
            if result.returncode == 75:
                checkpointed = read_json(directory / track / "SEGMENT_CHECKPOINTED.json")
                latest_path = directory / track / "checkpoints/LATEST.json"
                return _engine_recovery_snapshot(root, latest_path, checkpointed)

            result.check_returncode()
        results.append(compare_engine(root, mode))
    evidence = {k: v for result in results for k, v in result["artifact_hashes"].items()}
    for mode_result in results:
        name = f"engineering/engine/{mode_result['mode']}/COMPARISON.json"
        evidence[name] = file_hash(root / name)
    receipt = dict(
        plan_id=PLAN_ID,
        status="PASS_NONZERO_SURROGATE_KERNEL_RESUME"
        if len(results) == 2
        else "PASS_NATURAL_GRPO_RESUME",
        freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        common_start_sha256=file_hash(root / "COMMON_START.json"),
        natural=results[0],
        stress=results[1] if len(results) == 2 else None,
        artifact_hashes=evidence,
        claim=(
            "Native probability, gradient and recovery qualification only; not a scientific result"
        ),
    )
    atomic_json(gate, receipt, exclusive=True)
    return dict(status="COMPLETE", artifacts=[str(gate.relative_to(root))], metadata=receipt)


def _engine_recovery_snapshot(root, latest_path, checkpointed):
    """Immutable hard-linked evidence survives archival of an interrupted trial."""
    root, latest_path = Path(root), Path(latest_path)
    latest = read_json(latest_path)
    source = bounded_path(latest_path.parent, latest["path"])
    if file_hash(source) != latest["sha256"]:
        raise PermissionError("Interrupted ENGINE checkpoint bytes changed")
    destination = root / "recovery_checkpoints/ENGINE" / latest["sha256"]
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "state.pt"
    if not target.exists():
        os.link(source, target)
    if file_hash(target) != latest["sha256"]:
        raise PermissionError("Immutable ENGINE recovery snapshot differs")
    receipt = {**latest, "path": "state.pt"}
    marker = destination / "COMMIT.json"
    if marker.exists():
        if read_json(marker) != receipt:
            raise PermissionError("ENGINE recovery receipt differs")
    else:
        atomic_json(marker, receipt, exclusive=True)
    return dict(
        status="CHECKPOINTED",
        reason=checkpointed["reason"],
        artifacts=[str(marker.relative_to(root)), str(target.relative_to(root))],
        metadata={**checkpointed["metadata"], "restart_engine_track_required": True},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("natural", "stress"), required=True)
    parser.add_argument("--segment", choices=("continuous", "first", "resume"), required=True)
    args = parser.parse_args()
    result = run_segment(read_json(args.plan), args.root, mode=args.mode, segment=args.segment)
    print(json.dumps(result))
    sys.exit(75 if result["status"] == "CHECKPOINTED" else 0)
