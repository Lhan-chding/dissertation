"""SR-F1 final-advantage training, immutable raw slots and identity-bound recovery."""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import tempfile
from collections import Counter
from pathlib import Path

from mm_core.training import (
    _cpu_tree,
    capture_rng,
    configure_training,
    restore_rng,
    state_hash,
    trainable_state,
)
from mm_core.vl_runtime import hash_json, seed_all

from .contract import PLAN_ID, digest, reward_advantages, score
from .runtime import atomic_json, bounded_path, copy_parameters, file_hash, read_json

CHECKPOINT_FIELDS = frozenset(
    {
        "plan_id",
        "run_identity",
        "committed_logical_step",
        "parameters",
        "optimizer",
        "scheduler",
        "rng",
        "reference",
        "reference_hash",
        "input_stream_hash",
        "sampling_hash",
        "cursor",
        "diagnostics_hash",
        "token_path_hash",
    }
)


def sequence_objective(current, old, reference, coefficient):
    import torch

    if not current.requires_grad or old.requires_grad or reference.requires_grad:
        raise ValueError("Policy must require gradients; old and reference must be detached")
    if current.shape != old.shape or current.shape != reference.shape or current.numel() == 0:
        raise ValueError("Completion probability vectors must align")
    current, old, reference = current.float(), old.float(), reference.float()
    ratio = torch.exp(current - old)
    displacement = reference - current
    kl = torch.expm1(displacement) - displacement
    coefficient = torch.as_tensor(coefficient, dtype=torch.float32, device=current.device)
    if (
        coefficient.requires_grad
        or coefficient.numel() != 1
        or not bool(torch.isfinite(coefficient))
    ):
        raise ValueError("Final advantage must be one finite detached sequence coefficient")
    policy = -torch.minimum(ratio * coefficient, ratio.clamp(0.8, 1.2) * coefficient)
    for value in (current, old, reference, ratio, kl, policy):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite GRPO quantities; no clipping repair permitted")
    return dict(
        loss=(policy + 0.02 * kl).mean(),
        policy=policy.mean(),
        kl=kl.mean(),
        clip_fraction=((ratio < 0.8) | (ratio > 1.2)).float().mean(),
    )


def atomic_torch(path, state, *, exclusive=False):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-state-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_state(
    runtime,
    optimizer,
    scheduler,
    *,
    run_identity,
    step,
    reference,
    stream_hash,
    sampling_hash,
    diagnostics_hash=None,
    token_path_hash=None,
):
    return dict(
        plan_id=PLAN_ID,
        run_identity=run_identity,
        committed_logical_step=step,
        parameters=trainable_state(runtime.model),
        optimizer=_cpu_tree(optimizer.state_dict()),
        scheduler=scheduler.state_dict(),
        rng=capture_rng(),
        reference=_cpu_tree(reference),
        reference_hash=state_hash(reference),
        input_stream_hash=stream_hash,
        sampling_hash=sampling_hash,
        cursor=dict(next_logical_step=step + 1, next_slot=0, next_sample_index=0),
        diagnostics_hash=diagnostics_hash,
        token_path_hash=token_path_hash,
    )


def load_checkpoint(
    directory,
    runtime,
    optimizer,
    scheduler,
    *,
    run_identity,
    reference,
    stream_hash,
    sampling_hash,
    step=None,
):
    import torch

    directory = Path(directory)
    marker = directory / ("LATEST.json" if step is None else f"commit-{step:02d}.json")
    receipt = read_json(marker)
    path = bounded_path(directory, receipt["path"])
    if file_hash(path) != receipt["sha256"]:
        raise PermissionError("Checkpoint bytes changed")
    state = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(
        state, receipt, run_identity, stream_hash, sampling_hash, state_hash(reference)
    )
    copy_parameters(runtime.model, state["parameters"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    if state_hash(capture_rng()) != state_hash(state["rng"]):
        raise RuntimeError("RNG was not restored exactly")
    if state_hash(trainable_state(runtime.model)) != state_hash(state["parameters"]):
        raise RuntimeError("Policy parameters were not restored exactly")
    if state_hash(_cpu_tree(optimizer.state_dict())) != state_hash(state["optimizer"]):
        raise RuntimeError("Optimizer was not restored exactly")
    return state


def validate_checkpoint(state, receipt, run_identity, stream_hash, sampling_hash, reference_hash):
    if set(state) != CHECKPOINT_FIELDS or state["plan_id"] != PLAN_ID:
        raise PermissionError("Checkpoint schema/plan changed")
    if state_hash(state) != receipt["state_hash"] or receipt["field_hashes"] != {
        key: state_hash(value) for key, value in state.items()
    }:
        raise PermissionError("Checkpoint field/content hashes changed")
    if state["run_identity"] != run_identity or state["input_stream_hash"] != stream_hash:
        raise PermissionError("Checkpoint run/stream changed")
    if state["sampling_hash"] != sampling_hash or state["reference_hash"] != reference_hash:
        raise PermissionError("Checkpoint sampling/reference changed")
    if state_hash(state["reference"]) != reference_hash or not state["parameters"]:
        raise PermissionError("Empty policy or changed reference state")
    if state["committed_logical_step"] != receipt["step"] or state["cursor"] != dict(
        next_logical_step=receipt["step"] + 1, next_slot=0, next_sample_index=0
    ):
        raise PermissionError("Checkpoint cursor/step changed")


def recover_committed_metrics(directory, state):
    """Reconcile an interruption after checkpoint commit but before metrics publication."""
    step = state["committed_logical_step"]
    if step == 0:
        return
    directory = Path(directory)
    canonical = directory / "steps" / f"{step:02d}.json"
    expected = state["diagnostics_hash"]
    if canonical.exists():
        if digest(read_json(canonical)) != expected:
            raise PermissionError("Committed diagnostics changed")
        return
    candidates = [
        read_json(path) for path in (directory / "update_attempts").glob(f"{step:02d}-*.json")
    ]
    matches = [row for row in candidates if digest(row) == expected]
    if not matches or any(row != matches[0] for row in matches):
        raise PermissionError("Committed checkpoint has no matching executed-update evidence")
    atomic_json(canonical, matches[0], exclusive=True)


def update(
    runtime,
    optimizer,
    scheduler,
    reference,
    examples,
    *,
    stress=False,
    probability_gate=False,
    boundary=None,
):
    """Exactly 128 equal-weight sequence losses; no batch or token-count weighting."""
    import torch

    if len(examples) != 128:
        raise ValueError("SR-F1 requires the full effective batch of 16 x 8 sequences")
    before = trainable_state(runtime.model)
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    pg_accum = [torch.zeros_like(p) for p in parameters]
    optimizer.zero_grad(set_to_none=True)
    totals = Counter()
    difference_sum, difference_max, token_count = 0.0, 0.0, 0
    probability_differences = []
    probability_summaries = {
        key: dict(sum=0.0, minimum=None, maximum=None) for key in ("old", "reference", "current")
    }
    for example_index, example in enumerate(examples):
        if boundary and boundary["requested"]:
            optimizer.zero_grad(set_to_none=True)
            raise LeaseEnding("PREEMPTION_DURING_UNCOMMITTED_GRADIENT")
        record, row, coefficient = example["record"], example["row"], example["advantage"]
        prepared = runtime.prepare(row, example["root"])
        reference_values = runtime.reference_forward(prepared, record["tokens"], reference)
        ref = reference_values["logprobs"].detach()
        current = runtime.sequence_forward(
            prepared, record["tokens"], purpose="training_gradient", grad=True
        )["logprobs"]
        old = torch.tensor(record["old_logprobs"], dtype=torch.float32, device=current.device)
        for key, value in (("old", old), ("reference", ref), ("current", current.detach())):
            summary = probability_summaries[key]
            summary["sum"] += float(value.sum())
            low, high = float(value.min()), float(value.max())
            summary["minimum"] = low if summary["minimum"] is None else min(summary["minimum"], low)
            summary["maximum"] = (
                high if summary["maximum"] is None else max(summary["maximum"], high)
            )
        difference = (current.detach().float() - old).abs()
        probability_differences.extend(difference.cpu().tolist())
        difference_sum += float(difference.sum())
        difference_max = max(difference_max, float(difference.max()))
        token_count += len(record["tokens"])
        objective = sequence_objective(
            current, old, ref, (1.0 if example_index % 2 == 0 else -1.0) if stress else coefficient
        )
        pg = torch.autograd.grad(
            objective["policy"] / 128, parameters, retain_graph=True, allow_unused=False
        )
        for accumulated, gradient in zip(pg_accum, pg, strict=True):
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("Nonfinite isolated policy gradient")
            accumulated.add_(gradient.detach())
        (objective["loss"] / 128).backward()
        for key in ("loss", "policy", "kl", "clip_fraction"):
            totals[key] += float(objective[key].detach()) / 128
    if boundary and boundary["requested"]:
        optimizer.zero_grad(set_to_none=True)
        raise LeaseEnding("PREEMPTION_DURING_UNCOMMITTED_GRADIENT")
    probability = dict(
        mean_absolute_logprob_difference_nats=difference_sum / token_count,
        maximum_absolute_logprob_difference_nats=difference_max,
        token_count=token_count,
        absolute_differences=probability_differences if probability_gate else None,
    )
    if difference_sum / token_count > 0.005 or difference_max > 0.05:
        raise RuntimeError(
            "SAMPLER_TEACHER_FORCING_NUMERIC_GATE_FAILED: " + json.dumps(probability)
        )
    pg_norm = math.sqrt(sum(float(p.double().square().sum()) for p in pg_accum))
    total_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
    if not math.isfinite(pg_norm) or not math.isfinite(total_norm):
        raise FloatingPointError("Nonfinite gradient norm")
    runtime.reserve(
        "physical_optimizer_updates", 1, phase="ENGINE" if probability_gate else "SCIENCE"
    )
    optimizer.step()
    scheduler.step()
    after = trainable_state(runtime.model)
    delta = max(float((after[n] - before[n]).abs().max()) for n in before)
    return dict(
        **totals,
        policy_gradient_norm=pg_norm,
        total_gradient_norm_before_clip=total_norm,
        parameter_max_absolute_delta=delta,
        parameter_hash_before=state_hash(before),
        parameter_hash_after=state_hash(after),
        reference_hash=state_hash(reference),
        parameter_changed=state_hash(before) != state_hash(after),
        sampler_teacher_forcing=probability,
        completion_tokens=token_count,
        effective_sequences=128,
        logprob_summaries={
            key: dict(
                mean=value["sum"] / token_count, minimum=value["minimum"], maximum=value["maximum"]
            )
            for key, value in probability_summaries.items()
        },
        adam_nonzero_moments=all(
            bool(value["exp_avg"].count_nonzero()) and bool(value["exp_avg_sq"].count_nonzero())
            for value in optimizer.state.values()
        ),
        stress_coefficient_override=stress,
        final_advantages=[
            (1.0 if i % 2 == 0 else -1.0) if stress else float(e["advantage"])
            for i, e in enumerate(examples)
        ],
        optimizer_state_hash=state_hash(_cpu_tree(optimizer.state_dict())),
    )


class LeaseEnding(Exception):
    pass


@contextlib.contextmanager
def stop_at_committed_boundary():
    flags = {"requested": False, "signal": None}

    def request(signum, _frame):
        flags.update(requested=True, signal=signum)

    old = {number: signal.signal(number, request) for number in (signal.SIGTERM, signal.SIGUSR1)}
    try:
        yield flags
    finally:
        for number, handler in old.items():
            signal.signal(number, handler)


def commit_checkpoint(directory, state, *, retain_all=True):
    """Durable state bytes first, immutable commit second, atomic latest pointer last.

    Recovery checkpoints never erase previous evidence. Science invokes this at
    0/8/.../96 and before lease exit; ENGINE invokes it at every logical step.
    """
    directory = Path(directory)
    step, content_hash = state["committed_logical_step"], state_hash(state)
    name = f"step-{step:02d}-{content_hash}.pt"
    target = directory / name
    if not target.exists():
        atomic_torch(target, state, exclusive=True)
    receipt = dict(
        step=step,
        path=name,
        sha256=file_hash(target),
        state_hash=content_hash,
        field_hashes={key: state_hash(value) for key, value in state.items()},
    )
    marker = directory / f"commit-{step:02d}.json"
    if marker.exists():
        if read_json(marker) != receipt:
            raise RuntimeError("Committed logical step has conflicting state")
    else:
        atomic_json(marker, receipt, exclusive=True)
    atomic_json(directory / "LATEST.json", receipt)
    return receipt


def rollout_group(
    runtime, root, directory, run, schedule_row, row, task, policy_hash, sampling_hash
):
    records = []
    logical_step, slot = schedule_row["step"], schedule_row["slot"]
    if len(schedule_row["rollout_seeds"]) != 8:
        raise ValueError("Exactly eight preregistered sample seeds are required")
    image_hash = file_hash(bounded_path(root, row["image_file"]))
    for index, sample_seed in enumerate(schedule_row["rollout_seeds"]):
        identity = dict(
            request_id=digest([PLAN_ID, run["run_id"], logical_step, slot, index, policy_hash]),
            run_id=run["run_id"],
            logical_step=logical_step,
            slot=slot,
            sample_index=index,
            qid=row["qid"],
            root_id=task["root_id"],
            seed=sample_seed,
            policy_hash=policy_hash,
            sampling_hash=sampling_hash,
            image_sha256=image_hash,
            input_hash=digest({k: row[k] for k in ("qid", "image_file", "text")}),
        )
        path = Path(directory) / "rollouts" / f"{logical_step:02d}-{slot:02d}-{index}.json"
        if path.exists():
            record = read_json(path)
            if any(record.get(k) != v for k, v in identity.items()):
                raise PermissionError("Incomplete-update rollout identity mismatch")
            if record.get("record_hash") != digest(
                {k: v for k, v in record.items() if k != "record_hash"}
            ):
                raise PermissionError("Durable raw rollout changed")
        else:

            def persist(raw, path=path, identity=identity):
                record = {**identity, **raw}
                if any(raw.get(k) != identity[k] for k in ("seed", "sampling_hash")):
                    record["generation_status"] = "TECHNICAL_INVALID"
                    record["expected_identity"] = identity
                record["record_hash"] = digest(record)
                atomic_json(path, record, exclusive=True)

            runtime.generate_training(row, root, sample_seed, on_completion=persist)
            record = read_json(path)
        if record.get("generation_status") != "COMPLETE":
            raise PermissionError(
                "Technically invalid raw generation is retained but not trainable"
            )
        if (
            len(record["tokens"]) != len(record["old_logprobs"])
            or not record["tokens"]
            or record["old_logprobs"] != record["sampler_logprobs"]
            or not all(math.isfinite(v) for v in record["old_logprobs"])
        ):
            raise PermissionError("Saved sampling probabilities are invalid")
        records.append({**record, "score": score(record["raw_text"], task["world"], task["query"])})
    return records


def execute_path(
    runtime,
    root,
    directory,
    run,
    schedule,
    questions,
    tasks,
    *,
    stop_step=None,
    engine=False,
    stress=False,
    boundary=None,
    on_milestone=None,
):
    """Execute only the frozen input stream; independent groups fill one final batch."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if stress and not engine:
        raise PermissionError("Surrogate coefficients are confined to ENGINE")
    optimizer, scheduler, module_identity = configure_training(runtime)
    reference = trainable_state(runtime.model)
    if runtime.identity.get("trainable_state_hash") != state_hash(reference):
        raise PermissionError("Loaded common start differs from its published identity")
    stream_hash = digest(schedule)
    sampling_hash = hash_json(runtime.training_generation_config().to_dict())
    run_identity = {
        **run,
        "training_identity": module_identity,
        "start_policy_hash": state_hash(reference),
        "stress": stress,
    }
    seed_all(run["paired_seed"])
    manifest = dict(
        plan_id=PLAN_ID,
        run_identity=run_identity,
        stream_hash=stream_hash,
        sampling_hash=sampling_hash,
    )
    manifest_path = directory / "RUN_MANIFEST.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise PermissionError("Immutable run manifest differs")
    else:
        atomic_json(manifest_path, manifest, exclusive=True)
    checkpoint_dir = directory / "checkpoints"
    if (checkpoint_dir / "LATEST.json").exists():
        state = load_checkpoint(
            checkpoint_dir,
            runtime,
            optimizer,
            scheduler,
            run_identity=run_identity,
            reference=reference,
            stream_hash=stream_hash,
            sampling_hash=sampling_hash,
        )
        recover_committed_metrics(directory, state)
    else:
        state = checkpoint_state(
            runtime,
            optimizer,
            scheduler,
            run_identity=run_identity,
            step=0,
            reference=reference,
            stream_hash=stream_hash,
            sampling_hash=sampling_hash,
        )
        commit_checkpoint(checkpoint_dir, state)
    start_step = state["committed_logical_step"]
    process = dict(
        pid=os.getpid(),
        initial_step=start_step,
        restored_rng_hash=state_hash(capture_rng()),
        checkpoint_rng_hash=state_hash(state["rng"]),
        runtime_identity=runtime.identity,
        run_manifest_hash=digest(manifest),
    )
    atomic_json(directory / f"PROCESS-{os.getpid()}-{time_ns()}.json", process, exclusive=True)
    end = run["H"] if stop_step is None else stop_step
    if not 0 <= start_step <= end <= run["H"]:
        raise ValueError("Requested segment does not belong to registered path")
    if on_milestone and start_step in (32, 64, 96):
        # A committed milestone can precede a crash during adapter publication or
        # sentinel evaluation. Repair it before advancing the policy.
        milestone_result = on_milestone(runtime, start_step)
        if milestone_result and milestone_result.get("status") == "CHECKPOINTED":
            return _path_result(
                directory,
                state,
                run,
                "CHECKPOINTED",
                reason=milestone_result.get("reason", "PREEMPTION"),
            )
    for step in range(start_step + 1, end + 1):
        if boundary and (boundary["requested"] or (Path(root) / "STOP").exists()):
            commit_checkpoint(checkpoint_dir, state)
            return _path_result(
                directory,
                state,
                run,
                "CHECKPOINTED",
                reason="STOP_REQUESTED" if (Path(root) / "STOP").exists() else "PREEMPTION",
            )
        slots = sorted((s for s in schedule if s["step"] == step), key=lambda s: s["slot"])
        if [s["slot"] for s in slots] != list(range(16)):
            raise PermissionError("Each update must contain all 16 unique prompt slots")
        policy_hash = state_hash(trainable_state(runtime.model))
        grouped = []
        for slot in slots:
            qid = slot["qid"]
            grouped.append(
                rollout_group(
                    runtime,
                    root,
                    directory,
                    run,
                    slot,
                    questions[qid],
                    tasks[qid],
                    policy_hash,
                    sampling_hash,
                )
            )
            if boundary and (boundary["requested"] or (Path(root) / "STOP").exists()):
                commit_checkpoint(checkpoint_dir, state)
                return _path_result(
                    directory,
                    state,
                    run,
                    "CHECKPOINTED",
                    reason="STOP_REQUESTED" if (Path(root) / "STOP").exists() else "PREEMPTION",
                )
        if state_hash(trainable_state(runtime.model)) != policy_hash:
            raise RuntimeError("Policy changed while collecting its 16 x 8 batch")
        arm = ("DEC", "GATE", "DEC", "GATE")[step - 1] if engine else run["arm"]
        final_advantages, reward_audit = reward_advantages(
            arm, [[r["score"] for r in g] for g in grouped]
        )
        examples = [
            dict(
                record=record,
                row=questions[slot["qid"]],
                root=root,
                advantage=float(final_advantages[b, i]),
            )
            for b, (slot, records) in enumerate(zip(slots, grouped, strict=True))
            for i, record in enumerate(records)
        ]
        try:
            metrics = update(
                runtime,
                optimizer,
                scheduler,
                reference,
                examples,
                stress=stress,
                probability_gate=engine,
                boundary=boundary,
            )
        except LeaseEnding:
            # No optimizer update occurred. The entire raw batch remains durable;
            # recomputing its partial gradient is recorded as physical forward cost.
            commit_checkpoint(checkpoint_dir, state)
            return _path_result(directory, state, run, "CHECKPOINTED", reason="PREEMPTION")
        natural_groups = sum(bool(any(final_advantages[b])) for b in range(16))
        metrics.update(
            logical_step=step,
            arm=arm,
            reward_advantage_audit=reward_audit,
            scores=[[r["score"] for r in g] for g in grouped],
            natural_nonconstant_groups=natural_groups,
            all_natural_advantages_zero=not bool(final_advantages.any()),
        )
        token_path = [
            {
                key: e["record"][key]
                for key in ("qid", "seed", "tokens", "raw_text", "old_logprobs", "image_routing")
            }
            for e in examples
        ]
        metrics["token_path_hash"] = digest(token_path)
        atomic_json(
            directory / "update_attempts" / f"{step:02d}-{os.getpid()}-{time_ns()}.json",
            metrics,
            exclusive=True,
        )
        if engine and natural_groups and not stress and metrics["policy_gradient_norm"] <= 0:
            raise RuntimeError("GRADIENT_MISMATCH")
        if engine and not natural_groups and not stress and metrics["policy_gradient_norm"] != 0:
            raise RuntimeError("ZERO_ADVANTAGE_GRADIENT_MISMATCH")
        if stress and (metrics["policy_gradient_norm"] <= 0 or not metrics["parameter_changed"]):
            raise RuntimeError("SURROGATE_GRADIENT_MISMATCH")
        state = checkpoint_state(
            runtime,
            optimizer,
            scheduler,
            run_identity=run_identity,
            step=step,
            reference=reference,
            stream_hash=stream_hash,
            sampling_hash=sampling_hash,
            diagnostics_hash=digest(metrics),
            token_path_hash=metrics["token_path_hash"],
        )
        if (
            engine
            or step % 8 == 0
            or (boundary and (boundary["requested"] or (Path(root) / "STOP").exists()))
        ):
            commit_checkpoint(checkpoint_dir, state)
        canonical = directory / "steps" / f"{step:02d}.json"
        if canonical.exists():
            if read_json(canonical) != metrics:
                raise RuntimeError("Replayed step differs from preserved executed update")
        else:
            atomic_json(canonical, metrics, exclusive=True)
        if on_milestone and step in (32, 64, 96):
            milestone_result = on_milestone(runtime, step)
            if milestone_result and milestone_result.get("status") == "CHECKPOINTED":
                return _path_result(
                    directory,
                    state,
                    run,
                    "CHECKPOINTED",
                    reason=milestone_result.get("reason", "PREEMPTION"),
                )
        if boundary and (boundary["requested"] or (Path(root) / "STOP").exists()):
            return _path_result(
                directory,
                state,
                run,
                "CHECKPOINTED",
                reason="STOP_REQUESTED" if (Path(root) / "STOP").exists() else "PREEMPTION",
            )
    return _path_result(directory, state, run, "COMPLETE")


def time_ns():
    import time

    return time.time_ns()


def _path_result(directory, state, run, status, reason=None):
    return dict(
        status=status,
        reason=reason,
        directory=str(directory),
        final_step=state["committed_logical_step"],
        metadata=dict(
            full_state=True,
            identity_verified=True,
            next_update=state["committed_logical_step"] + 1,
            run_id=run["run_id"],
        ),
    )


def train_path(plan, root, run_id):
    from .data import load_inputs, load_schedule, load_tasks
    from .evaluation import evaluate_model
    from .runtime import load_runtime, publish_adapter, runtime_account

    root = Path(root)
    engine = read_json(root / "ENGINE_PROBABILITY_GRADIENT_RESUME.json")
    if engine.get("status") not in (
        "PASS_NATURAL_GRPO_RESUME",
        "PASS_NONZERO_SURROGATE_KERNEL_RESUME",
    ):
        raise PermissionError("Real GPU ENGINE qualification is required before science")
    runs = read_json(root / "manifests/RUN_MATRIX.json")
    matches = [r for r in runs if r["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError("Unknown frozen scientific run")
    run = matches[0]
    runtime = load_runtime(
        plan, root, state_id="SRF1_COMMON_START", account=runtime_account(root, run_id)
    )
    directory = root / "training" / run_id

    milestone_artifacts = []

    def milestone(runtime, step):
        publish_adapter(root, runtime, run_id, step=step)
        result = evaluate_model(
            runtime, root, run_id, stage="train_fit", step=step, boundary=boundary
        )
        milestone_artifacts.extend(result.get("artifacts", []))
        return result

    with stop_at_committed_boundary() as boundary:
        result = execute_path(
            runtime,
            root,
            directory,
            run,
            load_schedule(root, run["paired_seed"]),
            load_inputs(root),
            load_tasks(root),
            boundary=boundary,
            on_milestone=milestone,
        )
    receipt = dict(result, plan_id=PLAN_ID)
    atomic_json(directory / "PATH_STATUS.json", receipt)
    result["artifacts"] = [
        str(p.relative_to(root))
        for p in [
            directory / "PATH_STATUS.json",
            directory / "RUN_MANIFEST.json",
            directory / "checkpoints/LATEST.json",
        ]
    ]
    latest = read_json(directory / "checkpoints/LATEST.json")
    result["artifacts"].append(str((directory / "checkpoints" / latest["path"]).relative_to(root)))
    result["artifacts"].extend(sorted(set(milestone_artifacts)))
    if result["status"] == "COMPLETE":
        result["metadata"]["endpoint_adapter"] = str(Path("states") / run_id / "step-96")
    return result


def run_common_bridge(runtime, root, plan, boundary=None):
    """One common 16 x 8 completion-only SFT episode, resumable in full state."""
    import torch

    from mm_core.training import frozen_hash

    from .contract import gold_output
    from .data import load_inputs, load_tasks

    root = Path(root)
    directory = root / "engineering" / "bridge"
    inputs, tasks = load_inputs(root), load_tasks(root)
    qids = sorted(q for q, t in tasks.items() if t["pool"] == "BRIDGE")
    if len(qids) != 128:
        raise PermissionError("Common bridge must use all 128 registered completions exactly once")
    optimizer, scheduler, modules = configure_training(runtime)
    initial = trainable_state(runtime.model)
    base_hash = frozen_hash(runtime.model)
    identity = dict(
        plan_id=PLAN_ID,
        run_id="SRF1_COMMON_BRIDGE",
        H=16,
        B=8,
        sequence_exposures=128,
        module_identity=modules,
        start_policy_hash=state_hash(initial),
        gold_supervision_disclosed=True,
    )
    stream_hash, sampling_hash = digest(qids), "PROGRAM_GOLD_COMPLETION_NO_SAMPLING"
    seed_all(plan["model"]["lora"]["init_seed"])
    checkpoint_dir = directory / "checkpoints"
    if (checkpoint_dir / "LATEST.json").exists():
        state = load_checkpoint(
            checkpoint_dir,
            runtime,
            optimizer,
            scheduler,
            run_identity=identity,
            reference=initial,
            stream_hash=stream_hash,
            sampling_hash=sampling_hash,
        )
    else:
        state = checkpoint_state(
            runtime,
            optimizer,
            scheduler,
            run_identity=identity,
            step=0,
            reference=initial,
            stream_hash=stream_hash,
            sampling_hash=sampling_hash,
        )
        commit_checkpoint(checkpoint_dir, state)
    for step in range(state["committed_logical_step"] + 1, 17):
        optimizer.zero_grad(set_to_none=True)
        losses, lengths, completions = [], [], []
        selected = qids[(step - 1) * 8 : step * 8]
        for qid in selected:
            if boundary and boundary["requested"]:
                optimizer.zero_grad(set_to_none=True)
                commit_checkpoint(checkpoint_dir, state)
                return _bridge_checkpoint_result(root, directory, state)
            task = tasks[qid]
            text = json.dumps(
                gold_output(task["world"], task["query"]), ensure_ascii=False, separators=(",", ":")
            )
            tokens = runtime.encode_completion(text)
            if len(tokens) > 768:
                raise PermissionError("Bridge gold completion exceeds the frozen token channel")
            result = runtime.sequence_forward(
                runtime.prepare(inputs[qid], root),
                tokens,
                purpose="bridge_completion_gradient",
                grad=True,
            )
            loss = -result["logprobs"].float().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite bridge loss")
            losses.append(float(loss.detach()))
            lengths.append(len(tokens))
            completions.append(dict(qid=qid, gold_text=text, tokens=tokens))
            (loss / 8).backward()
        if boundary and boundary["requested"]:
            optimizer.zero_grad(set_to_none=True)
            commit_checkpoint(checkpoint_dir, state)
            return _bridge_checkpoint_result(root, directory, state)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in runtime.model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
        )
        runtime.reserve("physical_optimizer_updates", 1, phase="COMMON_BRIDGE", logical_step=step)
        optimizer.step()
        scheduler.step()
        metrics = dict(
            logical_step=step,
            qids=selected,
            sequence_losses=losses,
            completion_tokens=lengths,
            gold_completions=completions,
            gradient_norm=float(norm),
            parameter_hash=state_hash(trainable_state(runtime.model)),
        )
        atomic_json(
            directory / "update_attempts" / f"{step:02d}-{time_ns()}.json", metrics, exclusive=True
        )
        state = checkpoint_state(
            runtime,
            optimizer,
            scheduler,
            run_identity=identity,
            step=step,
            reference=initial,
            stream_hash=stream_hash,
            sampling_hash=sampling_hash,
            diagnostics_hash=digest(metrics),
            token_path_hash=digest(selected),
        )
        commit_checkpoint(checkpoint_dir, state)
    if frozen_hash(runtime.model) != base_hash:
        raise RuntimeError("Bridge modified a frozen base/vision/projector parameter")
    return dict(
        status="COMPLETE",
        logical_updates=16,
        logical_sequence_exposures=128,
        common_to_all_arms=True,
        gold_supervision_disclosed=True,
        before_parameter_hash=state_hash(initial),
        after_parameter_hash=state_hash(trainable_state(runtime.model)),
        checkpoint=str((checkpoint_dir / "LATEST.json").relative_to(root)),
        frozen_base_unchanged=True,
    )


def _bridge_checkpoint_result(root, directory, state):
    latest_path = directory / "checkpoints/LATEST.json"
    latest = read_json(latest_path)
    return dict(
        status="CHECKPOINTED",
        reason="PREEMPTION",
        artifacts=[
            str(latest_path.relative_to(root)),
            str((latest_path.parent / latest["path"]).relative_to(root)),
        ],
        metadata=dict(
            full_state=True,
            identity_verified=True,
            next_update=state["committed_logical_step"] + 1,
            run_id="SRF1_COMMON_BRIDGE",
        ),
    )
