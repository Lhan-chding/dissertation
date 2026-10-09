"""Registered F2 native GRPO, durable rollout slots, and exact full-state commits."""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import shutil
import signal
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

from mm_core.contracts import operate, parse_response, rational
from mm_core.training import (
    _cpu_tree,
    capture_rng,
    configure_training,
    restore_rng,
    state_hash,
    trainable_state,
)
from mm_core.vl_runtime import seed_all

from .contract import PLAN_ID, advantages, digest, seed
from .runtime import (
    adapter_identity,
    atomic_json,
    bounded_path,
    copy_parameters,
    file_hash,
    load_runtime,
    read_json,
)

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


def rewards(raw_text, row):
    truth = tuple(
        rational(v)
        for v in (
            row["true_values_decimal"] if "true_values_decimal" in row else row["true_values"]
        )
    )
    parsed = parse_response(raw_text, len(truth))
    answer = Fraction(int(parsed.L_A and parsed.answer == operate(truth, row["operation"])))
    read = (
        Fraction(sum(a == b for a, b in zip(parsed.readings, truth, strict=True)), len(truth))
        if parsed.L_P
        else Fraction(0)
    )
    return dict(
        A_full=int(answer),
        q_read=str(read),
        rA=str(answer),
        rAP=str((answer + read) / 2),
        L_P=parsed.L_P,
        L_A=parsed.L_A,
        format_valid=parsed.schema_ok,
        parser_errors=list(parsed.errors),
    )


def group_diagnostics(records, recipe):
    if len(records) != 8 or recipe not in {"A", "AP"}:
        raise ValueError("Registered reward group has eight sequences and recipe A/AP")
    a, ap = [[Fraction(record["reward"][key]) for record in records] for key in ("rA", "rAP")]
    selected = a if recipe == "A" else ap
    adv_a = [0.0] * 8 if len(set(a)) == 1 else advantages(a)
    adv_ap = [0.0] * 8 if len(set(ap)) == 1 else advantages(ap)
    mean = sum(selected) / 8
    variance = sum((r - mean) ** 2 for r in selected) / 8
    return dict(
        reward_recipe=recipe,
        rewards=[str(r) for r in selected],
        reward_mean=str(mean),
        reward_population_variance=str(variance),
        reward_population_std=math.sqrt(float(variance)),
        advantages=adv_a if recipe == "A" else adv_ap,
        zero_contrast=variance == 0,
        counterfactual_advantages_A=adv_a,
        counterfactual_advantages_AP=adv_ap,
        d_adv=math.sqrt(sum((x - y) ** 2 for x, y in zip(adv_a, adv_ap, strict=True)) / 8),
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


def commit_checkpoint(directory, state, *, retain_all=False):
    """Atomic pointer is the commit boundary; two previous physical states survive."""
    directory = Path(directory)
    step = state["committed_logical_step"]
    content_hash = state_hash(state)
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
    # Keep immutable scientific milestones and the latest two committed states.
    if not retain_all:
        for old in directory.glob("commit-*.json"):
            prior = read_json(old)
            if prior["step"] not in {0, 8, 16, 24, 32, step, step - 1}:
                prior_file = bounded_path(directory, prior["path"])
                if prior_file.exists():
                    prior_file.unlink()
    return receipt


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


def pin_checkpoint(root, directory, task_id):
    """Immutable hard-linked recovery evidence survives subsequent rolling cleanup."""
    root, directory = Path(root).resolve(), Path(directory)
    receipt = read_json(directory / "checkpoints/LATEST.json")
    source = bounded_path(directory / "checkpoints", receipt["path"])
    if file_hash(source) != receipt["sha256"]:
        raise PermissionError("Cannot pin a changed checkpoint")
    relative = Path("recovery_checkpoints") / task_id / os.environ["MM_DEV_ATTEMPT_ID"]
    destination = bounded_path(root, relative)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "state.pt"
    if target.exists():
        if file_hash(target) != receipt["sha256"]:
            raise PermissionError("Attempt recovery anchor already contains different state")
    else:
        os.link(source, target)
    marker = destination / "COMMIT.json"
    if not marker.exists():
        atomic_json(marker, receipt, exclusive=True)
    elif read_json(marker) != receipt:
        raise PermissionError("Attempt recovery anchor changed")
    return [str(target.relative_to(root)), str(marker.relative_to(root))]


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


def rollout_group(runtime, root, directory, run, schedule_row, row, policy_hash, sampling_hash):
    records = []
    logical_step, slot = schedule_row["logical_step"], schedule_row["slot"]
    for index in range(8):
        sample_seed = seed("rollout", run["phase"], run["repeat"], row["question_id"], index)
        request_id = digest(
            [PLAN_ID, run["phase"], run["run_id"], logical_step, slot, index, policy_hash]
        )
        identity = dict(
            request_id=request_id,
            run_id=run["run_id"],
            logical_step=logical_step,
            slot=slot,
            sample_index=index,
            question_id=row["question_id"],
            seed=sample_seed,
            policy_hash=policy_hash,
            sampling_hash=sampling_hash,
            input_hash=digest(
                {key: row[key] for key in ("question_id", "image_path", "image_sha256", "prompt")}
            ),
        )
        path = Path(directory) / "rollouts" / f"{logical_step:02d}-{slot:02d}-{index}.json"
        if path.exists():
            record = read_json(path)
            if any(record.get(key) != value for key, value in identity.items()):
                raise PermissionError("Incomplete-update rollout identity mismatch")
            if record["record_hash"] != digest(
                {k: v for k, v in record.items() if k != "record_hash"}
            ):
                raise PermissionError("Durable raw rollout changed")
        else:

            def persist(raw, path=path, identity=identity):
                record = {**identity, **raw}
                mismatches = [
                    key for key in ("seed", "sampling_hash") if raw.get(key) != identity[key]
                ]
                if "question_id" in raw and raw["question_id"] != identity["question_id"]:
                    mismatches.append("question_id")
                if mismatches:
                    record["generation_status"] = "TECHNICAL_INVALID"
                    record["expected_identity"] = identity
                    record["technical_validation_errors"] = [
                        *record.get("technical_validation_errors", []),
                        "ACTUAL_SAMPLER_IDENTITY_MISMATCH:" + ",".join(mismatches),
                    ]
                record["record_hash"] = digest(record)
                atomic_json(path, record, exclusive=True)
                if mismatches:
                    raise PermissionError("Actual sampler identity differs; invalid raw preserved")

            runtime.generate_training(row, root, sample_seed, on_completion=persist)
            record = read_json(path)
        if record.get("generation_status") != "COMPLETE":
            raise PermissionError("Technically invalid raw generation cannot be reused")
        record["reward"] = rewards(record["raw_text"], row)
        records.append(record)
    return records


def update(
    runtime, optimizer, scheduler, reference, examples, *, stress=False, probability_gate=False
):
    """Exactly 192 equal-weight sequence losses; no batch or token-count weighting."""
    import torch

    if len(examples) != 192:
        raise ValueError("F2 requires the full effective batch of 24 x 8 sequences")
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
    for example in examples:
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
        objective = sequence_objective(current, old, ref, 1.0 if stress else coefficient)
        pg = torch.autograd.grad(
            objective["policy"] / 192, parameters, retain_graph=True, allow_unused=False
        )
        for accumulated, gradient in zip(pg_accum, pg, strict=True):
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("Nonfinite isolated policy gradient")
            accumulated.add_(gradient.detach())
        (objective["loss"] / 192).backward()
        for key in ("loss", "policy", "kl", "clip_fraction"):
            totals[key] += float(objective[key].detach()) / 192
    probability = dict(
        mean_absolute_logprob_difference_nats=difference_sum / token_count,
        maximum_absolute_logprob_difference_nats=difference_max,
        token_count=token_count,
        absolute_differences=probability_differences if probability_gate else None,
    )
    if probability_gate and (difference_sum / token_count > 0.005 or difference_max > 0.05):
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
        effective_sequences=192,
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


def execute_path(
    runtime,
    root,
    directory,
    run,
    schedule,
    questions,
    *,
    stop_step=None,
    engine=False,
    stress=False,
    resume_directory=None,
    resume_step=None,
    boundary=None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    optimizer, scheduler, training_identity = configure_training(runtime)
    reference = trainable_state(runtime.model)
    if runtime.identity.get("trainable_state_hash") != state_hash(reference):
        raise PermissionError("Loaded starting adapter differs from published parameter identity")
    stream_hash = digest(schedule)
    sampling_hash = digest(runtime.training_generation_config().to_dict())
    run_identity = {
        **run,
        "training_identity": training_identity,
        "start_policy_hash": state_hash(reference),
        "stress": stress,
    }
    seed_all(
        seed("engine", 0)
        if engine
        else run.get("preparation_master_seed", run.get("future_master_seed"))
    )
    manifest = dict(
        plan_id=PLAN_ID,
        run_identity=run_identity,
        stream_hash=stream_hash,
        sampling_hash=sampling_hash,
    )
    manifest_path = directory / "RUN_MANIFEST.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise PermissionError("Immutable training run manifest differs")
    else:
        atomic_json(manifest_path, manifest, exclusive=True)
    checkpoint_directory = directory / "checkpoints"
    resume_from = Path(resume_directory) if resume_directory else checkpoint_directory
    if (resume_from / "LATEST.json").exists():
        state = load_checkpoint(
            resume_from,
            runtime,
            optimizer,
            scheduler,
            run_identity=run_identity,
            reference=reference,
            stream_hash=stream_hash,
            sampling_hash=sampling_hash,
            step=resume_step,
        )
        start_step = state["committed_logical_step"]
        recover_committed_metrics(directory, state)
    else:
        start_step = 0
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
        commit_checkpoint(checkpoint_directory, state, retain_all=engine)
    atomic_json(
        directory / f"PROCESS-{os.getpid()}.json",
        dict(
            pid=os.getpid(),
            initial_step=start_step,
            initial_parameters_hash=state_hash(state["parameters"]),
            restored_rng_hash=state_hash(capture_rng()),
            checkpoint_rng_hash=state_hash(state["rng"]),
            run_manifest_hash=digest(manifest),
            runtime_identity=runtime.identity,
        ),
        exclusive=True,
    )
    end = stop_step or run["steps"]
    for step in range(start_step + 1, end + 1):
        if boundary and boundary["requested"]:
            raise LeaseEnding("PREEMPTION")
        slots = [s for s in schedule if s["logical_step"] == step]
        if len(slots) != 24 or sorted(s["slot"] for s in slots) != list(range(24)):
            raise PermissionError("Frozen schedule does not contain all 24 unique slots")
        policy_hash = state_hash(trainable_state(runtime.model))
        examples, groups = [], []
        for slot in slots:
            row = questions[slot["question_id"]]
            records = rollout_group(
                runtime, root, directory, run, slot, row, policy_hash, sampling_hash
            )
            group = group_diagnostics(records, run.get("preparation_recipe", "A"))
            groups.append(
                {
                    **slot,
                    **group,
                    "completion_rewards": [r["reward"] for r in records],
                    "truncated_samples": sum(bool(r["truncated"]) for r in records),
                    "completion_tokens": sum(len(r["tokens"]) for r in records),
                }
            )
            examples.extend(
                dict(record=r, row=row, root=root, advantage=a)
                for r, a in zip(records, group["advantages"], strict=True)
            )
            if boundary and boundary["requested"]:
                raise LeaseEnding("PREEMPTION")
        if state_hash(trainable_state(runtime.model)) != policy_hash:
            raise RuntimeError("Rollout policy changed within update")
        metrics = update(
            runtime,
            optimizer,
            scheduler,
            reference,
            examples,
            stress=stress,
            probability_gate=engine,
        )
        metrics.update(
            logical_step=step,
            groups=groups,
            natural_nonconstant_groups=sum(not g["zero_contrast"] for g in groups),
            all_natural_advantages_zero=all(a["advantage"] == 0 for a in examples),
        )
        token_path_hash = digest(
            [
                {
                    "question_id": e["row"]["question_id"],
                    "tokens": e["record"]["tokens"],
                    "raw_text": e["record"]["raw_text"],
                    "image_routing": e["record"]["image_routing"],
                    "old_logprobs": e["record"]["old_logprobs"],
                    "seed": e["record"]["seed"],
                }
                for e in examples
            ]
        )
        metrics["token_path_hash"] = token_path_hash
        # Save executed-update evidence before committing; failed attempts remain visible.
        attempt_name = f"{step:02d}-{os.getpid()}-{state_hash(metrics)}.json"
        atomic_json(directory / "update_attempts" / attempt_name, metrics, exclusive=True)
        if (
            engine
            and metrics["natural_nonconstant_groups"]
            and not stress
            and (metrics["policy_gradient_norm"] <= 0 or not metrics["parameter_changed"])
        ):
            raise RuntimeError("GRPO_GRADIENT_MISMATCH")
        if (
            engine
            and not metrics["natural_nonconstant_groups"]
            and not stress
            and metrics["policy_gradient_norm"] != 0
        ):
            raise RuntimeError("ZERO_ADVANTAGE_POLICY_GRADIENT_MISMATCH")
        if stress and (metrics["policy_gradient_norm"] <= 0 or not metrics["parameter_changed"]):
            raise RuntimeError("SURROGATE_POLICY_GRADIENT_MISMATCH")
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
            token_path_hash=token_path_hash,
        )
        commit_checkpoint(checkpoint_directory, state, retain_all=engine)
        canonical_metrics = directory / "steps" / f"{step:02d}.json"
        if canonical_metrics.exists():
            if read_json(canonical_metrics) != metrics:
                raise RuntimeError("Conflicting canonical committed diagnostics")
        else:
            atomic_json(canonical_metrics, metrics, exclusive=True)
        if boundary and boundary["requested"]:
            raise LeaseEnding("PREEMPTION")
    return dict(
        checkpoint=read_json(checkpoint_directory / "LATEST.json"),
        run_identity=run_identity,
        final_step=end,
        process_id=os.getpid(),
    )


def publish_state(root, runtime, run, result):
    root = Path(root)
    path = root / "states" / run["run_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-adapter-", dir=path.parent))
    try:
        runtime.model.save_pretrained(staging, safe_serialization=True)
        if path.exists():
            expected = adapter_identity(root, str(staging.relative_to(root)))
            existing = adapter_identity(root, str(path.relative_to(root)))
            if expected["adapter_file_hashes"] != existing["adapter_file_hashes"]:
                raise PermissionError("Uncommitted endpoint differs from committed parameters")
        else:
            os.rename(staging, path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    identity = adapter_identity(root, str(path.relative_to(root)))
    identity.update(
        plan_id=PLAN_ID,
        freeze_sha256=file_hash(root / "manifests/F2_FREEZE.json"),
        base_model_weights_hash=read_json(root / "manifests/F2_FREEZE.json")["model_identity"][
            "model_weights_hash"
        ],
        producer_run_id=run["run_id"],
        checkpoint_sha256=result["checkpoint"]["sha256"],
        trainable_state_hash=state_hash(trainable_state(runtime.model)),
        logical_step=result["final_step"],
    )
    registry_path = root / "manifests/STATE_REGISTRY.json"
    lock = registry_path.with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        registry = (
            read_json(registry_path) if registry_path.exists() else dict(plan_id=PLAN_ID, states={})
        )
        for key in [run["run_id"], *([run["output_state"]] if "output_state" in run else [])]:
            if key in registry["states"] and registry["states"][key] != identity:
                raise PermissionError("State identity already published with different contents")
            registry["states"][key] = identity
        atomic_json(registry_path, registry)
    return identity


def run_train_path(plan_path, run_root, run_id):
    from .common import CostLedger, load_plan, require_allocation, verify_execution
    from .orchestration import checkpoint_task, complete_task, worker_lease

    root, plan_path = Path(run_root).resolve(), Path(plan_path).resolve()
    plan = load_plan(plan_path)
    freeze = verify_execution(plan_path, root, require_engine=True)
    matrix = read_json(plan_path.parent / "run_matrix.json")
    matches = [entry for entry in matrix if entry["run_id"] == run_id]
    if len(matches) != 1:
        raise PermissionError("Training only permits registered matrix run IDs")
    run = matches[0]
    task_id = os.environ.get("MM_DEV_TASK_ID", run_id)
    if task_id != run_id:
        raise PermissionError("Scheduler task does not match training path")
    require_allocation(root, task_id)
    with worker_lease(root, task_id), stop_at_committed_boundary() as boundary:
        ledger = CostLedger(root, task_id)
        runtime = load_runtime(
            plan, root, state_id=run["start_state"], account=ledger.reserve, freeze=freeze
        )
        schedule = [
            json.loads(line)
            for line in (plan_path.parent / "schedules" / (run["schedule_id"] + ".jsonl"))
            .read_text()
            .splitlines()
        ]
        questions = {
            row["question_id"]: row
            for row in map(json.loads, (root / "data/questions.jsonl").read_text().splitlines())
        }
        directory = root / "training" / run_id
        try:
            result = execute_path(
                runtime, root, directory, run, schedule, questions, boundary=boundary
            )
        except LeaseEnding as error:
            checkpoint_task(root, task_id, str(error), pin_checkpoint(root, directory, task_id))
            return dict(status="CHECKPOINTED", reason=str(error))
        identity = publish_state(root, runtime, run, result)
        completion = {**result, "state": identity}
        if (directory / "COMPLETE.json").exists():
            old = read_json(directory / "COMPLETE.json")
            if old["checkpoint"] != result["checkpoint"] or old["state"] != identity:
                raise PermissionError("Training completion identity changed")
        else:
            atomic_json(directory / "COMPLETE.json", completion, exclusive=True)
        complete_task(
            root,
            task_id,
            [
                str((directory / "COMPLETE.json").relative_to(root)),
                identity["adapter_path"] + "/adapter_model.safetensors",
            ],
            dict(run_id=run_id, final_step=32),
        )
        return result
