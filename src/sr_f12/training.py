"""SR-F1.2 single-update, full-sequence TIS objective and durable recovery.

No SR-F1 defaults or reward definitions are changed. Each group remains eight
responses, each update sixteen groups, and every response contributes 1/128.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from mm_core.training import _cpu_tree, capture_rng, restore_rng, state_hash, trainable_state
from sr_f1.contract import reward_advantages as reward_advantages
from sr_f1.runtime import copy_parameters, file_hash, read_json
from sr_f1.training import commit_checkpoint as _commit_checkpoint

PLAN_ID = "SR-F1.2"
BATCH_SIZE = 128
TIS_CAP = 2.0
KL_COEFFICIENT = 0.02
ALLOWED_LEARNING_RATES = (5e-5, 2e-5, 1e-4)
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
        "training_recipe",
    }
)


class ProbabilityMismatch(RuntimeError):
    """An uncommitted update failed the registered alignment/diagnosis gate."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class MicrobatchNumericalMismatch(RuntimeError):
    """A measured batch/single failure; args are (message, actual diagnostics)."""


def sequence_objective(current, sampler, reference, coefficient, *, kl_coefficient=0.02):
    """Token mean with detached train-old and detached cap-2 importance weights."""
    import torch

    if not current.requires_grad or sampler.requires_grad or reference.requires_grad:
        raise ValueError("Current logprobs require gradients; sampler/reference must be detached")
    if (
        current.ndim != 1
        or current.numel() == 0
        or current.shape != sampler.shape
        or (current.shape != reference.shape)
    ):
        raise ValueError("Nonempty unpadded completion token vectors must align")
    if kl_coefficient not in (0.0, KL_COEFFICIENT):
        raise ValueError("Only registered KL=0.02 or the zero-KL unit check is permitted")
    current, sampler, reference = current.float(), sampler.float(), reference.float()
    advantage = torch.as_tensor(coefficient, device=current.device, dtype=torch.float32)
    if advantage.numel() != 1 or advantage.requires_grad or not bool(torch.isfinite(advantage)):
        raise ValueError("One finite detached final reward advantage is required")
    if any(not bool(torch.isfinite(v).all()) for v in (current, sampler, reference)):
        raise FloatingPointError("Nonfinite completion logprob")
    old = current.detach()
    ratio = torch.exp(current - old)
    delta = old - sampler
    # Clamp before exp so a large finite diagnostic mismatch cannot overflow w.
    weights = torch.exp(delta.clamp(max=math.log(TIS_CAP))).detach()
    displacement = reference - current
    kl = torch.expm1(displacement) - displacement
    policy = -weights * torch.minimum(advantage * ratio, advantage * ratio.clamp(0.8, 1.2))
    loss = (policy + kl_coefficient * kl).mean()
    if any(not bool(torch.isfinite(v).all()) for v in (ratio, weights, kl, policy, loss)):
        raise FloatingPointError("Nonfinite SR-F1.2 objective")
    return {
        "loss": loss,
        "policy": policy.mean(),
        "kl": kl.mean(),
        "ratio": ratio,
        "weights": weights,
        "clip_fraction": ((ratio < 0.8) | (ratio > 1.2)).float().mean(),
        "absolute_difference": delta.abs().detach(),
        "tis_truncated": (delta > math.log(TIS_CAP)).detach(),
    }


def probability_diagnostics(differences, truncated):
    import torch

    values = torch.cat([v.detach().float().cpu().reshape(-1) for v in differences])
    masks = torch.cat([v.detach().bool().cpu().reshape(-1) for v in truncated])
    if values.numel() == 0 or values.shape != masks.shape or not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Invalid probability diagnostics")
    return dict(
        token_count=values.numel(),
        sampler_train_mean_absolute_difference=float(values.mean()),
        sampler_train_p99_absolute_difference=float(torch.quantile(values, 0.99)),
        sampler_train_max_absolute_difference=float(values.max()),
        tis_truncated_fraction=float(masks.float().mean()),
        tis_cap=TIS_CAP,
    )


def enforce_probability_gate(diagnostics):
    if diagnostics["sampler_train_mean_absolute_difference"] > 0.01:
        raise ProbabilityMismatch("Mean sampler/train discrepancy exceeds 0.01 nat", diagnostics)
    if diagnostics["tis_truncated_fraction"] > 0.01:
        raise ProbabilityMismatch(
            "TIS truncation exceeds 1%; diagnosis required before continuing", diagnostics
        )


def _qid(record):
    row = record.get("row", {})
    return record.get("qid", row.get("qid", row.get("question_id")))


def _microbatches(records, microbatch_size):
    if microbatch_size not in (4, 2, 1):
        raise ValueError("Frozen microbatch must be 4, 2, or 1")
    if len(records) != BATCH_SIZE:
        raise ValueError("Every update requires sixteen complete groups of eight")
    for start in range(0, BATCH_SIZE, 8):
        group = records[start : start + 8]
        identity = _qid(group[0])
        if identity is None or any(_qid(r) != identity for r in group):
            raise ValueError("Every consecutive eight records must share one question")
        for offset in range(0, 8, microbatch_size):
            yield start + offset, group[offset : offset + microbatch_size]


def _validate_lr(optimizer_state, scheduler_state, expected=None):
    groups = optimizer_state["param_groups"]
    if not groups:
        raise PermissionError("Optimizer has no registered groups")
    rates = [g["lr"] for g in groups]
    if any(rate not in ALLOWED_LEARNING_RATES for rate in rates) or len(set(rates)) != 1:
        raise PermissionError("Learning rate differs from SR-F1.2 allowed rates")
    if expected is not None and rates != [expected] * len(groups):
        raise PermissionError("Learning rate differs from frozen training recipe")
    if any(g.get("initial_lr") != rate for g, rate in zip(groups, rates, strict=True)):
        raise PermissionError("Initial learning rate differs")
    if scheduler_state.get("base_lrs") != rates or scheduler_state.get("_last_lr") != rates:
        raise PermissionError("Constant scheduler learning rate differs")


def _finite_tensor_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tensor_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensor_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def update(
    runtime,
    optimizer,
    scheduler,
    records,
    coefficients,
    *,
    reference,
    microbatch_size=4,
    save_train_logprobs=False,
):
    """A single optimizer update. Failed diagnostics never consume an Adam step.

    Runtime must expose batch_sequence_forward(records, purpose, grad) and
    batch_reference_forward(records, reference), returning unpadded tensor lists.
    Only forward/backward are microbatched; the registered effective batch is 128.
    """
    import torch

    batches = list(_microbatches(records, microbatch_size))
    if len(coefficients) != BATCH_SIZE:
        raise ValueError("Exactly 128 final advantages are required")
    _validate_lr(optimizer.state_dict(), scheduler.state_dict())
    for record in records:
        if "sampler_logprobs" not in record or len(record["tokens"]) != len(
            record["sampler_logprobs"]
        ):
            raise ValueError("Explicit sampler_logprobs must align with actual completion tokens")
    parameters = [p for p in runtime.model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    before = trainable_state(runtime.model)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    totals = dict(loss=0.0, policy=0.0, kl=0.0, clip_fraction=0.0)
    differences, truncated, train_values = [], [], []
    backward_calls = 0
    try:
        for offset, batch in batches:
            with torch.no_grad():
                references = runtime.batch_reference_forward(batch, reference)
            currents = runtime.batch_sequence_forward(batch, purpose="training_policy", grad=True)
            if len(currents) != len(batch) or len(references) != len(batch):
                raise ValueError("Runtime returned the wrong number of rows")
            losses = []
            for i, (record, current, ref) in enumerate(
                zip(batch, currents, references, strict=True)
            ):
                sampler = torch.as_tensor(
                    record["sampler_logprobs"], device=current.device, dtype=torch.float32
                )
                result = sequence_objective(current, sampler, ref, coefficients[offset + i])
                losses.append(result["loss"] / BATCH_SIZE)
                for name in totals:
                    totals[name] += float(result[name].detach()) / BATCH_SIZE
                differences.append(result["absolute_difference"].cpu())
                truncated.append(result["tis_truncated"].cpu())
                if save_train_logprobs:
                    train_values.append(current.detach().float().cpu().tolist())
            torch.stack(losses).sum().backward()
            backward_calls += 1
            del currents, references, losses
        diagnostics = probability_diagnostics(differences, truncated)
        enforce_probability_gate(diagnostics)
        if totals["clip_fraction"] != 0.0:
            raise RuntimeError("Detached train-old ratio self-check failed")
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError("Nonfinite accumulated gradient; Adam update withheld")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if any(not bool(torch.isfinite(p).all()) for p in parameters):
            raise FloatingPointError("Nonfinite parameters after Adam; do not commit this update")
        if not _finite_tensor_tree(optimizer.state_dict()):
            raise FloatingPointError("Nonfinite Adam state after update; do not commit")
        after = trainable_state(runtime.model)
        max_change = max(float((after[name] - value).abs().max()) for name, value in before.items())
        adam_nonzero = any(
            bool(torch.count_nonzero(state.get("exp_avg", torch.zeros(()))))
            or bool(torch.count_nonzero(state.get("exp_avg_sq", torch.zeros(()))))
            for state in optimizer.state.values()
        )
        result = {
            **totals,
            **diagnostics,
            "total_gradient_norm": float(norm),
            "parameter_max_absolute_change": max_change,
            "parameters_changed": max_change > 0,
            "adam_nonzero_moments": adam_nonzero,
            "microbatch_size": microbatch_size,
            "backward_calls": backward_calls,
            "optimizer_updates": 1,
            "sequence_count": BATCH_SIZE,
            "finite": True,
            "training_seconds": time.perf_counter() - started,
        }
        if save_train_logprobs:
            result["train_logprobs"] = train_values
        return result
    except BaseException:
        # Do not retry a smaller microbatch mid-science; this is a preflight choice.
        optimizer.zero_grad(set_to_none=True)
        raise


def select_microbatch(probe):
    """Technical-only selection; probe must not perform an optimizer update.

    probe(microbatch_size, gradient_checkpointing) tests a representative batch.
    Only CUDA OOM or a measured batch/single mismatch can select a smaller
    registered microbatch. A one-sequence numerical failure always stops; only
    a one-sequence OOM permits the final checkpointing fallback.
    Freeze the returned choice for every scientific arm before training.
    """
    import gc

    import torch

    attempts = []
    for size, checkpointing in ((4, False), (2, False), (1, False), (1, True)):
        try:
            evidence = probe(size, checkpointing)
        except MicrobatchNumericalMismatch as exc:
            attempts.append(
                dict(
                    microbatch_size=size,
                    gradient_checkpointing=checkpointing,
                    status="NUMERICAL_FAIL",
                    message=str(exc),
                    diagnostics=exc.args[1],
                )
            )
            if size == 1:
                raise MicrobatchNumericalMismatch(
                    "Single-sequence numerical gate failed; no smaller registered choice", attempts
                ) from exc
            continue
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append(
                dict(
                    microbatch_size=size,
                    gradient_checkpointing=checkpointing,
                    status="CUDA_OOM",
                    message=str(exc),
                )
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        attempts.append(
            dict(
                microbatch_size=size,
                gradient_checkpointing=checkpointing,
                status="PASS",
                evidence=evidence,
            )
        )
        return dict(microbatch_size=size, gradient_checkpointing=checkpointing, attempts=attempts)
    raise RuntimeError(f"No registered microbatch fits: {attempts}")


def first_batch_movement(before, after):
    """Token-weighted movement on exactly the original 128 pilot responses."""
    import torch

    if len(before) != BATCH_SIZE or len(after) != BATCH_SIZE:
        raise ValueError("Movement audit requires the first 128 pilot responses")
    differences = []
    for left, right in zip(before, after, strict=True):
        left = torch.as_tensor(left, dtype=torch.float32).detach().cpu()
        right = torch.as_tensor(right, dtype=torch.float32).detach().cpu()
        if left.ndim != 1 or left.numel() == 0 or left.shape != right.shape:
            raise ValueError("Movement audit token vectors differ")
        delta = (left - right).abs()
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("Nonfinite pilot movement")
        differences.append(delta)
    return float(torch.cat(differences).mean())


def select_pilot_learning_rate(metrics, movement, *, attempt=0, learning_rate=5e-5):
    """Preregistered stability/movement decision; no answer or joint score input."""
    if attempt not in (0, 1) or learning_rate not in ALLOWED_LEARNING_RATES:
        raise ValueError("At most one learning-rate adjustment is permitted")
    if attempt == 0 and learning_rate != 5e-5:
        raise ValueError("Initial eight-step pilot must use 5e-5")
    if attempt == 1 and learning_rate not in (2e-5, 1e-4):
        raise ValueError("Second pilot must use the selected adjusted learning rate")
    nonfinite = any(
        not row.get("finite", True)
        or any(
            name in row and not math.isfinite(float(row[name]))
            for name in ("loss", "kl", "total_gradient_norm")
        )
        for row in metrics
    )
    if not nonfinite and (len(metrics) != 8 or [r["step"] for r in metrics] != list(range(1, 9))):
        raise ValueError("A stable pilot requires all eight ordered updates")
    if not nonfinite and (movement is None or not math.isfinite(movement) or movement < 0):
        raise ValueError("Step-zero/eight first-batch token movement must be finite")
    # Equal 128-sequence step sizes make the mean equivalent to pooled failures.
    recent = [r["format_failure_rate"] for r in metrics if 5 <= r["step"] <= 8]
    if any(not 0 <= value <= 1 for value in recent):
        raise ValueError("Invalid format failure rate")
    late_failure = sum(recent) / len(recent) if recent else None
    unstable = (
        nonfinite
        or (late_failure is not None and late_failure > 0.20)
        or (len(metrics) == 8 and metrics[-1]["kl"] > 0.05)
    )
    if attempt == 1:
        rate, status = learning_rate, "STOP_UNSTABLE" if unstable else "PASS"
    else:
        rate = 2e-5 if unstable else 1e-4 if movement < 0.01 else 5e-5
        status = "RETRY_ONCE" if rate != learning_rate else "PASS"
    return dict(
        status=status,
        learning_rate=rate,
        prior_learning_rate=learning_rate,
        attempt=attempt,
        nonfinite=nonfinite,
        late_format_failure_rate=late_failure,
        step8_kl=metrics[-1].get("kl") if len(metrics) == 8 else None,
        first_batch_mean_absolute_logprob_movement=movement,
        score_metrics_used=False,
    )


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
    microbatch_size,
    diagnostics_hash=None,
    token_path_hash=None,
):
    _validate_lr(optimizer.state_dict(), scheduler.state_dict())
    if microbatch_size not in (4, 2, 1) or not isinstance(step, int) or not 0 <= step <= 96:
        raise ValueError("Invalid checkpoint cursor or microbatch")
    if not _finite_tensor_tree(optimizer.state_dict()) or not _finite_tensor_tree(
        trainable_state(runtime.model)
    ):
        raise FloatingPointError("Refusing to commit a nonfinite policy/Adam state")
    rate = optimizer.param_groups[0]["lr"]
    return dict(
        plan_id=PLAN_ID,
        run_identity=run_identity,
        committed_logical_step=step,
        parameters=trainable_state(runtime.model),
        optimizer=_cpu_tree(optimizer.state_dict()),
        scheduler=_cpu_tree(scheduler.state_dict()),
        rng=capture_rng(),
        reference=_cpu_tree(reference),
        reference_hash=state_hash(reference),
        input_stream_hash=stream_hash,
        sampling_hash=sampling_hash,
        cursor=dict(next_logical_step=step + 1, next_slot=0, next_sample_index=0),
        diagnostics_hash=diagnostics_hash,
        token_path_hash=token_path_hash,
        training_recipe=dict(
            learning_rate=rate,
            microbatch_size=microbatch_size,
            tis_cap=2.0,
            kl_coefficient=0.02,
            epochs_per_rollout=1,
        ),
    )


def commit_checkpoint(directory, state):
    if set(state) != CHECKPOINT_FIELDS or state["plan_id"] != PLAN_ID:
        raise ValueError("Not an SR-F1.2 checkpoint")
    return _commit_checkpoint(directory, state)


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
    microbatch_size,
    learning_rate,
    step=None,
):
    import torch

    directory = Path(directory).resolve()
    receipt = read_json(directory / ("LATEST.json" if step is None else f"commit-{step:02d}.json"))
    path = (directory / receipt["path"]).resolve()
    if path.parent != directory or file_hash(path) != receipt["sha256"]:
        raise PermissionError("Checkpoint path or bytes changed")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if set(state) != CHECKPOINT_FIELDS or state["plan_id"] != PLAN_ID:
        raise PermissionError("Checkpoint schema/plan differs")
    if state_hash(state) != receipt["state_hash"] or receipt["field_hashes"] != {
        key: state_hash(value) for key, value in state.items()
    }:
        raise PermissionError("Checkpoint content/field hashes differ")
    if (
        state["run_identity"] != run_identity
        or state["input_stream_hash"] != stream_hash
        or (state["sampling_hash"] != sampling_hash)
    ):
        raise PermissionError("Checkpoint run, input stream, or sampling differs")
    if state["reference_hash"] != state_hash(reference) or state_hash(
        state["reference"]
    ) != state_hash(reference):
        raise PermissionError("Checkpoint reference differs")
    expected_recipe = dict(
        learning_rate=learning_rate,
        microbatch_size=microbatch_size,
        tis_cap=2.0,
        kl_coefficient=0.02,
        epochs_per_rollout=1,
    )
    if state["training_recipe"] != expected_recipe:
        raise PermissionError("Checkpoint training recipe differs")
    if state["committed_logical_step"] != receipt["step"] or state["cursor"] != dict(
        next_logical_step=receipt["step"] + 1, next_slot=0, next_sample_index=0
    ):
        raise PermissionError("Checkpoint committed step/cursor differs")
    _validate_lr(state["optimizer"], state["scheduler"], learning_rate)
    if not _finite_tensor_tree(state["optimizer"]):
        raise PermissionError("Checkpoint optimizer contains nonfinite state")
    if not state["parameters"] or any(
        not bool(torch.isfinite(v).all()) for v in state["parameters"].values()
    ):
        raise PermissionError("Checkpoint policy is empty or nonfinite")
    copy_parameters(runtime.model, state["parameters"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    actual = {
        "parameters": trainable_state(runtime.model),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng(),
    }
    for name, value in actual.items():
        if state_hash(value) != state_hash(state[name]):
            raise RuntimeError(f"Restored {name} hash differs from committed state")
    return state
