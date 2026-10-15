"""Frozen SR-F3-CE single-update TIS plus independently reduced field CE.

This module owns no sampling, data selection, or scientific stopping decisions.
Each call consumes exactly sixteen cached groups of eight real student samples.
The old SR-F1.2 objective is reused, but none of its learning-rate gates are.
"""

from __future__ import annotations

import copy
import gc
import math
import time
from contextlib import contextmanager

from mm_core.training import _cpu_tree, capture_rng, restore_rng, state_hash, trainable_state
from sr_f1.runtime import copy_parameters
from sr_f12.training import (
    ProbabilityMismatch as ProbabilityMismatch,
)
from sr_f12.training import (
    enforce_probability_gate,
    probability_diagnostics,
)
from sr_f12.training import (
    sequence_objective as sequence_objective,
)

PLAN_ID = "SR-F3-CE-20261016"
ARMS = ("RL", "GOLD", "PREFIX", "EXEC")
BATCH_SIZE = 128
SUPPORT_SIZE = 4
LEARNING_RATE = 1e-5
KL_COEFFICIENT = 0.02
AUX_GAMMA = 0.1
DIAGNOSTIC_STEPS = frozenset((1, 16, 32, 64))
RECIPE = {
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "clip_grad_norm": 1.0,
    "scheduler": "constant",
    "warmup": 0,
    "prompts_per_update": 16,
    "samples_per_prompt": 8,
    "support_examples_per_update": SUPPORT_SIZE,
    "task_reduction": "mean_of_128_sequence_token_means",
    "aux_gamma": AUX_GAMMA,
    "exec_aux_weights": {"answer": 0.5, "work": 0.5},
    "tis_cap": 2.0,
    "clip_epsilon": 0.2,
    "kl_coefficient": KL_COEFFICIENT,
    "epochs_per_rollout": 1,
    "fresh_optimizer": True,
}


def _constant_schedule(_step):
    return 1.0


def fresh_optimizer(model):
    """Create fresh, empty Adam state; never accept a parent's Adam history."""
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parent adapter parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _constant_schedule)
    validate_optimizer(optimizer, scheduler, model=model)
    return optimizer, scheduler


def validate_optimizer_state(optimizer_state, scheduler_state):
    """Validate the new constant recipe without invoking an inherited LR gate."""
    groups = optimizer_state.get("param_groups", [])
    if not groups:
        raise PermissionError("AdamW has no parameter groups")
    for group in groups:
        expected = {
            "lr": LEARNING_RATE,
            "initial_lr": LEARNING_RATE,
            "eps": 1e-8,
            "weight_decay": 0.0,
        }
        if any(group.get(key) != value for key, value in expected.items()):
            raise PermissionError("Frozen SR-F3-CE optimizer recipe differs")
        if tuple(group.get("betas", ())) != (0.9, 0.999):
            raise PermissionError("Frozen AdamW betas differ")
        if any(group.get(key, False) for key in ("amsgrad", "maximize", "differentiable")):
            raise PermissionError("Unregistered AdamW mode")
    rates = [LEARNING_RATE] * len(groups)
    if scheduler_state.get("base_lrs") != rates or scheduler_state.get("_last_lr") != rates:
        raise PermissionError("Frozen constant scheduler differs")


def validate_optimizer(optimizer, scheduler, *, model=None):
    import torch

    if not isinstance(optimizer, torch.optim.AdamW):
        raise PermissionError("The registered optimizer is AdamW")
    if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        raise PermissionError("The registered scheduler is constant LambdaLR")
    if scheduler.optimizer is not optimizer:
        raise PermissionError("Scheduler is attached to another optimizer")
    if any(schedule is not _constant_schedule for schedule in scheduler.lr_lambdas):
        raise PermissionError("Use fresh_optimizer's registered constant scheduler")
    validate_optimizer_state(optimizer.state_dict(), scheduler.state_dict())
    if model is not None:
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise PermissionError("AdamW must own exactly all trainable adapter parameters")


def finite_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def group_advantages(rewards):
    """Population standard deviation, including all constant reward groups."""
    import numpy as np

    values = np.asarray(rewards, dtype=np.float64)
    if values.shape != (16, 8) or not np.isfinite(values).all():
        raise ValueError("Exactly 16 x 8 finite binary verified J rewards are required")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Only binary verified joint rewards are registered")
    std = values.std(axis=1, ddof=0, keepdims=True)
    return np.divide(
        values - values.mean(axis=1, keepdims=True),
        std + 1e-8,
        out=np.zeros_like(values),
        where=std > 0,
    )


def _qid(record):
    row = record.get("row", {})
    identity = record.get("qid", row.get("qid", row.get("question_id")))
    if not isinstance(identity, str) or not identity:
        raise ValueError("Every cached record needs a nonempty qid")
    return identity


def _task_batches(records, microbatch_size):
    if microbatch_size not in (4, 2, 1):
        raise ValueError("Registered microbatch sizes are 4, 2, 1")
    if len(records) != BATCH_SIZE:
        raise ValueError("Exactly 128 real on-policy records are required")
    qids = []
    for start in range(0, BATCH_SIZE, 8):
        group = records[start : start + 8]
        qid = _qid(group[0])
        if any(_qid(record) != qid for record in group):
            raise ValueError("Each consecutive group of eight must share one qid")
        qids.append(qid)
        for record in group:
            tokens, sampler = record.get("tokens", []), record.get("sampler_logprobs", [])
            if not 1 <= len(tokens) <= 1024 or len(tokens) != len(sampler):
                raise ValueError("Sampler logprobs must align with actual 1..1024 output tokens")
            if hasattr(sampler, "requires_grad") and sampler.requires_grad:
                raise ValueError("Sampler logprobs must be detached")
        for offset in range(0, 8, microbatch_size):
            yield start + offset, group[offset : offset + microbatch_size]
    if len(set(qids)) != 16:
        raise ValueError("Sixteen distinct task qids are required per update")


def _mask(record, name, *, required):
    import torch

    raw = record.get(name, [False] * len(record["tokens"]))
    mask = torch.as_tensor(raw)
    if mask.ndim != 1 or mask.numel() != len(record["tokens"]):
        raise ValueError(f"{name} must align with actual completion tokens")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(f"{name} is not a binary token mask")
    mask = mask.bool()
    if required and not bool(mask.any()):
        raise ValueError(f"{name} cannot be empty")
    return mask


def _validate_support(arm, records):
    if arm == "RL":
        if records is not None:
            raise PermissionError("RL has no support sequences, forward pass, labels, or KL")
        return []
    if records is None or len(records) != SUPPORT_SIZE:
        raise ValueError("Exactly four fixed support examples are required")
    if len({_qid(record) for record in records}) != SUPPORT_SIZE:
        raise ValueError("The four support qids must be distinct")
    masks = []
    for record in records:
        if not record.get("tokens"):
            raise ValueError("Support target tokens are empty")
        answer = _mask(record, "answer_mask", required=True)
        work = _mask(record, "work_mask", required=arm in ("PREFIX", "EXEC"))
        if bool((answer & work).any()):
            raise ValueError("Answer and work masks overlap")
        if arm == "GOLD" and bool(work.any()):
            raise ValueError("GOLD cannot contain a supervised work field")
        masks.append((answer, work))
    return masks


def masked_mean_nll(logprobs, mask):
    import torch

    mask = torch.as_tensor(mask, device=logprobs.device, dtype=torch.bool)
    if logprobs.ndim != 1 or logprobs.shape != mask.shape or not bool(mask.any()):
        raise ValueError("A nonempty aligned field mask is required")
    if not logprobs.requires_grad or not bool(torch.isfinite(logprobs).all()):
        raise FloatingPointError("Field CE requires finite differentiable logprobs")
    return -logprobs.float()[mask].mean()


def _gradient_norm(parameters):
    return math.sqrt(
        sum(
            float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None
        )
    )


def _component_gradient(loss, parameters, accumulator):
    import torch

    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    for index, gradient in enumerate(gradients):
        if gradient is not None:
            value = gradient.detach().to(device="cpu", dtype=torch.float64)
            if accumulator[index] is None:
                accumulator[index] = value.clone()
            else:
                accumulator[index].add_(value)


@contextmanager
def _timed_cost(costs, key):
    started = time.perf_counter()
    try:
        yield
    finally:
        costs[key] += time.perf_counter() - started


def _new_costs():
    return dict.fromkeys(
        (
            "task_reference_sequences_attempted",
            "task_reference_tokens_attempted",
            "task_policy_sequences_attempted",
            "task_policy_tokens_attempted",
            "support_sequences_attempted",
            "support_forward_tokens_attempted",
            "diagnostic_backward_calls_attempted",
            "diagnostic_gradient_tokens_attempted",
            "diagnostic_backward_host_wall_seconds",
            "update_backward_calls_attempted",
            "update_gradient_tokens_attempted",
            "update_backward_host_wall_seconds",
            "global_clip_calls",
            "optimizer_step_calls_attempted",
        ),
        0,
    )


def _gradient_diagnostics(components, gamma):
    norms = {
        name: math.sqrt(sum(float(g.square().sum()) for g in gradients if g is not None))
        for name, gradients in components.items()
    }
    angles = {}
    for left, right in (("policy", "kl"), ("policy", "aux_ce"), ("kl", "aux_ce")):
        if left not in components or right not in components:
            continue
        denominator = norms[left] * norms[right]
        dot = sum(
            float((a * b).sum())
            for a, b in zip(components[left], components[right], strict=True)
            if a is not None and b is not None
        )
        angles[f"{left}_vs_{right}"] = {
            "degrees": math.degrees(math.acos(max(-1.0, min(1.0, dot / denominator))))
            if denominator > 0
            else None,
            "status": "DEFINED" if denominator > 0 else "UNDEFINED_ZERO_NORM",
        }
    weights = {"policy": 1.0, "kl": KL_COEFFICIENT, "aux_ce": gamma}
    return {
        "unweighted_norms": norms,
        "weighted_norms": {name: weights[name] * norm for name, norm in norms.items()},
        "angles": angles,
        "method": "autograd.grad_on_same_graph_without_mutating_accumulated_update_gradient",
    }


def _snapshot(runtime, optimizer, scheduler):
    return {
        "parameters": trainable_state(runtime.model),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": capture_rng(),
        "train_mode": runtime.model.training,
    }


def _restore(runtime, optimizer, scheduler, state):
    copy_parameters(runtime.model, state["parameters"])
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    optimizer.zero_grad(set_to_none=True)
    restore_rng(state["rng"])
    runtime.model.train(state["train_mode"])
    for key, actual in (
        ("parameters", trainable_state(runtime.model)),
        ("optimizer", _cpu_tree(optimizer.state_dict())),
        ("scheduler", scheduler.state_dict()),
        ("rng", capture_rng()),
    ):
        if state_hash(actual) != state_hash(state[key]):
            raise RuntimeError(f"Whole-step rollback did not restore {key}")


def _attempt(
    runtime,
    optimizer,
    scheduler,
    records,
    advantages,
    *,
    arm,
    reference,
    support_records,
    support_masks,
    microbatch_size,
    step,
    gamma,
    save_train_logprobs,
    costs,
    apply_update=True,
    optimizer_step_callback=None,
):
    import torch

    parameters = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
    before = trainable_state(runtime.model)
    optimizer.zero_grad(set_to_none=True)
    totals = dict(policy=0.0, kl=0.0, aux_ce=0.0, answer_ce=0.0, work_ce=None)
    if arm in ("PREFIX", "EXEC"):
        totals["work_ce"] = 0.0
    differences, truncated, train_values, support_values = [], [], [], []
    diagnose = step in DIAGNOSTIC_STEPS
    components = {name: [None] * len(parameters) for name in ("policy", "kl")}
    if arm != "RL":
        components["aux_ce"] = [None] * len(parameters)
    for offset, batch in _task_batches(records, microbatch_size):
        tokens = sum(len(record["tokens"]) for record in batch)
        costs["task_reference_sequences_attempted"] += len(batch)
        costs["task_reference_tokens_attempted"] += tokens
        with torch.no_grad():
            references = runtime.batch_reference_forward(batch, reference)
        costs["task_policy_sequences_attempted"] += len(batch)
        costs["task_policy_tokens_attempted"] += tokens
        currents = runtime.batch_sequence_forward(batch, purpose="training_policy", grad=True)
        if len(currents) != len(batch) or len(references) != len(batch):
            raise ValueError("Runtime returned the wrong number of task sequences")
        policy_terms, kl_terms = [], []
        for index, (record, current, ref) in enumerate(
            zip(batch, currents, references, strict=True)
        ):
            if current.numel() != len(record["tokens"]):
                raise ValueError("Runtime logprobs do not align with actual generated tokens")
            sampler = torch.as_tensor(
                record["sampler_logprobs"], device=current.device, dtype=torch.float32
            )
            result = sequence_objective(current, sampler, ref, advantages[offset + index])
            policy_terms.append(result["policy"] / BATCH_SIZE)
            kl_terms.append(result["kl"] / BATCH_SIZE)
            differences.append(result["absolute_difference"].cpu())
            truncated.append(result["tis_truncated"].cpu())
            if float(result["clip_fraction"]) != 0.0:
                raise RuntimeError("Detached train-old ratio is not exactly one")
            if save_train_logprobs:
                train_values.append(current.detach().float().cpu().tolist())
        policy, kl = torch.stack(policy_terms).sum(), torch.stack(kl_terms).sum()
        totals["policy"] += float(policy.detach())
        totals["kl"] += float(kl.detach())
        if diagnose:
            for name, loss in (("policy", policy), ("kl", kl)):
                costs["diagnostic_backward_calls_attempted"] += 1
                costs["diagnostic_gradient_tokens_attempted"] += tokens
                with _timed_cost(costs, "diagnostic_backward_host_wall_seconds"):
                    _component_gradient(loss, parameters, components[name])
        costs["update_backward_calls_attempted"] += 1
        costs["update_gradient_tokens_attempted"] += tokens
        with _timed_cost(costs, "update_backward_host_wall_seconds"):
            (policy + KL_COEFFICIENT * kl).backward()
        del currents, references, policy_terms, kl_terms, policy, kl, result
    probabilities = probability_diagnostics(differences, truncated)
    enforce_probability_gate(probabilities)
    if arm != "RL":
        for start in range(0, SUPPORT_SIZE, microbatch_size):
            batch = support_records[start : start + microbatch_size]
            tokens = sum(len(record["tokens"]) for record in batch)
            costs["support_sequences_attempted"] += len(batch)
            costs["support_forward_tokens_attempted"] += tokens
            currents = runtime.batch_support_forward(batch, purpose="training_support", grad=True)
            if len(currents) != len(batch):
                raise ValueError("Runtime returned the wrong number of support sequences")
            terms = []
            supervised_tokens = 0
            for index, (record, current) in enumerate(zip(batch, currents, strict=True)):
                answer_mask, work_mask = support_masks[start + index]
                answer = masked_mean_nll(current, answer_mask)
                work = masked_mean_nll(current, work_mask) if arm in ("PREFIX", "EXEC") else None
                auxiliary = 0.5 * answer + 0.5 * work if arm == "EXEC" else answer
                terms.append(auxiliary / SUPPORT_SIZE)
                totals["answer_ce"] += float(answer.detach()) / SUPPORT_SIZE
                if work is not None:
                    totals["work_ce"] += float(work.detach()) / SUPPORT_SIZE
                supervised_tokens += int(answer_mask.sum())
                if arm == "EXEC":
                    supervised_tokens += int(work_mask.sum())
                support_values.append(
                    {
                        "qid": _qid(record),
                        "answer_ce": float(answer.detach()),
                        "work_ce": float(work.detach()) if work is not None else None,
                        "answer_tokens": int(answer_mask.sum()),
                        "work_tokens": int(work_mask.sum()),
                        "supervised_work_tokens": int(work_mask.sum()) if arm == "EXEC" else 0,
                    }
                )
            auxiliary = torch.stack(terms).sum()
            totals["aux_ce"] += float(auxiliary.detach())
            if diagnose:
                costs["diagnostic_backward_calls_attempted"] += 1
                costs["diagnostic_gradient_tokens_attempted"] += supervised_tokens
                with _timed_cost(costs, "diagnostic_backward_host_wall_seconds"):
                    _component_gradient(auxiliary, parameters, components["aux_ce"])
            costs["update_backward_calls_attempted"] += 1
            costs["update_gradient_tokens_attempted"] += supervised_tokens
            with _timed_cost(costs, "update_backward_host_wall_seconds"):
                (gamma * auxiliary).backward()
            del currents, terms, auxiliary, answer, work
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError("Nonfinite accumulated gradient; Adam step withheld")
    gradients = None
    if apply_update:
        norm_before = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        costs["global_clip_calls"] += 1
        norm_after = _gradient_norm(parameters)
        if optimizer_step_callback is not None:
            # The caller fsyncs an intent/budget reservation before any Adam
            # mutation. Unresolved intents can then count conservatively.
            optimizer_step_callback(
                {
                    "step": step,
                    "arm": arm,
                    "microbatch_size": microbatch_size,
                    "policy_hash": state_hash(before),
                }
            )
        costs["optimizer_step_calls_attempted"] += 1
        optimizer.step()
        scheduler.step()
    else:
        norm_before = norm_after = _gradient_norm(parameters)
        gradients = {
            name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
            for name, parameter in runtime.model.named_parameters()
            if parameter.requires_grad
        }
    if not finite_tree(optimizer.state_dict()) or not finite_tree(trainable_state(runtime.model)):
        raise FloatingPointError("Nonfinite state after Adam; entire update must roll back")
    validate_optimizer(optimizer, scheduler, model=runtime.model)
    after = trainable_state(runtime.model)
    max_change = max(float((after[name] - tensor).abs().max()) for name, tensor in before.items())
    l2_change = math.sqrt(
        sum(
            float((after[name] - tensor).double().square().sum()) for name, tensor in before.items()
        )
    )
    result = {
        **totals,
        **probabilities,
        "task_loss": totals["policy"] + KL_COEFFICIENT * totals["kl"],
        "policy_weighted": totals["policy"],
        "kl_weighted": KL_COEFFICIENT * totals["kl"],
        "aux_ce_weighted": gamma * totals["aux_ce"],
        "loss": totals["policy"] + KL_COEFFICIENT * totals["kl"] + gamma * totals["aux_ce"],
        "gradient_norm_before_clip": float(norm_before),
        "gradient_norm_after_clip": norm_after,
        "parameter_max_absolute_change": max_change,
        "parameter_l2_change": l2_change,
        "parameters_changed": max_change > 0,
        "optimizer_steps": sorted({int(value["step"]) for value in optimizer.state.values()}),
        "optimizer_updates": int(apply_update),
        "sequence_count": BATCH_SIZE,
        "support_sequence_count": 0 if arm == "RL" else SUPPORT_SIZE,
        "support_losses": support_values,
        "microbatch_size": microbatch_size,
        "finite": True,
        "gradient_diagnostics": _gradient_diagnostics(components, gamma) if diagnose else None,
    }
    if save_train_logprobs:
        result["train_logprobs"] = train_values
    if gradients is not None:
        result["unclipped_gradients"] = gradients
        result["unclipped_gradient_hash"] = state_hash(gradients)
    return result


def _validate_update_inputs(
    runtime,
    optimizer,
    scheduler,
    records,
    *,
    arm,
    support_records,
    step,
    microbatch_size,
    engineering_zero_aux,
):
    import numpy as np

    if arm not in ARMS or type(step) is not int or not 1 <= step <= 64:
        raise ValueError("Unknown arm or logical step outside 1..64")
    if engineering_zero_aux and (arm == "RL" or step > 4):
        raise PermissionError("The zero-CE check is only for the first four ENGINE updates")
    list(_task_batches(records, microbatch_size))
    masks = _validate_support(arm, support_records)
    for record in records:
        if "J" in record.get("score", {}) and record["reward"] != record["score"]["J"]:
            raise ValueError("Cached reward differs from the retained verified J score")
    rewards = np.asarray([record["reward"] for record in records], dtype=np.float64).reshape(16, 8)
    advantages = group_advantages(rewards).reshape(-1)
    validate_optimizer(optimizer, scheduler, model=runtime.model)
    if scheduler.state_dict().get("last_epoch") != step - 1:
        raise PermissionError("Logical step does not follow the current optimizer/scheduler cursor")
    adam_steps = {int(value["step"]) for value in optimizer.state.values()}
    if (step == 1 and adam_steps) or (step > 1 and adam_steps != {step - 1}):
        raise PermissionError("Adam state does not precede this logical update")
    return masks, rewards, advantages


def probe_gradients(
    runtime,
    optimizer,
    scheduler,
    records,
    *,
    arm,
    reference,
    support_records=None,
    step=1,
    microbatch_size=4,
    engineering_zero_aux=False,
):
    """ENGINE forward/backward probe with **zero** clip/Adam/scheduler updates.

    Returns CPU ``unclipped_gradients`` plus their hash, loss and measured costs.
    This permits CE-times-zero and microbatch comparisons using cached samples
    without exceeding the four-plus-two physical update budget. The entire live
    state is restored, even if the probe fails; this is never a science step.
    """
    masks, _rewards, advantages = _validate_update_inputs(
        runtime,
        optimizer,
        scheduler,
        records,
        arm=arm,
        support_records=support_records,
        step=step,
        microbatch_size=microbatch_size,
        engineering_zero_aux=engineering_zero_aux,
    )
    if step > 4:
        raise PermissionError("Standalone gradient probes are limited to ENGINE steps 1..4")
    before = _snapshot(runtime, optimizer, scheduler)
    reference_hash = state_hash(reference)
    costs = _new_costs()
    started = time.perf_counter()
    try:
        result = _attempt(
            runtime,
            optimizer,
            scheduler,
            records,
            advantages,
            arm=arm,
            reference=reference,
            support_records=support_records,
            support_masks=masks,
            microbatch_size=microbatch_size,
            step=step,
            gamma=0.0 if engineering_zero_aux else AUX_GAMMA,
            save_train_logprobs=False,
            costs=costs,
            apply_update=False,
        )
        if state_hash(reference) != reference_hash:
            raise RuntimeError("Fixed parent reference was mutated during gradient probe")
        result.update(
            plan_id=PLAN_ID,
            stage="ENGINE_GRADIENT_PROBE",
            arm=arm,
            step=step,
            scientific_recipe_eligible=False,
            total_cost=costs,
            training_seconds=time.perf_counter() - started,
        )
        return result
    except BaseException as exc:
        exc.training_probe_cost = copy.deepcopy(costs)
        raise
    finally:
        _restore(runtime, optimizer, scheduler, before)


def update(
    runtime,
    optimizer,
    scheduler,
    records,
    *,
    arm,
    reference,
    support_records=None,
    step,
    microbatch_size=4,
    save_train_logprobs=False,
    incident_callback=None,
    engineering_zero_aux=False,
    optimizer_step_callback=None,
):
    """Update once at fixed theta; CUDA OOM retries restart the *entire* update.

    Runtime exposes the historical task/reference batch methods and
    ``batch_support_forward(records, purpose=..., grad=True)``. Every forward
    returns an unpadded differentiable logprob vector per sequence. Cached raw
    records remain caller-owned; ``incident_callback`` durably records failures.
    Exceptions also carry ``training_attempts`` with cost and rollback evidence.

    The explicit ENGINE-only zero-CE check is never a scientific recipe. The
    caller must retain its receipt and may not use its adapter for science.
    """
    import torch

    masks, rewards, advantages = _validate_update_inputs(
        runtime,
        optimizer,
        scheduler,
        records,
        arm=arm,
        support_records=support_records,
        step=step,
        microbatch_size=microbatch_size,
        engineering_zero_aux=engineering_zero_aux,
    )
    original = _snapshot(runtime, optimizer, scheduler)
    reference_hash = state_hash(reference)
    gamma = 0.0 if engineering_zero_aux else AUX_GAMMA
    attempts = []
    started = time.perf_counter()
    sizes = (4, 2, 1)[(4, 2, 1).index(microbatch_size) :]
    for size in sizes:
        costs = _new_costs()
        attempt_started = time.perf_counter()
        try:
            result = _attempt(
                runtime,
                optimizer,
                scheduler,
                records,
                advantages,
                arm=arm,
                reference=reference,
                support_records=support_records,
                support_masks=masks,
                microbatch_size=size,
                step=step,
                gamma=gamma,
                save_train_logprobs=save_train_logprobs,
                costs=costs,
                optimizer_step_callback=optimizer_step_callback,
            )
            if state_hash(reference) != reference_hash:
                raise RuntimeError("Fixed parent reference was mutated")
            # Teacher forcing and its diagnostics do not advance sampling RNG.
            restore_rng(original["rng"])
            runtime.model.train(original["train_mode"])
            attempt = {
                "microbatch_size": size,
                "status": "APPLIED",
                "cost": costs,
                "wall_seconds": time.perf_counter() - attempt_started,
            }
            attempts.append(attempt)
            result.update(
                plan_id=PLAN_ID,
                arm=arm,
                step=step,
                attempts=attempts,
                total_cost={key: sum(a["cost"][key] for a in attempts) for key in costs},
                training_seconds=time.perf_counter() - started,
                teacher_forcing_rng_unchanged=state_hash(capture_rng())
                == state_hash(original["rng"]),
                reference_unchanged=True,
                engineering_zero_aux=engineering_zero_aux,
                scientific_recipe_eligible=not engineering_zero_aux,
                reward_groups=[
                    {
                        "qid": _qid(records[index * 8]),
                        "J": row.tolist(),
                        "mean": float(row.mean()),
                        "population_std": float(row.std(ddof=0)),
                        "advantages": advantages[index * 8 : (index + 1) * 8].tolist(),
                        "constant": bool(row.std(ddof=0) == 0),
                    }
                    for index, row in enumerate(rewards)
                ],
                constant_group_fraction=float((rewards.std(axis=1) == 0).mean()),
            )
            return result
        except BaseException as exc:
            _restore(runtime, optimizer, scheduler, original)
            oom = isinstance(exc, torch.cuda.OutOfMemoryError)
            attempt = {
                "microbatch_size": size,
                "status": "CUDA_OOM_ROLLED_BACK" if oom else "FAILED_ROLLED_BACK",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "cost": costs,
                "wall_seconds": time.perf_counter() - attempt_started,
                "parameters_optimizer_scheduler_rng_restored": True,
                "probability_diagnostics": getattr(exc, "diagnostics", None),
            }
            attempts.append(attempt)
            exc.training_attempts = copy.deepcopy(attempts)
            if incident_callback is not None:
                incident_callback(copy.deepcopy(attempt))
            if not oom or size == 1:
                raise
            # Never swallow an OOM at one sequence or enable checkpointing here:
            # that backend change requires the common cross-arm ENGINE gate.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise AssertionError("Registered microbatch loop exhausted unexpectedly")
