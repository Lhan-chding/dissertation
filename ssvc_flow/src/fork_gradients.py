"""R3 fixed-bank LoRA gradients and one isolated, real optimizer update.

The caller owns model/state restoration and certifies the trainable LoRA mask.
Gradient producers clear scratch gradients and restore the model's training mode;
they never step an optimizer, sample, rewrite old probabilities, or alter rewards.
Only CPU FP32 trainable-parameter tensors leave the gradient producers. Neither
score collection nor numerical comparison alone authorizes gradient reuse.
"""

from __future__ import annotations

import math

from .grpo_update import grouped_advantages, reward_channels, torch_ppo_loss
from .optimizer_fork import parameter_hash, state_hash

CATEGORIES = ("X", "S", "W", "I")
SCORE_CLIP_EPSILON = 0.2


def _number(value, name, *, positive=False):
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _parameters(model):
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not parameters:
        raise ValueError("No trainable LoRA parameters")
    if any(not p.is_floating_point() for p in parameters.values()):
        raise ValueError("Trainable parameters must be floating point")
    return parameters


def _old_tensor(row, device="cpu"):
    import torch

    return torch.as_tensor(row["old_logprobs"]).detach().to(device=device, dtype=torch.float32)


def _bank_binding(groups):
    import torch

    if not groups or not groups[0] or len({len(g) for g in groups}) != 1:
        raise ValueError("Nonempty equal-K prompt groups required")
    prompts, keys, records, prepared_cache = set(), set(), [], {}
    for group in groups:
        prompt = group[0].get("prompt_id")
        if not isinstance(prompt, str) or not prompt or prompt in prompts:
            raise ValueError("Each bank group requires a distinct nonempty prompt_id")
        prompts.add(prompt)
        prompt_input_hash = None
        for row in group:
            if row.get("prompt_id") != prompt or row.get("category") not in CATEGORIES:
                raise ValueError("Mixed prompt or unknown category in group")
            key = row.get("sample_key")
            if not isinstance(key, str) or not key or key in keys:
                raise ValueError("Unique nonempty sample_key required")
            keys.add(key)
            tokens = row["token_ids"]
            if not tokens or any(type(t) is not int or t < 0 for t in tokens):
                raise ValueError("Nonempty generated token IDs required")
            old = _old_tensor(row)
            if old.ndim != 1 or old.numel() != len(tokens) or not torch.isfinite(old).all():
                raise ValueError("Finite old token probabilities must match action length")
            prepared = row["prepared"]
            identity = id(prepared)
            if identity not in prepared_cache:
                prepared_cache[identity] = state_hash(prepared)
            input_hash = prepared_cache[identity]
            if prompt_input_hash is not None and input_hash != prompt_input_hash:
                raise ValueError("Same prompt has different prepared model inputs")
            prompt_input_hash = input_hash
            records.append(
                {
                    "prompt_id": prompt,
                    "sample_key": key,
                    "category": row["category"],
                    "token_ids": list(tokens),
                    "old_logprobs": old.tolist(),
                    "prepared_hash": input_hash,
                }
            )
    return {
        "bank_hash": state_hash(records),
        "B": len(groups),
        "K": len(groups[0]),
        "sequences": len(records),
        "total_tokens": sum(len(r["token_ids"]) for r in records),
    }


def _rewards(groups, arm, auxiliary_weight, epsilon):
    lam = _number(auxiliary_weight, "auxiliary_weight")
    epsilon = _number(epsilon, "epsilon")
    statistics = []
    for group in groups:
        values = [reward_channels(row["category"], arm, lam)["sum"] for row in group]
        statistics.append(
            {
                "prompt_id": group[0]["prompt_id"],
                "reward_vector": values,
                "category_counts": {c: sum(r["category"] == c for r in group) for c in CATEGORIES},
                **grouped_advantages(values, epsilon),
            }
        )
    return lam, epsilon, statistics


def _ratio_record(new, old, row, clip_epsilon):
    import torch

    if new.ndim != 1 or new.shape != old.shape or not new.numel():
        raise ValueError("Scored token/action masks or old probabilities differ")
    delta = new.detach().float() - old
    ratio = delta.exp()
    if not torch.isfinite(delta).all() or not torch.isfinite(ratio).all():
        raise FloatingPointError("Nonfinite token log-ratio or importance ratio")
    return {
        "sample_key": row["sample_key"],
        "prompt_id": row["prompt_id"],
        "category": row["category"],
        "tokens": new.numel(),
        "ratio_one_exact": bool((delta == 0).all()),
        "max_abs_token_log_ratio": float(delta.abs().max()),
        "sum_abs_token_log_ratio": float(delta.abs().double().sum()),
        "sequence_log_ratio": float(delta.double().sum()),
        "max_abs_ratio_minus_one": float((ratio - 1).abs().max()),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "tokens_below_clip": int((ratio < 1 - clip_epsilon).sum()),
        "tokens_above_clip": int((ratio > 1 + clip_epsilon).sum()),
    }


def _ratio_audit(records, clip_epsilon, advantages=None):
    tokens = sum(r["tokens"] for r in records)
    exact = all(r["ratio_one_exact"] for r in records)
    outside = sum(r["tokens_below_clip"] + r["tokens_above_clip"] for r in records)
    active = None
    if advantages is not None:
        active = sum(
            r["tokens_above_clip"] if a > 0 else r["tokens_below_clip"] if a < 0 else 0
            for r, a in zip(records, advantages, strict=True)
        )
    return {
        "ratio_one_exact": exact,
        "max_abs_token_log_ratio": max(r["max_abs_token_log_ratio"] for r in records),
        "mean_abs_token_log_ratio": math.fsum(r["sum_abs_token_log_ratio"] for r in records)
        / tokens,
        "max_abs_sequence_log_ratio": max(abs(r["sequence_log_ratio"]) for r in records),
        "max_abs_ratio_minus_one": max(r["max_abs_ratio_minus_one"] for r in records),
        "ratio_min": min(r["ratio_min"] for r in records),
        "ratio_max": max(r["ratio_max"] for r in records),
        "clip_epsilon": clip_epsilon,
        "tokens_outside_clip_range": outside,
        "clip_fraction": outside / tokens,
        "surrogate_clipped_tokens": active,
        "surrogate_clipping_active": None if active is None else active > 0,
        "cannot_adopt_reuse": not exact or outside > 0,
        "requires_direct_gradient_validation": True,
        "reuse_scope": "FIXED_THETA_RATIO_ONE_SINGLE_STEP_NO_ADDITIONAL_REGULARIZER",
        "ratio_validation": "EXACT_FP32_OLD_NEW_EQUALITY; no unregistered ratio tolerance applied",
        "reuse_status": "REQUIRES_DIRECT_GRADIENT_VALIDATION"
        if exact and not outside
        else "CANNOT_ADOPT_REUSE",
    }


def _copy_gradients(parameters):
    import torch

    if not any(p.grad is not None for p in parameters.values()):
        raise RuntimeError("No trainable gradient graph")
    result = {}
    for name, p in parameters.items():
        value = (
            p.grad.detach().float().cpu().clone()
            if p.grad is not None
            else torch.zeros_like(p, device="cpu", dtype=torch.float32)
        )
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite gradient: {name}")
        result[name] = value
    return result


def _norm(gradients):
    return math.sqrt(math.fsum(float(g.double().square().sum()) for g in gradients.values()))


def direct_loss_gradients(
    adapter, groups, *, arm="X_BASE", auxiliary_weight=0, lnorm=64, epsilon=1e-4, clip_epsilon=0.2
):
    """Unclipped parameter gradients of the real clipped PPO loss; no Adam step."""
    import torch

    binding = _bank_binding(groups)
    lnorm = _number(lnorm, "lnorm", positive=True)
    clip_epsilon = _number(clip_epsilon, "clip_epsilon", positive=True)
    if clip_epsilon >= 1:
        raise ValueError("clip_epsilon must be below one")
    lam, epsilon, statistics = _rewards(groups, arm, auxiliary_weight, epsilon)
    parameters = _parameters(adapter.model)
    origin = parameter_hash(adapter.model, trainable=True)
    training = adapter.model.training
    records, losses, advantages = [], [], []
    backward_calls = 0
    try:
        adapter.model.zero_grad(set_to_none=True)
        for group, stats in zip(groups, statistics, strict=True):
            for row, advantage in zip(group, stats["advantages"], strict=True):
                new = adapter.logprobs(row["prepared"], row["token_ids"], require_grad=True)
                old = _old_tensor(row, new.device)
                records.append(_ratio_record(new, old, row, clip_epsilon))
                loss, _ = torch_ppo_loss(
                    new[None],
                    old[None],
                    torch.tensor([advantage], device=new.device),
                    torch.ones_like(new[None], dtype=torch.bool),
                    lnorm=lnorm,
                    clip_epsilon=clip_epsilon,
                    total_sequences=binding["sequences"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite direct loss")
                loss.backward()
                backward_calls += 1
                losses.append(float(loss.detach()))
                advantages.append(advantage)
        if any(p.grad is not None for p in adapter.model.parameters() if not p.requires_grad):
            raise RuntimeError("Frozen parameter received a gradient")
        gradients = _copy_gradients(parameters)
        if parameter_hash(adapter.model, trainable=True) != origin:
            raise RuntimeError("Parameters changed during gradient computation")
        return {
            "gradients": gradients,
            "audit": {
                **binding,
                **_ratio_audit(records, clip_epsilon, advantages),
                "parameter_hash": origin,
                "arm": arm,
                "auxiliary_weight": lam,
                "epsilon": epsilon,
                "Lnorm": lnorm,
                "gradient_stage": "PRE_CLIP",
                "loss": math.fsum(losses),
                "grad_norm": _norm(gradients),
                "gradient_hash": state_hash(gradients),
                "backward_calls": backward_calls,
                "optimizer_updates": 0,
                "microbatch": 1,
                "group_statistics": statistics,
                "ratio_records": records,
                "old_probabilities_rewritten": False,
                "count_scope": (
                    "Observed successful loss.backward calls; "
                    "internal forwards belong to runtime hooks"
                ),
            },
        }
    finally:
        adapter.model.zero_grad(set_to_none=True)
        adapter.model.train(training)


def collect_category_scores(adapter, groups):
    """Sum sequence-score gradients in CPU FP32; retain no per-sample gradients."""
    import torch

    binding = _bank_binding(groups)
    parameters = _parameters(adapter.model)
    origin = parameter_hash(adapter.model, trainable=True)
    training = adapter.model.training
    sums, records, counts = {}, [], {}
    backward_calls = 0
    try:
        for group in groups:
            prompt = group[0]["prompt_id"]
            sums[prompt], counts[prompt] = {}, {c: 0 for c in CATEGORIES}
            for row in group:
                adapter.model.zero_grad(set_to_none=True)
                new = adapter.logprobs(row["prepared"], row["token_ids"], require_grad=True)
                records.append(
                    _ratio_record(new, _old_tensor(row, new.device), row, SCORE_CLIP_EPSILON)
                )
                new.float().sum().backward()
                backward_calls += 1
                if any(
                    p.grad is not None for p in adapter.model.parameters() if not p.requires_grad
                ):
                    raise RuntimeError("Frozen parameter received a gradient")
                # At most one transient sample's trainable gradients, discarded immediately.
                gradients = _copy_gradients(parameters)
                category = row["category"]
                if category not in sums[prompt]:
                    sums[prompt][category] = gradients
                else:
                    for name, value in gradients.items():
                        sums[prompt][category][name].add_(value)
                counts[prompt][category] += 1
                del gradients
        if parameter_hash(adapter.model, trainable=True) != origin:
            raise RuntimeError("Parameters changed during score collection")
        if any(
            not torch.isfinite(g).all()
            for cats in sums.values()
            for grads in cats.values()
            for g in grads.values()
        ):
            raise FloatingPointError("Nonfinite accumulated category score")
        binding = {**binding, "parameter_hash": origin, "score_sums_hash": state_hash(sums)}
        return {
            "score_sums": sums,
            "binding": binding,
            "audit": {
                **binding,
                **_ratio_audit(records, SCORE_CLIP_EPSILON),
                "backward_calls": backward_calls,
                "optimizer_updates": 0,
                "category_counts": counts,
                "missing_categories": {
                    p: [c for c in CATEGORIES if n[c] == 0] for p, n in counts.items()
                },
                "missing_category_interpretation": "EMPIRICAL_ZERO_CONDITIONAL_MEAN_UNESTIMABLE",
                "score_definition": (
                    "sum_sample sum_generated_token grad log pi; "
                    "includes real EOS, no per-length normalization"
                ),
                "storage": "CPU_FP32_TRAINABLE_SUBSPACE_ONLY",
                "microbatch": 1,
                "ratio_records": records,
                "old_probabilities_rewritten": False,
            },
        }
    finally:
        adapter.model.zero_grad(set_to_none=True)
        adapter.model.train(training)


def combine_category_gradients(
    score_bank, groups, *, arm, auxiliary_weight, lnorm=64, epsilon=1e-4
):
    """Combine complete joint advantages with category score sums, with loss sign."""
    import torch

    actual = _bank_binding(groups)
    binding = score_bank["binding"]
    if any(binding.get(key) != value for key, value in actual.items()):
        raise ValueError("Score-bank source binding mismatch")
    sums = score_bank["score_sums"]
    if state_hash(sums) != binding["score_sums_hash"]:
        raise ValueError("Score-bank gradient checksum mismatch")
    lnorm = _number(lnorm, "lnorm", positive=True)
    lam, epsilon, statistics = _rewards(groups, arm, auxiliary_weight, epsilon)
    template = next(iter(next(iter(sums.values())).values()))
    gradients = {name: torch.zeros_like(value) for name, value in template.items()}
    for group, stats in zip(groups, statistics, strict=True):
        prompt = group[0]["prompt_id"]
        by_category = {}
        for row, advantage in zip(group, stats["advantages"], strict=True):
            by_category[row["category"]] = advantage
        for category, advantage in by_category.items():
            coefficient = -advantage / (actual["sequences"] * lnorm)
            for name, value in sums[prompt][category].items():
                gradients[name].add_(value, alpha=coefficient)
    if any(not torch.isfinite(g).all() for g in gradients.values()):
        raise FloatingPointError("Nonfinite combined gradient")
    audit = score_bank["audit"]
    advantages = [value for stats in statistics for value in stats["advantages"]]
    return {
        "gradients": gradients,
        "audit": {
            **actual,
            **_ratio_audit(audit["ratio_records"], audit["clip_epsilon"], advantages),
            "parameter_hash": binding["parameter_hash"],
            "arm": arm,
            "auxiliary_weight": lam,
            "epsilon": epsilon,
            "Lnorm": lnorm,
            "gradient_stage": "PRE_CLIP",
            "grad_norm": _norm(gradients),
            "gradient_hash": state_hash(gradients),
            "backward_calls": 0,
            "optimizer_updates": 0,
            "group_statistics": statistics,
            "missing_categories": audit["missing_categories"],
            "missing_category_interpretation": audit["missing_category_interpretation"],
            "old_probabilities_rewritten": False,
            "score_collection_backward_calls": audit["backward_calls"],
        },
    }


def compare_gradient_bundles(direct, reused, atol=1e-6, rtol=1e-5):
    """Compare every pre-clip tensor; an all-zero comparison is not informative."""
    import torch

    atol, rtol = _number(atol, "atol"), _number(rtol, "rtol")
    left, right = direct["gradients"], reused["gradients"]
    same_keys = bool(left) and set(left) == set(right)
    per_parameter = {}
    differences, left_squares, right_squares = [], [], []
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        same_shape = a is not None and b is not None and a.shape == b.shape
        finite = same_shape and bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        if finite:
            a, b = a.detach().cpu().double(), b.detach().cpu().double()
            delta = a - b
            maximum = float(delta.abs().max()) if delta.numel() else 0.0
            difference = float(delta.square().sum())
            l2, r2 = float(a.square().sum()), float(b.square().sum())
            equal = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
            differences.append(difference)
            left_squares.append(l2)
            right_squares.append(r2)
        else:
            maximum = difference = l2 = r2 = None
            equal = False
        per_parameter[name] = {
            "same_shape": same_shape,
            "finite": finite,
            "allclose": equal,
            "max_abs_difference": maximum,
            "difference_l2": math.sqrt(difference) if finite else None,
            "direct_l2": math.sqrt(l2) if finite else None,
            "reused_l2": math.sqrt(r2) if finite else None,
        }
    finite = same_keys and all(p["finite"] for p in per_parameter.values())
    equal = finite and all(p["allclose"] for p in per_parameter.values())
    left_norm = math.sqrt(math.fsum(left_squares)) if finite else None
    right_norm = math.sqrt(math.fsum(right_squares)) if finite else None
    informative = finite and left_norm > 0 and right_norm > 0
    zero = finite and left_norm == 0 and right_norm == 0
    difference = math.sqrt(math.fsum(differences)) if finite else None
    return {
        "same_parameter_keys": same_keys,
        "finite": finite,
        "allclose": equal,
        "informative": informative,
        "zero_consistent": zero,
        "passed": bool(equal and informative),
        "atol": atol,
        "rtol": rtol,
        "direct_l2": left_norm,
        "reused_l2": right_norm,
        "difference_l2": difference,
        "relative_difference_l2": difference / left_norm if finite and left_norm else None,
        "max_abs_difference": max(p["max_abs_difference"] for p in per_parameter.values())
        if finite
        else None,
        "per_parameter": per_parameter,
        "scope": (
            "NUMERICAL_PRECLIP_COMPARISON_ONLY; "
            "caller must also enforce bank/state/ratio/adoption gates"
        ),
    }


def _finite_adam_state(optimizer):
    import torch

    def check(value):
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite Adam state; caller must restore scratch state")
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError("Nonfinite Adam state or hyperparameter")

    check(optimizer.state_dict())


def apply_gradient_update(model, optimizer, gradients, grad_clip=1):
    """Apply every explicit tensor, including zeros, then global clip and Adam step.

    No scheduler or grad scaler is configured by this API. The caller must use
    the same absence contract in its direct and reused candidates.
    """
    import torch

    parameters = _parameters(model)
    grad_clip = _number(grad_clip, "grad_clip", positive=True)
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        raise ValueError("R3 requires an Adam/AdamW optimizer")
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if len(owned) != len(parameters) or {id(p) for p in owned} != {
        id(p) for p in parameters.values()
    }:
        raise ValueError(
            "Optimizer must own every trainable parameter exactly once, and no frozen ones"
        )
    if set(gradients) != set(parameters):
        raise ValueError("Gradient names must cover every trainable parameter exactly")
    _finite_adam_state(optimizer)
    converted = {}
    for name, parameter in parameters.items():
        value = gradients[name]
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
            raise ValueError(f"Gradient shape/type mismatch: {name}")
        if value.is_sparse or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Finite dense floating gradient required: {name}")
        converted[name] = value.detach().to(device=parameter.device, dtype=parameter.dtype).clone()
        if not torch.isfinite(converted[name]).all():
            raise FloatingPointError(f"Gradient conversion overflow: {name}")
    before = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    parameter_before = parameter_hash(model, trainable=True)
    optimizer_before = state_hash(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in parameters.items():
        parameter.grad = converted[name]
    pre = torch.nn.utils.clip_grad_norm_(
        list(parameters.values()), grad_clip, error_if_nonfinite=True
    )
    post = _norm({name: p.grad.detach() for name, p in parameters.items()})
    optimizer.step()
    _finite_adam_state(optimizer)
    if any(not torch.isfinite(p).all() for p in parameters.values()):
        raise FloatingPointError(
            "Nonfinite parameters after Adam update; caller must restore scratch state"
        )
    delta = {
        name: p.detach().float().cpu() - before[name].float() for name, p in parameters.items()
    }
    return {
        "parameter_hash_before": parameter_before,
        "parameter_hash_after": parameter_hash(model, trainable=True),
        "optimizer_hash_before": optimizer_before,
        "optimizer_hash_after": state_hash(optimizer.state_dict()),
        "grad_norm_preclip": float(pre),
        "grad_norm_postclip": post,
        "actual_step_norm": _norm(delta),
        "optimizer_updates": 1,
        "gradient_hash_preclip": state_hash(gradients),
        "gradient_hash_postclip": state_hash(
            {name: p.grad.detach().float().cpu() for name, p in parameters.items()}
        ),
        "all_trainable_gradients_explicit": True,
        "scheduler": "NOT_CONFIGURED",
        "grad_scaler": "NOT_CONFIGURED",
        "zero_gradient_semantics": "EXPLICIT_TENSOR_AND_ALWAYS_ADAM_STEP",
        "finite_optimizer_state": True,
    }
