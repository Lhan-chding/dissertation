"""Joint group z-score GRPO with the predeclared fixed token denominator."""

from __future__ import annotations

import math


def grouped_advantages(rewards, epsilon=1e-4):
    values = [float(v) for v in rewards]
    if not values or epsilon < 0 or not all(math.isfinite(v) for v in [*values, epsilon]):
        raise ValueError("Finite nonempty rewards and nonnegative epsilon required")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((v - mean) ** 2 for v in values) / len(values)
    denominator = math.sqrt(variance + epsilon**2)
    advantages = [(v - mean) / denominator for v in values] if variance else [0.0] * len(values)
    return {
        "advantages": advantages,
        "mean": mean,
        "std": math.sqrt(variance),
        "variance": variance,
        "zero_variance_flag": variance == 0,
        "epsilon_convention": "sqrt(population_variance + epsilon^2)",
        "epsilon": epsilon,
    }


def ppo_surrogate(new_logprobs, old_logprobs, advantages, lnorm=64, clip_epsilon=0.2):
    if (
        lnorm <= 0
        or not advantages
        or len(new_logprobs) != len(advantages)
        or len(old_logprobs) != len(advantages)
    ):
        raise ValueError("Invalid group lengths or Lnorm")
    terms = []
    for new, old, advantage in zip(new_logprobs, old_logprobs, advantages, strict=False):
        if len(new) != len(old):
            raise ValueError("Old/new token lengths differ")
        for a, b in zip(new, old, strict=False):
            ratio = math.exp(a - b)
            clipped = max(1 - clip_epsilon, min(1 + clip_epsilon, ratio))
            terms.append(min(ratio * advantage, clipped * advantage))
    return -math.fsum(terms) / (len(advantages) * lnorm)


def torch_ppo_loss(new, old, advantages, mask, *, lnorm=64, clip_epsilon=0.2, total_sequences=None):
    import torch

    if new.shape != old.shape or new.shape != mask.shape or new.ndim != 2:
        raise ValueError("Expected equal [sequences,tokens] shapes")
    if advantages.shape != (new.shape[0],) or lnorm <= 0:
        raise ValueError("Invalid advantages or fixed Lnorm")
    denominator = total_sequences or new.shape[0]
    if denominator < new.shape[0] or not mask.any():
        raise ValueError("Invalid accumulation denominator or empty token mask")
    ratio = (new.float() - old.detach().float()).exp()
    adv = advantages.detach().float()[:, None]
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    loss = -torch.minimum(ratio * adv, clipped * adv).masked_fill(~mask, 0).sum() / (
        denominator * lnorm
    )
    audit = {
        "clip_fraction": float(((ratio != clipped) & mask).sum().detach() / mask.sum()),
        "loss_reduction": f"sum_generated_tokens/(B*K*{lnorm})",
        "masked_token_count": int(mask.sum()),
        "Lnorm": lnorm,
    }
    return loss, audit


def reward_channels(category, arm="X_BASE", auxiliary_weight=None):
    if category not in ("X", "S", "W", "I") or arm not in (
        "X_BASE",
        "X_VALID",
        "A_BASE",
        "A_VALID",
    ):
        raise ValueError("Unknown category or N-protocol arm")
    alpha_x = 2.0 if arm.startswith("X") else 0.0
    alpha_a = 2.0 if arm.startswith("A") else 0.0
    lam = float(arm.endswith("VALID")) if auxiliary_weight is None else float(auxiliary_weight)
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("Nonnegative finite auxiliary weight required")
    channels = {
        "exact": alpha_x * (category == "X"),
        "answer": alpha_a * (category in ("X", "S")),
        "validity": lam * (category != "I"),
    }
    return {**channels, "sum": sum(channels.values())}


def perform_update(adapter, optimizer, groups, *, lnorm=64, clip_epsilon=0.2, grad_clip=1.0):
    """One real Adam step after every K rollout has finished; microbatch=1."""
    import torch

    if not groups or len({len(group) for group in groups}) != 1 or not groups[0]:
        raise ValueError("Nonempty equal-K groups required")
    parameters = {name: p for name, p in adapter.model.named_parameters() if p.requires_grad}
    before = {name: p.detach().clone() for name, p in parameters.items()}
    optimizer.zero_grad(set_to_none=True)
    records = []
    total = sum(map(len, groups))
    for group in groups:
        stats = grouped_advantages([row["reward_sum"] for row in group])
        for row, advantage in zip(group, stats["advantages"], strict=False):
            new = adapter.logprobs(row["prepared"], row["token_ids"], require_grad=True)
            old = torch.tensor(row["old_logprobs"], device=new.device, dtype=torch.float32)
            loss, audit = torch_ppo_loss(
                new[None],
                old[None],
                torch.tensor([advantage], device=new.device),
                torch.ones_like(new[None], dtype=torch.bool),
                lnorm=lnorm,
                clip_epsilon=clip_epsilon,
                total_sequences=total,
            )
            loss.backward()
            records.append({**audit, "loss": float(loss.detach()), "advantage": advantage})
    if any(p.grad is not None for p in adapter.model.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen parameter received a gradient")
    if not any(p.grad is not None for p in parameters.values()):
        raise RuntimeError("No trainable gradients")
    pre = torch.nn.utils.clip_grad_norm_(
        list(parameters.values()), grad_clip, error_if_nonfinite=True
    )
    post = math.sqrt(
        sum(
            float(p.grad.detach().float().square().sum())
            for p in parameters.values()
            if p.grad is not None
        )
    )
    optimizer.step()
    step_norm = math.sqrt(
        sum(
            float((p.detach() - before[name]).float().square().sum())
            for name, p in parameters.items()
        )
    )
    policy_kls, reference_kls = [], []
    for group in groups:
        for row in group:
            new = adapter.logprobs(row["prepared"], row["token_ids"]).detach().cpu().double()
            log_ratio = float(new.sum()) - math.fsum(row["old_logprobs"])
            policy_kls.append(math.expm1(log_ratio) - log_ratio)
            if row.get("base_token_logprobs") is not None:
                ref_ratio = math.fsum(row["base_token_logprobs"]) - math.fsum(row["old_logprobs"])
                reference_kls.append(math.expm1(ref_ratio) - ref_ratio)
    return {
        "grad_norm_preclip": float(pre),
        "grad_norm_postclip": post,
        "actual_step_norm": step_norm,
        "sequences": total,
        "backward_calls": total,
        "optimizer_updates": 1,
        "clip_fraction": sum(row["clip_fraction"] for row in records) / total,
        "loss": sum(row["loss"] for row in records),
        "Lnorm": lnorm,
        "token_records": records,
        "empirical_same_training_bank_kl": math.fsum(policy_kls) / total,
        "reference_kl": math.fsum(reference_kls) / len(reference_kls) if reference_kls else None,
        "kl_estimator": (
            "sequence k3 = exp(log_ratio)-1-log_ratio; same training bank used to adapt candidate; "
            "empirical proxy, not independent population KL"
        ),
        "post_update_likelihood_forwards": total,
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pilot", "confirm"], required=True)
    parser.add_argument("--arm", choices=["X_BASE", "X_VALID", "A_BASE", "A_VALID"], required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-training", action="store_true")
    parser.parse_args(argv)
    parser.exit(
        2,
        "NOT_IMPLEMENTED_UNTIL_P1: complete the NTU GPU P1 audit and review its locked runtime "
        "before pilot; confirm requires separate researcher authorization.\n",
    )


if __name__ == "__main__":
    main()
