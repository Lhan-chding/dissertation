"""Exact hypergeometric reward-group expectations, conditional on a frozen bank."""

from __future__ import annotations

import itertools
import math
from functools import lru_cache

from reference.finite_group_reference import bank_group_rates, normalized_advantage_and_derivative

from .grpo_update import grouped_advantages

CATEGORIES = ("X", "S", "W", "I")


@lru_cache(maxsize=4096)
def bank_advantage_statistics(counts, k, lam, epsilon=1e-4):
    rates = bank_group_rates(counts, k)  # validates counts and K <= n
    denominator = math.comb(sum(counts), k)
    result = {
        "mean_squared_A_lambda_minus_A0": 0.0,
        "mean_squared_dA_dlambda": 0.0,
        "mean_squared_A_reward_minus_X_reward": 0.0,
        "zero_variance_X": 0.0,
        "zero_variance_X_VALID": 0.0,
        "zero_variance_A": 0.0,
        "zero_variance_A_VALID": 0.0,
    }
    derivative_defined = True
    for parts in itertools.product(*(range(min(n, k) + 1) for n in counts[:3])):
        ni = k - sum(parts)
        if not 0 <= ni <= counts[3]:
            continue
        selected = (*parts, ni)
        weight = (
            math.prod(math.comb(n, m) for n, m in zip(counts, selected, strict=False)) / denominator
        )
        categories = [c for c, n in zip(CATEGORIES, selected, strict=False) for _ in range(n)]
        base, _ = normalized_advantage_and_derivative(categories, 0.0, epsilon=epsilon)
        candidate, derivative = normalized_advantage_and_derivative(
            categories, lam, epsilon=epsilon
        )
        answer = grouped_advantages([2.0 * (c in ("X", "S")) for c in categories], epsilon)[
            "advantages"
        ]
        result["mean_squared_A_lambda_minus_A0"] += (
            weight * sum((a - b) ** 2 for a, b in zip(candidate, base, strict=False)) / k
        )
        result["mean_squared_A_reward_minus_X_reward"] += (
            weight * sum((a - b) ** 2 for a, b in zip(answer, base, strict=False)) / k
        )
        if all(math.isfinite(d) for d in derivative):
            result["mean_squared_dA_dlambda"] += weight * sum(d * d for d in derivative) / k
        else:
            derivative_defined = False
        for name, primary, auxiliary in [
            ("X", ("X",), 0.0),
            ("X_VALID", ("X",), lam),
            ("A", ("X", "S"), 0.0),
            ("A_VALID", ("X", "S"), lam),
        ]:
            rewards = [2.0 * (c in primary) + auxiliary * (c != "I") for c in categories]
            result["zero_variance_" + name] += weight * (len(set(rewards)) == 1)
    if not derivative_defined:
        result["mean_squared_dA_dlambda"] = None
    return {**rates, **result, "derivative_defined": derivative_defined}


def audit_categories(
    categories,
    ks=(2, 4, 8, 16),
    lambdas=(0.0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.25, 1.0, 2.0),
    epsilon=1e-4,
):
    if any(c not in CATEGORIES for c in categories):
        raise ValueError("unknown category")
    counts = tuple(categories.count(c) for c in CATEGORIES)
    return [
        {"K": k, "lambda": lam, **bank_advantage_statistics(counts, k, lam, epsilon)}
        for k in ks
        for lam in lambdas
    ]
