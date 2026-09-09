"""Reference joint X/validity advantages and lambda response on fixed banks."""

from __future__ import annotations

import math

from reference.finite_group_reference import normalized_advantage_and_derivative


def reward_vector(categories, lam, alpha=2.0):
    return [alpha * (c == "X") + lam * (c != "I") for c in categories]


def compare_lambdas(categories, lambdas, *, alpha=2.0, epsilon=1e-4):
    categories = list(categories)
    if not categories:
        raise ValueError("empty category bank")
    baseline, _ = normalized_advantage_and_derivative(categories, 0.0, alpha, epsilon)
    result = {}
    for lam in lambdas:
        advantages, derivative = normalized_advantage_and_derivative(
            categories, float(lam), alpha, epsilon
        )
        result[str(lam)] = {
            "advantages": advantages,
            "dA_dlambda": derivative,
            "delta_from_lambda0_l2": math.sqrt(
                sum((a - b) ** 2 for a, b in zip(advantages, baseline, strict=False))
                / len(categories)
            ),
            "zero_variance": all(v == 0.0 for v in advantages),
        }
    return {"alpha": alpha, "epsilon": epsilon, "baseline_lambda0": baseline, "lambdas": result}


def finite_difference(categories, lam, h=1e-6, *, alpha=2.0, epsilon=1e-4):
    if h <= 0:
        raise ValueError("h must be positive")
    left, _ = normalized_advantage_and_derivative(categories, lam - h, alpha, epsilon)
    right, _ = normalized_advantage_and_derivative(categories, lam + h, alpha, epsilon)
    return [(upper - lower) / (2 * h) for lower, upper in zip(left, right, strict=False)]
