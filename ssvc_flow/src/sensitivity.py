"""FP64 derivatives of joint group reward standardization.

These formulas are derived here; channelwise standardization is separately
labelled and is not represented as an exact decomposition or official GDPO.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _array(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("rewards must be a finite nonempty vector")
    return array


def _epsilon(value):
    if not np.isfinite(value) or value < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    return float(value)


def normalized_advantages(rewards, epsilon=1e-4):
    """Population variance (divisor K); constant groups are zero, even at eps=0."""
    rewards, epsilon = _array(rewards), _epsilon(epsilon)
    centered = rewards - rewards.mean()
    denominator = np.sqrt(np.mean(centered**2) + epsilon**2)
    return centered / denominator if denominator else np.zeros_like(centered)


def advantage_derivative(primary, validity, lam, epsilon=1e-4):
    """dA/dlambda = vc/s - rc*mean(rc*vc)/s**3, s²=var(r)+eps²."""
    primary, validity = _array(primary), _array(validity)
    epsilon = _epsilon(epsilon)
    if primary.shape != validity.shape or not np.isfinite(lam):
        raise ValueError("reward channels must have equal shapes and finite lambda")
    rewards = primary + float(lam) * validity
    centered, valid_centered = rewards - rewards.mean(), validity - validity.mean()
    scale = np.sqrt(np.mean(centered**2) + epsilon**2)
    if not scale:
        if np.any(valid_centered):
            raise ValueError("derivative undefined at epsilon=0 reward tie")
        return np.zeros_like(centered)
    covariance = np.mean(centered * valid_centered)
    return valid_centered / scale - centered * covariance / scale**3


def audit_derivative(primary, validity, lam, epsilon=1e-4, score_vectors=None):
    """Compare the derived formula, independent torch autograd, and three FD steps."""
    import torch

    primary, validity = _array(primary), _array(validity)
    analytic = advantage_derivative(primary, validity, lam, epsilon)
    variable = torch.tensor(float(lam), dtype=torch.float64, requires_grad=True)
    base, channel = torch.tensor(primary), torch.tensor(validity)

    def advantage(weight):
        reward = base + weight * channel
        centered = reward - reward.mean()
        return centered / torch.sqrt(torch.mean(centered**2) + epsilon**2)

    auto = torch.autograd.functional.jacobian(advantage, variable).detach().numpy()
    rows = []
    for step in [1e-2, 1e-3, 1e-4]:
        upper = normalized_advantages(primary + (lam + step) * validity, epsilon)
        lower = normalized_advantages(primary + (lam - step) * validity, epsilon)
        difference = (upper - lower) / (2 * step)
        error = np.abs(difference - analytic)
        nonzero = np.abs(analytic) > 1e-8
        relative = error[nonzero] / np.abs(analytic[nonzero])
        rows.append(
            {
                "step": step,
                "derivative": difference.tolist(),
                "max_abs_error": float(error.max()),
                "max_relative_error_nonzero": float(relative.max()) if relative.size else None,
                "passes_mixed_tolerance": bool(np.all(error <= 1e-8 + 1e-5 * np.abs(analytic))),
            }
        )
    joint = normalized_advantages(primary + lam * validity, epsilon)
    separate = normalized_advantages(primary, epsilon) + lam * normalized_advantages(
        validity, epsilon
    )
    # Default scores are those of a uniform n-action categorical policy with
    # each action represented once; caller-provided sequence scores can replace
    # this independent CPU fixture when a frozen model rollout bank is available.
    scores = (
        np.eye(len(primary)) - 1 / len(primary)
        if score_vectors is None
        else np.asarray(score_vectors, dtype=np.float64)
    )
    if scores.ndim != 2 or scores.shape[0] != len(primary) or not np.isfinite(scores).all():
        raise ValueError("score_vectors must have one finite row per reward")
    return {
        "lambda": float(lam),
        "lambda_over_alpha": float(lam / 2),
        "epsilon": epsilon,
        "variance_divisor": len(primary),
        "dtype": "float64",
        "analytic": analytic.tolist(),
        "autograd": auto.tolist(),
        "autograd_max_abs_error": float(np.max(np.abs(analytic - auto))),
        "finite_differences": rows,
        "joint_advantages": joint.tolist(),
        "channelwise_standardized_sum": separate.tolist(),
        "joint_vs_separate_norm": float(np.linalg.norm(joint - separate)),
        "joint_gradient": (joint @ scores / len(primary)).tolist(),
        "channelwise_gradient": (separate @ scores / len(primary)).tolist(),
        "lambda_gradient_derivative": (analytic @ scores / len(primary)).tolist(),
        "score_source": "uniform_categorical_fixture"
        if score_vectors is None
        else "caller_provided",
        "channelwise_method": "illustrative_separate_standardization_not_official_GDPO",
        "formula_source": "self_derived_from_joint_group_zscore",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from .core import phase_artifacts, write_json

    args.out.mkdir(parents=True, exist_ok=True)
    values = [
        audit_derivative([2, 0, 0, 0, 2, 0, 0, 0], [1, 1, 1, 0, 1, 0, 1, 1], lam)
        for lam in [0, 0.1, 0.25, 0.5, 1, 2, 4]
    ]
    path = args.out / "derivative_audit.json"
    write_json(path, values)
    phase_artifacts(
        args.out,
        "P6a",
        "CPU_TOY_PASSED",
        {"scope": "toy_mathematical_validation", "model_rollout_bank_audit": "pending_gpu"},
        [path],
    )


if __name__ == "__main__":
    main()
