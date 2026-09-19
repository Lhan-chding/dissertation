"""Frozen four-channel rewards and independently implemented GDPO/SAW baselines.

See docs/decision_modeling/BASELINE_MAPPING_zh.md for exact conventions and sources.
"""

from __future__ import annotations

import numpy as np

from ..grpo_update import grouped_advantages

CHANNELS = ("X", "A", "V", "C")
RECIPES = {
    "R0": (2.0, 0.0, 0.0, 0.0),
    "R1": (0.0, 2.0, 0.0, 0.0),
    "R2": (2.0, 0.0, 1.0, 0.0),
    "R3": (2.0, 0.0, 0.0, 1.0),
    "R4": (2.0, 0.0, 0.5, 0.5),
    "R5": (0.0, 2.0, 1.0, 0.0),
    "R6": (0.0, 2.0, 0.0, 1.0),
    "R7": (0.0, 2.0, 0.5, 0.5),
}
BASELINES = ("GDPO_R4", "SAW_R4")
SAW_DELTA = 1e-4


def reward_vector(semantic):
    event = semantic["event"]
    if event not in ("X", "S", "W", "I"):
        raise ValueError("Unknown semantic event")
    c = float(semantic["relation_score"])
    if not np.isfinite(c) or not 0 <= c <= 1 or (event == "I" and c != 0):
        raise ValueError("Relation reward must be finite in [0,1] and zero for I")
    a, v = int(event in ("X", "S")), int(event != "I")
    if semantic.get("answer_correct", a) != a or semantic.get("valid", v) != v:
        raise ValueError("Derived semantic rewards conflict with event")
    return [float(event == "X"), float(a), float(v), c]


def _tensor(values):
    rewards = np.asarray(values, dtype=np.float64)
    if rewards.ndim != 3 or rewards.shape[-1] != 4 or min(rewards.shape[:2]) < 1:
        raise ValueError("Rewards must have shape [B,K,4]")
    if rewards.shape[1] < 2 or not np.isfinite(rewards).all():
        raise ValueError("Finite rewards and K >= 2 required")
    if ((rewards < 0) | (rewards > 1)).any():
        raise ValueError("All four original reward channels lie in [0,1]")
    x, a, v, c = np.moveaxis(rewards, -1, 0)
    if (
        any(not np.isin(t, (0.0, 1.0)).all() for t in (x, a, v))
        or (x > a).any()
        or (a > v).any()
        or ((v == 0) & (c != 0)).any()
    ):
        raise ValueError("Four-channel semantic nesting violated")
    return rewards


def saw_weights(values, *, priorities=RECIPES["R4"], delta=SAW_DELTA):
    """Paper Algorithm 1, GRPO mode; known theoretical minimum 0, priority after CV."""
    rewards = _tensor(values)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("Positive CV stability constant required")
    priority = np.asarray(priorities, dtype=float)
    if (
        priority.shape != (4,)
        or not np.isfinite(priority).all()
        or (priority < 0).any()
        or not priority.any()
    ):
        raise ValueError("Four nonnegative priorities, at least one positive, required")
    active = priority > 0
    flat = rewards.reshape(-1, 4)
    # Algorithm 1 uses theoretical minimum, NOT empirical batch minimum.
    shifted = flat + delta
    means = shifted.mean(axis=0)
    stds = shifted.std(axis=0, ddof=1)
    cv = stds / (means + delta)
    total = cv[active].sum()
    weights = np.zeros(4)
    fallback = bool(total < delta)
    weights[active] = 1.0 if fallback else cv[active] / total
    return {
        "weights": (weights * priority).tolist(),
        "cv_weights": weights.tolist(),
        "priorities": priority.tolist(),
        "cv": cv.tolist(),
        "raw_batch_mean": flat.mean(axis=0).tolist(),
        "batch_std": stds.tolist(),
        "delta": float(delta),
        "offset": "theoretical_minimum_zero_plus_delta",
        "cv_mean_denominator": "raw_mean + 2*delta",
        "std_ddof": 1,
        "fallback_equal_active_weights": fallback,
        "mode": "GRPO_reward",
    }


def compute_advantages(values, recipe, *, epsilon=1e-4, saw_delta=SAW_DELTA):
    rewards = _tensor(values)
    if recipe not in (*RECIPES, *BASELINES):
        raise ValueError("Unknown reward recipe")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive normalization epsilon required")
    priority = np.asarray(RECIPES.get(recipe, RECIPES["R4"]))
    adaptation = (
        saw_weights(rewards, priorities=priority, delta=saw_delta) if recipe == "SAW_R4" else None
    )
    weights = np.asarray(adaptation["weights"] if adaptation else priority)
    total = rewards @ weights
    statistics = [grouped_advantages(group, epsilon) for group in total]
    if recipe == "GDPO_R4":
        # Official NVlabs TRL reference: std correction=1, epsilon outside sqrt.
        mean = rewards.mean(axis=1, keepdims=True)
        std = rewards.std(axis=1, keepdims=True, ddof=1)
        per_channel = (rewards - mean) / (std + epsilon)
        combined = per_channel @ priority
        batch_mean, batch_std = float(combined.mean()), float(combined.std(ddof=1))
        advantage = (combined - batch_mean) / (batch_std + epsilon)
        extra = {
            "group_channel_mean": mean[:, 0].tolist(),
            "group_channel_std": std[:, 0].tolist(),
            "per_channel_advantages": per_channel.tolist(),
            "pre_batch_advantages": combined.tolist(),
            "batch_mean": batch_mean,
            "batch_std": batch_std,
            "std_ddof": 1,
            "normalization": "channel_group_std_plus_epsilon_then_sequence_batch_std_plus_epsilon",
        }
    else:
        advantage = np.asarray([s["advantages"] for s in statistics])
        extra = {
            "normalization": "joint_sqrt_population_variance_plus_epsilon_squared",
            "std_ddof": 0,
        }
    if not np.isfinite(advantage).all():
        raise FloatingPointError("Nonfinite advantages")
    return {
        "recipe": recipe,
        "channels": list(CHANNELS),
        "reward_vectors": rewards.tolist(),
        "priorities": priority.tolist(),
        "actual_weights": weights.tolist(),
        "total_rewards": total.tolist(),
        "advantages": advantage.tolist(),
        "group_statistics": statistics,
        "epsilon": epsilon,
        "saw": adaptation,
        **extra,
    }
