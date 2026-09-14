"""Standard-library reference for the follow-up design, NOT a GPU runtime.

No model loading, network, filesystem writes, optimizer calls or training.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

CATEGORIES = ("X", "S", "W", "I")


def _number(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be bool")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and nonnegative/positive")
    return result


def group_statistics(
    categories: Sequence[str], *, auxiliary_weight: float,
    policy: str = "joint", epsilon: float = 1e-4,
) -> dict[str, Any]:
    if len(categories) < 2 or any(c not in CATEGORIES for c in categories):
        raise ValueError("A group needs at least two known categories")
    if policy not in ("joint", "no_x_off"):
        raise ValueError("Unknown group policy")
    lam = _number(auxiliary_weight, "auxiliary_weight")
    epsilon = _number(epsilon, "epsilon", positive=True)
    counts = {c: categories.count(c) for c in CATEGORIES}
    suppressed = policy == "no_x_off" and counts["X"] == 0
    effective = 0.0 if suppressed else lam
    rewards = [2.0 * (c == "X") + effective * (c != "I") for c in categories]
    mean = math.fsum(rewards) / len(rewards)
    centered = [v - mean for v in rewards]
    variance = math.fsum(v * v for v in centered) / len(rewards)
    denominator = math.sqrt(variance + epsilon * epsilon)
    advantage = [v / denominator for v in centered] if variance else [0.0] * len(rewards)
    # Derivative with respect to the requested lambda, on this fixed group.
    validity = [float(c != "I") for c in categories]
    mean_v = math.fsum(validity) / len(validity)
    centered_v = [v - mean_v for v in validity]
    covariance = math.fsum(r * v for r, v in zip(centered, centered_v)) / len(rewards)
    derivative = [0.0] * len(rewards) if suppressed else [
        v / denominator - r * covariance / denominator**3
        for r, v in zip(centered, centered_v)
    ]
    return {
        "counts": counts, "requested_lambda": lam, "effective_lambda": effective,
        "auxiliary_suppressed": suppressed, "rewards": rewards, "mean": mean,
        "variance": variance, "denominator": denominator, "advantages": advantage,
        "d_advantage_d_requested_lambda": derivative,
    }


def vector_comparison(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    if not left or len(left) != len(right):
        raise ValueError("Matching nonempty vectors required")
    a, b = list(map(float, left)), list(map(float, right))
    if not all(math.isfinite(v) for v in a + b):
        raise ValueError("Finite vector entries required")
    na = math.sqrt(math.fsum(v*v for v in a))
    nb = math.sqrt(math.fsum(v*v for v in b))
    dot = math.fsum(x*y for x, y in zip(a, b))
    difference = math.sqrt(math.fsum((x-y)**2 for x, y in zip(a, b)))
    return {"left_l2": na, "right_l2": nb, "difference_l2": difference,
            "cosine": max(-1.0, min(1.0, dot / (na*nb))) if na and nb else None,
            "exact_equal": a == b}


def fixed_panel_radius(weights: Sequence[float], sample_counts: Sequence[int],
                       *, alpha: float = 0.05, statements: int = 1) -> float:
    if not weights or len(weights) != len(sample_counts):
        raise ValueError("Matching nonempty weights and sample counts required")
    w = [_number(v, "weight") for v in weights]
    if not math.isclose(math.fsum(w), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Weights must sum to one")
    if any(type(n) is not int or n <= 0 for n in sample_counts):
        raise ValueError("Positive integer sample counts required")
    if isinstance(alpha, bool) or not 0 < alpha < 1 or type(statements) is not int or statements < 1:
        raise ValueError("Invalid confidence inputs")
    scale = math.fsum(v*v / n for v, n in zip(w, sample_counts))
    return math.sqrt(0.5 * scale * math.log(2 * statements / alpha))


def design_budget(design: dict[str, Any]) -> dict[str, int]:
    s1, opt, s2 = design["S1"], design["optimization"], design["S2"]
    units = s1["units"]
    if len({u["id"] for u in units}) != len(units):
        raise ValueError("Duplicate bank IDs")
    candidate_count = direct_count = replay_count = 0
    for unit in units:
        names = [c["id"] for c in unit["candidates"]]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate candidate IDs")
        if not set(unit["direct_candidate_ids"]).issubset(set(names)):
            raise ValueError("Evaluation references an absent candidate")
        candidate_count += len(names)
        direct_count += len(unit["direct_candidate_ids"])
        replay_count += unit["baseline_replay_updates"]
    total_updates = candidate_count + replay_count
    direct = s1["direct_evaluation"]
    eval_spec = s2["final_eval"]
    # Step64's panel is read from final N, not generated a second time.
    panel_steps = [s for s in s2["dev_panel"]["steps"] if s != s2["steps"]]
    per_run_panel = len(panel_steps) * s2["dev_panel"]["prompts"] * s2["dev_panel"]["samples_per_prompt"]
    final = sum(eval_spec[k] for k in ("N_prompts", "L_prompts", "OOD_prompts")) * eval_spec["samples_per_prompt"]
    initial_ood = eval_spec["OOD_prompts"] * eval_spec["samples_per_prompt"]
    diagnostic = s2["interface_diagnostic"]
    diagnostic_outputs = diagnostic["base_scenes"] * len(diagnostic["conditions"]) * diagnostic["samples_per_condition"] * len(diagnostic["steps"])
    per_run_eval = per_run_panel + final + initial_ood + diagnostic_outputs
    n_runs = len(s2["new_runs"])
    return {
        "new_candidates": candidate_count, "baseline_replays": replay_count,
        "candidate_and_replay_adam_calls": total_updates,
        "backward_sequence_calls": total_updates * opt["B"] * opt["K"],
        "logical_direct_candidates": direct_count,
        "S1_direct_outputs_upper": direct_count * direct["prompts"] * direct["samples_per_prompt"],
        "S1_optional_fresh64_outputs_upper": direct_count * direct["prompts"] * 64,
        "S2_new_runs": n_runs,
        "S2_training_outputs": n_runs * s2["steps"] * opt["B"] * opt["K"],
        "S2_evaluation_outputs_per_new_run_upper": per_run_eval,
        "S2_evaluation_outputs_new_runs_upper": n_runs * per_run_eval,
        "historical_two_endpoint_interface_outputs_upper": 2 * diagnostic["base_scenes"] * len(diagnostic["conditions"]) * diagnostic["samples_per_condition"],
    }
