"""Fixed-prompt estimands. Repeated completions are not independent scenes."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

CATEGORIES = ("X", "S", "W", "I")


def summarize(rows, prompt_weights=None):
    by_prompt = defaultdict(list)
    for row in rows:
        if row["category"] not in CATEGORIES:
            raise ValueError("unknown category")
        by_prompt[row["prompt_id"]].append(row)
    if not by_prompt:
        raise ValueError("no observations; probabilities are NA")
    weights = prompt_weights or {key: 1 / len(by_prompt) for key in by_prompt}
    if set(weights) != set(by_prompt) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("fixed prompt weights must cover all prompts and sum to one")
    if any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("prompt weights must be finite and nonnegative")
    probs = {category: 0.0 for category in CATEGORIES}
    macro = []
    for key, group in by_prompt.items():
        counts = Counter(row["category"] for row in group)
        probs = {c: probs[c] + weights[key] * counts[c] / len(group) for c in CATEGORIES}
        valid = len(group) - counts["I"]
        macro.append(counts["X"] / valid if valid else None)
    valid = probs["X"] + probs["S"] + probs["W"]
    answer = probs["X"] + probs["S"]
    observed_macro = [value for value in macro if value is not None]
    counts = Counter(row["category"] for row in rows)
    return {
        **{f"p{c}": probs[c] for c in CATEGORIES},
        "v": valid,
        "pA": answer,
        "qX": probs["X"] / valid if valid else None,
        "qS": probs["S"] / valid if valid else None,
        "truth_given_answer": probs["X"] / answer if answer else None,
        "macro_mean_per_prompt_qX": float(np.mean(observed_macro)) if observed_macro else None,
        "macro_qX_NA_fraction": sum(v is None for v in macro) / len(macro),
        "category_counts": {c: counts[c] for c in CATEGORIES},
        "missing_categories": [c for c in CATEGORIES if not counts[c]],
        "independent_scene_count": len({row["base_scene_id"] for row in rows}),
        "rollout_count": len(rows),
        "prompt_count": len(by_prompt),
        "estimator": "fixed_weight_mean_of_per_prompt_probabilities",
        "zero_count_interpretation": "unobserved support, not theoretical zero",
    }


def cluster_bootstrap(rows, metric="pX", repeats=10000, seed=17, alpha=0.05):
    """Single-seed scene cluster CI. Retains every prompt/interface/rollout."""
    if repeats < 2 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap arguments")
    if len({row.get("train_seed") for row in rows}) > 1:
        raise ValueError("multi-seed data require outer seed resampling, not this single-seed CI")
    scenes = sorted({row["base_scene_id"] for row in rows})
    if len(scenes) < 2:
        return {"status": "UNKNOWN", "reason": "fewer than two independent scenes"}
    groups = {s: [r for r in rows if r["base_scene_id"] == s] for s in scenes}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sampled = [
            {**row, "prompt_id": f"{index}:{row['prompt_id']}", "base_scene_id": str(index)}
            for index, scene in enumerate(rng.choice(scenes, len(scenes)))
            for row in groups[scene]
        ]
        value = summarize(sampled)[metric]
        if value is not None:
            estimates.append(value)
    # Discarding denominator-zero replicates would silently condition the CI.
    if len(estimates) != repeats:
        return {
            "status": "UNKNOWN",
            "reason": "undefined denominator in bootstrap replicates",
            "defined_replicates": len(estimates),
            "repeats": repeats,
        }
    lo, hi = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {
        "status": "ESTIMATED",
        "low": float(lo),
        "high": float(hi),
        "uncertainty": "single_training_seed_scene_cluster_percentile_bootstrap",
        "repeats": repeats,
        "seed": seed,
    }
