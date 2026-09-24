"""Pre-test precision planning; independent lineages, never checkpoint counts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

CELLS = (("R0", 32), ("R0", 96), ("R1", 32), ("R1", 96))
N_CHOICES = (12, 16, 20, 24, 32, 48)
PLANNING_Z = 1.96 + 0.841621


def sample_variance(values: np.ndarray) -> float:
    """Keep exactly constant observations at zero despite mean-roundoff noise."""
    return 0.0 if np.all(values == values[0]) else float(np.var(values, ddof=1))


def cell_key(row: Mapping) -> tuple[str, int]:
    cell = (str(row["source_recipe"]), int(row["origin_step"]))
    if cell not in CELLS:
        raise ValueError(f"Unregistered source/stage cell: {cell}")
    return cell


def plan_mc_samples(
    n_lineages: int,
    *,
    n_prompts: int = 288,
    repeats: int = 2,
    effect: float = 0.01,
    draws_choices: Sequence[int] = (16, 32, 64),
    prompt_weights: Sequence[float] | None = None,
) -> dict:
    """Worst-case independent Bernoulli streams for the complete test mean.

    Uses variance <= .25 per policy. Zero observed failures never reduce this
    bound. Prompt weights must reflect the six equally weighted strata.
    """
    if min(n_lineages, n_prompts, repeats) < 1 or not math.isfinite(effect) or effect <= 0:
        raise ValueError("Positive sample counts and effect required")
    choices = tuple(sorted(set(draws_choices)))
    if not choices or any(m < 1 or int(m) != m for m in choices):
        raise ValueError("Positive integer draw choices required")
    weights = (
        np.full(n_prompts, 1 / n_prompts)
        if prompt_weights is None
        else np.asarray(prompt_weights, dtype=float)
    )
    if (
        weights.shape != (n_prompts,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Prompt weights must be finite, nonnegative, and sum to one")
    budget = effect / 4
    candidates = [
        {
            "m": int(m),
            "mc_se_upper": math.sqrt(0.5 * float(weights @ weights) / (m * n_lineages * repeats)),
        }
        for m in choices
    ]
    selected = next((row for row in candidates if row["mc_se_upper"] <= budget), candidates[-1])
    return {
        **selected,
        "status": "MC_BUDGET_MET" if selected["mc_se_upper"] <= budget else "MC_BUDGET_UNMET",
        "mc_se_target": budget,
        "candidates": candidates,
        "n_lineages": n_lineages,
        "n_prompts": n_prompts,
        "repeats": repeats,
        "variance_source": "WORST_CASE_BERNOULLI_0.25",
        "estimand": "fixed-panel mean over all lineages and repeats",
        "includes_training_or_scene_variation": False,
    }


def plan_precision(
    anchor_rows: Sequence[Mapping],
    anchor_registry: Mapping[str, Mapping],
    *,
    effect: float = 0.01,
    bootstrap_repetitions: int = 10000,
    seed: int = 75001,
    test_outcomes_exist: bool = False,
) -> dict:
    """Plan N from LLO differences at metadata-predetermined single anchors.

    Rows contain lineage_id, origin_step, source_recipe, difference, and role
    (development or tuning). The registry is fixed before outcomes and maps
    lineage_id to exactly one source_recipe/origin_step. Other anchors are
    ignored, never averaged. A registered missing anchor is an error.
    """
    if test_outcomes_exist:
        raise ValueError("Precision planning is forbidden after test outcomes")
    if not math.isfinite(effect) or effect <= 0 or bootstrap_repetitions < 1:
        raise ValueError("Positive effect and bootstrap count required")
    registry = {str(k): v for k, v in anchor_registry.items()}
    if len(registry) != len(anchor_registry):
        raise ValueError("Duplicate lineage registry keys")
    selected = {}
    seen = set()
    for row in anchor_rows:
        if row.get("role") not in {"development", "tuning"}:
            raise ValueError("Precision uses only development/tuning LLO rows")
        lineage = str(row["lineage_id"])
        key = (lineage, int(row["origin_step"]))
        if key in seen:
            raise ValueError("Duplicate lineage/anchor; repeats must be averaged first")
        seen.add(key)
        if lineage not in registry:
            raise ValueError(f"Unregistered planning lineage: {lineage}")
        if row["source_recipe"] != registry[lineage]["source_recipe"]:
            raise ValueError("Source recipe disagrees with predetermined registry")
        value = float(row["difference"])
        if not math.isfinite(value) or abs(value) > 1:
            raise ValueError("Finite utility difference in [-1,1] required")
        if cell_key(row) == cell_key(registry[lineage]):
            selected[lineage] = dict(row)
    if set(selected) != set(registry):
        raise ValueError(f"Missing predetermined anchors: {sorted(set(registry) - set(selected))}")
    cells = {
        cell: np.asarray([float(r["difference"]) for r in selected.values() if cell_key(r) == cell])
        for cell in CELLS
    }
    reasons = []
    if len(selected) < 8:
        reasons.append("FEWER_THAN_8_INDEPENDENT_LINEAGES")
    if any(len(values) < 2 for values in cells.values()):
        reasons.append("CELL_VARIANCE_UNIDENTIFIED")
    if selected and all(float(r["difference"]) == 0 for r in selected.values()):
        reasons.append("ALL_OBSERVED_DIFFERENCES_ZERO")
    empirical_sd = None
    if not reasons:
        empirical_sd = math.sqrt(float(np.mean([sample_variance(v) for v in cells.values()])))
        if empirical_sd == 0:
            reasons.append("ZERO_WITHIN_CELL_VARIANCE")
    if reasons:
        s_plan = 0.02
    else:
        rng = np.random.default_rng(seed)
        variances = [
            np.var(
                rng.choice(v, size=(bootstrap_repetitions, len(v)), replace=True), axis=1, ddof=1
            )
            for v in cells.values()
        ]
        s_plan = float(np.quantile(np.sqrt(np.mean(variances, axis=0)), 0.75))
    required = math.ceil((PLANNING_Z * s_plan / effect) ** 2)
    selected_n = next((n for n in N_CHOICES if n >= required), N_CHOICES[-1])
    # Protocol fallback is 32 at the default effect; generalized effects retain
    # the same assumed SD and normal-approximation mapping.
    return {
        "status": "SD_UNIDENTIFIED" if reasons else "SD_ESTIMATED_FOR_PLANNING",
        "sd_source": "ASSUMED_0.02_FALLBACK"
        if reasons
        else "LLO_SINGLE_ANCHOR_FOUR_CELL_BOOTSTRAP_Q75",
        "unidentified_reasons": reasons,
        "n_independent_lineages": len(selected),
        "cell_counts": {f"{c[0]}:{c[1]}": len(v) for c, v in cells.items()},
        "anchor_registry": {k: dict(v) for k, v in registry.items()},
        "effective_sd": empirical_sd,
        "s_plan": s_plan,
        "effect": effect,
        "N_req": required,
        "N": selected_n,
        "n_choices": list(N_CHOICES),
        "target_power_approx": 0.8,
        "alpha": 0.05,
        "target_scale_attainable_approx": required <= selected_n,
        "remaining_mde_approx": PLANNING_Z * s_plan / math.sqrt(selected_n),
        "bootstrap_quantile": 0.75,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": seed,
        "finite_sample_power_guarantee": False,
        "mc": plan_mc_samples(selected_n, effect=effect),
    }
