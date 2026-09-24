"""Frozen-rule H32 comparisons, conditional on the fixed evaluation panel.

No function selects recipes from endpoint outcomes. Physical outcomes are
looked up only after the caller supplies registered, prospective decisions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass

import numpy as np
from scipy.stats import t

from .precision import CELLS, cell_key, sample_variance

PRIMARY = ("Z3_REPAIR_STRUCTURE", "Z2_REWARD_DISTRIBUTION")
SECONDARY = ("best_static", "GDPO_R4", "SAW_R4", "DIRECT_REPAIR_R4")


def _mapping(value) -> dict:
    return asdict(value) if is_dataclass(value) else dict(value)


def aggregate_lineage_differences(
    decisions: Sequence[Mapping],
    outcomes: Sequence[Mapping],
    registry: Mapping[str, Mapping],
    *,
    left: str = PRIMARY[0],
    right: str = PRIMARY[1],
    repeats: Sequence[int] = (1, 2),
    panel_id: str = "T",
    require_real: bool = True,
) -> list[dict]:
    """Average registered future repeats before counting independent lineages.

    One decision per lineage has ``selections: {method: recipe_id}``, and
    ``made_before_run: True``. Outcomes have a scalar ``J`` or ``metrics.J``.
    Every required physical result must exist, including same-action choices;
    equal actions reference the exact same object and produce exact zero.
    Duplicate physical outcome keys are rejected instead of silently counted.
    """
    if not repeats or len(set(repeats)) != len(repeats):
        raise ValueError("Distinct registered repeats required")
    registered = {str(k): v for k, v in registry.items()}
    if not registered:
        raise ValueError("Final analysis requires a nonempty frozen lineage registry")
    decision_by_lineage = {}
    for raw in decisions:
        row = _mapping(raw)
        lineage = str(row["lineage_id"])
        if lineage in decision_by_lineage:
            raise ValueError("Duplicate lineage decision")
        if lineage not in registered or cell_key(row) != cell_key(registered[lineage]):
            raise ValueError("Decision disagrees with frozen lineage/anchor registry")
        if row.get("made_before_run") is not True:
            raise ValueError("Analysis requires prospective decisions made before runs")
        decision_by_lineage[lineage] = row
    if set(decision_by_lineage) != set(registered):
        raise ValueError("Missing registered lineage decisions")
    index = {}
    for raw in outcomes:
        row = _mapping(raw)
        if row.get("panel_id") != panel_id or row.get("horizon") != 32:
            continue  # H8 and development panels are never optimized here.
        lineage = str(row["lineage_id"])
        if lineage not in registered or cell_key(row) != cell_key(registered[lineage]):
            raise ValueError("Outcome disagrees with frozen lineage/anchor registry")
        if row.get("continuation_repeat") not in repeats:
            raise ValueError("Unregistered future repeat")
        if require_real and row.get("execution_kind") != "REAL_CUDA_MODEL":
            raise ValueError("Final scientific analysis requires REAL_CUDA_MODEL outcomes")
        if int(row.get("n_draws", 0)) < 1:
            raise ValueError("Outcome requires a positive evaluated draw count")
        value = float(row["J"] if "J" in row else row.get("metrics", {})["J"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Nonfinite or out-of-range H32 utility")
        key = (
            lineage,
            int(row["origin_step"]),
            str(row["recipe_id"]),
            int(row["continuation_repeat"]),
        )
        if key in index:
            raise ValueError(f"Duplicate physical outcome: {key}")
        index[key] = value
    result = []
    for lineage, decision in decision_by_lineage.items():
        selections = decision["selections"]
        if left not in selections or right not in selections:
            raise ValueError(f"Missing precommitted comparison selection: {left}/{right}")
        a, b = selections[left], selections[right]
        differences, a_values, b_values, keys = [], [], [], []
        for repeat in repeats:
            ka = (lineage, int(decision["origin_step"]), str(a), int(repeat))
            kb = (lineage, int(decision["origin_step"]), str(b), int(repeat))
            if ka not in index or kb not in index:
                raise ValueError(f"Missing registered H32 outcome: {ka if ka not in index else kb}")
            va, vb = index[ka], index[kb]
            a_values.append(va)
            b_values.append(vb)
            differences.append(0.0 if a == b else va - vb)
            keys.append({"repeat": int(repeat), "left": list(ka), "right": list(kb)})
        result.append(
            {
                "lineage_id": lineage,
                "origin_step": int(decision["origin_step"]),
                "source_recipe": decision["source_recipe"],
                "left_method": left,
                "right_method": right,
                "left_recipe": a,
                "right_recipe": b,
                "left_mean": float(np.mean(a_values)),
                "right_mean": float(np.mean(b_values)),
                "difference": float(np.mean(differences)),
                "repeat_differences": differences,
                "same_action": a == b,
                "physical_outcome_references": keys,
                "n_future_repeats": len(repeats),
            }
        )
    return result


def four_cell_comparison(
    lineage_rows: Sequence[Mapping],
    *,
    alpha: float = 0.05,
    equivalence_margin: float = 0.01,
    bootstrap_repetitions: int = 10000,
    seed: int = 75002,
) -> dict:
    """Equal four-cell means and Welch-Satterthwaite paired-lineage interval.

    The block bootstrap resamples each whole lineage within its cell; repeats,
    actions and measures are already kept together in the lineage contrast.
    It performs no additional draw/token resampling.
    """
    if (
        not 0 < alpha < 1
        or not math.isfinite(equivalence_margin)
        or equivalence_margin <= 0
        or bootstrap_repetitions < 1
    ):
        raise ValueError("Invalid interval parameters")
    cells = {cell: [] for cell in CELLS}
    seen = set()
    for row in lineage_rows:
        lineage = str(row["lineage_id"])
        if lineage in seen:
            raise ValueError("Duplicate lineage: anchors/repeats are not independent units")
        seen.add(lineage)
        value = float(row["difference"])
        if not math.isfinite(value) or abs(value) > 1:
            raise ValueError("Finite utility difference in [-1,1] required")
        if row.get("same_action") and value != 0:
            raise ValueError("Same-action selections must have exact zero difference")
        cells[cell_key(row)].append(value)
    arrays = {c: np.asarray(v, dtype=float) for c, v in cells.items()}
    cell_reports = {
        f"{c[0]}:{c[1]}": {
            "n": len(v),
            "mean": float(v.mean()) if len(v) else None,
            "variance": sample_variance(v) if len(v) >= 2 else None,
        }
        for c, v in arrays.items()
    }
    result = {
        "n_independent_lineages": len(seen),
        "cell_statistics": cell_reports,
        "cell_weights": [0.25] * 4,
        "estimate": None,
        "standard_error": None,
        "degrees_of_freedom": None,
        "interval": None,
        "p_value": None,
        "alpha": alpha,
        "equivalence_margin": equivalence_margin,
        "approximately_equivalent": False,
        "bootstrap_interval": None,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": seed,
        "bootstrap_unit": "whole lineage within source/stage cell",
        "resample_tokens_or_draws": False,
        "finite_sample_guarantee": False,
        "estimand": "frozen-rule H32 selection difference conditional on fixed T",
        "scene_generalization_included": False,
        "all_selected_actions_identical": bool(lineage_rows)
        and all(r.get("same_action") is True for r in lineage_rows),
    }
    if any(len(v) == 0 for v in arrays.values()):
        return {
            **result,
            "status": "INCOMPLETE_CELLS",
            "interpretation": "Missing registered cell; no complete primary estimate.",
        }
    estimate = float(np.mean([v.mean() for v in arrays.values()]))
    result["estimate"] = estimate
    if any(len(v) < 2 for v in arrays.values()):
        return {
            **result,
            "status": "INSUFFICIENT_LINEAGES",
            "interpretation": "Within-cell variance is unidentified.",
        }
    rng = np.random.default_rng(seed)
    bootstrap_means = np.mean(
        [
            rng.choice(v, size=(bootstrap_repetitions, len(v)), replace=True).mean(axis=1)
            for v in arrays.values()
        ],
        axis=0,
    )
    result["bootstrap_interval"] = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2]).tolist()
    components = np.asarray([0.25**2 * sample_variance(v) / len(v) for v in arrays.values()])
    variance = float(components.sum())
    result["standard_error"] = math.sqrt(variance)
    if variance == 0:
        all_zero = all(np.all(v == 0) for v in arrays.values())
        return {
            **result,
            "status": "NO_OBSERVED_DIFFERENCE" if all_zero else "ZERO_VARIANCE_UNRESOLVED",
            "interpretation": (
                "Observed action choices coincide; this does not prove equivalence "
                "of all future policies."
            )
            if result["all_selected_actions_identical"]
            else (
                "Empirical zero variance does not remove fixed-panel MC or future-lineage "
                "uncertainty; no t-based equivalence claim."
            ),
        }
    df = variance**2 / sum(
        float(v) ** 2 / (len(arrays[c]) - 1) for c, v in zip(CELLS, components, strict=True)
    )
    half_width = float(t.ppf(1 - alpha / 2, df)) * math.sqrt(variance)
    interval = [estimate - half_width, estimate + half_width]
    p_value = float(2 * t.sf(abs(estimate / math.sqrt(variance)), df))
    equivalent = interval[0] >= -equivalence_margin and interval[1] <= equivalence_margin
    direction = (
        "POSITIVE_DIFFERENCE"
        if interval[0] > 0
        else "NEGATIVE_DIFFERENCE"
        if interval[1] < 0
        else "DIRECTION_UNRESOLVED"
    )
    return {
        **result,
        "status": direction,
        "degrees_of_freedom": df,
        "interval": interval,
        "p_value": p_value,
        "approximately_equivalent": equivalent,
        "interpretation": (
            "Approximate equivalence at the registered margin for this fixed-panel estimand."
        )
        if equivalent
        else "Interval excludes zero."
        if direction != "DIRECTION_UNRESOLVED"
        else "Insufficient evidence to determine the sign; crossing zero is not equivalence.",
    }


def holm_adjust(p_values: Mapping[str, float | None], *, alpha: float = 0.05) -> dict:
    """Holm familywise correction; unresolved p-values retain their family slot."""
    if not 0 < alpha < 1 or not p_values:
        raise ValueError("Nonempty p-value family and valid alpha required")
    for p in p_values.values():
        if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError("p-values must be finite in [0,1] or unresolved None")
    ordered = sorted(
        p_values, key=lambda name: (1 if p_values[name] is None else p_values[name], name)
    )
    adjusted, running = {}, 0.0
    for index, name in enumerate(ordered):
        p = p_values[name]
        running = max(running, min(1.0, (len(ordered) - index) * (1.0 if p is None else p)))
        adjusted[name] = {
            "p_value": p,
            "holm_p_value": None if p is None else running,
            "reject": p is not None and running <= alpha,
            "status": "UNRESOLVED" if p is None else "COMPUTED",
        }
    return {"alpha": alpha, "family_size": len(ordered), "comparisons": adjusted}


def analyze_final(
    decisions: Sequence[Mapping],
    outcomes: Sequence[Mapping],
    registry: Mapping[str, Mapping],
    *,
    panel_id: str = "T",
    require_real: bool = True,
    bootstrap_repetitions: int = 10000,
) -> dict:
    """Primary and four predeclared secondary comparisons, with honest status.

    Caller must validate freeze/decision provenance before invoking; the
    statistics layer additionally checks complete registered outcomes and
    explicit prospective decision flags. No result-based action is chosen.
    """
    common = dict(panel_id=panel_id, require_real=require_real)
    primary_rows = aggregate_lineage_differences(decisions, outcomes, registry, **common)
    primary = four_cell_comparison(primary_rows, bootstrap_repetitions=bootstrap_repetitions)
    secondary = {}
    for name in SECONDARY:
        rows = aggregate_lineage_differences(decisions, outcomes, registry, right=name, **common)
        secondary[name] = {
            "lineage_rows": rows,
            **four_cell_comparison(rows, bootstrap_repetitions=bootstrap_repetitions),
        }
    holm = holm_adjust({name: value["p_value"] for name, value in secondary.items()})
    primary_positive = primary["status"] == "POSITIVE_DIFFERENCE"
    all_controls_positive = all(
        secondary[name]["estimate"] is not None
        and secondary[name]["estimate"] > 0
        and holm["comparisons"][name]["reject"]
        for name in SECONDARY
    )
    return {
        "status": "ANALYZED_REAL_OUTCOMES" if require_real else "FIXTURE_ANALYSIS_NOT_EXPERIMENT",
        "primary": primary,
        "primary_lineage_rows": primary_rows,
        "secondary": secondary,
        "secondary_holm": holm,
        "extra_repair_beats_full_reward_information": primary_positive,
        "extra_repair_beats_all_registered_controls": all_controls_positive,
        "prospective_incremental_value_supported": require_real
        and primary_positive
        and all_controls_positive,
        "conclusion": (
            "Evidence supports incremental H32 selection value on unseen registered "
            "lineages conditional on fixed T."
        )
        if require_real and primary_positive and all_controls_positive
        else (
            "The complete claim of incremental value over full reward information "
            "and all registered controls is not established."
        ),
        "online_closed_loop_validated": False,
        "safety_certified": False,
    }
