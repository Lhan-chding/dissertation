# ruff: noqa: RUF001
"""CPU-only, fixed MM-DEV F2 analysis. No model access or outcome-adaptive rules.

The publication entry point requires the execution verifier before reading any
scientific response. Pure functions are independently testable on bounded data.
IDs and endpoint labels remain metadata, never predictor inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from mm_core.scoring import EXACT_CELLS, _field_metrics, json_exact, score_records

from .contract import CELLS, PLAN_ID, STRATA, advantages, run_matrix

STATES = ("S0", "S_A_0", "S_AP_0", "S_A_1", "S_AP_1")
ACTIONS = ("no_train", "a0", "aP", "aC")
FULL_METRICS = ("A_full", "P_full", "J_full")
BOOTSTRAP_SEED = 4311636612181726188
BOOTSTRAP_REPLICATES = 5000
INFERENCE_SCOPE = (
    "empirical-root descriptive intervals conditional on the registered fixed training paths; "
    "not independent-training-population or arbitrary-natural-chart confidence intervals"
)


def number(value: Any) -> float | None:
    """Decode exact scorer serialization without turning missing into zero."""
    if value is None:
        return None
    if isinstance(value, Mapping) and "numerator" in value and "denominator" in value:
        if value["denominator"] == 0:
            raise ValueError("ZERO_RATIONAL_DENOMINATOR")
        result = value["numerator"] / value["denominator"]
    else:
        result = float(Fraction(value)) if isinstance(value, str) else float(value)
    if not math.isfinite(result):
        raise ValueError("NONFINITE_ANALYSIS_INPUT")
    return result


def _mean(values: Sequence[float | None]) -> float | None:
    observed = [x for x in values if x is not None]
    return math.fsum(observed) / len(observed) if observed else None


def _square_float(value: float) -> float | None:
    result = value * value
    return result if math.isfinite(result) else None


def _sidecars(rows: Any, key: str) -> dict:
    if isinstance(rows, Mapping):
        return dict(rows)
    result = {}
    for row in rows or []:
        if row[key] in result:
            raise ValueError(f"DUPLICATE_SIDECAR:{key}")
        result[row[key]] = row
    return result


def score_panel(payload: Mapping[str, Any], *, state_id: str, panel: str) -> dict:
    """Normalize COPIES for the unchanged audit scorer; preserve F2 provenance.

    The evaluation loader is responsible for raw-file/hash/slot verification.
    This boundary rechecks exact completed-slot cardinality and sidecar coverage.
    Gold packets are shared for arithmetic only: unique physical counts are
    separately reported and never inferred from replicated packet references.
    """
    if panel not in {"PROBE", "DEV_EVAL"} or payload.get("complete") is not True:
        raise ValueError("INCOMPLETE_OR_UNREGISTERED_PANEL")
    questions = copy.deepcopy(payload["questions"])
    qmap = {q["question_id"]: q for q in questions}
    if len(qmap) != len(questions):
        raise ValueError("DUPLICATE_PANEL_QUESTION")
    selfs = _sidecars(payload.get("self_scores", []), "request_id")
    golds = _sidecars(payload.get("gold_scores", []), "question_id")
    outputs, seen = [], set()
    for original in payload["outputs"]:
        row = copy.deepcopy(original)
        key = (row["question_id"], row["sample_index"])
        if key in seen or key[0] not in qmap or type(key[1]) is not int or key[1] not in range(4):
            raise ValueError("DUPLICATE_OR_INVALID_COMPLETED_SLOT")
        if row.get("status") != "completed":
            raise ValueError("TECHNICAL_ATTEMPT_NOT_A_COMPLETED_RESPONSE")
        seen.add(key)
        row.update(stage="MEASUREMENT_AUDIT", f2_panel=panel, state_id=state_id)
        if panel == "PROBE":
            if row["request_id"] not in selfs or key[0] not in golds:
                raise ValueError("PROBE_FIELD_SCORES_INCOMPLETE")
            row["self_field_surprisal"] = copy.deepcopy(selfs[row["request_id"]]["field_scores"])
            row["gold_teacher_forced_field_nll"] = copy.deepcopy(golds[key[0]]["field_scores"])
        outputs.append(row)
    if seen != {(qid, k) for qid in qmap for k in range(4)}:
        raise ValueError("MISSING_COMPLETED_PANEL_SLOTS")
    if panel == "PROBE" and (
        set(selfs) != {x["request_id"] for x in outputs} or set(golds) != set(qmap)
    ):
        raise ValueError("ORPHAN_PROBE_FIELD_SCORE")
    if panel == "DEV_EVAL" and (selfs or golds):
        raise ValueError("UNREGISTERED_ENDPOINT_TEACHER_FORCING")
    for q in questions:
        if q["split"] != panel:
            raise ValueError("F2_PANEL_SPLIT_MISMATCH")
        q["split"] = "AUDIT_MEASURE"
    result = score_records(questions, outputs, stage="MEASUREMENT_AUDIT", bootstrap_replicates=0)
    for row in result["scored_outputs"]:
        row["split"] = panel
        row["stage"] = panel
        row["root_index"] = qmap[row["question_id"]]["root_index"]
    if panel == "PROBE":
        # Every gold forward physically happened once per question. The audit
        # scorer received shared packets for root arithmetic; fix measurement
        # denominators on the returned copy using unique question observations.
        unique_gold = [r for r in result["scored_outputs"] if r["sample_index"] == 0]
        gold_key = "gold_teacher_forced_field_nll"
        for cohort in result["tables"]["FIELD_METRICS"]["cohorts"]:
            cohort["overall"][gold_key] = json_exact(_field_metrics(unique_gold)[gold_key])
            for scope in ("chart_operation", "factor_cells"):
                for group in cohort[scope]:
                    keys = ("chart_type", "operation")
                    if scope == "factor_cells":
                        keys = (*keys, "visual_level", "numeric_level")
                    selected = [r for r in unique_gold if all(r[k] == group[k] for k in keys)]
                    group[gold_key] = json_exact(_field_metrics(selected)[gold_key])
            cohort["gold_unit"] = "one_cached_gold_completion_per_question_per_state"
    result.update(
        state_id=state_id,
        panel=panel,
        identity=copy.deepcopy(payload["identity"]),
        questions=copy.deepcopy(payload["questions"]),
        physical_field_measurements={"self": len(selfs), "gold": len(golds)},
        normalized_scorer_channel={"stage": "MEASUREMENT_AUDIT", "split": "AUDIT_MEASURE"},
        normalization_is_not_actual_panel_identity=True,
        technical_events=copy.deepcopy(payload.get("technical_events", [])),
    )
    return result


def panel_cube(scored_rows: Sequence[Mapping], root_count: int) -> np.ndarray:
    """Root x six strata x four V/D cells x (A,P,J), strict K=4."""
    sums = np.zeros((root_count, 6, 4, 3), dtype=np.float64)
    counts = np.zeros((root_count, 6, 4), dtype=np.int64)
    slots, qids, roots = set(), {}, {}
    for row in scored_rows:
        root = row["root_index"]
        if type(root) is not int or root not in range(root_count):
            raise ValueError("INVALID_ROOT_INDEX")
        stratum = STRATA.index((row["chart_type"], row["operation"]))
        cell = CELLS.index(row["visual_level"][0].upper() + row["numeric_level"][0].upper())
        key = (root, stratum, cell)
        slot = (*key, row["sample_index"])
        if slot in slots or row["sample_index"] not in range(4):
            raise ValueError("DUPLICATE_OR_INVALID_ANALYSIS_SLOT")
        slots.add(slot)
        if key in qids and qids[key] != row["question_id"]:
            raise ValueError("MULTIPLE_QUESTIONS_IN_FIXED_CELL")
        qids[key] = row["question_id"]
        if root in roots and roots[root] != row["root_family_id"]:
            raise ValueError("CONFLICTING_ROOT_IDENTITY")
        roots[root] = row["root_family_id"]
        for j, metric in enumerate(FULL_METRICS):
            if row[metric] not in (True, False, 0, 1):
                raise ValueError("INVALID_FULL_INDICATOR")
            sums[(*key, j)] += int(row[metric])
        counts[key] += 1
    if not np.all(counts == 4) or len(set(roots.values())) != root_count:
        raise ValueError("INCOMPLETE_FIXED_ROOT_PANEL")
    return sums / 4


def bootstrap_weights(
    root_count: int, *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    """One shared draw matrix for every model/action/future, preserving pairing."""
    if root_count < 1 or replicates < 1:
        raise ValueError("EMPTY_BOOTSTRAP")
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.integers(0, root_count, size=(replicates, root_count))
    weights = np.zeros((replicates, root_count), dtype=np.float64)
    for i, draw in enumerate(draws):
        weights[i] = np.bincount(draw, minlength=root_count) / root_count
    return weights


def interval(values: np.ndarray, level: float = 0.95) -> dict:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    q = (1 - level) / 2
    return {
        "level": level,
        "lower": float(np.quantile(valid, q, method="linear")) if len(valid) else None,
        "upper": float(np.quantile(valid, 1 - q, method="linear")) if len(valid) else None,
        "defined_replicates": len(valid),
        "undefined_replicates": len(values) - len(valid),
        "quantile_method": "linear",
    }


def utility_values(d: np.ndarray) -> dict:
    """d[..., six strata, metric(A,P,J)]; positive part AFTER stratum averaging."""
    d = np.asarray(d, dtype=float)
    if d.shape[-2:] != (6, 3) or not np.all(np.isfinite(d)):
        raise ValueError("UTILITY_REQUIRES_SIX_FINITE_STRATA")
    gain = d[..., 0].mean(axis=-1)
    interference = np.maximum(-d[..., 1], 0).mean(axis=-1)
    return {
        "G": gain,
        "I": interference,
        "U": gain - interference,
        "feasible": np.all(d[..., 1] >= -0.02, axis=-1),
    }


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _lineage(state: str) -> dict:
    if state == "S0":
        return {
            "origin": "shared_audited_common_start",
            "preparation_recipe": None,
            "preparation_repeat": None,
            "preparation_family": "common_base",
        }
    _, recipe, repeat = state.split("_")
    return {
        "origin": "shared_audited_common_start",
        "preparation_recipe": recipe,
        "preparation_repeat": int(repeat),
        "preparation_family": f"paired_preparation_p{repeat}",
    }


def response_analysis(cubes: Mapping[str, np.ndarray], plan: Mapping) -> dict:
    """Complete absolute/relative cells, paired main effects, utility and crossfit."""
    roots = plan["data"]["root_counts"]["DEV_EVAL"]
    expected = set(STATES) | {r["run_id"] for r in run_matrix() if r["phase"] == "CONTINUE"}
    if set(cubes) != expected or any(v.shape != (roots, 6, 4, 3) for v in cubes.values()):
        raise ValueError("REQUIRES_ALL_35_REGISTERED_MODEL_IDENTITIES")
    stats = plan["statistics"]
    if (
        stats["root_bootstrap_replicates"] != BOOTSTRAP_REPLICATES
        or stats["root_bootstrap_seed"] != BOOTSTRAP_SEED
    ):
        raise ValueError("UNREGISTERED_BOOTSTRAP_RECIPE")
    weights = bootstrap_weights(roots)
    # state x action(including no-train) x future x root x stratum x cell x metric
    absolute = np.zeros((5, 4, 2, roots, 6, 4, 3))
    for s, state in enumerate(STATES):
        for a, action in enumerate(ACTIONS[1:], 1):
            for f in (0, 1):
                absolute[s, a, f] = cubes[f"cont_{state}_{action}_f{f}"] - cubes[state]
    relative = absolute - absolute[:, 1:2]
    records, cells, root_records = [], [], []
    for s, state in enumerate(STATES):
        for a, action in enumerate(ACTIONS):
            for f in (0, 1):
                avg = absolute[s, a, f].mean(axis=0)
                rel = relative[s, a, f].mean(axis=0)
                for r, (chart, operation) in enumerate(STRATA):
                    root_d = absolute[s, a, f, :, r].mean(axis=1)
                    draws = weights @ root_d
                    rec = {
                        **_lineage(state),
                        "start_state": state,
                        "action": action,
                        "future_repeat": f,
                        "chart_type": chart,
                        "operation": operation,
                        "absolute_D": dict(
                            zip(FULL_METRICS, avg[r].mean(axis=0).tolist(), strict=True)
                        ),
                        "relative_Delta_vs_a0": dict(
                            zip(FULL_METRICS, rel[r].mean(axis=0).tolist(), strict=True)
                        ),
                        "absolute_root_CI95": {
                            m: interval(draws[:, j]) for j, m in enumerate(FULL_METRICS)
                        },
                        "root_count": roots,
                    }
                    records.append(rec)
                    for root_index in range(roots):
                        root_records.append(
                            {
                                **_lineage(state),
                                "start_state": state,
                                "action": action,
                                "future_repeat": f,
                                "evaluation_root_index": root_index,
                                "chart_type": chart,
                                "operation": operation,
                                "absolute_D_by_VD_cell": absolute[s, a, f, root_index, r].tolist(),
                                "relative_Delta_by_VD_cell": relative[
                                    s, a, f, root_index, r
                                ].tolist(),
                                "VD_cell_order": list(CELLS),
                                "metric_order": list(FULL_METRICS),
                            }
                        )
                    for c, cell in enumerate(CELLS):
                        cells.append(
                            {
                                **{
                                    k: v
                                    for k, v in rec.items()
                                    if k
                                    not in {
                                        "absolute_D",
                                        "relative_Delta_vs_a0",
                                        "absolute_root_CI95",
                                    }
                                },
                                "VD_cell": cell,
                                "absolute_D": dict(
                                    zip(FULL_METRICS, avg[r, c].tolist(), strict=True)
                                ),
                                "relative_Delta_vs_a0": dict(
                                    zip(FULL_METRICS, rel[r, c].tolist(), strict=True)
                                ),
                            }
                        )
    primary = []
    for m, metric in enumerate(FULL_METRICS[:2]):
        for a in (2, 3):
            root_effect = relative[:, a, :, :, :, :, m].mean(axis=(0, 1, 3, 4))
            draws = weights @ root_effect
            future = relative[:, a, :, :, :, :, m].mean(axis=(0, 2, 3, 4))
            primary.append(
                {
                    "contrast": f"macro_{metric[0]}_{ACTIONS[a]}_minus_a0",
                    "effect_probability": float(root_effect.mean()),
                    "effect_pp": float(root_effect.mean() * 100),
                    "CI95": interval(draws),
                    "CI98_75": interval(draws, 0.9875),
                    "future_effects": future.tolist(),
                    "future_1_minus_0": float(future[1] - future[0]),
                    "future_training_SE": None,
                }
            )
    utilities = []
    for s, state in enumerate(STATES):
        for a, action in enumerate(ACTIONS):
            per_future = absolute[s, a].mean(axis=(1, 3))  # future, stratum, metric
            root_mean_future = absolute[s, a].mean(axis=(0, 3))  # root,stratum,metric
            mean_strata = root_mean_future.mean(axis=0)
            point = utility_values(mean_strata)
            boot_d = np.einsum("br,rsm->bsm", weights, root_mean_future)
            boot_u = utility_values(boot_d)
            ci_p = [interval(boot_d[:, r, 1]) for r in range(6)]
            uncertainty = [
                "above_boundary"
                if q["lower"] >= -0.02
                else "below_boundary"
                if q["upper"] < -0.02
                else "uncertain_crosses_boundary"
                for q in ci_p
            ]
            utilities.append(
                {
                    "start_state": state,
                    "action": action,
                    **_plain(point),
                    "D_A": mean_strata[:, 0].tolist(),
                    "D_P": mean_strata[:, 1].tolist(),
                    "per_future": [_plain(utility_values(x)) for x in per_future],
                    "U_CI95": interval(boot_u["U"]),
                    "D_P_CI95_by_stratum": ci_p,
                    "protection_uncertainty": uncertainty,
                    "all_strata_uncertainty_above_boundary": all(
                        x == "above_boundary" for x in uncertainty
                    ),
                    "optimizer_updates": 0 if action == "no_train" else 32,
                    "shared_probe_cost_is_zero": False,
                    "no_train_relative_Delta_vs_a0": relative[s, a].mean(axis=(0, 1, 3)).tolist()
                    if a == 0
                    else None,
                }
            )
    preparations = []
    for left, right in [(s, "S0") for s in STATES[1:]] + [("S_AP_0", "S_A_0"), ("S_AP_1", "S_A_1")]:
        root_diff = (cubes[left] - cubes[right]).mean(axis=2)
        draws = np.einsum("br,rsm->bsm", weights, root_diff)
        preparations.append(
            {
                "left": left,
                "right": right,
                "paired_preparation_family": _lineage(left)["preparation_family"],
                "stratum_effects": root_diff.mean(axis=0).tolist(),
                "macro_effects": root_diff.mean(axis=(0, 1)).tolist(),
                "macro_CI95": {
                    m: interval(draws[:, :, j].mean(axis=1)) for j, m in enumerate(FULL_METRICS)
                },
            }
        )
    return {
        "primary": {
            "contrasts": primary,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "rng": "numpy.Generator(PCG64)",
            "resample_unit": "whole_root_all_24_cells_K_models_actions_futures",
            "scope": INFERENCE_SCOPE,
            "practical_attention_pp": 1,
            "scientific_pass_threshold": None,
        },
        "responses": records,
        "root_responses": root_records,
        "changes": cells,
        "utilities": {
            "lambda": 1,
            "epsilon": 0.02,
            "eta": 0,
            "rows": utilities,
            "scope": INFERENCE_SCOPE,
            "safety_guarantee": False,
        },
        "preparations": {
            "metric_order": FULL_METRICS,
            "stratum_order": STRATA,
            "comparisons": preparations,
            "independent_base_count": 1,
        },
        "opportunity": opportunity_analysis(absolute, plan),
        "absolute": absolute,
    }


def _opportunity(d: np.ndarray, constrained: bool) -> dict:
    # d: state,action,stratum,metric; no-train is always action index zero.
    u = utility_values(d)
    allowed = u["feasible"] if constrained else np.ones_like(u["feasible"], dtype=bool)
    selected = np.argmax(np.where(allowed, u["U"], -np.inf), axis=1)
    fixed_allowed = allowed.all(axis=0)
    fixed = int(np.argmax(np.where(fixed_allowed, u["U"].mean(axis=0), -np.inf)))
    state_value = float(u["U"][np.arange(5), selected].mean())
    fixed_value = float(u["U"][:, fixed].mean())
    return {
        "selected_action_indices": selected,
        "fixed_action_index": fixed,
        "state_selected_U": state_value,
        "fixed_U": fixed_value,
        "gap": state_value - fixed_value,
        "fixed_eligible_actions": [ACTIONS[i] for i in np.flatnonzero(fixed_allowed)],
    }


def opportunity_analysis(absolute: np.ndarray, plan: Mapping) -> dict:
    halves = plan["statistics"]["crossfit_root_halves"]
    h0, h1 = halves["half0_indices"], halves["half1_indices"]
    n = absolute.shape[3]
    if len(h0) != len(h1) or len(set(h0 + h1)) != n or set(h0 + h1) != set(range(n)):
        raise ValueError("INVALID_FROZEN_CROSSFIT_HALVES")
    result = {
        "exploratory_only": True,
        "pretraining_selector": False,
        "deployable_policy_proven": False,
        "tie_order": list(ACTIONS),
        "selection_and_fixed_baseline_share_splits": True,
        "in_sample": {},
        "crossfit": {},
        "ranking_by_future": [],
    }
    pooled = absolute.mean(axis=(2, 3, 5))
    for constrained in (False, True):
        key = "constrained" if constrained else "unconstrained"
        inside = _opportunity(pooled, constrained)
        result["in_sample"][key] = _plain(inside)
        folds = []
        for train_f, test_f, train_h, test_h in ((0, 1, h0, h1), (1, 0, h1, h0)):
            train = absolute[:, :, train_f][:, :, train_h].mean(axis=(2, 4))
            test = absolute[:, :, test_f][:, :, test_h].mean(axis=(2, 4))
            choice = _opportunity(train, constrained)
            u_test = utility_values(test)
            selected, fixed = choice["selected_action_indices"], choice["fixed_action_index"]
            value = float(u_test["U"][np.arange(5), selected].mean())
            fixed_value = float(u_test["U"][:, fixed].mean())
            folds.append(
                {
                    "selection_future": train_f,
                    "evaluation_future": test_f,
                    "selection_root_indices": train_h,
                    "evaluation_root_indices": test_h,
                    "selected_actions": [ACTIONS[x] for x in selected],
                    "fixed_action": ACTIONS[fixed],
                    "evaluation_state_selected_U": value,
                    "evaluation_fixed_U": fixed_value,
                    "evaluation_gap": value - fixed_value,
                    "evaluation_selected_feasible": u_test["feasible"][
                        np.arange(5), selected
                    ].tolist(),
                    "evaluation_fixed_feasible": u_test["feasible"][:, fixed].tolist(),
                    "evaluation_infeasible_values_retained": True,
                }
            )
        result["crossfit"][key] = {
            "folds": folds,
            "mean_gap": _mean([f["evaluation_gap"] for f in folds]),
            "safety_guarantee": False,
        }
    for s, state in enumerate(STATES):
        future_u = utility_values(absolute[s].mean(axis=(2, 4)))["U"]  # action,future
        ranks = [[ACTIONS[i] for i in np.argsort(-future_u[:, f], kind="stable")] for f in (0, 1)]
        result["ranking_by_future"].append(
            {
                "start_state": state,
                "rankings": ranks,
                "identical_order": ranks[0] == ranks[1],
                "utility_by_action_future": future_u.tolist(),
            }
        )
    return result


def _hierarchical(rows: Sequence[Mapping], value) -> dict:
    """Question means -> equally weighted supported cells -> supported roots.

    Fixed-panel mean exists only if every planned question/cell has support.
    The support mask is common to BNum and BGeo, never silently imputed.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["root_family_id"], row["question_id"])].append(row)
    roots = defaultdict(list)
    valid_responses = 0
    for (root, _), group in grouped.items():
        vals = [value(r) for r in group]
        valid_responses += sum(v is not None for v in vals)
        roots[root].append(_mean(vals))
    root_means = {root: _mean(vals) for root, vals in sorted(roots.items())}
    complete = all(all(v is not None for v in vals) for vals in roots.values())
    return {
        "conditional_root_equal_mean": _mean(list(root_means.values())),
        "complete_fixed_weight_mean": _mean(list(root_means.values())) if complete else None,
        "supported_root_count": sum(v is not None for v in root_means.values()),
        "total_root_count": len(roots),
        "supported_question_count": sum(v is not None for vs in roots.values() for v in vs),
        "total_question_count": len(grouped),
        "valid_response_count": valid_responses,
        "missing_response_count": len(rows) - valid_responses,
        "root_values": root_means,
        "root_supported_cells": {
            root: sum(v is not None for v in vals) for root, vals in sorted(roots.items())
        },
    }


def _conditional(rows: Sequence[Mapping], metric: str) -> dict:
    return _hierarchical(rows, lambda r: float(r[metric]) if r["L"] else None)


def _field_value(row: Mapping, source: str, field: str, metric: str) -> float | None:
    packet = row.get(source)
    if not packet or packet.get("status") != "MEASURED":
        return None
    return number(packet["field_nll"][field].get(metric))


def _shape_diagnostics(rows: Sequence[Mapping], weights: np.ndarray | None = None) -> dict:
    shape = _hierarchical(rows, lambda r: number(r["S"]))
    groups = defaultdict(list)
    single = []
    for row in rows:
        e = [number(v) for v in row["e"]] if row["L_P"] else []
        nonzero = sum(v != 0 for v in e)
        if row["S"] is not None:
            groups[(len(e), nonzero)].append(number(row["S"]))
            if nonzero == 1:
                single.append(abs(number(row["S"]) - 1 / len(e)))
    s_values = np.array([np.nan if v is None else v for v in shape["root_values"].values()])
    if weights is not None:
        supported = np.isfinite(s_values)
        mass = weights @ supported.astype(float)
        sums = weights @ np.nan_to_num(s_values, nan=0)
        draws = np.divide(sums, mass, out=np.full_like(sums, np.nan), where=mass > 0)
        shape["conditional_root_CI95"] = interval(draws)
    return {
        "S": shape,
        "by_arity_nonzero_coordinates": [
            {
                "arity": arity,
                "nonzero_coordinates": nz,
                "responses": len(values),
                "S_mean": _mean(values),
                "S_min": min(values),
                "S_max": max(values),
            }
            for (arity, nz), values in sorted(groups.items())
        ],
        "single_coordinate_count": len(single),
        "single_coordinate_expected_S": "1/arity",
        "single_coordinate_max_identity_residual": max(single) if single else None,
        "one_coordinate_fraction_among_S_defined": len(single) / shape["valid_response_count"]
        if shape["valid_response_count"]
        else None,
        "incremental_predictive_efficacy_established": False,
    }


def pretrain_features(probe: Mapping, history: Mapping | None = None) -> dict:
    """Nested information ladder; only PROBE of a registered starting state.

    No endpoint/model/path/seed/job ID is inside an input feature dictionary.
    Metadata preserves lineage for future grouped validation, outside inputs.
    """
    state = probe["state_id"]
    if state not in STATES or probe["panel"] != "PROBE":
        raise ValueError("ENDPOINT_FEATURE_LEAKAGE")
    rows = probe["scored_outputs"]
    result = {
        "state_id": state,
        "lineage": _lineage(state),
        "panel": "PROBE",
        "identity": probe["identity"],
        "label_source": "NONE",
        "information_sets_nested": True,
        "strata": [],
        "physical_field_measurements": probe["physical_field_measurements"],
        "gold_prefix_contains_gold_readings": True,
        "gold_packets_reused_for_question_arithmetic_not_extra_calls": True,
    }
    # Explicit allowlist prevents training-history identifiers becoming features.
    hist = {
        k: copy.deepcopy((history or {}).get(k))
        for k in (
            "preparation_recipe",
            "committed_updates",
            "exposure_by_stratum_cell",
            "recent_reward_support",
            "recent_gradient_summary",
        )
    }
    for chart, op in STRATA:
        selected = [r for r in rows if (r["chart_type"], r["operation"]) == (chart, op)]
        b0 = {
            "chart_type": chart,
            "operation": op,
            "arity": 3 if op == "range" else 2,
            "preparation_history": hist,
        }
        b0.update(
            {
                m: _hierarchical(selected, lambda r, key=m: float(r[key]))
                for m in ("A_full", "P_full", "L_P", "L_A", "L")
            }
        )
        b0.update({m.lower(): _conditional(selected, m) for m in ("A", "P", "C")})
        for source in ("self_field_surprisal", "gold_teacher_forced_field_nll"):
            field_rows = (
                [r for r in selected if r["sample_index"] == 0]
                if source.startswith("gold")
                else selected
            )
            b0[source] = {
                field: {
                    metric: _hierarchical(
                        field_rows,
                        lambda r, so=source, fi=field, me=metric: _field_value(r, so, fi, me),
                    )
                    for metric in ("mean_nll", "mean_token_entropy")
                }
                for field in ("readings", "answer")
            }
        bj = {"j": _hierarchical(selected, lambda r: float(r["J_full"]) if r["L"] else None)}
        btab = {
            "exact_six_cell_probabilities_conditional_L": {
                cell: _hierarchical(
                    selected, lambda r, c=cell: float(r["cell_exact"] == c) if r["L"] else None
                )
                for cell in EXACT_CELLS
            },
            "algebraic_dependencies": (
                "p=n111+n100; a=n111+n011+n001; c=n111+n011+n010; "
                "j=n111; z=n001 (rates conditional on L)"
            ),
        }
        arity = 3 if op == "range" else 2
        bnum = {
            "roles": [],
            "near_miss": _hierarchical(
                selected, lambda r: float(r["near_miss"]) if r["L_P"] else None
            ),
            "max_abs_error": _hierarchical(
                selected, lambda r: max(abs(number(v)) for v in r["e"]) if r["L_P"] else None
            ),
            "nonzero_coordinate_count": _hierarchical(
                selected, lambda r: float(sum(number(v) != 0 for v in r["e"])) if r["L_P"] else None
            ),
            "error_present": _hierarchical(
                selected, lambda r: float(not r["P_full"]) if r["L_P"] else None
            ),
            "S_support_indicator": _hierarchical(selected, lambda r: float(r["S"] is not None)),
            "arity": arity,
            "missing_readings": _hierarchical(selected, lambda r: float(not r["L_P"])),
            "nonzero_coordinate_distribution": {
                str(n): _hierarchical(
                    selected,
                    lambda r, k=n: (
                        float(sum(number(v) != 0 for v in r["e"]) == k) if r["L_P"] else None
                    ),
                )
                for n in range(arity + 1)
            },
        }
        for role in range(arity):
            metrics = {"role_index": role}
            for key, transform in (("bias", lambda x: x), ("MAE", abs), ("MSE", _square_float)):
                metrics[key] = _hierarchical(
                    selected,
                    lambda r, i=role, f=transform: f(number(r["e"][i])) if r["L_P"] else None,
                )
            mse = metrics["MSE"]["conditional_root_equal_mean"]
            metrics["RMSE"] = math.sqrt(mse) if mse is not None else None
            metrics["MSE_float_overflow_responses"] = sum(
                r["L_P"] and _square_float(number(r["e"][role])) is None for r in selected
            )
            metrics["overflow_policy"] = "float summary missing; exact rational geometry retained"
            bnum["roles"].append(metrics)
        # Expose exactly the same support/missing metadata at both numeric tiers.
        support = _hierarchical(selected, lambda r: number(r["S"]))
        bnum["shape_support"] = {
            k: v
            for k, v in support.items()
            if k not in {"conditional_root_equal_mean", "complete_fixed_weight_mean", "root_values"}
        }
        bgeo = {"S": support}
        result["strata"].append(
            {
                "chart_type": chart,
                "operation": op,
                "inputs": {
                    "B0": b0,
                    "BJ_addition": bj,
                    "BTab_addition": btab,
                    "BNum_addition": bnum,
                    "BGeo_S_addition": bgeo,
                },
            }
        )

    # Root identifiers remain audit metadata, never dimensions of a predictor.
    def separate_root_diagnostics(value, path, removed):
        if isinstance(value, dict):
            for key in list(value):
                if key in {"root_values", "root_supported_cells"}:
                    removed["/".join((*path, key))] = value.pop(key)
                else:
                    separate_root_diagnostics(value[key], (*path, key), removed)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                separate_root_diagnostics(item, (*path, str(i)), removed)

    result["non_feature_root_diagnostics"] = {}
    for row in result["strata"]:
        separate_root_diagnostics(
            row["inputs"],
            (row["chart_type"], row["operation"]),
            result["non_feature_root_diagnostics"],
        )
    result["fixed_low_dimensional_candidate"] = {
        "candidate": "six stratum conditional root-equal S means",
        "conditioning": "complete BNum including common support, missingness, arity and amplitude",
        "fitted": False,
        "controller_trained": False,
    }
    return result


def training_support(
    rollouts: Mapping[str, Sequence[Mapping]], updates: Mapping[str, Sequence[Mapping]]
) -> dict:
    """Recompute reward support on each path's own fixed groups, never resample.

    Canonical rollout adapter keys: logical_step, question_id, sample_index,
    rA, q_read, reward, chart_type, operation, visual_level, numeric_level.
    Actual advantage may be supplied; reward normalization is always recounted.
    """
    paths = []
    for run in run_matrix():
        run_id = run["run_id"]
        groups = defaultdict(list)
        for row in rollouts.get(run_id, []):
            groups[(row["logical_step"], row["question_id"])].append(row)
        groups_out, exposure = [], Counter()
        for (step, qid), group in sorted(groups.items()):
            if len(group) != 8 or {r["sample_index"] for r in group} != set(range(8)):
                raise ValueError(f"INCOMPLETE_TRAINING_REWARD_GROUP:{run_id}:{step}:{qid}")
            group = sorted(group, key=lambda r: r["sample_index"])
            exact_a = [Fraction(r["rA"]) for r in group]
            exact_ap = [
                (a + Fraction(r["q_read"])) / 2 for a, r in zip(exact_a, group, strict=True)
            ]
            exact_used = [Fraction(r["reward"]) for r in group]
            expected = exact_ap if run.get("preparation_recipe") == "AP" else exact_a
            if exact_used != expected:
                raise ValueError("RECORDED_REWARD_RECIPE_MISMATCH")
            aa, ap, actual = advantages(exact_a), advantages(exact_ap), advantages(exact_used)
            ra, rap, used = [list(map(float, values)) for values in (exact_a, exact_ap, exact_used)]
            for row, adv in zip(group, actual, strict=True):
                if "advantage" in row and not math.isclose(
                    number(row["advantage"]), adv, abs_tol=1e-7, rel_tol=1e-7
                ):
                    raise ValueError("RECORDED_ADVANTAGE_MISMATCH")
            d_adv = math.sqrt(sum((x - y) ** 2 for x, y in zip(aa, ap, strict=True)) / 8)
            first = group[0]
            label = {
                k: first[k]
                for k in ("chart_type", "operation", "visual_level", "numeric_level")
                if k in first
            }
            exposure[json.dumps(label, sort_keys=True)] += 1
            groups_out.append(
                {
                    "logical_step": step,
                    "question_id": qid,
                    **label,
                    "rA": ra,
                    "rA_exact": list(map(str, exact_a)),
                    "rAP_counterfactual": rap,
                    "rAP_counterfactual_exact": list(map(str, exact_ap)),
                    "used_reward": used,
                    "used_reward_exact": list(map(str, exact_used)),
                    "advantages": actual,
                    "reward_mean": _mean(used),
                    "reward_population_std": float(np.std(used)),
                    "zero_contrast": len(set(used)) == 1,
                    "d_adv": d_adv,
                    "d_adv_above_1e_6": d_adv > 1e-6,
                    "all_correct": all(a == 1 for a in ra),
                    "all_wrong": all(a == 0 for a in ra),
                    "format_failure_fraction": _mean(
                        [float(not r["format_ok"]) if "format_ok" in r else None for r in group]
                    ),
                    "truncated_fraction": _mean(
                        [float(r["truncated"]) if "truncated" in r else None for r in group]
                    ),
                }
            )
        stratified = []
        for chart, operation in STRATA:
            selected = [
                g
                for g in groups_out
                if (g.get("chart_type"), g.get("operation")) == (chart, operation)
            ]
            stratified.append(
                {
                    "chart_type": chart,
                    "operation": operation,
                    "groups": len(selected),
                    "zero_contrast_fraction": _mean([float(g["zero_contrast"]) for g in selected]),
                    "d_adv_above_1e_6_fraction": _mean(
                        [float(g["d_adv_above_1e_6"]) for g in selected]
                    ),
                }
            )
        path_updates = updates.get(run_id, [])
        paths.append(
            {
                "run_id": run_id,
                "phase": run["phase"],
                "preparation_recipe": run.get("preparation_recipe"),
                "groups": groups_out,
                "group_count": len(groups_out),
                "expected_group_count": 32 * 24,
                "zero_contrast_fraction": _mean([float(g["zero_contrast"]) for g in groups_out]),
                "d_adv_above_1e_6_fraction": _mean(
                    [float(g["d_adv_above_1e_6"]) for g in groups_out]
                ),
                "by_stratum": stratified,
                "exposure_by_stratum_cell": [
                    {**json.loads(k), "prompt_groups": v} for k, v in sorted(exposure.items())
                ],
                "recorded_update_count": len(path_updates),
                "updates": copy.deepcopy(path_updates),
                "missing_training_records": not groups_out or not path_updates,
            }
        )
    return {
        "paths": paths,
        "counterfactual_reward_is_arithmetic_on_own_trajectory_only": True,
        "zero_contrast_groups_retained": True,
        "scientific_logical_updates_expected": 1088,
        "engineering_excluded": True,
    }


def load_training_records(run_root: Path) -> tuple[dict, dict]:
    """Read committed native trainer diagnostics, not uncommitted attempts."""
    root = Path(run_root)
    rollouts, updates = {}, {}
    for run in run_matrix():
        run_id = run["run_id"]
        rows, steps = [], []
        for step in range(1, 33):
            path = root / "training" / run_id / "steps" / f"{step:02d}.json"
            packet = json.loads(path.read_text())
            if packet["logical_step"] != step or len(packet["groups"]) != 24:
                raise ValueError("INCOMPLETE_COMMITTED_STEP_DIAGNOSTICS")
            steps.append({k: v for k, v in packet.items() if k != "groups"})
            seen_slots = set()
            for group in packet["groups"]:
                if group["slot"] in seen_slots or group["slot"] not in range(24):
                    raise ValueError("INVALID_COMMITTED_GROUP_SLOTS")
                seen_slots.add(group["slot"])
                if not all(
                    len(group[k]) == 8 for k in ("completion_rewards", "rewards", "advantages")
                ):
                    raise ValueError("INVALID_COMMITTED_GROUP_CARDINALITY")
                cell = group["cell"]
                for i, reward in enumerate(group["completion_rewards"]):
                    rows.append(
                        {
                            "logical_step": step,
                            "question_id": group["question_id"],
                            "sample_index": i,
                            "chart_type": group["chart_type"],
                            "operation": group["operation"],
                            "visual_level": "low" if cell[0] == "L" else "high",
                            "numeric_level": "low" if cell[1] == "L" else "high",
                            "rA": reward["rA"],
                            "q_read": reward["q_read"],
                            "reward": group["rewards"][i],
                            "advantage": group["advantages"][i],
                            "format_ok": reward["format_valid"],
                        }
                    )
        rollouts[run_id], updates[run_id] = rows, steps
    return rollouts, updates


def load_cost_accounting(run_root: Path) -> dict:
    """All attempts retained; reservations never relabeled successful work."""
    from .evaluation import read_segment
    from .orchestration import digest, summarize_accounting

    paths = sorted((Path(run_root) / "accounting").glob("*.jsonl"))
    records, categories, ledger_tails = [], defaultdict(Counter), []
    scientific = {r["run_id"] for r in run_matrix()}
    for path in paths:
        entries, tails = read_segment(path, Path(run_root))
        ledger_tails.extend(tails)
        for row in entries:
            count = number(row["count"])
            if count is None or count < 0:
                raise ValueError("INVALID_PHYSICAL_COST")
            task = row["task_id"]
            category = (
                "scientific_training"
                if task in scientific
                else "engineering"
                if "engine" in task.lower()
                else "evaluation_or_other"
            )
            categories[category][row["kind"]] += count
            records.append(row)
    allocation_paths = sorted((Path(run_root) / "allocations").rglob("*.json"))
    final_path = Path(run_root) / "manifests/GPU_ACCOUNTING_FINAL.json"
    final = json.loads(final_path.read_text())
    registration = json.loads((Path(run_root) / "orchestration/REGISTRATION.json").read_text())
    state = json.loads((Path(run_root) / "orchestration/STATE.json").read_text())
    gpu_tasks = {k: v for k, v in registration["tasks"].items() if v["gpus"] > 0}
    if (
        final.get("status") != "COMPLETE"
        or final.get("policy") != "ACCOUNTING_ONLY"
        or final.get("plan_id") != PLAN_ID
        or registration.get("plan_id") != PLAN_ID
        or final.get("registration_hash") != digest(registration)
        or not gpu_tasks
        or final.get("gpu_task_count") != len(gpu_tasks)
    ):
        raise ValueError("MISSING_COMPLETE_GPU_ACCOUNTING")
    gpu_seconds = number(final["actual_gpu_seconds"])
    allocations = final["allocations"]
    expected_attempts = {}
    for task_id in gpu_tasks:
        task = state["tasks"][task_id]
        if task["status"] != "COMPLETE" or not task["attempts"]:
            raise ValueError("GPU_TASK_COST_INCOMPLETE")
        for attempt in task["attempts"]:
            key = (task_id, attempt["attempt_id"], attempt["job_id"])
            if key in expected_attempts:
                raise ValueError("DUPLICATE_GPU_ATTEMPT")
            expected_attempts[key] = attempt
    actual_keys = [(a["task_id"], a["attempt_id"], a["job_id"]) for a in allocations]
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != set(expected_attempts):
        raise ValueError("GPU_COST_ATTEMPT_COVERAGE_MISMATCH")
    if gpu_seconds < 0 or not math.isclose(number(final["actual_gpu_hours"]), gpu_seconds / 3600):
        raise ValueError("INVALID_GPU_COST_TOTAL")
    if not math.isclose(sum(number(a["gpu_seconds"]) for a in allocations), gpu_seconds):
        raise ValueError("GPU_ACCOUNTING_TOTAL_MISMATCH")
    for allocation in allocations:
        evidence = allocation["scheduler_observation"]
        source = (Path(run_root) / evidence["path"]).resolve(strict=True)
        if not source.is_relative_to(Path(run_root).resolve()):
            raise ValueError("GPU_ACCOUNTING_EVIDENCE_OUTSIDE_RUN")
        if hashlib.sha256(source.read_bytes()).hexdigest() != evidence["sha256"]:
            raise ValueError("GPU_ACCOUNTING_SOURCE_HASH_MISMATCH")
        attempt = expected_attempts[
            (allocation["task_id"], allocation["attempt_id"], allocation["job_id"])
        ]
        observation = json.loads(source.read_text())
        if observation["queue"]:
            raise ValueError("GPU_COST_NOT_TERMINAL")
        recounted = summarize_accounting(
            observation["accounting"],
            attempt,
            registration["permission"],
            gpu_tasks[allocation["task_id"]]["gpus"],
        )
        if any(allocation.get(k) != v for k, v in recounted.items()):
            raise ValueError("GPU_COST_RAW_RECOUNT_MISMATCH")
    return {
        "ledger_kind_totals_by_scope": {k: dict(v) for k, v in categories.items()},
        "ledger_rows": records,
        "ledger_count": len(records),
        "ledger_file_count": len(paths),
        "ledger_measure": "physical_attempt_or_reservation_not_successful_sequence_count",
        "retained_ledger_tail_events": ledger_tails,
        "source_files": [
            {
                "path": str(p.relative_to(run_root)),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in paths
        ],
        "allocation_evidence": [
            {"path": str(p.relative_to(run_root)), "record": json.loads(p.read_text())}
            for p in allocation_paths
        ],
        "actual_allocated_gpu_seconds": gpu_seconds,
        "actual_allocated_gpu_hours": gpu_seconds / 3600,
        "gpu_seconds_status": "TERMINAL_SCHEDULER_RECONCILED",
        "gpu_accounting_receipt": final,
        "gpu_accounting_receipt_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "queue_seconds": None,
        "wrapper_wall_seconds": None,
        "missing_time_is_not_zero": True,
        "saved_token_accounting": saved_token_accounting(Path(run_root), records),
    }


def saved_token_accounting(root: Path, ledger: Sequence[Mapping]) -> dict:
    """Stream persisted generations once; no sidecar or logical replay inflation."""
    from .evaluation import read_segment

    counts = defaultdict(Counter)
    observed, fingerprints, tails = {}, hashlib.sha256(), []

    def account(row, scope, location):
        # Persistence identity, distinct from a physical call counter in ledger.
        key = row.get("request_id") or json.dumps(
            [
                row.get(k)
                for k in (
                    "run_id",
                    "logical_step",
                    "slot",
                    "sample_index",
                    "policy_hash",
                    "question_id",
                )
            ]
        )
        key = (scope, key)
        body = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        identity = hashlib.sha256(body).hexdigest()
        fingerprints.update(location.encode() + b"\0" + identity.encode() + b"\n")
        counts[scope]["saved_physical_records"] += 1
        if key in observed:
            if observed[key] != identity:
                raise ValueError("CONFLICTING_SAVED_GENERATION_IDENTITY")
            counts[scope]["identical_persisted_records_deduplicated"] += 1
            return
        observed[key] = identity
        counts[scope]["unique_saved_generations"] += 1
        tokens = row.get("tokens")
        if not isinstance(tokens, list):
            counts[scope]["completion_token_records_unknown"] += 1
        else:
            count = row.get("completion_token_count", len(tokens))
            if count != len(tokens):
                raise ValueError("SAVED_COMPLETION_TOKEN_COUNT_MISMATCH")
            counts[scope]["known_completion_tokens"] += len(tokens)
            counts[scope]["completion_token_records_known"] += 1
        prompt = row.get("prompt_token_count")
        if type(prompt) is int and prompt >= 0:
            counts[scope]["known_prompt_tokens"] += prompt
            counts[scope]["prompt_token_records_known"] += 1
        else:
            counts[scope]["prompt_token_records_unknown"] += 1
        routing = row.get("image_routing", {})
        vision = routing.get("generation_vision_forward_calls")
        if type(vision) is int and vision >= 0:
            counts[scope]["known_generation_vision_calls"] += vision
        else:
            counts[scope]["generation_vision_call_records_unknown"] += 1

    for base in (root / "training", root / "engineering"):
        for path in sorted(base.rglob("rollouts/*.json")):
            scope = str(path.parent.parent.relative_to(root))
            account(json.loads(path.read_text()), scope, str(path.relative_to(root)))
    for path in sorted((root / "raw").rglob("outputs_*.jsonl")):
        entries, tail_events = read_segment(path, root)
        tails.extend(tail_events)
        for lineno, row in enumerate(entries, 1):
            scope = f"{row.get('panel', row.get('stage'))}/{row.get('state_id')}"
            account(row, scope, f"{path.relative_to(root)}:{lineno}")
    forwards = defaultdict(Counter)
    for row in ledger:
        if row["kind"] == "extra_forward_sequences":
            purpose = row.get("metadata", {}).get("purpose", "UNKNOWN")
            forwards[purpose]["registered_forward_sequences"] += row["count"]
            tokens = row.get("metadata", {}).get("completion_tokens")
            if type(tokens) is int and tokens >= 0:
                forwards[purpose]["registered_target_tokens"] += tokens
            else:
                forwards[purpose]["target_token_records_unknown"] += 1
    return {
        "by_scope": {k: dict(v) for k, v in sorted(counts.items())},
        "saved_source_inventory_sha256": fingerprints.hexdigest(),
        "retained_generation_tail_events": tails,
        "extra_forwards_by_purpose": {k: dict(v) for k, v in sorted(forwards.items())},
        "registered_forwards_are_not_successful_calls": True,
        "uncaptured_failed_generation_tokens": None,
        "unknown_failed_tokens_not_imputed_from_max_new_tokens": True,
        "physical_attempt_counts_remain_in_ledger": True,
    }


def build_artifacts(
    *,
    plan: Mapping,
    evaluations: Mapping[str, Mapping],
    probes: Mapping[str, Mapping],
    completeness: Mapping,
    training_rollouts: Mapping[str, Sequence[Mapping]],
    training_updates: Mapping[str, Sequence[Mapping]],
    cost_accounting: Mapping,
) -> dict[str, Any]:
    """Build all scientific files in memory only after a trusted readiness gate."""
    if set(probes) != set(STATES):
        raise ValueError("REQUIRES_ALL_FIVE_PROBES")
    if completeness.get("status") != "COMPLETE":
        raise ValueError("COMPLETE_EXECUTION_RECEIPT_REQUIRED")
    if (
        plan["utility"]["lambda"] != 1
        or plan["utility"]["epsilon_per_protected_stratum"] != 0.02
        or plan["utility"]["eta"] != 0
    ):
        raise ValueError("UNREGISTERED_UTILITY")
    cubes = {
        state: panel_cube(p["scored_outputs"], plan["data"]["root_counts"]["DEV_EVAL"])
        for state, p in evaluations.items()
    }
    # Prevent apparently rectangular but differently ordered root panels.
    canonical_questions = None
    for panel in evaluations.values():
        layout = sorted(
            (q["root_index"], q["root_family_id"], q["question_id"]) for q in panel["questions"]
        )
        if canonical_questions is not None and layout != canonical_questions:
            raise ValueError("UNPAIRED_EVALUATION_ROOT_IDENTITIES")
        canonical_questions = layout
    responses = response_analysis(cubes, plan)
    root_identities = {q["root_index"]: q["root_family_id"] for q in evaluations["S0"]["questions"]}
    for row in responses["root_responses"]:
        row["evaluation_root_family_id"] = root_identities[row["evaluation_root_index"]]
    support = training_support(training_rollouts, training_updates)
    if any(
        p["group_count"] != 768
        or p["recorded_update_count"] != 32
        or {u["logical_step"] for u in p["updates"]} != set(range(1, 33))
        for p in support["paths"]
    ):
        raise ValueError("INCOMPLETE_ALL_34_TRAINING_DIAGNOSTICS")
    history = {}
    for path in support["paths"]:
        if path["phase"] == "PREP":
            run = next(r for r in run_matrix() if r["run_id"] == path["run_id"])
            recent = path["updates"][-8:]
            recent_groups = [g for g in path["groups"] if g["logical_step"] > 24]
            history[run["output_state"]] = {
                "preparation_recipe": run["preparation_recipe"],
                "committed_updates": path["recorded_update_count"],
                "exposure_by_stratum_cell": path["exposure_by_stratum_cell"],
                "recent_reward_support": {
                    "logical_steps": [25, 32],
                    "group_count": len(recent_groups),
                    "zero_contrast_fraction": _mean(
                        [float(g["zero_contrast"]) for g in recent_groups]
                    ),
                    "d_adv_above_1e_6_fraction": _mean(
                        [float(g["d_adv_above_1e_6"]) for g in recent_groups]
                    ),
                },
                "recent_gradient_summary": {
                    key: _mean([number(r.get(key)) for r in recent])
                    for key in (
                        "policy_gradient_norm",
                        "total_gradient_norm_before_clip",
                        "policy",
                        "kl",
                        "clip_fraction",
                    )
                },
            }
    history["S0"] = {
        "preparation_recipe": "NONE",
        "committed_updates": 0,
        "exposure_by_stratum_cell": [],
        "recent_reward_support": None,
        "recent_gradient_summary": None,
    }
    features, geometry, joints, context = [], [], [], []
    for panel_name, panels in (("PROBE", probes), ("DEV_EVAL", evaluations)):
        for state, panel in panels.items():
            table = panel["tables"]
            ident = {"state_id": state, "panel": panel_name, "identity": panel["identity"]}
            joints.append(
                {
                    **ident,
                    "counts": table["COUNTS"],
                    "coverage": table["COVERAGE"],
                    "format": table["FORMAT_REPORT"],
                }
            )
            nroots = plan["data"]["root_counts"][panel_name]
            geometry.append(
                {
                    **ident,
                    "numeric_geometry": table["GEOMETRY"],
                    "root_equal": table["ROOT_EQUAL"],
                    "shape_support_and_redundancy": _shape_diagnostics(
                        panel["scored_outputs"], bootstrap_weights(nroots)
                    ),
                    "field_metrics": table["FIELD_METRICS"]
                    if panel_name == "PROBE"
                    else {"status": "NOT_REGISTERED_FOR_ENDPOINTS"},
                    "physical_field_measurements": panel["physical_field_measurements"],
                }
            )
            context.append({**ident, "audit": table["CONTEXT_AUDIT"], "primary_BGeo_input": False})
            if panel_name == "PROBE":
                panel_cube(panel["scored_outputs"], nroots)
                features.append(pretrain_features(panel, history.get(state)))
    files = {
        "summary/RUN_COMPLETENESS.json": dict(completeness),
        "summary/PRIMARY_EFFECTS.json": responses["primary"],
        "summary/RESPONSE_BY_STATE_ACTION_FUTURE_STRATUM.jsonl": responses["responses"],
        "summary/ABSOLUTE_AND_RELATIVE_CHANGES.jsonl": responses["changes"],
        "summary/RESPONSE_BY_ROOT.jsonl": responses["root_responses"],
        "summary/SECONDARY_EVENT_AND_NUMERIC_CHANGES.jsonl": secondary_changes(evaluations),
        "summary/PREPARATION_STATE_DIFFERENCES.json": responses["preparations"],
        "summary/REWARD_AND_ADVANTAGE_SUPPORT.json": support,
        "summary/JOINT_EVENT_TABLES.json": joints,
        "summary/NUMERIC_AND_GEOMETRY_SUPPORT.json": geometry,
        "summary/CONTEXT_AUDIT.json": context,
        "summary/UTILITY_AND_FEASIBILITY.json": responses["utilities"],
        "summary/EXPLORATORY_OPPORTUNITY_CROSSFIT.json": responses["opportunity"],
        "summary/ACTUAL_COST_ACCOUNTING.json": {
            "accounting": dict(cost_accounting),
            "expected_counts": plan["expected_counts"],
            "gpu_hours_policy": "ACCOUNTING_ONLY",
            "engineering_separate_from_scientific": True,
            "missing_is_unknown_not_zero": True,
        },
        "features/PRETRAIN_FEATURES_B0_BJ_BTAB_BNUM_BGEO.jsonl": features,
        "summary/FINAL_MM_DEV_REPORT_zh.md": scientific_report(responses, support),
        "summary/NEXT_STAGE_RECOMMENDATION_zh.md": next_stage_report(responses),
    }
    for panel_name, panels in (("PROBE", probes), ("DEV_EVAL", evaluations)):
        for state_id, panel in panels.items():
            if panel["state_id"] != state_id or any(
                r["state_id"] != state_id for r in panel["scored_outputs"]
            ):
                raise ValueError("SCORED_STATE_BINDING_MISMATCH")
            files[f"scoring/{panel_name}/{state_id}/SCORED_OUTPUTS.jsonl"] = panel["scored_outputs"]
    return _plain(files)


def secondary_changes(evaluations: Mapping[str, Mapping]) -> list[dict]:
    """Descriptive C/six-cell/numeric changes with changing support exposed."""
    summaries = {}
    for state, panel in evaluations.items():
        summaries[state] = {}
        for chart, operation in STRATA:
            rows = [
                r
                for r in panel["scored_outputs"]
                if (r["chart_type"], r["operation"]) == (chart, operation)
            ]
            metrics = {"C_given_L": _conditional(rows, "C")}
            metrics.update(
                {
                    f"cell_{c}_given_L": _hierarchical(
                        rows, lambda r, cell=c: float(r["cell_exact"] == cell) if r["L"] else None
                    )
                    for c in EXACT_CELLS
                }
            )
            for key in ("eps_P", "eps_C", "eps_O", "S"):
                metrics[key] = _hierarchical(rows, lambda r, k=key: number(r[k]))
            metrics["near_miss"] = _hierarchical(
                rows, lambda r: float(r["near_miss"]) if r["L_P"] else None
            )
            metrics["max_abs_error"] = _hierarchical(
                rows, lambda r: max(abs(number(e)) for e in r["e"]) if r["L_P"] else None
            )
            for role in range(3 if operation == "range" else 2):
                for key, transform in (("bias", lambda x: x), ("MAE", abs)):
                    metrics[f"role_{role}_{key}"] = _hierarchical(
                        rows,
                        lambda r, i=role, f=transform: f(number(r["e"][i])) if r["L_P"] else None,
                    )
            summaries[state][(chart, operation)] = metrics
    result = []
    for run in run_matrix():
        if run["phase"] != "CONTINUE":
            continue
        start = run["start_state"]
        comparator = f"cont_{start}_a0_f{run['repeat']}"
        for stratum in STRATA:
            changes = {}
            for metric, after in summaries[run["run_id"]][stratum].items():
                before, a0 = (
                    summaries[start][stratum][metric],
                    summaries[comparator][stratum][metric],
                )
                x, y, z = [r["conditional_root_equal_mean"] for r in (after, before, a0)]
                changes[metric] = {
                    "starting": before,
                    "endpoint": after,
                    "a0_endpoint": a0,
                    "absolute_D": x - y if x is not None and y is not None else None,
                    "relative_Delta_vs_a0": x - z if x is not None and z is not None else None,
                }
            result.append(
                {
                    **_lineage(start),
                    "start_state": start,
                    "action": run["action"],
                    "future_repeat": run["repeat"],
                    "chart_type": stratum[0],
                    "operation": stratum[1],
                    "secondary_not_independent_primary_endpoints": True,
                    "changes": changes,
                }
            )
    return result


def scientific_report(responses: Mapping, support: Mapping) -> str:
    lines = [
        "# MM-DEV F2 完整开发报告",
        "",
        "本轮检验固定多模态训练动作在不同准备起点上的绝对响应和相对均衡续训的替换效应。",
        "",
        "完成固定的四条32步准备、五个起点PROBE、三动作×两未来重复×五起点的30条32步续训及35模型DEV_EVAL。工程分支与科学训练分开。",
        "",
        "准备状态相对S0及配对A/AP差异见 PREPARATION_STATE_DIFFERENCES.json；"
        "五个状态名称不等于五个独立基座来源。",
        "",
        "| 准备比较 | 宏A变化(pp) | 宏P变化(pp) | 宏J变化(pp) |",
        "|---|---:|---:|---:|",
    ]
    for row in responses["preparations"]["comparisons"]:
        values = " | ".join(f"{100 * v:.4f}" for v in row["macro_effects"])
        lines.append(f"| {row['left']} - {row['right']} | {values} |")
    lines += [
        "",
        "## 四个预登记主比较",
        "",
        "| 对比 | 效应(pp) | 95%根簇区间(pp) | 98.75%根簇区间(pp) | f0/f1(pp) |",
        "|---|---:|---|---|---|",
    ]
    for x in responses["primary"]["contrasts"]:
        bounds = [
            f"[{x[k]['lower'] * 100:.4f}, {x[k]['upper'] * 100:.4f}]" for k in ("CI95", "CI98_75")
        ]
        lines.append(
            f"| {x['contrast']} | {x['effect_pp']:.4f} | {bounds[0]} | {bounds[1]} | "
            f"{x['future_effects'][0] * 100:.4f} / {x['future_effects'][1] * 100:.4f} |"
        )
    lines += [
        "",
        "这些是条件于实际固定训练路径的经验根簇描述区间；不是训练总体或任意自然图表总体的保证。1pp只是预登记关注量级，不是科学PASS门槛。",
        "",
        "全部起点×动作×未来重复×六层和24格分别见 "
        "RESPONSE_BY_STATE_ACTION_FUTURE_STRATUM.jsonl 与 "
        "ABSOLUTE_AND_RELATIVE_CHANGES.jsonl。D以该起点为锚，"
        "Δ以同未来重复a0为锚；no-train的Δ为−D(a0)。",
        "",
        "| 起点 | 动作 | 未来 | D宏A(pp) | D宏P(pp) | Δ宏A(pp) | Δ宏P(pp) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    grouped = defaultdict(list)
    for row in responses["responses"]:
        if row["action"] != "no_train":
            grouped[(row["start_state"], row["action"], row["future_repeat"])].append(row)
    for (state, action, future), rows in sorted(grouped.items()):
        vals = [
            100 * _mean([r[key][metric] for r in rows])
            for key in ("absolute_D", "relative_Delta_vs_a0")
            for metric in ("A_full", "P_full")
        ]
        text = " | ".join(f"{x:.4f}" for x in vals)
        lines.append(f"| {state} | {action} | {future} | {text} |")
    lines += [
        "",
        "## 奖励与语义支持",
        "",
        f"训练组诊断共保留 {sum(p['group_count'] for p in support['paths'])} 个组。"
        "零对比、all-correct、all-wrong和不利结果全部保留；"
        "rA/rAP反事实只在本路径真实轨迹上算数。",
        "",
        "JOINT_EVENT_TABLES.json 分开full指标、条件精确六格和容差八格；"
        "NUMERIC_AND_GEOMETRY_SUPPORT.json 保留误差、arity、非零坐标数、S的支持与缺失。"
        "S零误差不补0，单坐标误差的S=1/arity不能算作额外形状发现。"
        "SECONDARY_EVENT_AND_NUMERIC_CHANGES.jsonl 保留次要事件/数值变化及变化前后的支持。",
        "",
        "CONTEXT_AUDIT.json 保留跨提示及同提示collision和负噪声校正差值；"
        "它不进入主BGeo-S，也不识别内部推理机制。"
        "PROBE字段NLL/熵区分self与gold前缀，gold物理测量每题仅一次。",
        "",
        "## 效用与状态选择空间",
        "",
        "效用使用概率单位、λ=1、η=0；先平均两未来重复，再逐层取读数损害正部。保护边界每层−0.02，点估计及区间分别报告，不能宣称安全保证。",
        "有限样本下负部惩罚与max动作选择会产生估计偏差；bootstrap不消除全部选择偏差。",
        "",
    ]
    for key, v in responses["opportunity"]["crossfit"].items():
        lines.append(
            f"- {key} 双重交叉探索读数：相对匹配拆分固定动作的平均差 "
            f"{100 * v['mean_gap']:.4f}pp效用单位。"
        )
    lines += [
        "",
        "选择同时跨未来重复和冻结根半；固定动作基线使用同一拆分。此结果利用开发响应挑动作，只是事后机会诊断，不是训练前选择器或已实现策略。两个未来重复只报告各自效果与差值，不报告可靠训练方差。",
        "",
        "成本见 ACTUAL_COST_ACCOUNTING.json：科学、工程、失败、重做与GPU分配秒分别记账；"
        "GPU小时不作为停止门槛。",
        "",
        "本轮到固定合同完成为止。不自动延长H、调奖励、筛难题或启动MM-LOCK/MM-ONLINE/MM-CAL。"
        "后续建议见 NEXT_STAGE_RECOMMENDATION_zh.md。",
        "",
    ]
    return "\n".join(lines)


def next_stage_report(responses: Mapping) -> str:
    ranks = responses["opportunity"]["ranking_by_future"]
    stable = sum(r["identical_order"] for r in ranks)
    tops = {r["rankings"][f][0] for r in ranks for f in (0, 1)}
    observation = (
        "当前固定动作的首位排名一致，自适应选择空间可能有限。"
        if len(tops) == 1
        else "开发状态间出现不同首位动作，需要结合两未来重复稳定性判断。"
    )
    return (
        "# 下一阶段建议（未授权执行）\n\n"
        + observation
        + f" 五个状态中有 {stable} 个的两未来重复完整动作排序一致。\n\n"
        "先审查准备状态多样性、奖励优势支持、S支持与单坐标/arity冗余、完整BNum信息，以及经验根簇区间。稀疏支持或宽区间意味着本采样通道/剂量/精度下证据不足，不意味着语义信息普遍无效。\n\n"
        "若仍值得独立验证，应另行预登记新准备来源、按配对准备家系分组的训练/验证划分、完整BNum对BGeo-S候选、样本量及精度，并在揭晓新响应前提交预测和动作。不得以本轮五状态拟合认证新状态泛化；不得事后修改λ/ε、奖励或动作。\n\n"
        "MM-LOCK、在线控制、安全校准、奖励搜索和新符号训练均未启动，也不由本报告授权。\n"
    )


def write_artifacts(run_root: Path, files: Mapping[str, Any]) -> dict:
    """Immutable publication, byte-identical retries, manifest committed last."""
    root = Path(run_root).resolve(strict=True)
    model_ids = set(STATES) | {r["run_id"] for r in run_matrix() if r["phase"] == "CONTINUE"}

    def checked_output(name):
        relative = Path(name)
        path = root / relative
        ordinary = len(relative.parts) == 2 and relative.parts[0] in {"summary", "features"}
        scored = (
            len(relative.parts) == 4
            and relative.parts[0] == "scoring"
            and relative.parts[1] in {"PROBE", "DEV_EVAL"}
            and relative.parts[2] in model_ids
            and relative.parts[3] == "SCORED_OUTPUTS.jsonl"
        )
        if not (ordinary or scored) or any(
            (root / Path(*relative.parts[:i])).is_symlink()
            for i in range(1, len(relative.parts) + 1)
        ):
            raise ValueError("INVALID_ANALYSIS_OUTPUT_PATH")
        return path

    def serialize(name, value):
        if name.endswith(".jsonl"):
            data = "".join(
                json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                for r in value
            )
        elif name.endswith(".json"):
            data = (
                json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            )
        else:
            data = value
        return data.encode("utf-8")

    receipts = []
    for name, value in sorted(files.items()):
        path, encoded = checked_output(name), serialize(name, value)
        if path.exists() and path.read_bytes() != encoded:
            raise FileExistsError("DIFFERENT_ANALYSIS_ARTIFACT_ALREADY_PUBLISHED:" + name)
        receipts.append(
            {"path": name, "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
        )
    manifest = {
        "status": "ANALYZED",
        "files": receipts,
        "scientific_success_claimed": False,
        "next_stage_authorized": False,
    }
    published = {**files, "summary/ANALYSIS_MANIFEST.json": manifest}
    manifest_path = checked_output("summary/ANALYSIS_MANIFEST.json")
    manifest_bytes = serialize("summary/ANALYSIS_MANIFEST.json", manifest)
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise FileExistsError("DIFFERENT_ANALYSIS_MANIFEST_ALREADY_PUBLISHED")
    for name, value in published.items():
        path, encoded = checked_output(name), serialize(name, value)
        if path.exists():
            if path.read_bytes() != encoded:
                raise FileExistsError("ANALYSIS_ARTIFACT_CHANGED_DURING_PUBLICATION")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()
    return manifest
