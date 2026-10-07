"""Prespecified root-paired SER-J2 estimands, with fixed macro weights."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist
from typing import Any

import numpy as np

PARENTS = ("S96", "REP96")
ARMS = ("A_LOCAL_C1", "B_FORWARD_C4", "C_FORWARD_C3")
METRICS = ("X", "S", "W", "I", "value_present", "value_present_but_not_X")
BOOTSTRAP_SEED = 107079
BOOTSTRAP_REPLICATES = 5000


class IncompleteDataError(ValueError):
    """A missing registered outcome cannot be replaced with an observed subset."""


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("Invalid conditional counts")
    return numerator / denominator if denominator else None


def wilson(successes: int, draws: int, confidence: float = 0.95) -> list[float]:
    if (
        type(successes) is not int
        or type(draws) is not int
        or draws < 1
        or not 0 <= successes <= draws
    ):
        raise ValueError("Invalid binomial counts")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between zero and one")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, den = successes / draws, 1 + z * z / draws
    middle = (p + z * z / (2 * draws)) / den
    half = z * math.sqrt(p * (1 - p) / draws + z * z / (4 * draws * draws)) / den
    return [max(0.0, middle - half), min(1.0, middle + half)]


def pass_at_k(successes: int, draws: int, k: int) -> float:
    if (
        any(type(v) is not int for v in (successes, draws, k))
        or not 0 <= successes <= draws
        or not 1 <= k <= draws
    ):
        raise ValueError("Invalid finite-stream pass@k parameters")
    return (
        1.0
        if draws - successes < k
        else 1.0 - math.comb(draws - successes, k) / math.comb(draws, k)
    )


def _probabilities(values, shape):
    p = np.asarray(values, dtype=float)
    if p.shape != shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError(f"Complete probability array {shape} required")
    return p


def _effect(values: np.ndarray, samples: np.ndarray, *, bonferroni=False) -> dict[str, Any]:
    # values axes [two parents, three schedules, 128 paired roots]
    roots = values.mean(axis=(0, 1))
    boot = roots[samples].mean(axis=1)
    blocks = values.mean(axis=2)
    result = {
        "estimate": float(values.mean()),
        "ci95_root_conditional": np.quantile(boot, [0.025, 0.975]).tolist(),
        "per_parent_schedule": blocks.tolist(),
        "per_parent": values.mean(axis=(1, 2)).tolist(),
        "per_parent_schedule_sd": blocks.std(axis=1, ddof=1).tolist(),
        "per_parent_schedule_range": [[float(x.min()), float(x.max())] for x in blocks],
    }
    if bonferroni:
        result["ci98_75_bonferroni_four_secondary"] = np.quantile(boot, [0.00625, 0.99375]).tolist()
    return result


def primary_effect(
    probabilities,
    *,
    parent_probabilities=None,
    bootstrap_replicates=BOOTSTRAP_REPLICATES,
    seed=BOOTSTRAP_SEED,
):
    """Grid [parent2,schedule3,arm(A/B/C),root128,hub(3/4)]."""
    p = _probabilities(probabilities, (2, 3, 3, 128, 2))
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("Positive bootstrap replicate count required")
    samples = np.random.default_rng(seed).integers(0, 128, size=(bootstrap_replicates, 128))
    d4, d3 = p[:, :, 1, :, 1] - p[:, :, 2, :, 1], p[:, :, 1, :, 0] - p[:, :, 2, :, 0]
    z = (d4 - d3) / 2
    gamma = _effect(z, samples)
    secondary = {
        "B4-A4": _effect(p[:, :, 1, :, 1] - p[:, :, 0, :, 1], samples, bonferroni=True),
        "C3-A3": _effect(p[:, :, 2, :, 0] - p[:, :, 0, :, 0], samples, bonferroni=True),
        "B3-A3": _effect(p[:, :, 1, :, 0] - p[:, :, 0, :, 0], samples, bonferroni=True),
        "C4-A4": _effect(p[:, :, 2, :, 1] - p[:, :, 0, :, 1], samples, bonferroni=True),
    }
    parent_contrasts = {}
    if parent_probabilities is not None:
        baseline = _probabilities(parent_probabilities, (2, 128, 2))
        for arm_index, arm in enumerate(ARMS):
            for hub_index, hub in enumerate((3, 4)):
                parent_contrasts[f"{arm}:c{hub}j{hub}-PARENT"] = _effect(
                    p[:, :, arm_index, :, hub_index] - baseline[:, None, :, hub_index], samples
                )
    variance_bound = 1 / (4 * 8 * 6 * 128)
    return {
        "Gamma": gamma["estimate"],
        "Delta4": float(d4.mean()),
        "Delta3": float(d3.mean()),
        "ci95_root_conditional": gamma["ci95_root_conditional"],
        "per_parent_schedule_Gamma": gamma["per_parent_schedule"],
        "per_parent_Gamma": gamma["per_parent"],
        "Gamma_detail": gamma,
        "Delta4_detail": _effect(d4, samples),
        "Delta3_detail": _effect(d3, samples),
        "secondary": secondary,
        "relative_to_parent": parent_contrasts,
        "parent_order": list(PARENTS),
        "block_order": [0, 1, 2],
        "arm_order": list(ARMS),
        "roots": 128,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "condition": "two fixed parents, three actual schedules per parent, fixed 32 donor roots",
        "uncertainty_unit": "128 paired numeric roots; all arms/parents/schedules move together",
        "mc_variance_upper_bound": variance_bound,
        "mc95_halfwidth_upper_bound": 1.959963984540054 * math.sqrt(variance_bound),
        "mc_bound_scope": (
            "independent output sampling only; excludes root and training-path uncertainty"
        ),
        "practical_reference_pp": 5,
        "reference_is_not_gate": True,
    }


def cross_macro(cell_probabilities):
    if set(cell_probabilities) != {(c, j) for c in range(4) for j in range(4)}:
        raise ValueError("All 16 cross cells required")
    values = _probabilities(list(cell_probabilities.values()), (16,))
    return float(values.mean())


def trend_macro(cell_probabilities):
    if set(cell_probabilities) != set(range(4)):
        raise ValueError("All four trend positions required")
    return float(_probabilities(list(cell_probabilities.values()), (4,)).mean())


def summarize_tasks(rows):
    """Each checkpoint/task keeps its own stream, including invalid answers."""
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["checkpoint_id"], row["panel"], row["task_id"])
        draw_key = (*key, row["draw_index"])
        if draw_key in seen:
            raise ValueError("Duplicate formal generation identity")
        seen.add(draw_key)
        groups[key].append(row)
    result = []
    fields = (
        "checkpoint_id",
        "parent",
        "block",
        "arm",
        "step",
        "panel",
        "task_id",
        "root_id",
        "family",
        "center",
        "corrupted_index",
        "root_cohort",
    )
    for group in groups.values():
        first = group[0]
        if any(any(row.get(k) != first.get(k) for k in fields) for row in group):
            raise ValueError("Conflicting task or checkpoint metadata")
        if any(
            int(row["X"]) != int(row["value_present"]) - int(row["value_present_but_not_X"])
            or sum(bool(row[x]) for x in "XSWI") != 1
            for row in group
        ):
            raise ValueError("Semantic decomposition or event partition violated")
        item = {k: first.get(k) for k in fields}
        item["n_draws"] = len(group)
        item["draw_indices"] = sorted(r["draw_index"] for r in group)
        for metric in METRICS:
            item[metric + "_count"] = sum(bool(r[metric]) for r in group)
            item[metric] = item[metric + "_count"] / len(group)
        item["X_wilson95"] = wilson(item["X_count"], len(group))
        item["pass_at_1"] = pass_at_k(item["X_count"], len(group), 1)
        item["pass_at_8"] = pass_at_k(item["X_count"], len(group), 8) if len(group) >= 8 else None
        result.append(item)
    return result


def _checkpoint_key(row):
    return row["parent"], row["block"], row["arm"]


def _complete_confirmation(task_rows, expected_draws):
    expected = {(s, b, a) for s in PARENTS for b in range(3) for a in ARMS}
    expected |= {(s, None, "PARENT") for s in PARENTS}
    groups = defaultdict(dict)
    for row in task_rows:
        if row["panel"] != "E_CONFIRM":
            continue
        key = _checkpoint_key(row)
        if key not in expected or (row["arm"] != "PARENT" and row["step"] != 256):
            raise ValueError("Unregistered E_CONFIRM checkpoint")
        if row["task_id"] in groups[key]:
            raise ValueError("Multiple checkpoints for the same formal endpoint")
        groups[key][row["task_id"]] = row
    if set(groups) != expected:
        raise IncompleteDataError(
            f"Confirmation requires 18 students and two parents; present={len(groups)}"
        )
    reference = next(iter(groups.values()))
    if len(reference) != 800 or any(set(g) != set(reference) for g in groups.values()):
        raise IncompleteDataError(
            "All 20 endpoints must contain the same complete 800-task confirmation panel"
        )
    for group in groups.values():
        for task_id, row in group.items():
            if row["n_draws"] != expected_draws or row["draw_indices"] != list(
                range(expected_draws)
            ):
                raise IncompleteDataError(
                    "Every confirmation task requires the exact eight formal draws"
                )
            if any(
                row[field] != reference[task_id][field]
                for field in ("root_id", "family", "center", "corrupted_index", "root_cohort")
            ):
                raise ValueError("Checkpoint panels disagree on task/root identity")
    root_cells = defaultdict(set)
    cohorts = defaultdict(set)
    for row in reference.values():
        root_cells[row["root_id"]].add((row["family"], row["center"], row["corrupted_index"]))
        cohorts[row["root_cohort"]].add(row["root_id"])
    if {k: len(v) for k, v in cohorts.items()} != {
        "CORE32": 32,
        "EXTRA96": 96,
        "trend": 64,
        "duplicate_encoding": 32,
    }:
        raise IncompleteDataError(
            "Confirmation root cohorts must be CORE32/EXTRA96/trend64/duplicate32"
        )
    all_roots = [root for values in cohorts.values() for root in values]
    if len(all_roots) != len(set(all_roots)):
        raise ValueError("One numeric root assigned to multiple cohorts")
    for root, cells in root_cells.items():
        if root in cohorts["CORE32"]:
            wanted = {("cross_series", c, j) for c in range(4) for j in range(4)}
        elif root in cohorts["EXTRA96"]:
            wanted = {("cross_series", 2, 2), ("cross_series", 3, 3)}
        else:
            wanted = cells
            if len(cells) != 1 or next(iter(cells))[1] is not None:
                raise ValueError("Malformed trend/duplicate root")
        if cells != wanted:
            raise IncompleteDataError("Incomplete within-root structural pairing")
    for family, count in (("trend", 16), ("duplicate_encoding", 8)):
        if any(
            sum(r["family"] == family and r["corrupted_index"] == j for r in reference.values())
            != count
            for j in range(4)
        ):
            raise IncompleteDataError("Incomplete trend/duplicate position allocation")
    return groups, cohorts


def _primary_grid(groups, roots, metric):
    grid, parent = np.empty((2, 3, 3, 128, 2)), np.empty((2, 128, 2))
    rid = {root: i for i, root in enumerate(roots)}
    for key, tasks in groups.items():
        s, b, arm = key
        for row in tasks.values():
            if (
                row["family"] != "cross_series"
                or row["center"] != row["corrupted_index"]
                or row["center"] not in (2, 3)
            ):
                continue
            index = (PARENTS.index(s), rid[row["root_id"]], row["center"] - 2)
            if arm == "PARENT":
                parent[index] = row[metric]
            else:
                grid[index[0], b, ARMS.index(arm), index[1], index[2]] = row[metric]
    return grid, parent


def _macro_bootstrap(groups, cohorts, *, replicates, seed):
    """Four root-cohort resampling draws shared by every checkpoint and contrast."""
    keys = sorted(groups, key=lambda k: (PARENTS.index(k[0]), -1 if k[1] is None else k[1], k[2]))
    first = groups[keys[0]]
    tasks = sorted(first)
    position = {task: n for n, task in enumerate(tasks)}
    data = np.asarray(
        [[groups[key][task][metric] for task in tasks] for key in keys for metric in METRICS]
    ).reshape(len(keys), len(METRICS), 800)
    rng = np.random.default_rng(seed)
    root_counts = {}
    for cohort in ("CORE32", "EXTRA96", "trend", "duplicate_encoding"):
        roots = sorted(cohorts[cohort])
        indices = rng.integers(0, len(roots), size=(replicates, len(roots)))
        counts = np.zeros((replicates, len(roots)), dtype=float)
        np.add.at(counts, (np.arange(replicates)[:, None], indices), 1)
        root_counts.update({root: counts[:, i] for i, root in enumerate(roots)})
    cell_rows = defaultdict(list)
    for task, row in first.items():
        cell_rows[(row["family"], row["center"], row["corrupted_index"])].append(position[task])
    boot = {
        name: np.zeros((replicates, len(keys), len(METRICS)))
        for name in ("cross_macro", "trend_macro", "duplicate")
    }
    point = {name: np.zeros((len(keys), len(METRICS))) for name in boot}
    for (family, _center, _j), indices in cell_rows.items():
        name = {
            "cross_series": "cross_macro",
            "trend": "trend_macro",
            "duplicate_encoding": "duplicate",
        }[family]
        factor = 1 / (16 if family == "cross_series" else 4)
        weights = np.asarray([root_counts[first[tasks[i]]["root_id"]] for i in indices]).T
        den = weights.sum(axis=1)
        # A resample without a position has an undefined macro statistic; never silently fill zero.
        normalized = np.divide(
            weights, den[:, None], out=np.full_like(weights, np.nan), where=den[:, None] != 0
        )
        boot[name] += np.einsum("rt,kmt->rkm", normalized, data[:, :, indices]) * factor
        point[name] += data[:, :, indices].mean(axis=2) * factor
    boot["cross_trend_50_50"] = 0.5 * (boot["cross_macro"] + boot["trend_macro"])
    point["cross_trend_50_50"] = 0.5 * (point["cross_macro"] + point["trend_macro"])
    output = []
    for k, key in enumerate(keys):
        for name in point:
            for m, metric in enumerate(METRICS):
                values = boot[name][:, k, m]
                finite = np.isfinite(values)
                output.append(
                    {
                        "parent": key[0],
                        "block": key[1],
                        "arm": key[2],
                        "metric": metric,
                        "macro": name,
                        "estimate": float(point[name][k, m]),
                        "ci95_root_conditional": np.quantile(
                            values[finite], [0.025, 0.975]
                        ).tolist()
                        if finite.any()
                        else None,
                        "defined_bootstrap_replicates": int(finite.sum()),
                        "undefined_bootstrap_replicates": int((~finite).sum()),
                    }
                )
    contrasts = []
    index = {key: i for i, key in enumerate(keys)}
    for left, right in (
        [(ARMS[1], ARMS[2])] + [(a, ARMS[0]) for a in ARMS[1:]] + [(a, "PARENT") for a in ARMS]
    ):
        for name in point:
            for m, metric in enumerate(METRICS):
                pairs = [
                    (index[s, b, left], index[s, None if right == "PARENT" else b, right])
                    for s in PARENTS
                    for b in range(3)
                ]
                difference = np.mean(
                    [point[name][lhs, m] - point[name][rhs, m] for lhs, rhs in pairs]
                )
                values = np.mean(
                    [boot[name][:, lhs, m] - boot[name][:, rhs, m] for lhs, rhs in pairs], axis=0
                )
                finite = np.isfinite(values)
                contrasts.append(
                    {
                        "contrast": f"{left}-{right}",
                        "macro": name,
                        "metric": metric,
                        "estimate": float(difference),
                        "ci95_root_conditional": np.quantile(
                            values[finite], [0.025, 0.975]
                        ).tolist()
                        if finite.any()
                        else None,
                        "defined_bootstrap_replicates": int(finite.sum()),
                    }
                )
    return {
        "metrics": output,
        "contrasts": contrasts,
        "bootstrap_replicates": replicates,
        "seed": seed,
        "cohorts": {k: len(v) for k, v in cohorts.items()},
        "weights": (
            "cross16 cells equally; trend4 positions equally; "
            "cross/trend 0.5 each; duplicate separate"
        ),
        "scope": "descriptive conditional root intervals, preserving paired checkpoint vectors",
    }


def analyze_confirmation(
    rows, *, expected_draws=8, bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED
):
    if expected_draws != 8:
        raise ValueError("Frozen confirmation uses eight draws")
    task_rows = summarize_tasks(rows)
    groups, cohorts = _complete_confirmation(task_rows, expected_draws)
    roots = sorted(cohorts["CORE32"] | cohorts["EXTRA96"])
    components = {}
    for metric in ("X", "value_present", "value_present_but_not_X"):
        grid, parent = _primary_grid(groups, roots, metric)
        components[metric] = primary_effect(
            grid, parent_probabilities=parent, bootstrap_replicates=bootstrap_replicates, seed=seed
        )
    if not np.isclose(
        components["X"]["Gamma"],
        components["value_present"]["Gamma"] - components["value_present_but_not_X"]["Gamma"],
        atol=1e-14,
    ):
        raise AssertionError("Gamma must preserve exact value/extra decomposition")
    primary = dict(components["X"])
    primary.update(
        {
            "status": "COMPLETE_DESCRIPTIVE_INFERENCE",
            "metric": "raw_X",
            "root_ids": roots,
            "value_extra_decomposition": components,
            "scientific_pass_fail_gate": None,
        }
    )
    return {
        "primary": primary,
        "task_metrics": task_rows,
        "macro": _macro_bootstrap(groups, cohorts, replicates=bootstrap_replicates, seed=seed),
    }
