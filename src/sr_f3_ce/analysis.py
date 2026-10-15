"""Registered paired root statistics; no generalization to arbitrary retraining.

All failures remain in the fixed K denominator. Technical missingness raises.
Root resampling carries all paired arms and the three actual trained blocks.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

from .contract import FAMILIES
from .evaluation import ARMS, EXPERIMENT_ID, endpoint_models, object_hash

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 2026101651
CONFIDENCE = 1 - 0.05 / 3
PRIMARY = (
    ("EXEC", "PREFIX", "REPAIR", "value_correct"),
    ("EXEC", "GOLD", "REPAIR", "value_correct"),
    ("EXEC", "GOLD", "TEST_COMPOSITION", "value_correct"),
)
METRICS = (
    "value_correct",
    "certificate",
    "certificate_truth",
    "certificate_sufficient",
    "presentation",
    "finish",
    "J",
    "legacy_full",
    "value_parseable",
)


def contrast(
    matrix, family, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, alpha=0.05 / 3
):
    """Reference-compatible equal-family, equal-block paired cluster bootstrap.

    matrix[root, block] is the within-root arm difference. ``replicates`` may be
    reduced by unit tests only; ``primary_analysis`` always uses 10,000.
    """
    d, f = np.asarray(matrix, dtype=float), np.asarray(family)
    if d.ndim != 2 or d.shape[0] != len(f) or d.shape[1] != 3 or not np.isfinite(d).all():
        raise ValueError("Complete finite root x 3 block matrix required")
    if not len(f) or type(replicates) is not int or replicates < 1 or not 0 < alpha < 1:
        raise ValueError("Nonempty strata and valid bootstrap settings required")
    levels = np.unique(f)
    indices = [np.flatnonzero(f == level) for level in levels]
    if any(len(ii) < 2 for ii in indices):
        raise ValueError("At least two roots per stratum required")
    point = float(np.mean([d[ii].mean() for ii in indices]))
    per = np.mean([d[ii].mean(axis=0) for ii in indices], axis=0)
    rng = np.random.default_rng(seed)
    boots = np.zeros(replicates)
    for ii in indices:
        sampled = rng.choice(ii, size=(replicates, len(ii)), replace=True)
        boots += d[sampled].mean(axis=(1, 2)) / len(indices)
    ci95 = np.quantile(boots, [0.025, 0.975]).tolist()
    simultaneous = np.quantile(boots, [alpha / 2, 1 - alpha / 2]).tolist()
    sd = float(per.std(ddof=1))
    halfwidth = float(student_t.ppf(0.975, 2) * sd / math.sqrt(3))
    signs = dict(
        positive=int(np.sum(per > 0)), zero=int(np.sum(per == 0)), negative=int(np.sum(per < 0))
    )
    return dict(
        point=point,
        effect_pp=100 * point,
        per_block=per.tolist(),
        per_block_pp=(100 * per).tolist(),
        CI95=ci95,
        CI_simultaneous=simultaneous,
        CI95_pp=[100 * x for x in ci95],
        CI_simultaneous_pp=[100 * x for x in simultaneous],
        roots=len(f),
        strata={str(z): len(ii) for z, ii in zip(levels, indices, strict=True)},
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
        confidence_simultaneous=1 - alpha,
        paired_t_sensitivity=dict(
            df=2,
            per_block_sd_pp=100 * sd,
            CI95_pp=[100 * (point - halfwidth), 100 * (point + halfwidth)],
            role="descriptive sensitivity across only three actual blocks",
        ),
        signs=signs,
        family_effect_pp={
            str(z): 100 * float(d[ii].mean()) for z, ii in zip(levels, indices, strict=True)
        },
        cluster_scope="fixed trained blocks; root uncertainty only",
        arbitrary_seed_generalization=False,
        confirmation="POSITIVE_CONFIRMED" if simultaneous[0] > 0 else "NOT_CONFIRMED",
    )


def _task_map(tasks):
    rows = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)
    result = {t["qid"]: t for t in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate frozen task")
    return result


def _selected_tasks(tasks, scope, protocol):
    if scope in ("REPAIR", "LEGACY", "CANONICAL_REPAIR"):
        chosen = [
            t
            for t in tasks.values()
            if t["pool"] == "TEST_ID"
            and (t["mode"] == "repair" if scope != "LEGACY" else t["mode"].startswith("legacy"))
        ]
    elif scope in ("TEST_ID", "TEST_COMPOSITION", "TEST_LANGUAGE", "TEST_RENDER", "TEST_ARITY"):
        chosen = [t for t in tasks.values() if t["pool"] == scope]
    else:
        raise ValueError("Unknown frozen analysis scope")
    if protocol != "structured" or scope == "CANONICAL_REPAIR":
        if scope not in ("REPAIR", "CANONICAL_REPAIR"):
            raise ValueError("Extra protocols only cover canonical TEST_ID repair v2")
        chosen = [t for t in chosen if t["variant"] == 2]
    if not chosen or {t["family"] for t in chosen} != set(FAMILIES):
        raise ValueError("Every frozen family must be represented")
    return {t["qid"]: t for t in chosen}


def root_values(rows, tasks, arm, scope, metric="value_correct", *, protocol="structured"):
    if arm not in (*ARMS, "PARENT") or metric not in METRICS:
        raise ValueError("Unregistered arm or metric")
    task_map = _task_map(tasks)
    selected = _selected_tasks(task_map, scope, protocol)
    models = {m["model_id"]: m for m in endpoint_models() if m["arm"] == arm}
    observed = {b: {} for b in range(3)}
    for row in rows:
        if (
            row.get("model_id") not in models
            or row.get("protocol") != protocol
            or row.get("view") is not None
        ):
            continue
        if row.get("qid") not in selected:
            continue
        task, model = selected[row["qid"]], models[row["model_id"]]
        if any(row.get(k) != task[k] for k in ("pool", "family", "root_id")):
            raise PermissionError("Analysis metadata differs from frozen task")
        if row.get("block") != model["block"] or row.get("step") != model["step"]:
            raise PermissionError("Final model block/step differs")
        if row.get("generation_status") != "COMPLETE" or row.get("technical_validation_errors"):
            raise RuntimeError("Technical missingness cannot become a model failure")
        if type(row.get("draw")) is not int or row["draw"] not in range(4):
            raise ValueError("Four registered draws required")
        score = row.get("score", {}).get(metric)
        if score not in (0, 1):
            raise ValueError("Missing/nonbinary metric cannot be excluded or imputed")
        key = (row["qid"], row["draw"])
        if key in observed[model["block"]]:
            raise ValueError("Duplicate endpoint observation")
        observed[model["block"]][key] = float(score)
    expected = {(qid, draw) for qid in selected for draw in range(4)}
    if any(set(cells) != expected for cells in observed.values()):
        raise RuntimeError("Complete frozen qid x K4 x three-block support required")
    roots, families = defaultdict(list), {}
    for qid, task in selected.items():
        root = task["root_id"]
        if root in families and families[root] != task["family"]:
            raise ValueError("A root cannot belong to multiple families")
        roots[root].append(qid)
        families[root] = task["family"]
    ordered = sorted(roots)
    values = np.asarray(
        [
            [
                np.mean([np.mean([observed[b][(qid, d)] for d in range(4)]) for qid in roots[root]])
                for b in range(3)
            ]
            for root in ordered
        ]
    )
    return values, [families[root] for root in ordered], ordered


def paired_contrast(
    rows, tasks, arm1, arm0, scope, metric="value_correct", *, protocol="structured"
):
    left, families, roots = root_values(rows, tasks, arm1, scope, metric, protocol=protocol)
    right, other_families, other_roots = root_values(
        rows, tasks, arm0, scope, metric, protocol=protocol
    )
    if roots != other_roots or families != other_families:
        raise RuntimeError("Matched root identities required for paired contrast")
    result = contrast(left - right, families)
    result.update(
        arm1=arm1,
        arm0=arm0,
        scope=scope,
        metric=metric,
        protocol=protocol,
        root_ids_sha256=object_hash(roots),
    )
    levels = sorted(set(families))
    index = np.asarray(families)
    result["arm1_rate"] = float(np.mean([left[index == f].mean() for f in levels]))
    result["arm0_rate"] = float(np.mean([right[index == f].mean() for f in levels]))
    result["practical_reference_pp"] = 2
    result["power_guarantee"] = False
    return result


def retention_analysis(rows, tasks):
    results = {}
    for arm in ARMS:
        results[arm] = {}
        for metric in ("value_correct", "J"):
            item = paired_contrast(rows, tasks, arm, "PARENT", "LEGACY", metric)
            left, families, _ = root_values(rows, tasks, arm, "LEGACY", metric)
            right, _, _ = root_values(rows, tasks, "PARENT", "LEGACY", metric)
            family_results = {}
            f = np.asarray(families)
            for family in FAMILIES:
                family_results[family] = contrast((left - right)[f == family], f[f == family])
            worst = min(family_results, key=lambda name: family_results[name]["effect_pp"])
            item.update(
                by_family=family_results,
                worst_family=worst,
                worst_family_effect_pp=family_results[worst]["effect_pp"],
                preservation_margin_pp=2,
                any_family_point_below_margin=any(
                    x["effect_pp"] < -2 for x in family_results.values()
                ),
                safety_guarantee=False,
            )
            results[arm][metric] = item
    return results


def _assert_release(rows, receipt):
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "RELEASED"
        or receipt.get("experiment_id") != EXPERIMENT_ID
    ):
        raise PermissionError("New TEST analysis requires an explicit final release")
    if receipt.get("projection_kind") == "verified_raw_rescore_projection_v1":
        if receipt.get("raw_rescoring") != "PASS" or receipt.get(
            "analysis_rows_sha256"
        ) != object_hash(rows):
            raise PermissionError("Analysis projection differs from full raw rescoring receipt")
        return
    grouped = defaultdict(list)
    for row in rows:
        if row.get("sealed"):
            grouped[row["group_id"]].append({k: v for k, v in row.items() if k != "score"})
    for gid, raw in grouped.items():
        if receipt.get("raw_groups", {}).get(gid) != object_hash(
            sorted(raw, key=lambda x: x["draw"])
        ):
            raise PermissionError("Scored rows differ from released raw content")


def primary_analysis(rows, tasks, release_receipt):
    """Full fixed primary, secondary, retention and protocol results, adverse or not."""
    rows = list(rows)
    _assert_release(rows, release_receipt)
    primary = {f"C{i}": paired_contrast(rows, tasks, *spec) for i, spec in enumerate(PRIMARY, 1)}
    secondary = {}
    for scope in ("REPAIR", "TEST_COMPOSITION"):
        for arm1, arm0 in (("GOLD", "RL"), ("PREFIX", "GOLD"), ("EXEC", "RL")):
            secondary[f"{scope}:{arm1}-{arm0}"] = paired_contrast(rows, tasks, arm1, arm0, scope)
    protocols = {}
    for protocol in ("structured", "answer_only", "natural_work"):
        for arm1, arm0 in (("EXEC", "PREFIX"), ("EXEC", "GOLD")):
            protocols[f"{protocol}:{arm1}-{arm0}"] = paired_contrast(
                rows, tasks, arm1, arm0, "CANONICAL_REPAIR", protocol=protocol
            )
    protocol_rates = {}
    for protocol in ("structured", "answer_only", "natural_work"):
        protocol_rates[protocol] = {}
        metrics = (
            METRICS
            if protocol == "structured"
            else ("value_correct", "presentation", "finish", "value_parseable")
        )
        for arm in ("PARENT", *ARMS):
            protocol_rates[protocol][arm] = {}
            for metric in metrics:
                values, families, _ = root_values(
                    rows, tasks, arm, "CANONICAL_REPAIR", metric, protocol=protocol
                )
                f = np.asarray(families)
                by_family = {name: values[f == name].mean(axis=0).tolist() for name in FAMILIES}
                per_block = np.mean(list(by_family.values()), axis=0)
                protocol_rates[protocol][arm][metric] = dict(
                    rate=float(per_block.mean()),
                    per_block=per_block.tolist(),
                    family_per_block=by_family,
                )
    decomposition = {}
    for scope in (
        "REPAIR",
        "LEGACY",
        "TEST_COMPOSITION",
        "TEST_LANGUAGE",
        "TEST_RENDER",
        "TEST_ARITY",
    ):
        decomposition[scope] = {}
        for arm in ("PARENT", *ARMS):
            decomposition[scope][arm] = {}
            for metric in METRICS:
                values, families, _ = root_values(rows, tasks, arm, scope, metric)
                f = np.asarray(families)
                family_rates = {name: values[f == name].mean(axis=0).tolist() for name in FAMILIES}
                per = np.mean(list(family_rates.values()), axis=0)
                decomposition[scope][arm][metric] = dict(
                    rate=float(per.mean()), per_block=per.tolist(), family_per_block=family_rates
                )
    return dict(
        status="NUMERICAL_ANALYSIS_COMPLETE",
        experiment_id=EXPERIMENT_ID,
        primary=primary,
        secondary=secondary,
        canonical_protocols=protocols,
        canonical_protocol_rates=protocol_rates,
        retention=retention_analysis(rows, tasks),
        decomposition=decomposition,
        multiplicity="three primary contrasts; two-sided 98.33333333333333% intervals",
        all_registered_blocks_retained=True,
        adverse_results_retained=True,
        uncertainty_scope="root uncertainty conditional on the three actual trained blocks",
        report_instruction="Place all three block differences next to each primary contrast",
        scientific_completion_claimed=False,
    )
