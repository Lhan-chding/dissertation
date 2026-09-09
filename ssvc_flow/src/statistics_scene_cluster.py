"""Fixed-prompt, family-stratified scene-cluster statistics for next-stage audits."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

CATEGORIES = ("X", "S", "W", "I")
METRICS = ("pX", "pS", "pW", "pI", "v", "pA", "qX_pool", "qS_pool")


def _metrics(p):
    v = p[..., :3].sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack(
            [*np.moveaxis(p, -1, 0), v, p[..., 0] + p[..., 1], p[..., 0] / v, p[..., 1] / v],
            axis=-1,
        )


def weighted_prompt_metrics(rows, weights=None):
    groups = defaultdict(list)
    for row in rows:
        if row["category"] not in CATEGORIES:
            raise ValueError("unknown category")
        groups[row["prompt_id"]].append(row)
    if not groups:
        raise ValueError("no rows")
    weights = weights if weights is not None else {key: 1 / len(groups) for key in groups}
    if (
        set(weights) != set(groups)
        or not np.isclose(sum(weights.values()), 1)
        or any(not np.isfinite(w) or w < 0 for w in weights.values())
    ):
        raise ValueError("nonnegative finite weights must cover prompts and sum to one")
    p = np.array(
        [
            sum(
                weights[k] * sum(r["category"] == c for r in value) / len(value)
                for k, value in groups.items()
            )
            for c in CATEGORIES
        ]
    )
    values = _metrics(p)
    return {
        **{k: float(v) if np.isfinite(v) else None for k, v in zip(METRICS, values, strict=False)},
        "prompt_count": len(groups),
    }


def scene_cluster_intervals(prompt_counts, repeats=5000, seed=20260909):
    """Keep interfaces paired and stratify sampling by family; never resample outputs.

    The raw endpoint uses original prompt proportions. Standardized endpoint gives
    each observed family/interface cell equal weight. Missing pairs are UNKNOWN.
    """
    if repeats < 2:
        raise ValueError("at least two replicates required")
    groups = defaultdict(dict)
    for row in prompt_counts:
        if row.get("decode_mode", "sample") != "sample":
            continue
        key = (row["family"], row["base_scene_id"])
        interface = row["interface"]
        if interface in groups[key]:
            raise ValueError("duplicate scene/interface prompt")
        n = row["n_total"]
        if n <= 0:
            raise ValueError("nonempty bank required")
        groups[key][interface] = np.array([row["n_" + c] / n for c in CATEGORIES])
    interfaces = sorted({i for group in groups.values() for i in group})
    if not groups or any(set(group) != set(interfaces) for group in groups.values()):
        return {"status": "UNKNOWN", "reason": "missing common paired support"}
    families = sorted({key[0] for key in groups})
    rng = np.random.default_rng(seed)
    strata, draws = {}, {}
    for family in families:
        ids = sorted(k for k in groups if k[0] == family)
        if len(ids) < 2:
            return {"status": "UNKNOWN", "reason": "fewer than two scenes in a stratum"}
        strata[family] = np.stack([np.stack([groups[k][i] for i in interfaces]) for k in ids])
        idx = rng.integers(0, len(ids), size=(repeats, len(ids)))
        draws[family] = strata[family][idx].mean(axis=1)
    total = sum(len(v) for v in strata.values())
    raw = sum(draws[f] * len(strata[f]) / total for f in families)
    standard = sum(draws.values()) / len(families)

    def intervals(values):
        result = {}
        for j, metric in enumerate(METRICS):
            col = values[..., j]
            result[metric] = (
                {"status": "UNKNOWN", "reason": "undefined denominator in replicates"}
                if not np.isfinite(col).all()
                else {
                    "status": "ESTIMATED",
                    "low": float(np.quantile(col, 0.025)),
                    "high": float(np.quantile(col, 0.975)),
                }
            )
        return result

    paired = {}
    if len(interfaces) == 2:
        paired["contrast"] = f"{interfaces[1]} minus {interfaces[0]}"
        paired["metrics"] = intervals(_metrics(raw[:, 1]) - _metrics(raw[:, 0]))
    return {
        "status": "ESTIMATED",
        "repeats": repeats,
        "seed": seed,
        "unit": "base_scene",
        "strata": {f: len(v) for f, v in strata.items()},
        "double_resample_within_cluster": False,
        "CI_scope": "evaluation scene uncertainty; no training-seed uncertainty",
        "raw_overall": intervals(_metrics(raw.mean(axis=1))),
        "family_standardized": intervals(_metrics(standard.mean(axis=1))),
        "paired_interface_effect": paired,
    }


def cluster_bootstrap(rows, metric="pX", repeats=5000, seed=20260909):
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt_id"]].append(row)
    counts = [
        {
            **group[0],
            "family": group[0].get("family", "all"),
            "interface": group[0].get("interface", "one"),
            "n_total": len(group),
            **{"n_" + c: sum(r["category"] == c for r in group) for c in CATEGORIES},
        }
        for group in by_prompt.values()
    ]
    result = scene_cluster_intervals(counts, repeats, seed)
    return result.get("raw_overall", {}).get(metric, result)
