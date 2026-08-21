"""Paired scene-cluster inference for the single-seed Study-B pilot."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026082202
RELATIONAL_FAMILIES = frozenset({"pair_sum", "cross_series", "trend"})


class PairedMetricError(ValueError):
    """Paired Study-B rows do not define the frozen comparison."""


def _paired_rows(
    b2_rows: Sequence[Mapping[str, object]], b3_rows: Sequence[Mapping[str, object]]
) -> tuple[tuple[Mapping[str, object], Mapping[str, object]], ...]:
    b2 = {str(row.get("scene_id")): row for row in b2_rows}
    b3 = {str(row.get("scene_id")): row for row in b3_rows}
    if len(b2) != len(b2_rows) or len(b3) != len(b3_rows) or set(b2) != set(b3):
        raise PairedMetricError("B2/B3 evaluation scene identities are not one-to-one")
    pairs = tuple((b2[scene_id], b3[scene_id]) for scene_id in sorted(b2))
    for left, right in pairs:
        if any(
            left.get(field) != right.get(field)
            for field in ("semantic_scene_id", "family", "evaluation_axes", "truth")
        ):
            raise PairedMetricError("B2/B3 paired scene provenance differs")
    return pairs


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    index = round((len(sorted_values) - 1) * probability)
    return float(sorted_values[index])


def _contrast(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    *,
    axes: frozenset[str],
    relational_only: bool,
    metric: str,
    rng: random.Random,
    resamples: int,
) -> dict[str, object]:
    by_parent: dict[str, list[float]] = defaultdict(list)
    for b2, b3 in pairs:
        row_axes = b2.get("evaluation_axes")
        if not isinstance(row_axes, list) or len(row_axes) != 1 or row_axes[0] not in axes:
            continue
        if relational_only and b2.get("family") not in RELATIONAL_FAMILIES:
            continue
        left, right = b2.get(metric), b3.get(metric)
        if not isinstance(left, bool) or not isinstance(right, bool):
            raise PairedMetricError(f"paired {metric} values must be boolean")
        by_parent[str(b2["semantic_scene_id"])].append(float(right) - float(left))
    if not by_parent:
        raise PairedMetricError(f"paired contrast has no eligible {metric} scenes")
    cluster_effects = tuple(sum(values) / len(values) for values in by_parent.values())
    point = sum(cluster_effects) / len(cluster_effects)
    draws: list[float] = []
    for _ in range(resamples):
        sampled = rng.choices(cluster_effects, k=len(cluster_effects))
        draws.append(sum(sampled) / len(sampled))
    draws.sort()
    return {
        "semantic_scene_count": len(cluster_effects),
        "delta": point,
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
    }


def paired_bootstrap_contrasts(
    b2_rows: Sequence[Mapping[str, object]],
    b3_rows: Sequence[Mapping[str, object]],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Return fixed paired-bootstrap CIs using semantic scene as the cluster."""

    if type(seed) is not int or type(resamples) is not int or resamples != BOOTSTRAP_RESAMPLES:
        raise PairedMetricError("Study B requires the registered seed and 10,000 resamples")
    if seed != BOOTSTRAP_SEED:
        raise PairedMetricError("Study B paired-bootstrap seed drifted")
    pairs = _paired_rows(b2_rows, b3_rows)
    rng = random.Random(seed)
    graph = {
        metric: _contrast(
            pairs,
            axes=frozenset({"constraint_graph"}),
            relational_only=True,
            metric=metric,
            rng=rng,
            resamples=resamples,
        )
        for metric in ("exact_world", "genuine_recovery")
    }
    structural = {
        metric: _contrast(
            pairs,
            axes=frozenset(
                {"variable_permutation", "error_position", "fact_order", "constraint_graph"}
            ),
            relational_only=False,
            metric=metric,
            rng=rng,
            resamples=resamples,
        )
        for metric in ("exact_world", "genuine_recovery")
    }
    graph_lowers = [float(graph[metric]["ci95"][0]) for metric in graph]
    structural_lowers = [float(structural[metric]["ci95"][0]) for metric in structural]
    triggered = max(graph_lowers + structural_lowers) > 0.0
    return {
        "bootstrap": {
            "method": "paired_scene_cluster_percentile",
            "seed": seed,
            "resamples": resamples,
            "confidence_level": 0.95,
        },
        "relational_constraint_graph": graph,
        "structural_ood": structural,
        "stop_signal": {
            "rule": "B3_minus_B2_paired_CI95_lower_gt_zero",
            "triggered": triggered,
        },
    }


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "PairedMetricError",
    "paired_bootstrap_contrasts",
]
