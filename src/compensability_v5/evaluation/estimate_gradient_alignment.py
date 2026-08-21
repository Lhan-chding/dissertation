"""Pure-CPU metrics for v5 common-rollout reward-gradient alignment."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence


def _vector(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{label} must be a non-empty numeric vector")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{label} must contain finite numbers")
        result.append(float(item))
    return tuple(result)


def _cosine(left: Sequence[float], right: Sequence[float], weights: Sequence[float]) -> float:
    numerator = sum(a * b * weight for a, b, weight in zip(left, right, weights, strict=True))
    left_norm = math.sqrt(sum(a * a * weight for a, weight in zip(left, weights, strict=True)))
    right_norm = math.sqrt(sum(b * b * weight for b, weight in zip(right, weights, strict=True)))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("gradient alignment is undefined for a zero-norm gradient")
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def alignment_metrics(
    exact_gradient: object,
    reward_gradient: object,
    fisher_diagonal: object,
) -> dict[str, float]:
    """Compute Euclidean and diagonal-Fisher correction alignment."""

    exact = _vector(exact_gradient, "exact_gradient")
    reward = _vector(reward_gradient, "reward_gradient")
    fisher = _vector(fisher_diagonal, "fisher_diagonal")
    if len(exact) != len(reward) or len(exact) != len(fisher):
        raise ValueError("gradient and Fisher dimensions must match")
    if any(value <= 0.0 for value in fisher):
        raise ValueError("fisher_diagonal entries must be strictly positive")
    return {
        "euclidean_cosine": _cosine(exact, reward, (1.0,) * len(exact)),
        "diagonal_fisher_cosine": _cosine(
            exact, reward, tuple(1.0 / value for value in fisher)
        ),
    }


def correction_bearing_group_rate(groups: object) -> float:
    """Return the fraction of sampled groups containing both X and F."""

    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)) or not groups:
        raise ValueError("rollout_groups must be a non-empty sequence")
    correction_bearing = 0
    for group in groups:
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)) or not group:
            raise ValueError("each rollout group must be non-empty")
        values = tuple(group)
        if any(value not in {"X", "S", "F"} for value in values):
            raise ValueError("rollout categories must be X, S, or F")
        correction_bearing += "X" in values and "F" in values
    return correction_bearing / len(groups)


def _fiber_bin(size: int) -> str:
    if size == 1:
        return "singleton"
    if size <= 4:
        return "multi_2_4"
    return "multi_5_plus"


def estimate_gradient_alignment(
    records: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate same-rollout exact, answer, and consistency gradients by scene."""

    required = {
        "scene_id",
        "family",
        "graph_orbit",
        "fiber_size",
        "exact_gradient",
        "answer_gradient",
        "consistency_gradient",
        "fisher_diagonal",
        "rollout_groups",
    }
    rows: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for source in records:
        missing = required - set(source)
        if missing:
            raise ValueError(f"gradient-alignment record missing fields: {sorted(missing)}")
        scene_id, family, graph_orbit = (
            source["scene_id"],
            source["family"],
            source["graph_orbit"],
        )
        fiber_size = source["fiber_size"]
        if (
            not isinstance(scene_id, str)
            or not scene_id
            or scene_id in identifiers
            or not isinstance(family, str)
            or not family
            or not isinstance(graph_orbit, str)
            or not graph_orbit
            or isinstance(fiber_size, bool)
            or not isinstance(fiber_size, int)
            or fiber_size <= 0
        ):
            raise ValueError("gradient-alignment scene metadata is malformed or duplicated")
        identifiers.add(scene_id)
        exact = source["exact_gradient"]
        fisher = source["fisher_diagonal"]
        answer = alignment_metrics(exact, source["answer_gradient"], fisher)
        consistency = alignment_metrics(exact, source["consistency_gradient"], fisher)
        exact_self = alignment_metrics(exact, exact, fisher)
        rows.append(
            {
                "scene_id": scene_id,
                "family": family,
                "graph_orbit": graph_orbit,
                "fiber_size": fiber_size,
                "fiber_size_bin": _fiber_bin(fiber_size),
                "answer_euclidean_cosine": answer["euclidean_cosine"],
                "answer_diagonal_fisher_cosine": answer["diagonal_fisher_cosine"],
                "consistency_euclidean_cosine": consistency["euclidean_cosine"],
                "consistency_diagonal_fisher_cosine": consistency["diagonal_fisher_cosine"],
                "exact_self_diagonal_fisher_cosine": exact_self["diagonal_fisher_cosine"],
                "correction_bearing_group_rate": correction_bearing_group_rate(
                    source["rollout_groups"]
                ),
            }
        )
    if not rows:
        raise ValueError("gradient-alignment records must be non-empty")

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["fiber_size_bin"]), str(row["family"]), str(row["graph_orbit"]))
        grouped[key].append(row)
    strata: list[dict[str, object]] = []
    metric_names = (
        "answer_euclidean_cosine",
        "answer_diagonal_fisher_cosine",
        "consistency_euclidean_cosine",
        "consistency_diagonal_fisher_cosine",
        "correction_bearing_group_rate",
    )
    for (fiber_bin, family, graph_orbit), group in sorted(grouped.items()):
        strata.append(
            {
                "fiber_size_bin": fiber_bin,
                "family": family,
                "graph_orbit": graph_orbit,
                "scene_count": len(group),
                **{
                    name: sum(float(row[name]) for row in group) / len(group)
                    for name in metric_names
                },
            }
        )
    return {
        "schema_version": 1,
        "status": "V5_GRADIENT_ALIGNMENT_EVALUATED",
        "statistical_unit": "semantic_scene",
        "same_sampled_worlds_required": True,
        "scene_count": len(rows),
        "rows": rows,
        "strata": strata,
    }


def gradient_alignment_fixture() -> tuple[dict[str, object], ...]:
    return (
        {
            "scene_id": "fixture-singleton",
            "family": "known_value",
            "graph_orbit": "familiar",
            "fiber_size": 1,
            "exact_gradient": [1.0, 0.0],
            "answer_gradient": [1.0, 0.0],
            "consistency_gradient": [0.8, 0.2],
            "fisher_diagonal": [2.0, 1.0],
            "rollout_groups": [["X", "F", "F", "S"]],
        },
        {
            "scene_id": "fixture-multistate",
            "family": "pair_sum",
            "graph_orbit": "equivalent_basis",
            "fiber_size": 5,
            "exact_gradient": [1.0, 0.0],
            "answer_gradient": [-0.5, 0.5],
            "consistency_gradient": [0.7, 0.3],
            "fisher_diagonal": [2.0, 1.0],
            "rollout_groups": [["S", "F", "S", "F"]],
        },
    )


__all__ = [
    "alignment_metrics",
    "correction_bearing_group_rate",
    "estimate_gradient_alignment",
    "gradient_alignment_fixture",
]
