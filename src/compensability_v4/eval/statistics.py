"""Scene-clustered summaries, paired effects, and multiplicity controls."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SceneMetricAggregate:
    metric: str
    number_of_scenes: int
    number_of_rollouts: int
    point_estimate: float
    scene_values: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class StratifiedMetricAggregate:
    stratum_field: str
    strata: tuple[tuple[str, SceneMetricAggregate], ...]


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric or boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def aggregate_scene_metrics(
    rows: Iterable[Mapping[str, object]], *, metric: str
) -> SceneMetricAggregate:
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty string")
    grouped: dict[str, list[float]] = {}
    rollout_keys: set[tuple[str, object]] = set()
    number_of_rollouts = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each metric row must be a mapping")
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("scene_id must be a non-empty string")
        if metric not in row:
            raise ValueError(f"metric row is missing {metric!r}")
        if "rollout_id" in row:
            rollout_id = row["rollout_id"]
            try:
                key = (scene_id, rollout_id)
                if key in rollout_keys:
                    raise ValueError("rollout_id must be unique within scene")
                rollout_keys.add(key)
            except TypeError as error:
                raise TypeError("rollout_id must be hashable") from error
        grouped.setdefault(scene_id, []).append(_number(row[metric], metric))
        number_of_rollouts += 1
    if not grouped:
        raise ValueError("rows must not be empty")
    scene_values = tuple(
        (scene_id, sum(values) / len(values)) for scene_id, values in sorted(grouped.items())
    )
    estimate = sum(value for _scene, value in scene_values) / len(scene_values)
    return SceneMetricAggregate(
        metric=metric,
        number_of_scenes=len(scene_values),
        number_of_rollouts=number_of_rollouts,
        point_estimate=estimate,
        scene_values=scene_values,
    )


def aggregate_stratified_scene_metrics(
    rows: Iterable[Mapping[str, object]], *, metric: str, stratum_field: str = "family"
) -> StratifiedMetricAggregate:
    """Aggregate within registered strata while retaining scene as the unit."""

    if not isinstance(stratum_field, str) or not stratum_field:
        raise ValueError("stratum_field must be a non-empty string")
    grouped: dict[str, list[Mapping[str, object]]] = {}
    scene_strata: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each metric row must be a mapping")
        scene_id = row.get("scene_id")
        stratum = row.get(stratum_field)
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("scene_id must be a non-empty string")
        if not isinstance(stratum, str) or not stratum:
            raise ValueError(f"{stratum_field} must be a non-empty string")
        previous = scene_strata.setdefault(scene_id, stratum)
        if previous != stratum:
            raise ValueError("a scene cannot belong to multiple strata")
        grouped.setdefault(stratum, []).append(row)
    if not grouped:
        raise ValueError("rows must not be empty")
    return StratifiedMetricAggregate(
        stratum_field=stratum_field,
        strata=tuple(
            (stratum, aggregate_scene_metrics(grouped[stratum], metric=metric))
            for stratum in sorted(grouped)
        ),
    )


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    low: float
    high: float
    confidence: float
    number_of_scenes: int


def scene_clustered_bootstrap_ci(
    rows: Iterable[Mapping[str, object]],
    *,
    metric: str,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    aggregate = aggregate_scene_metrics(rows, metric=metric)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be numeric")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    values = tuple(value for _scene, value in aggregate.scene_values)
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(n_resamples)
    )
    alpha = (1.0 - float(confidence)) / 2.0
    low_index = max(0, min(n_resamples - 1, int(alpha * n_resamples)))
    high_index = max(0, min(n_resamples - 1, math.ceil((1.0 - alpha) * n_resamples) - 1))
    return BootstrapInterval(
        estimate=aggregate.point_estimate,
        low=estimates[low_index],
        high=estimates[high_index],
        confidence=float(confidence),
        number_of_scenes=aggregate.number_of_scenes,
    )


def paired_scene_difference(
    before: Mapping[str, float], after: Mapping[str, float]
) -> tuple[tuple[str, float], ...]:
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise TypeError("paired inputs must be mappings")
    if before.keys() != after.keys() or not before:
        raise ValueError("paired inputs must contain the same non-empty scene IDs")
    return tuple(
        (scene_id, _number(after[scene_id], "after") - _number(before[scene_id], "before"))
        for scene_id in sorted(before)
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(p_values, Mapping):
        raise TypeError("p_values must be a mapping")
    validated: list[tuple[int, str, float]] = []
    for position, (name, value) in enumerate(p_values.items()):
        number = _number(value, "p-value")
        if not isinstance(name, str) or not name or not 0.0 <= number <= 1.0:
            raise ValueError("p-values need named hypotheses and values in [0, 1]")
        validated.append((position, name, number))
    adjusted: dict[str, float] = {}
    running = 0.0
    ordered = sorted(validated, key=lambda item: (item[2], item[0]))
    for rank, (_position, name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}
