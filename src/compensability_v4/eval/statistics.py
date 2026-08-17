"""Scene-level aggregation only."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SceneAggregate:
    number_of_scenes: int
    number_of_rollouts: int
    point_estimate: float


def aggregate_scene_metrics(rows: list[dict[str, object]], *, metric: str) -> SceneAggregate:
    per_scene: dict[str, bool] = {}
    for row in rows:
        scene_id = row["scene_id"]
        value = row[metric]
        per_scene[str(scene_id)] = per_scene.get(str(scene_id), False) or bool(value)
    return SceneAggregate(
        number_of_scenes=len(per_scene),
        number_of_rollouts=len(rows),
        point_estimate=sum(per_scene.values()) / len(per_scene),
    )
