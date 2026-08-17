"""Validation and assembly of frozen-model natural single-error support data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from compensability_v4.schemas.observation import NaturalObservation
from compensability_v4.schemas.scene import RecoveryScene
from compensability_v4.theory.constraint_system import World

from .splits import CONFIRM_SPLITS


@dataclass(frozen=True, slots=True)
class NaturalErrorExample:
    scene_id: str
    observation_id: str
    truth: World
    observed_values: World
    error_index: int
    stage1_model_hash: str


def build_natural_error_pool(
    scenes: Iterable[RecoveryScene], observations: Iterable[NaturalObservation]
) -> tuple[NaturalErrorExample, ...]:
    """Keep all and only natural, exactly-one-position errors from non-confirm scenes."""

    scene_index: dict[str, RecoveryScene] = {}
    for scene in scenes:
        if not isinstance(scene, RecoveryScene):
            raise TypeError("scenes must contain RecoveryScene records")
        if scene.scene_id in scene_index:
            raise ValueError(f"duplicate scene_id {scene.scene_id!r}")
        if scene.split in CONFIRM_SPLITS:
            raise ValueError("confirm scenes must never enter the natural error support pool")
        scene_index[scene.scene_id] = scene
    examples: list[NaturalErrorExample] = []
    observation_ids: set[str] = set()
    for observation in observations:
        if not isinstance(observation, NaturalObservation):
            raise TypeError("observations must contain NaturalObservation records")
        if observation.observation_id in observation_ids:
            raise ValueError(f"duplicate observation_id {observation.observation_id!r}")
        observation_ids.add(observation.observation_id)
        if observation.scene_id not in scene_index:
            raise ValueError(f"unknown observation scene_id {observation.scene_id!r}")
        scene = scene_index[observation.scene_id]
        changed = tuple(
            index
            for index, (truth, observed) in enumerate(
                zip(scene.truth, observation.observed_values, strict=True)
            )
            if truth != observed
        )
        if changed != (observation.error_index,):
            raise ValueError(
                "natural observation must contain exactly its registered one-position error"
            )
        examples.append(
            NaturalErrorExample(
                scene_id=scene.scene_id,
                observation_id=observation.observation_id,
                truth=scene.truth,
                observed_values=observation.observed_values,
                error_index=observation.error_index,
                stage1_model_hash=observation.stage1_model_hash,
            )
        )
    return tuple(sorted(examples, key=lambda example: example.observation_id))
