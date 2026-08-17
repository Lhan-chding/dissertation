"""Immutable constructors for the preregistered style and constraint OOD axes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from compensability_v4.schemas.scene import RecoveryScene
from compensability_v4.theory.constraint_system import satisfies_all_facts

from .splits import DatasetSplit


def generate_style_ood_scene(
    source: RecoveryScene, *, scene_id: str, image_path: str
) -> RecoveryScene:
    """Change only rendering provenance while preserving pixel budget and task semantics."""

    if not isinstance(source, RecoveryScene):
        raise TypeError("source must be a RecoveryScene")
    payload = source.to_mapping()
    payload.update(
        {
            "scene_id": scene_id,
            "split": DatasetSplit.CONFIRM_STYLE_OOD.value,
            "image_path": image_path,
        }
    )
    return RecoveryScene.from_mapping(payload)


def generate_constraint_ood_scene(
    source: RecoveryScene,
    *,
    scene_id: str,
    constraint_graph_id: str,
    facts: Sequence[Mapping[str, object]],
    image_path: str,
) -> RecoveryScene:
    """Replace the constraint graph while keeping the same valid hidden world."""

    if not isinstance(source, RecoveryScene):
        raise TypeError("source must be a RecoveryScene")
    if not satisfies_all_facts(source.truth, facts):
        raise ValueError("constraint-OOD facts must support the source truth")
    payload = source.to_mapping()
    payload.update(
        {
            "scene_id": scene_id,
            "split": DatasetSplit.CONFIRM_CONSTRAINT_OOD.value,
            "constraint_graph_id": constraint_graph_id,
            "facts": [dict(fact) for fact in facts],
            "image_path": image_path,
        }
    )
    return RecoveryScene.from_mapping(payload)
