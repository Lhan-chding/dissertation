"""Preregistered v4 split names and confirm-set isolation checks."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compensability_v4.schemas.scene import RecoveryScene


class DatasetSplit(str, Enum):
    LEGACY_DIAGNOSTIC = "legacy_diagnostic"
    SYMBOLIC_SUPPORT_TRAIN = "symbolic_support_train"
    NATURAL_ERROR_SUPPORT_TRAIN = "natural_error_support_train"
    SUPPORT_DEV = "support_dev"
    CONFIRM_IID = "confirm_iid"
    CONFIRM_STYLE_OOD = "confirm_style_ood"
    CONFIRM_CONSTRAINT_OOD = "confirm_constraint_ood"


CONFIRM_SPLITS = frozenset(
    {
        DatasetSplit.CONFIRM_IID,
        DatasetSplit.CONFIRM_STYLE_OOD,
        DatasetSplit.CONFIRM_CONSTRAINT_OOD,
    }
)


class SplitIsolationError(ValueError):
    """A confirmatory scene overlaps pre-freeze data."""


def validate_split_isolation(scenes: Iterable[RecoveryScene]) -> None:
    scene_tuple = tuple(scenes)
    if len({scene.scene_id for scene in scene_tuple}) != len(scene_tuple):
        raise SplitIsolationError("scene_id values must be globally unique")
    preconfirm = tuple(scene for scene in scene_tuple if scene.split not in CONFIRM_SPLITS)
    confirm = tuple(scene for scene in scene_tuple if scene.split in CONFIRM_SPLITS)
    for key in ("semantic_scene_id", "numeric_table_id", "constraint_graph_id"):
        prior_values = {getattr(scene, key) for scene in preconfirm}
        leaked = sorted(
            {getattr(scene, key) for scene in confirm if getattr(scene, key) in prior_values}
        )
        if leaked:
            raise SplitIsolationError(f"confirm split leaks {key}: {', '.join(leaked)}")


def validate_fixed_visual_budget(scenes: Iterable[RecoveryScene]) -> None:
    budgets = {(scene.resized_height, scene.resized_width) for scene in scenes}
    if len(budgets) > 1:
        raise SplitIsolationError("all v4 scenes must use one fixed visual input budget")
