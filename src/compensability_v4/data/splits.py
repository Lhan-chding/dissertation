"""Registered dataset splits and isolation gates."""

from __future__ import annotations

from enum import Enum


class DatasetSplit(str, Enum):
    LEGACY_DIAGNOSTIC = "legacy_diagnostic"
    SYMBOLIC_SUPPORT_TRAIN = "symbolic_support_train"
    NATURAL_ERROR_SUPPORT_TRAIN = "natural_error_support_train"
    SUPPORT_DEV = "support_dev"
    CONFIRM_IID = "confirm_iid"
    CONFIRM_STYLE_OOD = "confirm_style_ood"
    CONFIRM_CONSTRAINT_OOD = "confirm_constraint_ood"


class SplitIsolationError(ValueError):
    pass


def validate_split_isolation(scenes: list[object] | tuple[object, ...]) -> None:
    seen: dict[str, dict[str, str]] = {
        "semantic_scene_id": {},
        "numeric_table_id": {},
        "constraint_graph_id": {},
    }
    for scene in scenes:
        split = getattr(scene, "split")
        for key in seen:
            value = getattr(scene, key)
            previous = seen[key].get(value)
            if previous is not None and previous != split.value:
                raise SplitIsolationError(f"{key} leaked across splits: {value}")
            seen[key][value] = split.value
