"""Immutable recovery-scene schema."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from compensability_v4.data.splits import DatasetSplit


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class RecoveryScene:
    scene_id: str
    split: DatasetSplit
    semantic_scene_id: str
    numeric_table_id: str
    constraint_graph_id: str
    truth: tuple[int, int, int, int]
    facts: tuple[MappingProxyType, ...]
    resized_height: int
    resized_width: int
    image_path: str

    @classmethod
    def from_mapping(cls, mapping: dict[str, object]) -> "RecoveryScene":
        expected = {
            "scene_id", "split", "semantic_scene_id", "numeric_table_id", "constraint_graph_id",
            "truth", "facts", "resized_height", "resized_width", "image_path",
        }
        unknown = sorted(set(mapping) - expected)
        if unknown:
            raise ValueError(f"unknown field: {unknown[0]}")
        truth = mapping["truth"]
        if not isinstance(truth, list) or len(truth) != 4:
            raise ValueError("truth must contain exactly four values")
        if any(type(item) is not int for item in truth):
            raise TypeError("truth must contain exact integers")
        for key in ("resized_height", "resized_width"):
            value = mapping[key]
            if type(value) is not int or value % 28 != 0:
                raise ValueError(f"{key} must be a multiple of 28")
        facts = mapping["facts"]
        if not isinstance(facts, list):
            raise TypeError("facts must be a list")
        return cls(
            scene_id=_sha(mapping["scene_id"], "scene_id"),
            split=DatasetSplit(mapping["split"]),
            semantic_scene_id=_sha(mapping["semantic_scene_id"], "semantic_scene_id"),
            numeric_table_id=_sha(mapping["numeric_table_id"], "numeric_table_id"),
            constraint_graph_id=_sha(mapping["constraint_graph_id"], "constraint_graph_id"),
            truth=(truth[0], truth[1], truth[2], truth[3]),
            facts=tuple(MappingProxyType(dict(item)) for item in facts),
            resized_height=mapping["resized_height"],
            resized_width=mapping["resized_width"],
            image_path=_sha(mapping["image_path"], "image_path"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "split": self.split.value,
            "semantic_scene_id": self.semantic_scene_id,
            "numeric_table_id": self.numeric_table_id,
            "constraint_graph_id": self.constraint_graph_id,
            "truth": list(self.truth),
            "facts": [dict(item) for item in self.facts],
            "resized_height": self.resized_height,
            "resized_width": self.resized_width,
            "image_path": self.image_path,
        }
