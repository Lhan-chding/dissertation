"""Immutable scene contract for CVA-Constraint-Recovery-v4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from compensability_v4.data.splits import DatasetSplit
from compensability_v4.theory.constraint_system import fact_from_mapping, validate_world

from ._common import (
    freeze_json,
    require_closed_keys,
    require_identifier,
    require_integer,
    require_mapping,
    require_relative_path,
    thaw_json,
)

_FIELDS = {
    "scene_id",
    "split",
    "semantic_scene_id",
    "numeric_table_id",
    "constraint_graph_id",
    "truth",
    "facts",
    "resized_height",
    "resized_width",
    "image_path",
}


@dataclass(frozen=True, slots=True)
class RecoveryScene:
    scene_id: str
    split: DatasetSplit
    semantic_scene_id: str
    numeric_table_id: str
    constraint_graph_id: str
    truth: tuple[int, int, int, int]
    facts: tuple[Mapping[str, object], ...]
    resized_height: int
    resized_width: int
    image_path: str

    def __post_init__(self) -> None:
        for name in (
            "scene_id",
            "semantic_scene_id",
            "numeric_table_id",
            "constraint_graph_id",
        ):
            object.__setattr__(self, name, require_identifier(getattr(self, name), name))
        if not isinstance(self.split, DatasetSplit):
            try:
                object.__setattr__(self, "split", DatasetSplit(self.split))
            except (TypeError, ValueError) as error:
                raise ValueError("split is not registered for v4") from error
        object.__setattr__(self, "truth", validate_world(self.truth, "truth"))
        if not isinstance(self.facts, Sequence) or isinstance(self.facts, (str, bytes)):
            raise TypeError("facts must be a sequence")
        frozen_facts: list[Mapping[str, object]] = []
        for index, fact in enumerate(self.facts):
            mapping = require_mapping(fact, f"facts[{index}]")
            fact_from_mapping(mapping)
            frozen = freeze_json(mapping, f"facts[{index}]")
            frozen_facts.append(frozen)  # type: ignore[arg-type]
        if not frozen_facts:
            raise ValueError("facts must not be empty")
        object.__setattr__(self, "facts", tuple(frozen_facts))
        for name in ("resized_height", "resized_width"):
            value = require_integer(getattr(self, name), name, minimum=28)
            if value % 28:
                raise ValueError(f"{name} must be an integer multiple of 28")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "image_path", require_relative_path(self.image_path, "image_path"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RecoveryScene:
        mapping = require_mapping(value, "scene")
        require_closed_keys(mapping, required=_FIELDS, name="scene")
        facts = mapping["facts"]
        if not isinstance(facts, Sequence) or isinstance(facts, (str, bytes)):
            raise TypeError("facts must be a sequence")
        return cls(
            scene_id=mapping["scene_id"],  # type: ignore[arg-type]
            split=mapping["split"],  # type: ignore[arg-type]
            semantic_scene_id=mapping["semantic_scene_id"],  # type: ignore[arg-type]
            numeric_table_id=mapping["numeric_table_id"],  # type: ignore[arg-type]
            constraint_graph_id=mapping["constraint_graph_id"],  # type: ignore[arg-type]
            truth=mapping["truth"],  # type: ignore[arg-type]
            facts=tuple(facts),  # type: ignore[arg-type]
            resized_height=mapping["resized_height"],  # type: ignore[arg-type]
            resized_width=mapping["resized_width"],  # type: ignore[arg-type]
            image_path=mapping["image_path"],  # type: ignore[arg-type]
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "split": self.split.value,
            "semantic_scene_id": self.semantic_scene_id,
            "numeric_table_id": self.numeric_table_id,
            "constraint_graph_id": self.constraint_graph_id,
            "truth": list(self.truth),
            "facts": [thaw_json(fact) for fact in self.facts],
            "resized_height": self.resized_height,
            "resized_width": self.resized_width,
            "image_path": self.image_path,
        }
