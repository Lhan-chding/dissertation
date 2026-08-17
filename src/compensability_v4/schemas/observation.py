"""Immutable natural Stage-1 observation with visual-token provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from compensability_v4.theory.constraint_system import validate_world

from ._common import (
    require_closed_keys,
    require_identifier,
    require_integer,
    require_mapping,
    require_sha256,
)

_FIELDS = {
    "observation_id",
    "scene_id",
    "observed_values",
    "error_index",
    "stage1_model_hash",
    "image_grid_thw",
    "visual_token_count",
}


@dataclass(frozen=True, slots=True)
class NaturalObservation:
    observation_id: str
    scene_id: str
    observed_values: tuple[int, int, int, int]
    error_index: int
    stage1_model_hash: str
    image_grid_thw: tuple[int, int, int]
    visual_token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", require_identifier(self.observation_id, "observation_id")
        )
        object.__setattr__(self, "scene_id", require_identifier(self.scene_id, "scene_id"))
        object.__setattr__(
            self, "observed_values", validate_world(self.observed_values, "observed_values")
        )
        error_index = require_integer(self.error_index, "error_index")
        if not 0 <= error_index < 4:
            raise ValueError("error_index must lie in [0, 3]")
        object.__setattr__(self, "error_index", error_index)
        object.__setattr__(
            self, "stage1_model_hash", require_sha256(self.stage1_model_hash, "stage1_model_hash")
        )
        if (
            not isinstance(self.image_grid_thw, Sequence)
            or isinstance(self.image_grid_thw, (str, bytes))
            or len(self.image_grid_thw) != 3
        ):
            raise ValueError("image_grid_thw must contain exactly three positive integers")
        grid = tuple(
            require_integer(item, f"image_grid_thw[{index}]", minimum=1)
            for index, item in enumerate(self.image_grid_thw)
        )
        object.__setattr__(self, "image_grid_thw", grid)
        token_count = require_integer(self.visual_token_count, "visual_token_count", minimum=1)
        if grid[1] % 2 or grid[2] % 2 or token_count != grid[0] * grid[1] * grid[2] // 4:
            raise ValueError("visual_token_count must match the Qwen 2x2 patch-merger grid")
        object.__setattr__(self, "visual_token_count", token_count)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> NaturalObservation:
        mapping = require_mapping(value, "observation")
        require_closed_keys(mapping, required=_FIELDS, name="observation")
        return cls(**mapping)  # type: ignore[arg-type]

    def to_mapping(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "scene_id": self.scene_id,
            "observed_values": list(self.observed_values),
            "error_index": self.error_index,
            "stage1_model_hash": self.stage1_model_hash,
            "image_grid_thw": list(self.image_grid_thw),
            "visual_token_count": self.visual_token_count,
        }
