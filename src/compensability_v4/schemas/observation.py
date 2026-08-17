"""Immutable natural-observation schema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NaturalObservation:
    observation_id: str
    scene_id: str
    observed_values: tuple[int, int, int, int]
    error_index: int
    stage1_model_hash: str
    image_grid_thw: tuple[int, int, int]
    visual_token_count: int

    @classmethod
    def from_mapping(cls, mapping: dict[str, object]) -> "NaturalObservation":
        observed_values = mapping["observed_values"]
        if not isinstance(observed_values, list) or len(observed_values) != 4:
            raise ValueError("observed_values must contain four integers")
        if any(type(item) is not int for item in observed_values):
            raise TypeError("observed_values must contain exact integers")
        error_index = mapping["error_index"]
        if type(error_index) is not int or not 0 <= error_index < 4:
            raise ValueError("error_index must be an integer from 0 to 3")
        grid = mapping["image_grid_thw"]
        if not isinstance(grid, list) or len(grid) != 3 or any(type(item) is not int for item in grid):
            raise TypeError("image_grid_thw must contain exactly three integers")
        return cls(
            observation_id=str(mapping["observation_id"]),
            scene_id=str(mapping["scene_id"]),
            observed_values=tuple(observed_values),
            error_index=error_index,
            stage1_model_hash=str(mapping["stage1_model_hash"]),
            image_grid_thw=(grid[0], grid[1], grid[2]),
            visual_token_count=int(mapping["visual_token_count"]),
        )

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
