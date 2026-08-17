"""Immutable experiment-record schema."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    record_id: str
    scene_id: str
    observation_id: str
    interface: str
    cue_condition: str
    prompt_hash: str
    tokenizer_version: str
    model_snapshot_hash: str
    output_text: str

    @classmethod
    def from_mapping(cls, mapping: dict[str, object]) -> "ExperimentRecord":
        return cls(
            record_id=str(mapping["record_id"]),
            scene_id=str(mapping["scene_id"]),
            observation_id=str(mapping["observation_id"]),
            interface=str(mapping["interface"]),
            cue_condition=str(mapping["cue_condition"]),
            prompt_hash=str(mapping["prompt_hash"]),
            tokenizer_version=str(mapping["tokenizer_version"]),
            model_snapshot_hash=str(mapping["model_snapshot_hash"]),
            output_text=str(mapping["output_text"]),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "scene_id": self.scene_id,
            "observation_id": self.observation_id,
            "interface": self.interface,
            "cue_condition": self.cue_condition,
            "prompt_hash": self.prompt_hash,
            "tokenizer_version": self.tokenizer_version,
            "model_snapshot_hash": self.model_snapshot_hash,
            "output_text": self.output_text,
        }
