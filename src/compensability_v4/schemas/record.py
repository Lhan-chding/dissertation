"""Immutable model-output record with reproducibility provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ._common import (
    require_closed_keys,
    require_identifier,
    require_mapping,
    require_sha256,
)

_FIELDS = {
    "record_id",
    "scene_id",
    "observation_id",
    "interface",
    "cue_condition",
    "prompt_hash",
    "tokenizer_version",
    "model_snapshot_hash",
    "output_text",
}
_INTERFACES = {
    "symbolic_downstream_recovery",
    "soft_report_diagnostic",
    "candidate_world_diagnostic",
    "same_conversation_visual_revision",
    "natural_visual_revision",
    "exact_cached_natural_continuation",
    "text_replay",
}
_CUES = {"no_cue", "valid_cue", "sham_cue", "counterfactual_cue"}


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

    def __post_init__(self) -> None:
        for name in ("record_id", "scene_id", "observation_id", "tokenizer_version"):
            object.__setattr__(self, name, require_identifier(getattr(self, name), name))
        interface = require_identifier(self.interface, "interface")
        if interface not in _INTERFACES:
            raise ValueError("interface is not registered by v4")
        object.__setattr__(self, "interface", interface)
        cue = require_identifier(self.cue_condition, "cue_condition")
        if cue not in _CUES:
            raise ValueError("cue_condition is not registered by v4")
        object.__setattr__(self, "cue_condition", cue)
        object.__setattr__(self, "prompt_hash", require_sha256(self.prompt_hash, "prompt_hash"))
        object.__setattr__(
            self,
            "model_snapshot_hash",
            require_sha256(self.model_snapshot_hash, "model_snapshot_hash"),
        )
        if not isinstance(self.output_text, str):
            raise TypeError("output_text must be a string")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ExperimentRecord:
        mapping = require_mapping(value, "experiment record")
        require_closed_keys(mapping, required=_FIELDS, name="experiment record")
        return cls(**mapping)  # type: ignore[arg-type]

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
