"""Claim-safe component regimes for the confirmatory VLM experiments."""

from __future__ import annotations

from dataclasses import dataclass

_UPDATES = frozenset({"frozen", "lora", "full"})
_INTERFACES = frozenset(
    {
        "natural_evidence_prefix",
        "structured_scene_json",
        "post_projector_activation",
        "mid_fusion_activation",
        "pre_answer_evidence_state",
    }
)
_REGIMES = {
    "lm_only": ("frozen", "frozen", "lora"),
    "projector_lm": ("frozen", "full", "lora"),
    "vision_lora": ("lora", "full", "lora"),
    "max_end_to_end": ("full", "full", "full"),
}


@dataclass(frozen=True, slots=True)
class VLMRegimeSpec:
    """Immutable update boundary and the claims it can identify.

    ``F0`` means the interface is frozen, ``F1`` means the upstream visual
    acquisition is frozen but the readout/mediator can change, and ``F2``
    means visual acquisition itself can change.  These labels prevent a
    frozen visual encoder experiment from being described as improved visual
    acquisition.
    """

    regime_id: str
    vision_update: str
    projector_update: str
    language_update: str

    def __post_init__(self) -> None:
        if self.regime_id not in _REGIMES:
            raise ValueError(f"unknown VLM regime: {self.regime_id!r}")
        updates = (self.vision_update, self.projector_update, self.language_update)
        if any(value not in _UPDATES for value in updates):
            raise ValueError("component updates must be frozen, lora, or full")
        if updates != _REGIMES[self.regime_id]:
            raise ValueError(
                f"component updates do not match the registered {self.regime_id} regime"
            )

    @property
    def acquisition_frozen(self) -> bool:
        return self.vision_update == "frozen"

    @property
    def allowed_claims(self) -> frozenset[str]:
        common = {"reasoning_change", "interaction_change"}
        if self.acquisition_frozen:
            common.add("readout_change")
        else:
            common.add("operational_perception_change")
        return frozenset(common)

    def interface_regime(self, interface: str) -> str:
        if interface not in _INTERFACES:
            raise ValueError(f"unknown mediator interface: {interface!r}")
        if interface == "post_projector_activation":
            if self.vision_update == "frozen" and self.projector_update == "frozen":
                return "F0"
            if self.vision_update == "frozen":
                return "F1"
            return "F2"
        return "F1" if self.acquisition_frozen else "F2"

    def to_mapping(self) -> dict[str, object]:
        return {
            "id": self.regime_id,
            "vision_update": self.vision_update,
            "projector_update": self.projector_update,
            "language_update": self.language_update,
            "acquisition_frozen": self.acquisition_frozen,
            "allowed_claims": sorted(self.allowed_claims),
            "forbidden_claims": (["acquisition_improvement"] if self.acquisition_frozen else []),
        }
