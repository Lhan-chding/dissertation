"""Pure measurements for the I0-I4 Qwen interface ladder."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Interface(str, Enum):
    I0_HARD_TEXT = "I0_hard_text_symbolic_recovery"
    I1_SOFT_REPORT = "I1_soft_report_diagnostic"
    I2_CANDIDATE_WORLD = "I2_candidate_world_diagnostic"
    I3_SAME_CONVERSATION = "I3_same_conversation_visual_revision"
    I4_EXACT_CACHE = "I4_exact_cached_natural_continuation"


class CueCondition(str, Enum):
    NO_CUE = "no_cue"
    VALID_CUE = "valid_cue"
    SHAM_CUE = "sham_cue"
    COUNTERFACTUAL_CUE = "counterfactual_cue"


@dataclass(frozen=True, slots=True)
class InterfaceOutcome:
    scene_id: str
    family: str
    interface: Interface
    condition: CueCondition
    true_world: tuple[int, int, int, int]
    observed_world: tuple[int, int, int, int]
    output_world: tuple[int, int, int, int] | None
    counterfactual_world: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not self.scene_id or not self.family:
            raise ValueError("scene_id and family are required")
        for name, world in (
            ("true_world", self.true_world),
            ("observed_world", self.observed_world),
        ):
            if len(world) != 4 or any(type(value) is not int for value in world):
                raise ValueError(f"{name} must contain exactly four integers")
        for name, world in (
            ("output_world", self.output_world),
            ("counterfactual_world", self.counterfactual_world),
        ):
            if world is not None and (
                len(world) != 4 or any(type(value) is not int for value in world)
            ):
                raise ValueError(f"{name} must be null or exactly four integers")
        if self.condition is CueCondition.COUNTERFACTUAL_CUE and self.counterfactual_world is None:
            raise ValueError("counterfactual cue requires its registered legal target world")

    @property
    def exact_world_recovery(self) -> bool:
        return self.output_world == self.true_world

    @property
    def observation_copy(self) -> bool:
        return self.output_world == self.observed_world

    @property
    def counterfactual_compliance(self) -> bool | None:
        if self.condition is not CueCondition.COUNTERFACTUAL_CUE:
            return None
        return self.output_world == self.counterfactual_world


@dataclass(frozen=True, slots=True)
class RevisionDecomposition:
    i0_no_cue_accuracy: float
    i4_no_cue_accuracy: float
    i4_valid_cue_accuracy: float
    spontaneous_visual_revision: float
    fact_conditioned_revision: float
    counterfactual_compliance: float | None


def _mean(indicators: Iterable[bool]) -> float:
    values = tuple(indicators)
    if not values:
        raise ValueError("an empirical rate requires at least one scene")
    return sum(values) / len(values)


def validate_interface_ladder(records: Iterable[InterfaceOutcome]) -> tuple[InterfaceOutcome, ...]:
    """Enforce pairing and uniqueness without imposing an empirical success threshold."""

    frozen = tuple(records)
    if not frozen:
        raise ValueError("interface ladder requires at least one scene")
    keys = [(row.scene_id, row.interface, row.condition) for row in frozen]
    if len(keys) != len(set(keys)):
        raise ValueError("interface records contain duplicate scene/interface/condition cells")
    by_scene: dict[str, set[tuple[Interface, CueCondition]]] = {}
    for row in frozen:
        by_scene.setdefault(row.scene_id, set()).add((row.interface, row.condition))
    primary_interfaces = (
        Interface.I0_HARD_TEXT,
        Interface.I3_SAME_CONVERSATION,
        Interface.I4_EXACT_CACHE,
    )
    required = {
        (interface, condition) for interface in primary_interfaces for condition in CueCondition
    }
    incomplete = sorted(scene for scene, cells in by_scene.items() if not required.issubset(cells))
    if incomplete:
        raise ValueError(f"interface ladder is incomplete for scenes: {incomplete[:5]}")
    return frozen


def revision_decomposition(records: Iterable[InterfaceOutcome]) -> RevisionDecomposition:
    """Compute plan-defined I0/I4 effects, with scenes retained as the unit."""

    rows = validate_interface_ladder(records)

    def selected(interface: Interface, condition: CueCondition) -> tuple[InterfaceOutcome, ...]:
        return tuple(
            row for row in rows if row.interface is interface and row.condition is condition
        )

    i0_no_rows = selected(Interface.I0_HARD_TEXT, CueCondition.NO_CUE)
    i4_no_rows = selected(Interface.I4_EXACT_CACHE, CueCondition.NO_CUE)
    i4_valid_rows = selected(Interface.I4_EXACT_CACHE, CueCondition.VALID_CUE)
    i0_no = _mean(row.exact_world_recovery for row in i0_no_rows)
    i4_no = _mean(row.exact_world_recovery for row in i4_no_rows)
    i4_valid = _mean(row.exact_world_recovery for row in i4_valid_rows)
    counterfactual = selected(Interface.I4_EXACT_CACHE, CueCondition.COUNTERFACTUAL_CUE)
    compliance = _mean(bool(row.counterfactual_compliance) for row in counterfactual)
    return RevisionDecomposition(
        i0_no_cue_accuracy=i0_no,
        i4_no_cue_accuracy=i4_no,
        i4_valid_cue_accuracy=i4_valid,
        spontaneous_visual_revision=i4_no - i0_no,
        fact_conditioned_revision=i4_valid - i4_no,
        counterfactual_compliance=compliance,
    )


def interface_claim_name(interface: Interface) -> str:
    """Return the only permitted mechanism label for an interface."""

    if interface is Interface.I0_HARD_TEXT:
        return "symbolic_downstream_recovery"
    if interface in {Interface.I3_SAME_CONVERSATION, Interface.I4_EXACT_CACHE}:
        return "natural_visual_revision"
    return "intervention_diagnostic"


__all__ = [
    "CueCondition",
    "Interface",
    "InterfaceOutcome",
    "RevisionDecomposition",
    "interface_claim_name",
    "revision_decomposition",
    "validate_interface_ladder",
]
