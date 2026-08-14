"""Exact preregistered call accounting for Recoverability v1."""

from __future__ import annotations

from dataclasses import dataclass

from .config import RecoverabilityProtocol


@dataclass(frozen=True, slots=True)
class DesignReport:
    phase_n_scenes: int
    bridge_scenes: int
    bridge_model_calls: int
    phase_c_intake_scenes: int
    selected_family_quotas: tuple[tuple[str, int], ...]
    selected_independent_scenes: int
    arms: tuple[str, ...]
    forks_per_arm: int
    total_downstream_forks: int
    confirmatory_forks: int
    diagnostic_forks: int
    independent_analysis_unit: str


def build_design_report(protocol: RecoverabilityProtocol) -> DesignReport:
    if not isinstance(protocol, RecoverabilityProtocol):
        raise TypeError("protocol must be a RecoverabilityProtocol")
    selected = sum(count for _family, count in protocol.phase_c.selected_family_quotas)
    total = selected * len(protocol.phase_c.arms) * protocol.phase_c.forks_per_arm
    confirmatory = (
        selected * len(protocol.phase_c.confirmatory_arms) * protocol.phase_c.forks_per_arm
    )
    diagnostic = selected * len(protocol.phase_c.diagnostic_arms) * protocol.phase_c.forks_per_arm
    return DesignReport(
        phase_n_scenes=protocol.phase_n.scenes,
        bridge_scenes=protocol.bridge.scenes,
        bridge_model_calls=protocol.bridge.scenes * protocol.bridge.protocols_per_scene,
        phase_c_intake_scenes=protocol.phase_c.intake_scenes,
        selected_family_quotas=protocol.phase_c.selected_family_quotas,
        selected_independent_scenes=selected,
        arms=protocol.phase_c.arms,
        forks_per_arm=protocol.phase_c.forks_per_arm,
        total_downstream_forks=total,
        confirmatory_forks=confirmatory,
        diagnostic_forks=diagnostic,
        independent_analysis_unit="semantic_scene",
    )
