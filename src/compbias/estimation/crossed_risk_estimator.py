"""Paired sample-level estimator for crossed perception/reasoner risks."""

from __future__ import annotations

from dataclasses import dataclass

from compbias.theory.crossed_risk import CrossedRiskResult, crossed_risk_decomposition
from compbias.trajectories.records import CrossedRiskRecord


@dataclass(frozen=True, slots=True)
class SampleCrossedRisk:
    sample_id: str
    interface_id: str
    seed: int
    result: CrossedRiskResult


@dataclass(frozen=True, slots=True)
class CrossedRiskEstimate:
    per_sample: tuple[SampleCrossedRisk, ...]
    aggregate: CrossedRiskResult
    maximum_identity_residual: float


def estimate_crossed_risks(records: tuple[CrossedRiskRecord, ...]) -> CrossedRiskEstimate:
    """Require exactly four paired cells before aggregating bounded losses."""

    if not records:
        raise ValueError("records must not be empty")
    groups: dict[tuple[str, str, int], dict[tuple[str, str], float]] = {}
    for record in records:
        if not isinstance(record, CrossedRiskRecord):
            raise TypeError("records must contain CrossedRiskRecord values")
        group = groups.setdefault((record.sample_id, record.interface_id, record.seed), {})
        cell = (record.perception_source, record.reasoner_source)
        if cell in group:
            raise ValueError("crossed risk cells must be unique within sample/interface/seed")
        group[cell] = record.loss
    required = {
        ("model", "model"),
        ("oracle", "model"),
        ("model", "oracle"),
        ("oracle", "oracle"),
    }
    estimates: list[SampleCrossedRisk] = []
    for (sample_id, interface_id, seed), cells in sorted(groups.items()):
        if set(cells) != required:
            raise ValueError("each sample/interface/seed requires all four crossed risk cells")
        result = crossed_risk_decomposition(
            l_mm=cells[("model", "model")],
            l_om=cells[("oracle", "model")],
            l_mo=cells[("model", "oracle")],
            l_oo=cells[("oracle", "oracle")],
        )
        estimates.append(SampleCrossedRisk(sample_id, interface_id, seed, result))
    aggregate = crossed_risk_decomposition(
        l_mm=sum(item.result.l_mm for item in estimates) / len(estimates),
        l_om=sum(item.result.l_om for item in estimates) / len(estimates),
        l_mo=sum(item.result.l_mo for item in estimates) / len(estimates),
        l_oo=sum(item.result.l_oo for item in estimates) / len(estimates),
    )
    return CrossedRiskEstimate(
        per_sample=tuple(estimates),
        aggregate=aggregate,
        maximum_identity_residual=max(item.result.identity_residual for item in estimates),
    )
