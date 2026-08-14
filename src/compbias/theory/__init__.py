"""Numerically verified selection, transport, risk, and regime theory."""

from .crossed_risk import CrossedRiskResult, crossed_risk_decomposition
from .density_ratio_identity import SelectionRatioResult, checkpoint_selection_ratio
from .frozen_regimes import FrozenRegimeAudit, audit_frozen_regime
from .natural_selection import (
    NaturalSelectionShift,
    natural_binary_selected_distribution,
    natural_selection_shift,
)
from .task_metric import task_induced_distance
from .transport_bounds import TransportDiagnostic, total_variation, transport_diagnostic

__all__ = [
    "CrossedRiskResult",
    "FrozenRegimeAudit",
    "NaturalSelectionShift",
    "SelectionRatioResult",
    "TransportDiagnostic",
    "audit_frozen_regime",
    "checkpoint_selection_ratio",
    "crossed_risk_decomposition",
    "natural_binary_selected_distribution",
    "natural_selection_shift",
    "task_induced_distance",
    "total_variation",
    "transport_diagnostic",
]
