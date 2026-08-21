"""v5 reward-fiber and structural-support utilities."""

from .audit_v4_raw import (
    answer_fiber_statistics,
    candidate_margin_summary,
    capability_chain_summary,
    confirm_error_cardinality_summary,
    interface_revision_summary,
    phase8_transition_counts,
    support_budget_summary,
)
from .theory import (
    correction_alignment,
    correction_bearing_group_probability,
    exact_state_update,
    is_sparse_correction_identifiable,
    orbit_risk_upper_bound,
)

__all__ = [
    "answer_fiber_statistics",
    "candidate_margin_summary",
    "capability_chain_summary",
    "confirm_error_cardinality_summary",
    "correction_alignment",
    "correction_bearing_group_probability",
    "exact_state_update",
    "interface_revision_summary",
    "is_sparse_correction_identifiable",
    "orbit_risk_upper_bound",
    "phase8_transition_counts",
    "support_budget_summary",
]
