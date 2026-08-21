"""Finite, side-effect-free contracts for the v5 correction theory."""

from .correction_alignment import alignment_coefficient, natural_gradient_direction
from .equivariance import equivariance_defect, orbit_risk, permute_world
from .grpo_signal import (
    answer_group_signal,
    correction_bearing_answer_signal,
    state_group_signal,
)
from .reward_fibers import conditional_distribution, kl_reward_projection
from .sparse_syndrome import (
    enumerate_sparse_errors,
    is_sparse_correction_unique,
    residual_signature,
)

__all__ = [
    "alignment_coefficient",
    "answer_group_signal",
    "conditional_distribution",
    "correction_bearing_answer_signal",
    "enumerate_sparse_errors",
    "equivariance_defect",
    "is_sparse_correction_unique",
    "kl_reward_projection",
    "natural_gradient_direction",
    "orbit_risk",
    "permute_world",
    "residual_signature",
    "state_group_signal",
]
