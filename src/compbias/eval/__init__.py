"""Deterministic evaluation utilities for compensability experiments."""

from .compensability import build_compensability_long_table, per_prompt_covariances
from .decomposition import (
    additive_errors,
    bregman_decomposition,
    coupling_metrics,
    squared_decomposition,
)
from .ood import compute_ood_metrics
from .statistics import bootstrap_mean_ci, holm_adjust, paired_bootstrap_delta

__all__ = [
    "additive_errors",
    "bootstrap_mean_ci",
    "bregman_decomposition",
    "build_compensability_long_table",
    "compute_ood_metrics",
    "coupling_metrics",
    "holm_adjust",
    "paired_bootstrap_delta",
    "per_prompt_covariances",
    "squared_decomposition",
]
