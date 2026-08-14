"""Controlled interventions for causal compensability experiments."""

from .counterfactual import CounterfactualPair, pair_error_mechanism_shift
from .error_catalog import apply_catalog_error, index_error_catalog, validate_error_catalog
from .state_injection import InterventionRecord, run_state_injection

__all__ = [
    "CounterfactualPair",
    "InterventionRecord",
    "apply_catalog_error",
    "index_error_catalog",
    "pair_error_mechanism_shift",
    "run_state_injection",
    "validate_error_catalog",
]
