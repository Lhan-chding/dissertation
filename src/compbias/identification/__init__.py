"""Operational identification contracts for perception--reasoning compensation."""

from .interface_spec import InterfaceSpec
from .nonidentifiability import RelabeledFactorization, relabel_factorization
from .partial_identification import (
    InterfaceGammaEstimate,
    PartialIdentificationResult,
    robust_compensation_interval,
)
from .validity_gates import (
    InterfaceValidityReport,
    InterfaceValidityThresholds,
    evaluate_interface_validity,
)

__all__ = [
    "InterfaceGammaEstimate",
    "InterfaceSpec",
    "InterfaceValidityReport",
    "InterfaceValidityThresholds",
    "PartialIdentificationResult",
    "RelabeledFactorization",
    "evaluate_interface_validity",
    "relabel_factorization",
    "robust_compensation_interval",
]
