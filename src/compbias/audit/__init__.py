"""Audit contracts for frozen and trainable model components."""

from .frozen_components import VLMRegimeSpec
from .representation_invariance import (
    RepresentationInvarianceReport,
    audit_representation_invariance,
)

__all__ = [
    "RepresentationInvarianceReport",
    "VLMRegimeSpec",
    "audit_representation_invariance",
]
