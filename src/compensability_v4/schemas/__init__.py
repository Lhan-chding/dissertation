"""Closed immutable schemas for v4 scenes, observations, and records."""

from .observation import NaturalObservation
from .record import ExperimentRecord
from .scene import RecoveryScene

__all__ = ["ExperimentRecord", "NaturalObservation", "RecoveryScene"]
