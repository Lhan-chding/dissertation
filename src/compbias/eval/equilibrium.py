"""Immutable summaries of terminal equilibrium labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class EquilibriumSummary:
    total: int
    counts: Mapping[str, int]
    proportions: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "proportions", MappingProxyType(dict(self.proportions)))

    def to_mapping(self) -> dict[str, object]:
        return {
            "total": self.total,
            "counts": dict(self.counts),
            "proportions": dict(self.proportions),
        }


def _label(item: object, field: str) -> str:
    value = item.get(field) if isinstance(item, Mapping) else item
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"every terminal {field} must be a non-empty string")
    return value


def summarize_endpoint_labels(
    endpoints: Iterable[object], *, label_field: str = "endpoint_label"
) -> EquilibriumSummary:
    """Count terminal labels without retaining or mutating caller-owned records."""

    if not isinstance(label_field, str) or not label_field:
        raise ValueError("label_field must be a non-empty string")
    labels = tuple(_label(item, label_field) for item in endpoints)
    if not labels:
        raise ValueError("endpoints must not be empty")
    raw_counts = Counter(labels)
    counts = {label: raw_counts[label] for label in sorted(raw_counts)}
    proportions = {label: count / len(labels) for label, count in counts.items()}
    return EquilibriumSummary(total=len(labels), counts=counts, proportions=proportions)


summarize_equilibria = summarize_endpoint_labels
