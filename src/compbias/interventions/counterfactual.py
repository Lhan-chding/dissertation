"""Construction of paired error-mechanism counterfactuals."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType


def _mapping(record: object) -> dict[str, object]:
    if isinstance(record, Mapping):
        return {str(key): _detach(value) for key, value in record.items()}
    to_mapping = getattr(record, "to_mapping", None)
    if callable(to_mapping):
        converted = to_mapping()
        if isinstance(converted, Mapping):
            return {str(key): _detach(value) for key, value in converted.items()}
    if is_dataclass(record) and not isinstance(record, type):
        return {field.name: _detach(getattr(record, field.name)) for field in fields(record)}
    raise TypeError("counterfactual records must be mappings or dataclasses")


def _detach(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _detach(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_detach(item) for item in value]
    return copy.deepcopy(value)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _mechanism(row: Mapping[str, object]) -> str:
    mechanism = row.get("error_mechanism")
    split_keys = row.get("split_keys")
    if mechanism is None and isinstance(split_keys, Mapping):
        mechanism = split_keys.get("error_mechanism")
    if not isinstance(mechanism, str) or not mechanism:
        raise ValueError("each paired record must identify its error_mechanism")
    return mechanism


def _mechanism_invariant_payload(row: Mapping[str, object]) -> dict[str, object]:
    """Return fields that must stay fixed in an error-mechanism intervention.

    The executable catalog is allowed to change because it defines the new
    mechanism.  All other observed and semantic fields, including visual
    style, must remain exactly paired.
    """

    payload = _mapping(row)
    payload.pop("error_mechanism", None)
    payload.pop("error_catalog", None)
    split_keys = payload.get("split_keys")
    if isinstance(split_keys, dict):
        split_keys.pop("error_mechanism", None)
    return payload


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    """One source record and its matched error-mechanism counterfactual."""

    sample_id: str
    source_error_mechanism: str
    counterfactual_error_mechanism: str
    source: Mapping[str, object]
    counterfactual: Mapping[str, object]
    shifted_factors: tuple[str, ...] = ("error_mechanism",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _freeze(self.source))
        object.__setattr__(self, "counterfactual", _freeze(self.counterfactual))

    def to_mapping(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "source_error_mechanism": self.source_error_mechanism,
            "counterfactual_error_mechanism": self.counterfactual_error_mechanism,
            "source": _detach(self.source),
            "counterfactual": _detach(self.counterfactual),
            "shifted_factors": list(self.shifted_factors),
        }


def _index(records: Iterable[object], label: str) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for record in records:
        row = _mapping(record)
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} record sample_id must be a non-empty string")
        if sample_id in indexed:
            raise ValueError(f"{label} records contain duplicate sample_id {sample_id!r}")
        indexed[sample_id] = row
    if not indexed:
        raise ValueError(f"{label} records must not be empty")
    return indexed


def pair_error_mechanism_shift(
    source_records: Iterable[object], counterfactual_records: Iterable[object]
) -> tuple[CounterfactualPair, ...]:
    """Pair records by ID and require an isolated error-mechanism change."""

    source = _index(source_records, "source")
    counterfactual = _index(counterfactual_records, "counterfactual")
    if source.keys() != counterfactual.keys():
        raise ValueError("source and counterfactual records must have paired sample_id values")

    pairs: list[CounterfactualPair] = []
    for sample_id in sorted(source):
        source_mechanism = _mechanism(source[sample_id])
        counterfactual_mechanism = _mechanism(counterfactual[sample_id])
        if source_mechanism == counterfactual_mechanism:
            raise ValueError(f"paired sample {sample_id!r} does not change the error_mechanism")
        if _mechanism_invariant_payload(source[sample_id]) != _mechanism_invariant_payload(
            counterfactual[sample_id]
        ):
            raise ValueError(
                f"paired sample {sample_id!r} changes fields other than "
                "error_mechanism or error_catalog"
            )
        pairs.append(
            CounterfactualPair(
                sample_id=sample_id,
                source_error_mechanism=source_mechanism,
                counterfactual_error_mechanism=counterfactual_mechanism,
                source=source[sample_id],
                counterfactual=counterfactual[sample_id],
            )
        )
    return tuple(pairs)
