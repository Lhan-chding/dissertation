"""Construction of separately typed synthetic mediator stress tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .records import SyntheticMediatorRecord


class SyntheticMediatorAdapter(Protocol):
    def synthetic_mediator(
        self,
        sample: Mapping[str, object],
        interface: str,
        error_type: str,
        seed: int,
    ) -> Mapping[str, object]: ...


def build_synthetic_mediators(
    model: SyntheticMediatorAdapter,
    samples: Sequence[Mapping[str, object]],
    *,
    interface_id: str,
    checkpoint_id: str,
    error_types: Sequence[str],
    seeds: Sequence[int],
) -> tuple[SyntheticMediatorRecord, ...]:
    """Build edited states only on the explicitly synthetic evidence path."""

    if not samples:
        raise ValueError("samples must not be empty")
    errors = tuple(error_types)
    seed_values = tuple(seeds)
    if not errors or len(errors) != len(seed_values):
        raise ValueError("error_types and seeds must be non-empty and equally sized")
    if len(set(zip(errors, seed_values, strict=True))) != len(errors):
        raise ValueError("error_type and seed pairs must be unique")
    records: list[SyntheticMediatorRecord] = []
    for sample in samples:
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("each sample requires a non-empty sample_id")
        for index, (error_type, seed) in enumerate(zip(errors, seed_values, strict=True)):
            if not isinstance(error_type, str) or not error_type:
                raise ValueError("error_types must be non-empty strings")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seeds must be non-negative integers")
            payload = model.synthetic_mediator(sample, interface_id, error_type, seed)
            if not isinstance(payload, Mapping):
                raise TypeError("synthetic_mediator must return a mapping")
            records.append(
                SyntheticMediatorRecord(
                    mediator_record_id=f"{sample_id}:synthetic:{index}",
                    sample_id=sample_id,
                    checkpoint_id=checkpoint_id,
                    interface_id=interface_id,
                    target_error_type=error_type,
                    construction_method=payload.get("construction_method"),
                    synthetic_mediator=payload.get("synthetic_mediator"),
                    nearest_natural_state_ids=tuple(payload.get("nearest_natural_state_ids", ())),
                    transport_signature=tuple(payload.get("transport_signature", ())),
                )
            )
    return tuple(records)
