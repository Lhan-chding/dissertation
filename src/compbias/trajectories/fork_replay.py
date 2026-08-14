"""Exact natural mediator replay with explicit image access removal."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from compbias.identification.interface_spec import InterfaceSpec

from .records import ForkedContinuationRecord, NaturalMediatorRecord


class ForkReplayAdapter(Protocol):
    def fork_continuation(
        self,
        record: NaturalMediatorRecord,
        interface: str | InterfaceSpec,
        seed: int,
        *,
        image_access: bool,
    ) -> object: ...


def _attribute(output: object, name: str) -> object:
    if not hasattr(output, name):
        raise TypeError(f"fork adapter output is missing attribute {name!r}")
    return getattr(output, name)


def fork_natural_mediators(
    model: ForkReplayAdapter,
    records: Sequence[NaturalMediatorRecord],
    *,
    interface_id: str | InterfaceSpec,
    n_forks: int,
    seeds: Sequence[int],
) -> tuple[ForkedContinuationRecord, ...]:
    """Fork every exact natural state with image access hard-disabled."""

    if not records:
        raise ValueError("records must not be empty")
    seed_values = tuple(seeds)
    if isinstance(n_forks, bool) or not isinstance(n_forks, int) or not 1 <= n_forks <= 64:
        raise ValueError("n_forks must be an integer from 1 to 64")
    if len(seed_values) != n_forks or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must contain exactly n_forks unique values")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seed_values):
        raise ValueError("seeds must be non-negative integers")

    forks: list[ForkedContinuationRecord] = []
    for record in records:
        if not isinstance(record, NaturalMediatorRecord):
            raise TypeError("records must contain NaturalMediatorRecord values")
        for fork_index, seed in enumerate(seed_values):
            output = model.fork_continuation(
                record,
                interface_id,
                seed,
                image_access=False,
            )
            forks.append(
                ForkedContinuationRecord(
                    mediator_record_id=record.mediator_record_id,
                    fork_id=f"{record.mediator_record_id}:fork:{fork_index}",
                    image_cut_mode="remove_image_context",
                    continuation_seed=seed,
                    answer=str(_attribute(output, "answer")),
                    reward=_attribute(output, "reward"),
                    replay_fidelity_metadata=_attribute(output, "metadata"),
                    source_kind="natural",
                )
            )
    return tuple(forks)
