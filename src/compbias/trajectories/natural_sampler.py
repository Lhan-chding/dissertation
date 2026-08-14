"""Collection of natural operational mediators from model adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .records import NaturalMediatorRecord


class NaturalForwardAdapter(Protocol):
    def natural_forward(
        self, sample: Mapping[str, object], interface: str, seed: int
    ) -> object: ...


def _sample_field(sample: Mapping[str, object], name: str) -> object:
    if name not in sample:
        raise ValueError(f"sample is missing required field {name!r}")
    return sample[name]


def _attribute(output: object, name: str) -> object:
    if not hasattr(output, name):
        raise TypeError(f"natural adapter output is missing attribute {name!r}")
    return getattr(output, name)


def collect_natural_mediators(
    model: NaturalForwardAdapter,
    samples: Sequence[Mapping[str, object]],
    *,
    interface_id: str,
    checkpoint_id: str,
    n_mediators: int,
    seeds: Sequence[int],
) -> tuple[NaturalMediatorRecord, ...]:
    """Run natural image-conditioned forwards; never edit the returned mediator."""

    if not samples:
        raise ValueError("samples must not be empty")
    seed_values = tuple(seeds)
    if (
        isinstance(n_mediators, bool)
        or not isinstance(n_mediators, int)
        or not 1 <= n_mediators <= 128
    ):
        raise ValueError("n_mediators must be an integer from 1 to 128")
    if len(seed_values) != n_mediators or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must contain exactly n_mediators unique values")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seed_values):
        raise ValueError("seeds must be non-negative integers")

    records: list[NaturalMediatorRecord] = []
    sample_ids: set[str] = set()
    for sample in samples:
        sample_id = _sample_field(sample, "sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a non-empty string")
        if sample_id in sample_ids:
            raise ValueError("samples contain duplicate sample_id values")
        sample_ids.add(sample_id)
        for rollout_index, seed in enumerate(seed_values):
            output = model.natural_forward(sample, interface_id, seed)
            records.append(
                NaturalMediatorRecord(
                    mediator_record_id=f"{sample_id}:natural:{rollout_index}",
                    sample_id=sample_id,
                    checkpoint_id=checkpoint_id,
                    interface_id=interface_id,
                    rollout_id=f"rollout-{rollout_index}",
                    image_path=str(_sample_field(sample, "image_path")),
                    question=str(_sample_field(sample, "question")),
                    gold_scene=_sample_field(sample, "gold_scene"),
                    natural_mediator_raw=str(_attribute(output, "raw")),
                    natural_mediator_parsed=_attribute(output, "parsed"),
                    parser_confidence=_attribute(output, "parser_confidence"),
                    parse_failed=_attribute(output, "parse_failed"),
                    error_type=str(_attribute(output, "error_type")),
                    task_severity=_attribute(output, "task_severity"),
                    original_answer=str(_attribute(output, "answer")),
                    original_reward=_attribute(output, "reward"),
                    rng_seed=seed,
                    activation_ref=_attribute(output, "activation_ref"),
                )
            )
    return tuple(records)
