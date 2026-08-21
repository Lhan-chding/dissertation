"""Finite permutation actions, equivariance defect, and orbit risk."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Integral
from typing import TypeVar

Value = TypeVar("Value")
World = tuple[Value, ...]


def _tuple(values: object, *, name: str) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} must be an iterable")
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _permutation(values: object, *, dimension: int) -> tuple[int, ...]:
    raw = _tuple(values, name="permutation")
    if len(raw) != dimension:
        raise ValueError("permutation must have the same length as the world")
    normalized: list[int] = []
    for position, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"permutation[{position}] must be an integer")
        normalized.append(int(value))
    if set(normalized) != set(range(dimension)):
        raise ValueError("permutation must be a bijection of the world indices")
    return tuple(normalized)


def permute_world(world: Iterable[Value], permutation: Iterable[int]) -> tuple[Value, ...]:
    """Apply a finite coordinate permutation, returning a new immutable world."""

    normalized_world = _tuple(world, name="world")
    normalized_permutation = _permutation(permutation, dimension=len(normalized_world))
    return tuple(normalized_world[index] for index in normalized_permutation)  # type: ignore[return-value]


def _decoder_output(
    decoder: Callable[[tuple[object, ...]], Iterable[object]],
    instance: tuple[object, ...],
) -> tuple[object, ...]:
    output = _tuple(decoder(instance), name="decoder output")
    if len(output) != len(instance):
        raise ValueError("decoder output must have the same length as its instance")
    return output


def _audit_inputs(
    decoder: object,
    instances: Iterable[Iterable[object]],
    permutations: Iterable[Iterable[int]],
) -> tuple[
    Callable[[tuple[object, ...]], Iterable[object]],
    tuple[tuple[object, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    if not callable(decoder):
        raise TypeError("decoder must be callable")
    raw_instances = _tuple(instances, name="instances")
    normalized_instances = tuple(
        _tuple(instance, name=f"instances[{index}]")
        for index, instance in enumerate(raw_instances)
    )
    dimension = len(normalized_instances[0])
    if any(len(instance) != dimension for instance in normalized_instances):
        raise ValueError("all instances must have the same dimension")
    raw_permutations = _tuple(permutations, name="permutations")
    normalized_permutations = tuple(
        _permutation(permutation, dimension=dimension) for permutation in raw_permutations
    )
    return decoder, normalized_instances, normalized_permutations  # type: ignore[return-value]


def equivariance_defect(
    decoder: Callable[[tuple[Value, ...]], Iterable[Value]],
    instances: Iterable[Iterable[Value]],
    permutations: Iterable[Iterable[int]],
) -> float:
    r"""Return ``P[D(gm) != gD(m)]`` under the finite empirical product measure."""

    canonical_decoder, canonical_instances, canonical_permutations = _audit_inputs(
        decoder, instances, permutations
    )
    mismatches = 0
    comparisons = len(canonical_instances) * len(canonical_permutations)
    for instance in canonical_instances:
        decoded = _decoder_output(canonical_decoder, instance)
        for permutation in canonical_permutations:
            transformed_instance = permute_world(instance, permutation)
            transformed_decoded = _decoder_output(canonical_decoder, transformed_instance)
            expected = permute_world(decoded, permutation)
            mismatches += transformed_decoded != expected
    return mismatches / comparisons


def orbit_risk(
    decoder: Callable[[tuple[Value, ...]], Iterable[Value]],
    instances: Iterable[Iterable[Value]],
    truths: Iterable[Iterable[Value]],
    permutations: Iterable[Iterable[int]],
) -> float:
    r"""Return ``P[D(gm) != gx]`` under the finite empirical orbit distribution."""

    canonical_decoder, canonical_instances, canonical_permutations = _audit_inputs(
        decoder, instances, permutations
    )
    raw_truths = _tuple(truths, name="truths")
    canonical_truths = tuple(
        _tuple(truth, name=f"truths[{index}]") for index, truth in enumerate(raw_truths)
    )
    if len(canonical_truths) != len(canonical_instances):
        raise ValueError("truths must contain one world per instance")
    dimension = len(canonical_instances[0])
    if any(len(truth) != dimension for truth in canonical_truths):
        raise ValueError("every truth must have the instance dimension")

    errors = 0
    comparisons = len(canonical_instances) * len(canonical_permutations)
    for instance, truth in zip(canonical_instances, canonical_truths, strict=True):
        for permutation in canonical_permutations:
            transformed_instance = permute_world(instance, permutation)
            decoded = _decoder_output(canonical_decoder, transformed_instance)
            transformed_truth = permute_world(truth, permutation)
            errors += decoded != transformed_truth
    return errors / comparisons


__all__ = ["equivariance_defect", "orbit_risk", "permute_world"]
