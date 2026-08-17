"""Finite one-edit candidate enumeration and exact constraint projection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .constraint_system import World, satisfies_all_facts, validate_world


class ProjectionError(ValueError):
    """Base class for invalid or non-unique external projections."""


class NoProjectionError(ProjectionError):
    """No allowed one-edit candidate satisfies every fact."""


class AmbiguousProjectionError(ProjectionError):
    """More than one allowed one-edit candidate satisfies every fact."""


def _value_domain(value_domain: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(value_domain)
    except TypeError as error:
        raise TypeError("value_domain must be an iterable of integers") from error
    if not values:
        raise ValueError("value_domain must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("value_domain must contain only integers")
    if len(set(values)) != len(values):
        raise ValueError("value_domain must not contain duplicates")
    return tuple(sorted(values))


def enumerate_one_edit_candidates(
    observed: tuple[int, int, int, int] | Sequence[int],
    value_domain: Iterable[int],
) -> list[World]:
    """Enumerate ``Ham(x, observed) <= 1`` without duplicates, in sorted order."""

    canonical_observed = validate_world(observed, "observed")
    values = _value_domain(value_domain)
    candidates: set[World] = set()
    if all(value in values for value in canonical_observed):
        candidates.add(canonical_observed)
    for index in range(4):
        for value in values:
            if value == canonical_observed[index]:
                continue
            candidate = list(canonical_observed)
            candidate[index] = value
            if all(item in values for item in candidate):
                candidates.add(tuple(candidate))  # type: ignore[arg-type]
    return sorted(candidates)


def constraint_supported_candidates(
    observed: Sequence[int], facts: Iterable[object], value_domain: Iterable[int]
) -> tuple[World, ...]:
    fact_tuple = tuple(facts)
    return tuple(
        candidate
        for candidate in enumerate_one_edit_candidates(observed, value_domain)
        if satisfies_all_facts(candidate, fact_tuple)
    )


def unique_constraint_projection(
    observed: tuple[int, int, int, int] | Sequence[int],
    facts: Iterable[object],
    value_domain: Iterable[int],
) -> World:
    """Return the unique fact-supported one-edit world or fail closed."""

    supported = constraint_supported_candidates(observed, facts, value_domain)
    if not supported:
        raise NoProjectionError("no one-edit candidate satisfies all facts")
    if len(supported) != 1:
        raise AmbiguousProjectionError(
            f"constraint projection is ambiguous across {len(supported)} candidates"
        )
    return supported[0]
