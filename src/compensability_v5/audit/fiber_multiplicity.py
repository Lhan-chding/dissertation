"""Exact answer-fiber enumeration for the frozen v4 four-value task."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

World = tuple[int, int, int, int]
RawRow = Mapping[str, Any]


def validate_world(value: object, *, label: str) -> World:
    """Return a four-integer world, rejecting booleans and lossy coercions."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError(f"{label} must contain exactly four integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{label} must contain exactly four integers")
    return value[0], value[1], value[2], value[3]


def validate_domain(value_domain: Iterable[int]) -> tuple[int, ...]:
    """Freeze a nonempty, duplicate-free integer candidate domain."""

    values = tuple(value_domain)
    if not values:
        raise ValueError("value_domain must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("value_domain must contain only integers")
    if len(values) != len(set(values)):
        raise ValueError("value_domain must not contain duplicates")
    return tuple(sorted(values))


def enumerate_one_edit_worlds(
    observed: Sequence[int], value_domain: Iterable[int] = range(2, 19)
) -> tuple[World, ...]:
    """Enumerate unique worlds at Hamming distance at most one from ``observed``.

    The observed world is included only when it is itself inside the frozen domain.
    If any observed coordinate is out of domain, no admissible one-edit world exists,
    because changing one coordinate cannot make every unchanged coordinate admissible.
    """

    frozen = validate_world(observed, label="observed")
    domain = validate_domain(value_domain)
    if any(value not in domain for value in frozen):
        return ()
    candidates: set[World] = {frozen}
    for index in range(4):
        for replacement in domain:
            if replacement == frozen[index]:
                continue
            candidate = list(frozen)
            candidate[index] = replacement
            candidates.add((candidate[0], candidate[1], candidate[2], candidate[3]))
    return tuple(sorted(candidates))


def apply_answer_operation(world: Sequence[int], operation: str) -> int:
    """Apply the three frozen v4 chart-question operations."""

    values = validate_world(world, label="world")
    if operation == "sum":
        return values[0] + values[1]
    if operation == "difference":
        return values[0] - values[1]
    if operation == "max_minus_min":
        return max(values) - min(values)
    raise ValueError(f"unsupported answer operation: {operation!r}")


def answer_fiber_size(
    observed: Sequence[int],
    *,
    operation: str,
    answer: int,
    value_domain: Iterable[int] = range(2, 19),
) -> int:
    """Count answer-equivalent worlds in the frozen one-edit neighborhood."""

    if isinstance(answer, bool) or not isinstance(answer, int):
        raise TypeError("answer must be an integer")
    return sum(
        apply_answer_operation(candidate, operation) == answer
        for candidate in enumerate_one_edit_worlds(observed, value_domain)
    )


def _summarize_sizes(sizes: Sequence[int]) -> dict[str, float | int]:
    if not sizes:
        raise ValueError("fiber-size collection must not be empty")
    return {
        "scene_count": len(sizes),
        "mean_size": sum(sizes) / len(sizes),
        "median_size": float(statistics.median(sizes)),
        "max_size": max(sizes),
        "singleton_count": sum(size == 1 for size in sizes),
        "singleton_rate": sum(size == 1 for size in sizes) / len(sizes),
        "empty_count": sum(size == 0 for size in sizes),
    }


def answer_fiber_statistics(
    rows: Iterable[RawRow], *, value_domain: Iterable[int] = range(2, 19)
) -> dict[str, Any]:
    """Audit v4 RL answer fibers overall and by operation and family."""

    domain = validate_domain(value_domain)
    all_sizes: list[int] = []
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    per_scene: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        scene_id = str(row["scene_id"])
        if scene_id in seen:
            raise ValueError(f"duplicate RL scene_id: {scene_id}")
        seen.add(scene_id)
        observed = validate_world(row["observed"], label="observed")
        truth = validate_world(row["truth"], label="truth")
        operation = str(row["operation"])
        answer = row["answer"]
        if isinstance(answer, bool) or not isinstance(answer, int):
            raise TypeError("answer must be an integer")
        if apply_answer_operation(truth, operation) != answer:
            raise ValueError(f"answer does not match truth for scene {scene_id}")
        size = answer_fiber_size(observed, operation=operation, answer=answer, value_domain=domain)
        family = str(row.get("family", "unknown"))
        all_sizes.append(size)
        grouped["operation"][operation].append(size)
        grouped["family"][family].append(size)
        per_scene.append({"scene_id": scene_id, "fiber_size": size})
    if not all_sizes:
        raise ValueError("RL rows must not be empty")
    return {
        **_summarize_sizes(all_sizes),
        "candidate_definition": {
            "distance": "hamming_at_most_one",
            "value_domain": list(domain),
            "includes_observed_if_admissible": True,
        },
        "by_operation": {
            key: _summarize_sizes(values) for key, values in sorted(grouped["operation"].items())
        },
        "by_family": {
            key: _summarize_sizes(values) for key, values in sorted(grouped["family"].items())
        },
        "per_scene": sorted(per_scene, key=lambda item: item["scene_id"]),
    }


__all__ = [
    "answer_fiber_size",
    "answer_fiber_statistics",
    "apply_answer_operation",
    "enumerate_one_edit_worlds",
    "validate_domain",
    "validate_world",
]
