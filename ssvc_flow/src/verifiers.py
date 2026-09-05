"""Strict action validity, deterministic executors, and full-domain answer fibers."""

import json
from functools import lru_cache

from .constraint_solver import satisfies


def strict_parse(raw, domain_size=100):
    """Parse the complete JSON string, with no extraction or output repair."""
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError, RecursionError):
        return None
    if (
        not isinstance(parsed, list)
        or len(parsed) != 4
        or any(type(value) is not int or not 0 <= value < domain_size for value in parsed)
    ):
        return None
    return parsed


def executor(world, operation):
    if (
        not isinstance(world, (list, tuple))
        or len(world) != 4
        or any(type(value) is not int or not 0 <= value < 100 for value in world)
    ):
        raise ValueError("world must be four integers in 0..99")
    a, b, c, d = world
    if operation == "sum4":
        return a + b + c + d
    if operation == "difference_pairs":
        return a + b - c - d
    if operation == "range4":
        return max(world) - min(world)
    raise ValueError(f"unknown operation: {operation}")


def classify(raw, truth_world, operation):
    parsed = strict_parse(raw)
    if parsed is None:
        return "I"
    if parsed == list(truth_world):
        return "X"
    return "S" if executor(parsed, operation) == executor(truth_world, operation) else "W"


def annotate(raw, truth_world, operation, cue=None):
    parsed = strict_parse(raw)
    return {
        "parsed_world": parsed,
        "category": classify(raw, truth_world, operation),
        "syntax_valid": parsed is not None,
        "constraint_satisfaction": satisfies(parsed, cue)
        if parsed is not None and cue is not None
        else None,
        "executor_answer": executor(parsed, operation) if parsed is not None else None,
    }


@lru_cache(maxsize=16)
def _sum_counts(domain_size, count):
    result = (1,)
    for _ in range(count):
        # Integer convolution; tuples prevent accidental modification of cached counts.
        result = tuple(
            sum(
                result[total - value]
                for value in range(domain_size)
                if 0 <= total - value < len(result)
            )
            for total in range(len(result) + domain_size - 1)
        )
    return result


def fiber_size(answer, operation, domain_size=100):
    """Count h^{-1}(answer) in D^4, never the constrained repair candidate set."""
    if type(domain_size) is not int or domain_size < 2 or type(answer) is not int:
        raise ValueError("fiber inputs require integer answer and domain_size >= 2")
    if operation == "sum4":
        counts = _sum_counts(domain_size, 4)
        return counts[answer] if 0 <= answer < len(counts) else 0
    if operation == "difference_pairs":
        counts = _sum_counts(domain_size, 2)
        return sum(
            value * counts[index - answer]
            for index, value in enumerate(counts)
            if 0 <= index - answer < len(counts)
        )
    if operation == "range4":
        if not 0 <= answer < domain_size:
            return 0
        if answer == 0:
            return domain_size
        r = answer
        return (domain_size - r) * ((r + 1) ** 4 - 2 * r**4 + (r - 1) ** 4)
    raise ValueError(f"unknown operation: {operation}")
