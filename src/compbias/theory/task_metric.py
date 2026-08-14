"""Task-induced behavioral geometry without latent Euclidean assumptions."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from numbers import Real
from typing import TypeVar

StateT = TypeVar("StateT")
QueryT = TypeVar("QueryT")


def task_induced_distance(
    state_a: StateT,
    state_b: StateT,
    diagnostic_queries: Sequence[QueryT],
    solver: Callable[[StateT, QueryT], object],
    answer_metric: Callable[[object, object], float],
    p: float = 2.0,
) -> float:
    """Return the equal-weighted Lp distance over diagnostic query behavior."""

    if isinstance(diagnostic_queries, (str, bytes)) or not diagnostic_queries:
        raise ValueError("diagnostic_queries must be a non-empty sequence")
    if not callable(solver) or not callable(answer_metric):
        raise TypeError("solver and answer_metric must be callable")
    if isinstance(p, bool) or not isinstance(p, Real):
        raise TypeError("p must be numeric")
    exponent = float(p)
    if not math.isfinite(exponent) or exponent < 1.0:
        raise ValueError("p must be finite and at least one")

    powered: list[float] = []
    for query in diagnostic_queries:
        distance = answer_metric(solver(state_a, query), solver(state_b, query))
        if isinstance(distance, bool) or not isinstance(distance, Real):
            raise TypeError("answer_metric must return numeric distances")
        numeric = float(distance)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("answer_metric must return finite non-negative distances")
        powered.append(numeric**exponent)
    return float((math.fsum(powered) / len(powered)) ** (1.0 / exponent))
