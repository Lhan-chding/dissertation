from __future__ import annotations

import math

from compbias.theory.task_metric import task_induced_distance

QUERIES = ("a", "b", "sum")


def _solver(state: dict[str, float], query: str) -> float:
    if query == "sum":
        return state["a"] + state["b"]
    return state[query]


def _absolute(left: object, right: object) -> float:
    return abs(float(left) - float(right))


def test_task_distance_is_a_pseudometric() -> None:
    states = (
        {"a": 1.0, "b": 2.0, "unused": 10.0},
        {"a": 2.0, "b": 2.0, "unused": -1.0},
        {"a": 4.0, "b": 3.0, "unused": 7.0},
    )
    distance = lambda x, y: task_induced_distance(  # noqa: E731
        x, y, QUERIES, _solver, _absolute, p=2.0
    )

    for state in states:
        assert distance(state, state) == 0.0
    for left in states:
        for right in states:
            assert distance(left, right) >= 0.0
            assert distance(left, right) == distance(right, left)
    assert (
        distance(states[0], states[2])
        <= distance(states[0], states[1]) + distance(states[1], states[2]) + 1e-12
    )


def test_task_equivalent_states_can_have_zero_distance() -> None:
    left = {"a": 1.0, "b": 2.0, "unused": 10.0}
    right = {"a": 1.0, "b": 2.0, "unused": -999.0}
    assert math.isclose(
        task_induced_distance(left, right, QUERIES, _solver, _absolute),
        0.0,
        abs_tol=0.0,
    )
