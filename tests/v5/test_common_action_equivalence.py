"""Common four-integer action-space and reward-isolation contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation
from compensability_v5.training.common_space_rewards import answer_reward, exact_state_reward


def test_world_action_round_trip_is_strict_and_immutable() -> None:
    payload = {"world": [5, 2, 7, 4]}

    action = WorldAction.from_mapping(payload)

    assert action.world == (5, 2, 7, 4)
    assert action.to_mapping() == payload
    payload["world"][0] = 99
    assert action.world == (5, 2, 7, 4)
    with pytest.raises(FrozenInstanceError):
        action.world = (1, 2, 3, 4)  # type: ignore[misc]


@pytest.mark.parametrize(
    "payload",
    [
        {"world": [1, 2, 3]},
        {"world": [1, 2, 3, 4, 5]},
        {"world": [1, 2, 3, 4.0]},
        {"world": [1, 2, True, 4]},
        {"world": [1, 2, 3, 4], "answer": 10},
        {"values": [1, 2, 3, 4]},
    ],
)
def test_world_action_rejects_noncanonical_or_extra_fields(payload: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError), match=r"world|integer|unknown"):
        WorldAction.from_mapping(payload)


def test_answer_and_state_rewards_use_the_same_world_action() -> None:
    truth = WorldAction.from_mapping({"world": [5, 2, 7, 4]})
    shortcut = WorldAction.from_mapping({"world": [4, 3, 7, 4]})
    operation = {"operator": "sum", "indices": [0, 1]}

    assert apply_answer_operation(truth, operation) == 7
    assert apply_answer_operation(shortcut, operation) == 7
    assert answer_reward(truth, truth, operation) == 1.0
    assert answer_reward(shortcut, truth, operation) == 1.0
    assert exact_state_reward(truth, truth) == 1.0
    assert exact_state_reward(shortcut, truth) == 0.0


def test_query_operator_changes_reward_not_action_protocol() -> None:
    action = WorldAction.from_mapping({"world": [8, 3, 5, 2]})

    assert apply_answer_operation(action, {"operator": "sum", "indices": [0, 2]}) == 13
    assert apply_answer_operation(action, {"operator": "difference", "indices": [0, 2]}) == 3
    assert (
        apply_answer_operation(
            action,
            {"operator": "max_minus_min", "indices": [0, 1, 2, 3]},
        )
        == 6
    )
    assert action.to_mapping() == {"world": [8, 3, 5, 2]}


def test_max_minus_min_requires_the_complete_four_value_query() -> None:
    action = WorldAction.from_mapping({"world": [8, 3, 5, 2]})

    with pytest.raises(ValueError, match="all four"):
        apply_answer_operation(
            action,
            {"operator": "max_minus_min", "indices": [0, 1]},
        )
