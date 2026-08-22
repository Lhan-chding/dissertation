from __future__ import annotations

from compensability_v5.study_c2.action_protocol import (
    audit_action_censoring,
    parse_first_world_action,
    parse_first_world_tuple,
    parse_legacy_exact_world,
)
from compensability_v5.study_c2.rewards import RolloutKind, classify_completion


def test_first_line_is_the_only_action_and_legacy_parser_remains_read_only() -> None:
    completion = "4,5,6,7\nI changed the first value because the facts require it."

    assert parse_first_world_action(completion) == (4, 5, 6, 7)
    assert parse_legacy_exact_world(completion) is None
    assert parse_first_world_action("prefix 4,5,6,7") is None
    assert parse_first_world_action("4,5,6,19\n") is None
    assert parse_first_world_tuple("4,5,6,19\nignored") == (4, 5, 6, 19)
    assert parse_first_world_action("4,5,6,7 extra\n") is None


def test_reward_classification_separates_exact_shortcut_failure_and_invalid() -> None:
    truth = (4, 5, 6, 7)
    operation = {"operator": "sum", "indices": [0, 1]}

    exact = classify_completion("4,5,6,7\n", truth=truth, operation=operation)
    shortcut = classify_completion("3,6,6,7\n", truth=truth, operation=operation)
    failure = classify_completion("3,5,6,7\n", truth=truth, operation=operation)
    invalid = classify_completion("The answer is nine", truth=truth, operation=operation)

    assert (exact.kind, exact.answer_reward, exact.state_reward) == (RolloutKind.X, 1, 1)
    assert (shortcut.kind, shortcut.answer_reward, shortcut.state_reward) == (
        RolloutKind.S,
        1,
        0,
    )
    assert (failure.kind, failure.answer_reward, failure.state_reward) == (
        RolloutKind.F,
        0,
        0,
    )
    assert (invalid.kind, invalid.answer_reward, invalid.state_reward) == (
        RolloutKind.U,
        0,
        0,
    )


def test_censoring_bridge_keeps_formal_rewards_immutable() -> None:
    rows = audit_action_censoring(
        ["4,5,6,7", "4,5,6,7\nexplanation", "bad"],
        truth=(4, 5, 6, 7),
        operation={"operator": "sum", "indices": [0, 1]},
    )

    assert [row["first_line_kind"] for row in rows] == ["X", "X", "U"]
    assert [row["legacy_parse_success"] for row in rows] == [True, False, False]
    assert rows[1]["informatively_censored"] is True
