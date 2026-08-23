from __future__ import annotations

from compensability.study_c3.action_taxonomy import ActionClass
from compensability.study_c3.reward_channels import (
    ARM_IDS,
    rewards_for_class,
    verify_argmax_preservation,
)


def test_registered_reward_matrix_preserves_all_three_channels() -> None:
    expected = {
        "A_BIN": [1, 1, 0, 0],
        "X_BIN": [1, 0, 0, 0],
        "A_LEX": [3, 3, 1, 0],
        "X_LEX": [3, 1, 1, 0],
    }
    classes = (ActionClass.X, ActionClass.S, ActionClass.W, ActionClass.I)
    for arm in ARM_IDS:
        rows = [rewards_for_class(kind, arm=arm) for kind in classes]
        assert [row["combined_reward"] for row in rows] == expected[arm]
        assert all(
            set(row) == {"semantic_reward", "validity_reward", "combined_reward"}
            for row in rows
        )


def test_finite_argmax_preservation() -> None:
    assert verify_argmax_preservation() == {
        "A_LEX_equals_A_BIN": True,
        "X_LEX_equals_X_BIN": True,
    }
