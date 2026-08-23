from __future__ import annotations

import pytest

from compensability.study_c3.gradient_audit import (
    shared_reward_gradient_diagnostics,
    validate_shared_action_batch,
)


def test_all_reward_gradients_use_one_action_batch() -> None:
    rows = [
        {"action_id": f"a{index}", "completion": str(index)} for index in range(4)
    ]
    assert validate_shared_action_batch(rows, expected_action_ids=["a0", "a1", "a2", "a3"])

    diagnostics = shared_reward_gradient_diagnostics(
        reward_vectors={
            "A": [1, 1, 0, 0],
            "X": [1, 0, 0, 0],
            "V": [1, 1, 1, 0],
            "A_LEX": [3, 3, 1, 0],
            "X_LEX": [3, 1, 1, 0],
        },
        score_vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
    )

    assert set(diagnostics["gradient_norms"]) == {"A", "X", "V", "A_LEX", "X_LEX"}
    assert -1.0 <= diagnostics["cosine_X_V"] <= 1.0
    assert -1.0 <= diagnostics["cosine_A_V"] <= 1.0
    assert diagnostics["difference_X_A"] >= 0
    assert diagnostics["difference_X_LEX_X"] >= 0
    assert diagnostics["difference_A_LEX_A"] >= 0


def test_shared_buffer_rejects_different_action_identity() -> None:
    with pytest.raises(ValueError, match="same action batch"):
        validate_shared_action_batch(
            [{"action_id": "a", "completion": "2,3,4,5"}],
            expected_action_ids=["b"],
        )

