from __future__ import annotations

import pytest

from compensability.study_c3.factorial_design import summarize_training_group


def _row(action_class: str, reward: int, failure: str, length: int) -> dict[str, object]:
    semantic = int(action_class == "X")
    validity = int(action_class != "I")
    return {
        "action_class": action_class,
        "semantic_reward": semantic,
        "validity_reward": validity,
        "combined_reward": reward,
        "completion_token_length": length,
        "failure_category": failure,
    }


def test_group_diagnostics_records_registered_mechanisms() -> None:
    rows = (
        _row("X", 3, "canonical_valid_in_domain", 8),
        _row("S", 1, "canonical_valid_in_domain", 9),
        _row("W", 1, "semantic_valid_noncanonical", 10),
        _row("W", 1, "canonical_valid_in_domain", 11),
        _row("I", 0, "labeled_incomplete_or_truncated", 16),
        _row("I", 0, "free_prose_without_complete_action", 16),
        _row("I", 0, "other_invalid", 4),
        _row("I", 0, "complete_out_of_domain", 12),
    )

    result = summarize_training_group(rows, expected_size=8)

    assert result["counts"] == {"X": 1, "S": 1, "W": 2, "I": 4}
    assert result["validity_bearing_group"] is True
    assert result["truth_bearing_group"] is True
    assert result["answer_disagreement_group"] is True
    assert result["all_zero"] is False and result["all_one"] is False
    assert result["semantic_reward_variance"] > 0
    assert result["validity_reward_variance"] > 0
    assert result["combined_reward_variance"] > 0
    assert result["mean_completion_token_length"] == 10.75
    assert result["parse_failure_taxonomy"]["labeled_incomplete_or_truncated"] == 1


def test_group_diagnostics_rejects_wrong_size_and_nonfinite_values() -> None:
    rows = [_row("I", 0, "other_invalid", 1)] * 8
    with pytest.raises(ValueError, match="group size"):
        summarize_training_group(rows[:-1], expected_size=8)
    rows[0] = dict(rows[0], combined_reward=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        summarize_training_group(rows, expected_size=8)


def test_all_one_is_measured_on_the_binary_semantic_channel_for_lex_rewards() -> None:
    rows = [_row("X", 3, "canonical_valid_in_domain", 8)] * 8

    result = summarize_training_group(rows, expected_size=8)

    assert result["all_one"] is True
    assert result["combined_constant"] is True
    assert result["combined_constant_value"] == 3.0
