"""Small arithmetic checks; no checkpoints or actual observations are loaded."""

import math

import pytest

from src.modeling_v4.bridge_support_bounds import support_bounds


def row(tokens, event, left, right):
    return {
        "token_ids": tokens,
        "category": event,
        "max_new_tokens": 64,
        "eos_seen": True,
        "left_logp": math.log(left),
        "right_logp": math.log(right),
    }


def test_deduplicated_mass_retains_unseen_events_and_contains_truth():
    a = row([10, 99], "X", 0.7, 0.8)
    b = row([11, 99], "W", 0.1, 0.05)
    result = support_bounds([a, b, a])
    assert result["unique_complete_actions"] == 2
    assert result["duplicate_rows_removed"] == 1
    assert result["unseen_mass_left"] == pytest.approx(0.2)
    assert result["unseen_mass_right"] == pytest.approx(0.15)
    x, s = result["events"][:2]
    assert [x["lower"], x["upper"]] == pytest.approx([-0.25, 0.1])
    assert [s["lower"], s["upper"]] == pytest.approx([-0.15, 0.2])
    # Allocate remaining L mass to S and remaining R mass to X.
    truth = [-0.25, 0.2, 0.05, 0]
    for event, value in zip(result["events"], truth, strict=True):
        assert event["lower"] - 1e-14 <= value <= event["upper"] + 1e-14


def test_repeated_scores_and_excess_mass_invalidate_instead_of_clipping():
    a = row([10, 99], "X", 0.7, 0.8)
    changed = {**a, "left_logp": a["left_logp"] + 1e-8}
    result = support_bounds([a, changed])
    assert not result["valid_conditional_bound"]
    assert "DUPLICATE_ACTION_SCORE_DISAGREEMENT" in result["invalid_reasons"]
    assert result["events"][0]["lower"] is None
    result = support_bounds([a, row([11, 99], "W", 0.5, 0.1)])
    assert "OBSERVED_MASS_EXCEEDS_ONE" in result["invalid_reasons"]
    assert result["unseen_mass_left"] < 0


def test_complete_token_identity_and_event_partition_are_required():
    a = row([10, 99], "X", 0.7, 0.8)
    with pytest.raises(ValueError, match="conflicting event"):
        support_bounds([a, {**a, "category": "I"}])
    with pytest.raises(ValueError, match="before EOS"):
        support_bounds([{**a, "eos_seen": False}])
    result = support_bounds([{**a, "token_ids": [10] * 64, "eos_seen": False}])
    assert result["valid_conditional_bound"]
