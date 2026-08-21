"""Fail-closed budget matching contracts for B0--B3 support LoRA."""

from __future__ import annotations

import copy

import pytest
from compensability_v5.audit.budget_audit import BudgetMismatchError, assert_budget_matched


def _arm(**updates: object) -> dict[str, object]:
    arm: dict[str, object] = {
        "unique_source_scenes": 128,
        "rows": 768,
        "target_tokens": 10_000,
        "steps": 96,
        "optimizer": {"name": "adamw", "learning_rate": 2e-4, "weight_decay": 0.01},
        "lora_rank": 16,
        "lora_targets": ["q_proj", "v_proj"],
        "gradient_accumulation": 8,
        "approximate_flops": 1.25e15,
    }
    arm.update(updates)
    return arm


def test_budget_match_accepts_only_preregistered_target_token_tolerance() -> None:
    budgets = {
        name: _arm(target_tokens=tokens)
        for name, tokens in zip(
            ("B0", "B1", "B2", "B3"), (10_000, 10_080, 10_020, 10_060), strict=True
        )
    }

    assert assert_budget_matched(budgets, target_token_relative_tolerance=0.01) is None


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("unique_source_scenes", 127),
        ("rows", 767),
        ("steps", 95),
        ("optimizer", {"name": "sgd", "learning_rate": 2e-4, "weight_decay": 0.01}),
        ("lora_rank", 8),
        ("lora_targets", ["q_proj"]),
        ("gradient_accumulation", 4),
        ("approximate_flops", 1.20e15),
    ],
)
def test_any_nontoken_budget_mismatch_fails_closed(field: str, different_value: object) -> None:
    budgets = {name: _arm() for name in ("B0", "B1", "B2", "B3")}
    budgets["B3"] = copy.deepcopy(budgets["B3"])
    budgets["B3"][field] = different_value

    with pytest.raises(BudgetMismatchError, match=field):
        assert_budget_matched(budgets, target_token_relative_tolerance=0.01)


def test_target_tokens_outside_tolerance_fail_closed() -> None:
    budgets = {name: _arm() for name in ("B0", "B1", "B2", "B3")}
    budgets["B3"] = _arm(target_tokens=10_101)

    with pytest.raises(BudgetMismatchError, match="target_tokens"):
        assert_budget_matched(budgets, target_token_relative_tolerance=0.01)


def test_budget_audit_requires_exactly_the_four_registered_arms() -> None:
    with pytest.raises(BudgetMismatchError, match=r"B0.*B1.*B2.*B3|arms"):
        assert_budget_matched({"B0": _arm(), "B1": _arm(), "B2": _arm()})
