"""Compensability is estimated per prompt from a rollout-level long table."""

import pandas as pd
import pytest

from compbias.eval.compensability import (
    build_compensability_long_table,
    per_prompt_covariances,
)

EXPECTED_COLUMNS = [
    "sample_id",
    "error_id",
    "severity",
    "base_probability",
    "checkpoint",
    "rollout_seed",
    "reward",
    "c_hat",
    "c_ci_low",
    "c_ci_high",
]


def _records() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    profiles = {
        "prompt_a": (("truth", 0.0, 1.0, 0.8), ("offset", 1.0, 0.0, 0.2)),
        "prompt_b": (("truth", 100.0, 0.0, 0.25), ("offset", 101.0, 1.0, 0.75)),
    }
    for sample_id, errors in profiles.items():
        for error_id, severity, reward, base_probability in errors:
            for seed in range(4):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "error_id": error_id,
                        "severity": severity,
                        "base_probability": base_probability,
                        "checkpoint": "step_100",
                        "rollout_seed": seed,
                        "reward": reward,
                        "view": "interventional",
                    }
                )
    return tuple(rows)


def test_long_table_keeps_one_row_per_rollout_and_group_estimates() -> None:
    records = _records()

    table = build_compensability_long_table(records, confidence=0.95)

    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == EXPECTED_COLUMNS
    assert len(table) == len(records)
    assert not table.duplicated(["sample_id", "error_id", "checkpoint", "rollout_seed"]).any()
    grouped = table.groupby(["sample_id", "error_id"], sort=True)["c_hat"].first()
    assert grouped.loc[("prompt_a", "truth")] == 1.0
    assert grouped.loc[("prompt_a", "offset")] == 0.0
    assert grouped.loc[("prompt_b", "truth")] == 0.0
    assert grouped.loc[("prompt_b", "offset")] == 1.0
    assert ((table.c_ci_low >= 0) & (table.c_ci_high <= 1)).all()


def test_covariance_is_computed_per_prompt_not_only_after_pooling() -> None:
    table = build_compensability_long_table(_records())

    covariances = per_prompt_covariances(table)

    assert list(covariances.columns) == [
        "sample_id",
        "checkpoint",
        "severity_compensability_covariance",
        "n_errors",
        "base_probability_sum",
    ]
    by_prompt = covariances.set_index("sample_id")
    assert by_prompt.loc["prompt_a", "severity_compensability_covariance"] == pytest.approx(-0.16)
    assert by_prompt.loc["prompt_b", "severity_compensability_covariance"] == pytest.approx(0.1875)
    assert (by_prompt.n_errors == 2).all()


def test_natural_and_interventional_views_cannot_be_mixed() -> None:
    records = list(_records())
    records[0] = {**records[0], "view": "natural"}

    with pytest.raises(ValueError, match=r"natural.*interventional|view"):
        build_compensability_long_table(records)


def test_duplicate_seed_or_inconsistent_severity_is_rejected() -> None:
    records = list(_records())
    with pytest.raises(ValueError, match=r"duplicate|rollout_seed"):
        build_compensability_long_table((*records, records[0]))

    records[1] = {**records[1], "severity": 99.0}
    with pytest.raises(ValueError, match="severity"):
        build_compensability_long_table(records)


def test_compensability_requires_binary_rewards_and_normalized_base_policy() -> None:
    records = list(_records())
    records[0] = {**records[0], "reward": 0.5}
    with pytest.raises(ValueError, match=r"binary|0 or 1"):
        build_compensability_long_table(records)

    table = build_compensability_long_table(_records())
    table.loc[table.sample_id == "prompt_a", "base_probability"] = (0.6, 0.6) * 4
    with pytest.raises(ValueError, match=r"base_probability|sum"):
        per_prompt_covariances(table)
