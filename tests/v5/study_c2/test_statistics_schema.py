from __future__ import annotations

import pytest

from compensability_v5.study_c2.schemas import (
    C2ContractError,
    build_reward_arm_configs,
    validate_study_c2_config,
)
from compensability_v5.study_c2.statistics import paired_collision_difference_in_differences


def _config() -> dict[str, object]:
    return {
        "schema_version": 2,
        "seed": 2026082401,
        "value_domain": [2, 18],
        "group_candidates": [8, 16, 32],
        "support_rollouts_per_prompt": 64,
        "training": {
            "precision": "bf16",
            "learning_rate": 1.0e-6,
            "kl_beta": 0.04,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "temperature": 0.7,
            "top_p": 1.0,
            "max_prompt_length": 512,
            "max_completion_length": 16,
            "optimizer": "adamw_torch",
            "epochs": 1,
        },
        "evaluation": {
            "sampled_rollouts": 16,
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 2026082403,
        },
    }


def test_two_arm_manifests_differ_only_in_registered_fields() -> None:
    contract = validate_study_c2_config(_config())
    arms = build_reward_arm_configs(contract, initialization_hash="a" * 64)
    left, right = arms
    differing = {key for key in left if left[key] != right[key]}
    assert differing == {"name", "reward_function_id", "output_directory"}

    drifted = _config()
    drifted["training"] = {**drifted["training"], "max_completion_length": 32}  # type: ignore[arg-type]
    with pytest.raises(C2ContractError, match="training"):
        validate_study_c2_config(drifted)


def test_scene_paired_collision_difference_in_differences() -> None:
    rows = [
        {"pair_id": "p1", "condition": "collision", "arm": "state", "exact": 1.0},
        {"pair_id": "p1", "condition": "collision", "arm": "answer", "exact": 0.0},
        {"pair_id": "p1", "condition": "separating", "arm": "state", "exact": 0.5},
        {"pair_id": "p1", "condition": "separating", "arm": "answer", "exact": 0.5},
        {"pair_id": "p2", "condition": "collision", "arm": "state", "exact": 0.5},
        {"pair_id": "p2", "condition": "collision", "arm": "answer", "exact": 0.0},
        {"pair_id": "p2", "condition": "separating", "arm": "state", "exact": 0.0},
        {"pair_id": "p2", "condition": "separating", "arm": "answer", "exact": 0.0},
    ]
    result = paired_collision_difference_in_differences(rows, resamples=1000, seed=7)
    assert result["estimate"] == 0.75
    assert result["pair_count"] == 2
    assert result["bootstrap_95_ci"][0] <= result["estimate"] <= result["bootstrap_95_ci"][1]
