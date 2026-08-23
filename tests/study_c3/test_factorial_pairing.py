from __future__ import annotations

import copy

import pytest

from compensability.study_c3.factorial_design import (
    ARM_IDS,
    build_factorial_arms,
    validate_factorial_pairing,
    validate_study_c3_config,
)


def _config() -> dict[str, object]:
    return {
        "schema_version": 3,
        "model": "Qwen2.5-VL-3B-Instruct",
        "value_domain": [2, 18],
        "training_seeds": [2026082501],
        "single_seed_mechanism_pilot": True,
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
            "group_size": 8,
            "optimizer": "adamw_torch",
            "epochs": 1,
            "optimizer_steps": 192,
            "checkpoint_steps": [48, 96, 144, 192],
        },
        "evaluation": {
            "sampled_rollouts": 16,
            "free_max_completion_length": 48,
            "bootstrap_resamples": 10_000,
            "bootstrap_seed": 2026082503,
            "decoders": ["free_48", "valid_world_fsa"],
        },
        "existing_checkpoint_intervention": {
            "sampled_rollouts": 16,
            "decoders": ["free_16", "free_48", "valid_world_fsa"],
        },
        "shared_gradient": {"sampled_rollouts": 8, "decoder": "free_48"},
        "lexicographic_semantic_coefficient": 2,
    }


def test_closed_config_and_four_reward_only_arms() -> None:
    config = validate_study_c3_config(_config())
    arms = build_factorial_arms(config, initialization_hash="a" * 64)

    assert tuple(arm["arm"] for arm in arms) == ARM_IDS
    assert all(arm["single_seed_mechanism_pilot"] is True for arm in arms)
    common_keys = set(arms[0]) - {
        "arm",
        "verifier",
        "validity_channel",
        "reward_function_id",
        "output_directory",
    }
    assert all(
        {key: arm[key] for key in common_keys} == {key: arms[0][key] for key in common_keys}
        for arm in arms
    )


def test_pairing_fails_closed_on_any_nonreward_drift() -> None:
    arms = build_factorial_arms(validate_study_c3_config(_config()), initialization_hash="a" * 64)
    records = {
        str(arm["arm"]): {
            "arm_config": arm,
            "scene_ids": ["scene-a", "scene-b"],
            "prompt_sha256s": ["1" * 64, "2" * 64],
            "rollout_seed_contract": {
                "algorithm": "trl_global_seed_common_stream_v1",
                "seed": 2026082501,
            },
            "initial_checkpoint_sha256": "a" * 64,
        }
        for arm in arms
    }
    result = validate_factorial_pairing(records)
    assert result["reward_only_factorial_verified"] is True

    drifted = copy.deepcopy(records)
    drifted["X_LEX"]["scene_ids"] = ["scene-b", "scene-a"]
    with pytest.raises(ValueError, match="scene IDs"):
        validate_factorial_pairing(drifted)

    drifted = copy.deepcopy(records)
    drifted["A_LEX"]["initial_checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="initial checkpoint"):
        validate_factorial_pairing(drifted)
