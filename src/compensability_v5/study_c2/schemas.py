"""Closed Study C2 configuration and reward-arm identity contract."""

from __future__ import annotations

from collections.abc import Mapping

SEED = 2026082401
_TRAINING = {
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
}
_EVALUATION = {
    "sampled_rollouts": 16,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 2026082403,
}


class C2ContractError(ValueError):
    """The frozen Study C2 design contract drifted."""


def validate_study_c2_config(payload: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema_version",
        "seed",
        "value_domain",
        "group_candidates",
        "support_rollouts_per_prompt",
        "training",
        "evaluation",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise C2ContractError("Study C2 config schema drifted")
    if (
        payload["schema_version"] != 2
        or payload["seed"] != SEED
        or payload["value_domain"] != [2, 18]
        or payload["group_candidates"] != [8, 16, 32]
        or payload["support_rollouts_per_prompt"] != 64
    ):
        raise C2ContractError("Study C2 top-level contract drifted")
    if payload["training"] != _TRAINING:
        raise C2ContractError("Study C2 training contract drifted")
    if payload["evaluation"] != _EVALUATION:
        raise C2ContractError("Study C2 evaluation contract drifted")
    return {
        "schema_version": 2,
        "seed": SEED,
        "value_domain": [2, 18],
        "group_candidates": [8, 16, 32],
        "support_rollouts_per_prompt": 64,
        "training": dict(_TRAINING),
        "evaluation": dict(_EVALUATION),
    }


def build_reward_arm_configs(
    contract: Mapping[str, object], *, initialization_hash: str
) -> tuple[dict[str, object], dict[str, object]]:
    if len(initialization_hash) != 64 or any(
        char not in "0123456789abcdef" for char in initialization_hash
    ):
        raise C2ContractError("B3 initialization hash must be lowercase SHA-256")
    common = {
        "schema_version": contract["schema_version"],
        "seed": contract["seed"],
        "initialization": "B3",
        "initialization_hash": initialization_hash,
        "action_protocol": "anchored_first_line_world_v1",
        "training": dict(contract["training"]),  # type: ignore[arg-type]
    }
    return (
        {
            **common,
            "name": "C2_answer_reward",
            "reward_function_id": "answer_reward_v1",
            "output_directory": "artifacts/v5/study_c2/training/C2_answer_reward",
        },
        {
            **common,
            "name": "C2_exact_state_reward",
            "reward_function_id": "exact_state_reward_v1",
            "output_directory": "artifacts/v5/study_c2/training/C2_exact_state_reward",
        },
    )


__all__ = ["SEED", "C2ContractError", "build_reward_arm_configs", "validate_study_c2_config"]
