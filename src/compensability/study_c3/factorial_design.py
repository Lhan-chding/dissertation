"""Closed 2x2 Study C3 factorial contract and group diagnostics."""

from __future__ import annotations

import copy
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence

from .action_taxonomy import ActionClass, FailureCategory

ARM_IDS = ("A_BIN", "X_BIN", "A_LEX", "X_LEX")
TRAINING_SEED = 2026082501
CHECKPOINT_STEPS = (48, 96, 144, 192)

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
    "group_size": 8,
    "optimizer": "adamw_torch",
    "epochs": 1,
    "optimizer_steps": 192,
    "checkpoint_steps": [48, 96, 144, 192],
}
_EVALUATION = {
    "sampled_rollouts": 16,
    "free_max_completion_length": 48,
    "bootstrap_resamples": 10_000,
    "bootstrap_seed": 2026082503,
    "decoders": ["free_48", "valid_world_fsa"],
}
_INTERVENTION = {
    "sampled_rollouts": 16,
    "decoders": ["free_16", "free_48", "valid_world_fsa"],
}
_SHARED_GRADIENT = {"sampled_rollouts": 8, "decoder": "free_48"}


def validate_study_c3_config(payload: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema_version",
        "model",
        "value_domain",
        "training_seeds",
        "single_seed_mechanism_pilot",
        "training",
        "evaluation",
        "existing_checkpoint_intervention",
        "shared_gradient",
        "lexicographic_semantic_coefficient",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Study C3 config schema drifted")
    if (
        payload["schema_version"] != 3
        or payload["model"] != "Qwen2.5-VL-3B-Instruct"
        or payload["value_domain"] != [2, 18]
        or payload["training_seeds"] != [TRAINING_SEED]
        or payload["single_seed_mechanism_pilot"] is not True
        or payload["lexicographic_semantic_coefficient"] != 2
        or payload["training"] != _TRAINING
        or payload["evaluation"] != _EVALUATION
        or payload["existing_checkpoint_intervention"] != _INTERVENTION
        or payload["shared_gradient"] != _SHARED_GRADIENT
    ):
        raise ValueError("Study C3 frozen design drifted")
    return copy.deepcopy(dict(payload))


def _digest(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("Study C3 initialization hash must be lowercase SHA-256")


def build_factorial_arms(
    contract: Mapping[str, object], *, initialization_hash: str
) -> tuple[dict[str, object], ...]:
    _digest(initialization_hash)
    validate_study_c3_config(contract)
    result: list[dict[str, object]] = []
    for arm in ARM_IDS:
        verifier, channel = arm.split("_", maxsplit=1)
        result.append(
            {
                "schema_version": 3,
                "seed": TRAINING_SEED,
                "single_seed_mechanism_pilot": True,
                "initialization": "Study_B_B3",
                "initialization_hash": initialization_hash,
                "action_protocol": "canonical_semantic_world_v1",
                "training": copy.deepcopy(_TRAINING),
                "arm": arm,
                "verifier": "answer" if verifier == "A" else "state",
                "validity_channel": "binary" if channel == "BIN" else "lex",
                "reward_function_id": f"study_c3_{arm.lower()}_v1",
                "output_directory": f"artifacts/v5/study_c3/training/{arm}",
            }
        )
    return tuple(result)


def validate_factorial_pairing(records: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    if tuple(records) != ARM_IDS:
        raise ValueError("Study C3 pairing requires A_BIN/X_BIN/A_LEX/X_LEX in frozen order")
    first = records[ARM_IDS[0]]
    reference_scenes = first.get("scene_ids")
    reference_prompts = first.get("prompt_sha256s")
    reference_seeds = first.get("rollout_seed_contract")
    reference_initial = first.get("initial_checkpoint_sha256")
    allowed = {
        "arm",
        "verifier",
        "validity_channel",
        "reward_function_id",
        "output_directory",
    }
    reference_config = first.get("arm_config")
    if not isinstance(reference_config, Mapping):
        raise ValueError("Study C3 pairing lacks arm configs")
    common = {key: value for key, value in reference_config.items() if key not in allowed}
    for arm in ARM_IDS:
        record = records[arm]
        if record.get("scene_ids") != reference_scenes:
            raise ValueError(f"Study C3 scene IDs/order drifted for {arm}")
        if record.get("prompt_sha256s") != reference_prompts:
            raise ValueError(f"Study C3 prompt order drifted for {arm}")
        if record.get("rollout_seed_contract") != reference_seeds:
            raise ValueError(f"Study C3 rollout seeds drifted for {arm}")
        if record.get("initial_checkpoint_sha256") != reference_initial:
            raise ValueError(f"Study C3 initial checkpoint drifted for {arm}")
        config = record.get("arm_config")
        if not isinstance(config, Mapping):
            raise ValueError(f"Study C3 pairing lacks arm config for {arm}")
        if {key: value for key, value in config.items() if key not in allowed} != common:
            raise ValueError(f"Study C3 optimizer/nonreward config drifted for {arm}")
    return {
        "reward_only_factorial_verified": True,
        "arm_count": 4,
        "scene_count": len(reference_scenes) if isinstance(reference_scenes, list) else 0,
        "initial_checkpoint_sha256": reference_initial,
    }


def summarize_training_group(
    rows: Sequence[Mapping[str, object]], *, expected_size: int
) -> dict[str, object]:
    if len(rows) != expected_size:
        raise ValueError("Study C3 reward trace group size drifted")
    classes = [str(row.get("action_class")) for row in rows]
    if any(value not in {kind.value for kind in ActionClass} for value in classes):
        raise ValueError("Study C3 group contains an invalid action class")
    channels: dict[str, list[float]] = {}
    for key in ("semantic_reward", "validity_reward", "combined_reward"):
        values = [row.get(key) for row in rows]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("Study C3 group reward channels must be finite")
        channels[key] = [float(value) for value in values]  # type: ignore[arg-type]
    lengths = [row.get("completion_token_length") for row in rows]
    if any(type(value) is not int or int(value) < 0 for value in lengths):
        raise ValueError("Study C3 completion token lengths are invalid")
    failures = [str(row.get("failure_category")) for row in rows]
    registered_failures = {kind.value for kind in FailureCategory}
    if any(value not in registered_failures for value in failures):
        raise ValueError("Study C3 group failure taxonomy drifted")
    counts = Counter(classes)
    failure_counts = Counter(failures)
    semantic = channels["semantic_reward"]
    combined = channels["combined_reward"]
    return {
        "counts": {kind.value: counts[kind.value] for kind in ActionClass},
        "semantic_reward_variance": statistics.pvariance(channels["semantic_reward"]),
        "validity_reward_variance": statistics.pvariance(channels["validity_reward"]),
        "combined_reward_variance": statistics.pvariance(combined),
        "validity_bearing_group": counts["I"] > 0 and sum(counts[kind] for kind in "XSW") > 0,
        "truth_bearing_group": counts["X"] > 0 and counts["S"] + counts["W"] > 0,
        "answer_disagreement_group": counts["X"] > 0 and counts["S"] > 0,
        "all_zero": all(value == 0.0 for value in semantic),
        "all_one": all(value == 1.0 for value in semantic),
        "combined_constant": len(set(combined)) == 1,
        "combined_constant_value": combined[0] if len(set(combined)) == 1 else None,
        "constant_reward_group": len(set(combined)) == 1,
        "mean_completion_token_length": sum(int(value) for value in lengths) / len(lengths),
        "truncated_or_labeled_rate": failure_counts[
            FailureCategory.LABELED_INCOMPLETE_OR_TRUNCATED.value
        ]
        / len(rows),
        "prose_rate": failure_counts[FailureCategory.FREE_PROSE_WITHOUT_COMPLETE_ACTION.value]
        / len(rows),
        "parse_failure_taxonomy": {
            kind.value: failure_counts[kind.value] for kind in FailureCategory
        },
    }


__all__ = [
    "ARM_IDS",
    "CHECKPOINT_STEPS",
    "TRAINING_SEED",
    "build_factorial_arms",
    "summarize_training_group",
    "validate_factorial_pairing",
    "validate_study_c3_config",
]
