"""Frozen provenance and common-random-number contract for C3-C."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from compensability_v4.qwen.phase5_runtime import phase5_rollout_seed

from .factorial_design import (
    ARM_IDS,
    CHECKPOINT_STEPS,
    TRAINING_SEED,
    build_factorial_arms,
    validate_factorial_pairing,
    validate_study_c3_config,
)
from .io import canonical_sha256
from .reward_channels import verify_argmax_preservation

_SOURCE_KEYS = {
    "package_lock_sha256",
    "fiber_rows_sha256",
    "c2_training_pair_manifest_sha256",
    "action_audit_manifest_sha256",
    "decoder_intervention_manifest_sha256",
}


def _valid_digest(value: object, *, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_training_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[str], list[str]]:
    if len(rows) != 192:
        raise ValueError("Study C3 requires exactly 192 training prompts")
    scenes: list[str] = []
    prompts: list[str] = []
    for row in rows:
        scene_id = row.get("scene_id")
        prompt_sha = row.get("prompt_sha256")
        if (
            row.get("split") != "train"
            or not isinstance(scene_id, str)
            or not _valid_digest(prompt_sha)
        ):
            raise ValueError("Study C3 training prompt rows are malformed")
        scenes.append(scene_id)
        prompts.append(str(prompt_sha))
    if len(set(scenes)) != len(scenes):
        raise ValueError("Study C3 training scene IDs are duplicated")
    return scenes, prompts


def build_execution_contract(
    *,
    config: Mapping[str, object],
    training_rows: Sequence[Mapping[str, object]],
    b3_adapter_sha256: str,
    model_snapshot_sha256: str,
    source_sha256s: Mapping[str, str],
    git_commit_sha: str,
) -> dict[str, object]:
    frozen = validate_study_c3_config(config)
    if (
        not _valid_digest(b3_adapter_sha256)
        or not _valid_digest(model_snapshot_sha256)
        or set(source_sha256s) != _SOURCE_KEYS
        or any(not _valid_digest(value) for value in source_sha256s.values())
        or not _valid_digest(git_commit_sha, length=40)
    ):
        raise ValueError("Study C3 execution-contract provenance is malformed")
    scenes, prompts = _validate_training_rows(training_rows)
    arms = build_factorial_arms(frozen, initialization_hash=b3_adapter_sha256)
    rollout_seeds = [
        [phase5_rollout_seed(TRAINING_SEED, scene_id, index) for index in range(8)]
        for scene_id in scenes
    ]
    records = {
        str(arm["arm"]): {
            "arm_config": arm,
            "scene_ids": scenes,
            "prompt_sha256s": prompts,
            "rollout_seeds": rollout_seeds,
            "initial_checkpoint_sha256": b3_adapter_sha256,
        }
        for arm in arms
    }
    pairing = validate_factorial_pairing(records)
    arm_payload = {
        arm: {
            "arm_config": records[arm]["arm_config"],
            "scene_ids": scenes,
            "prompt_sha256s": prompts,
            "rollout_seeds": rollout_seeds,
            "initial_checkpoint_sha256": b3_adapter_sha256,
        }
        for arm in ARM_IDS
    }
    return {
        "schema_version": 3,
        "status": "STUDY_C3_FACTORIAL_EXECUTION_CONTRACT_FROZEN",
        "model": frozen["model"],
        "model_snapshot_sha256": model_snapshot_sha256,
        "b3_adapter_sha256": b3_adapter_sha256,
        "git_commit_sha": git_commit_sha,
        "config_sha256": canonical_sha256(frozen),
        **dict(source_sha256s),
        "training_seed_list": [TRAINING_SEED],
        "single_seed_mechanism_pilot": True,
        "training_prompt_count": len(scenes),
        "group_size": 8,
        "optimizer_steps": 192,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "rollout_seed_algorithm": "phase5_rollout_seed_v1",
        "arms": arm_payload,
        "reward_only_factorial_verified": pairing["reward_only_factorial_verified"],
        "argmax_preservation": verify_argmax_preservation(),
        "training_invoked": False,
        "optimizer_step_invoked": False,
        "rl_invoked": False,
        "gpu_invoked": False,
    }


__all__ = ["build_execution_contract"]
