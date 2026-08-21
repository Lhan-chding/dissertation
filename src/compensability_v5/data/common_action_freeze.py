"""Freeze the common-action-space RL package before any server execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence

from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation

ACTION_PARSER_ID = "compensability_v4.qwen.phase5_support.parse_world:v1"
PILOT_SEED = 2026082301

_REGISTERED_ARMS = (
    "B3_answer",
    "B3_exact_state",
    "B2_answer",
    "B2_exact_state",
    "Base_exact_state",
)
_PAIR_FIELDS = (
    "prompt_hash",
    "initialization_hash",
    "action_parser_hash",
    "rollout_seed_hash",
    "scene_metadata_hash",
)
_FORBIDDEN_PROMPT_FIELDS = frozenset(
    {
        "prompt_path",
        "prompt_file",
        "prompt_files",
        "prompts",
        "answer_prompt",
        "exact_state_prompt",
    }
)
_INITIALIZATIONS = ("B3", "B2", "Base")


class CommonActionFreezeError(ValueError):
    """The common-action RL package drifted from the preregistered local freeze."""


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CommonActionFreezeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _freeze_scene(scene: object, *, index: int) -> dict[str, object]:
    if not isinstance(scene, Mapping):
        raise CommonActionFreezeError(f"scene {index} must be a mapping")
    forbidden = sorted(_FORBIDDEN_PROMPT_FIELDS & set(scene))
    if forbidden:
        raise CommonActionFreezeError(
            "common-space RL must use a single prompt surface; prompt file drift detected: "
            + ", ".join(forbidden)
        )
    required = {"scene_id", "prompt", "truth", "answer_operation"}
    missing = sorted(required - set(scene))
    if missing:
        raise CommonActionFreezeError(f"scene {index} is missing required fields: {missing}")
    scene_id = scene["scene_id"]
    prompt = scene["prompt"]
    if not isinstance(scene_id, str) or not scene_id:
        raise CommonActionFreezeError("scene_id must be a non-empty string")
    if not isinstance(prompt, str) or not prompt:
        raise CommonActionFreezeError("prompt must be a non-empty string")
    family = scene.get("family")
    if family is not None and (not isinstance(family, str) or not family):
        raise CommonActionFreezeError("family must be a non-empty string when provided")
    fiber_size = scene.get("fiber_size")
    if fiber_size is not None and (not _is_integer(fiber_size) or fiber_size <= 0):
        raise CommonActionFreezeError("fiber_size must be a positive integer when provided")
    policy_support = scene.get("policy_support")
    if policy_support is not None and (
        isinstance(policy_support, bool)
        or not isinstance(policy_support, (int, float))
        or not 0.0 <= float(policy_support) <= 1.0
    ):
        raise CommonActionFreezeError("policy_support must be a probability when provided")
    candidate_worlds = scene.get("candidate_worlds")
    if candidate_worlds is not None:
        if (
            not isinstance(candidate_worlds, Sequence)
            or isinstance(candidate_worlds, (str, bytes))
            or not candidate_worlds
        ):
            raise CommonActionFreezeError("candidate_worlds must be a non-empty sequence")
        validated_candidates: list[list[int]] = []
        for candidate in candidate_worlds:
            try:
                action = WorldAction.from_mapping({"world": candidate})
            except (TypeError, ValueError) as error:
                raise CommonActionFreezeError(
                    "candidate_worlds must contain four-integer worlds"
                ) from error
            validated_candidates.append(list(action.world))
    else:
        validated_candidates = []
    truth_payload = {"world": scene["truth"]}
    truth = WorldAction.from_mapping(truth_payload)
    answer_operation = scene["answer_operation"]
    answer_label = apply_answer_operation(truth, answer_operation)
    result = {
        "schema_version": 1,
        "scene_id": scene_id,
        "prompt": prompt,
        "prompt_hash": _sha256_text(prompt),
        "prompt_sha256": _sha256_text(prompt),
        "truth": list(truth.world),
        "answer_operation": dict(answer_operation),
        "reward_labels": {
            "answer": answer_label,
            "exact_state": list(truth.world),
        },
    }
    for field in ("family", "fiber_size", "fiber_bin", "support_bin", "policy_support", "role"):
        if field in scene:
            value = scene[field]
            if field in {"fiber_bin", "support_bin"} and (not isinstance(value, str) or not value):
                raise CommonActionFreezeError(f"{field} must be a non-empty string when provided")
            result[field] = value
    if candidate_worlds is not None:
        result["candidate_worlds"] = validated_candidates
    return result


def _derive_bins_and_roles(scenes: list[dict[str, object]]) -> None:
    if any(
        "family" not in scene or "fiber_size" not in scene or "policy_support" not in scene
        for scene in scenes
    ):
        raise CommonActionFreezeError("family, fiber_size, and Study-A policy_support are required")
    ranked = sorted(
        scenes, key=lambda scene: (float(scene["policy_support"]), str(scene["scene_id"]))
    )
    for rank, scene in enumerate(ranked):
        size = int(scene["fiber_size"])
        scene["fiber_bin"] = (
            "singleton" if size == 1 else ("multi_2_4" if size <= 4 else "multi_5_plus")
        )
        scene["support_bin"] = (
            "medium"
            if len(ranked) == 1
            else (
                "low"
                if rank * 3 < len(ranked)
                else "medium"
                if rank * 3 < 2 * len(ranked)
                else "high"
            )
        )
    evaluation_count = 24 if len(scenes) == 96 else len(scenes) // 4
    strata: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for scene in scenes:
        key = str(scene["family"]), str(scene["fiber_bin"]), str(scene["support_bin"])
        strata.setdefault(key, []).append(scene)
    quotas = {key: len(group) * evaluation_count // len(scenes) for key, group in strata.items()}
    remainder = evaluation_count - sum(quotas.values())
    order = sorted(
        strata,
        key=lambda key: (-(len(strata[key]) * evaluation_count % len(scenes)), key),
    )
    for key in order[:remainder]:
        quotas[key] += 1
    eval_ids: set[str] = set()
    for key, group in strata.items():
        ranked_group = sorted(
            group,
            key=lambda scene: _sha256_text(f"{PILOT_SEED}:{scene['scene_id']}"),
        )
        eval_ids.update(str(scene["scene_id"]) for scene in ranked_group[: quotas[key]])
    for scene in scenes:
        scene["role"] = "rl_eval" if str(scene["scene_id"]) in eval_ids else "rl_train"


def _scene_metadata_hash(scenes: Sequence[Mapping[str, object]]) -> str:
    return _sha256_json(
        [
            {
                key: scene[key]
                for key in (
                    "scene_id",
                    "truth",
                    "answer_operation",
                    "family",
                    "fiber_size",
                    "fiber_bin",
                    "support_bin",
                    "candidate_worlds",
                    "policy_support",
                    "role",
                )
                if key in scene
            }
            for scene in scenes
        ]
    )


def freeze_common_action_space(
    scenes: Iterable[Mapping[str, object]],
    *,
    initialization_hashes: Mapping[str, str],
    action_parser_id: str,
    rollout_seeds: Sequence[int],
) -> dict[str, object]:
    """Freeze one prompt per scene with both reward labels and arm hashes."""

    if not isinstance(initialization_hashes, Mapping):
        raise CommonActionFreezeError("initialization_hashes must be a mapping")
    if set(initialization_hashes) != set(_INITIALIZATIONS):
        raise CommonActionFreezeError("initialization_hashes must contain exactly B3, B2, Base")
    if action_parser_id != ACTION_PARSER_ID:
        raise CommonActionFreezeError(
            f"action_parser_id must remain the registered runtime parser: {ACTION_PARSER_ID}"
        )
    if (
        not isinstance(rollout_seeds, Sequence)
        or isinstance(rollout_seeds, (str, bytes))
        or tuple(rollout_seeds) != (PILOT_SEED,)
    ):
        raise CommonActionFreezeError(
            f"rollout_seeds must contain exactly one fixed pilot seed: {PILOT_SEED}"
        )

    source_scenes = tuple(scenes)
    if any(
        isinstance(scene, Mapping)
        and ("fiber_bin" in scene or "support_bin" in scene or "role" in scene)
        for scene in source_scenes
    ):
        raise CommonActionFreezeError("fiber/support bins and role are derived, not user inputs")
    frozen_list = [_freeze_scene(scene, index=index) for index, scene in enumerate(source_scenes)]
    _derive_bins_and_roles(frozen_list)
    frozen_scenes = tuple(frozen_list)
    if not frozen_scenes:
        raise CommonActionFreezeError("at least one scene is required for the RL freeze")
    scene_ids = tuple(scene["scene_id"] for scene in frozen_scenes)
    if len(set(scene_ids)) != len(scene_ids):
        raise CommonActionFreezeError("scene_id values must be unique across frozen RL scenes")

    prompt_hash = _sha256_json(
        [
            {"scene_id": scene["scene_id"], "prompt_hash": scene["prompt_hash"]}
            for scene in frozen_scenes
        ]
    )
    action_parser_hash = _sha256_text(action_parser_id)
    rollout_seed_hash = _sha256_json(list(rollout_seeds))
    scene_metadata_hash = _scene_metadata_hash(frozen_scenes)
    shared_arm_fields = {
        "prompt_hash": prompt_hash,
        "action_parser_hash": action_parser_hash,
        "rollout_seeds": list(rollout_seeds),
        "rollout_seed_hash": rollout_seed_hash,
        "scene_metadata_hash": scene_metadata_hash,
        "rl_train_hash": _sha256_json(
            [scene for scene in frozen_scenes if scene["role"] == "rl_train"]
        ),
        "rl_eval_hash": _sha256_json(
            [scene for scene in frozen_scenes if scene["role"] == "rl_eval"]
        ),
        "role_counts": {
            "rl_train": sum(scene["role"] == "rl_train" for scene in frozen_scenes),
            "rl_eval": sum(scene["role"] == "rl_eval" for scene in frozen_scenes),
        },
        "decoding": {"mode": "shared_from_training_config"},
        "optimizer": {"mode": "shared_from_training_config"},
        "steps": "shared_from_training_config",
        "group_size": "shared_from_training_config",
        "action_space": "four_integer_world",
    }
    arms: dict[str, dict[str, object]] = {}
    for initialization in ("B3", "B2"):
        initialization_hash = _require_sha256(
            initialization_hashes[initialization], f"{initialization} initialization hash"
        )
        for reward_function in ("answer", "exact_state"):
            arms[f"{initialization}_{reward_function}"] = {
                **shared_arm_fields,
                "initialization": initialization,
                "initialization_hash": initialization_hash,
                "reward_function": reward_function,
            }
    arms["Base_exact_state"] = {
        **shared_arm_fields,
        "initialization": "Base",
        "initialization_hash": _require_sha256(
            initialization_hashes["Base"], "Base initialization hash"
        ),
        "reward_function": "exact_state",
    }
    package = {
        "schema_version": 1,
        "status": "V5_COMMON_ACTION_SPACE_FROZEN",
        "action_parser_id": action_parser_id,
        "rollout_seeds": list(rollout_seeds),
        "prompt_hash": prompt_hash,
        "action_parser_hash": action_parser_hash,
        "rollout_seed_hash": rollout_seed_hash,
        "scene_metadata_hash": scene_metadata_hash,
        "rl_train_hash": shared_arm_fields["rl_train_hash"],
        "rl_eval_hash": shared_arm_fields["rl_eval_hash"],
        "role_counts": shared_arm_fields["role_counts"],
        "initialization_hashes": dict(initialization_hashes),
        "scenes": [dict(scene) for scene in frozen_scenes],
        "arms": arms,
        "preflight": {"only_reward_function_differs": True},
    }
    assert_common_action_preflight(package)
    return package


def assert_common_action_preflight(package: Mapping[str, object]) -> None:
    """Fail closed if the reward-isolation freeze drifts from the specification."""

    if not isinstance(package, Mapping):
        raise CommonActionFreezeError("common-action package must be a mapping")
    arms = package.get("arms")
    if not isinstance(arms, Mapping):
        raise CommonActionFreezeError("common-action package arms must be a mapping")
    if set(arms) != set(_REGISTERED_ARMS):
        raise CommonActionFreezeError("common-action arm set must match the registered package")

    normalized_arms: dict[str, Mapping[str, object]] = {}
    for name, arm in arms.items():
        if not isinstance(arm, Mapping):
            raise CommonActionFreezeError(f"{name} arm must be a mapping")
        normalized_arms[str(name)] = arm
        for field in (
            *_PAIR_FIELDS,
            "rl_train_hash",
            "rl_eval_hash",
            "role_counts",
            "initialization",
            "reward_function",
        ):
            if field not in arm:
                raise CommonActionFreezeError(f"{name} is missing {field}")
        for field in _PAIR_FIELDS:
            _require_sha256(arm[field], f"{name}.{field}")

    scenes = package.get("scenes")
    if not isinstance(scenes, Sequence) or isinstance(scenes, (str, bytes)) or not scenes:
        raise CommonActionFreezeError("common-action package scenes must be a non-empty sequence")
    frozen_scenes = tuple(_freeze_scene(scene, index=index) for index, scene in enumerate(scenes))
    derived = [dict(scene) for scene in frozen_scenes]
    for scene in derived:
        for field in ("fiber_bin", "support_bin", "role"):
            scene.pop(field, None)
    _derive_bins_and_roles(derived)
    if any(
        original.get(field) != expected[field]
        for original, expected in zip(frozen_scenes, derived, strict=True)
        for field in ("fiber_bin", "support_bin", "role")
    ):
        raise CommonActionFreezeError("derived fiber/support bins or train/eval role drifted")
    for original, regenerated in zip(scenes, frozen_scenes, strict=True):
        if (
            not isinstance(original, Mapping)
            or original.get("prompt_hash") != regenerated["prompt_hash"]
            or original.get("prompt_sha256") != regenerated["prompt_sha256"]
            or original.get("reward_labels") != regenerated["reward_labels"]
        ):
            raise CommonActionFreezeError(
                "scene prompt hashes and both reward labels must match frozen content"
            )
    if package.get("action_parser_id") != ACTION_PARSER_ID:
        raise CommonActionFreezeError("action_parser_id drifted from the registered runtime parser")
    if package.get("rollout_seeds") != [PILOT_SEED]:
        raise CommonActionFreezeError("rollout seeds drifted from the fixed pilot seed")
    expected_hashes = {
        "prompt_hash": _sha256_json(
            [
                {"scene_id": scene["scene_id"], "prompt_hash": scene["prompt_hash"]}
                for scene in frozen_scenes
            ]
        ),
        "action_parser_hash": _sha256_text(str(package.get("action_parser_id", ""))),
        "rollout_seed_hash": _sha256_json(list(package.get("rollout_seeds", []))),
        "scene_metadata_hash": _scene_metadata_hash(frozen_scenes),
        "rl_train_hash": _sha256_json(
            [scene for scene in frozen_scenes if scene["role"] == "rl_train"]
        ),
        "rl_eval_hash": _sha256_json(
            [scene for scene in frozen_scenes if scene["role"] == "rl_eval"]
        ),
    }
    for field, expected in expected_hashes.items():
        if package.get(field) != expected:
            raise CommonActionFreezeError(f"{field} does not match the frozen package content")
    expected_counts = {
        "rl_train": sum(scene["role"] == "rl_train" for scene in frozen_scenes),
        "rl_eval": sum(scene["role"] == "rl_eval" for scene in frozen_scenes),
    }
    if package.get("role_counts") != expected_counts:
        raise CommonActionFreezeError("role_counts do not match the frozen split")
    if len(frozen_scenes) == 96 and expected_counts != {"rl_train": 72, "rl_eval": 24}:
        raise CommonActionFreezeError("Study C split must be exactly 72 rl_train / 24 rl_eval")

    baseline = normalized_arms["B3_answer"]
    for field in (
        "prompt_hash",
        "action_parser_hash",
        "rollout_seed_hash",
        "rl_train_hash",
        "rl_eval_hash",
        "role_counts",
    ):
        if baseline[field] != package[field] or any(
            arm[field] != baseline[field] for arm in normalized_arms.values()
        ):
            raise CommonActionFreezeError(f"{field} must remain identical across reward arms")
    if any(
        arm["scene_metadata_hash"] != package["scene_metadata_hash"]
        for arm in normalized_arms.values()
    ):
        raise CommonActionFreezeError(
            "scene_metadata_hash must remain identical across reward arms"
        )

    for initialization in ("B3", "B2"):
        answer = normalized_arms[f"{initialization}_answer"]
        exact_state = normalized_arms[f"{initialization}_exact_state"]
        if (
            answer["initialization"] != initialization
            or exact_state["initialization"] != initialization
        ):
            raise CommonActionFreezeError(f"{initialization} initialization label drifted")
        differing = sorted(
            field
            for field in set(answer) | set(exact_state)
            if answer.get(field) != exact_state.get(field)
        )
        if differing != ["reward_function"]:
            raise CommonActionFreezeError(
                f"{initialization} reward pair differs outside reward_function: {differing}"
            )

    base_arm = normalized_arms["Base_exact_state"]
    if base_arm["initialization"] != "Base":
        raise CommonActionFreezeError("Base_exact_state initialization must remain Base")
    if base_arm["reward_function"] != "exact_state":
        raise CommonActionFreezeError("Base_exact_state reward_function must remain exact_state")
    initialization_hashes = package.get("initialization_hashes")
    if not isinstance(initialization_hashes, Mapping):
        raise CommonActionFreezeError("package initialization_hashes mapping is missing")
    for name, arm in normalized_arms.items():
        initialization = arm["initialization"]
        if (
            initialization not in initialization_hashes
            or arm["initialization_hash"] != (initialization_hashes[initialization])
        ):
            raise CommonActionFreezeError(
                f"{name} initialization_hash differs from the frozen package"
            )
    preflight = package.get("preflight")
    if not isinstance(preflight, Mapping):
        raise CommonActionFreezeError("preflight summary is missing")
    if preflight.get("only_reward_function_differs") is not True:
        raise CommonActionFreezeError("preflight must certify only_reward_function_differs")
    from compensability_v5.training.train_common_space_grpo import (
        assert_common_space_reward_isolation,
    )

    try:
        assert_common_space_reward_isolation(normalized_arms)
    except (RuntimeError, TypeError, ValueError) as error:
        raise CommonActionFreezeError(str(error)) from error


__all__ = [
    "ACTION_PARSER_ID",
    "PILOT_SEED",
    "CommonActionFreezeError",
    "assert_common_action_preflight",
    "freeze_common_action_space",
]
