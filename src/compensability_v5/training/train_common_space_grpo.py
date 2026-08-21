"""Common-action-space isolation gates for v5 GRPO.

No accelerator library is imported here.  A validated manifest may be handed
to a separately installed server runner only after every reward-isolation
field has been checked.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from compensability_v5.training.train_support_lora import (
    ServerExecutionBlocked,
    ValidatedExecution,
)

COMMON_SPACE_GRPO_ACK = "I_UNDERSTAND_THIS_STARTS_V5_COMMON_SPACE_GRPO"
REGISTERED_ARMS = frozenset(
    {
        "B3_answer",
        "B3_exact_state",
        "B2_answer",
        "B2_exact_state",
        "Base_exact_state",
    }
)
SHARED_FIELDS = (
    "prompt_hash",
    "initialization_hash",
    "action_parser_hash",
    "rollout_seeds",
    "decoding",
    "optimizer",
    "steps",
    "group_size",
)


def assert_common_space_reward_isolation(
    arms: Mapping[str, Mapping[str, object]],
) -> None:
    """Require registered arms to differ only in reward and initialization arm.

    B2/B3/Base intentionally identify different initial checkpoints.  Within
    each initialization, reward arms must share every other registered field.
    The Base arm is a support-negative exact-state control and has no answer
    counterpart.
    """

    if set(arms) != REGISTERED_ARMS:
        missing = sorted(REGISTERED_ARMS - set(arms))
        extra = sorted(set(arms) - REGISTERED_ARMS)
        raise ServerExecutionBlocked(
            f"common-space arm set drifted; missing={missing}, extra={extra}"
        )
    required = {*SHARED_FIELDS, "initialization", "reward_function", "action_space"}
    for name, arm in arms.items():
        absent = sorted(required - set(arm))
        if absent:
            raise ServerExecutionBlocked(f"{name} is missing common-space fields: {absent}")
        if arm.get("action_space") != "four_integer_world":
            raise ServerExecutionBlocked(f"{name} changed the common four-integer action space")
        expected_reward = "answer" if name.endswith("_answer") else "exact_state"
        if arm.get("reward_function") != expected_reward:
            raise ServerExecutionBlocked(f"{name} reward function is inconsistent with its arm")

    global_fields = (
        "prompt_hash",
        "action_parser_hash",
        "rollout_seeds",
        "decoding",
        "optimizer",
        "steps",
        "group_size",
    )
    baseline = arms["B3_answer"]
    for field in global_fields:
        if any(arm.get(field) != baseline.get(field) for arm in arms.values()):
            raise ServerExecutionBlocked(f"common-space field differs across reward arms: {field}")
    for initialization in ("B3", "B2"):
        answer = arms[f"{initialization}_answer"]
        state = arms[f"{initialization}_exact_state"]
        if (
            answer.get("initialization") != initialization
            or state.get("initialization") != initialization
        ):
            raise ServerExecutionBlocked(f"{initialization} initialization label drifted")
        if answer.get("initialization_hash") != state.get("initialization_hash"):
            raise ServerExecutionBlocked(
                f"{initialization} initialization hash differs across reward arms"
            )
        differing = {key for key in set(answer) | set(state) if answer.get(key) != state.get(key)}
        if differing != {"reward_function"}:
            raise ServerExecutionBlocked(
                f"{initialization} reward pair differs outside reward_function: {sorted(differing)}"
            )
    if arms["Base_exact_state"].get("initialization") != "Base":
        raise ServerExecutionBlocked("Base support-negative control initialization drifted")


def load_common_space_arms(path: Path) -> dict[str, dict[str, object]]:
    """Load and validate the arms from a hash-bound common-action freeze."""

    if path.is_symlink() or not path.is_file():
        raise ServerExecutionBlocked(f"common-action manifest is missing or unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ServerExecutionBlocked(f"common-action manifest is invalid JSON: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ServerExecutionBlocked("common-action manifest must be a schema_version 1 mapping")
    from compensability_v5.data.common_action_freeze import assert_common_action_preflight

    assert_common_action_preflight(payload)
    source = payload.get("arms")
    if not isinstance(source, Mapping):
        raise ServerExecutionBlocked("common-action manifest arms mapping is missing")
    arms: dict[str, dict[str, object]] = {}
    for name, arm in source.items():
        if not isinstance(name, str) or not isinstance(arm, Mapping):
            raise ServerExecutionBlocked("common-action arm entries must be named mappings")
        arms[name] = dict(arm)
    assert_common_space_reward_isolation(arms)
    return arms


def execute_common_space_grpo(
    validation: ValidatedExecution,
    arms: Mapping[str, Mapping[str, object]],
    *,
    runner: Callable[[Mapping[str, object], Mapping[str, Mapping[str, object]]], Any] | None = None,
) -> Any:
    """Validate reward isolation, then delegate only to an injected server runner."""

    assert_common_space_reward_isolation(arms)
    if runner is None:
        raise ServerExecutionBlocked(
            "server GRPO runner is not installed in the local package; transfer the frozen bundle"
        )
    detached = {name: dict(arm) for name, arm in arms.items()}
    return runner(validation.to_mapping(), detached)


def common_space_fixture() -> dict[str, dict[str, object]]:
    """Return a small immutable-by-construction fixture for local gate checks."""

    shared: dict[str, object] = {
        "prompt_hash": "a" * 64,
        "action_parser_hash": "b" * 64,
        "rollout_seeds": [11],
        "decoding": {"temperature": 0.7, "top_p": 1.0},
        "optimizer": {"name": "adamw", "learning_rate": 2e-5},
        "steps": 8,
        "group_size": 4,
        "action_space": "four_integer_world",
    }
    result: dict[str, dict[str, object]] = {}
    for initialization in ("B3", "B2"):
        initialization_hash = ("c" if initialization == "B3" else "d") * 64
        for reward in ("answer", "exact_state"):
            result[f"{initialization}_{reward}"] = {
                **shared,
                "initialization": initialization,
                "initialization_hash": initialization_hash,
                "reward_function": reward,
            }
    result["Base_exact_state"] = {
        **shared,
        "initialization": "Base",
        "initialization_hash": "e" * 64,
        "reward_function": "exact_state",
    }
    return result


__all__: Sequence[str] = (
    "COMMON_SPACE_GRPO_ACK",
    "REGISTERED_ARMS",
    "SHARED_FIELDS",
    "assert_common_space_reward_isolation",
    "common_space_fixture",
    "execute_common_space_grpo",
    "load_common_space_arms",
)
