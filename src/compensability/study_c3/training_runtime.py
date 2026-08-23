"""Three-channel reward tracing and group diagnostics for C3-C training."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .action_taxonomy import classify_action, classify_failure
from .factorial_design import ARM_IDS, summarize_training_group
from .io import append_jsonl, read_jsonl
from .reward_channels import rewards_for_class
from .semantic_action_parser import parse_p1

TokenCounter = Callable[[str], int]


def _completion_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        messages = tuple(value)
        if messages and all(isinstance(message, Mapping) for message in messages):
            content = messages[-1].get("content")  # type: ignore[union-attr]
            if isinstance(content, str):
                return content
    raise ValueError("Study C3 completion has an unsupported structure")


def _expand(values: object, *, size: int, label: str) -> tuple[object, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"Study C3 reward metadata {label} is malformed")
    items = tuple(values)
    if len(items) == size:
        return items
    if size % len(items):
        raise ValueError(f"Study C3 reward metadata {label} cannot align to completions")
    return tuple(item for item in items for _ in range(size // len(items)))


def build_traced_reward(
    *,
    arm_config: Mapping[str, object],
    training_rows: Sequence[Mapping[str, object]],
    trace_path: Path,
    group_size: int,
    token_counter: TokenCounter,
) -> Callable[..., list[float]]:
    arm = arm_config.get("arm")
    if arm not in ARM_IDS or arm_config.get("action_protocol") != "canonical_semantic_world_v1":
        raise ValueError("Study C3 reward arm is unregistered")
    index = {str(row.get("scene_id")): row for row in training_rows}
    if not index or len(index) != len(training_rows) or "None" in index:
        raise ValueError("Study C3 reward rows are empty or duplicated")
    group_index = 0
    reward_call_index = 0
    if trace_path.exists():
        existing = read_jsonl(trace_path)
        if existing:
            group_index = max(int(row["group_index"]) for row in existing) + 1
            reward_call_index = max(int(row["reward_call_index"]) for row in existing) + 1

    def reward(completions: Sequence[object], **kwargs: object) -> list[float]:
        nonlocal group_index, reward_call_index
        if (
            not isinstance(completions, Sequence)
            or isinstance(completions, (str, bytes))
            or not completions
            or len(completions) % group_size
        ):
            raise ValueError(f"Study C3 reward batch must be a multiple of K={group_size}")
        scene_ids = _expand(kwargs.get("scene_id"), size=len(completions), label="scene_id")
        state = kwargs.get("trainer_state")
        raw_step = getattr(state, "global_step", -1)
        trainer_step = raw_step if type(raw_step) is int else -1
        selected: list[float] = []
        trace_rows: list[dict[str, object]] = []
        for start in range(0, len(completions), group_size):
            group = completions[start : start + group_size]
            group_scenes = scene_ids[start : start + group_size]
            if len(set(group_scenes)) != 1:
                raise ValueError("Study C3 reward group crosses prompt boundaries")
            scene_id = group_scenes[0]
            if not isinstance(scene_id, str) or scene_id not in index:
                raise ValueError(f"Study C3 reward received unknown scene: {scene_id}")
            source = index[scene_id]
            truth = source.get("truth")
            operation = source.get("operation")
            if (
                not isinstance(truth, list)
                or len(truth) != 4
                or any(type(value) is not int for value in truth)
                or not isinstance(operation, Mapping)
            ):
                raise ValueError("Study C3 reward source truth/operation is malformed")
            for position, completion in enumerate(group):
                text = _completion_text(completion)
                label = classify_action(
                    parse_p1(text),
                    truth=tuple(truth),  # type: ignore[arg-type]
                    operation=operation,
                )
                channels = rewards_for_class(label.action_class, arm=str(arm))
                token_length = token_counter(text)
                if type(token_length) is not int or token_length < 0:
                    raise ValueError("Study C3 tokenizer returned an invalid completion length")
                combined = float(channels["combined_reward"])
                selected.append(combined)
                action_id = hashlib.sha256(
                    f"{scene_id}:{group_index}:{position}:{text}".encode()
                ).hexdigest()
                trace_rows.append(
                    {
                        "schema_version": 3,
                        "arm": arm,
                        "reward_function_id": arm_config["reward_function_id"],
                        "trainer_step": trainer_step,
                        "reward_call_index": reward_call_index,
                        "group_index": group_index,
                        "position": position,
                        "scene_id": scene_id,
                        "pair_id": source.get("pair_id"),
                        "condition": source.get("condition"),
                        "family": source.get("family"),
                        "prompt_sha256": source.get("prompt_sha256"),
                        "action_id": action_id,
                        "completion": text,
                        "completion_token_length": token_length,
                        "parsed_world": (
                            None if label.parsed_world is None else list(label.parsed_world)
                        ),
                        "action_class": label.action_class.value,
                        "failure_category": classify_failure(text).value,
                        **channels,
                    }
                )
            group_index += 1
        append_jsonl(trace_path, trace_rows)
        reward_call_index += 1
        return selected

    reward.__name__ = str(arm_config["reward_function_id"])
    return reward


def build_group_diagnostics(
    trace_path: Path, *, expected_group_count: int, group_size: int
) -> tuple[dict[str, object], ...]:
    rows = read_jsonl(trace_path)
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        group_index = row.get("group_index")
        if type(group_index) is not int or group_index < 0:
            raise ValueError("Study C3 trace group index is invalid")
        grouped.setdefault(group_index, []).append(row)
    if set(grouped) != set(range(expected_group_count)):
        raise ValueError("Study C3 trace group count/order drifted")
    diagnostics: list[dict[str, object]] = []
    for group_index in range(expected_group_count):
        group = grouped[group_index]
        if len({row.get("scene_id") for row in group}) != 1:
            raise ValueError("Study C3 trace group crosses scenes")
        measured = summarize_training_group(group, expected_size=group_size)
        diagnostics.append(
            {
                "schema_version": 3,
                "group_index": group_index,
                "scene_id": group[0]["scene_id"],
                "pair_id": group[0].get("pair_id"),
                "condition": group[0].get("condition"),
                "family": group[0].get("family"),
                **measured,
            }
        )
    return tuple(diagnostics)


__all__ = ["build_group_diagnostics", "build_traced_reward"]
