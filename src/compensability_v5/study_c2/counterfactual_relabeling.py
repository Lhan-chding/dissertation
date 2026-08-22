"""Counterfactual verifier relabeling on an immutable action buffer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation


def counterfactual_reward_matrix(
    *,
    actions: Sequence[tuple[int, int, int, int]],
    truth: tuple[int, int, int, int],
    operations: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if not operations:
        raise ValueError("counterfactual relabeling requires at least one operation")
    truth_action = WorldAction(truth)
    rows: list[dict[str, object]] = []
    for action_index, action in enumerate(actions):
        candidate = WorldAction(action)
        rewards: dict[str, int] = {}
        for operation_index, operation in enumerate(operations):
            operation_id = operation.get("operation_id", f"operation-{operation_index}")
            if not isinstance(operation_id, str):
                raise ValueError("operation_id must be a string")
            closed = {key: operation[key] for key in ("operator", "indices")}
            rewards[operation_id] = int(
                apply_answer_operation(candidate, closed)
                == apply_answer_operation(truth_action, closed)
            )
        state = int(action == truth)
        rows.append(
            {
                "action_index": action_index,
                "action": list(action),
                "state_reward": state,
                "answer_rewards": rewards,
                "shortcut_operation_count": sum(
                    value == 1 and state == 0 for value in rewards.values()
                ),
                "reward_label_flip_count": sum(
                    value != next(iter(rewards.values())) for value in rewards.values()
                ),
            }
        )
    return tuple(rows)


__all__ = ["counterfactual_reward_matrix"]
