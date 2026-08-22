"""Answer-reward quotient visibility and null-direction diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from compensability_v5.data.common_action_schema import WorldAction, apply_answer_operation


def reward_visibility(
    truth: tuple[int, int, int, int],
    observation: tuple[int, int, int, int],
    operations: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    truth_action, observed_action = WorldAction(truth), WorldAction(observation)
    result: dict[str, int] = {}
    for index, operation in enumerate(operations):
        operation_id = operation.get("operation_id", f"operation-{index}")
        if not isinstance(operation_id, str):
            raise ValueError("operation_id must be a string")
        closed = {key: operation[key] for key in ("operator", "indices")}
        result[operation_id] = int(
            apply_answer_operation(truth_action, closed)
            != apply_answer_operation(observed_action, closed)
        )
    return result


__all__ = ["reward_visibility"]
