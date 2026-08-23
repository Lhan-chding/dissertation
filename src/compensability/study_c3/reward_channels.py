"""Binary and lexicographic Study C3 reward channels."""

from __future__ import annotations

from .action_taxonomy import ActionClass

ARM_IDS = ("A_BIN", "X_BIN", "A_LEX", "X_LEX")


def rewards_for_class(action_class: ActionClass, *, arm: str) -> dict[str, int]:
    if not isinstance(action_class, ActionClass):
        raise ValueError("Study C3 reward requires a registered X/S/W/I action class")
    if arm not in ARM_IDS:
        raise ValueError(f"unregistered Study C3 arm: {arm}")
    verifier = arm.split("_", maxsplit=1)[0]
    semantic = int(
        action_class in ({ActionClass.X, ActionClass.S} if verifier == "A" else {ActionClass.X})
    )
    validity = int(action_class is not ActionClass.I)
    combined = semantic if arm.endswith("BIN") else 2 * semantic + validity
    return {
        "semantic_reward": semantic,
        "validity_reward": validity,
        "combined_reward": combined,
    }


def _argmax(arm: str) -> frozenset[ActionClass]:
    values = {kind: rewards_for_class(kind, arm=arm)["combined_reward"] for kind in ActionClass}
    maximum = max(values.values())
    return frozenset(kind for kind, value in values.items() if value == maximum)


def verify_argmax_preservation() -> dict[str, bool]:
    return {
        "A_LEX_equals_A_BIN": _argmax("A_LEX") == _argmax("A_BIN"),
        "X_LEX_equals_X_BIN": _argmax("X_LEX") == _argmax("X_BIN"),
    }


__all__ = ["ARM_IDS", "rewards_for_class", "verify_argmax_preservation"]
