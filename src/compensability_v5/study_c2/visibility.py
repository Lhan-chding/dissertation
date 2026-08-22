"""Reward visibility bins for verifier-null and separating interventions."""

from __future__ import annotations


def visibility_bin(answer_difference: int) -> str:
    magnitude = abs(int(answer_difference))
    if magnitude == 0:
        return "reward_null"
    if magnitude == 1:
        return "low_visibility"
    return "high_visibility"


__all__ = ["visibility_bin"]
