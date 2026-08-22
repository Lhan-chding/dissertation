"""Reward contrast rank and normalized strength."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def reward_contrast(
    state_rewards: Sequence[float], answer_rewards: Sequence[float]
) -> dict[str, object]:
    if len(state_rewards) != len(answer_rewards) or not state_rewards:
        raise ValueError("reward vectors must be non-empty and aligned")
    matrix = np.asarray((state_rewards, answer_rewards), dtype=float)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    denominator = np.linalg.norm(centered[0]) + np.linalg.norm(centered[1]) + 1e-12
    return {
        "contrast_rank": int(np.linalg.matrix_rank(centered)),
        "second_singular_value": float(singular[1] if len(singular) > 1 else 0.0),
        "normalized_contrast_strength": float(
            np.linalg.norm(centered[0] - centered[1]) / denominator
        ),
        "centered_state": centered[0].tolist(),
        "centered_answer": centered[1].tolist(),
    }


__all__ = ["reward_contrast"]
