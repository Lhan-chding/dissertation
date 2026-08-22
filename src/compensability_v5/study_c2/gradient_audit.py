"""Shared-action reward and score-vector gradient diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def centered_advantages(rewards: Sequence[float]) -> np.ndarray:
    values = np.asarray(rewards, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("reward vector must contain at least two finite values")
    return values - values.mean()


def shared_gradient_diagnostics(
    *,
    state_rewards: Sequence[float],
    answer_rewards: Sequence[float],
    score_vectors: Sequence[Sequence[float]],
) -> dict[str, object]:
    state = centered_advantages(state_rewards)
    answer = centered_advantages(answer_rewards)
    scores = np.asarray(score_vectors, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != len(state) or len(answer) != len(state):
        raise ValueError("score vectors and rewards must align")
    g_state = state @ scores
    g_answer = answer @ scores
    difference = g_state - g_answer
    state_norm = float(np.linalg.norm(g_state))
    answer_norm = float(np.linalg.norm(g_answer))
    if state_norm == 0.0 and answer_norm == 0.0:
        cosine = 1.0
    elif state_norm == 0.0 or answer_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(g_state, g_answer) / (state_norm * answer_norm))
    return {
        "reward_hamming_distance": sum(
            float(left) != float(right)
            for left, right in zip(state_rewards, answer_rewards, strict=True)
        ),
        "gradient_state_norm": state_norm,
        "gradient_answer_norm": answer_norm,
        "gradient_difference_norm": float(np.linalg.norm(difference)),
        "gradient_cosine": cosine,
        "state_gradient": g_state.tolist(),
        "answer_gradient": g_answer.tolist(),
    }


__all__ = ["centered_advantages", "shared_gradient_diagnostics"]
