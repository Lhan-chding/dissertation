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


def autograd_gradient_diagnostics(
    *,
    log_probabilities: object,
    trainable_parameters: Sequence[object],
    state_rewards: Sequence[float],
    answer_rewards: Sequence[float],
) -> dict[str, object]:
    """Measure both verifier gradients on one differentiable rollout group."""

    import torch

    if (
        not isinstance(log_probabilities, torch.Tensor)
        or log_probabilities.ndim != 1
        or len(log_probabilities) < 2
        or len(state_rewards) != len(log_probabilities)
        or len(answer_rewards) != len(log_probabilities)
    ):
        raise ValueError(
            "log probabilities and reward vectors must be aligned one-dimensional data"
        )
    parameters = tuple(trainable_parameters)
    if not parameters or any(
        not isinstance(parameter, torch.Tensor) or not parameter.requires_grad
        for parameter in parameters
    ):
        raise ValueError("gradient audit requires non-empty trainable parameters")
    if not torch.isfinite(log_probabilities).all():
        raise ValueError("log probabilities must be finite")

    state_values = centered_advantages(state_rewards)
    answer_values = centered_advantages(answer_rewards)
    state_advantages = torch.as_tensor(
        state_values, dtype=log_probabilities.dtype, device=log_probabilities.device
    )
    answer_advantages = torch.as_tensor(
        answer_values, dtype=log_probabilities.dtype, device=log_probabilities.device
    )
    state_objective = torch.dot(state_advantages, log_probabilities)
    answer_objective = torch.dot(answer_advantages, log_probabilities)
    state_gradients = torch.autograd.grad(
        state_objective,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    answer_gradients = torch.autograd.grad(
        answer_objective,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    state_square = torch.zeros((), dtype=torch.float64, device=log_probabilities.device)
    answer_square = torch.zeros((), dtype=torch.float64, device=log_probabilities.device)
    difference_square = torch.zeros((), dtype=torch.float64, device=log_probabilities.device)
    dot_product = torch.zeros((), dtype=torch.float64, device=log_probabilities.device)
    finite = True
    for parameter, state_gradient, answer_gradient in zip(
        parameters, state_gradients, answer_gradients, strict=True
    ):
        state_tensor = torch.zeros_like(parameter) if state_gradient is None else state_gradient
        answer_tensor = torch.zeros_like(parameter) if answer_gradient is None else answer_gradient
        finite = (
            finite
            and bool(torch.isfinite(state_tensor).all())
            and bool(torch.isfinite(answer_tensor).all())
        )
        state_float = state_tensor.detach().to(dtype=torch.float64)
        answer_float = answer_tensor.detach().to(dtype=torch.float64)
        state_square = state_square + torch.sum(state_float * state_float)
        answer_square = answer_square + torch.sum(answer_float * answer_float)
        difference = state_float - answer_float
        difference_square = difference_square + torch.sum(difference * difference)
        dot_product = dot_product + torch.sum(state_float * answer_float)

    state_norm = float(torch.sqrt(state_square).item())
    answer_norm = float(torch.sqrt(answer_square).item())
    difference_norm = float(torch.sqrt(difference_square).item())
    if state_norm == 0.0 and answer_norm == 0.0:
        cosine = 1.0
    elif state_norm == 0.0 or answer_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(dot_product.item() / (state_norm * answer_norm))
        cosine = min(1.0, max(-1.0, cosine))
    return {
        "reward_hamming_distance": sum(
            float(left) != float(right)
            for left, right in zip(state_rewards, answer_rewards, strict=True)
        ),
        "state_advantages": state_values.tolist(),
        "answer_advantages": answer_values.tolist(),
        "gradient_state_norm": state_norm,
        "gradient_answer_norm": answer_norm,
        "gradient_difference_norm": difference_norm,
        "gradient_cosine": cosine,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "finite": finite,
    }


__all__ = [
    "autograd_gradient_diagnostics",
    "centered_advantages",
    "shared_gradient_diagnostics",
]
