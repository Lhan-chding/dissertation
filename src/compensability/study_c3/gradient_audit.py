"""Same-action-buffer gradient measurements for Study C3 reward channels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def validate_shared_action_batch(
    rows: Sequence[Mapping[str, object]], *, expected_action_ids: Sequence[str]
) -> bool:
    observed = [row.get("action_id") for row in rows]
    if (
        not rows
        or len(rows) != len(expected_action_ids)
        or observed != list(expected_action_ids)
        or len(set(observed)) != len(observed)
        or any(not isinstance(row.get("completion"), str) for row in rows)
    ):
        raise ValueError("Study C3 gradients must use the same action batch")
    return True


def _gradient(rewards: Sequence[float], scores: np.ndarray) -> np.ndarray:
    values = np.asarray(rewards, dtype=float)
    if values.ndim != 1 or len(values) != len(scores) or not np.isfinite(values).all():
        raise ValueError("Study C3 reward vectors must align and be finite")
    return (values - values.mean()) @ scores


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return min(1.0, max(-1.0, float(np.dot(left, right) / (left_norm * right_norm))))


def shared_reward_gradient_diagnostics(
    *,
    reward_vectors: Mapping[str, Sequence[float]],
    score_vectors: Sequence[Sequence[float]],
) -> dict[str, object]:
    required = {"A", "X", "V", "A_LEX", "X_LEX"}
    if set(reward_vectors) != required:
        raise ValueError("Study C3 gradient audit requires A/X/V/A_LEX/X_LEX rewards")
    scores = np.asarray(score_vectors, dtype=float)
    if scores.ndim != 2 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("Study C3 score vectors must be an aligned finite matrix")
    gradients = {name: _gradient(values, scores) for name, values in reward_vectors.items()}
    norms = {name: float(np.linalg.norm(value)) for name, value in gradients.items()}
    result = {
        "gradient_norms": norms,
        "cosine_X_V": _cosine(gradients["X"], gradients["V"]),
        "cosine_A_V": _cosine(gradients["A"], gradients["V"]),
        "difference_X_A": float(np.linalg.norm(gradients["X"] - gradients["A"])),
        "difference_X_LEX_X": float(np.linalg.norm(gradients["X_LEX"] - gradients["X"])),
        "difference_A_LEX_A": float(np.linalg.norm(gradients["A_LEX"] - gradients["A"])),
        "same_action_batch": True,
        "finite": all(math.isfinite(value) for value in norms.values()),
    }
    if not result["finite"]:
        raise ValueError("Study C3 gradient diagnostics are non-finite")
    return result


def autograd_reward_gradient_diagnostics(
    *,
    log_probabilities: object,
    trainable_parameters: Sequence[object],
    reward_vectors: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Compute all five gradients from the same differentiable action batch."""

    import torch

    required = ("A", "X", "V", "A_LEX", "X_LEX")
    if (
        not isinstance(log_probabilities, torch.Tensor)
        or log_probabilities.ndim != 1
        or len(log_probabilities) < 2
        or set(reward_vectors) != set(required)
        or not torch.isfinite(log_probabilities).all()
    ):
        raise ValueError("Study C3 log probabilities/reward vectors are malformed")
    parameters = tuple(trainable_parameters)
    if not parameters or any(
        not isinstance(parameter, torch.Tensor) or not parameter.requires_grad
        for parameter in parameters
    ):
        raise ValueError("Study C3 gradient audit requires trainable parameters")
    flattened: dict[str, torch.Tensor] = {}
    for index, name in enumerate(required):
        raw = np.asarray(reward_vectors[name], dtype=float)
        if raw.shape != (len(log_probabilities),) or not np.isfinite(raw).all():
            raise ValueError("Study C3 autograd reward vectors must align")
        advantages = torch.as_tensor(
            raw - raw.mean(),
            dtype=log_probabilities.dtype,
            device=log_probabilities.device,
        )
        gradients = torch.autograd.grad(
            torch.dot(advantages, log_probabilities),
            parameters,
            retain_graph=index < len(required) - 1,
            allow_unused=True,
        )
        flattened[name] = torch.cat(
            [
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.detach().reshape(-1)
                for parameter, gradient in zip(parameters, gradients, strict=True)
            ]
        ).to(dtype=torch.float32)
    if any(not torch.isfinite(flattened[name]).all() for name in required):
        raise ValueError("Study C3 autograd gradients are non-finite")
    gradients = {name: flattened[name] for name in required}

    def distance(left: str, right: str) -> float:
        return float(torch.linalg.vector_norm(gradients[left] - gradients[right]).item())

    def cosine(left: str, right: str) -> float:
        left_norm = float(torch.linalg.vector_norm(gradients[left]).item())
        right_norm = float(torch.linalg.vector_norm(gradients[right]).item())
        if left_norm == right_norm == 0.0:
            return 1.0
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return min(
            1.0,
            max(
                -1.0,
                float(torch.dot(gradients[left], gradients[right]).item())
                / (left_norm * right_norm),
            ),
        )

    return {
        "gradient_norms": {
            name: float(torch.linalg.vector_norm(gradients[name]).item()) for name in required
        },
        "cosine_X_V": cosine("X", "V"),
        "cosine_A_V": cosine("A", "V"),
        "difference_X_A": distance("X", "A"),
        "difference_X_LEX_X": distance("X_LEX", "X"),
        "difference_A_LEX_A": distance("A_LEX", "A"),
        "same_action_batch": True,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "finite": True,
    }


__all__ = [
    "autograd_reward_gradient_diagnostics",
    "shared_reward_gradient_diagnostics",
    "validate_shared_action_batch",
]
