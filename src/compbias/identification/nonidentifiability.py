"""Constructive witness for end-to-end factorization non-identifiability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _stochastic_matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain finite non-negative probabilities")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"each row of {name} must sum to one")
    return np.array(matrix, copy=True)


@dataclass(frozen=True, slots=True)
class RelabeledFactorization:
    """A distinct latent labeling with exactly the same observable kernel."""

    perception: np.ndarray
    reasoning: np.ndarray
    permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        perception = np.array(self.perception, dtype=np.float64, copy=True)
        reasoning = np.array(self.reasoning, dtype=np.float64, copy=True)
        perception.setflags(write=False)
        reasoning.setflags(write=False)
        object.__setattr__(self, "perception", perception)
        object.__setattr__(self, "reasoning", reasoning)


def relabel_factorization(
    perception: object,
    reasoning: object,
    permutation: tuple[int, ...],
) -> RelabeledFactorization:
    """Relabel a finite mediator without changing end-to-end behavior.

    This is a constructive theorem witness, not an estimator of a privileged
    perception boundary.
    """

    perception_matrix = _stochastic_matrix(perception, "perception")
    reasoning_matrix = _stochastic_matrix(reasoning, "reasoning")
    latent_size = perception_matrix.shape[1]
    if reasoning_matrix.shape[0] != latent_size:
        raise ValueError("perception columns must equal reasoning rows")
    if (
        isinstance(permutation, (str, bytes))
        or len(permutation) != latent_size
        or any(isinstance(index, bool) or not isinstance(index, int) for index in permutation)
        or set(permutation) != set(range(latent_size))
    ):
        raise ValueError("permutation must be a bijection over mediator indices")

    transformed_perception = np.empty_like(perception_matrix)
    transformed_reasoning = np.empty_like(reasoning_matrix)
    for old_index, new_index in enumerate(permutation):
        transformed_perception[:, new_index] = perception_matrix[:, old_index]
        transformed_reasoning[new_index, :] = reasoning_matrix[old_index, :]
    return RelabeledFactorization(
        perception=transformed_perception,
        reasoning=transformed_reasoning,
        permutation=tuple(permutation),
    )
