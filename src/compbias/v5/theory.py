"""Low-cost v5 theory helpers for exact numerical verification."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from compbias.rl import _positive_int
from compbias.rl.exact_kl import exact_kl_projection


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return converted


def exact_state_update(
    reference: ArrayLike,
    rewards: ArrayLike,
    beta: float,
) -> NDArray[np.float64]:
    return exact_kl_projection(reference, rewards, beta)


def correction_bearing_group_probability(p_x: float, p_s: float, k: int) -> float:
    exact = _probability(p_x, name="p_x")
    shortcut = _probability(p_s, name="p_s")
    size = _positive_int(k, name="k")
    if exact + shortcut > 1.0:
        raise ValueError("p_x + p_s must not exceed 1")
    return 1.0 - (1.0 - exact) ** size - (exact + shortcut) ** size + shortcut**size


def correction_alignment(
    exact_gradient: ArrayLike,
    reward_gradient: ArrayLike,
    fisher: ArrayLike,
) -> float:
    exact = np.array(exact_gradient, dtype=np.float64, copy=True)
    reward = np.array(reward_gradient, dtype=np.float64, copy=True)
    fisher_matrix = np.array(fisher, dtype=np.float64, copy=True)
    if exact.ndim != 1 or reward.ndim != 1 or exact.shape != reward.shape:
        raise ValueError("gradients must be one-dimensional and shape-aligned")
    if fisher_matrix.shape != (exact.size, exact.size):
        raise ValueError("fisher must be square and match gradient dimension")
    exact_nat = np.linalg.solve(fisher_matrix, exact)
    reward_nat = np.linalg.solve(fisher_matrix, reward)
    numerator = float(exact @ reward_nat)
    exact_norm = float(exact @ exact_nat)
    reward_norm = float(reward @ reward_nat)
    if exact_norm <= 0.0 or reward_norm <= 0.0:
        raise ValueError("fisher-weighted norms must be positive")
    return numerator / math.sqrt(exact_norm * reward_norm)


def orbit_risk_upper_bound(*, base_risk: float, equivariance_defect: float) -> float:
    base = _probability(base_risk, name="base_risk")
    defect = _probability(equivariance_defect, name="equivariance_defect")
    return base + defect


def is_sparse_correction_identifiable(
    matrix: ArrayLike,
    *,
    deltas: Iterable[int],
    sparsity: int = 1,
) -> bool:
    array = np.array(matrix, dtype=np.int64, copy=True)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("matrix must be a non-empty 2D integer array")
    if sparsity != 1:
        raise NotImplementedError("current helper verifies only the one-error case")
    allowed = tuple(int(delta) for delta in deltas)
    if not allowed or any(delta == 0 for delta in allowed):
        raise ValueError("deltas must be a non-empty iterable of non-zero integers")
    seen: set[tuple[int, ...]] = set()
    for column_index in range(array.shape[1]):
        column = array[:, column_index]
        for delta in allowed:
            signature = tuple(int(value) for value in (delta * column))
            if signature in seen:
                return False
            seen.add(signature)
    return True


__all__ = [
    "correction_alignment",
    "correction_bearing_group_probability",
    "exact_state_update",
    "is_sparse_correction_identifiable",
    "orbit_risk_upper_bound",
]
