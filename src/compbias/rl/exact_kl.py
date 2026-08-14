"""Exact KL-regularized selection on a finite joint trajectory table."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from compbias.rl import _frozen_array, _positive_float, _probability_array, _reward_array


def exact_kl_projection(
    reference: ArrayLike,
    rewards: ArrayLike,
    beta: float,
) -> NDArray[np.float64]:
    r"""Return ``reference * exp(rewards / beta)`` after stable normalization.

    The input can be any non-empty finite table.  Zero-probability entries stay
    outside the selected support, matching the forward-KL constraint exactly.
    """

    reference_array = _probability_array(reference, name="reference")
    reward_array = _reward_array(rewards, shape=reference_array.shape)
    beta_value = _positive_float(beta, name="beta")
    support = reference_array > 0.0

    supported_rewards = reward_array[support]
    reward_pivot = float(np.max(supported_rewards))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled_rewards = (supported_rewards - reward_pivot) / beta_value
        log_weights = np.log(reference_array[support]) + scaled_rewards
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)

    normalizer = float(weights.sum(dtype=np.float64))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise FloatingPointError("the KL projection could not be normalized")
    selected = np.zeros_like(reference_array)
    selected[support] = weights / normalizer
    return _frozen_array(selected)


__all__ = ["exact_kl_projection"]
