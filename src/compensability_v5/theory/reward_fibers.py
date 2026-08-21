"""KL projections and conditional distributions on finite reward fibers."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

_DISTRIBUTION_TOLERANCE = 1e-12


def _one_dimensional_floats(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional finite numeric sequence") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True)


def _probability_vector(values: ArrayLike, *, name: str = "prior") -> NDArray[np.float64]:
    probabilities = _one_dimensional_floats(values, name=name)
    if np.any(probabilities < 0.0):
        raise ValueError(f"{name} cannot contain negative values")
    total = float(np.sum(probabilities, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=_DISTRIBUTION_TOLERANCE):
        raise ValueError(f"{name} must sum to one")
    if not np.any(probabilities > 0.0):
        raise ValueError(f"{name} must have non-empty support")
    return probabilities


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def kl_reward_projection(
    prior: ArrayLike,
    rewards: ArrayLike,
    beta: float,
) -> tuple[float, ...]:
    r"""Return the stable finite-space optimum ``prior * exp(reward / beta)``."""

    prior_array = _probability_vector(prior)
    reward_array = _one_dimensional_floats(rewards, name="rewards")
    if reward_array.shape != prior_array.shape:
        raise ValueError("prior and rewards must have the same length")
    beta_value = _positive_finite(beta, name="beta")

    support = prior_array > 0.0
    supported_prior = prior_array[support].astype(np.longdouble)
    supported_rewards = reward_array[support].astype(np.longdouble)
    reward_pivot = np.max(supported_rewards)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_weights = np.log(supported_prior) + (supported_rewards - reward_pivot) / np.longdouble(
            beta_value
        )
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
    normalizer = np.sum(weights, dtype=np.longdouble)
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise FloatingPointError("the KL reward projection could not be normalized")

    projected = np.zeros(prior_array.shape, dtype=np.longdouble)
    projected[support] = weights / normalizer
    result = tuple(float(value) for value in projected)
    if not np.isclose(sum(result), 1.0, rtol=0.0, atol=_DISTRIBUTION_TOLERANCE):
        raise FloatingPointError("the KL reward projection lost probability mass")
    return result


def conditional_distribution(
    distribution: ArrayLike,
    indices: Iterable[int],
) -> tuple[float, ...]:
    """Condition a finite distribution on the ordered set of ``indices``."""

    probabilities = _probability_vector(distribution, name="distribution")
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Iterable):
        raise TypeError("indices must be an iterable of integer positions")
    raw_indices = tuple(indices)
    if not raw_indices:
        raise ValueError("indices must be non-empty")

    normalized: list[int] = []
    for position, value in enumerate(raw_indices):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"indices[{position}] must be an integer")
        index = int(value)
        if not 0 <= index < probabilities.size:
            raise ValueError(f"indices[{position}] is outside the distribution")
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError("indices must not contain duplicates")

    selected = probabilities[normalized]
    mass = float(np.sum(selected, dtype=np.float64))
    if mass <= 0.0:
        raise ValueError("the conditioning event must have positive probability")
    return tuple(float(value / mass) for value in selected)


__all__ = ["conditional_distribution", "kl_reward_projection"]
