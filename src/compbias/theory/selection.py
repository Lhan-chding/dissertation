"""Exact KL selection identities for NumPy and PyTorch arrays."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np

try:  # Torch is an optional project dependency.
    import torch
except ImportError:  # pragma: no cover - exercised in NumPy-only installations
    torch = None  # type: ignore[assignment]


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _uses_torch(*values: Any) -> bool:
    tensor_flags = tuple(_is_torch_tensor(value) for value in values)
    if any(tensor_flags) and not all(tensor_flags):
        raise TypeError("array arguments must all use the same NumPy or Torch backend")
    return any(tensor_flags)


def _as_float_array(value: Any, name: str, *, use_torch: bool) -> Any:
    if use_torch:
        if not _is_torch_tensor(value):  # guarded by _uses_torch, useful for direct calls
            raise TypeError(f"{name} must be a Torch tensor")
        if value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{name} must contain real numeric values")
        return value if value.is_floating_point() else value.to(dtype=torch.float64)

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    return array


def _all_finite(value: Any, *, use_torch: bool) -> bool:
    if use_torch:
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(value).all())


def _validate_vector(value: Any, name: str, *, use_torch: bool) -> Any:
    array = _as_float_array(value, name, use_torch=use_torch)
    size = array.numel() if use_torch else array.size
    if array.ndim != 1 or size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not _all_finite(array, use_torch=use_torch):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_probability_vector(value: Any, name: str, *, use_torch: bool) -> Any:
    probabilities = _validate_vector(value, name, use_torch=use_torch)
    if use_torch:
        if bool((probabilities < 0).any().item()):
            raise ValueError(f"{name} must be non-negative")
        total = probabilities.sum()
        one = total.new_tensor(1.0)
        valid_total = bool(torch.isclose(total, one, rtol=1e-7, atol=1e-8).item())
    else:
        if bool(np.any(probabilities < 0)):
            raise ValueError(f"{name} must be non-negative")
        valid_total = bool(np.isclose(probabilities.sum(), 1.0, rtol=1e-7, atol=1e-8))
    if not valid_total:
        raise ValueError(f"{name} must sum to one")
    return probabilities / probabilities.sum()


def _validate_positive_vector(value: Any, name: str, *, use_torch: bool) -> Any:
    vector = _validate_vector(value, name, use_torch=use_torch)
    invalid = bool((vector <= 0).any().item()) if use_torch else bool(np.any(vector <= 0))
    if invalid:
        raise ValueError(f"{name} must be strictly positive")
    return vector


def _validate_same_shape(left: Any, right: Any, left_name: str, right_name: str) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape")
    if _is_torch_tensor(left) and left.device != right.device:
        raise ValueError(f"{left_name} and {right_name} must be on the same device")


def _validate_beta(beta: float) -> float:
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise ValueError("beta must be a finite positive scalar")
    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value <= 0:
        raise ValueError("beta must be a finite positive scalar")
    return beta_value


def _safe_log_nonnegative(value: Any, *, use_torch: bool) -> Any:
    if use_torch:
        positive = value > 0
        safe_value = torch.where(positive, value, torch.ones_like(value))
        negative_infinity = torch.full_like(value, -torch.inf)
        return torch.where(positive, torch.log(safe_value), negative_infinity)

    result = np.full_like(value, -np.inf, dtype=np.float64)
    np.log(value, out=result, where=value > 0)
    return result


def _normalise_log_weights(log_weights: Any, *, use_torch: bool) -> Any:
    if use_torch:
        if not bool(torch.isfinite(log_weights).any().item()):
            raise ValueError("at least one log weight must be finite")
        return torch.softmax(log_weights, dim=0)

    maximum = float(np.max(log_weights))
    if not math.isfinite(maximum):
        raise ValueError("at least one log weight must be finite")
    weights = np.exp(log_weights - maximum)
    return weights / weights.sum()


def boltzmann_projection(base_probs: Any, rewards: Any, beta: float) -> Any:
    """Return the KL-regularized Boltzmann projection of a discrete policy."""

    beta_value = _validate_beta(beta)
    use_torch = _uses_torch(base_probs, rewards)
    base = _validate_probability_vector(base_probs, "base_probs", use_torch=use_torch)
    reward = _validate_vector(rewards, "rewards", use_torch=use_torch)
    _validate_same_shape(base, reward, "base_probs", "rewards")

    if use_torch:
        if bool(((reward < 0) | (reward > 1)).any().item()):
            raise ValueError("rewards must lie in the interval [0, 1]")
        support = base > 0
        reference = torch.max(reward[support])
        scaled_reward = (reward - reference) / beta_value
    else:
        if bool(np.any((reward < 0) | (reward > 1))):
            raise ValueError("rewards must lie in the interval [0, 1]")
        support = base > 0
        reference = np.max(reward[support])
        with np.errstate(over="ignore", under="ignore"):
            scaled_reward = (reward - reference) / beta_value

    log_weights = _safe_log_nonnegative(base, use_torch=use_torch) + scaled_reward
    return _normalise_log_weights(log_weights, use_torch=use_torch)


def reward_moment_multiplier(cond_reward_samples: Any, beta: float) -> Any:
    """Average ``exp(reward / beta)`` over each final sample dimension.

    If the unscaled moments exceed the backend's finite range, a single common
    positive factor is removed.  This leaves every downstream selection law
    unchanged while keeping the returned relative multipliers finite.
    """

    beta_value = _validate_beta(beta)
    use_torch = _uses_torch(cond_reward_samples)
    rewards = _as_float_array(cond_reward_samples, "cond_reward_samples", use_torch=use_torch)
    size = rewards.numel() if use_torch else rewards.size
    if rewards.ndim == 0 or size == 0 or rewards.shape[-1] == 0:
        raise ValueError("cond_reward_samples must contain at least one reward sample")
    if not _all_finite(rewards, use_torch=use_torch):
        raise ValueError("cond_reward_samples must contain only finite values")
    if use_torch:
        if bool(((rewards < 0) | (rewards > 1)).any().item()):
            raise ValueError("conditional rewards must lie in the interval [0, 1]")
        reference = torch.max(rewards)
        shifted = (rewards - reference) / beta_value
        relative_log_moments = torch.log(torch.mean(torch.exp(shifted), dim=-1))
        offset = reference / beta_value
        log_moments = relative_log_moments + offset
        log_limit = math.log(torch.finfo(rewards.dtype).max)
        if bool((torch.max(log_moments) <= log_limit).item()):
            return torch.exp(log_moments)
        relative_log_moments = relative_log_moments - torch.max(relative_log_moments)
        log_floor = math.log(torch.finfo(rewards.dtype).tiny)
        return torch.exp(torch.clamp(relative_log_moments, min=log_floor))

    if bool(np.any((rewards < 0) | (rewards > 1))):
        raise ValueError("conditional rewards must lie in the interval [0, 1]")
    reference = float(np.max(rewards))
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        shifted = (rewards - reference) / beta_value
        relative_log_moments = np.log(np.mean(np.exp(shifted), axis=-1))
        log_moments = relative_log_moments + reference / beta_value
    log_limit = math.log(np.finfo(np.float64).max)
    if bool(np.max(log_moments) <= log_limit):
        return np.exp(log_moments)
    relative_log_moments = relative_log_moments - np.max(relative_log_moments)
    log_floor = math.log(np.finfo(np.float64).tiny)
    return np.exp(np.maximum(relative_log_moments, log_floor))


def selected_error_distribution(mu0: Any, multiplier: Any) -> Any:
    """Reweight an error distribution by positive reward moment multipliers."""

    use_torch = _uses_torch(mu0, multiplier)
    base = _validate_probability_vector(mu0, "mu0", use_torch=use_torch)
    moments = _validate_positive_vector(multiplier, "multiplier", use_torch=use_torch)
    _validate_same_shape(base, moments, "mu0", "multiplier")
    log_weights = _safe_log_nonnegative(base, use_torch=use_torch)
    log_weights = log_weights + _safe_log_nonnegative(moments, use_torch=use_torch)
    return _normalise_log_weights(log_weights, use_torch=use_torch)


def expectation_shift(mu0: Any, values: Any, multiplier: Any) -> Any:
    """Return the exact selected-minus-base expectation shift."""

    use_torch = _uses_torch(mu0, values, multiplier)
    base = _validate_probability_vector(mu0, "mu0", use_torch=use_torch)
    statistic = _validate_vector(values, "values", use_torch=use_torch)
    moments = _validate_positive_vector(multiplier, "multiplier", use_torch=use_torch)
    _validate_same_shape(base, statistic, "mu0", "values")
    _validate_same_shape(base, moments, "mu0", "multiplier")
    selected = selected_error_distribution(base, moments)
    if use_torch:
        base_mean = torch.sum(base * statistic)
        return torch.sum((selected - base) * (statistic - base_mean))
    base_mean = np.sum(base * statistic)
    return np.sum((selected - base) * (statistic - base_mean))


def binary_compensability_multiplier(compensability: Any, beta: float) -> Any:
    """Return ``1 + (exp(1 / beta) - 1) * compensability`` stably."""

    beta_value = _validate_beta(beta)
    use_torch = _uses_torch(compensability)
    probability = _validate_vector(compensability, "compensability", use_torch=use_torch)
    if use_torch:
        if bool(((probability < 0) | (probability > 1)).any().item()):
            raise ValueError("compensability must lie in the interval [0, 1]")
        if not bool((probability > 0).any().item()):
            return torch.ones_like(probability)
        inverse_beta = 1.0 / beta_value
        log_limit = math.log(torch.finfo(probability.dtype).max)
        if inverse_beta <= log_limit:
            return 1.0 + torch.expm1(probability.new_tensor(inverse_beta)) * probability
        scaled = probability + (1.0 - probability) * math.exp(-inverse_beta)
        return torch.clamp(scaled, min=torch.finfo(probability.dtype).tiny)

    if bool(np.any((probability < 0) | (probability > 1))):
        raise ValueError("compensability must lie in the interval [0, 1]")
    if not bool(np.any(probability > 0)):
        return np.ones_like(probability)
    inverse_beta = 1.0 / beta_value
    log_limit = math.log(np.finfo(np.float64).max)
    if inverse_beta <= log_limit:
        return 1.0 + np.expm1(inverse_beta) * probability
    scaled = probability + (1.0 - probability) * math.exp(-inverse_beta)
    return np.maximum(scaled, np.finfo(np.float64).tiny)


__all__ = [
    "binary_compensability_multiplier",
    "boltzmann_projection",
    "expectation_shift",
    "reward_moment_multiplier",
    "selected_error_distribution",
]
