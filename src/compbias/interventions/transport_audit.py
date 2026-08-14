"""Behavior-signature diagnostics for synthetic mediator transport."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np


def _matrix(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a non-empty finite two-dimensional array")
    return np.array(matrix, copy=True)


def _rewards(value: object, name: str, count: int) -> np.ndarray:
    rewards = np.asarray(value, dtype=np.float64)
    if rewards.shape != (count,) or not np.all(np.isfinite(rewards)):
        raise ValueError(f"{name} must have one finite value per signature")
    if np.any((rewards < 0.0) | (rewards > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return np.array(rewards, copy=True)


@dataclass(frozen=True, slots=True)
class SyntheticTransportReport:
    error_support_overlap: float
    two_sample_accuracy: float
    mean_nearest_task_distance: float
    reward_gap: float
    reward_gap_ci_low: float
    reward_gap_ci_high: float
    off_support_stress_test: bool


def _cross_validated_centroid_accuracy(
    natural: np.ndarray,
    synthetic: np.ndarray,
    *,
    seed: int,
) -> float:
    labels = np.concatenate(
        (np.zeros(len(natural), dtype=np.int8), np.ones(len(synthetic), dtype=np.int8))
    )
    values = np.vstack((natural, synthetic))
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(len(values), dtype=np.int8)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        fold_ids[indices[rng.permutation(len(indices))]] = np.arange(len(indices)) % 5
    predictions = np.empty_like(labels)
    for fold in range(5):
        test = fold_ids == fold
        train = ~test
        natural_center = values[train & (labels == 0)].mean(axis=0)
        synthetic_center = values[train & (labels == 1)].mean(axis=0)
        natural_distance = np.sum((values[test] - natural_center) ** 2, axis=1)
        synthetic_distance = np.sum((values[test] - synthetic_center) ** 2, axis=1)
        predictions[test] = (synthetic_distance < natural_distance).astype(np.int8)
    raw_accuracy = float(np.mean(predictions == labels))
    return max(raw_accuracy, 1.0 - raw_accuracy)


def _nearest_distance_mean(natural: np.ndarray, synthetic: np.ndarray) -> float:
    nearest: list[np.ndarray] = []
    for start in range(0, len(synthetic), 128):
        batch = synthetic[start : start + 128]
        distances = np.sqrt(np.sum((batch[:, None, :] - natural[None, :, :]) ** 2, axis=2))
        nearest.append(np.min(distances, axis=1))
    return float(np.concatenate(nearest).mean())


def _bootstrap_means(
    values: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    unique, counts = np.unique(values, return_counts=True)
    if len(unique) <= 256:
        sampled_counts = rng.multinomial(
            len(values),
            counts.astype(np.float64) / len(values),
            size=draws,
        )
        return sampled_counts @ unique / len(values)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 128):
        stop = min(start + 128, draws)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return means


def audit_synthetic_transport(
    *,
    natural_signatures: object,
    synthetic_signatures: object,
    natural_rewards: object,
    synthetic_rewards: object,
    natural_error_types: tuple[str, ...],
    synthetic_error_types: tuple[str, ...],
    bootstrap_draws: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    classifier_threshold: float = 0.75,
) -> SyntheticTransportReport:
    """Audit support, behavioral separability, task distance, and reward transport."""

    natural = _matrix(natural_signatures, "natural_signatures")
    synthetic = _matrix(synthetic_signatures, "synthetic_signatures")
    if natural.shape[1] != synthetic.shape[1] or min(len(natural), len(synthetic)) < 5:
        raise ValueError("signature groups need matching dimensions and at least five rows each")
    natural_reward = _rewards(natural_rewards, "natural_rewards", len(natural))
    synthetic_reward = _rewards(synthetic_rewards, "synthetic_rewards", len(synthetic))
    if len(natural_error_types) != len(natural) or len(synthetic_error_types) != len(synthetic):
        raise ValueError("error type labels must align with signature rows")
    natural_support = set(natural_error_types)
    synthetic_support = set(synthetic_error_types)
    if not natural_support or not synthetic_support or "" in natural_support | synthetic_support:
        raise ValueError("error type labels must be non-empty strings")
    union = natural_support | synthetic_support
    overlap = len(natural_support & synthetic_support) / len(union)

    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise TypeError("bootstrap_draws must be an integer")
    if not 1_000 <= bootstrap_draws <= 1_000_000:
        raise ValueError("bootstrap_draws must be between 1000 and 1000000")
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise TypeError("confidence must be numeric")
    confidence_value = float(confidence)
    if not 0.5 < confidence_value < 1.0:
        raise ValueError("confidence must lie strictly between 0.5 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0.5 <= classifier_threshold <= 1.0:
        raise ValueError("classifier_threshold must lie in [0.5, 1]")

    accuracy = _cross_validated_centroid_accuracy(natural, synthetic, seed=seed)
    mean_nearest_distance = _nearest_distance_mean(natural, synthetic)
    reward_gap = float(synthetic_reward.mean() - natural_reward.mean())
    rng = np.random.default_rng(seed)
    natural_means = _bootstrap_means(natural_reward, draws=bootstrap_draws, rng=rng)
    synthetic_means = _bootstrap_means(synthetic_reward, draws=bootstrap_draws, rng=rng)
    draws = synthetic_means - natural_means
    tail = (1.0 - confidence_value) / 2.0
    low, high = np.quantile(draws, (tail, 1.0 - tail), method="linear")
    return SyntheticTransportReport(
        error_support_overlap=float(overlap),
        two_sample_accuracy=accuracy,
        mean_nearest_task_distance=mean_nearest_distance,
        reward_gap=reward_gap,
        reward_gap_ci_low=float(low),
        reward_gap_ci_high=float(high),
        off_support_stress_test=overlap < 1.0 or accuracy > classifier_threshold,
    )
