"""Joint covariance in candidate-major coordinates, with explicit oracle moments."""

from __future__ import annotations

import math

import numpy as np


def probability(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Finite nonnegative probability vector required")
    if not np.isclose(p.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("Probabilities must sum to one")
    return p


def multinomial_covariance(p, n):
    p = probability(p)
    if type(n) is not int or n <= 0:
        raise ValueError("Positive integer sample count required")
    return (np.diag(p) - np.outer(p, p)) / n


def counts_covariance(counts, *, prior=0.5):
    counts = np.asarray(counts)
    if (
        counts.shape != (4,)
        or counts.dtype.kind not in "iu"
        or (counts < 0).any()
        or counts.sum() <= 0
    ):
        raise ValueError("Four nonnegative integer counts with positive total required")
    if not math.isfinite(prior) or prior < 0:
        raise ValueError("Finite nonnegative covariance-only prior required")
    p = (counts + prior) / (counts.sum() + 4 * prior)
    return multinomial_covariance(p, int(counts.sum()))


def contrast_covariance(level_covariance, incidence, coordinates=1):
    covariance, incidence = map(
        lambda v: np.asarray(v, dtype=np.float64), (level_covariance, incidence)
    )
    if (
        incidence.ndim != 2
        or not np.isfinite(incidence).all()
        or not np.allclose(incidence.sum(1), 0, atol=1e-14)
    ):
        raise ValueError("Finite zero-sum incidence rows required")
    transform = np.kron(incidence, np.eye(coordinates))
    if (
        covariance.shape != (transform.shape[1], transform.shape[1])
        or not np.isfinite(covariance).all()
    ):
        raise ValueError("Aligned finite level covariance required")
    return transform @ covariance @ transform.T


def covariance_of_mean(contributions):
    contributions = np.asarray(contributions, dtype=np.float64)
    if contributions.ndim != 2 or len(contributions) < 2 or not np.isfinite(contributions).all():
        raise ValueError("At least two finite joint contribution rows required")
    centered = contributions - contributions.mean(0)
    return centered.T @ centered / (len(contributions) * (len(contributions) - 1))


def shrink_covariance(covariance, eta=0.01, *, observation_scale=None, n=None, floor_ratio=1e-8):
    covariance = np.asarray(covariance, dtype=np.float64)
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not covariance.size
        or not np.isfinite(covariance).all()
    ):
        raise ValueError("Finite square nonempty covariance required")
    if not 0 < eta <= 1 or not math.isfinite(floor_ratio) or floor_ratio <= 0:
        raise ValueError("Positive shrinkage and spectral floor required")
    covariance = (covariance + covariance.T) / 2
    eigen = np.linalg.eigvalsh(covariance)
    empirical_scale = float(np.trace(covariance) / len(covariance))
    if eigen.min() < -1e-10 * max(1.0, abs(eigen).max()):
        raise ValueError("Covariance is not positive semidefinite")
    # 1/n^2 is a measurement-scale regularizer, not a confidence guarantee.
    fallback = 1 / float(n) ** 2 if n is not None and type(n) is int and n > 0 else None
    scale = (
        observation_scale
        if observation_scale is not None
        else max(empirical_scale, fallback or 0.0)
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Positive observation scale or sample count required for empirical zero covariance"
        )
    diagonal = np.diag(np.maximum(np.diag(covariance), eta * scale))
    shrunk = (1 - eta) * covariance + eta * diagonal
    values, vectors = np.linalg.eigh(shrunk)
    floor = max(scale * floor_ratio, np.finfo(np.float64).tiny)
    result = (vectors * np.maximum(values, floor)) @ vectors.T
    return result, {
        "status": "EMPIRICAL_DEGENERATE" if empirical_scale <= 0 else "REGULARIZED",
        "eta": eta,
        "observation_scale": scale,
        "spectral_floor": floor,
        "empirical_rank": int(np.count_nonzero(eigen > max(eigen.max() * 1e-10, 1e-14))),
        "regularized_rank_is_not_information_rank": True,
    }


def weighted_moments(values, distribution):
    p = probability(distribution)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != len(p) or not np.isfinite(values).all():
        raise ValueError("Finite contribution matrix aligned to support required")
    mean = p @ values
    centered = values - mean
    return mean, (centered.T * p) @ centered


def lr_moments(p0, p1, rho, features):
    p0, p1, rho = map(probability, (p0, p1, rho))
    if p0.shape != p1.shape or p0.shape != rho.shape:
        raise ValueError("Policy support shape mismatch")
    difference = p1 - p0
    if ((rho == 0) & (difference != 0)).any():
        raise ValueError("BLOCKED_SUPPORT_MISMATCH")
    weights = np.divide(difference, rho, out=np.zeros_like(difference), where=rho > 0)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    mean, covariance = weighted_moments(weights[:, None] * features, rho)
    return mean, covariance, weights


def inverse_cdf_joint(p0, p1):
    p0, p1 = map(probability, (p0, p1))
    if p0.shape != p1.shape:
        raise ValueError("Identical action support required")
    a, b = np.r_[0, np.cumsum(p0)], np.r_[0, np.cumsum(p1)]
    a[-1] = b[-1] = 1.0
    return np.maximum(
        0, np.minimum(a[1:, None], b[None, 1:]) - np.maximum(a[:-1, None], b[None, :-1])
    )


def stable_exp_difference(loga, logb):
    a, b = np.broadcast_arrays(np.asarray(loga, float), np.asarray(logb, float))
    if np.any(np.isnan(a) | np.isnan(b) | np.isposinf(a) | np.isposinf(b)):
        raise ValueError("NaN/positive infinity in log weights")
    maximum = np.maximum(a, b)
    if np.any(maximum > math.log(np.finfo(float).max)):
        raise FloatingPointError("NONFINITE_WEIGHT; no clipping permitted")
    with np.errstate(invalid="ignore"):
        difference = a - b
        result = np.sign(difference) * np.exp(maximum) * (-np.expm1(-np.abs(difference)))
    result = np.where(np.isneginf(maximum), 0.0, result)
    if not np.isfinite(result).all():
        raise FloatingPointError("NONFINITE_WEIGHT")
    return result
