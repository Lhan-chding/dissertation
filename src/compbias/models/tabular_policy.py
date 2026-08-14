"""Immutable finite categorical policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated_probabilities(values: ArrayLike) -> NDArray[np.float64]:
    try:
        probabilities = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError("probabilities must be a finite numeric vector") from error

    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must contain only finite values")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities cannot contain negative values")
    total = float(np.sum(probabilities, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    probabilities.setflags(write=False)
    return probabilities


def _validated_labels(labels: Any, *, size: int) -> tuple[str, ...]:
    if labels is None:
        return tuple(str(index) for index in range(size))
    if isinstance(labels, (str, bytes)):
        raise ValueError("labels must be a sequence with one unique label per action")
    try:
        copied = tuple(labels)
    except TypeError as error:
        raise ValueError("labels must be a sequence") from error
    if len(copied) != size:
        raise ValueError("labels must contain one entry per action")
    if any(not isinstance(label, str) or not label for label in copied):
        raise ValueError("labels must be non-empty strings")
    if len(set(copied)) != size:
        raise ValueError("labels must be unique")
    return copied


@dataclass(frozen=True, slots=True, init=False)
class CategoricalPolicy:
    """A categorical policy whose stored distribution cannot be mutated."""

    _probabilities: NDArray[np.float64] = field(repr=False)
    labels: tuple[str, ...]

    def __init__(self, probabilities: ArrayLike, labels: Any = None) -> None:
        copied = _validated_probabilities(probabilities)
        copied_labels = _validated_labels(labels, size=copied.size)
        object.__setattr__(self, "_probabilities", copied)
        object.__setattr__(self, "labels", copied_labels)

    @classmethod
    def from_probabilities(
        cls,
        probabilities: ArrayLike,
        labels: Any = None,
    ) -> CategoricalPolicy:
        """Create a policy after copying and validating the input vector."""

        return cls(probabilities, labels)

    @property
    def probabilities(self) -> NDArray[np.float64]:
        """Return a defensive copy, never the policy's stored array."""

        return self._probabilities.copy()

    def updated(self, logit_delta: ArrayLike) -> CategoricalPolicy:
        """Return the stable exponentiated update without changing this policy."""

        try:
            delta = np.array(logit_delta, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("logit_delta must be a finite numeric vector") from error
        if delta.shape != self._probabilities.shape:
            raise ValueError("logit_delta must have one entry per action")
        if not np.all(np.isfinite(delta)):
            raise ValueError("logit_delta must contain only finite values")

        support = self._probabilities > 0.0
        support_delta = delta[support]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            centered_delta = support_delta - np.max(support_delta)
            log_weights = np.log(self._probabilities[support]) + centered_delta
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
        normalizer = float(weights.sum(dtype=np.float64))
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            raise FloatingPointError("the logit update could not be normalized")

        probabilities = np.zeros_like(self._probabilities)
        probabilities[support] = weights / normalizer
        return type(self).from_probabilities(probabilities, self.labels)

    def sample(
        self,
        *,
        size: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Sample action indices using the caller-supplied generator."""

        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)):
            raise TypeError("size must be a positive integer")
        if int(size) <= 0:
            raise ValueError("size must be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        samples = np.asarray(
            rng.choice(self._probabilities.size, size=int(size), p=self._probabilities),
            dtype=np.int64,
        )
        samples.setflags(write=False)
        return samples


__all__ = ["CategoricalPolicy"]
