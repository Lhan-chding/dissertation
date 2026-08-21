"""Exact finite checks for sparse-syndrome correction identifiability."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from numbers import Integral
from typing import TypeAlias

IntegerVector: TypeAlias = tuple[int, ...]
IntegerMatrix: TypeAlias = tuple[IntegerVector, ...]


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _items(values: object, *, name: str) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} must be an iterable")
    return tuple(values)


def _vector(values: object, *, name: str, allow_empty: bool = False) -> IntegerVector:
    raw = _items(values, name=name)
    if not raw and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    return tuple(_integer(value, name=f"{name}[{index}]") for index, value in enumerate(raw))


def _matrix(values: object, *, dimension: int | None = None) -> IntegerMatrix:
    rows = _items(values, name="matrix")
    if not rows:
        raise ValueError("matrix must contain at least one row")
    normalized = tuple(_vector(row, name=f"matrix[{index}]") for index, row in enumerate(rows))
    width = len(normalized[0]) if dimension is None else dimension
    if width <= 0 or any(len(row) != width for row in normalized):
        raise ValueError("matrix must be rectangular with the error dimension as its width")
    return normalized


def enumerate_sparse_errors(
    dimension: int,
    magnitudes: Iterable[int],
    max_sparsity: int,
) -> tuple[IntegerVector, ...]:
    """Enumerate all admissible errors in deterministic lexicographic order.

    The zero vector is included once.  ``magnitudes`` denotes non-zero integer
    coordinate errors; duplicate magnitudes are harmless and are canonicalized.
    """

    dimension_value = _integer(dimension, name="dimension")
    sparsity_value = _integer(max_sparsity, name="max_sparsity")
    if dimension_value <= 0:
        raise ValueError("dimension must be positive")
    if not 0 <= sparsity_value <= dimension_value:
        raise ValueError("max_sparsity must lie between zero and dimension")

    raw_magnitudes = _items(magnitudes, name="magnitudes")
    if not raw_magnitudes:
        raise ValueError("magnitudes must be non-empty")
    canonical_magnitudes = tuple(
        sorted(
            {
                _integer(value, name=f"magnitudes[{index}]")
                for index, value in enumerate(raw_magnitudes)
            }
        )
    )
    if 0 in canonical_magnitudes:
        raise ValueError("magnitudes must contain only non-zero errors")

    errors: set[IntegerVector] = {(0,) * dimension_value}
    for support_size in range(1, sparsity_value + 1):
        for support in itertools.combinations(range(dimension_value), support_size):
            for values in itertools.product(canonical_magnitudes, repeat=support_size):
                error = [0] * dimension_value
                for coordinate, magnitude in zip(support, values, strict=True):
                    error[coordinate] = magnitude
                errors.add(tuple(error))
    return tuple(sorted(errors))


def residual_signature(matrix: Iterable[Iterable[int]], error: Iterable[int]) -> IntegerVector:
    """Return the exact linear signature ``A error`` for an integer matrix."""

    normalized_error = _vector(error, name="error")
    normalized_matrix = _matrix(matrix, dimension=len(normalized_error))
    return tuple(
        sum(coefficient * value for coefficient, value in zip(row, normalized_error, strict=True))
        for row in normalized_matrix
    )


def is_sparse_correction_unique(
    matrix: Iterable[Iterable[int]],
    errors: Iterable[Iterable[int]],
) -> bool:
    """Return whether every admissible error has a distinct residual signature."""

    raw_errors = _items(errors, name="errors")
    if not raw_errors:
        raise ValueError("errors must be non-empty")
    normalized_errors = tuple(
        _vector(error, name=f"errors[{index}]") for index, error in enumerate(raw_errors)
    )
    dimension = len(normalized_errors[0])
    if any(len(error) != dimension for error in normalized_errors):
        raise ValueError("all errors must have the same dimension")
    if len(set(normalized_errors)) != len(normalized_errors):
        raise ValueError("errors must not contain duplicates")

    normalized_matrix = _matrix(matrix, dimension=dimension)
    signatures = {
        tuple(
            sum(coefficient * value for coefficient, value in zip(row, error, strict=True))
            for row in normalized_matrix
        )
        for error in normalized_errors
    }
    return len(signatures) == len(normalized_errors)


__all__ = ["enumerate_sparse_errors", "is_sparse_correction_unique", "residual_signature"]
