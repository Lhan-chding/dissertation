"""Finite contracts for sparse-syndrome correction identifiability."""

from __future__ import annotations

import itertools

import pytest
from compensability_v5.theory.sparse_syndrome import (
    enumerate_sparse_errors,
    is_sparse_correction_unique,
    residual_signature,
)

THEORY_TOLERANCE = 1e-8


def _matvec(matrix: tuple[tuple[int, ...], ...], vector: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sum(coefficient * value for coefficient, value in zip(row, vector, strict=True))
        for row in matrix
    )


def test_sparse_error_enumeration_is_exhaustive_unique_and_deterministic() -> None:
    errors = enumerate_sparse_errors(dimension=3, magnitudes=(-2, -1, 1, 2), max_sparsity=1)

    expected = {(0, 0, 0)}
    expected.update(
        tuple(magnitude if index == coordinate else 0 for index in range(3))
        for coordinate in range(3)
        for magnitude in (-2, -1, 1, 2)
    )
    assert tuple(errors) == tuple(sorted(expected))
    assert len(errors) == 13


def test_sparse_uniqueness_matches_exhaustive_residual_collision_theorem() -> None:
    matrix = ((1, 0, 1), (0, 1, 1))
    errors = enumerate_sparse_errors(dimension=3, magnitudes=(-2, -1, 1, 2), max_sparsity=1)
    signatures = [residual_signature(matrix, error) for error in errors]

    assert signatures == [_matvec(matrix, error) for error in errors]
    assert len(signatures) == len(set(signatures))
    assert is_sparse_correction_unique(matrix, errors) is True

    for left, right in itertools.combinations(errors, 2):
        difference = tuple(a - b for a, b in zip(left, right, strict=True))
        kernel_norm = max(abs(value) for value in _matvec(matrix, difference))
        assert kernel_norm > THEORY_TOLERANCE


def test_scaled_column_collision_is_a_nonunique_correction_witness() -> None:
    matrix = ((1, 2, 0),)
    errors = enumerate_sparse_errors(dimension=3, magnitudes=(1, 2), max_sparsity=1)
    first = (2, 0, 0)
    second = (0, 1, 0)

    assert residual_signature(matrix, first) == residual_signature(matrix, second) == (2,)
    assert is_sparse_correction_unique(matrix, errors) is False


@pytest.mark.parametrize("max_sparsity", [-1, 4, True])
def test_sparse_error_enumeration_rejects_invalid_sparsity(max_sparsity: object) -> None:
    with pytest.raises((TypeError, ValueError), match="sparsity"):
        enumerate_sparse_errors(3, (-1, 1), max_sparsity)  # type: ignore[arg-type]
