"""Exact, model-free linear-constraint helpers for the v5 factorial freeze."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import TypeAlias

Matrix: TypeAlias = tuple[tuple[int, int, int, int], ...]
Vector: TypeAlias = tuple[int, ...]


def matrix_rank(matrix: Matrix) -> int:
    """Return the exact rational row rank of a four-column integer matrix."""

    invalid_rows = any(
        len(row) != 4 or any(type(value) is not int for value in row) for row in matrix
    )
    if not matrix or invalid_rows:
        raise ValueError("constraint matrix must contain nonempty four-integer rows")
    work = [[Fraction(value) for value in row] for row in matrix]
    rank = 0
    for column in range(4):
        pivot = next((index for index in range(rank, len(work)) if work[index][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][column]
        work[rank] = [value / scale for value in work[rank]]
        for index, row in enumerate(work):
            if index != rank and row[column]:
                factor = row[column]
                work[index] = [
                    left - factor * right
                    for left, right in zip(row, work[rank], strict=True)
                ]
        rank += 1
    return rank


def apply_matrix(matrix: Matrix, world: tuple[int, int, int, int]) -> Vector:
    if len(world) != 4 or any(type(value) is not int for value in world):
        raise ValueError("world must contain exactly four integers")
    return tuple(
        sum(coefficient * value for coefficient, value in zip(row, world, strict=True))
        for row in matrix
    )


def elementary_equivalent_basis(matrix: Matrix, targets: Vector) -> tuple[Matrix, Vector]:
    """Apply a fixed invertible row operation without changing the solution set."""

    if len(matrix) != len(targets) or len(matrix) < 2:
        raise ValueError("basis requires matching matrix and target rows")
    if matrix_rank(matrix) != 4:
        raise ValueError("basis must uniquely determine a four-value world")
    first = tuple(left + right for left, right in zip(matrix[0], matrix[1], strict=True))
    transformed = (first, *matrix[1:])
    transformed_targets = (targets[0] + targets[1], *targets[1:])
    if matrix_rank(transformed) != 4:
        raise AssertionError("invertible row operation changed rank")
    return transformed, transformed_targets


def graph_signature(matrix: Matrix) -> str:
    payload = json.dumps(matrix, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
