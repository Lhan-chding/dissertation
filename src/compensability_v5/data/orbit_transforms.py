"""Deterministic structural transformations used by the Phase 2a orbit freeze."""

from __future__ import annotations

from .constraint_basis import Matrix, Vector, elementary_equivalent_basis


def permute_system(
    world: tuple[int, int, int, int],
    matrix: Matrix,
    targets: Vector,
    permutation: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], Matrix, Vector]:
    if tuple(sorted(permutation)) != (0, 1, 2, 3):
        raise ValueError("variable permutation must be a permutation of 0..3")
    transformed_world = tuple(world[index] for index in permutation)
    transformed_matrix = tuple(tuple(row[index] for index in permutation) for row in matrix)
    return transformed_world, transformed_matrix, targets


def inverse_permutation(permutation: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if tuple(sorted(permutation)) != (0, 1, 2, 3):
        raise ValueError("variable permutation must be a permutation of 0..3")
    return tuple(permutation.index(index) for index in range(4))  # type: ignore[return-value]


def transform_linear_system(
    *,
    world: tuple[int, int, int, int],
    matrix: Matrix,
    targets: Vector,
    graph_axis: str,
) -> tuple[tuple[int, int, int, int], Matrix, Vector, dict[str, object]]:
    """Return one registered orbit member and its explicit inverse metadata."""

    if graph_axis == "familiar":
        return world, matrix, targets, {"kind": "identity", "inverse": {"kind": "identity"}}
    if graph_axis == "variable_permuted":
        permutation = (1, 2, 3, 0)
        transformed = permute_system(world, matrix, targets, permutation)
        metadata = {
            "kind": graph_axis,
            "permutation": list(permutation),
            "inverse": {"permutation": list(inverse_permutation(permutation))},
        }
        return (*transformed, metadata)
    if graph_axis == "fact_order_permuted":
        order = tuple(reversed(range(len(matrix))))
        return (
            world,
            tuple(matrix[index] for index in order),
            tuple(targets[index] for index in order),
            {"kind": graph_axis, "order": list(order), "inverse": {"order": list(order)}},
        )
    if graph_axis == "equivalent_basis":
        transformed_matrix, transformed_targets = elementary_equivalent_basis(matrix, targets)
        metadata = {
            "kind": graph_axis,
            "row_operation": "row0_plus_row1",
            "inverse": {"row_operation": "row0_minus_row1"},
        }
        return world, transformed_matrix, transformed_targets, metadata
    if graph_axis == "sparse_mixed_ood":
        mixed_matrix: Matrix = ((1, 0, 0, 0), matrix[1], matrix[2], matrix[-1])
        mixed_targets = (world[0], targets[1], targets[2], targets[-1])
        metadata = {
            "kind": graph_axis,
            "construction": "unary_plus_relational",
            "inverse": {"kind": "identity_on_world"},
        }
        return world, mixed_matrix, mixed_targets, metadata
    raise ValueError("graph axis is not registered")
