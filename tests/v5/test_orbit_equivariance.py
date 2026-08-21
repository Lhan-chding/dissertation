"""Finite-group tests for equivariance and the orbit-risk bound."""

from __future__ import annotations

from compensability_v5.theory.equivariance import (
    equivariance_defect,
    orbit_risk,
    permute_world,
)

THEORY_TOLERANCE = 1e-8
PERMUTATIONS = ((0, 1, 2, 3), (1, 0, 2, 3), (3, 2, 1, 0))
INSTANCES = ((1, 4, 2, 3), (7, 5, 8, 6), (2, 9, 1, 4))
TRUTHS = INSTANCES


def test_world_permutations_form_the_expected_finite_action() -> None:
    world = (3, 5, 7, 11)

    assert permute_world(world, (0, 1, 2, 3)) == world
    assert permute_world(world, (1, 0, 3, 2)) == (5, 3, 11, 7)
    assert permute_world(permute_world(world, (1, 0, 2, 3)), (1, 0, 2, 3)) == world


def test_equivariant_decoder_has_zero_defect_and_equal_base_orbit_risk() -> None:
    def decoder(instance: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(instance)

    base_risk = sum(
        decoder(instance) != truth for instance, truth in zip(INSTANCES, TRUTHS, strict=True)
    ) / len(INSTANCES)

    defect = equivariance_defect(decoder, INSTANCES, PERMUTATIONS)
    transformed_risk = orbit_risk(decoder, INSTANCES, TRUTHS, PERMUTATIONS)

    assert abs(defect) < THEORY_TOLERANCE
    assert abs(transformed_risk - base_risk) < THEORY_TOLERANCE


def test_orbit_risk_is_bounded_by_base_risk_plus_equivariance_defect() -> None:
    def decoder(instance: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sorted(instance))

    base_risk = sum(
        decoder(instance) != truth for instance, truth in zip(INSTANCES, TRUTHS, strict=True)
    ) / len(INSTANCES)
    defect = equivariance_defect(decoder, INSTANCES, PERMUTATIONS)
    transformed_risk = orbit_risk(decoder, INSTANCES, TRUTHS, PERMUTATIONS)

    assert transformed_risk <= base_risk + defect + THEORY_TOLERANCE
    assert 0.0 <= defect <= 1.0
    assert 0.0 <= transformed_risk <= 1.0
