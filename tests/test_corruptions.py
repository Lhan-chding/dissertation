"""Executable error catalogs must be validated, reversible, and non-mutating."""

import pytest

from compbias.envs.cva_world.corruptions import (
    apply_error,
    reverse_error,
    validate_error_spec,
)
from compbias.envs.cva_world.schema import ErrorSpec


def _error(error_id: str, family: str, severity: float, parameters: dict[str, object]) -> ErrorSpec:
    return ErrorSpec.from_mapping(
        {
            "error_id": error_id,
            "family": family,
            "severity": severity,
            "parameters": parameters,
        }
    )


@pytest.mark.parametrize(
    ("scene", "error", "expected"),
    [
        (
            {"value": 7},
            _error(
                "numeric_offset:+2",
                "numeric_offset",
                2,
                {"field": "value", "delta": 2},
            ),
            {"value": 9},
        ),
        (
            {"count": 5},
            _error("omission:1", "omission", 1, {"field": "count", "amount": 1}),
            {"count": 4},
        ),
        (
            {"count": 5},
            _error(
                "duplication:2",
                "duplication",
                2,
                {"field": "count", "amount": 2},
            ),
            {"count": 7},
        ),
        (
            {"reading": 2.5},
            _error(
                "gauge:+0.125",
                "numeric_offset",
                0.125,
                {"field": "reading", "delta": 0.125},
            ),
            {"reading": 2.625},
        ),
        (
            {"bars": [2, 5, 3]},
            _error(
                "bar:1:-1",
                "local_offset",
                1,
                {"field": "bars", "index": 1, "delta": -1},
            ),
            {"bars": [2, 4, 3]},
        ),
    ],
)
def test_numeric_corruptions_round_trip_without_mutating_scene(
    scene: dict[str, object], error: ErrorSpec, expected: dict[str, object]
) -> None:
    original = repr(scene)

    perceived = apply_error(scene, error)
    recovered = reverse_error(perceived, error)

    assert perceived == expected
    if "reading" in scene:
        assert recovered["reading"] == pytest.approx(scene["reading"], abs=1e-12)
    else:
        assert recovered == scene
    assert perceived is not scene
    assert repr(scene) == original


def test_truth_returns_an_equal_but_distinct_immutable_state() -> None:
    scene = {"value": 7}
    truth = _error("truth", "truth", 0, {})

    perceived = apply_error(scene, truth)

    assert perceived == scene
    assert perceived is not scene
    with pytest.raises(TypeError):
        perceived["value"] = 8  # type: ignore[index]


@pytest.mark.parametrize(
    ("relation", "flipped"),
    [
        ("left_of", "right_of"),
        ("right_of", "left_of"),
        ("above", "below"),
        ("below", "above"),
        ("parallel", "intersect"),
        ("intersect", "parallel"),
    ],
)
def test_relation_flip_is_an_involution(relation: str, flipped: str) -> None:
    error = _error(
        "relation_flip",
        "relation_flip",
        1,
        {
            "field": "relation",
            "pairs": {
                "left_of": "right_of",
                "right_of": "left_of",
                "above": "below",
                "below": "above",
                "parallel": "intersect",
                "intersect": "parallel",
            },
        },
    )
    scene = {"relation": relation}

    once = apply_error(scene, error)
    twice = apply_error(once, error)

    assert once["relation"] == flipped
    assert twice == scene
    assert reverse_error(once, error) == scene


@pytest.mark.parametrize(
    "error",
    [
        _error("numeric_offset:+2", "numeric_offset", 1, {"field": "value", "delta": 2}),
        _error("omission:-1", "omission", 1, {"field": "count", "amount": -1}),
        _error("truth", "truth", 0, {"delta": 1}),
    ],
)
def test_invalid_error_parameters_or_semantic_severity_are_rejected(error: ErrorSpec) -> None:
    with pytest.raises(ValueError):
        validate_error_spec(error)
