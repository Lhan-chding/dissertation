"""The canonical solver is the sole, pure source of task answers."""

from dataclasses import replace

import pytest

from compbias.envs.cva_world.canonical_solver import (
    canonical_reasoning,
    solve,
    solve_sample,
)
from compbias.envs.cva_world.schema import CVASample, TaskFamily


@pytest.mark.parametrize(
    ("family", "scene", "question", "expected_answer", "expected_reasoning"),
    [
        (
            TaskFamily.DIGIT_OFFSET,
            {"value": 7},
            {"template": "add_constant", "operand": 3},
            10,
            {"operation": "add", "operand": 3},
        ),
        (
            TaskFamily.COUNT_TRANSFORM,
            {"count": 4},
            {"template": "affine_transform", "scale": 3, "offset": -1},
            11,
            {"operation": "affine", "scale": 3, "offset": -1},
        ),
        (
            TaskFamily.GAUGE_CALIBRATION,
            {"reading": 2.5},
            {"template": "calibrate", "scale": 2.0, "offset": 1.0},
            6.0,
            {"operation": "affine", "scale": 2.0, "offset": 1.0},
        ),
        (
            TaskFamily.BAR_CHART_AGGREGATE,
            {"bars": [2, 5, 3, 6]},
            {"template": "aggregate", "operation": "sum", "indices": [0, 1]},
            7,
            {"operation": "sum", "indices": [0, 1]},
        ),
        (
            TaskFamily.BAR_CHART_AGGREGATE,
            {"bars": [11, 4, 9, 12]},
            {"template": "aggregate", "operation": "difference", "indices": [0, 1]},
            7,
            {"operation": "difference", "indices": [0, 1]},
        ),
        (
            TaskFamily.BAR_CHART_AGGREGATE,
            {"bars": [21, 3, 8, 22]},
            {"template": "aggregate", "operation": "ratio", "indices": [0, 1]},
            7.0,
            {"operation": "ratio", "indices": [0, 1]},
        ),
        (
            TaskFamily.RELATION_RULE,
            {"relation": "left_of"},
            {
                "template": "relation_lookup",
                "rule": {"left_of": "near", "right_of": "far"},
            },
            "near",
            {"operation": "lookup", "relation": "left_of", "result": "near"},
        ),
    ],
)
def test_solver_covers_every_task_family(
    family: TaskFamily,
    scene: dict[str, object],
    question: dict[str, object],
    expected_answer: object,
    expected_reasoning: dict[str, object],
) -> None:
    scene_before = repr(scene)
    question_before = repr(question)

    assert solve(scene, question, family) == expected_answer
    assert canonical_reasoning(scene, question, family) == expected_reasoning
    assert repr(scene) == scene_before
    assert repr(question) == question_before


def _digit_sample() -> CVASample:
    return CVASample.from_mapping(
        {
            "sample_id": "digit_offset_train_000001",
            "image_path": "images/one.png",
            "task_family": "digit_offset",
            "scene": {"value": 7},
            "question": {"template": "add_constant", "operand": 3},
            "canonical_answer": 10,
            "canonical_reasoning": {"operation": "add", "operand": 3},
            "error_catalog": [
                {
                    "error_id": "truth",
                    "family": "truth",
                    "severity": 0.0,
                    "parameters": {},
                }
            ],
            "split_keys": {
                "semantic_split": "train",
                "visual_style": "font_a",
                "error_mechanism": "standard",
            },
        }
    )


def test_solve_sample_self_checks_stored_answer_and_reasoning() -> None:
    sample = _digit_sample()
    result = solve_sample(sample)

    assert result.answer == sample.canonical_answer
    assert result.reasoning == sample.canonical_reasoning
    assert result.is_consistent is True


def test_solve_sample_rejects_stale_generated_labels() -> None:
    sample = replace(_digit_sample(), canonical_answer=999)

    with pytest.raises(ValueError, match="canonical_answer"):
        solve_sample(sample)


@pytest.mark.parametrize(
    ("family", "scene", "question"),
    [
        (TaskFamily.DIGIT_OFFSET, {}, {"template": "add_constant", "operand": 1}),
        (TaskFamily.COUNT_TRANSFORM, {"count": -1}, {"template": "affine_transform"}),
        (TaskFamily.BAR_CHART_AGGREGATE, {"bars": []}, {"operation": "sum"}),
        (TaskFamily.RELATION_RULE, {"relation": "unknown"}, {"rule": {}}),
    ],
)
def test_solver_fails_explicitly_on_invalid_task_inputs(
    family: TaskFamily, scene: dict[str, object], question: dict[str, object]
) -> None:
    with pytest.raises((TypeError, ValueError, KeyError)):
        solve(scene, question, family)


@pytest.mark.parametrize("operation", ["mean", "max", "min", "unknown"])
def test_bar_solver_rejects_unregistered_operations(operation: str) -> None:
    with pytest.raises(ValueError, match="unsupported aggregate operation"):
        solve(
            {"bars": [8, 2, 5, 9]},
            {"template": "aggregate", "operation": operation, "indices": [0, 1]},
            TaskFamily.BAR_CHART_AGGREGATE,
        )


def test_bar_ratio_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="denominator"):
        solve(
            {"bars": [8, 0, 5, 9]},
            {"template": "aggregate", "operation": "ratio", "indices": [0, 1]},
            TaskFamily.BAR_CHART_AGGREGATE,
        )
