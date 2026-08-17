"""RED contracts for the T1--T6 capability chain and world-level evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from compensability_v4.diagnostics.capability_chain import (
    CapabilityRecord,
    CapabilityTask,
    CapabilityTaskType,
    evaluate_capability_record,
)
from compensability_v4.eval.counterfactual import validate_counterfactual_world
from compensability_v4.eval.statistics import aggregate_scene_metrics
from compensability_v4.eval.world_recovery import RecoveryClassification, classify_world_recovery


def _task(task_type: CapabilityTaskType, expected_output: str) -> CapabilityTask:
    return CapabilityTask(
        scene_id="scene-001",
        task_type=task_type,
        expected_output=expected_output,
    )


def test_capability_task_enum_is_exactly_t1_through_t6() -> None:
    assert {task.value for task in CapabilityTaskType} == {"T1", "T2", "T3", "T4", "T5", "T6"}


@pytest.mark.parametrize(
    ("task_type", "expected_output", "raw_output", "parsed_output"),
    [
        (CapabilityTaskType.T1, "YES", "YES", "YES"),
        (CapabilityTaskType.T2, "CONFLICT", "CONFLICT", "CONFLICT"),
        (CapabilityTaskType.T3, "0", "0", 0),
        (CapabilityTaskType.T4, "9", "9", 9),
        (CapabilityTaskType.T5, "A", "A", "A"),
        (CapabilityTaskType.T6, "9,4,5,6", "9,4,5,6", (9, 4, 5, 6)),
    ],
)
def test_capability_chain_accepts_only_the_minimal_registered_outputs(
    task_type: CapabilityTaskType,
    expected_output: str,
    raw_output: str,
    parsed_output: object,
) -> None:
    record = evaluate_capability_record(_task(task_type, expected_output), raw_output)

    assert isinstance(record, CapabilityRecord)
    assert record.parse_success is True
    assert record.is_correct is True
    assert record.parsed_output == parsed_output


@pytest.mark.parametrize(
    ("task_type", "expected_output", "raw_output"),
    [
        (CapabilityTaskType.T1, "YES", "Yes, it satisfies the fact."),
        (CapabilityTaskType.T2, "CONFLICT", "There is a CONFLICT"),
        (CapabilityTaskType.T3, "0", "index 0"),
        (CapabilityTaskType.T4, "9", "9 because 9+4=13"),
        (CapabilityTaskType.T5, "A", "Option A"),
        (CapabilityTaskType.T6, "9,4,5,6", "[9, 4, 5, 6]"),
    ],
)
def test_capability_chain_rejects_explanations_dsl_and_noncanonical_format(
    task_type: CapabilityTaskType, expected_output: str, raw_output: str
) -> None:
    record = evaluate_capability_record(_task(task_type, expected_output), raw_output)

    assert record.parse_success is False
    assert record.is_correct is False
    assert record.parsed_output is None


def test_capability_task_and_record_are_immutable() -> None:
    task = _task(CapabilityTaskType.T6, "9,4,5,6")
    record = evaluate_capability_record(task, "9,4,5,6")

    with pytest.raises(FrozenInstanceError):
        task.expected_output = "8,4,5,6"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        record.is_correct = False  # type: ignore[misc]


def test_t5_success_is_explicitly_diagnostic_not_full_recovery() -> None:
    record = evaluate_capability_record(_task(CapabilityTaskType.T5, "B"), "B")

    assert record.is_correct is True
    assert record.establishes_full_recovery is False


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ((8, 4, 5, 6), RecoveryClassification.COPY),
        ((7, 4, 5, 6), RecoveryClassification.SINGLE_EDIT),
        ((7, 3, 5, 6), RecoveryClassification.OVEREDIT),
        ((9, 4, 5, 6), RecoveryClassification.TRUE_RECOVERY),
    ],
)
def test_world_recovery_taxonomy_is_mutually_exclusive_and_complete(
    prediction: tuple[int, int, int, int], expected: RecoveryClassification
) -> None:
    classification = classify_world_recovery(
        truth=(9, 4, 5, 6),
        observed=(8, 4, 5, 6),
        prediction=prediction,
    )

    assert classification is expected
    assert classification in set(RecoveryClassification)


def test_counterfactual_world_must_be_distinct_legal_and_uniquely_fact_supported() -> None:
    facts = [
        {"type": "known_value", "index": 1, "value": 4},
        {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": 14},
    ]

    assert (
        validate_counterfactual_world(
            original_truth=(9, 4, 5, 6),
            observed=(8, 4, 5, 6),
            counterfactual_world=(10, 4, 5, 6),
            counterfactual_facts=facts,
            value_domain=range(0, 12),
        )
        is None
    )

    with pytest.raises(ValueError, match="distinct"):
        validate_counterfactual_world(
            original_truth=(9, 4, 5, 6),
            observed=(8, 4, 5, 6),
            counterfactual_world=(9, 4, 5, 6),
            counterfactual_facts=facts,
            value_domain=range(0, 12),
        )


def test_statistical_aggregation_uses_scene_not_rollout_as_the_unit() -> None:
    rows = [
        {"scene_id": "scene-a", "rollout_id": index, "success": True}
        for index in range(10)
    ] + [{"scene_id": "scene-b", "rollout_id": 0, "success": False}]

    result = aggregate_scene_metrics(rows, metric="success")

    assert result.number_of_scenes == 2
    assert result.number_of_rollouts == 11
    assert result.point_estimate == pytest.approx(0.5)
