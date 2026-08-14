"""Contract tests for immutable, auditable CVA-World records."""

from dataclasses import FrozenInstanceError

import pytest

from compbias.envs.cva_world.schema import (
    CVASample,
    ErrorSpec,
    SemanticSplit,
    SplitKeys,
    TaskFamily,
)


def _sample_payload() -> dict[str, object]:
    return {
        "sample_id": "digit_offset_train_000001",
        "image_path": "images/digit_offset_train_000001.png",
        "task_family": "digit_offset",
        "scene": {"value": 7},
        "question": {
            "template": "add_constant",
            "operand": 3,
            "text": "Add 3 to the number shown in the image.",
        },
        "canonical_answer": 10,
        "canonical_reasoning": {"operation": "add", "operand": 3},
        "error_catalog": [
            {
                "error_id": "truth",
                "family": "truth",
                "severity": 0.0,
                "parameters": {},
            },
            {
                "error_id": "numeric_offset:+2",
                "family": "numeric_offset",
                "severity": 2.0,
                "parameters": {"field": "value", "delta": 2},
            },
        ],
        "split_keys": {
            "semantic_split": "train",
            "visual_style": "font_a",
            "error_mechanism": "standard",
        },
    }


def test_sample_mapping_round_trip_has_explicit_schema_and_no_aliasing() -> None:
    payload = _sample_payload()

    sample = CVASample.from_mapping(payload)

    assert sample.sample_id == "digit_offset_train_000001"
    assert sample.task_family is TaskFamily.DIGIT_OFFSET
    assert sample.split_keys.semantic_split is SemanticSplit.TRAIN
    assert isinstance(sample.split_keys, SplitKeys)
    assert all(isinstance(error, ErrorSpec) for error in sample.error_catalog)
    assert sample.to_mapping() == payload

    # Input and serialized output are detached from the frozen canonical record.
    payload["scene"]["value"] = 99  # type: ignore[index]
    serialized = sample.to_mapping()
    serialized["scene"]["value"] = -1
    assert sample.scene["value"] == 7


def test_optional_source_id_round_trips_for_explicit_ood_pairing() -> None:
    payload = _sample_payload()
    payload["sample_id"] = "digit_offset_ood_test_000001"
    payload["source_id"] = "digit_offset_iid_test_000001"
    payload["split_keys"]["semantic_split"] = "ood_test"  # type: ignore[index]

    sample = CVASample.from_mapping(payload)

    assert sample.source_id == "digit_offset_iid_test_000001"
    assert sample.to_mapping() == payload


def test_schema_is_deeply_immutable() -> None:
    sample = CVASample.from_mapping(_sample_payload())

    with pytest.raises(FrozenInstanceError):
        sample.sample_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        sample.scene["value"] = 8  # type: ignore[index]
    with pytest.raises(TypeError):
        sample.question["operand"] = 4  # type: ignore[index]
    with pytest.raises(TypeError):
        sample.error_catalog[1].parameters["delta"] = 3  # type: ignore[index]


@pytest.mark.parametrize("nested_field", ["scene", "question"])
def test_sample_rejects_unknown_nested_scene_or_question_fields(nested_field: str) -> None:
    payload = _sample_payload()
    payload[nested_field]["email"] = "alice@example.com"  # type: ignore[index]

    with pytest.raises(ValueError, match=rf"unknown {nested_field} fields.*email"):
        CVASample.from_mapping(payload)


def test_relation_question_requires_exact_complete_rule_table() -> None:
    payload = _sample_payload()
    payload.update(
        {
            "task_family": "relation_rule",
            "scene": {"relation": "left_of", "entity_pair": "pair_0_0_1"},
            "question": {
                "template": "relation_lookup",
                "rule": {"left_of": "class_0"},
                "text": "Use the supplied relation rule to name the class.",
            },
            "canonical_answer": "class_0",
        }
    )

    with pytest.raises(ValueError, match="relation rule keys"):
        CVASample.from_mapping(payload)


def test_sample_rejects_unregistered_values_in_allowed_nested_fields() -> None:
    payload = _sample_payload()
    payload["question"]["text"] = "Contact alice@example.com for the answer."  # type: ignore[index]
    with pytest.raises(ValueError, match="question text"):
        CVASample.from_mapping(payload)

    payload = _sample_payload()
    payload["error_catalog"][1]["parameters"]["note"] = "alice@example.com"  # type: ignore[index]
    with pytest.raises(ValueError, match="parameters fields"):
        CVASample.from_mapping(payload)

    payload = _sample_payload()
    payload["error_catalog"][1]["secret"] = "alice@example.com"  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown error fields"):
        CVASample.from_mapping(payload)


@pytest.mark.parametrize("operation", ["mean", "max", "min", "unknown"])
def test_bar_schema_rejects_unregistered_operations(operation: str) -> None:
    payload = _sample_payload()
    payload.update(
        {
            "task_family": "bar_chart_aggregate",
            "scene": {"bars": [8, 2, 9, 10], "maximum": 100.0},
            "question": {
                "template": "aggregate",
                "operation": operation,
                "indices": [0, 1],
                "text": "Sum the first two bar heights.",
            },
            "canonical_answer": 10,
            "canonical_reasoning": {"operation": operation, "indices": [0, 1]},
        }
    )

    with pytest.raises(ValueError, match="bar chart operation"):
        CVASample.from_mapping(payload)


def test_bar_schema_binds_question_text_and_registered_indices_to_operation() -> None:
    payload = _sample_payload()
    payload.update(
        {
            "task_family": "bar_chart_aggregate",
            "scene": {"bars": [8, 2, 9, 10], "maximum": 100.0},
            "question": {
                "template": "aggregate",
                "operation": "ratio",
                "indices": [1, 0],
                "text": "Sum the first two bar heights.",
            },
            "canonical_answer": 4.0,
            "canonical_reasoning": {"operation": "ratio", "indices": [1, 0]},
        }
    )

    with pytest.raises(ValueError, match="indices"):
        CVASample.from_mapping(payload)

    payload["question"]["indices"] = [0, 1]  # type: ignore[index]
    with pytest.raises(ValueError, match="question text"):
        CVASample.from_mapping(payload)


def test_bar_schema_rejects_zero_ratio_denominator() -> None:
    payload = _sample_payload()
    payload.update(
        {
            "task_family": "bar_chart_aggregate",
            "scene": {"bars": [8, 0, 9, 10], "maximum": 100.0},
            "question": {
                "template": "aggregate",
                "operation": "ratio",
                "indices": [0, 1],
                "text": "Divide the first bar height by the second.",
            },
            "canonical_answer": 0,
            "canonical_reasoning": {"operation": "ratio", "indices": [0, 1]},
        }
    )

    with pytest.raises(ValueError, match="denominator"):
        CVASample.from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_id", "", "sample_id"),
        ("task_family", "unknown", "task_family"),
        ("canonical_answer", None, "canonical_answer"),
    ],
)
def test_invalid_required_fields_fail_fast(field: str, value: object, message: str) -> None:
    payload = _sample_payload()
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        CVASample.from_mapping(payload)


def test_all_five_task_families_and_all_dataset_splits_are_closed_enums() -> None:
    assert {family.value for family in TaskFamily} == {
        "digit_offset",
        "count_transform",
        "gauge_calibration",
        "bar_chart_aggregate",
        "relation_rule",
    }
    assert {split.value for split in SemanticSplit} == {
        "train",
        "calibration",
        "val",
        "iid_test",
        "ood_test",
    }


def test_error_identifiers_are_unique_and_truth_is_well_formed() -> None:
    payload = _sample_payload()
    payload["error_catalog"].append(payload["error_catalog"][0])  # type: ignore[union-attr,index]

    with pytest.raises(ValueError, match="error_id"):
        CVASample.from_mapping(payload)

    malformed_truth = _sample_payload()
    malformed_truth["error_catalog"][0]["severity"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="truth"):
        CVASample.from_mapping(malformed_truth)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_schema_rejects_nonfinite_floats(value: float) -> None:
    payload = _sample_payload()
    payload["scene"]["value"] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="non-finite"):
        CVASample.from_mapping(payload)

    payload = _sample_payload()
    payload["error_catalog"][1]["severity"] = value  # type: ignore[index]
    with pytest.raises(ValueError, match="severity"):
        CVASample.from_mapping(payload)
