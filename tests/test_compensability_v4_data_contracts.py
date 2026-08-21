"""RED contracts for immutable v4 records and confirm-set isolation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from compensability_v4.data.splits import (
    DatasetSplit,
    SplitIsolationError,
    validate_split_isolation,
)
from compensability_v4.schemas.observation import NaturalObservation
from compensability_v4.schemas.record import ExperimentRecord
from compensability_v4.schemas.scene import RecoveryScene


def _scene_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "scene_id": "scene-001",
        "split": "symbolic_support_train",
        "semantic_scene_id": "semantic-001",
        "numeric_table_id": "numbers-001",
        "constraint_graph_id": "graph-001",
        "truth": [9, 4, 5, 6],
        "facts": [
            {"type": "known_value", "index": 1, "value": 4},
            {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": 13},
        ],
        "resized_height": 280,
        "resized_width": 280,
        "image_path": "images/scene-001.png",
    }
    payload.update(updates)
    return payload


def test_v4_split_enum_is_closed_and_uses_the_preregistered_names() -> None:
    assert {member.value for member in DatasetSplit} == {
        "legacy_diagnostic",
        "symbolic_support_train",
        "natural_error_support_train",
        "support_dev",
        "confirm_iid",
        "confirm_style_ood",
        "confirm_constraint_ood",
        "confirm_error_mechanism_ood",
    }


def test_recovery_scene_round_trip_is_strict_and_deeply_immutable() -> None:
    payload = _scene_payload()

    scene = RecoveryScene.from_mapping(payload)

    assert scene.split is DatasetSplit.SYMBOLIC_SUPPORT_TRAIN
    assert scene.truth == (9, 4, 5, 6)
    assert scene.to_mapping() == payload
    payload["truth"][0] = 99  # type: ignore[index]
    payload["facts"][0]["value"] = 99  # type: ignore[index]
    assert scene.truth == (9, 4, 5, 6)
    assert scene.facts[0]["value"] == 4
    with pytest.raises(FrozenInstanceError):
        scene.scene_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        scene.facts[0]["value"] = 7  # type: ignore[index]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"truth": [1, 2, 3]}, "four"),
        ({"truth": [1, 2, 3, 4.0]}, "integer"),
        ({"resized_height": 281}, "28"),
        ({"resized_width": 252, "unexpected": "field"}, "unknown"),
    ],
)
def test_recovery_scene_rejects_noncanonical_worlds_and_visual_budgets(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        RecoveryScene.from_mapping(_scene_payload(**updates))


def test_natural_observation_preserves_image_token_provenance() -> None:
    payload = {
        "observation_id": "obs-001",
        "scene_id": "scene-001",
        "observed_values": [8, 4, 5, 6],
        "error_index": 0,
        "stage1_model_hash": "a" * 64,
        "image_grid_thw": [1, 20, 20],
        "visual_token_count": 100,
    }

    observation = NaturalObservation.from_mapping(payload)

    assert observation.observed_values == (8, 4, 5, 6)
    assert observation.image_grid_thw == (1, 20, 20)
    assert observation.to_mapping() == payload
    with pytest.raises(FrozenInstanceError):
        observation.error_index = 1  # type: ignore[misc]


@pytest.mark.parametrize("error_index", [-1, 4, True, "0"])
def test_natural_observation_rejects_invalid_error_locations(error_index: object) -> None:
    payload = {
        "observation_id": "obs-001",
        "scene_id": "scene-001",
        "observed_values": [8, 4, 5, 6],
        "error_index": error_index,
        "stage1_model_hash": "a" * 64,
        "image_grid_thw": [1, 20, 20],
        "visual_token_count": 100,
    }
    with pytest.raises((TypeError, ValueError), match="error_index"):
        NaturalObservation.from_mapping(payload)


def test_experiment_record_requires_reproducibility_and_interface_provenance() -> None:
    payload = {
        "record_id": "record-001",
        "scene_id": "scene-001",
        "observation_id": "obs-001",
        "interface": "symbolic_downstream_recovery",
        "cue_condition": "valid_cue",
        "prompt_hash": "b" * 64,
        "tokenizer_version": "fixed-snapshot-tokenizer",
        "model_snapshot_hash": "c" * 64,
        "output_text": "9,4,5,6",
    }

    record = ExperimentRecord.from_mapping(payload)

    assert record.to_mapping() == payload
    with pytest.raises(FrozenInstanceError):
        record.output_text = "8,4,5,6"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("leakage_key", "shared_value"),
    [
        ("semantic_scene_id", "semantic-shared"),
        ("numeric_table_id", "numbers-shared"),
        ("constraint_graph_id", "graph-shared"),
    ],
)
def test_confirm_splits_reject_training_or_legacy_leakage(
    leakage_key: str, shared_value: str
) -> None:
    train = RecoveryScene.from_mapping(
        _scene_payload(scene_id="train", **{leakage_key: shared_value})
    )
    confirm_updates: dict[str, object] = {
        "scene_id": "confirm",
        "split": "confirm_iid",
        "semantic_scene_id": "semantic-confirm",
        "numeric_table_id": "numbers-confirm",
        "constraint_graph_id": "graph-confirm",
        leakage_key: shared_value,
    }
    confirm = RecoveryScene.from_mapping(_scene_payload(**confirm_updates))

    with pytest.raises(SplitIsolationError, match=leakage_key):
        validate_split_isolation([train, confirm])


def test_disjoint_train_dev_and_confirm_splits_pass_isolation() -> None:
    train = RecoveryScene.from_mapping(_scene_payload(scene_id="train"))
    dev = RecoveryScene.from_mapping(
        _scene_payload(
            scene_id="dev",
            split="support_dev",
            semantic_scene_id="semantic-dev",
            numeric_table_id="numbers-dev",
            constraint_graph_id="graph-dev",
        )
    )
    confirm = RecoveryScene.from_mapping(
        _scene_payload(
            scene_id="confirm",
            split="confirm_iid",
            semantic_scene_id="semantic-confirm",
            numeric_table_id="numbers-confirm",
            constraint_graph_id="graph-confirm",
        )
    )

    assert validate_split_isolation([train, dev, confirm]) is None
