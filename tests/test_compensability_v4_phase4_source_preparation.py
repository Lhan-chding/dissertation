"""RED contracts for autonomous, split-isolated Phase 4 source preparation."""

from __future__ import annotations

import pytest

from compensability_v4.data.splits import DatasetSplit


def _record(scene_id: str, *, family: str = "trend", values: tuple[int, int, int, int] = (6, 9, 12, 15)) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "family": family,
        "values": list(values),
        "image": f"images/{scene_id}.png",
    }


def test_phase4_source_preparation_selects_disjoint_natural_scenes_deterministically():
    from compensability_v4.training.phase4_sources import (  # noqa: PLC0415
        build_independent_natural_scenes,
    )

    records = (
        _record("phase-c-screen-000001", family="cross_series", values=(2, 3, 5, 7)),
        _record("phase-c-screen-000002", family="duplicate_encoding", values=(4, 4, 8, 12)),
        _record("phase-c-screen-000003", family="trend", values=(6, 9, 12, 15)),
        _record("phase-c-screen-000004", family="trend", values=(7, 10, 13, 16)),
    )

    first = build_independent_natural_scenes(
        records,
        confirm_scene_ids=frozenset({"phase-c-screen-000001"}),
        candidate_cap=3,
        selection_seed=2026081806,
    )
    second = build_independent_natural_scenes(
        records,
        confirm_scene_ids=frozenset({"phase-c-screen-000001"}),
        candidate_cap=3,
        selection_seed=2026081806,
    )

    assert first == second
    assert len(first) == 3
    assert all(scene.scene_id != "phase-c-screen-000001" for scene in first)
    assert all(scene.split is DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN for scene in first)
    assert all(scene.image_path.startswith("images/") for scene in first)
    assert all(scene.facts for scene in first)


def test_phase4_source_preparation_keeps_only_frozen_stage1_single_errors():
    from compensability_v4.training.phase4_sources import (  # noqa: PLC0415
        NaturalObservationCapture,
        retain_natural_single_error_scenes,
    )
    from compensability_v4.schemas.scene import RecoveryScene  # noqa: PLC0415

    scenes = (
        RecoveryScene(
            scene_id="natural-a",
            split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN,
            semantic_scene_id="semantic-natural-a",
            numeric_table_id="numbers-natural-a",
            constraint_graph_id="graph-natural-a",
            truth=(2, 3, 5, 7),
            facts=(
                {"type": "known_value", "index": 1, "value": 3},
                {"type": "known_value", "index": 2, "value": 5},
                {"type": "known_value", "index": 3, "value": 7},
                {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": 5},
            ),
            resized_height=280,
            resized_width=280,
            image_path="images/natural-a.png",
        ),
        RecoveryScene(
            scene_id="natural-b",
            split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN,
            semantic_scene_id="semantic-natural-b",
            numeric_table_id="numbers-natural-b",
            constraint_graph_id="graph-natural-b",
            truth=(4, 4, 8, 12),
            facts=(
                {"type": "known_value", "index": 1, "value": 4},
                {"type": "known_value", "index": 2, "value": 8},
                {"type": "known_value", "index": 3, "value": 12},
                {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": 8},
            ),
            resized_height=280,
            resized_width=280,
            image_path="images/natural-b.png",
        ),
        RecoveryScene(
            scene_id="natural-c",
            split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN,
            semantic_scene_id="semantic-natural-c",
            numeric_table_id="numbers-natural-c",
            constraint_graph_id="graph-natural-c",
            truth=(6, 9, 12, 15),
            facts=(
                {"type": "known_value", "index": 1, "value": 9},
                {"type": "known_value", "index": 2, "value": 12},
                {"type": "known_value", "index": 3, "value": 15},
                {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": 15},
            ),
            resized_height=280,
            resized_width=280,
            image_path="images/natural-c.png",
        ),
    )
    captures = (
        NaturalObservationCapture("natural-a", "2,3,8,7", (1, 10, 10), 25),
        NaturalObservationCapture("natural-b", "4,4,8,12", (1, 10, 10), 25),
        NaturalObservationCapture("natural-c", "not-a-world", (1, 10, 10), 25),
    )

    retained, observations, traces = retain_natural_single_error_scenes(
        scenes,
        captures,
        model_snapshot_sha256="a" * 64,
        target_count=1,
    )

    assert tuple(scene.scene_id for scene in retained) == ("natural-a",)
    assert observations[0].scene_id == "natural-a"
    assert observations[0].observed_values == (2, 3, 8, 7)
    assert observations[0].error_index == 2
    assert [trace["selection_status"] for trace in traces] == [
        "accepted_single_error",
        "rejected_zero_error",
        "rejected_unparseable",
    ]


def test_phase4_source_preparation_fails_closed_when_natural_error_target_is_not_met():
    from compensability_v4.training.phase4_sources import (  # noqa: PLC0415
        NaturalObservationCapture,
        retain_natural_single_error_scenes,
    )
    from compensability_v4.schemas.scene import RecoveryScene  # noqa: PLC0415

    scene = RecoveryScene(
        scene_id="natural",
        split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN,
        semantic_scene_id="semantic-natural",
        numeric_table_id="numbers-natural",
        constraint_graph_id="graph-natural",
        truth=(2, 3, 5, 7),
        facts=(
            {"type": "known_value", "index": 1, "value": 3},
            {"type": "known_value", "index": 2, "value": 5},
            {"type": "known_value", "index": 3, "value": 7},
            {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": 5},
        ),
        resized_height=280,
        resized_width=280,
        image_path="images/natural.png",
    )

    with pytest.raises(RuntimeError, match="single-error"):
        retain_natural_single_error_scenes(
            (scene,),
            (NaturalObservationCapture("natural", "2,3,5,7", (1, 10, 10), 25),),
            model_snapshot_sha256="a" * 64,
            target_count=1,
        )
