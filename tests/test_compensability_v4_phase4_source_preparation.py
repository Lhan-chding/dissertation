"""RED contracts for autonomous, split-isolated Phase 4 source preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compensability_v4.data.splits import DatasetSplit


def _record(
    scene_id: str,
    *,
    family: str = "trend",
    values: tuple[int, int, int, int] = (6, 9, 12, 15),
) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "family": family,
        "values": list(values),
        "image": f"images/{scene_id}.png",
    }


def _s6_rows(
    scene_id: str,
    *,
    family: str,
    truth: tuple[int, int, int, int],
    raw_output: str,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for interface in (
        "I0_hard_text_symbolic_recovery",
        "I2_candidate_world_diagnostic",
        "I3_same_conversation_visual_revision",
        "I4_exact_cached_natural_continuation",
    ):
        for cue in ("no_cue", "valid_cue", "sham_cue", "counterfactual_cue"):
            rows.append(
                {
                    "call_id": f"S6:{scene_id}:{interface}:{cue}",
                    "scene_id": scene_id,
                    "family": family,
                    "interface": interface,
                    "cue_condition": cue,
                    "true_world": list(truth),
                }
            )
    rows.append(
        {
            "call_id": f"S6:{scene_id}:I1_soft_report_diagnostic:no_cue",
            "scene_id": scene_id,
            "family": family,
            "interface": "I1_soft_report_diagnostic",
            "cue_condition": "no_cue",
            "true_world": list(truth),
            "diagnostic_payload": {"raw_output": raw_output},
            "source_stage": "S6_runtime",
            "source_branch": "stage1_soft_report_runtime",
        }
    )
    return tuple(rows)


def test_phase4_source_preparation_selects_disjoint_natural_scenes_deterministically():
    from compensability_v4.training.phase4_sources import (
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
    from compensability_v4.schemas.scene import RecoveryScene
    from compensability_v4.training.phase4_sources import (
        NaturalObservationCapture,
        retain_natural_single_error_scenes,
    )

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


def test_phase4_source_preparation_records_out_of_domain_stage1_output_without_blocking():
    from compensability_v4.schemas.scene import RecoveryScene
    from compensability_v4.training.phase4_sources import (
        NaturalObservationCapture,
        retain_natural_single_error_scenes,
    )

    scene = RecoveryScene(
        scene_id="natural-domain",
        split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN,
        semantic_scene_id="semantic-natural-domain",
        numeric_table_id="numbers-natural-domain",
        constraint_graph_id="graph-natural-domain",
        truth=(2, 3, 5, 7),
        facts=(
            {"type": "known_value", "index": 1, "value": 3},
            {"type": "known_value", "index": 2, "value": 5},
            {"type": "known_value", "index": 3, "value": 7},
            {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": 5},
        ),
        resized_height=280,
        resized_width=280,
        image_path="images/natural-domain.png",
    )

    retained, observations, traces = retain_natural_single_error_scenes(
        (scene,),
        (NaturalObservationCapture("natural-domain", "2,3,5,99", (1, 10, 10), 25),),
        model_snapshot_sha256="a" * 64,
        target_count=0,
        value_domain=range(2, 19),
    )

    assert retained == ()
    assert observations == ()
    assert traces[0]["selection_status"] == "rejected_outside_frozen_domain"


def test_phase4_source_preparation_fails_closed_when_natural_error_target_is_not_met():
    from compensability_v4.schemas.scene import RecoveryScene
    from compensability_v4.training.phase4_sources import (
        NaturalObservationCapture,
        retain_natural_single_error_scenes,
    )

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


def test_phase4_source_preparation_reuses_frozen_s6_i1_legacy_observations():
    from compensability_v4.training.phase4_sources import (
        NaturalObservationCapture,
        build_legacy_s6_natural_candidates,
    )

    scenes, captures = build_legacy_s6_natural_candidates(
        (
            {
                "call_id": "S6:legacy-001:I1:no_cue",
                "scene_id": "legacy-001",
                "family": "trend",
                "interface": "I1_soft_report_diagnostic",
                "cue_condition": "no_cue",
                "true_world": [2, 4, 6, 8],
                "diagnostic_payload": {"raw_output": "2,4,7,8"},
                "source_stage": "S6_runtime",
                "source_branch": "stage1_soft_report_runtime",
            },
        ),
        image_paths={"legacy-001": "images/legacy-001.png"},
        image_grid_thw=(1, 20, 20),
    )

    assert scenes[0].split is DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN
    assert scenes[0].truth == (2, 4, 6, 8)
    assert captures == (NaturalObservationCapture("legacy-001", "2,4,7,8", (1, 20, 20), 100),)


def test_phase4_preparation_builds_and_publishes_hash_bound_sources(tmp_path: Path):
    from compensability_v4.training.phase4 import sha256_path
    from compensability_v4.training.phase4_sources import (
        prepare_legacy_s6_support_sources,
        validate_prepared_source_summary,
        write_prepared_support_sources,
    )

    records = (
        *_s6_rows("legacy-001", family="trend", truth=(2, 4, 6, 8), raw_output="2,4,7,8"),
        *_s6_rows(
            "legacy-002",
            family="duplicate_encoding",
            truth=(3, 3, 6, 9),
            raw_output="3,3,6,9",
        ),
    )
    summary = {
        "status": "PHASE_3_INTERFACE_LADDER_EXECUTED_WITH_DIAGNOSTICS",
        "number_of_source_scenes": 2,
        "number_of_cells": 34,
        "model_snapshot_sha256": "a" * 64,
        "training_invoked": False,
        "rl_invoked": False,
        "subjective_success_threshold_applied": False,
    }
    dataset = (
        _record("legacy-001", family="trend", values=(2, 4, 6, 8)),
        _record("legacy-002", family="duplicate_encoding", values=(3, 3, 6, 9)),
    )

    prepared = prepare_legacy_s6_support_sources(
        interface_records=records,
        interface_summary=summary,
        dataset_records=dataset,
        model_snapshot_sha256="a" * 64,
        expected_scenes=2,
        symbolic_scene_count=3,
        symbolic_seed=17,
        value_domain=range(2, 19),
        image_grid_thw=(1, 20, 20),
    )

    assert len(prepared.symbolic_scenes) == 3
    assert tuple(scene.scene_id for scene in prepared.natural_scenes) == ("legacy-001",)
    assert prepared.natural_observations[0].observed_values == (2, 4, 7, 8)
    assert prepared.selection_counts == {
        "accepted_single_error": 1,
        "rejected_zero_error": 1,
    }

    source_hashes = {
        "s6_per_scene": "b" * 64,
        "s6_summary": "c" * 64,
        "dataset_manifest": "d" * 64,
        "dataset_records": "e" * 64,
    }
    paths = write_prepared_support_sources(
        output_root=tmp_path / "sources",
        prepared=prepared,
        source_hashes=source_hashes,
    )
    validated = validate_prepared_source_summary(paths.summary, paths=paths)

    assert validated == {
        "symbolic_scenes": sha256_path(paths.symbolic_scenes),
        "natural_scenes": sha256_path(paths.natural_scenes),
        "natural_observations": sha256_path(paths.natural_observations),
    }
    payload = json.loads(paths.summary.read_text(encoding="utf-8"))
    assert payload["contains_confirmatory_data"] is False
    assert payload["source_hashes"] == source_hashes
    assert payload["counts"]["natural_single_error_scenes"] == 1
    with pytest.raises(FileExistsError, match="overwrite"):
        write_prepared_support_sources(
            output_root=tmp_path / "sources",
            prepared=prepared,
            source_hashes=source_hashes,
        )


def test_phase4_preparation_rejects_incomplete_s6_cell_closure():
    from compensability_v4.training.phase4_sources import prepare_legacy_s6_support_sources

    rows = _s6_rows("legacy-001", family="trend", truth=(2, 4, 6, 8), raw_output="2,4,7,8")
    with pytest.raises(RuntimeError, match="17-cell"):
        prepare_legacy_s6_support_sources(
            interface_records=rows[:-1],
            interface_summary={
                "status": "PHASE_3_INTERFACE_LADDER_EXECUTED_WITH_DIAGNOSTICS",
                "number_of_source_scenes": 1,
                "number_of_cells": 17,
                "model_snapshot_sha256": "a" * 64,
                "training_invoked": False,
                "rl_invoked": False,
                "subjective_success_threshold_applied": False,
            },
            dataset_records=(_record("legacy-001", values=(2, 4, 6, 8)),),
            model_snapshot_sha256="a" * 64,
            expected_scenes=1,
            symbolic_scene_count=1,
            symbolic_seed=17,
            value_domain=range(2, 19),
            image_grid_thw=(1, 20, 20),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidate_cap": 0, "selection_seed": 1}, "candidate_cap"),
        ({"candidate_cap": 1, "selection_seed": True}, "selection_seed"),
    ],
)
def test_phase4_independent_selection_rejects_invalid_controls(kwargs, message):
    from compensability_v4.training.phase4_sources import build_independent_natural_scenes

    with pytest.raises((TypeError, ValueError), match=message):
        build_independent_natural_scenes(
            (_record("source-001"),),
            confirm_scene_ids=frozenset(),
            **kwargs,
        )


def test_phase4_independent_selection_rejects_duplicates_and_insufficient_pool():
    from compensability_v4.training.phase4_sources import build_independent_natural_scenes

    with pytest.raises(ValueError, match="not unique"):
        build_independent_natural_scenes(
            (_record("source-001"), _record("source-001")),
            confirm_scene_ids=frozenset(),
            candidate_cap=1,
            selection_seed=1,
        )
    with pytest.raises(RuntimeError, match="smaller"):
        build_independent_natural_scenes(
            (_record("source-001"),),
            confirm_scene_ids=frozenset({"source-001"}),
            candidate_cap=1,
            selection_seed=1,
        )


@pytest.mark.parametrize(
    ("records", "image_paths", "grid", "message"),
    [
        ((), {}, (1, 19, 20), "image_grid_thw"),
        ((object(),), {}, (1, 20, 20), "must be mappings"),
        (
            (
                {
                    "interface": "I1_soft_report_diagnostic",
                    "cue_condition": "valid_cue",
                    "source_stage": "S6_runtime",
                    "source_branch": "stage1_soft_report_runtime",
                },
            ),
            {},
            (1, 20, 20),
            "provenance",
        ),
        ((), {}, (1, 20, 20), "contains no I1"),
    ],
)
def test_phase4_legacy_candidate_builder_fails_closed(records, image_paths, grid, message):
    from compensability_v4.training.phase4_sources import build_legacy_s6_natural_candidates

    with pytest.raises((RuntimeError, TypeError, ValueError), match=message):
        build_legacy_s6_natural_candidates(
            records,
            image_paths=image_paths,
            image_grid_thw=grid,
        )


def test_phase4_prepared_summary_detects_missing_or_drifted_files(tmp_path: Path):
    from compensability_v4.training.phase4_sources import validate_prepared_source_summary

    with pytest.raises(RuntimeError, match="missing"):
        validate_prepared_source_summary(tmp_path / "missing.json")

    summary = tmp_path / "source_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "phase_4_prepared_support_sources",
                "status": "PHASE_4_SUPPORT_SOURCES_PREPARED_FROM_FROZEN_S6",
                "contains_confirmatory_data": False,
                "counts": {"symbolic_scenes": 1, "natural_single_error_scenes": 1},
                "output_hashes": {
                    "symbolic_scenes": "a" * 64,
                    "natural_scenes": "b" * 64,
                    "natural_observations": "c" * 64,
                    "selection_trace": "d" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_prepared_source_summary(summary)


def test_phase4_prepared_summary_rejects_boolean_counts(tmp_path: Path):
    from compensability_v4.training.phase4_sources import validate_prepared_source_summary

    summary = tmp_path / "source_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "phase_4_prepared_support_sources",
                "status": "PHASE_4_SUPPORT_SOURCES_PREPARED_FROM_FROZEN_S6",
                "contains_confirmatory_data": False,
                "counts": {
                    "symbolic_scenes": True,
                    "natural_single_error_scenes": 1,
                },
                "output_hashes": {
                    "symbolic_scenes": "a" * 64,
                    "natural_scenes": "b" * 64,
                    "natural_observations": "c" * 64,
                    "selection_trace": "d" * 64,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="malformed"):
        validate_prepared_source_summary(summary)


def test_phase4_natural_filter_records_multiple_errors_and_target_truncation():
    from compensability_v4.training.phase4_sources import (
        NaturalObservationCapture,
        build_independent_natural_scenes,
        retain_natural_single_error_scenes,
    )

    scene = build_independent_natural_scenes(
        (_record("natural-filter", values=(6, 9, 12, 15)),),
        confirm_scene_ids=frozenset(),
        candidate_cap=1,
        selection_seed=1,
    )[0]
    retained, observations, traces = retain_natural_single_error_scenes(
        (scene,),
        (NaturalObservationCapture(scene.scene_id, "7,10,12,15", (1, 20, 20), 100),),
        model_snapshot_sha256="a" * 64,
        target_count=0,
    )
    assert retained == observations == ()
    assert traces[0]["selection_status"] == "rejected_multiple_errors"

    retained, observations, traces = retain_natural_single_error_scenes(
        (scene,),
        (NaturalObservationCapture(scene.scene_id, "7,9,12,15", (1, 20, 20), 100),),
        model_snapshot_sha256="a" * 64,
        target_count=0,
    )
    assert retained == observations == ()
    assert traces[0]["retained"] is False


def test_phase4_preparation_rejects_summary_and_source_hash_drift(tmp_path: Path):
    from compensability_v4.training.phase4_sources import (
        prepare_legacy_s6_support_sources,
        write_prepared_support_sources,
    )

    rows = _s6_rows("legacy-001", family="trend", truth=(2, 4, 6, 8), raw_output="2,4,7,8")
    with pytest.raises(RuntimeError, match="summary contract"):
        prepare_legacy_s6_support_sources(
            interface_records=rows,
            interface_summary={"status": "wrong"},
            dataset_records=(_record("legacy-001", values=(2, 4, 6, 8)),),
            model_snapshot_sha256="a" * 64,
            expected_scenes=1,
            symbolic_scene_count=1,
            symbolic_seed=17,
            value_domain=range(2, 19),
            image_grid_thw=(1, 20, 20),
        )

    prepared = prepare_legacy_s6_support_sources(
        interface_records=rows,
        interface_summary={
            "status": "PHASE_3_INTERFACE_LADDER_EXECUTED_WITH_DIAGNOSTICS",
            "number_of_source_scenes": 1,
            "number_of_cells": 17,
            "model_snapshot_sha256": "a" * 64,
            "training_invoked": False,
            "rl_invoked": False,
            "subjective_success_threshold_applied": False,
        },
        dataset_records=(_record("legacy-001", values=(2, 4, 6, 8)),),
        model_snapshot_sha256="a" * 64,
        expected_scenes=1,
        symbolic_scene_count=1,
        symbolic_seed=17,
        value_domain=range(2, 19),
        image_grid_thw=(1, 20, 20),
    )
    with pytest.raises(ValueError, match="input hashes"):
        write_prepared_support_sources(
            output_root=tmp_path / "bad-hashes",
            prepared=prepared,
            source_hashes={"s6_per_scene": "not-a-hash"},
        )
