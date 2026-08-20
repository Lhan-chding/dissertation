from __future__ import annotations

import csv
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from compensability_v4.data.splits import DatasetSplit
from compensability_v4.qwen.phase5_support import (
    CheckpointSceneMeasurement,
    HeldOutNaturalError,
    PolicyCheckpoint,
    build_support_dev_candidates,
    retain_held_out_natural_errors,
    summarize_phase5_policy_support,
    write_phase5_outputs,
)


SHA = "a" * 64


def _dataset_rows(count: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    families = ("cross_series", "duplicate_encoding", "trend")
    for index in range(count):
        rows.append(
            {
                "scene_id": f"scene-{index:03d}",
                "family": families[index % len(families)],
                "values": [2 + index, 3 + index, 4 + index, 5 + index],
                "image": f"images/scene-{index:03d}.png",
            }
        )
    return rows


def test_support_dev_candidates_are_deterministic_and_exclude_all_phase4_scenes() -> None:
    first = build_support_dev_candidates(
        _dataset_rows(), excluded_scene_ids=frozenset({"scene-000", "scene-003"}), count=4, seed=7
    )
    second = build_support_dev_candidates(
        reversed(_dataset_rows()),
        excluded_scene_ids=frozenset({"scene-000", "scene-003"}),
        count=4,
        seed=7,
    )

    assert first == second
    assert len(first) == 4
    assert all(scene.split is DatasetSplit.SUPPORT_DEV for scene in first)
    assert not {scene.scene_id for scene in first} & {"scene-000", "scene-003"}
    assert len({scene.semantic_scene_id for scene in first}) == 4
    with pytest.raises(FrozenInstanceError):
        first[0].scene_id = "changed"  # type: ignore[misc]


def test_support_dev_candidates_fail_closed_on_duplicates_or_insufficient_pool() -> None:
    duplicate = [*_dataset_rows(2), _dataset_rows(2)[0]]
    with pytest.raises(ValueError, match="duplicated"):
        build_support_dev_candidates(
            duplicate, excluded_scene_ids=frozenset(), count=1, seed=1
        )
    with pytest.raises(RuntimeError, match="smaller"):
        build_support_dev_candidates(
            _dataset_rows(2), excluded_scene_ids=frozenset({"scene-000"}), count=2, seed=1
        )


def test_natural_error_retention_keeps_every_parseable_error_and_audits_all_candidates() -> None:
    scenes = build_support_dev_candidates(
        _dataset_rows(5), excluded_scene_ids=frozenset(), count=5, seed=3
    )
    output_by_scene: dict[str, str] = {}
    for ordinal, scene in enumerate(scenes):
        if ordinal == 0:
            output_by_scene[scene.scene_id] = ",".join(map(str, scene.truth))
        elif ordinal == 1:
            output_by_scene[scene.scene_id] = "not-a-world"
        elif ordinal == 2:
            changed = (scene.truth[0] + 1, *scene.truth[1:])
            output_by_scene[scene.scene_id] = ",".join(map(str, changed))
        else:
            changed = (scene.truth[0] + 1, scene.truth[1] + 1, *scene.truth[2:])
            output_by_scene[scene.scene_id] = ",".join(map(str, changed))

    errors, traces = retain_held_out_natural_errors(
        scenes, output_by_scene=output_by_scene, stage1_model_sha256=SHA
    )

    assert len(errors) == 3
    assert [len(error.error_indices) for error in errors] == [1, 2, 2]
    assert len(traces) == 5
    assert {trace["selection_status"] for trace in traces} == {
        "excluded_correct",
        "excluded_unparseable",
        "included_natural_error",
    }
    assert all(error.split is DatasetSplit.SUPPORT_DEV for error in errors)
    assert all(error.stage1_model_sha256 == SHA for error in errors)


def _error(scene_id: str, family: str = "cross_series") -> HeldOutNaturalError:
    return HeldOutNaturalError(
        scene_id=scene_id,
        family=family,
        split=DatasetSplit.SUPPORT_DEV,
        truth=(2, 3, 4, 5),
        observed=(9, 3, 4, 5),
        error_indices=(0,),
        facts=(
            {"type": "known_value", "index": 1, "value": 3},
            {"type": "known_value", "index": 2, "value": 4},
            {"type": "known_value", "index": 3, "value": 5},
            {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": 5},
        ),
        image_path="images/example.png",
        stage1_model_sha256=SHA,
        stage1_raw_output="9,3,4,5",
    )


def _measurement(
    scene_id: str,
    checkpoint: PolicyCheckpoint,
    outcomes: tuple[bool, ...],
    *,
    greedy: tuple[int, int, int, int] = (2, 3, 4, 5),
) -> CheckpointSceneMeasurement:
    return CheckpointSceneMeasurement(
        scene_id=scene_id,
        family="cross_series",
        split=DatasetSplit.SUPPORT_DEV,
        checkpoint=checkpoint,
        checkpoint_sha256=(checkpoint.value[0].lower() * 64),
        truth=(2, 3, 4, 5),
        observed=(9, 3, 4, 5),
        greedy_output=greedy,
        greedy_parse_success=True,
        greedy_success=greedy == (2, 3, 4, 5),
        greedy_observation_copy=greedy == (9, 3, 4, 5),
        candidate_logp_true=-1.0,
        candidate_logp_observed=-2.0,
        candidate_margin_true_observed=1.0,
        sample_outputs=tuple((2, 3, 4, 5) if success else (9, 3, 4, 5) for success in outcomes),
        sample_parse_success=tuple(True for _ in outcomes),
        sample_success=outcomes,
        sample_observation_copy=tuple(not success for success in outcomes),
    )


def test_phase5_summary_reports_p_i_g_k_pass_at_k_and_copy_rate_without_thresholds() -> None:
    errors = (_error("scene-a"), _error("scene-b"))
    measurements = tuple(
        _measurement(scene.scene_id, checkpoint, (True, False, False, True))
        for scene in errors
        for checkpoint in PolicyCheckpoint
    )

    summary = summarize_phase5_policy_support(
        errors=errors,
        measurements=measurements,
        pass_at_k=(1, 2, 4),
        informative_group_size=4,
        sampling_temperature=0.7,
        sampling_seed=2026082005,
    )

    assert summary["status"] == "PHASE_5_POLICY_SUPPORT_EXECUTED"
    assert summary["number_of_held_out_natural_errors"] == 2
    assert summary["number_of_checkpoint_scene_rows"] == 8
    assert summary["sampling_rollouts_per_scene"] == 4
    assert summary["pass_at_k"] == [1, 2, 4]
    assert summary["informative_group_size"] == 4
    assert summary["subjective_success_threshold_applied"] is False
    for checkpoint in PolicyCheckpoint:
        row = summary["by_checkpoint"][checkpoint.value]
        assert row["mean_p_i"] == pytest.approx(0.5)
        assert row["mean_G_K"] == pytest.approx(0.875)
        assert row["observation_copy_rate"] == pytest.approx(0.5)
        assert row["greedy_success_rate"] == pytest.approx(1.0)


def test_phase5_summary_requires_complete_four_checkpoint_scene_closure() -> None:
    error = _error("scene-a")
    rows = tuple(_measurement(error.scene_id, checkpoint, (True, False)) for checkpoint in PolicyCheckpoint)
    with pytest.raises(ValueError, match="closure"):
        summarize_phase5_policy_support(
            errors=(error,),
            measurements=rows[:-1],
            pass_at_k=(1, 2),
            informative_group_size=2,
            sampling_temperature=0.7,
            sampling_seed=1,
        )


def test_phase5_writer_emits_required_parquet_json_and_csv_without_overwrite(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    error = _error("scene-a")
    rows = tuple(
        _measurement(error.scene_id, checkpoint, (True, False, True, False))
        for checkpoint in PolicyCheckpoint
    )
    summary = summarize_phase5_policy_support(
        errors=(error,),
        measurements=rows,
        pass_at_k=(1, 2, 4),
        informative_group_size=4,
        sampling_temperature=0.7,
        sampling_seed=1,
    )
    parquet_path = tmp_path / "policy_support_by_scene.parquet"
    informative_path = tmp_path / "informative_group_rate.json"
    pass_path = tmp_path / "pass_at_k.csv"

    write_phase5_outputs(
        parquet_path=parquet_path,
        informative_path=informative_path,
        pass_at_k_path=pass_path,
        measurements=rows,
        summary=summary,
        source_sha256={"support_dev": SHA, "Base": "b" * 64, "C0": "c" * 64, "C1": "d" * 64, "T": "e" * 64},
    )

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    assert table.num_rows == 4
    payload = json.loads(informative_path.read_text(encoding="utf-8"))
    assert payload["number_of_held_out_natural_errors"] == 1
    with pass_path.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(PolicyCheckpoint) * 3
    assert {int(row["k"]) for row in csv_rows} == {1, 2, 4}
    with pytest.raises(FileExistsError, match="overwrite"):
        write_phase5_outputs(
            parquet_path=parquet_path,
            informative_path=informative_path,
            pass_at_k_path=pass_path,
            measurements=rows,
            summary=summary,
            source_sha256={"support_dev": SHA, "Base": "b" * 64, "C0": "c" * 64, "C1": "d" * 64, "T": "e" * 64},
        )
