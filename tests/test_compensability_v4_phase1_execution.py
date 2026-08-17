"""TDD contracts for the real Phase-1 T1--T6 server runner."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from compensability_v4.diagnostics.capability_chain import (
    CapabilityTaskType,
    build_capability_calls,
    evaluate_capability_call,
    load_legacy_capability_scenes,
    summarize_capability_run,
)
from compensability_v4.qwen.capability_runner import (
    execute_capability_calls,
    write_capability_outputs,
)


PROMPTS = {task.value: f"frozen prompt for {task.value}" for task in CapabilityTaskType}


def _screen_row(index: int, *, eligible: bool = True) -> dict[str, object]:
    truth = [8 + index, 4, 5, 9]
    return {
        "scene_id": f"scene-{index:03d}",
        "family": "cross_series",
        "chart_type": "grouped_bar",
        "operation": "sum",
        "values": truth,
        "perceived_values": [truth[0] - 1, *truth[1:]],
        "parse_success": True,
        "natural_perception_error": True,
        "one_position_error": True,
        "operator_sensitive": True,
        "design_recoverability_validated": True,
        "eligible": eligible,
    }


def test_legacy_scene_loader_keeps_only_hash_bound_eligible_rows(tmp_path) -> None:
    source = tmp_path / "screen_records.jsonl"
    rows = [_screen_row(0), _screen_row(1, eligible=False)]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    scenes = load_legacy_capability_scenes(
        source,
        expected_scenes=1,
        expected_family_counts={"cross_series": 1},
    )

    assert scenes[0].scene_id == "scene-000"
    assert scenes[0].error_index == 0
    assert scenes[0].facts


def test_legacy_scene_loader_rejects_ineligible_semantic_drift(tmp_path) -> None:
    source = tmp_path / "screen_records.jsonl"
    row = _screen_row(0)
    row["operator_sensitive"] = False
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="eligible predicate"):
        load_legacy_capability_scenes(
            source,
            expected_scenes=1,
            expected_family_counts={"cross_series": 1},
        )


def test_capability_plan_is_complete_balanced_and_deterministic(tmp_path) -> None:
    source = tmp_path / "screen_records.jsonl"
    source.write_text(
        "\n".join(json.dumps(_screen_row(index)) for index in range(4)) + "\n",
        encoding="utf-8",
    )
    scenes = load_legacy_capability_scenes(
        source,
        expected_scenes=4,
        expected_family_counts={"cross_series": 4},
    )

    calls = build_capability_calls(
        scenes,
        prompts=PROMPTS,
        candidate_labels=("A", "B", "C", "D"),
        seed=2026081701,
    )

    assert calls == build_capability_calls(
        scenes,
        prompts=PROMPTS,
        candidate_labels=("A", "B", "C", "D"),
        seed=2026081701,
    )
    assert len(calls) == 24
    assert Counter(call.task_type for call in calls) == Counter(
        {task: 4 for task in CapabilityTaskType}
    )
    t1 = [call for call in calls if call.task_type is CapabilityTaskType.T1]
    assert Counter(call.expected_output for call in t1) == Counter({"YES": 2, "NO": 2})
    t5 = [call for call in calls if call.task_type is CapabilityTaskType.T5]
    assert {call.expected_output for call in t5} == {"A", "B", "C", "D"}
    assert all(len(call.candidate_worlds) == 4 for call in t5)
    assert all(len(set(call.candidate_worlds)) == 4 for call in t5)
    assert all(call.messages[0]["content"] == PROMPTS[call.task_type.value] for call in calls)


def test_phase1_scoring_reports_paired_gaps_without_success_thresholds(tmp_path) -> None:
    source = tmp_path / "screen_records.jsonl"
    source.write_text(
        "\n".join(json.dumps(_screen_row(index)) for index in range(4)) + "\n",
        encoding="utf-8",
    )
    scenes = load_legacy_capability_scenes(
        source,
        expected_scenes=4,
        expected_family_counts={"cross_series": 4},
    )
    calls = build_capability_calls(
        scenes,
        prompts=PROMPTS,
        candidate_labels=("A", "B", "C", "D"),
        seed=2026081701,
    )
    records = []
    for call in calls:
        raw = call.expected_output
        if call.scene_id in {"scene-002", "scene-003"} and call.task_type in {
            CapabilityTaskType.T3,
            CapabilityTaskType.T6,
        }:
            raw = "INVALID"
        records.append(evaluate_capability_call(call, raw))

    summary_rows, gaps = summarize_capability_run(tuple(records), bootstrap_resamples=200)

    assert len(summary_rows) == 6
    assert gaps["G_search"]["estimate"] == pytest.approx(0.5)
    assert gaps["G_loc"]["estimate"] == pytest.approx(0.5)
    assert gaps["subjective_success_threshold_applied"] is False
    assert gaps["T5_establishes_full_recovery"] is False


class _FakeModel:
    def __init__(self, outputs: tuple[str, ...]) -> None:
        self._outputs = iter(outputs)

    def complete_text(self, messages: object) -> str:
        assert messages
        return next(self._outputs)


def test_runtime_executes_each_frozen_call_once(tmp_path) -> None:
    source = tmp_path / "screen_records.jsonl"
    source.write_text(json.dumps(_screen_row(0)) + "\n", encoding="utf-8")
    scenes = load_legacy_capability_scenes(
        source,
        expected_scenes=1,
        expected_family_counts={"cross_series": 1},
    )
    calls = build_capability_calls(
        scenes,
        prompts=PROMPTS,
        candidate_labels=("A", "B", "C", "D"),
        seed=2026081701,
    )
    progress: list[tuple[int, int]] = []

    records = execute_capability_calls(
        _FakeModel(tuple(call.expected_output for call in calls)),
        object(),
        calls,
        max_new_tokens=32,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert len(records) == 6
    assert all(record.is_correct for record in records)
    assert progress[-1] == (6, 6)


def test_output_writer_emits_exact_three_no_overwrite_artifacts(tmp_path) -> None:
    source = tmp_path / "screen_records.jsonl"
    source.write_text(json.dumps(_screen_row(0)) + "\n", encoding="utf-8")
    scenes = load_legacy_capability_scenes(
        source,
        expected_scenes=1,
        expected_family_counts={"cross_series": 1},
    )
    calls = build_capability_calls(
        scenes,
        prompts=PROMPTS,
        candidate_labels=("A", "B", "C", "D"),
        seed=2026081701,
    )
    records = tuple(evaluate_capability_call(call, call.expected_output) for call in calls)
    summaries, computed_gaps = summarize_capability_run(records, bootstrap_resamples=20)
    gaps = {
        **computed_gaps,
        "subjective_success_threshold_applied": False,
        "T5_establishes_full_recovery": False,
    }
    output = tmp_path / "capability_chain"

    write_capability_outputs(output, records=records, summaries=summaries, gaps=gaps)

    assert {path.name for path in output.iterdir()} == {
        "per_scene.csv",
        "summary_by_family.csv",
        "paired_gaps.json",
    }
    assert json.loads((output / "paired_gaps.json").read_text())[
        "subjective_success_threshold_applied"
    ] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        write_capability_outputs(output, records=records, summaries=summaries, gaps=gaps)
