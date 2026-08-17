"""RED contracts for real Phase-2 teacher-forced candidate scoring."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import ClassVar

import pytest

from compensability_v4.diagnostics.capability_chain import LegacyCapabilityScene
from compensability_v4.qwen.phase2_candidate import (
    CueCondition,
    build_candidate_label_evidence,
    build_candidate_scoring_plan,
    execute_candidate_scoring_plan,
    summarize_candidate_scoring,
    validate_phase1_candidate_source,
    write_candidate_scoring_outputs,
)
from compensability_v4.theory.candidate_space import constraint_supported_candidates


def _facts(truth: tuple[int, int, int, int]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "type": "known_value",
            "index": index,
            "value": value,
            "fact_id": f"truth-{index}",
        }
        for index, value in enumerate(truth)
    )


def _scene(index: int) -> LegacyCapabilityScene:
    truth = (8 + index, 4, 5, 9)
    return LegacyCapabilityScene(
        scene_id=f"scene-{index:03d}",
        family="duplicate_encoding",
        truth=truth,
        observed=(truth[0] - 1, *truth[1:]),
        facts=_facts(truth),
        error_index=0,
        value_domain=tuple(range(2, 19)),
    )


def test_candidate_plan_is_four_condition_balanced_and_prompt_paired() -> None:
    calls = build_candidate_scoring_plan(
        tuple(_scene(index) for index in range(4)),
        prompt="Choose one candidate label and output only that label.",
        candidate_labels=("A", "B", "C", "D"),
        seed=2026081701,
    )

    assert len(calls) == 16
    assert Counter(call.condition for call in calls) == Counter(
        {condition: 4 for condition in CueCondition}
    )
    assert len({call.call_id for call in calls}) == 16
    valid_calls = tuple(call for call in calls if call.condition is CueCondition.VALID_CUE)
    assert Counter(call.true_label for call in valid_calls) == Counter(
        {"A": 1, "B": 1, "C": 1, "D": 1}
    )
    for scene_id in {call.scene_id for call in calls}:
        group = tuple(call for call in calls if call.scene_id == scene_id)
        assert len({call.candidate_worlds for call in group}) == 1
        assert len({call.candidate_labels for call in group}) == 1
        assert all(len(set(call.candidate_worlds)) == 4 for call in group)
        no_cue = next(call for call in group if call.condition is CueCondition.NO_CUE)
        valid = next(call for call in group if call.condition is CueCondition.VALID_CUE)
        no_payload = json.loads(no_cue.messages[1]["content"])
        valid_payload = json.loads(valid.messages[1]["content"])
        assert no_payload.pop("facts") == []
        assert valid_payload.pop("facts")
        assert no_payload == valid_payload
        counterfactual = next(
            call for call in group if call.condition is CueCondition.COUNTERFACTUAL_CUE
        )
        assert constraint_supported_candidates(
            counterfactual.observed_world,
            counterfactual.facts,
            counterfactual.value_domain,
        ) == (counterfactual.counterfactual_world,)
        sham = next(call for call in group if call.condition is CueCondition.SHAM_CUE)
        assert (
            len(
                constraint_supported_candidates(
                    sham.observed_world,
                    sham.facts,
                    sham.value_domain,
                )
            )
            > 1
        )


class _Tokenizer:
    table: ClassVar[dict[str, int]] = {"A": 1, "B": 2, "C": 3, "D": 4}

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self.table[value]]


class _Processor:
    tokenizer = _Tokenizer()


class _ScoringModel:
    def __init__(self) -> None:
        self.calls = 0
        self.generate_calls = 0

    def score_candidate_logits(self, messages, labels):
        self.calls += 1
        assert messages and tuple(labels) == ("A", "B", "C", "D")
        return {label: float(4 - index) for index, label in enumerate(labels)}

    def generate(self, **_kwargs):
        self.generate_calls += 1
        raise AssertionError("Phase 2 candidate scoring must never call generate()")


def test_candidate_execution_is_teacher_forced_and_reports_registered_metrics() -> None:
    calls = build_candidate_scoring_plan(
        (_scene(0),),
        prompt="Choose one label.",
        candidate_labels=("A", "B", "C", "D"),
        seed=2026081701,
    )
    model = _ScoringModel()
    progress: list[tuple[int, int]] = []

    records = execute_candidate_scoring_plan(
        model,
        _Processor(),
        calls,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert len(records) == 4
    assert model.calls == 4
    assert model.generate_calls == 0
    assert progress[-1] == (4, 4)
    assert all(record.generation_invoked is False for record in records)
    assert all(math.isfinite(record.logp_true) for record in records)
    assert all(math.isfinite(record.logp_observed) for record in records)
    assert all(math.isfinite(record.margin_true_observed) for record in records)
    assert all(1 <= record.true_rank <= 4 for record in records)
    assert all(1 <= record.observed_rank <= 4 for record in records)
    assert all(len(record.candidate_logits) == 4 for record in records)
    assert all(len(record.candidate_log_probabilities) == 4 for record in records)

    summary = summarize_candidate_scoring(records, bootstrap_resamples=100)
    assert summary["number_of_scenes"] == 1
    assert summary["number_of_forward_calls"] == 4
    assert summary["subjective_success_threshold_applied"] is False
    assert summary["generation_invoked"] is False
    assert set(summary["paired_effects"]) == {
        "valid_minus_no_cue_margin",
        "sham_minus_no_cue_margin",
        "counterfactual_minus_no_cue_target_margin",
    }


def test_candidate_labels_and_outputs_are_immutable_and_no_overwrite(tmp_path: Path) -> None:
    labels = build_candidate_label_evidence(
        _Tokenizer(),
        ("A", "B", "C", "D"),
        model_snapshot_sha256="a" * 64,
    )
    assert labels["labels"] == [
        {"label": "A", "token_id": 1},
        {"label": "B", "token_id": 2},
        {"label": "C", "token_id": 3},
        {"label": "D", "token_id": 4},
    ]
    calls = build_candidate_scoring_plan(
        (_scene(0),),
        prompt="Choose one label.",
        candidate_labels=("A", "B", "C", "D"),
        seed=2026081701,
    )
    records = execute_candidate_scoring_plan(_ScoringModel(), _Processor(), calls)
    summary = summarize_candidate_scoring(records, bootstrap_resamples=100)
    labels_path = tmp_path / "tokenizer/candidate_labels.json"
    records_path = tmp_path / "candidate_scoring/per_scene.jsonl"
    summary_path = tmp_path / "candidate_scoring/summary.json"

    write_candidate_scoring_outputs(
        labels_path=labels_path,
        records_path=records_path,
        summary_path=summary_path,
        label_evidence=labels,
        records=records,
        summary=summary,
    )

    assert json.loads(labels_path.read_text())["generation_invoked"] is False
    rows = tuple(json.loads(line) for line in records_path.read_text().splitlines())
    assert len(rows) == 4
    assert {row["cue_condition"] for row in rows} == {
        "no_cue",
        "valid_cue",
        "sham_cue",
        "counterfactual_cue",
    }
    assert json.loads(summary_path.read_text())["number_of_forward_calls"] == 4
    with pytest.raises(FileExistsError, match="overwrite"):
        write_candidate_scoring_outputs(
            labels_path=labels_path,
            records_path=records_path,
            summary_path=summary_path,
            label_evidence=labels,
            records=records,
            summary=summary,
        )


def test_phase1_candidate_source_validation_is_structural_not_outcome_gated(
    tmp_path: Path,
) -> None:
    scenes = (_scene(0), _scene(1))
    per_scene = tmp_path / "per_scene.csv"
    with per_scene.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("call_id", "scene_id", "family", "task_type"),
        )
        writer.writeheader()
        for scene in scenes:
            for task in ("T1", "T2", "T3", "T4", "T5", "T6"):
                writer.writerow(
                    {
                        "call_id": f"{scene.scene_id}.{task}",
                        "scene_id": scene.scene_id,
                        "family": scene.family,
                        "task_type": task,
                    }
                )
    summary = tmp_path / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("family", "task_type", "number_of_scenes", "parse_rate", "accuracy"),
        )
        writer.writeheader()
        for task in ("T1", "T2", "T3", "T4", "T5", "T6"):
            writer.writerow(
                {
                    "family": "duplicate_encoding",
                    "task_type": task,
                    "number_of_scenes": 2,
                    "parse_rate": 0.0,
                    "accuracy": 0.0,
                }
            )
    gaps = tmp_path / "gaps.json"
    gaps.write_text(
        json.dumps(
            {
                "status": "PHASE_1_EXECUTED",
                "source_eligible_scenes": 3,
                "world_recoverable_scenes": 2,
                "model_calls": 12,
                "training_invoked": False,
                "rl_invoked": False,
                "subjective_success_threshold_applied": False,
            }
        ),
        encoding="utf-8",
    )

    evidence = validate_phase1_candidate_source(
        scenes,
        per_scene_path=per_scene,
        summary_path=summary,
        gaps_path=gaps,
        expected_source_scenes=3,
        expected_family_counts={"duplicate_encoding": 2},
    )

    assert evidence == {"number_of_scenes": 2, "number_of_records": 12}
