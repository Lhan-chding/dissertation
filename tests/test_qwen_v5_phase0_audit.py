from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import pytest

from compbias.v5.audit_v4_raw import (
    answer_fiber_statistics,
    candidate_margin_summary,
    capability_chain_summary,
    confirm_error_cardinality_summary,
    interface_revision_summary,
    phase8_transition_counts,
    support_budget_summary,
)
from scripts.v5.audit_v4_raw import main as audit_main


def test_capability_chain_summary_groups_by_family_and_task() -> None:
    rows = (
        {
            "family": "cross_series",
            "task_type": "T1",
            "parse_success": True,
            "is_correct": False,
        },
        {
            "family": "cross_series",
            "task_type": "T1",
            "parse_success": True,
            "is_correct": True,
        },
        {
            "family": "trend",
            "task_type": "T6",
            "parse_success": False,
            "is_correct": False,
        },
    )

    summary = capability_chain_summary(rows)

    assert summary["cross_series"]["T1"]["scene_count"] == 2
    assert summary["cross_series"]["T1"]["parse_rate"] == pytest.approx(1.0)
    assert summary["cross_series"]["T1"]["accuracy"] == pytest.approx(0.5)
    assert summary["trend"]["T6"]["parse_rate"] == pytest.approx(0.0)


def test_candidate_margin_summary_computes_family_level_validity_effects() -> None:
    rows = (
        {
            "scene_id": "scene-a",
            "family": "cross_series",
            "cue_condition": "valid_cue",
            "margin_true_observed": -0.5,
        },
        {
            "scene_id": "scene-a",
            "family": "cross_series",
            "cue_condition": "sham_cue",
            "margin_true_observed": -0.8,
        },
        {
            "scene_id": "scene-b",
            "family": "cross_series",
            "cue_condition": "valid_cue",
            "margin_true_observed": -1.5,
        },
        {
            "scene_id": "scene-b",
            "family": "cross_series",
            "cue_condition": "sham_cue",
            "margin_true_observed": -1.4,
        },
    )

    summary = candidate_margin_summary(rows)

    assert summary["cross_series"]["valid_minus_sham_target_margin_mean"] == pytest.approx(
        0.1
    )
    assert summary["cross_series"]["scene_count"] == 2


def test_interface_revision_summary_uses_i3_exact_world_revision() -> None:
    rows = (
        {
            "family": "duplicate_encoding",
            "interface": "I3_same_conversation_visual_revision",
            "cue_condition": "valid_cue",
            "true_world": [1, 2, 3, 4],
            "output_world": [1, 2, 3, 4],
        },
        {
            "family": "duplicate_encoding",
            "interface": "I3_same_conversation_visual_revision",
            "cue_condition": "no_cue",
            "true_world": [1, 2, 3, 4],
            "output_world": [0, 2, 3, 4],
        },
    )

    summary = interface_revision_summary(rows)

    assert summary["duplicate_encoding"]["valid_exact_revision_rate"] == pytest.approx(1.0)
    assert summary["duplicate_encoding"]["no_cue_exact_revision_rate"] == pytest.approx(0.0)
    assert summary["duplicate_encoding"]["valid_minus_no_cue"] == pytest.approx(1.0)


def test_answer_fiber_statistics_enumerates_one_edit_worlds() -> None:
    rows = (
        {
            "scene_id": "scene-a",
            "observed": [8, 11, 5, 17],
            "truth": [9, 11, 5, 17],
            "operation": "difference",
            "answer": -2,
        },
        {
            "scene_id": "scene-b",
            "observed": [15, 18, 17, 14],
            "truth": [16, 18, 17, 14],
            "operation": "max_minus_min",
            "answer": 4,
        },
    )

    summary = answer_fiber_statistics(rows)

    assert summary["scene_count"] == 2
    assert summary["singleton_count"] == 0
    assert summary["max_size"] >= summary["median_size"] >= 1
    assert summary["mean_size"] == pytest.approx(6.0)


def test_support_budget_summary_counts_rows_per_variant() -> None:
    rows = (
        {"variant": "C0_format_only"},
        {"variant": "C0_format_only"},
        {"variant": "T_constraint_recovery"},
    )

    summary = support_budget_summary(rows)

    assert summary["counts_by_variant"] == {
        "C0_format_only": 2,
        "T_constraint_recovery": 1,
    }
    assert summary["budget_ratio_T_to_C0"] == pytest.approx(0.5)


def test_confirm_error_cardinality_summary_tracks_multi_error_and_domain_drift() -> None:
    rows = (
        {"error_indices": [0], "observed": [2, 3, 4, 5]},
        {"error_indices": [1, 2], "observed": [1, 3, 4, 5]},
        {"error_indices": [0, 1, 2, 3], "observed": [2, 3, 4, 25]},
    )

    summary = confirm_error_cardinality_summary(rows, in_domain=range(2, 19))

    assert summary["error_count_histogram"] == {"1": 1, "2": 1, "4": 1}
    assert summary["out_of_domain_scene_count"] == 2


def test_phase8_transition_counts_compares_seeded_rl_against_t_checkpoint() -> None:
    rows = (
        {
            "scene_id": "scene-a",
            "checkpoint": "T",
            "free_generation_answer_exact": True,
            "post_revision_world_exact": False,
        },
        {
            "scene_id": "scene-a",
            "checkpoint": "Recovery_LoRA_AnswerOnly_RL",
            "free_generation_answer_exact": False,
            "post_revision_world_exact": False,
        },
        {
            "scene_id": "scene-a",
            "checkpoint": "Recovery_LoRA_RecoveryOutcome_RL",
            "free_generation_answer_exact": True,
            "post_revision_world_exact": True,
        },
        {
            "scene_id": "scene-b",
            "checkpoint": "T",
            "free_generation_answer_exact": False,
            "post_revision_world_exact": False,
        },
        {
            "scene_id": "scene-b",
            "checkpoint": "Recovery_LoRA_AnswerOnly_RL",
            "free_generation_answer_exact": False,
            "post_revision_world_exact": False,
        },
        {
            "scene_id": "scene-b",
            "checkpoint": "Recovery_LoRA_RecoveryOutcome_RL",
            "free_generation_answer_exact": True,
            "post_revision_world_exact": True,
        },
    )

    summary = phase8_transition_counts(rows)

    assert summary["free_generation_answer_exact"]["Recovery_LoRA_AnswerOnly_RL"]["lost"] == 1
    assert (
        summary["free_generation_answer_exact"]["Recovery_LoRA_RecoveryOutcome_RL"]["gained"] == 1
    )
    assert summary["post_revision_world_exact"]["Recovery_LoRA_RecoveryOutcome_RL"]["gained"] == 2


def test_phase0_cli_writes_derived_analysis_json(tmp_path: Path) -> None:
    capability = io.StringIO()
    writer = csv.DictWriter(
        capability,
        fieldnames=[
            "scene_id",
            "family",
            "task_type",
            "parse_success",
            "is_correct",
        ],
    )
    writer.writeheader()
    writer.writerow(
        {
            "scene_id": "scene-a",
            "family": "cross_series",
            "task_type": "T1",
            "parse_success": "True",
            "is_correct": "True",
        }
    )
    candidate_rows = "\n".join(
        (
            json.dumps(
                {
                    "scene_id": "scene-a",
                    "family": "cross_series",
                    "cue_condition": "valid_cue",
                    "margin_true_observed": -0.5,
                }
            ),
            json.dumps(
                {
                    "scene_id": "scene-a",
                    "family": "cross_series",
                    "cue_condition": "sham_cue",
                    "margin_true_observed": -0.7,
                }
            ),
        )
    )
    interface_rows = json.dumps(
        {
            "scene_id": "scene-a",
            "family": "cross_series",
            "interface": "I3_same_conversation_visual_revision",
            "cue_condition": "valid_cue",
            "true_world": [9, 11, 5, 17],
            "output_world": [9, 11, 5, 17],
        }
    )
    rl_rows = json.dumps(
        {
            "scene_id": "scene-a",
            "answer": 20,
            "operation": "sum",
            "observed": [8, 11, 5, 17],
            "truth": [9, 11, 5, 17],
        }
    )
    support_rows = json.dumps({"variant": "C0_format_only"})
    confirm_rows = json.dumps({"error_indices": [0], "observed": [8, 11, 5, 17]})
    phase8_rows = "\n".join(
        (
            json.dumps(
                {
                    "scene_id": "scene-a",
                    "checkpoint": "T",
                    "free_generation_answer_exact": True,
                    "post_revision_world_exact": False,
                }
            ),
            json.dumps(
                {
                    "scene_id": "scene-a",
                    "checkpoint": "Recovery_LoRA_AnswerOnly_RL",
                    "free_generation_answer_exact": False,
                    "post_revision_world_exact": False,
                }
            ),
        )
    )

    raw_tar = tmp_path / "raw.tar.gz"
    with tarfile.open(raw_tar, "w:gz") as archive:
        payloads = {
            "artifacts/v4/capability_chain/per_scene.csv": capability.getvalue(),
            "artifacts/v4/candidate_scoring/per_scene.jsonl": candidate_rows,
            "artifacts/v4/interface_ladder/per_scene.jsonl": interface_rows,
            "artifacts/v4/rl/data/answer_only.jsonl": rl_rows,
            "artifacts/v4/training/support.jsonl": support_rows,
            "artifacts/v4/phase8/confirm_data/selection_trace.jsonl": confirm_rows,
            "artifacts/v4/phase8/evaluation/per_scene.jsonl": phase8_rows,
        }
        for path, text in payloads.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(path)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    facts_path = tmp_path / "facts.md"
    facts_path.write_text("fact bundle placeholder\n", encoding="utf-8")
    output_path = tmp_path / "derived_analysis.json"

    assert (
        audit_main(
            [
                "--raw-archive",
                str(raw_tar),
                "--fact-file",
                str(facts_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["inputs"]["raw_archive"]["sha256"]
    assert payload["sections"]["capability_chain"]["cross_series"]["T1"]["accuracy"] == (
        pytest.approx(1.0)
    )
    assert payload["sections"]["answer_fiber_statistics"]["scene_count"] == 1
