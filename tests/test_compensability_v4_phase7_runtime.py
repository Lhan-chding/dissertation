from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path

import pytest

from compensability_v4.data.splits import DatasetSplit

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64

CHECKPOINTS = (
    "Base",
    "C0",
    "C1",
    "T",
    "Base_AnswerOnly_RL",
    "Recovery_LoRA_RecoveryOutcome_RL",
    "Recovery_LoRA_AnswerOnly_RL",
)
METRICS = (
    "stage1_visual_exact",
    "post_revision_world_exact",
    "reasoning_operator_exact",
    "final_answer_exact",
    "operator_invariant_correct",
    "genuine_recovery",
    "error_cancellation",
    "trace_mismatch",
    "error_mechanism_shift",
)


def _subject():
    return importlib.import_module("compensability_v4.qwen.phase7_runtime")


def _row(
    *,
    scene_id: str,
    checkpoint: str,
    family: str = "trend",
    ood_axis: str = "iid",
    split: DatasetSplit = DatasetSplit.SUPPORT_DEV,
    seed: int = 11,
    rollout_id: int = 0,
    stage1_visual_exact: bool = False,
    post_revision_world_exact: bool = True,
    reasoning_operator_exact: bool = True,
    final_answer_exact: bool = True,
    operator_invariant_correct: bool = False,
    genuine_recovery: bool = True,
    error_cancellation: bool = False,
    trace_mismatch: bool = False,
    error_mechanism_shift: bool = False,
) -> Mapping[str, object]:
    return {
        "scene_id": scene_id,
        "checkpoint": checkpoint,
        "checkpoint_sha256": SHA,
        "family": family,
        "split": split.value,
        "ood_axis": ood_axis,
        "seed": seed,
        "rollout_id": rollout_id,
        "image_sha256": "b" * 64,
        "stage1_visual_exact": stage1_visual_exact,
        "post_revision_world_exact": post_revision_world_exact,
        "reasoning_operator_exact": reasoning_operator_exact,
        "final_answer_exact": final_answer_exact,
        "operator_invariant_correct": operator_invariant_correct,
        "genuine_recovery": genuine_recovery,
        "error_cancellation": error_cancellation,
        "trace_mismatch": trace_mismatch,
        "error_mechanism_shift": error_mechanism_shift,
    }


def test_phase7_config_freezes_full_chain_metrics_ood_axes_and_objective_statistics() -> None:
    subject = _subject()
    config = subject.load_phase7_config(ROOT / "configs/recoverability/v4_phase_7.yaml")

    assert tuple(config.chain) == (
        "image",
        "natural_observation",
        "revision_or_recovery",
        "chart_operation",
        "final_answer",
    )
    assert tuple(config.required_metrics) == METRICS
    assert set(config.ood_axes) == {
        "iid",
        "style_ood",
        "constraint_graph_ood",
        "error_mechanism_ood",
    }
    assert tuple(config.checkpoints) == CHECKPOINTS
    assert config.bootstrap_confidence == 0.95
    assert config.bootstrap_resamples == 10_000
    assert config.tost_margin > 0.0
    assert config.confirmatory_evaluation_authorized is False
    assert config.subjective_success_threshold is None


def test_phase7_chain_row_requires_complete_trace_and_consistent_derived_labels() -> None:
    subject = _subject()
    valid = subject.Phase7ChainRow.from_mapping(_row(scene_id="scene-a", checkpoint="T"))
    assert valid.genuine_recovery is True

    missing = dict(_row(scene_id="scene-a", checkpoint="T"))
    missing.pop("reasoning_operator_exact")
    with pytest.raises(ValueError, match="reasoning_operator_exact"):
        subject.Phase7ChainRow.from_mapping(missing)

    contradictory_recovery = _row(
        scene_id="scene-a",
        checkpoint="T",
        stage1_visual_exact=True,
        genuine_recovery=True,
    )
    with pytest.raises(ValueError, match="genuine recovery"):
        subject.Phase7ChainRow.from_mapping(contradictory_recovery)

    contradictory_cancellation = _row(
        scene_id="scene-a",
        checkpoint="T",
        post_revision_world_exact=True,
        final_answer_exact=True,
        genuine_recovery=False,
        error_cancellation=True,
    )
    with pytest.raises(ValueError, match="error cancellation"):
        subject.Phase7ChainRow.from_mapping(contradictory_cancellation)


def test_phase7_support_dev_diagnostics_are_allowed_but_confirm_splits_fail_closed() -> None:
    subject = _subject()
    support_row = subject.Phase7ChainRow.from_mapping(_row(scene_id="support-a", checkpoint="Base"))
    assert subject.validate_phase7_rows(
        (support_row,), confirmatory_evaluation_authorized=False
    ) == (support_row,)

    for split in (
        DatasetSplit.CONFIRM_IID,
        DatasetSplit.CONFIRM_STYLE_OOD,
        DatasetSplit.CONFIRM_CONSTRAINT_OOD,
    ):
        confirm_row = subject.Phase7ChainRow.from_mapping(
            _row(scene_id=f"confirm-{split.value}", checkpoint="Base", split=split)
        )
        with pytest.raises(ValueError, match="confirmatory"):
            subject.validate_phase7_rows((confirm_row,), confirmatory_evaluation_authorized=False)


def test_phase7_summary_reports_every_metric_globally_by_family_and_by_ood_axis() -> None:
    subject = _subject()
    rows = tuple(
        subject.Phase7ChainRow.from_mapping(
            _row(
                scene_id=scene_id,
                checkpoint="Base",
                family=family,
                ood_axis=axis,
                final_answer_exact=scene_id != "s1",
                post_revision_world_exact=scene_id == "s0",
                reasoning_operator_exact=scene_id != "s1",
                genuine_recovery=scene_id == "s0",
                error_cancellation=scene_id == "s2",
                trace_mismatch=scene_id == "s2",
                error_mechanism_shift=axis == "error_mechanism_ood",
            )
        )
        for scene_id, family, axis in (
            ("s0", "trend", "iid"),
            ("s1", "cross_series", "style_ood"),
            ("s2", "duplicate_encoding", "error_mechanism_ood"),
        )
    )

    summary = subject.summarize_phase7(
        rows,
        bootstrap_resamples=100,
        bootstrap_seed=7,
        tost_margin=0.02,
    )

    assert summary["status"] == "PHASE_7_MULTIMODAL_DIAGNOSTIC_EVALUATED"
    assert summary["scene_is_statistical_unit"] is True
    assert summary["rollout_is_statistical_unit"] is False
    assert summary["subjective_success_threshold_applied"] is False
    assert summary["confirmatory_data_used"] is False
    assert set(summary["metrics"]) == set(METRICS)
    assert set(summary["by_checkpoint"]) == {"Base"}
    for metric in METRICS:
        global_result = summary["metrics"][metric]["global"]
        assert {
            "estimate",
            "ci_low",
            "ci_high",
            "confidence",
            "number_of_scenes",
        } <= set(global_result)
        assert set(summary["metrics"][metric]["by_family"]) == {
            "cross_series",
            "duplicate_encoding",
            "trend",
        }
        assert set(summary["metrics"][metric]["by_ood_axis"]) == {
            "error_mechanism_ood",
            "iid",
            "style_ood",
        }
        assert metric in summary["by_checkpoint"]["Base"]["metrics"]


def test_phase7_rollouts_are_clustered_to_scenes_and_never_inflate_sample_size() -> None:
    subject = _subject()
    rows = tuple(
        subject.Phase7ChainRow.from_mapping(
            _row(
                scene_id=scene_id,
                checkpoint="Base",
                rollout_id=rollout_id,
                final_answer_exact=success,
                post_revision_world_exact=False,
                reasoning_operator_exact=success,
                genuine_recovery=False,
                error_cancellation=success,
                trace_mismatch=success,
            )
        )
        for scene_id, outcomes in (("s0", (True, False)), ("s1", (True, True)))
        for rollout_id, success in enumerate(outcomes)
    )

    summary = subject.summarize_phase7(
        rows,
        bootstrap_resamples=100,
        bootstrap_seed=7,
        tost_margin=0.02,
    )

    answer = summary["metrics"]["final_answer_exact"]["global"]
    assert answer["number_of_scenes"] == 2
    assert answer["number_of_rollouts"] == 4
    assert answer["estimate"] == pytest.approx(0.75)


def test_phase7_registered_paired_effects_include_holm_tost_and_seed_variability() -> None:
    subject = _subject()
    outcomes = {
        "Base": (False, False),
        "C0": (False, False),
        "C1": (False, True),
        "T": (True, True),
        "Base_AnswerOnly_RL": (False, True),
        "Recovery_LoRA_RecoveryOutcome_RL": (True, True),
        "Recovery_LoRA_AnswerOnly_RL": (True, True),
    }
    rows = tuple(
        subject.Phase7ChainRow.from_mapping(
            _row(
                scene_id=scene_id,
                checkpoint=checkpoint,
                seed=seed,
                rollout_id=0,
                stage1_visual_exact=success,
                post_revision_world_exact=success,
                reasoning_operator_exact=success,
                final_answer_exact=success,
                genuine_recovery=False,
            )
        )
        for checkpoint, scene_outcomes in outcomes.items()
        for seed in (11, 17)
        for scene_id, success in zip(("s0", "s1"), scene_outcomes, strict=True)
    )

    summary = subject.summarize_phase7(
        rows,
        bootstrap_resamples=100,
        bootstrap_seed=7,
        tost_margin=0.02,
    )

    required_effects = {
        "T_minus_C0",
        "T_minus_C1",
        "seeded_rl_minus_base_rl",
        "recovery_reward_rl_minus_answer_only_rl",
    }
    assert required_effects <= set(summary["registered_effects"])
    assert set(summary["registered_effects_by_family"]) == {"trend"}
    assert set(summary["registered_effects_by_ood_axis"]) == {"iid"}
    for effect_name in required_effects:
        effect = summary["registered_effects"][effect_name]
        assert {
            "estimate",
            "ci_low",
            "ci_high",
            "paired_scene_count",
            "two_sided_sign_flip_p_value",
            "holm_adjusted_p_value",
            "tost",
        } <= set(effect)
        assert effect["tost"]["margin"] == 0.02
    assert set(summary["seed_level_variability"]) == {"11", "17"}


def test_phase7_execution_manifest_is_hash_bound_and_has_no_subjective_gate() -> None:
    subject = _subject()
    config = subject.load_phase7_config(ROOT / "configs/recoverability/v4_phase_7.yaml")
    source_hashes = {
        "dataset_records": "1" * 64,
        "support_dev": "2" * 64,
        "phase4_summary": "3" * 64,
        "phase5_summary": "4" * 64,
        "phase6_evaluation": "5" * 64,
    }
    checkpoint_hashes = {name: f"{index:x}" * 64 for index, name in enumerate(CHECKPOINTS, 1)}

    manifest = subject.build_phase7_execution_manifest(
        config=config,
        source_sha256=source_hashes,
        checkpoint_sha256=checkpoint_hashes,
        config_sha256="6" * 64,
        package_lock_sha256="7" * 64,
    )

    assert manifest["status"] == "PHASE_7_MULTIMODAL_EXECUTION_MANIFEST_PREPARED"
    assert manifest["source_sha256"] == source_hashes
    assert manifest["checkpoint_sha256"] == checkpoint_hashes
    assert manifest["confirmatory_evaluation_authorized"] is False
    assert manifest["support_dev_diagnostic_authorized"] is True
    assert manifest["subjective_success_threshold_applied"] is False
    assert "success_threshold" not in manifest
