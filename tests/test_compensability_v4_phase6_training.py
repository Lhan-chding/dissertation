from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from compensability_v4.data.splits import DatasetSplit
from compensability_v4.schemas.observation import NaturalObservation
from compensability_v4.training.phase6 import (
    Phase6Example,
    Phase6TrainingConfig,
    Phase6Variant,
    RewardGroupTrace,
    RewardKind,
    build_phase6_examples,
    score_phase6_completion,
    summarize_reward_groups,
    validate_phase5_policy_support,
)


def _scene(scene_id: str = "natural-a") -> dict[str, object]:
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "split": DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN.value,
        "semantic_scene_id": f"semantic-{scene_id}",
        "numeric_table_id": f"numeric-{scene_id}",
        "constraint_graph_id": f"graph-{scene_id}",
        "truth": [2, 3, 5, 7],
        "facts": [
            {"type": "known_value", "index": 1, "value": 3},
            {"type": "known_value", "index": 2, "value": 5},
            {"type": "known_value", "index": 3, "value": 7},
            {"type": "pair_sum", "left_index": 0, "right_index": 1, "total": 5},
        ],
        "resized_height": 280,
        "resized_width": 280,
        "image_path": f"images/{scene_id}.png",
    }


def _observation(scene_id: str = "natural-a") -> dict[str, object]:
    return {
        "schema_version": 1,
        "observation_id": f"obs-{scene_id}",
        "scene_id": scene_id,
        "split": DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN.value,
        "model_snapshot_sha256": "a" * 64,
        "observed_values": [2, 3, 8, 7],
        "raw_output": "2,3,8,7",
        "parse_success": True,
        "error_count": 1,
        "error_index": 2,
    }


def _record(scene_id: str = "natural-a") -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "family": "cross_series",
        "chart_type": "grouped_bar",
        "operation": "sum",
        "values": [2, 3, 5, 7],
        "question": "What is the total?",
        "answer": 17,
        "image": f"images/{scene_id}.png",
    }


def test_phase6_config_and_variant_matrix_are_frozen() -> None:
    config = Phase6TrainingConfig.from_mapping(
        {
            "precision": "bf16",
            "learning_rate": 1.0e-6,
            "max_steps": 64,
            "group_size": 8,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 0,
            "kl_beta": 0.04,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_prompt_length": 512,
            "max_completion_length": 32,
            "checkpoint_steps": 16,
            "seed": 2026082006,
            "use_vllm": False,
        }
    )

    assert config.group_size == 8
    assert tuple(Phase6Variant) == (
        Phase6Variant.BASE_ANSWER_ONLY,
        Phase6Variant.RECOVERY_OUTCOME,
        Phase6Variant.RECOVERY_ANSWER_ONLY,
    )
    assert Phase6Variant.BASE_ANSWER_ONLY.initial_checkpoint == "Base"
    assert Phase6Variant.RECOVERY_OUTCOME.initial_checkpoint == "T"
    assert Phase6Variant.RECOVERY_ANSWER_ONLY.reward_kind is RewardKind.ANSWER_ONLY
    with pytest.raises(FrozenInstanceError):
        config.max_steps = 1  # type: ignore[misc]


def test_phase6_training_rows_are_natural_train_only_and_keep_both_rewards() -> None:
    rows = build_phase6_examples(
        natural_scenes=(_scene(),),
        natural_observations=(_observation(),),
        dataset_records=(_record(),),
    )

    assert len(rows) == 2
    assert {row.reward_kind for row in rows} == {
        RewardKind.RECOVERY_OUTCOME,
        RewardKind.ANSWER_ONLY,
    }
    assert all(row.split is DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN for row in rows)
    recovery = next(row for row in rows if row.reward_kind is RewardKind.RECOVERY_OUTCOME)
    answer = next(row for row in rows if row.reward_kind is RewardKind.ANSWER_ONLY)
    assert recovery.expected_completion == "2,3,5,7"
    assert recovery.observed == (2, 3, 8, 7)
    assert answer.expected_completion == "17"
    assert "What is the total?" in answer.prompt

    leaked = _scene()
    leaked["split"] = DatasetSplit.SUPPORT_DEV.value
    with pytest.raises(ValueError, match="natural_error_support_train"):
        build_phase6_examples(
            natural_scenes=(leaked,),
            natural_observations=(_observation(),),
            dataset_records=(_record(),),
        )


def test_phase6_accepts_the_frozen_phase4_observation_schema_without_a_split_field() -> None:
    observation = NaturalObservation(
        observation_id="phase4-stage1-natural-a",
        scene_id="natural-a",
        observed_values=(2, 3, 8, 7),
        error_index=2,
        stage1_model_hash="a" * 64,
        image_grid_thw=(1, 20, 20),
        visual_token_count=100,
    ).to_mapping()

    assert "split" not in observation
    rows = build_phase6_examples(
        natural_scenes=(_scene(),),
        natural_observations=(observation,),
        dataset_records=(_record(),),
    )

    assert len(rows) == 2
    assert all(row.split is DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN for row in rows)


def test_phase6_rewards_distinguish_world_recovery_answer_and_copy() -> None:
    recovery = Phase6Example.from_mapping(
        next(
            row.to_mapping()
            for row in build_phase6_examples(
                natural_scenes=(_scene(),),
                natural_observations=(_observation(),),
                dataset_records=(_record(),),
            )
            if row.reward_kind is RewardKind.RECOVERY_OUTCOME
        )
    )
    answer = Phase6Example.from_mapping(
        next(
            row.to_mapping()
            for row in build_phase6_examples(
                natural_scenes=(_scene(),),
                natural_observations=(_observation(),),
                dataset_records=(_record(),),
            )
            if row.reward_kind is RewardKind.ANSWER_ONLY
        )
    )

    recovered = score_phase6_completion(recovery, "2,3,5,7")
    copied = score_phase6_completion(recovery, "2,3,8,7")
    answer_hit = score_phase6_completion(answer, "17")
    answer_world = score_phase6_completion(answer, "2,3,5,7")

    assert (recovered.reward, recovered.exact_world_recovery) == (1.0, True)
    assert (copied.reward, copied.observation_copy) == (0.0, True)
    assert (answer_hit.reward, answer_hit.answer_exact) == (1.0, True)
    assert (answer_world.reward, answer_world.exact_world_recovery) == (0.0, True)


def test_phase6_group_summary_reports_signal_diagnostics_without_thresholds() -> None:
    traces = (
        RewardGroupTrace("g0", "s0", Phase6Variant.RECOVERY_OUTCOME, (0.0,) * 8, 0.1, 1.0),
        RewardGroupTrace("g1", "s1", Phase6Variant.RECOVERY_OUTCOME, (1.0,) * 8, 0.2, 0.9),
        RewardGroupTrace(
            "g2",
            "s2",
            Phase6Variant.RECOVERY_OUTCOME,
            (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            0.3,
            0.8,
            exact_world_recovery_count=4,
            observation_copy_count=2,
        ),
    )

    summary = summarize_reward_groups(traces, group_size=8)

    assert summary["all_zero_group_rate"] == pytest.approx(1 / 3)
    assert summary["all_one_group_rate"] == pytest.approx(1 / 3)
    assert summary["non_degenerate_group_rate"] == pytest.approx(1 / 3)
    assert summary["mean_group_reward_variance"] == pytest.approx(1 / 12)
    assert summary["exact_world_recovery_rate"] == pytest.approx(4 / 24)
    assert summary["observation_copy_rate"] == pytest.approx(2 / 24)
    assert summary["subjective_success_threshold_applied"] is False


def test_phase6_requires_completed_phase5_evidence_without_numeric_gate() -> None:
    payload = {
        "schema_version": 1,
        "status": "PHASE_5_POLICY_SUPPORT_EXECUTED",
        "number_of_held_out_natural_errors": 32,
        "number_of_checkpoint_scene_rows": 128,
        "sampling_rollouts_per_scene": 16,
        "informative_group_size": 8,
        "subjective_success_threshold_applied": False,
        "confirmatory_data_used": False,
        "training_invoked": False,
        "rl_invoked": False,
    }

    validate_phase5_policy_support(payload)
    payload["subjective_success_threshold_applied"] = True
    with pytest.raises(ValueError, match="threshold"):
        validate_phase5_policy_support(payload)
