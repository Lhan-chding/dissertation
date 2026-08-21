from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from compensability_v4.data.splits import DatasetSplit
from compensability_v4.schemas.observation import NaturalObservation
from compensability_v4.schemas.scene import RecoveryScene

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
OOD_AXES = (
    "iid",
    "style_ood",
    "constraint_graph_ood",
    "error_mechanism_ood",
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
PRIOR_SEEDS = frozenset(
    {
        2026082005,
        2026082006,
        2026082007,
        2026082101,
        2026082102,
    }
)
AXIS_SPLITS = {
    "iid": "confirm_iid",
    "style_ood": "confirm_style_ood",
    "constraint_graph_ood": "confirm_constraint_ood",
    "error_mechanism_ood": "confirm_error_mechanism_ood",
}


def _subject():
    return importlib.import_module("compensability_v4.qwen.phase8_confirm_runtime")


def _scene(
    scene_id: str,
    *,
    split: DatasetSplit,
    semantic_scene_id: str | None = None,
    numeric_table_id: str | None = None,
    constraint_graph_id: str | None = None,
) -> RecoveryScene:
    return RecoveryScene(
        scene_id=scene_id,
        split=split,
        semantic_scene_id=semantic_scene_id or f"semantic-{scene_id}",
        numeric_table_id=numeric_table_id or f"numeric-{scene_id}",
        constraint_graph_id=constraint_graph_id or f"graph-{scene_id}",
        truth=(2, 3, 5, 7),
        facts=(
            {"type": "known_value", "index": 0, "value": 2},
            {"type": "known_value", "index": 3, "value": 7},
        ),
        resized_height=280,
        resized_width=280,
        image_path=f"images/{scene_id}.png",
    )


def _observation(
    scene_id: str,
    *,
    observed_values: tuple[int, int, int, int],
    error_index: int,
) -> NaturalObservation:
    return NaturalObservation(
        observation_id=f"observation-{scene_id}",
        scene_id=scene_id,
        observed_values=observed_values,
        error_index=error_index,
        stage1_model_hash=SHA,
        image_grid_thw=(1, 20, 20),
        visual_token_count=100,
    )


def _result_row(
    *,
    scene_id: str,
    checkpoint: str,
    ood_axis: str = "iid",
    family: str = "trend",
    seed: int = 31,
    success: bool = True,
) -> Mapping[str, object]:
    return {
        "scene_id": scene_id,
        "checkpoint": checkpoint,
        "checkpoint_sha256": SHA,
        "family": family,
        "split": AXIS_SPLITS[ood_axis],
        "ood_axis": ood_axis,
        "seed": seed,
        "rollout_id": 0,
        "image_sha256": "b" * 64,
        "stage1_visual_exact": False,
        "post_revision_world_exact": success,
        "reasoning_operator_exact": success,
        "final_answer_exact": success,
        "operator_invariant_correct": False,
        "genuine_recovery": success,
        "error_cancellation": False,
        "trace_mismatch": not success,
        "error_mechanism_shift": ood_axis == "error_mechanism_ood",
        "free_generation_answer_exact": success,
        "deterministic_chain_answer_exact": success,
        "answer_source": "genuine_recovery" if success else "unresolved",
    }


def test_phase8_config_freezes_new_seeds_counts_axes_checkpoints_and_statistics() -> None:
    subject = _subject()
    config = subject.load_phase8_config(ROOT / "configs/recoverability/v4_phase_8.yaml")

    assert set(config.ood_axes) == set(OOD_AXES)
    assert tuple(config.checkpoints) == CHECKPOINTS
    assert tuple(config.required_metrics) == METRICS
    assert set(config.fixed_scene_counts) == set(OOD_AXES)
    assert all(type(count) is int and count > 0 for count in config.fixed_scene_counts.values())
    assert config.generation_seed not in PRIOR_SEEDS
    assert config.evaluation_seed not in PRIOR_SEEDS
    assert config.bootstrap_seed not in PRIOR_SEEDS
    assert len({config.generation_seed, config.evaluation_seed, config.bootstrap_seed}) == 3
    assert config.bootstrap_confidence == 0.95
    assert config.bootstrap_resamples == 10_000
    assert config.tost_margin > 0.0
    assert config.confirmatory_evaluation_authorized is True
    assert config.require_explicit_ack is True
    assert config.subjective_success_threshold is None


def test_phase8_registers_error_mechanism_confirm_split() -> None:
    assert DatasetSplit.CONFIRM_ERROR_MECHANISM_OOD.value == "confirm_error_mechanism_ood"


def test_phase8_confirm_scenes_are_triply_isolated_from_every_preconfirm_regime() -> None:
    subject = _subject()
    prior = (
        _scene("legacy", split=DatasetSplit.LEGACY_DIAGNOSTIC),
        _scene("symbolic", split=DatasetSplit.SYMBOLIC_SUPPORT_TRAIN),
        _scene("natural", split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN),
        _scene("support", split=DatasetSplit.SUPPORT_DEV),
    )
    confirm = (
        _scene("confirm-iid", split=DatasetSplit.CONFIRM_IID),
        _scene("confirm-style", split=DatasetSplit.CONFIRM_STYLE_OOD),
        _scene("confirm-constraint", split=DatasetSplit.CONFIRM_CONSTRAINT_OOD),
        _scene(
            "confirm-error",
            split=DatasetSplit.CONFIRM_ERROR_MECHANISM_OOD,
        ),
    )
    assert subject.validate_phase8_isolation(confirm, prior) == confirm

    for field in ("semantic_scene_id", "numeric_table_id", "constraint_graph_id"):
        payload = confirm[0].to_mapping()
        payload[field] = getattr(prior[0], field)
        leaked = (RecoveryScene.from_mapping(payload), *confirm[1:])
        with pytest.raises(ValueError, match=field):
            subject.validate_phase8_isolation(leaked, prior)


def test_phase8_freeze_uses_fixed_candidate_counts_and_includes_all_natural_stage1_errors() -> None:
    subject = _subject()
    scenes = (
        _scene("error-a", split=DatasetSplit.CONFIRM_IID),
        _scene("exact-b", split=DatasetSplit.CONFIRM_IID),
        _scene("error-c", split=DatasetSplit.CONFIRM_STYLE_OOD),
        _scene("exact-d", split=DatasetSplit.CONFIRM_STYLE_OOD),
    )
    observations = (
        _observation("error-a", observed_values=(2, 4, 5, 7), error_index=1),
        _observation("exact-b", observed_values=(2, 3, 5, 7), error_index=0),
        _observation("error-c", observed_values=(2, 3, 6, 7), error_index=2),
        _observation("exact-d", observed_values=(2, 3, 5, 7), error_index=0),
    )

    frozen = subject.freeze_phase8_natural_errors(
        scenes,
        observations,
        fixed_scene_counts={"iid": 2, "style_ood": 2},
    )

    assert {example.scene_id for example in frozen.examples} == {"error-a", "error-c"}
    assert frozen.candidate_scene_count == 4
    assert frozen.natural_error_count == 2
    assert frozen.all_natural_stage1_errors_included is True
    assert frozen.selection_uses_model_outcome_threshold is False

    with pytest.raises(ValueError, match="fixed scene count"):
        subject.freeze_phase8_natural_errors(
            scenes,
            observations,
            fixed_scene_counts={"iid": 3, "style_ood": 2},
        )


def test_phase8_freeze_keeps_multi_position_natural_stage1_errors() -> None:
    subject = _subject()
    scene = _scene("multi-error", split=DatasetSplit.CONFIRM_ERROR_MECHANISM_OOD)
    observation = _observation(
        "multi-error",
        observed_values=(1, 4, 5, 7),
        error_index=0,
    )

    frozen = subject.freeze_phase8_natural_errors(
        (scene,),
        (observation,),
        fixed_scene_counts={"error_mechanism_ood": 1},
    )

    assert frozen.natural_error_count == 1
    assert frozen.examples[0].error_indices == (0, 1)


def test_phase8_result_rows_require_both_answer_endpoints_nine_metrics_and_answer_source() -> None:
    subject = _subject()
    valid = subject.Phase8ConfirmRow.from_mapping(_result_row(scene_id="confirm-a", checkpoint="T"))
    assert valid.free_generation_answer_exact is True
    assert valid.deterministic_chain_answer_exact is True
    assert valid.answer_source.value == "genuine_recovery"

    for missing in (*METRICS, "free_generation_answer_exact", "deterministic_chain_answer_exact"):
        payload = dict(_result_row(scene_id="confirm-a", checkpoint="T"))
        payload.pop(missing)
        with pytest.raises(ValueError, match=missing):
            subject.Phase8ConfirmRow.from_mapping(payload)

    contradiction = dict(_result_row(scene_id="confirm-a", checkpoint="T", success=False))
    contradiction["answer_source"] = "genuine_recovery"
    with pytest.raises(ValueError, match="answer source"):
        subject.Phase8ConfirmRow.from_mapping(contradiction)

    reread = dict(_result_row(scene_id="confirm-reread", checkpoint="T"))
    reread.update(
        {
            "stage1_visual_exact": True,
            "genuine_recovery": False,
            "answer_source": "visual_reread",
        }
    )
    assert subject.Phase8ConfirmRow.from_mapping(reread).answer_source.value == "visual_reread"


def test_phase8_summary_analyzes_metrics_endpoints_and_answer_sources_by_registered_strata() -> (
    None
):
    subject = _subject()
    rows = tuple(
        subject.Phase8ConfirmRow.from_mapping(
            _result_row(
                scene_id=f"scene-{axis}",
                checkpoint="T",
                ood_axis=axis,
                family="trend" if index % 2 == 0 else "cross_series",
                success=index != 1,
            )
        )
        for index, axis in enumerate(OOD_AXES)
    )

    summary = subject.summarize_phase8(
        rows,
        bootstrap_resamples=100,
        bootstrap_seed=41,
        tost_margin=0.02,
    )

    assert summary["status"] == "PHASE_8_CONFIRMATORY_EVALUATED"
    assert summary["confirmatory_data_used"] is True
    assert summary["scene_is_statistical_unit"] is True
    assert summary["rollout_is_statistical_unit"] is False
    assert summary["subjective_success_threshold_applied"] is False
    assert set(summary["metrics"]) == set(METRICS)
    assert set(summary["answer_endpoints"]) == {
        "free_generation_answer_exact",
        "deterministic_chain_answer_exact",
    }
    assert set(summary["answer_source_counts"]) == {"genuine_recovery", "unresolved"}
    assert set(summary["by_checkpoint"]) == {"T"}
    for metric in (*METRICS, *summary["answer_endpoints"]):
        metric_summary = summary["metrics"].get(metric) or summary["answer_endpoints"][metric]
        assert {
            "estimate",
            "ci_low",
            "ci_high",
            "confidence",
            "number_of_scenes",
        } <= set(metric_summary["global"])
        assert set(metric_summary["by_family"]) == {"cross_series", "trend"}
        assert set(metric_summary["by_ood_axis"]) == set(OOD_AXES)


def test_phase8_registered_effects_use_paired_scenes_holm_tost_and_all_seven_checkpoints() -> None:
    subject = _subject()
    rows = tuple(
        subject.Phase8ConfirmRow.from_mapping(
            _result_row(
                scene_id=scene_id,
                checkpoint=checkpoint,
                seed=seed,
                success=checkpoint
                in {
                    "T",
                    "Recovery_LoRA_RecoveryOutcome_RL",
                    "Recovery_LoRA_AnswerOnly_RL",
                },
            )
        )
        for checkpoint in CHECKPOINTS
        for seed in (31, 37)
        for scene_id in ("paired-a", "paired-b")
    )

    summary = subject.summarize_phase8(
        rows,
        bootstrap_resamples=100,
        bootstrap_seed=41,
        tost_margin=0.02,
    )

    assert set(summary["by_checkpoint"]) == set(CHECKPOINTS)
    assert set(summary["seed_level_variability"]) == {"31", "37"}
    required_effects = {
        "T_minus_C0",
        "T_minus_C1",
        "seeded_rl_minus_base_rl",
        "recovery_reward_rl_minus_answer_only_rl",
    }
    assert required_effects <= set(summary["registered_effects"])
    assert set(summary["registered_effects_by_answer_endpoint"]) == {
        "free_generation_answer_exact",
        "deterministic_chain_answer_exact",
    }
    for endpoint_effects in summary["registered_effects_by_answer_endpoint"].values():
        assert required_effects <= set(endpoint_effects)
    for effect in summary["registered_effects"].values():
        assert {
            "estimate",
            "ci_low",
            "ci_high",
            "paired_scene_count",
            "two_sided_sign_flip_p_value",
            "holm_adjusted_p_value",
            "tost",
        } <= set(effect)
        assert effect["paired_scene_count"] == 2
        assert effect["tost"]["margin"] == 0.02


def test_phase8_manifest_binds_frozen_data_checkpoints_and_authorization_ack() -> None:
    subject = _subject()
    config = subject.load_phase8_config(ROOT / "configs/recoverability/v4_phase_8.yaml")
    source_hashes = {
        "legacy_diagnostic": "1" * 64,
        "symbolic_support_train": "2" * 64,
        "natural_error_support_train": "3" * 64,
        "support_dev": "4" * 64,
        "phase7_evaluation": "5" * 64,
        "confirm_scenes": "6" * 64,
        "confirm_observations": "7" * 64,
        "confirm_summary": "8" * 64,
        "confirm_image_bundle": "b" * 64,
        "prompt_config": "c" * 64,
    }
    checkpoint_hashes = {name: f"{index:x}" * 64 for index, name in enumerate(CHECKPOINTS, 1)}

    with pytest.raises(PermissionError, match="ACK"):
        subject.build_phase8_execution_manifest(
            config=config,
            source_sha256=source_hashes,
            checkpoint_sha256=checkpoint_hashes,
            config_sha256="9" * 64,
            package_lock_sha256="a" * 64,
            authorization_ack=None,
        )

    manifest = subject.build_phase8_execution_manifest(
        config=config,
        source_sha256=source_hashes,
        checkpoint_sha256=checkpoint_hashes,
        config_sha256="9" * 64,
        package_lock_sha256="a" * 64,
        authorization_ack="I_UNDERSTAND_THIS_CONSUMES_THE_FROZEN_PHASE_8_CONFIRM_SET",
    )
    assert manifest["status"] == "PHASE_8_CONFIRMATORY_EXECUTION_MANIFEST_PREPARED"
    assert manifest["source_sha256"] == source_hashes
    assert manifest["checkpoint_sha256"] == checkpoint_hashes
    assert manifest["confirmatory_evaluation_authorized"] is True
    assert manifest["authorization_ack_verified"] is True
    assert manifest["subjective_success_threshold_applied"] is False


def test_phase8_config_and_isolation_validation_fail_closed(tmp_path: Path) -> None:
    subject = _subject()
    source = ROOT / "configs/recoverability/v4_phase_8.yaml"
    payload = json.loads(source.read_text(encoding="utf-8"))
    bad_seed = tmp_path / "bad-seed.json"
    bad_seed.write_text(
        json.dumps({**payload, "seeds": {**payload["seeds"], "evaluation": 2026082103}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="distinct"):
        subject.load_phase8_config(bad_seed)
    with pytest.raises(ValueError, match="must not be empty"):
        subject.validate_phase8_isolation((), ())
    prior = (_scene("prior", split=DatasetSplit.SUPPORT_DEV),)
    with pytest.raises(ValueError, match="registered confirm splits"):
        subject.validate_phase8_isolation(prior, ())
    confirm = (_scene("confirm", split=DatasetSplit.CONFIRM_IID),)
    with pytest.raises(ValueError, match="prior regimes"):
        subject.validate_phase8_isolation(confirm, confirm)
    with pytest.raises(ValueError, match="scene_id"):
        subject.validate_phase8_isolation(
            confirm, (_scene("confirm", split=DatasetSplit.SUPPORT_DEV),)
        )


def test_phase8_freeze_and_result_row_structural_errors_fail_closed() -> None:
    subject = _subject()
    scene = _scene("scene-a", split=DatasetSplit.CONFIRM_IID)
    observation = _observation("scene-a", observed_values=(2, 4, 5, 7), error_index=0)
    with pytest.raises(ValueError, match="candidate scenes"):
        subject.freeze_phase8_natural_errors((), (), fixed_scene_counts={"iid": 1})
    with pytest.raises(ValueError, match="unknown OOD"):
        subject.freeze_phase8_natural_errors(
            (scene,), (observation,), fixed_scene_counts={"unknown": 1}
        )
    with pytest.raises(ValueError, match="duplicate scene"):
        subject.freeze_phase8_natural_errors(
            (scene,), (observation, observation), fixed_scene_counts={"iid": 1}
        )
    with pytest.raises(ValueError, match="error_index"):
        subject.freeze_phase8_natural_errors(
            (scene,), (observation,), fixed_scene_counts={"iid": 1}
        )
    with pytest.raises(ValueError, match="must not be empty"):
        subject.validate_phase8_rows(())
    valid = subject.Phase8ConfirmRow.from_mapping(_result_row(scene_id="scene-a", checkpoint="T"))
    with pytest.raises(ValueError, match="identity"):
        subject.validate_phase8_rows((valid, valid))


def test_phase8_summary_and_manifest_boundary_types_fail_closed() -> None:
    subject = _subject()
    valid = subject.Phase8ConfirmRow.from_mapping(_result_row(scene_id="scene-a", checkpoint="T"))
    with pytest.raises(ValueError, match="positive integer"):
        subject.summarize_phase8((valid,), bootstrap_resamples=0)
    with pytest.raises(TypeError, match="integer"):
        subject.summarize_phase8((valid,), bootstrap_seed=1.5)
    with pytest.raises(ValueError, match="positive and finite"):
        subject.summarize_phase8((valid,), tost_margin=float("inf"))
    config = subject.load_phase8_config(ROOT / "configs/recoverability/v4_phase_8.yaml")
    with pytest.raises(ValueError, match="source hashes"):
        subject.build_phase8_execution_manifest(
            config=config,
            source_sha256={},
            checkpoint_sha256={name: SHA for name in CHECKPOINTS},
            config_sha256=SHA,
            package_lock_sha256=SHA,
            authorization_ack="I_UNDERSTAND_THIS_CONSUMES_THE_FROZEN_PHASE_8_CONFIRM_SET",
        )
    with pytest.raises(ValueError, match="seven checkpoints"):
        subject.build_phase8_execution_manifest(
            config=config,
            source_sha256={name: SHA for name in subject._SOURCE_KEYS},
            checkpoint_sha256={"Base": SHA},
            config_sha256=SHA,
            package_lock_sha256=SHA,
            authorization_ack="I_UNDERSTAND_THIS_CONSUMES_THE_FROZEN_PHASE_8_CONFIRM_SET",
        )
