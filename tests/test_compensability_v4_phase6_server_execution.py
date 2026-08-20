from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from compensability_v4.data.splits import DatasetSplit
from compensability_v4.qwen.phase6_runtime import (
    PHASE6_LOCKED_PATHS,
    verify_phase6_package_lock,
)
from compensability_v4.training.phase6 import Phase6Example, Phase6Variant, RewardKind

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/v4/11_prepare_phase6_rl.py"
SCRIPT_DIR = ROOT / "scripts/v4"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_phase6_manifest_script_is_importable_and_execute_gated() -> None:
    module = _load("test_phase6_prepare_manifest", SCRIPT_PATH)

    assert callable(module.main)

    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert (
        "BLOCKED: Phase 6 RL manifest preparation requires explicit --execute." in string_constants
    )
    assert "--policy-support-summary-sha256" in source


def test_phase6_manifest_script_does_not_construct_rl_training() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "Trainer(" not in source
    assert "GRPOTrainer" not in source
    assert "optimizer.step" not in source
    assert "torch.optim" not in source


def test_phase6_execution_scripts_cover_data_grpo_resume_and_evaluation() -> None:
    prepare = _load("test_phase6_prepare_data", SCRIPT_DIR / "12_prepare_phase6_rl_data.py")
    train = _load("test_phase6_train", SCRIPT_DIR / "13_train_phase6_grpo.py")
    evaluate = _load("test_phase6_evaluate", SCRIPT_DIR / "14_evaluate_phase6_rl.py")

    assert all(callable(module.main) for module in (prepare, train, evaluate))
    for filename in (
        "12_prepare_phase6_rl_data.py",
        "13_train_phase6_grpo.py",
        "14_evaluate_phase6_rl.py",
    ):
        source = (SCRIPT_DIR / filename).read_text(encoding="utf-8")
        assert "--execution-manifest-sha256" in source
        assert "PHASE6_LOCKED_PATHS" in source
    training_source = (SCRIPT_DIR / "13_train_phase6_grpo.py").read_text(encoding="utf-8")
    training_tree = ast.parse(training_source)
    assert any(
        node.id == "GRPOTrainer" for node in ast.walk(training_tree) if isinstance(node, ast.Name)
    )
    assert "--preflight-only" in training_source
    assert "COMPBIAS_V4_PHASE6_RL_ACK" in training_source
    assert "resume_from_checkpoint" in training_source
    assert "reward_trace.jsonl" in training_source
    assert "RewardTraceCheckpointCallback" in training_source
    assert "cannot verify all required frozen model components" in training_source
    assert "package_lock_sha256" in training_source
    evaluation_source = (SCRIPT_DIR / "14_evaluate_phase6_rl.py").read_text(encoding="utf-8")
    assert "support-dev errors drifted from the execution manifest" in evaluation_source
    assert "Recovery_LoRA baseline drifted from the execution manifest" in evaluation_source
    assert "execution_manifest_sha256" in evaluation_source
    prepare_source = (SCRIPT_DIR / "12_prepare_phase6_rl_data.py").read_text(encoding="utf-8")
    assert "Phase 5 summary no longer matches the execution manifest" in prepare_source
    assert "Phase 5 source hashes drifted from the execution manifest" in prepare_source


def test_phase6_reward_trace_has_stable_call_groups_and_checkpoint_restore(
    tmp_path: Path,
) -> None:
    train = _load("test_phase6_train_trace", SCRIPT_DIR / "13_train_phase6_grpo.py")
    example = Phase6Example(
        example_id="scene-a:answer_only",
        scene_id="scene-a",
        family="trend",
        split=DatasetSplit.NATURAL_ERROR_SUPPORT_TRAIN,
        reward_kind=RewardKind.ANSWER_ONLY,
        prompt="Return the integer answer only.",
        expected_completion="17",
        truth=(2, 3, 5, 7),
        observed=(2, 3, 8, 7),
        answer=17,
        operation="sum",
    )
    trace_path = tmp_path / "run/reward_trace.jsonl"
    reward = train._reward_function(
        examples=(example,),
        variant=Phase6Variant.BASE_ANSWER_ONLY,
        trace_path=trace_path,
    )
    assert reward(["17"] * 8, example_id=[example.example_id]) == [1.0] * 8
    assert reward(["0"] * 8, example_id=[example.example_id]) == [0.0] * 8
    rows = tuple(json.loads(line) for line in trace_path.read_text().splitlines())
    assert {row["reward_call_index"] for row in rows[:8]} == {0}
    assert {row["reward_call_index"] for row in rows[8:]} == {1}

    checkpoint = tmp_path / "run/checkpoint-16"
    checkpoint.mkdir()
    snapshot = checkpoint / "reward_trace.jsonl"
    snapshot.write_text("\n".join(json.dumps(row) for row in rows[:8]) + "\n")
    train._restore_reward_trace(str(checkpoint), trace_path)
    assert trace_path.read_text() == snapshot.read_text()
    with pytest.raises(RuntimeError, match="stale reward trace"):
        train._restore_reward_trace(None, trace_path)


def test_phase6_evaluation_reports_paired_intervals_family_effects_and_holm() -> None:
    evaluate = _load("test_phase6_evaluate_summary", SCRIPT_DIR / "14_evaluate_phase6_rl.py")
    checkpoints = (
        "Base",
        Phase6Variant.BASE_ANSWER_ONLY.value,
        "Recovery_LoRA",
        Phase6Variant.RECOVERY_OUTCOME.value,
        Phase6Variant.RECOVERY_ANSWER_ONLY.value,
    )
    rows = tuple(
        {
            "checkpoint": checkpoint,
            "scene_id": scene_id,
            "family": family,
            "greedy_exact_world_recovery": checkpoint == Phase6Variant.RECOVERY_OUTCOME.value,
            "greedy_observation_copy": checkpoint == "Base",
            "sample_exact_world_recovery": [checkpoint == Phase6Variant.RECOVERY_OUTCOME.value] * 2,
            "sample_observation_copy": [checkpoint == "Base"] * 2,
            "answer_exact": checkpoint
            in (
                Phase6Variant.BASE_ANSWER_ONLY.value,
                Phase6Variant.RECOVERY_ANSWER_ONLY.value,
            ),
        }
        for checkpoint in checkpoints
        for scene_id, family in (("s0", "trend"), ("s1", "cross_series"))
    )

    summary = evaluate._summary(
        rows,
        2,
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )

    effect = summary["registered_effects"]["recovery_outcome_rl_effect_from_recovery_on_world"]
    assert effect["estimate"] == 1.0
    assert effect["ci_low"] == 1.0
    assert "holm_adjusted_p_value" in effect
    assert set(summary["registered_effects_by_family"]) == {"cross_series", "trend"}
    assert summary["scene_is_statistical_unit"] is True


def test_phase6_package_lock_closes_manifest_freeze_surface() -> None:
    digest = verify_phase6_package_lock(
        lock_path=ROOT / "configs/recoverability/v4/server_package_lock_phase_6.yaml",
        repository_root=ROOT,
        expected_paths=PHASE6_LOCKED_PATHS,
    )

    assert len(digest) == 64
