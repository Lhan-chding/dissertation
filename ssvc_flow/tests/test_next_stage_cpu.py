from pathlib import Path

import pytest

from src.constraint_solver import solve
from src.group_support_audit import bank_advantage_statistics
from src.joint_advantage_reference import compare_lambdas, finite_difference
from src.next_stage_common import dry_run_plan, load_yaml
from src.statistics_scene_cluster import scene_cluster_intervals, weighted_prompt_metrics


def test_next_stage_yaml_and_dry_run_are_locked():
    config = load_yaml(Path(__file__).parents[1] / "configs/next_stage.yaml")
    plan = dry_run_plan(config)
    assert config["model"]["id"] == "Qwen/Qwen3.5-9B"
    assert plan["generation_counts"]["R4_train"] == 4096
    assert plan["automatic_sbatch_submission"] is False


def test_group_hypergeometric_support_and_advantage_boundary():
    result = bank_advantage_statistics((1, 3, 3, 1), 8, 0.01)
    assert 0 <= result["no_X"] <= 1
    assert result["three_level_X_B_I"] > 0
    all_valid = compare_lambdas(["X", "S", "W", "X"], [0, 1])
    assert all_valid["lambdas"]["1"]["delta_from_lambda0_l2"] == pytest.approx(0)


def test_derivative_matches_central_difference():
    categories = ["X", "S", "W", "I", "W", "S"]
    analytic = compare_lambdas(categories, [0.25])["lambdas"]["0.25"]["dA_dlambda"]
    numeric = finite_difference(categories, 0.25)
    assert analytic == pytest.approx(numeric, abs=2e-7)


def test_prompt_macro_and_pooled_metrics_are_explicit():
    rows = [{"prompt_id": "a", "base_scene_id": "s1", "category": "X"}] * 8 + [
        {"prompt_id": "b", "base_scene_id": "s2", "category": "I"}
    ] * 8
    metrics = weighted_prompt_metrics(rows)
    assert metrics["pX"] == pytest.approx(0.5)
    assert metrics["v"] == pytest.approx(0.5)
    assert metrics["qX_pool"] == pytest.approx(1.0)


def test_scene_cluster_requires_paired_interfaces():
    rows = []
    for scene in ("s1", "s2"):
        for interface in ("IMAGE_CUE_FRESH", "SYMBOLIC_FRESH"):
            rows.append(
                {
                    "prompt_id": scene + interface,
                    "base_scene_id": scene,
                    "family": "cross_series",
                    "interface": interface,
                    "n_total": 2,
                    "n_X": 1,
                    "n_S": 0,
                    "n_W": 1,
                    "n_I": 0,
                }
            )
    result = scene_cluster_intervals(rows, repeats=20)
    assert result["status"] == "ESTIMATED"
    assert result["unit"] == "base_scene"


def test_independent_solver_finds_unique_repair():
    truth = [4, 8, 12, 16]
    cue = {"family": "trend"}
    observed = [4, 8, 12, 17]
    assert solve(observed, cue) == [truth]
