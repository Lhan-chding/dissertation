from __future__ import annotations

import dataclasses

from compbias.rl.small_natural_replay import (
    SmallNaturalReplayConfig,
    run_small_natural_replay,
)


def test_confirmatory_default_matches_v2_preregistration() -> None:
    config = SmallNaturalReplayConfig()
    assert config.semantic_state_count == 1_000
    assert config.visual_realization_count == 8_000
    assert config.n_mediators == 32
    assert config.n_forks == 32
    assert len(config.training_seeds) == 20
    assert len(config.task_families) == 5


def test_tiny_pil_cnn_replay_produces_three_estimands_and_crossed_risk() -> None:
    config = SmallNaturalReplayConfig(
        samples_per_family_per_split=1,
        realizations_per_semantic=2,
        n_mediators=4,
        n_forks=4,
        training_seeds=(0, 1),
        training_steps=2,
        batch_size=16,
        bootstrap_draws=1_000,
        confirmatory=False,
    )
    result = run_small_natural_replay(config)

    assert result.semantic_state_count == 25
    assert result.visual_realization_count == 50
    assert result.natural_mediator_count == 25 * 4 * 2
    assert result.forked_continuation_count == 25 * 4 * 4 * 2
    assert result.synthetic_mediator_count == 25 * 2
    assert result.error_families == (
        "bar_chart_aggregate",
        "count_transform",
        "digit_offset",
        "gauge_calibration",
        "relation_rule",
    )
    assert len(result.seed_results) == 2
    assert all(len(seed.by_error_family) == 5 for seed in result.seed_results)
    assert all(
        {row.error_family for row in seed.by_error_family} == set(result.error_families)
        for seed in result.seed_results
    )
    assert all(
        0.0 <= row.c_sel <= 1.0 and 0.0 <= row.c_fork <= 1.0 and 0.0 <= row.c_syn <= 1.0
        for seed in result.seed_results
        for row in seed.by_error_family
    )
    assert all(seed.crossed_risk.identity_residual < 1e-12 for seed in result.seed_results)
    assert all(seed.c_sel_error >= 0.0 for seed in result.seed_results)
    assert all(seed.c_fork_error >= 0.0 for seed in result.seed_results)
    assert all(seed.c_syn_error >= 0.0 for seed in result.seed_results)
    assert all(len(seed.model_sha256) == 64 for seed in result.seed_results)
    assert any(abs(seed.transport_gap) > 1e-4 for seed in result.seed_results)
    assert result.input_source == "cva_renderer_pil"
    assert result.model_path == "cnn_perceiver_to_image_blind_mlp_reasoner"


def test_tiny_replay_is_seed_reproducible() -> None:
    config = SmallNaturalReplayConfig(
        samples_per_family_per_split=1,
        realizations_per_semantic=2,
        n_mediators=2,
        n_forks=2,
        training_seeds=(3,),
        training_steps=1,
        batch_size=16,
        bootstrap_draws=1_000,
        confirmatory=False,
    )
    first = run_small_natural_replay(config)
    second = run_small_natural_replay(dataclasses.replace(config))
    assert first == second
