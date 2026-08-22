from __future__ import annotations

import math

import pytest

from compensability_v5.study_c2.contrast_rank import reward_contrast
from compensability_v5.study_c2.gradient_audit import shared_gradient_diagnostics
from compensability_v5.study_c2.group_metrics import (
    choose_group_size,
    exact_shortcut_group_probability,
    summarize_group,
)


def test_group_metrics_distinguish_generic_and_corrective_excitation() -> None:
    summary = summarize_group(("X", "S", "F", "U"))

    assert summary == {
        "AIGR": True,
        "SIGR": True,
        "RDGR": True,
        "ESGR": True,
        "counts": {"X": 1, "S": 1, "F": 1, "U": 1},
    }
    expected = 1 - 0.9**8 - 0.8**8 + 0.7**8
    assert math.isclose(exact_shortcut_group_probability(0.1, 0.2, 8), expected)


def test_k_selection_maximizes_per_rollout_exact_shortcut_efficiency() -> None:
    selected = choose_group_size(
        [{"p_X": 0.1, "p_S": 0.2}, {"p_X": 0.05, "p_S": 0.1}],
        candidates=(8, 16, 32),
    )
    efficiencies = selected["efficiency_by_k"]
    best = max(efficiencies.values())
    assert efficiencies[selected["selected_k"]] == best
    assert selected["selected_k"] == min(k for k, value in efficiencies.items() if value == best)


def test_contrast_rank_zero_implies_identical_centered_advantages_and_gradients() -> None:
    identical = reward_contrast([1, 0, 0, 0], [1, 0, 0, 0])
    diagnostics = shared_gradient_diagnostics(
        state_rewards=[1, 0, 0, 0],
        answer_rewards=[1, 0, 0, 0],
        score_vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
    )

    assert identical["normalized_contrast_strength"] == 0.0
    assert diagnostics["reward_hamming_distance"] == 0
    assert diagnostics["gradient_difference_norm"] == 0.0
    assert diagnostics["gradient_cosine"] == pytest.approx(1.0)


def test_shortcut_creates_rank_two_reward_and_gradient_contrast() -> None:
    contrast = reward_contrast([1, 0, 0, 0], [1, 1, 0, 0])
    diagnostics = shared_gradient_diagnostics(
        state_rewards=[1, 0, 0, 0],
        answer_rewards=[1, 1, 0, 0],
        score_vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
    )

    assert contrast["contrast_rank"] == 2
    assert contrast["second_singular_value"] > 0
    assert diagnostics["reward_hamming_distance"] == 1
    assert diagnostics["gradient_difference_norm"] > 0
