"""Behavioral mathematical contracts, written before implementation."""

import itertools
import json
import math

import numpy as np
import pytest

from src.decoding_audit import enumerate_decoders, toy_decoder_audit, weighted_pooled_purity
from src.sensitivity import advantage_derivative, audit_derivative, normalized_advantages
from src.tabular_flow import (
    count_vectors,
    finite_group_flow,
    group_updates,
    mean_field_flow,
    multinomial_weights,
    run_benchmark,
    self_derived_nonclosure_example,
    validity_purity_derivative,
)


@pytest.mark.parametrize("k", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("p", [[0.02, 0.18, 0.3, 0.5], [0.0, 0.3, 0.2, 0.5], [1.0, 0.0, 0.0, 0.0]])
def test_multinomial_support_and_probability(k, p):
    counts = count_vectors(k)
    assert counts.shape == (math.comb(k + 3, 3), 4)
    assert np.all(counts.sum(axis=1) == k)
    weights = multinomial_weights(counts, p)
    assert weights.sum() == pytest.approx(1.0, abs=2e-13)
    assert weights @ counts / k == pytest.approx(p, abs=2e-13)


def test_exact_finite_step_equals_independent_ordered_draw_enumeration():
    p, rewards, k, eta = np.array([0.15, 0.2, 0.35, 0.3]), [3, 1, 1, 0], 3, 0.3
    result = finite_group_flow(p, rewards, k, eta, epsilon=1e-4)
    probability = np.zeros(4)
    second = np.zeros((4, 4))
    eq = 0.0
    for draws in itertools.product(range(4), repeat=k):
        np.bincount(draws, minlength=4)
        values = np.array(rewards)[list(draws)]
        advantage = (values - values.mean()) / np.sqrt(values.var() + 1e-8)
        update = np.bincount(draws, weights=advantage, minlength=4) / k
        mass = float(np.prod(p[list(draws)]))
        shifted = p * np.exp(eta * update)
        shifted /= shifted.sum()
        probability += mass * shifted
        second += mass * np.outer(update, update)
        eq += mass * shifted[0] / shifted[:3].sum()
    assert result["exact_p_next"] == pytest.approx(probability, abs=1e-13)
    assert result["Xi_K"] == pytest.approx(second, abs=1e-13)
    assert result["expected_q_next"] == pytest.approx(eq, abs=1e-13)
    h = np.array(result["H_K"])
    softmax_mean = p * np.exp(eta * h)
    softmax_mean /= softmax_mean.sum()
    assert np.max(abs(probability - softmax_mean)) > 1e-5
    assert abs(result["expected_q_next"] - result["pooled_q_next"]) > 1e-6


def test_mass_conservation_second_order_and_step_scaling():
    p, reward = [0.02, 0.18, 0.3, 0.5], [3, 1, 1, 0]
    errors = []
    for eta in [0.05, 0.01, 0.001]:
        row = finite_group_flow(p, reward, 8, eta)
        assert sum(row["H_K"]) == pytest.approx(0.0, abs=1e-13)
        assert sum(row["delta_p"]) == pytest.approx(0.0, abs=1e-13)
        errors.append((row["first_order_error"], row["second_order_error"]))
        assert row["second_order_error"] < row["first_order_error"]
    assert errors[1][0] < errors[0][0] / 20
    assert errors[2][0] < errors[1][0] / 90
    assert errors[1][1] < errors[0][1] / 100


def test_reward_ordering_ties_and_epsilon_zero():
    p = [0.15, 0.2, 0.35, 0.3]
    a = finite_group_flow(p, [3, 2, 1, 0], 2, 0.05, epsilon=0)
    b = finite_group_flow(p, [101, 40, 0.2, -6], 2, 0.05, epsilon=0)
    assert a["H_K"] == pytest.approx(b["H_K"], abs=1e-13)
    tied = finite_group_flow(p, [2, 0, 0, 0], 2, 0.05, epsilon=0)
    untied = finite_group_flow(p, [2.1, 0.1, 0.1, 0], 2, 0.05, epsilon=0)
    assert not np.allclose(tied["H_K"], untied["H_K"])
    zero = finite_group_flow(p, [1, 1, 1, 1], 8, 0.05, epsilon=0)
    assert zero["H_K"] == [0.0, 0.0, 0.0, 0.0]
    assert zero["zero_variance_probability"] == pytest.approx(1.0)
    smooth_a = finite_group_flow(p, [3, 2, 1, 0], 2, 0.05, epsilon=0.2)
    smooth_b = finite_group_flow(p, [101, 40, 0.2, -6], 2, 0.05, epsilon=0.2)
    assert not np.allclose(smooth_a["H_K"], smooth_b["H_K"])


def test_all_valid_shift_and_degenerate_support():
    p = [0.2, 0.3, 0.5, 0.0]
    baseline = finite_group_flow(p, [2, 0, 0, 0], 8, 0.05)
    valid = finite_group_flow(p, [3, 1, 1, 0], 8, 0.05)
    assert baseline["exact_p_next"] == pytest.approx(valid["exact_p_next"], abs=1e-14)
    for p in [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]:
        row = finite_group_flow(p, [3, 1, 1, 0], 4, 0.01, epsilon=0)
        assert row["exact_p_next"] == pytest.approx(p)
        assert np.isfinite(np.array(row["Xi_K"])).all()
    assert finite_group_flow([0.0, 0.0, 0.0, 1.0], [1, 1, 1, 0], 2, 0.01)["expected_q_next"] is None


def test_validity_q_derivative_and_meanfield_limit():
    p = np.array([0.6, 0.1, 0.2, 0.1])
    row = finite_group_flow(p, [1, 1, 1, 0], 16, 1e-5)
    derivative = validity_purity_derivative(p, 16)
    assert derivative == pytest.approx(
        (row["expected_q_next"] - p[0] / sum(p[:3])) / 1e-5, abs=2e-7
    )
    target = np.array(mean_field_flow(p, [3, 1, 1, 0])["H_infinity"])
    errors = [
        np.linalg.norm(np.array(finite_group_flow(p, [3, 1, 1, 0], k, 0.01)["H_K"]) - target)
        for k in [4, 16, 32]
    ]
    assert errors[-1] < errors[0]


def test_self_derived_nonclosure_is_honestly_named_and_equal_start_probability():
    result = self_derived_nonclosure_example()
    assert result["source"] == "self_derived_fixture"
    assert result["theory_review_verified"] is False
    assert result["p_minus"] == pytest.approx(result["p_plus"])
    assert not np.allclose(result["dp_minus"], result["dp_plus"])


@pytest.mark.parametrize(
    "p,k,eta,eps",
    [
        ([0.2, 0.2, 0.2, 0.2], 2, 0.01, 0.1),
        ([0.1, -0.1, 0.5, 0.5], 2, 0.01, 0.1),
        ([0.25] * 4, 0, 0.01, 0.1),
        ([0.25] * 4, 2, -0.1, 0.1),
        ([0.25] * 4, 2, 0.1, -1),
    ],
)
def test_invalid_flow_arguments(p, k, eta, eps):
    with pytest.raises(ValueError):
        finite_group_flow(p, [3, 1, 1, 0], k, eta, epsilon=eps)


@pytest.mark.parametrize("lam", [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
def test_lambda_derivative_matches_autograd_and_finite_difference(lam):
    primary = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
    valid = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    audit = audit_derivative(primary, valid, lam)
    assert audit["autograd_max_abs_error"] < 1e-12
    for row in audit["finite_differences"]:
        if row["step"] <= 1e-3:
            assert row["max_abs_error"] < 1e-5
    # Both constructions reduce to normalized primary rewards at lambda=0.
    if lam == 0:
        assert audit["joint_vs_separate_norm"] == 0
    else:
        assert audit["joint_vs_separate_norm"] > 1e-3


def test_zero_variance_sensitivity_and_tie_undefined():
    assert normalized_advantages([1, 1, 1], epsilon=0).tolist() == [0, 0, 0]
    assert advantage_derivative([2, 2], [1, 1], 0.1, epsilon=0).tolist() == [0, 0]
    with pytest.raises(ValueError, match=r"tie|undefined"):
        advantage_derivative([0, 0], [1, 0], 0, epsilon=0)
    assert np.isfinite(advantage_derivative([0, 0], [1, 0], 0, epsilon=1e-4)).all()


def test_toy_rejection_and_fsa_enumeration_distinguishes_distributions():
    result = toy_decoder_audit()
    assert result["rejection_distribution"] == pytest.approx(
        result["free_conditional_distribution"]
    )
    assert result["total_variation_fsa_vs_rejection"] > 0.1
    assert result["parser_language_agrees_with_fsa"] is True
    assert result["scope"] == "toy_mathematical_validation"


def test_sequence_enumeration_and_correct_prompt_weighting():
    transitions = {
        (): {"a": 0.5, "b": 0.5},
        ("a",): {"x": 0.9, "y": 0.1},
        ("b",): {"x": 0.1, "y": 0.9},
    }
    results = enumerate_decoders(transitions, {("a", "x"), ("b", "x")}, 2)
    assert results["free_valid_probability"] == pytest.approx(0.5)
    assert sorted(results["rejection_distribution"].values()) == pytest.approx([0.1, 0.9])
    assert sorted(results["fsa_distribution"].values()) == pytest.approx([0.5, 0.5])
    assert weighted_pooled_purity([0.9, 0.1], [0.1, 0.9], [0.5, 0.5]) == pytest.approx(0.18)


def test_benchmark_writes_traceable_artifacts(tmp_path):
    config = {
        "tabular": {
            "K": [2, 4],
            "eta": [0.001, 0.01],
            "probabilities": [[0.15, 0.2, 0.35, 0.3]],
            "epsilon": 1e-4,
        }
    }
    result = run_benchmark(config, tmp_path)
    for name in [
        "status.json",
        "report.md",
        "manifest.json",
        "finite_group_flow.csv",
        "derivative_audit.json",
        "decoder_toy_audit.json",
    ]:
        assert (tmp_path / name).is_file()
    assert result["scope"] == "toy_mathematical_validation"
    assert json.loads((tmp_path / "status.json").read_text())["phase"] == "P2"


def test_roundoff_cannot_create_a_signal_in_tied_groups():
    updates, zero = group_updates(count_vectors(3), [0.1, 0.1, 0.1, 0.1], epsilon=0)
    assert np.array_equal(updates, np.zeros_like(updates))
    assert zero.all()
    varied, _ = group_updates(count_vectors(16), [3.0, 1.0, 1.0, 0.0])
    assert np.max(np.abs(varied.sum(axis=1))) < 1e-13


@pytest.mark.parametrize(
    "counts",
    [[], [[-1, 0, 0, 3]], [[0.5, 0.5, 0, 1]], [[1, 0, 0, 1], [1, 0, 0, 2]], [[0, 0, 0, 0]]],
)
def test_invalid_multinomial_counts(counts):
    with pytest.raises(ValueError):
        multinomial_weights(counts, [0.25] * 4)


@pytest.mark.parametrize("values,eps", [([], 0), ([float("nan")], 0), ([1, 2], -1)])
def test_invalid_sensitivity_arguments(values, eps):
    with pytest.raises(ValueError):
        normalized_advantages(values, eps)
    with pytest.raises(ValueError):
        advantage_derivative([1, 2], [1], 0.1)


def test_gradient_comparison_uses_same_sequence_scores():
    primary, valid = [2.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]
    scores = np.array([[0.2, 0.3], [0.1, 0.2], [-0.3, 0.1], [0.4, -0.5]])
    result = audit_derivative(primary, valid, 1.0, score_vectors=scores)
    assert result["joint_gradient"] == pytest.approx(
        np.array(result["joint_advantages"]) @ scores / 4
    )
    assert result["channelwise_gradient"] == pytest.approx(
        np.array(result["channelwise_standardized_sum"]) @ scores / 4
    )
    assert not np.allclose(result["joint_gradient"], result["channelwise_gradient"])
    with pytest.raises(ValueError):
        audit_derivative(primary, valid, 1.0, score_vectors=[[1, 2]])


@pytest.mark.parametrize(
    "transition,language,steps",
    [
        ({(): {"a": 0.9}}, {("a",)}, 1),
        ({(): {"a": float("nan")}}, {("a",)}, 1),
        ({}, set(), 1),
        ({}, {("a",)}, 0),
    ],
)
def test_invalid_decoder_inputs(transition, language, steps):
    with pytest.raises(ValueError):
        enumerate_decoders(transition, language, steps)


def test_unreachable_valid_language_does_not_invent_accepts():
    result = enumerate_decoders({(): {"a": 1.0}}, {("b",)}, 1)
    assert result["rejection_distribution"] == {}
    assert result["rejection_expected_proposals_per_accept"] is None
    assert result["fsa_completion_probability"] == 0
    assert weighted_pooled_purity([0.0, 0.0], [0.1, 0.8], [0.5, 0.5]) is None
    assert validity_purity_derivative([0.0, 0.0, 0.0, 1.0], 2) is None


@pytest.mark.parametrize(
    "rates,purity,weights",
    [
        ([], [], []),
        ([0.5], [0.2, 0.3], [1]),
        ([1.1], [0.5], [1]),
        ([0.5], [-0.2], [1]),
        ([0.5], [0.2], [0.2]),
    ],
)
def test_invalid_pooled_purity_inputs(rates, purity, weights):
    with pytest.raises(ValueError):
        weighted_pooled_purity(rates, purity, weights)


def test_cli_entrypoints_and_dry_run(tmp_path, capsys):
    from src.decoding_audit import main as decoder_main
    from src.sensitivity import main as sensitivity_main
    from src.tabular_flow import main as flow_main

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tabular": {
                    "K": [2],
                    "eta": [0.01],
                    "meanfield_group_sizes": [],
                    "probabilities": [[0.15, 0.2, 0.35, 0.3]],
                }
            }
        )
    )
    flow_main(["--config", str(config_path), "--out", str(tmp_path / "dry"), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["rollout_count"] == 0
    assert not (tmp_path / "dry").exists()
    flow_main(["--config", str(config_path), "--out", str(tmp_path / "flow")])
    sensitivity_main(["--out", str(tmp_path / "sensitivity")])
    decoder_main(["--out", str(tmp_path / "decoder")])
    assert json.loads((tmp_path / "flow/status.json").read_text())["status"] == "PARTIAL"
    assert json.loads((tmp_path / "sensitivity/status.json").read_text())["phase"] == "P6a"
    assert json.loads((tmp_path / "decoder/status.json").read_text())["phase"] == "P8a"
