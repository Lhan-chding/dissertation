"""Registered formulas, literal update boundary, and finite-precision regressions."""

import copy
import math

import numpy as np
import pytest

from sr_f1.contract import reward_advantages as old_advantages
from sr_f2.advantages import (
    DEC_PRIME_ABSOLUTE_BOUND,
    gate_one,
    reward_advantages,
    zscore,
)


def scored(a=0, j=0, e=0, p=0, j6=None):
    return dict(A=a, J=j, E=e, p_read=p, J6=j if j6 is None else j6)


def batch():
    group = [
        scored(1, 1, 1, 1),
        scored(1, 1, 1, 1),
        scored(1, 0, 1, 0.75),
        scored(0, 0, 1, 0.5),
        scored(1, 0, 0, 0.75),
        scored(),
        scored(),
        scored(),
    ]
    return [copy.deepcopy(group) for _ in range(16)]


@pytest.mark.parametrize("successes", range(1, 8))
@pytest.mark.parametrize("failed_p", [0.0, 1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6, 1.0])
def test_49_constant_h_groups_remain_bitwise_J(successes, failed_p):
    j = np.array([1] * successes + [0] * (8 - successes))
    p = np.array([1.0] * successes + [failed_p] * (8 - successes))
    actual, branch = gate_one(j, np.ones(8), p)
    np.testing.assert_array_equal(actual, zscore(j))
    assert branch == "target_J_refined"


def test_2000_random_gate_groups_preserve_success_credit_sign_bound_and_zero_sum():
    rng = np.random.default_rng(71002)
    for _ in range(2000):
        j = np.zeros(8)
        j[rng.choice(8, size=rng.integers(1, 8), replace=False)] = 1
        e = np.where(j, 1, rng.integers(0, 2, size=8))
        p = np.where(j, 1, rng.integers(0, 9, size=8) / 8)
        expected = zscore(j)
        actual, branch = gate_one(j, e, p)
        adjustment = actual[j == 0] - expected[j == 0]
        np.testing.assert_array_equal(actual[j == 1], expected[j == 1])
        assert abs(adjustment.sum()) < 1e-9
        assert np.all(actual[j == 0] < 0)
        assert np.all(np.abs(adjustment) <= 0.25 * min(-expected[j == 0]) + 1e-9)
        assert abs(actual.sum()) < 1e-9
        assert branch == "target_J_refined"


def test_gate_skips_sub_tolerance_contrast():
    j = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 + 1e-10])
    np.testing.assert_array_equal(gate_one(j, np.ones(8), p)[0], zscore(j))


@pytest.mark.parametrize(
    "j,e,p,branch",
    [
        ([1] * 8, [1] * 8, [1] * 8, "all_joint_correct"),
        ([0] * 8, [0, 1] * 4, [0, 1] * 4, "evidence"),
        ([0] * 8, [1] * 8, [0, 1] * 4, "readings"),
        ([0] * 8, [1] * 8, [1] * 8, "no_semantic_contrast"),
    ],
)
def test_gate_fallback_branches(j, e, p, branch):
    actual, observed = gate_one(j, e, p)
    assert observed == branch
    expected = (
        zscore(e) if branch == "evidence" else zscore(p) if branch == "readings" else np.zeros(8)
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("step", [1, 7, 8, 12, 13, 96])
def test_early_registered_12_to_13_boundary_bitwise_matches_rule(step):
    scores = batch()
    expected_arm = "GATE" if step <= 12 else "J"
    actual, audit = reward_advantages("EARLY", scores, step=step)
    expected, _ = reward_advantages(expected_arm, scores, step=step)
    np.testing.assert_array_equal(actual, expected)
    assert audit["effective_arm"] == expected_arm
    assert audit["step"] == step


@pytest.mark.parametrize("arm", ["A", "J", "DEC"])
def test_reference_arms_reproduce_frozen_old_advantages(arm):
    scores = batch()
    actual, _ = reward_advantages(arm, scores, step=1)
    expected, _ = old_advantages(arm, scores)
    np.testing.assert_array_equal(actual, expected)


def test_j6_changes_only_topk_reward_and_supports_six_readings():
    scores = batch()
    families = ["CROSS", "THRESHOLD", "TOPK", "INTERVAL"] * 4
    for index, family in enumerate(families):
        if family == "TOPK":
            scores[index][4].update(J6=1, E6=1, P6=1)
    actual, _ = reward_advantages("J6", scores, step=1, families=families)
    baseline, _ = reward_advantages("J", scores, step=1)
    for index, family in enumerate(families):
        if family != "TOPK":
            np.testing.assert_array_equal(actual[index], baseline[index])
        else:
            assert not np.array_equal(actual[index], baseline[index])
            np.testing.assert_array_equal(
                actual[index], zscore([row["J6"] for row in scores[index]])
            )


def test_dec_prime_single_active_group_disproves_checklist_265_bound():
    scores = [[scored() for _ in range(8)] for _ in range(16)]
    scores[0][0] = scored(1, 1, 1, 1)
    actual, audit = reward_advantages("DEC_PRIME", scores, step=1)
    np.testing.assert_array_equal(actual[0], 2 * zscore(np.array([1] + [0] * 7)))
    assert np.max(np.abs(actual)) > 5.29 > 2.65
    assert np.max(np.abs(actual)) < 2 * math.sqrt(7) == DEC_PRIME_ABSOLUTE_BOUND
    assert not actual[1:].any()
    assert audit["batch_normalized"] is False
    assert audit["contrast_groups"] == 1
    old, old_audit = old_advantages("DEC", scores)
    np.testing.assert_array_equal(actual, np.array(old_audit["pre_batch_advantage"]))
    assert old[0, 0] > actual[0, 0]


def test_dec_prime_is_exact_registered_weighted_sum_without_batch_rescale():
    scores = batch()
    actual, audit = reward_advantages("DEC_PRIME", scores, step=96)
    expected = np.stack(
        [
            zscore([row["J"] for row in group])
            + 0.5 * zscore([row["E"] for row in group])
            + 0.5 * zscore([row["E"] * row["p_read"] for row in group])
            for group in scores
        ]
    )
    np.testing.assert_array_equal(actual, expected)
    assert audit["advantage_sum_squares"] == float((actual * actual).sum())


@pytest.mark.parametrize(
    "active", [(), ("TOPK",), ("TOPK", "THRESHOLD"), ("CROSS", "THRESHOLD", "TOPK", "INTERVAL")]
)
def test_adapt_family_routing_and_usage_audit(active):
    families = ["CROSS", "THRESHOLD", "TOPK", "INTERVAL"] * 4
    scores = batch()
    before = copy.deepcopy(scores)
    actual, audit = reward_advantages(
        "ADAPT", scores, step=9, families=families, active_families=active
    )
    for index, family in enumerate(families):
        key = "J" if family in active else "A"
        np.testing.assert_array_equal(actual[index], zscore([row[key] for row in scores[index]]))
        assert audit["group_reward_arms"][index] == key
    assert audit["semantic_groups"] == 4 * len(active)
    assert audit["semantic_group_fraction"] == len(active) / 4
    assert scores == before


@pytest.mark.parametrize("arm", ["EARLY", "J6", "DEC_PRIME", "ADAPT"])
def test_full_batch_required_and_constant_batches_have_zero_advantage(arm):
    kwargs = dict(step=1, families=["TOPK"] * 16)
    with pytest.raises(ValueError, match="16 x 8"):
        reward_advantages(arm, [[scored()] * 8], **kwargs)
    for row in (scored(), scored(1, 1, 1, 1)):
        actual, audit = reward_advantages(arm, [[row] * 8 for _ in range(16)], **kwargs)
        np.testing.assert_array_equal(actual, np.zeros((16, 8)))
        assert audit["zero_signal_step"] is True


@pytest.mark.parametrize("step", [0, 97, -1, True, 1.0])
def test_invalid_update_boundary_rejected(step):
    with pytest.raises(ValueError, match="One-based"):
        reward_advantages("EARLY", batch(), step=step)


def test_invalid_scores_and_routing_rejected():
    scores = batch()
    scores[0][0]["p_read"] = float("nan")
    with pytest.raises(ValueError):
        reward_advantages("DEC_PRIME", scores, step=1)
    with pytest.raises(ValueError, match="requires one family"):
        reward_advantages("ADAPT", batch(), step=1)
    with pytest.raises(ValueError, match="Unknown activated"):
        reward_advantages(
            "ADAPT", batch(), step=1, families=["TOPK"] * 16, active_families=["secret"]
        )
    scores = batch()
    scores[0][0]["J6"] = 0
    with pytest.raises(ValueError, match="non-TOPK"):
        reward_advantages("J6", scores, step=1, families=["CROSS"] * 16)
