"""Small deterministic contracts; large repeated campaigns run on the server."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.modeling_v3.covariance_pilot import (
    bootstrap_crossfit,
    cost_matched_estimates,
    crossfit_estimate,
    estimate_pilot,
    oracle_b,
    pilot_coefficient,
)
from src.modeling_v3.observation_geometry import (
    B_DROP_W,
    B_EQUAL,
    B_PRESERVE_XI,
    ContributionBatch,
    correction,
    covariance_after,
    decode_raw,
    encode_raw,
    estimate_geometry,
    helmert,
    sample_mixture_sources,
    stable_origin_weight,
    stable_pair_weight,
    stable_three_policy_weight,
)


def make_batch(z, namespace="main", pairs=None, ids=None):
    z = np.asarray(z, float)
    if z.ndim == 2:
        z = z[:, None, :]
    n, c, _ = z.shape
    return ContributionBatch(
        contributions=z,
        sample_ids=tuple(ids or [f"{namespace}:{i}" for i in range(n)]),
        rng_stream_id=f"rng:{namespace}",
        prompt_id="prompt:one",
        contrast_ids=tuple(f"contrast:{i}" for i in range(c)),
        policy_pairs=tuple(pairs or [("baseline", f"candidate:{i}") for i in range(c)]),
        token_ids=tuple((i, 2) for i in range(n)),
        proposal_kind="equal_pair_mixture",
        support_status="VERIFIED",
    )


@pytest.fixture
def exact():
    b = np.array([0.1, 0.2, 0.3, 0.4])
    u = np.array([0.11, 0.19, 0.32, 0.38])
    rho = (b + u) / 2
    z = np.eye(4) * stable_pair_weight(np.log(u), np.log(b))[:, None]
    mu = rho @ z
    centered = z - mu
    cov = (centered.T * rho) @ centered
    return z, rho, mu, cov


def test_helmert_and_noisy_raw_are_losslessly_invertible():
    h = helmert()
    assert_allclose(h.T @ h, np.eye(3), atol=1e-15)
    assert_allclose(h.T @ np.ones(4), 0, atol=1e-15)
    raw = np.array([[0.1, -0.2, 0.7, -0.3], [0.1, -0.2, 0.3, -0.2]])
    transformed, mass = encode_raw(raw)
    assert_allclose(decode_raw(transformed, mass), raw, atol=1e-15)
    assert mass[0] != 0
    assert_allclose(transformed[1] @ h.T, raw[1], atol=1e-15)


def test_equal_l2_identity_does_not_imply_x_improves():
    truth = np.array([0.2, -0.1, 0.1, -0.2])
    raw = truth + np.array([0, 2, 0, 0])
    corrected = correction(raw, B_EQUAL)
    assert_allclose(
        np.sum((raw - truth) ** 2) - np.sum((corrected - truth) ** 2), raw.sum() ** 2 / 4
    )
    assert abs(corrected[0] - truth[0]) > abs(raw[0] - truth[0])


@pytest.mark.parametrize("b,unchanged", [(B_DROP_W, [0, 1, 3]), (B_PRESERVE_XI, [0, 3])])
def test_fixed_correction_preserves_requested_events(exact, b, unchanged):
    z, _, _, _ = exact
    fixed = correction(z, b)
    assert_allclose(fixed.sum(-1), 0, atol=1e-16)
    assert_allclose(fixed[:, unchanged], z[:, unchanged], atol=0)


def test_stable_weights_extremes_and_nearly_equal():
    assert_allclose(
        stable_pair_weight([-1, -10000, -1, -np.inf], [-10000, -1, -1, -1]), [2, -2, 0, -2]
    )
    assert_allclose(stable_pair_weight([-np.inf], [-np.inf]), 0)
    tiny = stable_pair_weight(-20 + 1e-9, -20)
    assert 0 < tiny < 2e-9
    assert_allclose(
        stable_origin_weight(
            [-1, -np.inf, -900, -900 + 1e-8], [-2, -1, -901, -900], [-1, -1, -900, -900]
        ),
        [1 - np.exp(-1), -1, 1 - np.exp(-1), np.expm1(1e-8)],
        atol=1e-13,
    )
    # Each ratio overflows; their signed difference remains representable.
    assert stable_origin_weight(0.0, -1e-13, -720.0) > 0
    with pytest.raises(FloatingPointError, match="NONFINITE_WEIGHT"):
        stable_origin_weight(0.0, -1.0, -1000.0)
    with pytest.raises(ValueError, match="SUPPORT"):
        stable_origin_weight(-1.0, -2.0, -np.inf)
    with pytest.raises(ValueError):
        stable_pair_weight(np.nan, 0)


def test_three_policy_bound_is_three_not_two():
    logs = np.array([[-10000, 0, -10000], [0, -10000, -10000], [-3, -3, -3]])
    assert_allclose(
        stable_three_policy_weight(logs, candidate_index=1, baseline_index=0), [3, -3, 0]
    )
    p = np.array([[0.1, 0.15, 0.2], [0.8, 0.2, 0.1]])
    assert_allclose(
        stable_three_policy_weight(np.log(p), candidate_index=2, baseline_index=0),
        (p[:, 2] - p[:, 0]) / p.mean(-1),
    )
    with pytest.raises(ValueError):
        stable_three_policy_weight(np.zeros((4, 2)))


def test_mixture_sources_use_one_iid_draw_each_not_balanced_quota():
    class RecordedRng:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 2, 7)
            return np.zeros(size, dtype=int)

    assert_allclose(sample_mixture_sources(7, 2, RecordedRng()), 0)


def test_raw_estimate_is_unprojected_with_singleton_covariance_unavailable():
    raw = np.array([[0, 2, 0, 0.0]])
    result = estimate_geometry(make_batch(raw), "RAW4")
    assert_allclose(result.estimate, raw)
    assert_allclose(result.raw_contributions[:, 0], raw)
    assert_allclose(result.mass_residual, [2])
    assert result.covariance_of_mean is None
    assert result.diagnostics["covariance_status"] == "INSUFFICIENT_DRAWS"
    fixed = estimate_geometry(make_batch(raw), "PRESERVE_XI")
    assert_allclose(fixed.estimate, [[0, 1, -1, 0]])
    assert_allclose(fixed.mass_residual, [2])
    assert not fixed.raw_contributions.flags.writeable


def test_joint_covariance_keeps_shared_and_alias_correlations():
    z = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, -0.5]])
    joint = np.stack([z, z, -z], axis=1)
    batch = make_batch(joint, pairs=[("b", "u"), ("b", "u"), ("u", "b")])
    result = estimate_geometry(batch, "PRESERVE_XI")
    expected = np.cov(correction(joint, B_PRESERVE_XI).reshape(4, 12), rowvar=False, ddof=1)
    assert_allclose(result.covariance_per_draw, expected)
    assert_allclose(result.covariance_of_mean, expected / 4)
    assert_allclose(expected[:4, :4], expected[:4, 4:8])
    assert_allclose(expected[:4, :4], -expected[:4, 8:12])
    assert result.diagnostics["unique_contrasts"] == 1
    assert result.diagnostics["independent_draws"] == 4


def test_batch_rejects_identity_support_and_alias_inconsistency():
    z = np.array([[1, 0, 0, 0], [0, -1, 0, 0.0]])
    batch = make_batch(z)
    with pytest.raises(ValueError, match="sample"):
        replace(batch, sample_ids=("same", "same"))
    with pytest.raises(ValueError, match="SUPPORT"):
        replace(batch, support_status="UNKNOWN")
    with pytest.raises(ValueError, match="alias"):
        replace(batch, policy_pairs=(("b", "b"),))
    with pytest.raises(ValueError, match="alias"):
        make_batch(np.stack([z, z + 1], 1), pairs=[("b", "u"), ("b", "u")])
    with pytest.raises(ValueError, match="token"):
        replace(batch, token_ids=((1,),))


def test_oracle_covariance_and_coefficient_error_identities(exact):
    z, rho, mu, cov = exact
    optimum = oracle_b(cov)
    assert_allclose(optimum.sum(), 1)
    c = cov @ np.ones(4)
    vm = c.sum()
    oracle = cov - np.outer(c, c) / vm
    assert_allclose(covariance_after(cov, optimum), oracle, atol=1e-17)
    for b in [B_EQUAL, B_DROP_W, B_PRESERVE_XI, np.array([-0.2, 0.5, 0.4, 0.3])]:
        observed = correction(z, b)
        assert_allclose(rho @ observed, mu, atol=1e-17)
        enumeration = ((observed - mu).T * rho) @ (observed - mu)
        assert_allclose(covariance_after(cov, b, n=17), enumeration / 17, atol=1e-17)
        assert_allclose(enumeration, oracle + vm * np.outer(b - optimum, b - optimum), atol=1e-17)


def test_zero_mass_variance_and_insufficient_pilot_are_explicit():
    z = np.array([[1, -1, 0, 0], [-1, 1, 0, 0.0]])
    assert oracle_b(np.cov(z, rowvar=False)) is None
    b, diag = pilot_coefficient(z, shrink=1)
    assert_allclose(b, B_PRESERVE_XI)
    assert diag["status"] == "PILOT_UNINFORMATIVE"
    assert diag["zero_mass_variance"]
    b, diag = pilot_coefficient([[1, 0, 0, 0]], shrink=0.1)
    assert diag["status"] == "PILOT_INSUFFICIENT_DRAWS"
    assert_allclose(b, B_PRESERVE_XI)


def test_pilot_shrink_l1_and_raw_coefficient_receipt():
    z = np.array([[1, -0.999, 0, 0], [-1, 0.999, 0, 0], [2, -1.998, 0, 0]])
    b, diag = pilot_coefficient(z, shrink=0)
    assert abs(b).sum() <= 8 + 1e-12
    assert_allclose(b.sum(), 1, atol=1e-12)
    assert abs(np.asarray(diag["unbounded_b"])).sum() > 8
    assert 0 < diag["mix"] < 1
    assert diag["l1_limited"]
    z = np.diag([1, -1, 2, -2.0])
    b, diag = pilot_coefficient(z, shrink=1)
    assert_allclose(b, B_EQUAL)
    assert diag["shrink"] == 1


def test_pilot_cannot_use_main_draws_or_rng_or_wrong_prompt():
    z = np.diag([1, -1, 2, -2.0])
    pilot, main = make_batch(z, "pilot"), make_batch(z, "main")
    for bad in [
        replace(main, sample_ids=pilot.sample_ids),
        replace(main, rng_stream_id=pilot.rng_stream_id),
        replace(main, prompt_id="other"),
    ]:
        with pytest.raises(ValueError):
            estimate_pilot(pilot, bad)
    for bad in [
        replace(main, token_ids=((1, 2), (2,), (3,), (4,))),
        replace(main, proposal_kind="origin"),
    ]:
        # Equal tokens are permitted across independent draws; proposal identity is not.
        if bad.proposal_kind == "origin":
            with pytest.raises(ValueError):
                estimate_pilot(pilot, bad)


def test_same_cost_baselines_use_all_draws_and_keep_paired_subset():
    pilot = make_batch(np.diag([1, -1, 2, -2.0]), "pilot")
    main = make_batch(np.diag([2, -2, 1, -1.0]), "main")
    result = cost_matched_estimates(pilot, main, shrink=0.1)
    assert result["cost_matched"]["RAW4"].n == 8
    assert result["paired_main_only"]["RAW4"].n == 4
    adaptive = result["cost_matched"]["PILOT_SHRINK_ZERO_SUM"]
    assert adaptive.n == 4
    assert adaptive.diagnostics["total_draws_cost"] == 8
    assert adaptive.diagnostics["covariance_scope"] == "CONDITIONAL_ON_FROZEN_PILOT"
    assert adaptive.diagnostics["pilot_estimation_uncertainty_included"] is False
    assert_allclose(
        result["cost_matched"]["RAW4"].estimate,
        np.concatenate([pilot.contributions, main.contributions]).mean(0),
    )


def test_finite_pilot_uncertainty_can_exceed_raw_variance(exact):
    _, _, _, cov = exact
    optimum = oracle_b(cov)
    bad = np.array([8.0, -7.0, 0.0, 0.0])
    vm = np.ones(4) @ cov @ np.ones(4)
    expected = covariance_after(cov, optimum) + vm * np.outer(bad - optimum, bad - optimum)
    assert_allclose(covariance_after(cov, bad), expected, atol=1e-15)
    assert np.trace(expected) > np.trace(cov)


def test_crossfit_refuses_naive_independent_fold_covariance():
    a = make_batch(np.diag([1, -1, 2, -2.0]), "a")
    b = make_batch(np.diag([2, -2, 1, -1.0]), "b")
    result = crossfit_estimate(a, b)
    assert result.n == 8
    assert result.covariance_of_mean is None
    assert result.diagnostics["covariance_status"] == "CROSSFIT_REQUIRES_FULL_REFIT"
    ab = estimate_pilot(a, b)
    ba = estimate_pilot(b, a)
    assert_allclose(result.estimate, (ab.estimate + ba.estimate) / 2)
    boot = bootstrap_crossfit(a, b, repetitions=12, seed=31)
    assert boot["replicate_estimates"].shape == (12, 1, 4)
    assert boot["covariance_of_mean"].shape == (4, 4)
    assert boot["refits_per_replicate"] == 2
    assert boot["status"] == "BOOTSTRAP_ESTIMATE_NOT_COVERAGE_CERTIFICATE"


def test_crossfit_mean_unbiased_by_small_full_enumeration():
    # Four equally likely contributions have zero expected mass. Enumerating
    # both independent two-draw folds tests the entire adaptive fit procedure.
    support = np.diag([1.0, -1.0, 2.0, -2.0])
    estimates = []
    for indices in product(range(4), repeat=4):
        a = make_batch(support[list(indices[:2])], "a")
        b = make_batch(support[list(indices[2:])], "b")
        estimates.append(crossfit_estimate(a, b).estimate)
    assert_allclose(np.mean(estimates, axis=0), support.mean(0)[None, :], atol=1e-14)


def test_exhaustive_events_and_nonfinite_contributions_are_not_dropped():
    from src.modeling_v3.observation_geometry import event_contributions

    z = event_contributions([1.0, -2.0, 3.0, -4.0], np.array([0, 1, 2, 3]))
    assert_allclose(z, np.diag([1, -2, 3, -4]))
    for labels in [np.array([0, 1, 2, 4]), np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 1, 2])]:
        with pytest.raises(ValueError):
            event_contributions([1.0, -2.0, 3.0, -4.0], labels)
    with pytest.raises(ValueError):
        event_contributions([np.inf], np.array([0]))
    # Empty generated output is an action, not a missing observation.
    batch = replace(make_batch([[0, 0, 0, 1.0], [1, 0, 0, 0.0]]), token_ids=((), (2,)))
    assert batch.n == 2


def test_primary_validity_is_consistently_minus_i_and_preserved():
    batch = make_batch([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 2.0, -1.0]])
    raw = estimate_geometry(batch, "RAW4")
    corrected = estimate_geometry(batch, "PRESERVE_XI")
    assert_allclose(raw.delta_pX, corrected.delta_pX)
    assert_allclose(raw.delta_v, corrected.delta_v)
    assert_allclose(raw.raw_valid_event_sum - raw.delta_v, raw.mass_residual)


def test_actual_finite_pilot_covariance_penalty_by_enumeration(exact):
    z, rho, _mu, cov = exact
    optimum = oracle_b(cov)
    expected_cov = np.zeros((4, 4))
    coefficient_error = np.zeros((4, 4))
    for i, j in product(range(4), repeat=2):
        b, _ = pilot_coefficient(z[[i, j]], shrink=0.0)
        probability = rho[i] * rho[j]
        expected_cov += probability * covariance_after(cov, b, n=2)
        coefficient_error += probability * np.outer(b - optimum, b - optimum)
    vm = np.ones(4) @ cov @ np.ones(4)
    assert_allclose(
        expected_cov, covariance_after(cov, optimum, n=2) + vm * coefficient_error / 2, atol=1e-17
    )
    # Strong simple RAW4 receives all four paid draws; the two-draw pilot
    # does not come free and can reverse the known-covariance oracle gain.
    assert np.trace(expected_cov) > np.trace(cov / 4)


def test_covariance_rejects_invalid_matrix_and_divisor():
    for cov in [np.diag([1, 1, 1, -1]), np.ones((3, 4)), np.array([[np.nan]])]:
        with pytest.raises(ValueError):
            oracle_b(cov)
    with pytest.raises(ValueError):
        covariance_after(np.eye(4), B_EQUAL, n=0)
    with pytest.raises(ValueError):
        correction(np.zeros(4), np.ones(4))


def test_join_shared_banks_preserves_joint_correlation_and_rejects_token_mismatch():
    from src.modeling_v3.observation_geometry import join_shared_batches

    a = replace(make_batch(np.diag([1, -1, 2, -2.0])), proposal_kind="origin")
    b = replace(
        a,
        contrast_ids=("other:contrast",),
        policy_pairs=(("baseline", "other"),),
        contributions=a.contributions * 2,
    )
    result = estimate_geometry(join_shared_batches([a, b]), "RAW4")
    assert_allclose(
        result.raw_covariance_of_mean[:4, 4:], 2 * result.raw_covariance_of_mean[:4, :4]
    )
    with pytest.raises(ValueError, match="token_ids"):
        join_shared_batches([a, replace(b, token_ids=((6,), (7,), (8,), (9,)))])
    with pytest.raises(ValueError, match="sample_ids"):
        join_shared_batches([a, replace(b, sample_ids=tuple(reversed(b.sample_ids)))])
