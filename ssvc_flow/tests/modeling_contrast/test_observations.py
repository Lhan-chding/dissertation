import numpy as np
import pytest

from src.modeling_contrast.observations import (
    OracleAudit,
    ToyWorldService,
    known_event_logp,
    measure,
)


def fixture():
    categories = np.array([[2, 3, 0, 1, 2, 3]])
    p = np.array([[0.2, 0.15, 0.2, 0.1, 0.2, 0.15]])
    q = p + np.array([[0.001, -0.002, 0.002, 0, 0, -0.001]])
    return {"o": p, "b": p, "u": q, "a": p.copy()}, categories


@pytest.mark.parametrize("method", ["O_IND", "O_CRN", "O_LR_ORIGIN", "O_LR_MIX"])
def test_packets_are_finite_and_aliases_exact_zero(method):
    tables, categories = fixture()
    service = ToyWorldService.from_action_probabilities(tables, categories)
    packet = measure(service, [("b", "u"), ("b", "a")], origin_id="o", method=method, n=64, seed=13)
    assert packet.event_estimate.shape == (2, 1, 4)
    assert packet.covariance.shape == (1, 6, 6)
    np.testing.assert_array_equal(packet.event_estimate[1], 0)
    assert np.isfinite(packet.event_estimate).all()
    assert not hasattr(packet, "oracle") and not hasattr(packet, "service")
    assert not packet.event_estimate.flags.writeable
    assert packet.cost["generated_sequences"] == 0


def test_likelihood_raw_and_helmert_are_distinct_views():
    tables, cats = fixture()
    service = ToyWorldService.from_action_probabilities(tables, cats)
    packet = measure(service, [("b", "u")], origin_id="o", method="O_LR_ORIGIN", n=64, seed=33)
    raw = packet.raw_event_estimate
    np.testing.assert_allclose(
        packet.event_estimate, raw - raw.sum(-1, keepdims=True) / 4, atol=1e-16
    )
    np.testing.assert_allclose(packet.mass_residual, raw.sum(-1), atol=1e-16)


def test_legacy_independent_counts_reused_without_new_sampling():
    tables, cats = fixture()
    service = ToyWorldService.from_action_probabilities(tables, cats)
    counts = {"b": np.array([[4, 2, 7, 3]]), "u": np.array([[5, 2, 7, 2]])}
    packet = measure(
        service, [("b", "u")], origin_id="o", method="O_IND", n=16, seed=1, legacy_counts=counts
    )
    np.testing.assert_allclose(packet.event_estimate[0, 0], [1 / 16, 0, 0, -1 / 16])
    assert packet.cost["generated_actions"] == 0
    assert packet.cost["physical_policy_forwards"] == 0
    assert packet.cost["reused_actions"] == 32


def test_known_event_exact_scores_cached_and_no_generation():
    tables, cats = fixture()
    service = ToyWorldService.from_action_probabilities(tables, cats)
    got = known_event_logp(service, ["b", "u", "a"])
    assert got["cost"]["cross_policy_action_scores"] == 6
    assert got["cost"]["generated_actions"] == 0
    np.testing.assert_allclose(got["pX"], [[0.2], [0.202], [0.2]])
    np.testing.assert_allclose(got["v"], [[0.7], [0.703], [0.7]])
    assert known_event_logp(service, ["b", "u"])["cost"]["cross_policy_action_scores"] == 0


def test_oracle_moments_are_separate_and_unbiased():
    tables, cats = fixture()
    oracle = OracleAudit.from_action_probabilities(tables, cats)
    for method in ("O_IND", "O_CRN", "O_LR_ORIGIN", "O_LR_MIX"):
        result = oracle.moments([("b", "u")], origin_id="o", method=method, n=64)
        np.testing.assert_allclose(
            result["event_mean"][0, 0], [0.002, 0, 0.001, -0.003], atol=2e-16
        )
        assert result["status"] == "ORACLE_DIAGNOSTIC_ONLY"


def test_missing_support_is_a_packet_failure():
    p = np.array([[1.0, 0, 0, 0]])
    q = np.array([[0.9, 0.1, 0, 0]])
    service = ToyWorldService.from_action_probabilities({"o": p, "b": p, "u": q}, [[0, 1, 2, 3]])
    packet = measure(service, [("b", "u")], origin_id="o", method="O_LR_ORIGIN", n=64, seed=5)
    assert packet.status == "BLOCKED_SUPPORT_MISMATCH"
    assert packet.cost["generated_actions"] == 0


def test_joint_origin_covariance_retains_shared_packet_terms():
    tables, cats = fixture()
    tables["v"] = tables["b"] + 2 * (tables["u"] - tables["b"])
    packet = measure(
        ToyWorldService.from_action_probabilities(tables, cats),
        [("b", "u"), ("b", "v")],
        origin_id="o",
        method="O_LR_ORIGIN",
        n=64,
        seed=11,
    )
    np.testing.assert_allclose(
        packet.covariance[0, :3, 3:], 2 * packet.covariance[0, :3, :3], rtol=1e-10, atol=1e-18
    )


def test_packet_reuse_does_not_increase_information_or_cost():
    tables, cats = fixture()
    service = ToyWorldService.from_action_probabilities(tables, cats)
    args = dict(pairs=[("b", "u")], origin_id="o", method="O_LR_MIX", n=64, seed=11)
    first = measure(service, **args)
    before = service.ledger.snapshot()
    second = measure(service, **args)
    assert first is second
    assert before == service.ledger.snapshot()
    assert first.sample_ids == second.sample_ids


def test_legacy_unsigned_counts_preserve_negative_contrast():
    tables, cats = fixture()
    service = ToyWorldService.from_action_probabilities(tables, cats)
    counts = {
        "b": np.array([[5, 2, 7, 2]], dtype=np.uint16),
        "u": np.array([[4, 2, 7, 3]], dtype=np.uint16),
    }
    packet = measure(service, [("b", "u")], method="O_IND", n=16, seed=2, legacy_counts=counts)
    np.testing.assert_array_equal(packet.event_estimate[0, 0], [-1 / 16, 0, 0, 1 / 16])


def test_redundant_mixture_pairs_share_sampling_and_joint_covariance():
    tables, cats = fixture()
    service = ToyWorldService.from_action_probabilities(tables, cats)
    packet = measure(
        service, [("b", "u"), ("a", "u"), ("u", "b")], method="O_LR_MIX", n=64, seed=14
    )
    assert packet.cost["generated_actions"] == 64
    assert len(packet.sample_ids) == 1
    np.testing.assert_array_equal(packet.event_estimate[0], packet.event_estimate[1])
    np.testing.assert_allclose(packet.event_estimate[0], -packet.event_estimate[2], atol=1e-16)
    np.testing.assert_allclose(packet.covariance[0, :3, 3:6], packet.covariance[0, :3, :3])
    np.testing.assert_allclose(packet.covariance[0, :3, 6:9], -packet.covariance[0, :3, :3])


def test_parent_frozen_forward_and_known_scores_match_exact_parent():
    from src.modeling_qualification.toy import (
        _numpy_forward,
        flatten_parameters,
        make_model,
    )

    theta = flatten_parameters(make_model())
    rng = np.random.default_rng(9)
    features = rng.normal(size=(2, 16, 44))
    cats = np.tile([2, 3, 0, 1, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], (2, 1))
    service = ToyWorldService.from_parameters({"b": theta, "a": theta.copy()}, features, cats)
    assert service.ledger.snapshot()["physical_policy_forwards"] == 0
    result = known_event_logp(service, ["b", "a"])
    expected = _numpy_forward(theta, features, cats)[3]
    np.testing.assert_allclose(result["pX"], np.stack([expected[:, 0]] * 2), rtol=1e-15)
    np.testing.assert_allclose(result["v"], np.stack([expected[:, :3].sum(-1)] * 2), atol=2e-16)
    assert result["cost"]["physical_policy_forwards"] == 2
    assert result["cost"]["cross_policy_action_scores"] == 6
    assert result["cost"]["optimizer_updates"] == 0
    assert result["cost"]["exact_probability_calls"] == 0


@pytest.mark.parametrize("method", ["O_IND", "O_CRN", "O_LR_ORIGIN", "O_LR_MIX"])
def test_empirical_estimator_mean_and_covariance_match_finite_oracle(method):
    tables, cats = fixture()
    tables["u"] = tables["b"] + [[0.04, -0.05, 0.06, 0, 0, -0.05]]
    pairs = [("b", "u")]
    exact = OracleAudit.from_action_probabilities(tables, cats).moments(
        pairs, origin_id="o", method=method, n=64
    )
    service = ToyWorldService.from_action_probabilities(tables, cats)
    estimates = np.stack(
        [
            measure(
                service,
                pairs,
                origin_id="o",
                method=method,
                n=64,
                seed=rep,
                purpose="observation_study",
            ).helmert_estimate.ravel()
            for rep in range(512)
        ]
    )
    expected = (
        exact["event_mean"]
        @ np.array(
            [
                [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
                [-1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
                [0, -2 / np.sqrt(6), 1 / np.sqrt(12)],
                [0, 0, -3 / np.sqrt(12)],
            ]
        )
    ).ravel()
    se = np.sqrt(np.diag(exact["covariance"][0]) / len(estimates))
    assert np.all(np.abs(estimates.mean(0) - expected) < 6 * se)
    empirical = np.cov(estimates, rowvar=False)
    assert (
        np.linalg.norm(empirical - exact["covariance"][0]) / np.linalg.norm(exact["covariance"][0])
        < 0.3
    )


def test_packet_identity_binds_labels_and_fit_evaluation_rng():
    tables, cats = fixture()
    service = ToyWorldService.from_action_probabilities(tables, cats)
    args = dict(pairs=[("b", "u")], origin_id="o", method="O_CRN", n=64, seed=31)
    fit = measure(service, **args, purpose="fit")
    evaluation = measure(service, **args, purpose="evaluation")
    assert fit.packet_id != evaluation.packet_id
    assert fit.sample_ids != evaluation.sample_ids
    flipped = cats[:, ::-1]
    other = measure(
        ToyWorldService.from_action_probabilities(tables, flipped), **args, purpose="fit"
    )
    assert fit.packet_id != other.packet_id


def test_oracle_mixture_redundant_and_reverse_covariance_matches_packet_contract():
    tables, cats = fixture()
    result = OracleAudit.from_action_probabilities(tables, cats).moments(
        [("b", "u"), ("a", "u"), ("u", "b")], method="O_LR_MIX", n=64
    )
    cov = result["covariance"][0]
    np.testing.assert_allclose(cov[:3, 3:6], cov[:3, :3], atol=1e-18)
    np.testing.assert_allclose(cov[:3, 6:9], -cov[:3, :3], atol=1e-18)
