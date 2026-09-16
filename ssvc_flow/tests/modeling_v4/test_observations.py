"""Small full-refit observation uncertainty fixtures; no model calls."""

import numpy as np
import pytest

from src.modeling_v3.observation_geometry import ContributionBatch
from src.modeling_v4.observations import independent_packet_uncertainty, observe_packet


def packet(seed=0, n=24):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2, 4))
    z[..., 1] += 2 * z[..., 0]
    return ContributionBatch(
        z,
        tuple(f"{seed}:{i}" for i in range(n)),
        f"stream:{seed}",
        "p",
        ("a", "b"),
        (("base", "left"), ("base", "right")),
        tuple((1, 99) for _ in range(n)),
        "origin",
    )


def test_raw_preserve_and_crossfit_share_draws_and_keep_mass():
    batch = packet()
    report = observe_packet(
        batch, bootstrap_repetitions=24, costs={"forward_calls": 48, "generated_sequences": 24}
    )
    np.testing.assert_array_equal(report["raw_contributions"], batch.contributions)
    np.testing.assert_allclose(report["raw_mass_per_draw"], batch.contributions.sum(-1))
    raw, preserve, crossfit = [
        report["methods"][name] for name in ("RAW4", "PRESERVE_XI", "CROSSFIT_COV_ZERO_SUM")
    ]
    np.testing.assert_allclose(raw["estimate"], batch.contributions.mean(0))
    np.testing.assert_allclose(preserve["estimate"][:, [0, 3]], raw["estimate"][:, [0, 3]])
    assert report["actual_costs"]["forward_calls"] == 48
    assert report["actual_costs"]["generated_sequences"] == 24
    assert crossfit["uncertainty"]["refits_per_replicate"] == 2
    assert crossfit["uncertainty"]["repetitions"] == 24
    assert crossfit["diagnostics"]["folds_are_independent_for_interval"] is False
    assert crossfit["diagnostics"]["total_draws_cost"] == batch.n
    assert crossfit["uncertainty"]["model_calls"] == 0


def test_crossfit_without_bootstrap_does_not_claim_iid_covariance():
    result = observe_packet(packet(), bootstrap_repetitions=None)["methods"][
        "CROSSFIT_COV_ZERO_SUM"
    ]
    assert result["covariance_of_mean"] is None
    assert result["uncertainty"]["status"] == "UNAVAILABLE_WITHOUT_FULL_REFIT_OR_PACKETS"


def test_crossfit_rejects_reference_leakage_and_non_iid_design():
    batch = packet()
    with pytest.raises(ValueError, match="reference"):
        observe_packet(batch, reference_sample_ids=[batch.sample_ids[0]])
    with pytest.raises(ValueError, match="reference"):
        observe_packet(batch, reference_rng_stream_ids=[batch.rng_stream_id])
    with pytest.raises(ValueError, match="iid"):
        observe_packet(batch, sampling_design="fixed_quota")


def test_independent_packet_uncertainty_is_across_complete_refitted_packets():
    packets = [packet(seed) for seed in range(5)]
    result = independent_packet_uncertainty(packets, method="CROSSFIT_COV_ZERO_SUM")
    means = np.asarray(
        [
            observe_packet(batch, bootstrap_repetitions=None)["methods"]["CROSSFIT_COV_ZERO_SUM"][
                "estimate"
            ]
            for batch in packets
        ]
    )
    expected = np.cov(means.reshape(5, -1), rowvar=False, ddof=1)
    np.testing.assert_allclose(result["packet_covariance"], expected)
    np.testing.assert_allclose(result["covariance_of_packet_mean"], expected / 5)
    assert result["independent_packets"] == 5 and not result["training_seed_uncertainty_included"]
    with pytest.raises(ValueError, match="reused"):
        independent_packet_uncertainty([packets[0], packets[0]], method="CROSSFIT_COV_ZERO_SUM")
