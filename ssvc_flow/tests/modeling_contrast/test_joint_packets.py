import numpy as np
import pytest

from src.modeling_contrast.covariance import counts_covariance
from src.modeling_contrast.joint_packets import fit_packet_covariance, retained_fit_rows
from src.modeling_contrast.targets import helmert


def real_excerpt_packet():
    # Exact paid-count excerpt, seed103_X_VALID anchor24 n64 noise0, banks0/1.
    # Source counts SHA256: 266d462873e8fb43d666afe8bae7278fd4cdf6629db86255548968f924ded09b
    counts = np.array(
        [
            [
                [[3, 12, 48, 1], [3, 6, 54, 1]],
                [[3, 5, 56, 0], [1, 5, 57, 1]],
                [[3, 12, 48, 1], [3, 6, 54, 1]],
            ],
            [
                [[3, 12, 48, 1], [3, 6, 54, 1]],
                [[7, 9, 48, 0], [2, 5, 57, 0]],
                [[3, 12, 48, 1], [3, 6, 54, 1]],
            ],
        ],
        dtype=np.uint16,
    )
    rows = [
        dict(
            bank=b,
            noise_replica=0,
            old_counts_reused=True,
            packet_id=f"bank{b}",
            policy_fingerprints=[["alias42", f"alias{candidate}"], ["alias42", "alias42"]],
        )
        for b, candidate in enumerate((43, 46))
    ]
    return {
        "legacy_total_levels": counts[None].astype(float) / 64,
        "covariance": np.zeros((1, 2, 2, 6, 6)),
        "metadata": {"method": "O_IND", "n": 64, "packets": rows},
    }


def test_actual_legacy_shared_baseline_produces_cross_bank_covariance():
    packet = real_excerpt_packet()
    cov = fit_packet_covariance(packet, 0, [0, 1])
    assert cov.shape == (2, 12, 12)
    expected = helmert().T @ counts_covariance(np.array([3, 12, 48, 1])) @ helmert()
    np.testing.assert_allclose(cov[0, :3, 6:9], expected, atol=1e-18)
    assert np.linalg.norm(cov[0, :3, 6:9]) > 0
    np.testing.assert_array_equal(cov[:, 3:6], 0)
    np.testing.assert_array_equal(cov[:, 9:12], 0)
    assert np.linalg.eigvalsh(cov).min() > -1e-15


def test_new_independent_evaluation_is_not_correlated_with_legacy_fit():
    packet = real_excerpt_packet()
    packet["metadata"]["packets"][1]["old_counts_reused"] = False
    packet["covariance"][0, 1] = np.eye(6)[None] * 0.25
    cov = fit_packet_covariance(packet, 0, [0, 1])
    np.testing.assert_array_equal(cov[:, :6, 6:], 0)
    np.testing.assert_array_equal(cov[:, 6:, 6:], packet["covariance"][0, 1])


def test_alias_count_disagreement_and_noninteger_recovery_fail():
    packet = real_excerpt_packet()
    packet["legacy_total_levels"][0, 1, 0, 0] += [1 / 64, -1 / 64, 0, 0]
    with pytest.raises(ValueError, match="alias"):
        fit_packet_covariance(packet, 0, [0, 1])
    packet = real_excerpt_packet()
    packet["legacy_total_levels"][0, 0, 0, 0] += [0.1 / 64, -0.1 / 64, 0, 0]
    with pytest.raises(ValueError, match="integer"):
        fit_packet_covariance(packet, 0, [0, 1])


def test_legacy_duplicate_rows_removed_but_independent_measurements_retained():
    packet = real_excerpt_packet()
    packet["metadata"]["packets"][1]["policy_fingerprints"][0][1] = "alias43"
    assert retained_fit_rows(packet, 0, [0, 1]).tolist() == [0]
    packet["metadata"]["packets"][1]["old_counts_reused"] = False
    assert retained_fit_rows(packet, 0, [0, 1]).tolist() == [0, 2]


def test_non_ind_method_preserves_original_packet_covariance_blocks():
    packet = real_excerpt_packet()
    packet["metadata"]["method"] = "O_LR_ORIGIN"
    packet["covariance"][:] = np.eye(6) * 0.1
    cov = fit_packet_covariance(packet, 0, [1, 0])
    np.testing.assert_allclose(cov[0], np.eye(12) * 0.1)
    with pytest.raises(ValueError):
        fit_packet_covariance(packet, 0, [0, 0])
