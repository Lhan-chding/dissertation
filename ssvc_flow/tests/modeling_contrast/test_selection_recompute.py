import numpy as np

from src.modeling_contrast.selection_recompute import (
    bootstrap_nrmse,
    canonical,
    recompute_candidates,
)


def test_independent_bootstrap_resamples_whole_seed_sufficient_statistics():
    seeds = np.array([202, 201, 202, 201])
    method = np.array([[4.0, 25.0], [1.0, 25.0], [4.0, 25.0], [1.0, 25.0]])
    baseline = np.array([[25.0, 25.0]] * 4)
    result = bootstrap_nrmse(method, baseline, seeds, 5000, 2026091503)
    assert result["cluster_count"] == 2
    np.testing.assert_allclose(result["paired_mean"], np.sqrt(0.1) - 1)
    np.testing.assert_allclose([result["ci025"], result["ci975"]], [-0.8, -0.6])
    assert result["bootstrap_replicates"] == 5000


def test_recompute_detects_bootstrap_table_corruption_and_nonzero_gate():
    rows = []
    for seed in (201, 202):
        for method, err, rank, norm in [
            ("C0", 1.0, 0, 0.0),
            ("C1_LEGACY_EXACT_RULE", 0.9, 1, 0.1),
            ("C2", 0.25, 1, 0.2),
        ]:
            rows.append(
                dict(
                    configuration_id=method,
                    method=method,
                    observation="O_IND",
                    n=64,
                    target="joint_1_minus_joint_0",
                    population="all",
                    seed=seed,
                    arm="X_BASE",
                    anchor=8,
                    noise_replica=0,
                    actual_rank=rank,
                    predicted_difference_norm=norm,
                    pX_error_ss=err,
                    pX_truth_ss=1.0,
                    v_error_ss=err,
                    v_truth_ss=1.0,
                )
            )
    cfg = {
        "data_roles": {"legacy_selection_seeds": [201, 202]},
        "fresh_cpu": {"main_arms": ["X_BASE"]},
        "branches": {"anchors": [8]},
        "observation": {"noise_replicas": 1, "methods": ["O_IND"]},
        "primary_target": {"name": "joint_1_minus_joint_0"},
        "statistics": {
            "bootstrap_replicates": 5000,
            "bootstrap_seed": 2026091503,
            "nrmse_pX_max": 0.75,
            "nrmse_v_max": 0.75,
        },
    }
    result = recompute_candidates(rows, cfg)
    assert result["candidates"][0]["science_without_nonalias"]
    assert result["candidates"][0]["pX_nrmse"] == 0.5
    assert len(result["bootstrap"]) == 4
    rows[-1]["actual_rank"] = 0
    rows[-2]["actual_rank"] = 0
    # The two seed/model rows must both become zero before excitation vanishes.
    for row in rows:
        if row["method"] == "C2":
            row["actual_rank"] = 0
    assert not recompute_candidates(rows, cfg)["candidates"][0]["science_without_nonalias"]


def test_canonical_hash_does_not_depend_on_dictionary_order():
    assert canonical({"x": 1, "y": [2]}) == canonical({"y": [2], "x": 1})
