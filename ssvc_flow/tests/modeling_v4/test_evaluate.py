import copy

import numpy as np
import pytest

from src.modeling_v4.evaluate import evaluate_records, seed_cluster_bootstrap


def record(seed=1, method="FULL", **updates):
    value = {
        "seed": seed,
        "method": method,
        "origin_id": f"o{seed}",
        "bank_id": "b",
        "role": "development",
        "target": "joint_1_minus_joint_0",
        "prompt_ids": ["a", "b"],
        "groups": ["g", "g"],
        "prediction": [[0.1, 0, 0, -0.1], [0.3, 0, 0, -0.3]],
        "reference": [[0.0, 0, 0, 0], [0.0, 0, 0, 0]],
        "reference_kind": "EXACT",
        "reference_se": [[0.0] * 4] * 2,
        "reference_resolved": True,
        "is_alias": False,
        "prediction_status": "PREDICTED",
        "diagnostics": {"rho": 0.9},
    }
    return {**value, **updates}


def get(result, channel="RAW4", scope="all"):
    return next(r for r in result["summary"] if r["channel"] == channel and r["scope"] == scope)


def test_four_event_mean_group_v_and_large_rho_are_retained():
    result = evaluate_records(iter([record()]))
    assert get(result)["mse"] == pytest.approx(0.025)
    assert get(result, "group_v")["mse"] == pytest.approx(0.04)
    assert get(result, "X")["all_count"] == 2
    assert get(result)["signal_rms"] == 0
    assert get(result)["nrmse"] is None
    assert result["risk_coverage"][0]["threshold"] == 0.9
    assert result["risk_coverage"][0]["coverage"] == 1


def test_unknown_and_reference_unresolved_do_not_become_zero_or_disappear():
    unknown = record(bank_id="unknown", prediction=None, prediction_status="UNKNOWN")
    unresolved = record(bank_id="uncertain", reference_resolved=False)
    result = evaluate_records([record(), unknown, unresolved])
    row = get(result)
    assert row["all_count"] == 24 and row["finite_resolved_count"] == 8
    assert row["mse"] is None and row["finite_subset_mse"] == pytest.approx(0.025)
    assert get(result, scope="unknown")["all_count"] == 8
    assert get(result, scope="reference_unresolved")["all_count"] == 8
    assert result["risk_coverage"][0]["coverage"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        evaluate_records([record(prediction=None)])


def test_reference_variance_correction_negative_is_secondary_and_no_q95_correction():
    result = evaluate_records(
        [
            record(
                prediction=np.zeros((2, 4)),
                reference_kind="MONTE_CARLO",
                reference_se=np.full((2, 4), 0.2),
                reference_independent=True,
                reference_prompt_independent=True,
            )
        ]
    )
    row = get(result)
    assert row["mse"] == 0 and row["reference_variance_corrected_mse"] == pytest.approx(-0.04)
    assert row["q95_absolute"] == 0 and row["reference_se_rms"] == pytest.approx(0.2)
    assert get(result, "group_v")["reference_variance_corrected_mse"] == pytest.approx(-0.02)
    dependent = evaluate_records(
        [record(reference_kind="MONTE_CARLO", reference_se=np.full((2, 4), 0.2))]
    )
    assert get(dependent)["reference_variance_corrected_mse"] is None
    assert get(dependent, "group_v")["reference_se_rms"] is None


def test_alias_membership_uses_identity_not_zero_truth_and_duplicates_rejected():
    result = evaluate_records([record(), record(bank_id="alias", is_alias=True)])
    assert get(result, scope="alias")["all_count"] == 8
    assert get(result, scope="nonalias")["all_count"] == 8
    with pytest.raises(ValueError):
        evaluate_records([record(), record()])
    changed = record(method="OTHER")
    changed["reference"] = [[0.01, 0, 0, -0.01]] * 2
    with pytest.raises(ValueError):
        evaluate_records([record(), changed])


def test_seed_cluster_bootstrap_keeps_all_rows_of_each_seed_together():
    values = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    result = seed_cluster_bootstrap(values, [1, 1, 2, 2, 3, 3])
    repeated = seed_cluster_bootstrap(values.repeat(3), np.repeat([1, 1, 2, 2, 3, 3], 3))
    assert result["repetitions"] == 2000 and result["unit"] == "training_seed"
    assert result["interval"] == repeated["interval"]
    assert result["independent_seed_count"] == 3 and result["certified_95"] is False
    assert seed_cluster_bootstrap([0, 1, 2], [1, 1, 2])["interval"] is None


def test_query_block_and_prompt_rows_agree_and_sink_preserves_caller_data():
    block = record()
    before = copy.deepcopy(block)
    single = []
    for i in range(2):
        single.append(
            {
                **block,
                "prompt_ids": [block["prompt_ids"][i]],
                "groups": ["g"],
                "prediction": block["prediction"][i],
                "reference": block["reference"][i],
                "reference_se": block["reference_se"][i],
            }
        )
    output = []
    result = evaluate_records([block], core_sink=output.append)
    assert result["core_results"] == [] and len(output) == 1 and block == before
    assert get(result)["mse"] == get(evaluate_records(single))["mse"]
    assert get(result, "group_v")["mse"] == get(evaluate_records(single), "group_v")["mse"]
    assert result["risk_coverage"] == evaluate_records(single)["risk_coverage"]


def test_risk_curve_ties_are_kept_together_and_missing_diagnostics_are_counted():
    result = evaluate_records(
        [record(), record(bank_id="b2"), record(bank_id="b3", diagnostics={})]
    )
    curve = result["risk_coverage"]
    assert len(curve) == 1 and curve[0]["selected_queries"] == 2
    assert curve[0]["diagnostic_missing_queries"] == 1
    assert curve[0]["coverage"] == pytest.approx(2 / 3)


def test_roles_keep_independent_seed_bootstrap_populations_separate():
    result = evaluate_records([record(s) for s in (1, 2, 3)] + [record(9, role="test")])
    dev = next(x for x in result["bootstrap"] if x["role"] == "development" and x["scope"] == "all")
    test = next(x for x in result["bootstrap"] if x["role"] == "test" and x["scope"] == "all")
    assert dev["seed_order"] == [1, 2, 3] and dev["interval"] is not None
    assert test["seed_order"] == [9] and test["interval"] is None


def test_crossfit_naive_fold_se_is_rejected():
    with pytest.raises(ValueError, match="Crossfit"):
        evaluate_records(
            [
                record(
                    reference_kind="MONTE_CARLO",
                    reference_estimator="CROSSFIT_COV_ZERO_SUM",
                    reference_se_method="IID_CORRECTED_FOLDS",
                )
            ]
        )


def test_writer_binds_outputs_and_failed_input_has_no_completion(tmp_path):
    import json

    from src.modeling_v3.io import verify_manifest
    from src.modeling_v4.evaluate import write_evaluation

    out = tmp_path / "evaluation"
    result = write_evaluation(
        [record()], out, binding={"config_hash": "fixture", "source": "fixture"}
    )
    assert result["status"] == "COMPLETE" and verify_manifest(out)["status"] == "COMPLETE"
    assert json.loads((out / "METRICS.json").read_text())["kind"] == "V4_EVALUATION"
    with pytest.raises(FileExistsError):
        write_evaluation([record()], out, binding={"fixture": True})
    with pytest.raises(ValueError):
        write_evaluation([record(prediction=None)], tmp_path / "failed", binding={"fixture": True})
    assert not (tmp_path / "failed" / "COMPLETE.json").exists()


def test_nonalias_risk_retains_separate_error_denominator():
    result = evaluate_records(
        [record(), record(bank_id="alias", is_alias=True, prediction=np.zeros((2, 4)))]
    )
    curve = result["risk_coverage"][0]
    assert curve["finite_subset_mse"] == pytest.approx(0.0125)
    assert curve["nonalias_finite_subset_mse"] == pytest.approx(0.025)
    assert curve["finite_resolved_cells"] == 16
    assert curve["nonalias_finite_resolved_cells"] == 8


def test_scalar_zero_reference_se_is_explicit_exact_uncertainty():
    assert get(evaluate_records([record(reference_se=0)]))["reference_se_rms"] == 0
