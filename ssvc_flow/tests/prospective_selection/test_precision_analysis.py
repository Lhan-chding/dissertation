"""CPU fixtures only: these tests are not real model experiment evidence."""

import copy
import math

import numpy as np
import pytest
from scipy.stats import t

from src.prospective_selection.analysis import (
    PRIMARY,
    SECONDARY,
    aggregate_lineage_differences,
    analyze_final,
    four_cell_comparison,
    holm_adjust,
)
from src.prospective_selection.precision import CELLS, plan_mc_samples, plan_precision


def planning_rows(n_per_cell=4):
    rows, registry = [], {}
    for index, (recipe, step) in enumerate(CELLS):
        for repeat in range(n_per_cell):
            lineage = str(index * 10 + repeat)
            record = {
                "lineage_id": lineage,
                "source_recipe": recipe,
                "origin_step": step,
                "difference": (repeat - (n_per_cell - 1) / 2) * 0.02,
                "role": "development",
            }
            rows.append(record)
            registry[lineage] = {"source_recipe": recipe, "origin_step": step}
    return rows, registry


def final_fixture(n_per_cell=3, same_action=False):
    decisions, outcomes, registry = [], [], {}
    for index, (source, step) in enumerate(CELLS):
        for repeat in range(n_per_cell):
            lineage = str(index * 10 + repeat)
            origin = {"lineage_id": lineage, "source_recipe": source, "origin_step": step}
            registry[lineage] = {"source_recipe": source, "origin_step": step}
            selections = {
                PRIMARY[0]: "R3",
                PRIMARY[1]: "R3" if same_action else "R4",
                "best_static": "R0",
                "GDPO_R4": "GDPO_R4",
                "SAW_R4": "SAW_R4",
                "DIRECT_REPAIR_R4": "DIRECT_REPAIR_R4",
            }
            decisions.append({**origin, "selections": selections, "made_before_run": True})
            for recipe in set(selections.values()):
                for future_repeat in (1, 2):
                    outcomes.append(
                        {
                            **origin,
                            "recipe_id": recipe,
                            "continuation_repeat": future_repeat,
                            "horizon": 32,
                            "panel_id": "T",
                            "n_draws": 16,
                            "execution_kind": "CPU_FIXTURE",
                            "schedule_id": str(future_repeat),
                            "J": 0.5
                            + (0.02 + repeat * 0.01 if recipe == "R3" else 0)
                            + future_repeat * 0.001,
                        }
                    )
    return decisions, outcomes, registry


def test_one_predetermined_anchor_per_lineage_not_average_stages():
    rows, registry = planning_rows()
    expected = plan_precision(rows, registry, bootstrap_repetitions=500)
    extra = [
        {**row, "origin_step": 96 if row["origin_step"] == 32 else 32, "difference": 0.99}
        for row in rows
    ]
    actual = plan_precision(rows + extra, registry, bootstrap_repetitions=500)
    assert actual == expected
    assert actual["n_independent_lineages"] == 16
    assert actual["effective_sd"] == pytest.approx(np.std([-0.03, -0.01, 0.01, 0.03], ddof=1))
    assert actual["bootstrap_quantile"] == 0.75
    assert actual["finite_sample_power_guarantee"] is False


def test_sd_bootstrap_matches_independent_manual_four_cell_calculation():
    rows, registry = planning_rows()
    got = plan_precision(rows, registry, bootstrap_repetitions=400, seed=5)
    rng = np.random.default_rng(5)
    variances = []
    for cell in CELLS:
        values = [r["difference"] for r in rows if (r["source_recipe"], r["origin_step"]) == cell]
        draws = rng.choice(values, size=(400, len(values)), replace=True)
        variances.append(np.var(draws, axis=1, ddof=1))
    expected = np.quantile(np.sqrt(np.mean(variances, axis=0)), 0.75)
    assert got["s_plan"] == expected


def test_sd_unidentified_uses_32_not_zero_for_equal_actions_or_few_lineages():
    rows, registry = planning_rows()
    for row in rows:
        row.update(difference=0, same_action=True)
    plan = plan_precision(rows, registry)
    assert plan["status"] == "SD_UNIDENTIFIED"
    assert plan["N"] == 32 and plan["N_req"] == 32 and plan["s_plan"] == 0.02
    few, few_registry = planning_rows(n_per_cell=1)
    assert plan_precision(few, few_registry)["N"] == 32


def test_precision_rejects_test_leakage_duplicate_lineages_and_missing_anchor():
    rows, registry = planning_rows()
    with pytest.raises(ValueError, match="after test"):
        plan_precision(rows, registry, test_outcomes_exist=True)
    with pytest.raises(ValueError, match="only development"):
        plan_precision([{**rows[0], "role": "locked_test"}, *rows[1:]], registry)
    with pytest.raises(ValueError, match="Duplicate"):
        plan_precision([*rows, rows[0]], registry)
    with pytest.raises(ValueError, match="Missing predetermined"):
        plan_precision(rows[1:], registry)


def test_mde_report_keeps_registered_maximum_without_chasing_significance():
    rows, registry = planning_rows()
    for row in rows:
        row["difference"] *= 10
    plan = plan_precision(rows, registry, bootstrap_repetitions=100)
    assert plan["N"] == 48 and plan["N_req"] > 48
    assert plan["remaining_mde_approx"] > 0.01
    assert plan["target_scale_attainable_approx"] is False


def test_mc_plans_experiment_mean_and_preserves_worst_case_at_saturation():
    result = plan_mc_samples(12)
    assert result["m"] == 16
    assert result["mc_se_upper"] == pytest.approx(math.sqrt(0.5 / (288 * 16 * 12 * 2)))
    assert result["status"] == "MC_BUDGET_MET"
    assert result["includes_training_or_scene_variation"] is False
    assert plan_mc_samples(12, n_prompts=1)["status"] == "MC_BUDGET_UNMET"
    with pytest.raises(ValueError, match="weights"):
        plan_mc_samples(12, n_prompts=2, prompt_weights=[0.8, 0.8])


def test_repeats_are_averaged_and_same_action_references_exact_same_runs():
    decisions, outcomes, registry = final_fixture(same_action=True)
    rows = aggregate_lineage_differences(decisions, outcomes, registry, require_real=False)
    assert len(rows) == 12
    assert all(r["difference"] == 0 and r["repeat_differences"] == [0, 0] for r in rows)
    assert all(
        ref["left"] == ref["right"] for r in rows for ref in r["physical_outcome_references"]
    )
    report = four_cell_comparison(rows, bootstrap_repetitions=100)
    assert report["status"] == "NO_OBSERVED_DIFFERENCE"
    assert report["all_selected_actions_identical"] is True
    assert report["interval"] is None and report["p_value"] is None
    assert not report["approximately_equivalent"]


def test_final_missing_repeat_duplicate_run_and_nonreal_outcomes_fail():
    decisions, outcomes, registry = final_fixture()
    with pytest.raises(ValueError, match="REAL_CUDA_MODEL"):
        aggregate_lineage_differences(decisions, outcomes, registry)
    needed = next(i for i, r in enumerate(outcomes) if r["recipe_id"] == "R3")
    with pytest.raises(ValueError, match="Missing registered H32"):
        aggregate_lineage_differences(
            decisions, outcomes[:needed] + outcomes[needed + 1 :], registry, require_real=False
        )
    with pytest.raises(ValueError, match="Duplicate physical"):
        aggregate_lineage_differences(
            decisions, [*outcomes, outcomes[0]], registry, require_real=False
        )
    bad = copy.deepcopy(decisions)
    bad[0]["made_before_run"] = False
    with pytest.raises(ValueError, match="prospective"):
        aggregate_lineage_differences(bad, outcomes, registry, require_real=False)


def test_four_cell_welch_weights_cells_not_raw_observation_counts():
    rows = []
    arrays = ([0.1, 0.2], [0, 0.03, 0.06], [-0.2, -0.1, 0, 0.1], [0.1, 0.2, 0.3])
    for index, (cell, values) in enumerate(zip(CELLS, arrays, strict=False)):
        rows.extend(
            {
                "lineage_id": f"{index}-{i}",
                "source_recipe": cell[0],
                "origin_step": cell[1],
                "difference": value,
            }
            for i, value in enumerate(values)
        )
    result = four_cell_comparison(rows, bootstrap_repetitions=200)
    estimate = np.mean([np.mean(a) for a in arrays])
    components = [np.var(a, ddof=1) / len(a) / 16 for a in arrays]
    variance = sum(components)
    df = variance**2 / sum(v * v / (len(a) - 1) for v, a in zip(components, arrays, strict=False))
    assert result["estimate"] == pytest.approx(estimate)
    assert result["degrees_of_freedom"] == pytest.approx(df)
    assert result["interval"] == pytest.approx(
        [
            estimate - t.ppf(0.975, df) * math.sqrt(variance),
            estimate + t.ppf(0.975, df) * math.sqrt(variance),
        ]
    )
    assert result["resample_tokens_or_draws"] is False
    assert result["n_independent_lineages"] == 12
    with pytest.raises(ValueError, match="Duplicate lineage"):
        four_cell_comparison([*rows, rows[0]])


def test_zero_variance_different_actions_does_not_assert_equivalence():
    rows, _ = planning_rows()
    for row in rows:
        row.update(difference=0, same_action=False)
    result = four_cell_comparison(rows, bootstrap_repetitions=100)
    assert result["status"] == "NO_OBSERVED_DIFFERENCE"
    assert result["interval"] is None and not result["approximately_equivalent"]
    assert not result["all_selected_actions_identical"]


def test_constant_nonzero_differences_do_not_gain_certainty_from_float_roundoff():
    rows, registry = planning_rows(n_per_cell=3)
    for row in rows:
        row["difference"] = 0.1
    result = four_cell_comparison(rows, bootstrap_repetitions=100)
    assert result["status"] == "ZERO_VARIANCE_UNRESOLVED"
    assert result["interval"] is None and result["p_value"] is None
    assert plan_precision(rows, registry)["status"] == "SD_UNIDENTIFIED"


def test_holm_keeps_unresolved_family_member_and_monotone_adjustment():
    result = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04, "d": None})["comparisons"]
    assert result["a"]["holm_p_value"] == 0.04
    assert result["b"]["holm_p_value"] == 0.09
    assert result["c"]["holm_p_value"] == 0.09
    assert result["a"]["reject"] and not result["b"]["reject"]
    assert result["d"]["holm_p_value"] is None and not result["d"]["reject"]


def test_fixture_analysis_never_claims_real_prospective_evidence():
    decisions, outcomes, registry = final_fixture()
    result = analyze_final(
        decisions, outcomes, registry, require_real=False, bootstrap_repetitions=100
    )
    assert result["status"] == "FIXTURE_ANALYSIS_NOT_EXPERIMENT"
    assert result["primary"]["n_independent_lineages"] == 12
    assert result["primary"]["estimate"] == pytest.approx(0.03)
    assert result["secondary_holm"]["family_size"] == len(SECONDARY) == 4
    assert not result["prospective_incremental_value_supported"]
