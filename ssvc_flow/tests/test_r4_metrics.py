"""R4 pure-CPU KL and paired endpoint statistics; no new rollouts."""

import copy
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_r4_metrics_api_exists():
    assert importlib.util.find_spec("src.r4_metrics") is not None


@pytest.fixture
def api():
    return importlib.import_module("src.r4_metrics")


FAMILIES = ("cross_series", "duplicate", "trend")
INTERFACES = ("IMAGE_CUE_FRESH", "SYMBOLIC_FRESH")


def control(lengths=(1, 3)):
    rows = []
    for family in FAMILIES:
        for interface in INTERFACES:
            for i, length in enumerate(lengths):
                prompt = f"{family}-p-{interface}"
                rows.append(
                    {
                        "sample_key": f"{prompt}-{i}",
                        "prompt_id": prompt,
                        "base_scene_id": f"{family}-p",
                        "family": family,
                        "interface": interface,
                        "token_ids": [1] * (length - 1) + [2],
                        "old_logprobs": [-3.0] * length,
                        "stop_reason": "eos",
                    }
                )
    return rows


def scored(rows, delta):
    return {
        r["sample_key"]: [
            p + (delta(r, i) if callable(delta) else delta) for i, p in enumerate(r["old_logprobs"])
        ]
        for r in rows
    }


def test_kl_is_conditional_token_mean_with_eos_and_prompt_weighting(api):
    rows = control()
    values = scored(rows, lambda r, i: 0.2 if len(r["token_ids"]) == 1 else 0.0)
    result = api.control_kl_diagnostic(rows, values, eos_token_ids=[2])
    assert result["mean_token_kl"] == pytest.approx((np.expm1(0.2) - 0.2) / 4)
    assert result["denominator"]["selected_tokens"] == 24
    assert result["denominator"]["included_eos_tokens"] == 12
    assert result["denominator"]["prompt_count"] == 6
    assert not result["should_stop"]
    assert (
        result["direction"] == "KL(step0_token_policy || candidate_token_policy) at step0 prefixes"
    )
    assert not result["trajectory_kl"] and not result["candidate_state_occupancy_kl"]
    assert len(result["sequence_records"]) == 12
    assert all(p["selected_token_count"] == 4 for p in result["by_prompt"].values())
    json.dumps(result, allow_nan=False)


def test_kl_prompt_counts_do_not_change_fixed_six_group_weights(api):
    rows = control(lengths=(1,))
    p = rows[0]
    rows.extend({**p, "sample_key": f"extra-{i}"} for i in range(20))
    result = api.control_kl_diagnostic(
        rows,
        scored(rows, lambda r, i: 0.2 if r["prompt_id"] == p["prompt_id"] else 0),
        eos_token_ids=[2],
    )
    assert result["mean_token_kl"] == pytest.approx((np.expm1(0.2) - 0.2) / 6)


def test_kl_stop_thresholds_use_strict_greater_and_distinct_alarms(api):
    rows = control(lengths=(1,))
    exactly_two = api.control_kl_diagnostic(rows, scored(rows, 2.0), eos_token_ids=[2])
    assert exactly_two["sequence_log_ratio_p99_abs"] == 2.0
    assert not exactly_two["alarms"]["sequence_log_ratio_p99_abs"]
    assert exactly_two["alarms"]["mean_token_kl"]
    rows = control(lengths=(64,))
    sequence_alarm = api.control_kl_diagnostic(rows, scored(rows, 0.04), eos_token_ids=[2])
    assert not sequence_alarm["alarms"]["mean_token_kl"]
    assert sequence_alarm["alarms"]["sequence_log_ratio_p99_abs"]
    assert sequence_alarm["should_stop"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1000.0])
def test_kl_nonfinite_and_exp_overflow_fail_without_clipping(api, bad):
    rows = control()
    new = scored(rows, 0)
    new[rows[0]["sample_key"]][0] = bad
    with pytest.raises((ValueError, FloatingPointError)):
        api.control_kl_diagnostic(rows, new, eos_token_ids=[2])


def test_kl_rejects_action_key_and_eos_mismatch(api):
    rows = control()
    new = scored(rows, 0)
    with pytest.raises(ValueError):
        api.control_kl_diagnostic(rows, new, eos_token_ids=[99])
    new.pop(rows[0]["sample_key"])
    with pytest.raises(ValueError):
        api.control_kl_diagnostic(rows, new, eos_token_ids=[2])


def endpoints(scenes=10, track="N", families=FAMILIES):
    rows, initial = [], []
    patterns = (
        ("X", "I", "I", "I"),
        ("X", "S", "I", "I"),
        ("X", "X", "S", "I"),
        ("S", "W", "X", "X"),
    )
    interfaces = ("collision", "separating") if track == "L" else INTERFACES
    for family in families:
        for s in range(scenes):
            for interface in interfaces:
                prompt = f"{family}-{s}-{interface}"
                meta = {
                    "prompt_id": prompt,
                    "base_scene_id": f"{family}-{s}",
                    "family": family,
                    "interface": interface,
                    "track": track,
                    "decode_mode": "sample",
                }
                base = list(patterns[s % 4])
                valid = ["S", "S", *base[2:]]
                for arm, values in (("X_BASE", base), ("X_VALID", valid)):
                    for j, c in enumerate(values):
                        rows.append(
                            {
                                **meta,
                                "arm": arm,
                                "checkpoint_step": 64,
                                "sample_key": f"{prompt}-{arm}-{j}",
                                "category": c,
                            }
                        )
                for j, c in enumerate(patterns[0]):
                    initial.append(
                        {
                            **meta,
                            "arm": "INITIAL",
                            "checkpoint_step": 0,
                            "sample_key": f"{prompt}-init-{j}",
                            "category": c,
                        }
                    )
    return rows, initial


def analyze(api, rows, initial=None, **kwargs):
    return api.analyze_sampled_endpoints(
        rows, initial, track=kwargs.pop("track", "N"), bootstrap_replicates=257, **kwargs
    )


def test_endpoint_counts_fixed_prompt_weights_and_paired_arms(api):
    rows, initial = endpoints()
    result = analyze(api, rows, initial)
    comparison = result["comparisons"]["X_VALID_minus_X_BASE"]
    metric = comparison["responses"]["overall"]["pX"]
    assert metric["estimate"] == pytest.approx(-0.25)
    assert comparison["panel"]["base_scenes"] == 30
    assert comparison["panel"]["groups"] == [f"{f}/{i}" for f in FAMILIES for i in INTERFACES]
    assert result["bootstrap_contract"]["interfaces_and_arms_paired_within_scene"]
    assert result["bootstrap_contract"]["training_seed_uncertainty_included"] is False
    one_prompt = rows[0]["prompt_id"]
    copied = [
        {**r, "sample_key": f"copy{j}-{r['sample_key']}"}
        for j in range(7)
        for r in rows
        if r["prompt_id"] == one_prompt
    ]
    repeated = analyze(api, rows + copied, initial)
    assert repeated["comparisons"]["X_VALID_minus_X_BASE"]["responses"] == comparison["responses"]
    assert (
        repeated["comparisons"]["X_BASE_minus_initial"]["responses"]
        == result["comparisons"]["X_BASE_minus_initial"]["responses"]
    )
    json.dumps(result, allow_nan=False)


def test_initial_change_uses_only_preexisting_initial_panel(api):
    rows, initial = endpoints()
    initial = [r for r in initial if r["base_scene_id"].endswith(("-0", "-1"))]
    result = analyze(api, rows, initial)
    full = result["comparisons"]["X_VALID_minus_X_BASE"]
    change = result["comparisons"]["X_BASE_minus_initial"]
    assert full["panel"]["base_scenes"] == 30
    assert change["panel"]["base_scenes"] == 6
    assert change["responses"]["overall"]["pX"]["estimate"] == 0
    assert change["panel_role"] == "FIXED_INITIAL_PANEL_ONLY"


def test_six_group_max_stat_band_is_not_pointwise_percentile_ci(api):
    rows, _ = endpoints()
    result = analyze(api, rows)
    comparison = result["comparisons"]["X_VALID_minus_X_BASE"]
    joint = comparison["simultaneous_contract"]["pX"]
    assert joint["method"] == "BOOTSTRAP_CENTERED_MAX_ABS_DEVIATION"
    assert joint["group_count"] == 6
    assert joint["coverage_scope"] == "ONE_METRIC_ACROSS_DECLARED_GROUPS_ONLY"
    rng = np.random.default_rng(20260909)
    # Each scene's pX change is -1/4,-1/4,-1/2,0 in both paired interfaces.
    differences = np.asarray([-0.25, -0.25, -0.5, 0] * 3)[:10]
    deviations = []
    for _ in FAMILIES:
        idx = rng.integers(0, 10, size=(257, 10))
        deviations.append(np.abs(differences[idx].mean(axis=1) - differences.mean()))
    critical = np.quantile(np.max(deviations, axis=0), 0.95)
    assert joint["critical_absolute_deviation"] == pytest.approx(critical)
    metric = comparison["responses"]["trend/IMAGE_CUE_FRESH"]["pX"]
    assert metric["simultaneous_ci"]["low"] == pytest.approx(metric["estimate"] - critical)
    assert metric["pointwise_ci"]["method"] == "PAIRED_SCENE_CLUSTER_PERCENTILE"
    assert comparison["responses"]["overall"]["pX"]["simultaneous_ci"]["status"] == "NOT_INCLUDED"


def test_decline_exclusion_uses_interval_bound_and_does_not_mean_no_harm(api):
    rows, _ = endpoints()
    result = analyze(api, rows)
    metric = result["comparisons"]["X_VALID_minus_X_BASE"]["responses"]["overall"]["pX"]
    assert metric["point_direction"] == "NEGATIVE_POINT_TREND"
    assert metric["pointwise_evidence"] == "NEGATIVE_INTERVAL_EXCLUDES_ZERO"
    assert metric["excluded_decline_magnitude_above_pointwise"] == pytest.approx(
        max(0, -metric["pointwise_ci"]["low"])
    )
    assert metric["absence_of_significance_implies_no_harm"] is False


def test_degenerate_bootstrap_is_unstable_and_not_evidence_for_reversal(api):
    rows, _ = endpoints()
    for row in rows:
        index = int(row["sample_key"].rsplit("-", 1)[1])
        row["category"] = (
            ("X", "X", "I", "I")[index] if row["arm"] == "X_BASE" else ("X", "S", "S", "I")[index]
        )
    result = analyze(api, rows)
    comparison = result["comparisons"]["X_VALID_minus_X_BASE"]
    metric = comparison["responses"]["overall"]["pX"]
    assert metric["pointwise_ci"]["status"] == "UNSTABLE"
    assert metric["pointwise_evidence"] == "UNSTABLE_NO_INFERENCE"
    assert metric["excluded_decline_magnitude_above_pointwise"] is None
    reversal = comparison["reversal_diagnostics"]["overall"]
    assert reversal["point_estimate_reversal_candidate"]
    assert not reversal["joint_statistical_evidence"]


def test_zero_denominator_never_gets_smoothed_and_small_panels_are_unstable(api):
    rows, _ = endpoints(scenes=1)
    for row in rows:
        row["category"] = "I"
    result = analyze(api, rows)
    metric = result["comparisons"]["X_VALID_minus_X_BASE"]["responses"]["overall"]["qX_pool"]
    assert metric["estimate"] is None
    assert metric["pointwise_ci"]["status"] == "UNSTABLE"
    assert metric["pointwise_ci"]["low"] is None
    assert metric["pointwise_ci"]["undefined_replicates"] == 257


def test_protocols_are_separate_and_ood_does_not_claim_six_families(api):
    rows, initial = endpoints(track="L")
    result = analyze(api, rows, initial, track="L")
    assert result["track"] == "L"
    bad_initial = copy.deepcopy(initial)
    bad_initial[0]["track"] = "N"
    with pytest.raises(ValueError, match="track"):
        analyze(api, rows, bad_initial, track="L")
    ood, _ = endpoints(track="OOD", families=("cross_series",))
    result = analyze(api, ood, track="OOD")
    assert (
        result["comparisons"]["X_VALID_minus_X_BASE"]["simultaneous_contract"]["pX"]["group_count"]
        == 2
    )
    assert result["comparisons"]["X_BASE_minus_initial"]["status"] == "NOT_MEASURED"
    assert result["generalization_scope"] == "CROSS_SERIES_GRAPH_STRUCTURE_ONLY"


@pytest.mark.parametrize(
    "kind", ["duplicate", "greedy", "arm_missing_prompt", "initial_extra", "unpaired", "step"]
)
def test_bad_endpoint_pairings_are_rejected(api, kind):
    rows, initial = endpoints()
    if kind == "duplicate":
        rows.append(dict(rows[0]))
    elif kind == "greedy":
        rows[0]["decode_mode"] = "greedy"
    elif kind == "arm_missing_prompt":
        rows = [
            r
            for r in rows
            if not (r["arm"] == "X_VALID" and r["prompt_id"] == rows[0]["prompt_id"])
        ]
    elif kind == "initial_extra":
        initial[0]["prompt_id"] = "missing-in-endpoint"
    elif kind == "unpaired":
        rows = [r for r in rows if r["prompt_id"] != rows[0]["prompt_id"]]
    else:
        rows[0]["checkpoint_step"] = 32
    with pytest.raises(ValueError):
        analyze(api, rows, initial)


def test_l_primary_weights_follow_fixed_prompt_distribution_and_equal_groups_are_sensitivity(api):
    rows, initial = endpoints(scenes=12, track="L")
    # Duplicate has 4 scenes; the other two protocol families have 12 each.
    rows = [
        r
        for r in rows
        if r["family"] != "duplicate" or int(r["base_scene_id"].rsplit("-", 1)[1]) < 4
    ]
    initial = [
        r
        for r in initial
        if r["family"] != "duplicate" or int(r["base_scene_id"].rsplit("-", 1)[1]) < 4
    ]
    for row in rows:
        row["category"] = "X" if row["arm"] == "X_VALID" and row["family"] == "duplicate" else "I"
    primary = analyze(api, rows, initial, track="L")
    comparison = primary["comparisons"]["X_VALID_minus_X_BASE"]
    assert comparison["responses"]["overall"]["pX"]["estimate"] == pytest.approx(4 / 28)
    assert comparison["panel"]["fixed_group_weights"]["duplicate/collision"] == pytest.approx(
        4 / 56
    )
    assert comparison["panel"]["weighting_role"] == "PRIMARY_FIXED_PROMPT_DISTRIBUTION"
    equal = {g: 1 / 6 for g in comparison["panel"]["groups"]}
    sensitivity = analyze(api, rows, initial, track="L", group_weights=equal)
    change = sensitivity["comparisons"]["X_VALID_minus_X_BASE"]
    assert change["responses"]["overall"]["pX"]["estimate"] == pytest.approx(1 / 3)
    assert change["panel"]["weighting_role"] == "EXPLICIT_WEIGHT_SENSITIVITY"


def test_endpoint_checkpoint_step_is_required_but_historical_initial_can_omit_it(api):
    rows, initial = endpoints()
    for row in initial:
        row.pop("checkpoint_step")
    analyze(api, rows, initial)
    for row in rows:
        row.pop("checkpoint_step")
    with pytest.raises(ValueError, match="step"):
        analyze(api, rows, initial)


def test_endpoint_pool_ratio_is_recomputed_inside_every_cluster_draw(api):
    rows, _ = endpoints()
    result = analyze(api, rows)
    metric = result["comparisons"]["X_VALID_minus_X_BASE"]["responses"][
        "cross_series/IMAGE_CUE_FRESH"
    ]["qX_pool"]
    values = {}
    for arm in ("X_BASE", "X_VALID"):
        components = []
        for scene in range(10):
            group = [
                r
                for r in rows
                if r["arm"] == arm
                and r["base_scene_id"] == f"cross_series-{scene}"
                and r["interface"] == "IMAGE_CUE_FRESH"
            ]
            components.append(
                [
                    sum(r["category"] == "X" for r in group) / len(group),
                    sum(r["category"] != "I" for r in group) / len(group),
                ]
            )
        values[arm] = np.asarray(components)
    expected = (
        values["X_VALID"].mean(0)[0] / values["X_VALID"].mean(0)[1]
        - values["X_BASE"].mean(0)[0] / values["X_BASE"].mean(0)[1]
    )
    assert metric["estimate"] == pytest.approx(expected)
    idx = np.random.default_rng(20260909).integers(0, 10, size=(257, 10))
    left = values["X_VALID"][idx].mean(axis=1)
    right = values["X_BASE"][idx].mean(axis=1)
    expected_ci = np.quantile(left[:, 0] / left[:, 1] - right[:, 0] / right[:, 1], [0.025, 0.975])
    assert metric["pointwise_ci"]["low"] == pytest.approx(expected_ci[0])
    assert metric["pointwise_ci"]["high"] == pytest.approx(expected_ci[1])
    wrong_macro = np.mean(
        values["X_VALID"][:, 0] / values["X_VALID"][:, 1]
        - values["X_BASE"][:, 0] / values["X_BASE"][:, 1]
    )
    assert metric["estimate"] != pytest.approx(wrong_macro)


def _sensitivity_inputs():
    rows, initial = endpoints(scenes=10, track="L")
    for row in rows:
        scene = int(row["base_scene_id"].rsplit("-", 1)[1])
        sample = int(row["sample_key"].rsplit("-", 1)[1])
        offset = 2 * FAMILIES.index(row["family"]) + int(row["interface"] == "separating")
        score = (scene + sample + offset + int(row["arm"] == "X_VALID")) % 11
        row["category"] = "X" if score < 2 else "S" if score < 5 else "W" if score < 8 else "I"
    return rows, initial


def test_l_sensitivity_weights_reordering_is_bitwise_stable(api):
    from src.core import canonical_hash

    rows, initial = _sensitivity_inputs()
    groups = sorted({r["family"] + "/" + r["interface"] for r in rows})
    weights = {g: 1 / 6 for g in groups}
    reversed_weights = {g: weights[g] for g in reversed(groups)}
    first = analyze(api, rows, initial, track="L", group_weights=weights)
    second = analyze(api, rows, initial, track="L", group_weights=reversed_weights)
    assert canonical_hash(first) == canonical_hash(second)


def test_l_sensitivity_report_is_bitwise_stable_across_python_hash_seeds():
    script = f"""
import importlib.util
import json
from src.core import canonical_hash
from src.r4_metrics import analyze_sampled_endpoints
spec = importlib.util.spec_from_file_location("r4_fixture", {str(Path(__file__).resolve())!r})
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
rows, initial = fixture._sensitivity_inputs()
groups = {{r["family"] + "/" + r["interface"] for r in rows}}
weights = {{g: 1 / len(groups) for g in groups}}
result = analyze_sampled_endpoints(rows, initial, track="L", group_weights=weights)
print(json.dumps({{"order": list(weights), "hash": canonical_hash(result)}}))
"""
    reports = [
        json.loads(
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONHASHSEED": seed},
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        for seed in ("0", "1", "2")
    ]
    assert len({tuple(r["order"]) for r in reports}) > 1
    assert len({r["hash"] for r in reports}) == 1
