"""Pure CPU control-bank importance statistics; no model execution."""

import copy
import importlib
import importlib.util
import json

import numpy as np
import pytest


def test_response_api_exists():
    assert importlib.util.find_spec("src.r3_response") is not None


@pytest.fixture
def api():
    return importlib.import_module("src.r3_response")


FAMILIES = ("cross_series", "duplicate", "trend")
INTERFACES = ("IMAGE_CUE_FRESH", "SYMBOLIC_FRESH")


def panel(scenes=2, samples=4):
    rows = []
    for family in FAMILIES:
        for scene in range(scenes):
            for interface in INTERFACES:
                prompt = f"{family}-{scene}-{interface}"
                for sample in range(samples):
                    # Each scene has one X and a different valid denominator.
                    category = "X" if sample == 0 else "S" if sample <= scene else "I"
                    rows.append(
                        {
                            "sample_key": f"{prompt}-{sample}",
                            "prompt_id": prompt,
                            "base_scene_id": f"{family}-{scene}",
                            "family": family,
                            "interface": interface,
                            "category": category,
                            "token_ids": [1] * (sample % 3 + 1),
                            "old_logprobs": [-10.0] * (sample % 3 + 1),
                        }
                    )
    return rows


def candidate(rows, weight):
    result = {}
    for row in rows:
        w = weight(row) if callable(weight) else weight
        values = list(row["old_logprobs"])
        values[0] += float(np.log(w))
        result[row["sample_key"]] = values
    return result


def analyze(api, rows, candidates, *, scenes=2, samples=4, repeats=101):
    return api.analyze_control_responses(
        rows,
        candidates,
        baseline_key="zero",
        bank_index=0,
        bootstrap_replicates=repeats,
        expected_base_scenes_per_family=scenes,
        expected_samples_per_prompt=samples,
    )


def test_ordinary_is_baseline_absolute_change_and_ratio_of_pools(api):
    rows = panel()
    original = copy.deepcopy(rows)
    candidates = {"zero": candidate(rows, 2.0), "one": candidate(rows, 4.0)}
    result = analyze(api, rows, candidates)
    zero = result["candidates"]["zero"]["responses"]["overall"]
    one = result["candidates"]["one"]["responses"]["overall"]
    assert zero["pX"]["estimate"] == pytest.approx(0.5)
    assert zero["pX"]["absolute_delta"] == pytest.approx(0.25)
    assert zero["pX"]["auxiliary_delta"] == 0
    assert one["pX"]["estimate"] == pytest.approx(1.0)
    assert one["v"]["estimate"] == pytest.approx(1.5)
    assert one["pX"]["auxiliary_delta"] == pytest.approx(0.5)
    assert one["qX_pool"]["estimate"] == pytest.approx(2 / 3)
    assert one["qX_pool"]["estimate"] != pytest.approx((1.0 + 0.5) / 2)
    assert one["qS_pool"]["estimate"] == pytest.approx(1 / 3)
    assert one["pX"]["predicted_delta"] == one["pX"]["auxiliary_delta"]
    assert one["pX"]["observed_delta"] is None
    assert one["pX"]["prediction_residual"] is None
    assert result["method"] == "ORDINARY_UNCLIPPED_IMPORTANCE_SAMPLING"
    assert result["safety_status"] == "NOT_CERTIFIED"
    assert rows == original
    json.dumps(result, allow_nan=False)


def test_auxiliary_difference_is_computed_on_paired_sample_weights(api):
    rows = panel()
    baseline = candidate(rows, 1e12)
    current = candidate(rows, lambda r: 1e12 + (0.02 if r["category"] == "X" else 0))
    result = analyze(api, rows, {"zero": baseline, "one": current})
    deltas = []
    for row in rows:
        key = row["sample_key"]
        w0 = np.exp(np.sum(np.asarray(baseline[key]) - row["old_logprobs"]))
        w1 = np.exp(np.sum(np.asarray(current[key]) - row["old_logprobs"]))
        deltas.append((w1 - w0) * (row["category"] == "X"))
    expected = np.mean(deltas)
    actual = result["candidates"]["one"]["responses"]["overall"]["pX"]["auxiliary_delta"]
    assert actual == expected
    assert actual != pytest.approx(0, abs=1e-8)


def test_cluster_bootstrap_recomputes_ratios_and_pairs_interfaces(api):
    rows = panel(scenes=4)
    repeats = 173
    result = analyze(api, rows, {"zero": candidate(rows, 1.0)}, scenes=4, repeats=repeats)
    rng = np.random.default_rng(20260909)
    draws = {family: rng.integers(0, 4, size=(repeats, 4)) for family in FAMILIES}
    valid = np.arange(1, 5) / 4
    v_by_family = {family: valid[idx].mean(axis=1) for family, idx in draws.items()}
    pooled = 0.25 / np.mean(list(v_by_family.values()), axis=0)
    expected = np.quantile(pooled, [0.025, 0.975])
    metric = result["candidates"]["zero"]["responses"]["overall"]["qX_pool"]
    ci = metric["ci"]["estimate"]
    assert ci["low"] == pytest.approx(expected[0])
    assert ci["high"] == pytest.approx(expected[1])
    wrong_macro = np.mean([(0.25 / valid)[idx].mean(axis=1) for idx in draws.values()], axis=0)
    assert not np.allclose(expected, np.quantile(wrong_macro, [0.025, 0.975]))
    assert result["bootstrap_contract"]["interfaces_paired_within_scene"]
    assert result["bootstrap_contract"]["candidate_draws_shared"]
    assert not result["bootstrap_contract"]["within_scene_outputs_resampled"]
    # Identical interfaces use the same scene draw, including finite-replicate randomness.
    groups = result["candidates"]["zero"]["responses"]
    assert groups[f"trend/{INTERFACES[0]}"] == groups[f"trend/{INTERFACES[1]}"]


def test_prompt_overall_scope_warnings_at_registered_k16(api):
    rows = panel(scenes=8, samples=16)
    result = api.analyze_control_responses(
        rows,
        {"zero": candidate(rows, 1.0)},
        baseline_key="zero",
        bank_index=6,
        bootstrap_replicates=31,
    )
    diagnostics = result["candidates"]["zero"]["diagnostics"]
    prompt = next(iter(diagnostics["by_prompt"].values()))
    assert prompt["ESS_fraction"] == pytest.approx(1)
    assert prompt["max_normalized_weight"] == pytest.approx(1 / 16)
    assert "OVERLAP_WARNING" in prompt["warnings"]
    assert diagnostics["overall"]["max_normalized_weight"] == pytest.approx(1 / 768)
    assert "OVERLAP_WARNING" not in diagnostics["overall"]["warnings"]
    assert "OVERLAP_WARNING" in result["candidates"]["zero"]["warnings"]
    assert not result["candidates"]["zero"]["can_use_for_safety_decision"]


def test_missing_x_support_and_zero_valid_denominators_remain_na(api):
    rows = panel()
    for row in rows:
        row["category"] = "I"
    result = analyze(api, rows, {"zero": candidate(rows, 1.0)})
    candidate_result = result["candidates"]["zero"]
    metric = candidate_result["responses"]["overall"]
    assert metric["pX"]["estimate"] == 0
    assert metric["qX_pool"]["estimate"] is None
    assert metric["qX_pool"]["auxiliary_delta"] is None
    assert metric["qX_pool"]["ci"]["estimate"]["status"] == "NA"
    assert "NO_OBSERVED_X_SUPPORT" in candidate_result["warnings"]
    support = candidate_result["diagnostics"]["overall"]["categories"]["X"]
    assert support["count"] == 0 and support["ESS"] is None
    json.dumps(result, allow_nan=False)


def test_fixed_group_weights_do_not_follow_family_scene_counts(api):
    rows = panel(scenes=3)
    # Keep two scenes in one family, but preserve both interfaces and their rows.
    rows = [r for r in rows if r["base_scene_id"] != "cross_series-2"]
    candidates = {"zero": candidate(rows, lambda r: 3 if r["family"] == "cross_series" else 1)}
    result = analyze(api, rows, candidates, scenes=None)
    actual = result["candidates"]["zero"]["responses"]["overall"]["pX"]["estimate"]
    assert actual == pytest.approx((0.75 + 0.25 + 0.25) / 3)
    assert result["panel"]["overall_group_weights"] == {g: 1 / 6 for g in result["panel"]["groups"]}


@pytest.mark.parametrize(
    "kind", ["duplicate", "missing", "extra", "length", "unpaired", "metadata"]
)
def test_bad_sample_or_panel_bindings_are_rejected(api, kind):
    rows = panel()
    candidates = {"zero": candidate(rows, 1.0)}
    if kind == "duplicate":
        rows[1]["sample_key"] = rows[0]["sample_key"]
    elif kind == "missing":
        candidates["zero"].pop(rows[0]["sample_key"])
    elif kind == "extra":
        candidates["zero"]["extra"] = [0]
    elif kind == "length":
        candidates["zero"][rows[0]["sample_key"]].append(0)
    elif kind == "unpaired":
        rows = [r for r in rows if r["prompt_id"] != rows[0]["prompt_id"]]
        candidates = {"zero": candidate(rows, 1.0)}
    else:
        rows[0]["family"] = "trend"
    with pytest.raises(ValueError):
        analyze(api, rows, candidates)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1000.0])
def test_nonfinite_or_overflowing_likelihood_never_gets_clipped(api, bad):
    rows = panel()
    candidates = {"zero": candidate(rows, 1.0)}
    candidates["zero"][rows[0]["sample_key"]][0] = bad
    with pytest.raises((ValueError, FloatingPointError)):
        analyze(api, rows, candidates)


def test_unscored_bank_cannot_claim_response(api):
    rows = panel()
    with pytest.raises(ValueError, match="0 or 6"):
        api.analyze_control_responses(
            rows,
            {"zero": candidate(rows, 1)},
            baseline_key="zero",
            bank_index=3,
            expected_base_scenes_per_family=2,
            expected_samples_per_prompt=4,
        )


def test_undefined_bootstrap_draws_are_not_dropped_even_when_point_ratio_exists(api):
    rows = panel()
    for row in rows:
        if row["base_scene_id"].endswith("-0"):
            row["category"] = "I"
    result = analyze(api, rows, {"zero": candidate(rows, 1)}, repeats=173)
    metric = result["candidates"]["zero"]["responses"]["trend/IMAGE_CUE_FRESH"]["qX_pool"]
    assert metric["estimate"] == pytest.approx(0.5)
    assert metric["auxiliary_delta"] == 0
    assert metric["ci"]["estimate"]["status"] == "NA"
    assert metric["ci"]["estimate"]["undefined_replicates"] > 0
    assert metric["ci"]["estimate"]["low"] is None
    assert metric["ci"]["auxiliary_delta"]["status"] == "NA"


def test_ess_uses_raw_weights_and_counts_missing_categories(api):
    rows = panel()
    result = analyze(
        api, rows, {"zero": candidate(rows, lambda r: 100 if r["category"] == "X" else 1)}
    )
    diagnostic = next(iter(result["candidates"]["zero"]["diagnostics"]["by_prompt"].values()))
    assert diagnostic["ESS"] == pytest.approx(103**2 / (100**2 + 3))
    assert diagnostic["ESS_fraction"] < 0.5
    assert diagnostic["max_normalized_weight"] == pytest.approx(100 / 103)
    assert diagnostic["mean_weight"] == pytest.approx(103 / 4)
    assert diagnostic["categories"]["X"]["count"] == 1
    assert diagnostic["categories"]["X"]["ESS"] == pytest.approx(1)
    assert diagnostic["categories"]["W"]["count"] == 0
    assert diagnostic["categories"]["W"]["ESS"] is None
