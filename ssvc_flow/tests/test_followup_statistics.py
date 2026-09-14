"""Independent CPU checks of the follow-up estimands and uncertainty contract."""

import copy
import importlib
import json
import math
from statistics import mean, stdev

import numpy as np
import pytest


def api():
    return importlib.import_module("src.followup_statistics")


def panel(scenes=4, samples=4, families=("a", "b")):
    rows = []
    for family in families:
        for scene in range(scenes):
            for interface in ("IMAGE_CUE_FRESH", "SYMBOLIC_FRESH"):
                prompt = f"{family}/{scene}/{interface}"
                for sample in range(samples):
                    rows.append(
                        {
                            "prompt_id": prompt,
                            "base_scene_id": f"{family}/{scene}",
                            "family": family,
                            "interface": interface,
                            "train_seed": 29,
                            "sample_seed": sample,
                            "bank_id": "composite",
                            "sample_key": f"{prompt}/{sample}",
                            "category": "X" if sample == 0 else "S" if sample <= scene else "I",
                        }
                    )
    return rows


def weights(rows):
    prompts = {r["prompt_id"] for r in rows}
    return {p: 1 / len(prompts) for p in prompts}


def count_panel():
    return [
        {
            "prompt_id": "p0",
            "base_scene_id": "s0",
            "family": "f",
            "interface": "i",
            "seed_block": "b",
            "n": 4,
            "counts": {"X": 1, "S": 0, "W": 0, "I": 3},
        },
        {
            "prompt_id": "p1",
            "base_scene_id": "s1",
            "family": "f",
            "interface": "i",
            "seed_block": "b",
            "n": 4,
            "counts": {"X": 1, "S": 3, "W": 0, "I": 0},
        },
    ]


def test_s01_s02_fixed_weight_counts_and_ratio_of_means():
    rows = count_panel()
    original = copy.deepcopy(rows)
    out = api().summarize_counts(rows, fixed_weights={"p0": 0.25, "p1": 0.75})
    assert out["pX"] == 0.25
    assert out["v"] == 0.8125
    assert out["qX"] == pytest.approx(4 / 13)
    assert out["qX"] != pytest.approx(0.25 * 1 + 0.75 * 0.25)
    assert out["rollout_count"] == 8
    assert out["category_counts"] == {"X": 2, "S": 3, "W": 0, "I": 3}
    assert rows == original


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r[0]["counts"].update(X=-1, I=5),
        lambda r: r[0]["counts"].update(Z=0),
        lambda r: r[0].update(n=9),
        lambda r: r[0]["counts"].update(X=0.5, I=3.5),
        lambda r: r[0]["counts"].update(X=True),
    ],
)
def test_s01_reject_invalid_counts(mutation):
    rows = count_panel()
    mutation(rows)
    with pytest.raises(ValueError):
        api().summarize_counts(rows, fixed_weights={"p0": 0.5, "p1": 0.5})


@pytest.mark.parametrize(
    "fixed",
    [{"p0": 1}, {"p0": 0.4, "p1": 0.4}, {"p0": -0.5, "p1": 1.5}, {"p0": float("nan"), "p1": 0.5}],
)
def test_s01_reject_invalid_fixed_weights(fixed):
    with pytest.raises(ValueError):
        api().summarize_counts(count_panel(), fixed_weights=fixed)


def test_s03_right_minus_left_probability_and_percentage_points():
    left = panel()
    right = [{**r, "category": "X" if r["category"] == "S" else r["category"]} for r in left]
    out = api().summarize_direct_response(
        left, right, fixed_weights=weights(left), bootstrap_replicates=41
    )
    assert out["direction"] == "right-left"
    assert out["delta_probability"]["pX"] == 0.375
    assert out["delta_pp"]["pX"] == 37.5
    assert out["left"]["estimator"] == "DIRECT_SAMPLING"
    assert out["safety_status"] == "NOT_CERTIFIED"
    assert out["fixed_panel"]["uncertainty"] != out["scene_bootstrap"]["uncertainty"]
    assert len(out["per_scope"]) == 5
    assert out["per_scope"]["a/SYMBOLIC_FRESH"]["delta_pp"]["pX"] == 37.5
    assert out["fixed_panel"]["union_M"] == 2 * 5 * 5
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_id", "changed"),
        ("base_scene_id", "changed"),
        ("interface", "changed"),
        ("family", "changed"),
        ("sample_seed", 100),
        ("train_seed", 41),
        ("bank_id", "other-bank"),
    ],
)
def test_s04_reject_unpaired_coverage(field, value):
    left = panel()
    right = copy.deepcopy(left)
    right[0][field] = value
    with pytest.raises(ValueError):
        api().summarize_direct_response(
            left, right, fixed_weights=weights(left), bootstrap_replicates=11
        )


def test_s04_reject_missing_or_duplicate_samples_and_cross_bank_baseline():
    left = panel()
    for right in (left[:-1], [*left, left[0]], [{**r, "bank_id": "different"} for r in left]):
        with pytest.raises(ValueError):
            api().summarize_direct_response(left, right, fixed_weights=weights(left))


def test_s05_shared_family_scene_draws_recompute_ratios_without_inner_resampling():
    rows = panel()
    right = [{**r, "category": "X" if r["category"] == "S" else r["category"]} for r in rows]
    repeats, seed = 137, 77
    out = api().paired_scene_bootstrap(
        {"base": rows, "candidate": right},
        fixed_weights=weights(rows),
        baseline_key="base",
        repeats=repeats,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    draws = {f: rng.integers(0, 4, size=(repeats, 4)) for f in ("a", "b")}
    expected_v = np.mean([((d + 1) / 4).mean(axis=1) for d in draws.values()], axis=0)
    expected_q = 0.25 / expected_v
    ci = out["candidates"]["base"]["overall"]["qX"]
    assert [ci["low"], ci["high"]] == pytest.approx(np.quantile(expected_q, [0.025, 0.975]))
    assert out["scene_draws"] == {f: d.tolist() for f, d in draws.items()}
    base = out["candidates"]["base"]
    assert base["a/IMAGE_CUE_FRESH"] == base["a/SYMBOLIC_FRESH"]
    candidate_delta = out["deltas"]["candidate"]["overall"]["pX"]
    assert [candidate_delta["low"], candidate_delta["high"]] == pytest.approx(
        np.quantile(expected_v - 0.25, [0.025, 0.975])
    )
    assert out["contract"]["candidate_draws_shared"]
    assert out["contract"]["interfaces_paired_within_scene"]
    assert not out["contract"]["within_scene_outputs_resampled"]


def test_s06_zero_event_support_and_undefined_ratio_remain_distinct():
    rows = [{**r, "category": "I"} for r in panel()]
    out = api().summarize_direct_response(
        rows, rows, fixed_weights=weights(rows), bootstrap_replicates=31
    )
    assert out["left"]["pX"] == 0
    assert out["left"]["qX"] is None
    assert out["left"]["event_support"]["X"] == "UNOBSERVED"
    assert out["left"]["prompts_without_X"] == len(weights(rows))
    ci = out["scene_bootstrap"]["candidates"]["left"]["overall"]["pX"]
    assert ci["status"] == "UNINFORMATIVE"
    assert ci["low"] is None and ci["high"] is None
    assert out["fixed_panel"]["candidates"]["left"]["pX"]["high"] > 0
    assert out["fixed_panel"]["candidates"]["left"]["qX"]["status"] == "UNINFORMATIVE"


def test_s07_hoeffding_union_and_ratio_difference_against_standard_library():
    left = count_panel()
    right = copy.deepcopy(left)
    right[0]["counts"] = {"X": 2, "S": 0, "W": 0, "I": 2}
    # Larger n produces a nontrivial lower bound for valid mass.
    for rows in (left, right):
        for row in rows:
            row["n"] *= 1000
            row["counts"] = {k: v * 1000 for k, v in row["counts"].items()}
    w = {"p0": 0.25, "p1": 0.75}
    out = api().fixed_panel_bounds(
        {"left": left, "right": right}, fixed_weights=w, baseline_key="left", alpha=0.05
    )
    assert out["union_M"] == 10  # pX, pS, pW, pI, v for each candidate.
    half = math.sqrt(0.5 * (0.25**2 / 4000 + 0.75**2 / 4000) * math.log(20 / 0.05))
    assert out["candidates"]["left"]["half_width"] == pytest.approx(half)
    low_x, high_x = 0.25 - half, 0.25 + half
    low_v, high_v = 0.8125 - half, 0.8125 + half
    ci = out["candidates"]["left"]["qX"]
    assert ci["low"] == pytest.approx(low_x / high_v)
    assert ci["high"] == pytest.approx(high_x / low_v)
    delta = out["deltas"]["right"]["pX"]
    assert delta["low"] == pytest.approx(0.0625 - 2 * half)
    assert delta["high"] == pytest.approx(0.0625 + 2 * half)
    with pytest.raises(ValueError, match="union"):
        api().fixed_panel_bounds({"left": left, "right": right}, fixed_weights=w, union_M=9)


def test_s08_s09_uniform_16_overlap_does_not_imply_event_support():
    out = api().importance_sampling_diagnostics([0.0] * 16, ["I"] * 16)
    assert out["ESS"] == pytest.approx(16)
    assert out["ESS_fraction"] == pytest.approx(1)
    assert out["max_normalized_weight"] == pytest.approx(1 / 16)
    assert out["n_times_max_normalized_weight"] == pytest.approx(1)
    assert "OVERLAP_WARNING" not in out["warnings"]
    assert out["events"]["X"]["support"] == "UNOBSERVED"
    assert out["events"]["X"]["ESS"] is None
    assert out["safety_status"] == "NOT_CERTIFIED"


def test_s10_extreme_logweights_stay_stable_and_json_finite():
    out = api().importance_sampling_diagnostics([10000.0, 10000.0, -10000.0], ["X", "S", "I"])
    assert out["ESS"] == pytest.approx(2)
    assert out["mean_weight"] is None
    assert out["mean_weight_status"] == "OVERFLOW"
    assert out["log_mean_weight"] == pytest.approx(10000 + math.log(2 / 3))
    assert out["SNIS"]["pX"] == pytest.approx(0.5)
    assert out["OIS"]["pX"] is None
    assert out["OIS"]["log_probabilities"]["pX"] == pytest.approx(10000 - math.log(3))
    json.dumps(out, allow_nan=False)
    under = api().importance_sampling_diagnostics([-10000.0] * 4, ["X"] * 4)
    assert under["mean_weight_status"] == "UNDERFLOW"
    assert under["SNIS"]["pX"] == 1
    assert under["OIS"]["pX"] is None


@pytest.mark.parametrize("bad", [[float("nan")], [float("inf")], [-float("inf")], [], [[0.0]]])
def test_s10_nonfinite_or_invalid_logweights_rejected(bad):
    with pytest.raises(ValueError):
        api().importance_sampling_diagnostics(bad, ["X"])


def test_s11_ois_snis_direct_are_explicit_and_no_clipping():
    out = api().importance_sampling_diagnostics([math.log(8), math.log(2)], ["X", "I"])
    assert out["OIS"]["estimator"] == "ORDINARY_IMPORTANCE_SAMPLING"
    assert out["OIS"]["pX"] == pytest.approx(4)
    assert out["SNIS"]["pX"] == pytest.approx(0.8)
    assert out["DIRECT"]["pX"] == 0.5
    assert "OIS_OUTSIDE_PROBABILITY_RANGE" in out["warnings"]
    event = out["events"]["X"]
    assert event["count"] == 1 and event["ESS"] == 1


def test_s12_aliases_reuse_raw_samples_and_not_independent_candidates_or_seeds():
    rows = panel()
    out = api().summarize_candidate_panel(
        {"base": rows},
        fixed_weights=weights(rows),
        baseline_key="base",
        aliases={"clone": "base", "clone2": "clone"},
        bootstrap_replicates=17,
    )
    assert out["independent_candidate_count"] == 1
    assert out["unique_rollout_count"] == len(rows)
    assert out["aliases"] == {"clone": "base", "clone2": "base"}
    assert out["fixed_panel"]["union_M"] == 5 * 5  # Five events in each reported scope.
    assert out["independent_training_seeds"] == [29]
    with pytest.raises(ValueError):
        api().resolve_aliases(["base"], {"a": "b", "b": "a"})
    with pytest.raises(ValueError):
        api().resolve_aliases(["base"], {"a": "missing"})


def test_s13_per_seed_paired_deltas_and_exploratory_prospective_labels():
    records = []
    for seed, base, arm in [(17, 0.2, 0.3), (29, 0.4, 0.35), (41, 0.5, 0.7)]:
        records.extend(
            [
                {"seed": seed, "arm": "joint0", "metrics": {"pX": base}},
                {"seed": seed, "arm": "joint1", "metrics": {"pX": arm}},
            ]
        )
    out = api().summarize_seed_results(records, baseline_arm="joint0")
    expected = [0.3 - 0.2, 0.35 - 0.4, 0.7 - 0.5]
    for entry, delta in zip(out["per_seed"], expected, strict=True):
        assert entry["arms"]["joint1"]["delta_probability"]["pX"] == delta
    assert out["per_seed"][0]["seed_role"] == "EXPLORATORY_ALREADY_OBSERVED"
    assert all(r["seed_role"] == "PROSPECTIVE" for r in out["per_seed"][1:])
    summary = out["aggregate"]["joint1"]["pX"]["paired_delta"]
    assert summary["mean"] == mean(expected)
    assert summary["std"] == stdev(expected)
    assert summary["directions"] == ["positive", "negative", "positive"]
    assert summary["all_same_direction"] is False
    assert out["prospective_aggregate"]["joint1"]["pX"]["paired_delta"]["count"] == 2
    assert out["uncertainty"] == "TRAINING_SEED_DESCRIPTIVE_SUMMARY"


def test_s13_seed_coverage_and_alias_records_cannot_inflate_seed_counts():
    records = [
        {"seed": 29, "arm": "base", "metrics": {"pX": 0.2}},
        {"seed": 29, "arm": "new", "metrics": {"pX": 0.3}},
        {"seed": 29, "arm": "alias", "alias_of": "new"},
    ]
    out = api().summarize_seed_results(records, baseline_arm="base")
    assert out["training_seed_count"] == 1
    assert set(out["aggregate"]) == {"base", "new"}
    records.append({"seed": 41, "arm": "base", "metrics": {"pX": 0.2}})
    with pytest.raises(ValueError):
        api().summarize_seed_results(records, baseline_arm="base")


def test_s14_few_scenes_and_undefined_replicates_never_filtered():
    one = panel(scenes=1)
    out = api().paired_scene_bootstrap({"base": one}, fixed_weights=weights(one))
    assert out["status"] == "UNINFORMATIVE"
    assert "scene" in out["reason"].lower()
    rows = panel(scenes=2, families=("a",))
    rows = [{**r, "category": "I" if r["base_scene_id"] == "a/0" else "X"} for r in rows]
    out = api().paired_scene_bootstrap(
        {"base": rows}, fixed_weights=weights(rows), repeats=101, seed=17
    )
    ci = out["candidates"]["base"]["overall"]["qX"]
    assert ci["status"] == "UNINFORMATIVE"
    assert ci["undefined_replicates"] > 0
    assert ci["defined_replicates"] + ci["undefined_replicates"] == 101
    assert ci["low"] is None and ci["high"] is None


def test_s07_group_bounds_normalize_weights_and_union_covers_all_groups():
    rows = panel()
    fixed = weights(rows)
    out = api().fixed_panel_bounds({"base": rows}, fixed_weights=fixed, include_groups=True)
    assert out["union_M"] == 25
    group = out["by_scope"]["a/SYMBOLIC_FRESH"]["candidates"]["base"]
    expected_half = math.sqrt(0.5 * (4 * 0.25**2 / 4) * math.log(50 / 0.05))
    assert group["half_width"] == pytest.approx(expected_half)
    assert out["by_scope"]["overall"]["candidates"] == out["candidates"]


def test_s10_s11_valid_ratio_stays_defined_for_tiny_valid_importance_mass():
    out = api().importance_sampling_diagnostics([-10000.0, -10000.0, 10000.0], ["X", "S", "I"])
    assert out["OIS"]["v"] is None
    assert out["OIS"]["qX"] == pytest.approx(0.5)
    assert out["SNIS"]["qX"] == pytest.approx(0.5)
    assert out["DIRECT"]["qX"] == 0.5


def test_s05_unequal_weights_keep_family_mass_and_resample_whole_scenes():
    rows = panel()
    scene_weight = [0.1, 0.2, 0.3, 0.4]
    fixed = {
        prompt: (0.25 if prompt.startswith("a/") else 0.75)
        * scene_weight[int(prompt.split("/")[1])]
        / 2
        for prompt in weights(rows)
    }
    out = api().paired_scene_bootstrap({"base": rows}, fixed_weights=fixed, repeats=173, seed=13)
    rng = np.random.default_rng(13)
    draws = {family: rng.integers(0, 4, size=(173, 4)) for family in ("a", "b")}
    expected = []
    for replicate in range(173):
        valid = 0
        for family, family_mass in (("a", 0.25), ("b", 0.75)):
            chosen = draws[family][replicate]
            denominator = sum(scene_weight[i] for i in chosen)
            numerator = sum(scene_weight[i] * (i + 1) / 4 for i in chosen)
            valid += family_mass * numerator / denominator
        expected.append(0.25 / valid)
    ci = out["candidates"]["base"]["overall"]["qX"]
    assert [ci["low"], ci["high"]] == pytest.approx(np.quantile(expected, [0.025, 0.975]))


def test_s02_s03_s07_qs_ratio_and_intervals_use_same_union_events():
    rows = count_panel()
    fixed = {"p0": 0.25, "p1": 0.75}
    out = api().summarize_counts(rows, fixed_weights=fixed)
    assert out["qS"] == pytest.approx(9 / 13)
    for row in rows:
        row["n"] *= 1000
        row["counts"] = {key: value * 1000 for key, value in row["counts"].items()}
    bounds = api().fixed_panel_bounds({"base": rows}, fixed_weights=fixed)
    assert bounds["union_M"] == 5
    half = math.sqrt(0.5 * (0.25**2 / 4000 + 0.75**2 / 4000) * math.log(10 / 0.05))
    qs = bounds["candidates"]["base"]["qS"]
    assert [qs["low"], qs["high"]] == pytest.approx(
        [(0.5625 - half) / (0.8125 + half), (0.5625 + half) / (0.8125 - half)]
    )
    left = panel()
    right = [
        {**row, "category": "X" if row["category"] == "S" else row["category"]} for row in left
    ]
    response = api().summarize_direct_response(
        left, right, fixed_weights=weights(left), bootstrap_replicates=41
    )
    assert response["delta_probability"]["qS"] == pytest.approx(-0.6)
    assert response["delta_pp"]["qS"] == pytest.approx(-60)
    assert "qS" in response["scene_bootstrap"]["candidates"]["left"]["overall"]
    zeros = [{**row, "category": "I"} for row in left]
    assert api().summarize_counts(zeros, fixed_weights=weights(zeros))["qS"] is None
    importance = api().importance_sampling_diagnostics([0.0, 0.0, 0.0], ["X", "S", "I"])
    for estimator in ("OIS", "SNIS", "DIRECT"):
        assert importance[estimator]["qS"] == pytest.approx(0.5)
