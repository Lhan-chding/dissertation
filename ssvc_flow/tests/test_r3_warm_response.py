"""CPU checks for independently sampled warm responses and common scene bootstrap."""

import copy
import importlib
import importlib.util
import json

import numpy as np
import pytest

FAMILIES = ("cross_series", "duplicate", "trend")
INTERFACES = ("IMAGE_CUE_FRESH", "SYMBOLIC_FRESH")


def test_direct_validation_api_exists():
    assert importlib.util.find_spec("src.r3_warm_response") is not None


def _categories(counts):
    return [category for category, count in zip("XSWI", counts, strict=True) for _ in range(count)]


def case(*, proposal_counts=None, baseline_counts=None, candidate_counts=None, weight=None):
    proposal_counts = proposal_counts or (lambda scene: [4, 4, 4, 4])
    baseline_counts = baseline_counts or (lambda scene: [4, 8, 0, 4])
    candidate_counts = candidate_counts or (lambda scene: [8, 0, 4, 4])
    weight = weight or (lambda key, row, scene: 2.0 if key == "zero" else 4.0)
    proposal, direct, scores = [], [], {"zero": {}, "one": {}}
    for family in FAMILIES:
        for scene in range(8):
            for interface in INTERFACES:
                prompt = f"{family}-{scene}-{interface}"
                metadata = {
                    "prompt_id": prompt,
                    "base_scene_id": f"{family}-{scene}",
                    "family": family,
                    "interface": interface,
                }
                for sample, category in enumerate(_categories(proposal_counts(scene))):
                    row = {
                        **metadata,
                        "sample_key": f"proposal-{prompt}-{sample}",
                        "category": category,
                        "token_ids": [sample + 1] * (sample % 3 + 1),
                        "old_logprobs": [-10.0] * (sample % 3 + 1),
                    }
                    proposal.append(row)
                    for key in scores:
                        values = list(row["old_logprobs"])
                        values[0] += float(np.log(weight(key, row, scene)))
                        scores[key][row["sample_key"]] = values
                for key, counts in (("zero", baseline_counts), ("one", candidate_counts)):
                    direct.extend(
                        {
                            **metadata,
                            "sample_key": f"direct-{key}-{prompt}-{sample}",
                            "candidate_id": key,
                            "category": category,
                            "checkpoint_step": 64,
                            "bank_index": 0,
                        }
                        for sample, category in enumerate(_categories(counts(scene)))
                    )
    return proposal, scores, direct


def analyze(values, **kwargs):
    api = importlib.import_module("src.r3_warm_response")
    return api.analyze_direct_validation(
        *values,
        bank_index=kwargs.pop("bank_index", 0),
        baseline_key=kwargs.pop("baseline_key", "zero"),
        candidate_key=kwargs.pop("candidate_key", "one"),
        **kwargs,
    )


def test_absolute_auxiliary_and_residual_use_ordinary_is_and_correct_sign():
    values = case()
    original = copy.deepcopy(values)
    result = analyze(values)
    assert values == original
    assert result["analysis_completed"] is True
    assert result["execution_kind"] == "CPU_MATH"
    assert result["direct_sequences"] == 1536
    assert result["safety_status"] == "NOT_CERTIFIED"
    assert result["can_use_for_safety_decision"] is False
    baseline = result["comparisons"]["absolute_baseline"]["responses"]["overall"]
    target = result["comparisons"]["absolute_candidate"]["responses"]["overall"]
    auxiliary = result["comparisons"]["auxiliary"]["responses"]["overall"]
    assert baseline["pX"]["predicted_delta"] == pytest.approx(0.25)
    assert baseline["pX"]["observed_delta"] == 0
    assert target["pX"]["predicted_delta"] == pytest.approx(0.75)
    assert target["pX"]["observed_delta"] == pytest.approx(0.25)
    assert auxiliary["pX"]["predicted_delta"] == pytest.approx(0.5)
    assert auxiliary["pX"]["observed_delta"] == pytest.approx(0.25)
    assert auxiliary["pX"]["residual"] == pytest.approx(-0.25)
    assert auxiliary["v"]["predicted_delta"] == pytest.approx(1.5)
    assert auxiliary["v"]["observed_delta"] == 0
    assert auxiliary["qX_pool"]["observed_delta"] == pytest.approx(1 / 3)
    assert auxiliary["qS_pool"]["observed_delta"] == pytest.approx(-2 / 3)
    assert len(result["comparisons"]["auxiliary"]["responses"]) == 7
    assert result["control_IS"]["primary_self_normalized"] is False
    assert result["control_IS"]["primary_weights_clipped"] is False
    assert "OVERLAP_WARNING" in result["control_IS"]["candidates"]["zero"]["warnings"]
    json.dumps(result, allow_nan=False)


def test_common_scene_draws_cancel_prediction_residual_without_output_pairing():
    def target_counts(scene):
        x = scene % 4 + 1
        return [x, 6 - x, 6, 4]

    def weights(key, row, scene):
        return 1.0 if key == "zero" else target_counts(scene)["XSWI".index(row["category"])] / 4

    result = analyze(
        case(
            baseline_counts=lambda scene: [4, 4, 4, 4],
            candidate_counts=target_counts,
            weight=weights,
        )
    )
    metric = result["comparisons"]["auxiliary"]["responses"]["overall"]["pX"]
    assert metric["ci"]["predicted_delta"]["half_width"] > 0
    assert metric["ci"]["observed_delta"]["half_width"] > 0
    assert metric["residual"] == pytest.approx(0, abs=1e-14)
    assert metric["ci"]["residual"]["half_width"] < 1e-14
    contract = result["bootstrap_contract"]
    assert contract["replicates"] == 5000
    assert contract["seed"] == 20260909
    assert contract["proposal_and_direct_draws_shared"] is True
    assert contract["interfaces_paired_within_scene"] is True
    assert contract["individual_outputs_paired"] is False
    assert (
        contract["draw_indices_sha256"]
        == result["control_IS"]["bootstrap_contract"]["draw_indices_sha256"]
    )


def test_ratio_is_recomputed_in_each_joint_bootstrap_replicate():
    def baseline(scene):
        return [1, scene % 4 + 1, 0, 14 - scene % 4]

    def target(scene):
        return [scene % 3 + 1, 4, 1, 10 - scene % 3]

    result = analyze(case(baseline_counts=baseline, candidate_counts=target))
    rng = np.random.default_rng(20260909)
    base_boot, target_boot = [], []
    base = np.asarray([[baseline(s)[0], sum(baseline(s)[:3])] for s in range(8)]) / 16
    candidate = np.asarray([[target(s)[0], sum(target(s)[:3])] for s in range(8)]) / 16
    for _family in FAMILIES:
        draws = rng.integers(0, 8, size=(5000, 8))
        base_boot.append(base[draws].mean(axis=1))
        target_boot.append(candidate[draws].mean(axis=1))
    left, right = np.mean(target_boot, axis=0), np.mean(base_boot, axis=0)
    samples = left[:, 0] / left[:, 1] - right[:, 0] / right[:, 1]
    expected = np.quantile(samples, [0.025, 0.975])
    metric = result["comparisons"]["auxiliary"]["responses"]["overall"]["qX_pool"]
    assert [metric["ci"]["observed_delta"][k] for k in ("low", "high")] == pytest.approx(expected)
    wrong = np.mean(candidate[:, 0] / candidate[:, 1] - base[:, 0] / base[:, 1])
    assert metric["observed_delta"] != pytest.approx(wrong)


def test_zero_effect_and_unobserved_support_are_inconclusive_not_pass():
    values = case(
        proposal_counts=lambda scene: [0, 0, 0, 16],
        baseline_counts=lambda scene: [0, 0, 0, 16],
        candidate_counts=lambda scene: [0, 0, 0, 16],
        weight=lambda key, row, scene: 1.0,
    )
    result = analyze(values)
    assert result["status"] == "INCONCLUSIVE"
    metric = result["comparisons"]["auxiliary"]["responses"]["overall"]["pX"]
    for ci in metric["ci"].values():
        assert ci["half_width_to_abs_point_estimate"] is None
        assert ci["half_width_to_abs_predicted_delta"] is None
        assert "DEGENERATE_BOOTSTRAP" in ci["warnings"]
    ratio = result["comparisons"]["auxiliary"]["responses"]["overall"]["qX_pool"]
    assert ratio["observed_delta"] is ratio["predicted_delta"] is ratio["residual"] is None
    assert ratio["ci"]["observed_delta"]["undefined_replicates"] == 5000
    assert "NO_OBSERVED_X_SUPPORT" in result["warnings"]
    assert "SMALL_SCENE_PANEL" in result["warnings"]
    json.dumps(result, allow_nan=False)


def test_direct_record_order_and_output_identifiers_cannot_create_pairing():
    values = case()
    first = analyze(values, bootstrap_replicates=111)
    direct = values[2]
    for i, row in enumerate(direct):
        row["sample_key"] = f"independent-id-{len(direct) - i}"
    direct.reverse()
    second = analyze(values, bootstrap_replicates=111)
    assert first["comparisons"] == second["comparisons"]
    assert first["direct_records_sha256"] != second["direct_records_sha256"]


def test_predicted_points_and_intervals_exactly_match_existing_control_analysis():
    result = analyze(case(), bootstrap_replicates=111)
    for comparison, cid, interval in (
        ("absolute_baseline", "zero", "absolute_delta"),
        ("absolute_candidate", "one", "absolute_delta"),
        ("auxiliary", "one", "auxiliary_delta"),
    ):
        for scope, metrics in result["comparisons"][comparison]["responses"].items():
            for metric, value in metrics.items():
                expected = result["control_IS"]["candidates"][cid]["responses"][scope][metric]
                assert value["predicted_delta"] == expected[interval]
                for field in ("low", "high", "half_width", "undefined_replicates"):
                    assert value["ci"]["predicted_delta"][field] == expected["ci"][interval][field]


def test_undefined_direct_ratio_replicates_are_not_dropped():
    result = analyze(
        case(baseline_counts=lambda scene: [1, 0, 0, 15] if scene == 0 else [0, 0, 0, 16])
    )
    metric = result["comparisons"]["auxiliary"]["responses"]["overall"]["qX_pool"]
    ci = metric["ci"]["observed_delta"]
    assert metric["observed_delta"] is not None
    assert 0 < ci["undefined_replicates"] < 5000
    assert ci["low"] is ci["high"] is ci["half_width"] is None
    assert "UNDEFINED_VALID_DENOMINATOR" in ci["warnings"]


def test_bank_six_preserves_other_is_candidates_and_does_not_pair_shared_output_ids():
    values = case()
    values[1]["unmeasured"] = copy.deepcopy(values[1]["one"])
    for row in values[2]:
        row["bank_index"] = 6
        row["sample_key"] = (
            row["sample_key"].replace("direct-zero-", "direct-").replace("direct-one-", "direct-")
        )
    result = analyze(values, bank_index=6, bootstrap_replicates=111)
    assert result["bank_index"] == 6
    assert result["direct_not_measured_candidate_keys"] == ["unmeasured"]
    assert set(result["direct_estimates"]) == {"zero", "one"}
    assert result["bootstrap_contract"]["individual_outputs_paired"] is False


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "prompt", "scene", "category", "candidate", "cold", "bank"]
)
def test_rejects_bad_direct_panel(mutation):
    values = case()
    row = values[2][0]
    if mutation == "missing":
        values[2].pop()
    elif mutation == "duplicate":
        values[2][-1] = copy.deepcopy(values[2][-2])
    elif mutation == "prompt":
        row["prompt_id"] = "foreign-prompt"
    elif mutation == "scene":
        row["base_scene_id"] = "foreign-scene"
    elif mutation == "category":
        row["category"] = "A"
    elif mutation == "candidate":
        row["candidate_id"] = "unplanned"
    elif mutation == "cold":
        row["checkpoint_step"] = 0
    else:
        row["bank_index"] = 6
    with pytest.raises(ValueError):
        analyze(values)


@pytest.mark.parametrize("mutation", ["keys", "same_candidate", "nan", "overflow", "bad_bank"])
def test_rejects_invalid_prediction_inputs(mutation):
    values = case()
    kwargs = {}
    key = values[0][0]["sample_key"]
    if mutation == "keys":
        del values[1]["one"][key]
    elif mutation == "same_candidate":
        kwargs["candidate_key"] = "zero"
    elif mutation == "bad_bank":
        kwargs["bank_index"] = 1
    else:
        values[1]["one"][key][0] = float("nan") if mutation == "nan" else 1000.0
    with pytest.raises((ValueError, FloatingPointError)):
        analyze(values, **kwargs)
