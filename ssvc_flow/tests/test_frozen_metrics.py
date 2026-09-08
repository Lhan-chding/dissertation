"""Frozen sampling estimands retain fixed prompts and separate greedy outputs."""

import copy
import json

import pytest

from src.frozen_metrics import build_frozen_metrics


def observations(categories=None, prompt="p0", family="trend", interface="SYMBOLIC_FRESH"):
    categories = categories or ["X", "S", "W", "I"] * 4
    result = []
    for index, category in enumerate([*categories, "X"]):
        greedy = index == len(categories)
        result.append(
            {
                "prompt_id": prompt,
                "base_scene_id": "scene-" + prompt,
                "constraint_family": family,
                "interface": interface,
                "category": category,
                "decode_mode": "greedy" if greedy else "sample",
                "rollout_index": 0 if greedy else index,
                "sample_key": f"{prompt}:{'greedy' if greedy else index}",
                "completion_length": index + 1,
                "stop_reason": "eos",
                "syntax_valid": category != "I",
                "copy_observation": category == "W",
                "constraint_satisfaction": None if category == "I" else category == "X",
            }
        )
    return result


def test_sampling_is_distinct_from_greedy_and_json_safe():
    rows = observations()
    original = copy.deepcopy(rows)
    result = build_frozen_metrics(rows)
    assert result["overall"]["rollout_count"] == 16
    assert result["overall"]["pooled"]["pX"] == 0.25
    assert result["overall"]["pooled"]["qX"] == pytest.approx(1 / 3)
    assert result["overall"]["pooled"]["truth_given_answer"] == 0.5
    assert result["greedy"]["overall"]["pooled"]["pX"] == 1
    assert result["greedy"]["overall"]["rollout_count"] == 1
    assert len(result["by_group"]) == 6
    assert result["by_group"]["trend/IMAGE_CUE_FRESH"]["status"] == "NA"
    assert rows == original
    json.dumps(result, allow_nan=False)


def test_macro_and_pooled_ratios_differ_and_na_prompts_are_retained():
    rows = observations(["X"] + ["I"] * 15)
    rows += observations(["S"] * 16, "p1")
    rows += observations(["I"] * 16, "p2")
    result = build_frozen_metrics(rows)["overall"]
    assert result["pooled"]["qX"] == pytest.approx(1 / 17)
    assert result["macro_mean_per_prompt_qX"] == 0.5
    assert result["macro_qX_NA_fraction"] == pytest.approx(1 / 3)


def test_undefined_denominators_are_null():
    result = build_frozen_metrics(observations(["I"] * 16))["overall"]
    assert result["pooled"]["qX"] is None
    assert result["pooled"]["qS"] is None
    assert result["pooled"]["truth_given_answer"] is None
    assert result["constraint_satisfaction"]["rate_among_known_valid"] is None
    assert result["constraint_satisfaction"]["unknown_count"] == 16


def test_constraints_copy_and_truncation_have_explicit_denominators():
    rows = observations()
    rows[0]["stop_reason"] = "length"
    rows[3]["stop_reason"] = "length"
    result = build_frozen_metrics(rows)["overall"]
    assert result["copy_observation_rate"] == 0.25
    assert result["constraint_satisfaction"] == {
        "satisfied_count": 4,
        "known_valid_count": 12,
        "unknown_count": 4,
        "invalid_count": 4,
        "rate_among_known_valid": pytest.approx(1 / 3),
        "satisfied_fraction_all_outputs": 0.25,
    }
    assert result["generation"]["actual_token_count"] == 136
    assert result["generation"]["length_stop_complete_valid_count"] == 1
    assert result["generation"]["length_stop_partial_or_invalid_count"] == 1
    assert result["generation"]["completion_lengths"] == list(range(1, 17))


def test_binomial_intervals_do_not_claim_zero_or_one_probability():
    for category in ["X", "I"]:
        result = build_frozen_metrics(observations([category] * 16))["by_prompt"]["p0"]
        low, high = result["pX_interval"]["low"], result["pX_interval"]["high"]
        if category == "I":
            assert low == 0
            assert high == pytest.approx(1 - 0.025 ** (1 / 16))
            assert result["group_probability_estimates"]["8"]["no_X"]["low"] < 1
        else:
            assert low == pytest.approx(0.025 ** (1 / 16))
            assert high == 1
            assert result["group_probability_estimates"]["8"]["no_X"]["high"] > 0


def test_informative_interval_checks_interior_maximum():
    result = build_frozen_metrics(observations(["X", "S"] * 8))["by_prompt"]["p0"]
    for k in [2, 4, 8, 16, 32]:
        bounds = result["group_probability_estimates"][str(k)]["informative_X_reward"]
        assert bounds["high"] == pytest.approx(1 - 2 * 0.5**k)
        assert bounds["low"] < bounds["high"]
        assert bounds["estimate"] == bounds["high"]


def test_actual_groups_are_nonoverlapping_index_order_not_combinations():
    rows = observations(["X"] * 8 + ["S"] * 7 + ["I"])
    rows = list(reversed(rows))
    result = build_frozen_metrics(rows)["by_prompt"]["p0"]["actual_groups"]
    assert result["8"]["group_count"] == 2
    assert result["8"]["zero_reward_variance_X"]["count"] == 2
    assert result["8"]["contains_X"]["fraction"] == 0.5
    assert result["8"]["contains_S"]["fraction"] == 0.5
    assert result["8"]["contains_I"]["fraction"] == 0.5
    assert result["16"]["zero_reward_variance_X"]["count"] == 0


def test_incomplete_groups_are_na_not_padded_or_overlapping():
    result = build_frozen_metrics(observations(["S"] * 4), expected_rollouts=4)
    assert result["overall"]["actual_groups"]["8"]["status"] == "NA"
    assert result["overall"]["actual_groups"]["8"]["unused_rollout_count"] == 4


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(0),
        lambda rows: rows.append(dict(rows[0])),
        lambda rows: rows.pop(),
        lambda rows: rows.append(dict(rows[-1], sample_key="extra-greedy")),
        lambda rows: rows[0].update(rollout_index=16),
        lambda rows: rows[0].update(base_scene_id="other-scene"),
        lambda rows: rows[0].update(sample_key=rows[1]["sample_key"]),
        lambda rows: rows[0].pop("sample_key"),
        lambda rows: rows[0].update(category="Z"),
        lambda rows: rows[0].update(syntax_valid=False),
        lambda rows: rows[0].update(stop_reason="unknown"),
        lambda rows: rows[0].update(completion_length=-1),
        lambda rows: rows[3].update(constraint_satisfaction=True),
    ],
)
def test_malformed_or_incomplete_input_is_rejected(mutation):
    rows = observations()
    mutation(rows)
    with pytest.raises(ValueError):
        build_frozen_metrics(rows)


def test_sample_keys_are_optional_but_rollout_identity_remains_required():
    rows = observations()
    for row in rows:
        row.pop("sample_key")
    assert build_frozen_metrics(rows)["overall"]["rollout_count"] == 16


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="no observations"):
        build_frozen_metrics([])


def test_legacy_conditions_have_observed_groups_and_unknown_relations():
    rows = observations(family="legacy-sum", interface="collision")
    rows += observations(prompt="p1", family="legacy-range", interface="separating")
    for row in rows:
        row["constraint_satisfaction"] = None
        row["base_scene_id"] = "pair0"
    result = build_frozen_metrics(rows, track="L")
    assert set(result["by_group"]) == {"legacy-sum/collision", "legacy-range/separating"}
    assert result["overall"]["independent_scene_count"] == 1
    assert result["overall"]["constraint_satisfaction"]["known_valid_count"] == 0
    assert result["overall"]["constraint_satisfaction"]["rate_among_known_valid"] is None


def test_legacy_complete_out_of_domain_action_is_invalid_not_partial_syntax():
    rows = observations(family="legacy-sum", interface="collision")
    for row in rows:
        row["constraint_satisfaction"] = None
        row["semantic_parse_success"] = row["category"] != "I"
    rows[3]["syntax_valid"] = True
    rows[3]["stop_reason"] = "length"
    result = build_frozen_metrics(rows, track="L")["overall"]
    assert result["pooled"]["v"] == 0.75
    assert result["constraint_satisfaction"]["invalid_count"] == 4
    assert result["generation"]["length_stop_complete_valid_count"] == 0
    assert result["generation"]["length_stop_partial_or_invalid_count"] == 1
    assert result["generation"]["length_stop_complete_syntax_invalid_action_count"] == 1
    assert result["generation"]["length_stop_incomplete_or_invalid_syntax_count"] == 0


def test_all_unknown_constraint_satisfaction_is_not_zero_satisfaction():
    rows = observations(family="legacy-sum", interface="collision")
    for row in rows:
        row["constraint_satisfaction"] = None
    result = build_frozen_metrics(rows, track="L")["overall"]["constraint_satisfaction"]
    assert result["satisfied_fraction_all_outputs"] is None
    assert result["rate_among_known_valid"] is None
    assert result["known_valid_count"] == 0


def test_optional_semantic_parse_flag_must_agree_with_action_category():
    rows = observations(family="legacy-sum", interface="collision")
    rows[0]["semantic_parse_success"] = False
    with pytest.raises(ValueError, match="semantic"):
        build_frozen_metrics(rows, track="L")
