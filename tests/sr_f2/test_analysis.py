"""Paired inference, denominators, and the clarified C4 decision boundary."""

import copy
import json

import numpy as np
import pytest
from scipy.stats import t as student_t

from sr_f2.analysis import (
    FAMILIES,
    SEEDS,
    _noninferiority,
    _t_sensitivity,
    _untriggered,
    _usage,
    contrast,
    report_main_contrasts,
)


def event(value, **updates):
    return dict(
        A=value, J=value, E=value, E6=value, P=value, P6=value, J6=value, p_read=value, **updates
    )


def rows_for(arm, function, *, roots=2, questions=1):
    prefix = "SRF2" if arm in ("EARLY", "J6", "DEC_PRIME", "ADAPT") else "SRF1_2"
    rows = []
    for seed_index, seed in enumerate(SEEDS):
        for family in FAMILIES:
            count = roots[family] if isinstance(roots, dict) else roots
            for root in range(count):
                question_count = questions(root) if callable(questions) else questions
                for question in range(question_count):
                    for draw in range(4):
                        value = function(seed_index, family, root, question, draw)
                        rows.append(
                            dict(
                                model_id=f"{prefix}_{arm}_s{seed}",
                                family=family,
                                root_id=f"{family}-r{root}",
                                qid=f"{family}-r{root}-q{question}",
                                draw=draw,
                                score=event(value) if not isinstance(value, dict) else value,
                            )
                        )
    return rows


def test_equal_seed_family_root_question_weighting_not_pooled_draw_weighting():
    roots = dict(CROSS=2, THRESHOLD=3, TOPK=4, INTERVAL=5)

    def count(root):
        return root + 1

    new = rows_for("EARLY", lambda s, f, r, q, d: int(r == 0), roots=roots, questions=count)
    old = rows_for("J", lambda *args: 0, roots=roots, questions=count)
    result = contrast(new, old, "EARLY")
    expected = np.mean([1 / roots[family] for family in FAMILIES])
    assert result["new_rate"] == pytest.approx(expected)
    assert result["seed_effects_pp"] == pytest.approx([100 * expected] * 3)
    for seed in result["per_seed"]:
        assert seed["denominators"]["roots"] == 14
        assert seed["denominators"]["by_family"]["CROSS"]["questions"] == 3
        assert seed["denominators"]["by_family"]["CROSS"]["samples"] == 12


def test_bootstrap_keeps_root_pairing_shared_across_all_three_seeds():
    new = rows_for("EARLY", lambda s, f, r, q, d: int(r == s) if s < 2 else int(d < 2))
    old = rows_for("J", lambda s, f, r, q, d: int(d < 2))
    result = contrast(new, old, "EARLY")
    assert result["effect_pp"] == 0
    assert result["CI98_75_pp"] == [0, 0]
    assert result["contrast_confirmed"] is False
    assert result["bootstrap_seed"] == 812901
    assert result["bootstrap_replicates"] == 5000


def test_bootstrap_matches_existing_frozen_root_stratification():
    from sr_f12.analysis import paired_root_ci

    new = rows_for("EARLY", lambda s, f, r, q, d: int(d <= (s + r) % 4), roots=5)
    old = rows_for("J", lambda s, f, r, q, d: int(d < 2), roots=5)
    result = contrast(new, old, "EARLY")
    x = np.array([[(s + r) % 4 / 4 + 0.25 for _ in FAMILIES for r in range(5)] for s in range(3)])
    expected = paired_root_ci(x, np.full_like(x, 0.5), [f for f in FAMILIES for _ in range(5)])
    assert result["effect_pp"] == pytest.approx(expected["effect_pp"])
    assert result["CI98_75_pp"] == pytest.approx(expected["CI98_75_pp"])


def test_root_interval_does_not_replace_three_seed_same_sign_rule():
    new = rows_for("EARLY", lambda s, f, r, q, d: int(s < 2))
    old = rows_for("J", lambda s, f, r, q, d: int(d < 2))
    result = contrast(new, old, "EARLY")
    assert result["seed_effects_pp"] == [50, 50, -50]
    assert result["root_interval_excludes_zero"] is True
    assert result["all_three_same_sign"] is False
    assert result["contrast_confirmed"] is False
    assert result["status"] == "ROOT_LEVEL_ONLY"
    assert result["paired_t_sensitivity"]["CI98_75_pp"][0] < 0


@pytest.mark.parametrize("positive", [True, False])
def test_sign_confirmation_is_directional_and_zero_is_not_a_sign(positive):
    new = rows_for("EARLY", lambda *args: int(positive))
    old = rows_for("J", lambda *args: int(not positive))
    result = contrast(new, old, "EARLY")
    assert result["contrast_confirmed"] is True
    assert result["superiority_confirmed"] is positive
    assert result["status"] == ("CONFIRMED_POSITIVE" if positive else "CONFIRMED_NEGATIVE")
    json.dumps(result, allow_nan=False)


def test_seed_t_sensitivity_uses_sample_sd_df2_and_both_intervals():
    result = _t_sensitivity([1, 2, 3])
    assert result["paired_sd_pp"] == 1
    assert result["degrees_of_freedom"] == 2
    half_width = student_t.ppf(0.99375, 2) / np.sqrt(3)
    assert result["CI98_75_pp"] == pytest.approx([2 - half_width, 2 + half_width])
    assert _t_sensitivity([0, 0, 0])["two_sided_p_value"] == 1


def test_irrelevant_panels_steps_and_models_do_not_enter_contrast():
    new = rows_for("J6", lambda s, f, r, q, d: int(f == "TOPK"))
    old = rows_for("J", lambda *args: 0)
    extra = []
    for metadata in (
        {"pool": "COMPOSITION"},
        {"protocol": "answer_only"},
        {"view": "V1"},
        {"step": 32},
        {"model_id": "SRF2_PILOT_J6"},
    ):
        extra.append(dict(new[0], score=event(1), **metadata))
    result = contrast(new + extra, old, "J6", metric="A", family="TOPK")
    assert result["effect_pp"] == 100
    assert result["per_seed"][0]["denominators"]["roots"] == 2


@pytest.mark.parametrize(
    "problem",
    ["missing", "duplicate", "missing_seed", "missing_metric", "nan", "fractional_binary", "draw"],
)
def test_incomplete_or_invalid_scientific_support_is_never_zero_filled(problem):
    new = rows_for("EARLY", lambda *args: 1)
    old = rows_for("J", lambda *args: 0)
    if problem == "missing":
        new.pop()
    elif problem == "duplicate":
        new.append(copy.deepcopy(new[0]))
    elif problem == "missing_seed":
        new = [row for row in new if not row["model_id"].endswith("71003")]
    elif problem == "missing_metric":
        del new[0]["score"]["J"]
    elif problem == "nan":
        new[0]["score"]["J"] = float("nan")
    elif problem == "fractional_binary":
        new[0]["score"]["J"] = 0.5
    else:
        new[0]["draw"] = 4
    with pytest.raises(ValueError):
        contrast(new, old, "EARLY")


@pytest.mark.parametrize(
    "lower,seeds,confirmed",
    [
        (-1.9, [-1, 0, 1], True),
        (-2, [-1, 0, 1], False),
        (-2.01, [-1, 0, 1], False),
        (-1.9, [-2, 1, 1], False),
        (-1.9, [-2.1, 1, 1], False),
    ],
)
def test_c4_noninferiority_is_strict_at_minus2_and_allows_zero_crossing(lower, seeds, confirmed):
    result = _noninferiority(dict(CI98_75_pp=[lower, 1.5], seed_effects_pp=seeds))
    assert result["confirmed"] is confirmed
    assert result["zero_crossing_is_allowed"] is True


def trigger_states():
    active = [["TOPK"], ["CROSS", "THRESHOLD"], []]
    return {
        f"SRF2_ADAPT_s{seed}": dict(active_families=active[index])
        for index, seed in enumerate(SEEDS)
    }


def usage(used=900):
    return {f"SRF2_ADAPT_s{seed}": dict(semantic_groups=used, total_groups=1536) for seed in SEEDS}


def test_untriggered_selection_is_per_seed_and_each_seed_keeps_equal_weight():
    new = rows_for("ADAPT", lambda s, f, r, q, d: int(f in ({"CROSS"}, {"TOPK"}, {"INTERVAL"})[s]))
    old = rows_for("J", lambda *args: 0)
    result = _untriggered(new, old, trigger_states())
    assert result["seed_effects_pp"] == pytest.approx([100 / 3, 50, 25])
    assert result["effect_pp"] == pytest.approx((100 / 3 + 50 + 25) / 3)
    assert [row["selected_denominators"]["roots"] for row in result["per_seed"]] == [6, 4, 8]
    assert result["plus_2pp_is_decision_gate"] is False


def test_untriggered_selection_is_undefined_when_any_seed_has_no_family():
    states = trigger_states()
    states["SRF2_ADAPT_s71001"]["active_families"] = list(FAMILIES)
    result = _untriggered([], [], states)
    assert result["status"] == "UNDEFINED_NO_UNTRIGGERED_FAMILY"
    assert result["superiority_confirmed"] is False


def test_semantic_usage_retains_counts_and_rejects_partial_denominator():
    result = _usage(usage())
    assert result["semantic_groups"] == 2700
    assert result["total_groups"] == 4608
    assert result["criterion_met"] is True
    assert _usage(usage(1000))["criterion_met"] is False
    bad = usage()
    bad["SRF2_ADAPT_s71002"]["total_groups"] = 1520
    with pytest.raises(ValueError, match="96 x 16"):
        _usage(bad)


@pytest.fixture(scope="module")
def full_report():
    def scores(arm):
        def value(seed, family, root, question, draw):
            base = int(draw < 2)
            position = question * 4 + draw
            if arm == "J":
                a = j = base
            elif arm == "GATE":
                a = j = int(base or position in (2, 3, 6, 7))
            elif arm == "EARLY":
                a = j = int(base or position in (2, 3))
            elif arm == "J6":
                a, j = int(base or position in (2, 3, 6, 7)), 0
            elif arm == "DEC_PRIME":
                a = j = int(base or position == 2)
            else:
                a = int(base or position == 2)
                j = int(base or position == 2) if root % 2 else int(base and position != 0)
            return dict(
                A=a,
                J=j,
                E=int(arm != "J6"),
                E6=1,
                P=j,
                P6=1 if arm == "J6" else j,
                J6=a if arm == "J6" else j,
                p_read=0.75 if arm == "J6" else 1 if j else 0.5,
            )

        return value

    new = [
        row
        for arm in ("EARLY", "J6", "DEC_PRIME", "ADAPT")
        for row in rows_for(arm, scores(arm), roots=32, questions=16)
    ]
    old = [
        row for arm in ("J", "GATE") for row in rows_for(arm, scores(arm), roots=32, questions=16)
    ]
    return report_main_contrasts(
        new, old, trigger_states=trigger_states(), semantic_group_usage=usage()
    )


def test_complete_report_separates_prediction_from_confirmation(full_report):
    assert full_report["status"] == "MAIN_CONTRASTS_COMPLETE"
    assert full_report["conclusions"] == "NUMERICAL_STATUSES_ONLY"
    c1, c2, c3, c4 = [full_report["contrasts"][key] for key in ("C1", "C2", "C3", "C4")]
    assert c1["effect_pp"] == 3.125
    assert c1["contrast_confirmed"] is True
    assert c1["prediction_met"] is False  # 3.125 < .6 * 6.25.
    assert c2["effect_pp"] == 6.25
    assert c2["prediction_met"] is True
    assert c2["certificate_rates"]["E6"]["rate"] == 1
    assert c2["certificate_rates"]["E"]["rate"] == 0
    assert c3["effect_pp"] == 1.5625
    assert c3["prediction_met"] is True
    assert c4["CI98_75_pp"][0] < 0 < c4["CI98_75_pp"][1]
    assert c4["contrast_confirmed"] is False
    assert c4["noninferiority"]["confirmed"] is True
    assert c4["primary_status"] == "NONINFERIORITY_CONFIRMED"
    assert c4["untriggered_A"]["effect_pp"] == 1.5625
    assert c4["untriggered_A"]["plus_2pp_prediction_met"] is False
    assert c4["untriggered_A"]["superiority_confirmed"] is True
    assert c4["criteria_met"] is True
    assert c4["criteria_status"] == "MET"
    assert c1["per_seed"][0]["denominators"]["samples"] == 8192
    assert c2["per_seed"][0]["denominators"]["samples"] == 2048
    json.dumps(full_report, allow_nan=False)


def test_main_report_rejects_partial_root_panels():
    new = [
        row
        for arm in ("EARLY", "J6", "DEC_PRIME", "ADAPT")
        for row in rows_for(arm, lambda *args: 1)
    ]
    old = [row for arm in ("J", "GATE") for row in rows_for(arm, lambda *args: 0)]
    with pytest.raises(ValueError, match="complete TEST_ID"):
        report_main_contrasts(
            new, old, trigger_states=trigger_states(), semantic_group_usage=usage()
        )
