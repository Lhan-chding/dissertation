"""Frozen planning and gate boundaries; no real data or model execution."""

import copy
import hashlib

import pytest

from sr_f6.data import public_prompt
from sr_f6.interface_repair import screen_inputs as v2_screen_inputs
from sr_f6.interface_repair import screen_plan as v2_screen_plan
from sr_f6.interface_v3_plan import (
    FACTORS,
    MASTER_SEED,
    METRICS,
    PARENTS,
    PHASE,
    screen_inputs,
    screen_plan,
    stage1_gate,
    stage2_gate,
    stage_decision,
)
from sr_f6.interface_v3_protocol import VERSION
from srf6_ref.core import FAMILIES, seed_for
from srf6_ref.design import build_bank


@pytest.fixture(scope="module")
def original():
    tasks = [
        task
        for root in build_bank("DEV_CAL", 8, 2026102711)
        for rows in root["tasks"].values()
        for task in rows
    ]
    public = []
    for task in tasks:
        prompt = public_prompt(task)
        public.append(
            {
                **{
                    key: task[key]
                    for key in ("qid", "root_id", "family", "factor", "variant", "split")
                },
                "protocol": "structured_v4",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "model_input": {
                    "prompt": prompt,
                    "image_file": "DATA/images/example.png",
                    "image_sha256": "a" * 64,
                },
            }
        )
    return public, tasks


def counts(n=173, **changes):
    return {**dict.fromkeys(METRICS, n), **changes}


def parents(b0=173, b1=173, b2=173):
    return {parent: counts(n) for parent, n in zip(PARENTS, (b0, b1, b2), strict=True)}


def test_both_fixed_stages_preserve_original_identity_and_images(original):
    public, private = original
    before = copy.deepcopy(original)
    original_by_qid = {row["qid"]: row for row in public}
    stages = []
    for stage, indices in ((1, (0, 1)), (2, (2, 3))):
        selected = screen_inputs(public, private, stage)
        expected = {
            f"DEV_CAL_{family}_{index:04d}_{factor}_v{variant}"
            for family in FAMILIES
            for index in indices
            for factor in FACTORS
            for variant in range(4)
        }
        assert len(selected) == 96 and {row["qid"] for row in selected} == expected
        for row in selected:
            old = original_by_qid[row["qid"]]
            assert row["screen_stage"] == stage and row["prompt_version"] == VERSION
            assert row["source_prompt_sha256"] == old["prompt_sha256"]
            assert row["model_input"]["prompt"] != old["model_input"]["prompt"]
            for key in ("image_file", "image_sha256"):
                assert row["model_input"][key] == old["model_input"][key]
        stages.append(selected)
    assert not ({row["qid"] for row in stages[0]} & {row["qid"] for row in stages[1]})
    assert original == before
    stages[0][0]["model_input"]["image_file"] = "changed"
    assert original == before


@pytest.mark.parametrize("stage", [1, 2])
def test_complete_source_is_checked_even_outside_selected_stage(original, stage):
    public, private = copy.deepcopy(original)
    private[-1]["factor"] = "wrong"
    with pytest.raises(ValueError):
        screen_inputs(public, private, stage)


@pytest.mark.parametrize("stage", [True, False, 0, 3, 1.0, "1", None])
def test_only_integer_two_stage_ids_exist(original, stage):
    with pytest.raises(ValueError):
        screen_inputs(*original, stage)
    with pytest.raises(ValueError):
        screen_plan([], stage)
    with pytest.raises(ValueError):
        stage_decision(parents(), stage=stage)


def test_seed_domain_budget_factor_pairing_and_frozen_confirmation(original):
    public, private = original
    old = v2_screen_plan(v2_screen_inputs(public, private))
    old_seeds = {a["seed"] for a in old["attempts"]}
    all_new_seeds = []
    for stage in (1, 2):
        selected = screen_inputs(public, private, stage)
        before = copy.deepcopy(selected)
        plan = screen_plan(selected, stage)
        assert selected == before
        assert plan == screen_plan(list(reversed(selected)), stage)
        assert plan["phase"] == PHASE and plan["stage"] == stage
        assert plan["expected_new_answers"] == len(plan["attempts"]) == 576
        assert plan["max_global_answers"] == 1152
        assert plan["both_stage_plans_required_before_first_output"]
        assert not plan["generation_allowed"] and not plan["training_allowed"]
        assert not plan["scorer_changed"] and not plan["parser_changed"]
        assert plan["confirmation"]["required_counts"] == dict.fromkeys(METRICS, 1383)
        assert plan["confirmation"]["registered_observability_checks"] == 18
        paired = {}
        for row in plan["attempts"]:
            assert row["stage"] == stage
            key = row["parent"], row["root_id"], row["variant"], row["draw"]
            assert paired.setdefault(key, row["seed"]) == row["seed"]
            assert row["seed"] == seed_for(MASTER_SEED, VERSION, "screen", stage, *key)
        assert len(paired) == 288
        all_new_seeds.append(set(paired.values()))
        assert not (all_new_seeds[-1] & old_seeds)
        plan["confirmation"]["required_counts"]["V"] = 0
        assert screen_plan(selected, stage)["confirmation"]["required_counts"]["V"] == 1383
    assert not (all_new_seeds[0] & all_new_seeds[1])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "stage",
        "factor",
        "variant_bool",
        "prompt",
        "extra_input",
        "image",
        "qid_type",
    ],
)
def test_plan_rejects_changed_or_wrong_stage_inputs(original, mutation):
    selected = screen_inputs(*original)
    if mutation == "missing":
        selected.pop()
    elif mutation == "duplicate":
        selected[-1] = selected[0]
    elif mutation == "stage":
        selected[0]["screen_stage"] = 2
    elif mutation == "factor":
        selected[0]["factor"] = "P1" if selected[0]["factor"] == "P0" else "P0"
    elif mutation == "variant_bool":
        selected[0]["variant"] = False
    elif mutation == "prompt":
        selected[0]["model_input"]["prompt"] += " modified"
    elif mutation == "extra_input":
        selected[0]["model_input"]["gold"] = 42
    elif mutation == "qid_type":
        selected[0]["qid"] = []
    else:
        selected[0]["model_input"]["image_file"] = "DATA/images/../../outside"
    with pytest.raises(ValueError):
        screen_plan(selected)


@pytest.mark.parametrize(
    "n,want",
    [(0, "FAIL"), (163, "FAIL"), (164, "GRAY"), (172, "GRAY"), (173, "PASS"), (192, "PASS")],
)
def test_stage1_exact_boundaries(n, want):
    assert stage1_gate(counts(n)) == want


def test_all_metrics_pass_and_any_metric_fail_control_stage1():
    assert stage1_gate(counts(192, S_known=172)) == "GRAY"
    assert stage1_gate(counts(192, B_known=163)) == "FAIL"
    assert stage1_gate(counts(164, B_known=163)) == "FAIL"


@pytest.mark.parametrize("bad", [True, False, -1, 193, 173.0, "173", None])
def test_counts_reject_boolean_wrong_type_and_range(bad):
    with pytest.raises(ValueError):
        stage1_gate(counts(V=bad))
    with pytest.raises(ValueError):
        stage2_gate(counts(172), counts(V=bad))


@pytest.mark.parametrize("bad", [{}, {"V": 173}, {**counts(), "extra": 1}, [], None])
def test_count_keys_are_exact(bad):
    with pytest.raises(ValueError):
        stage1_gate(bad)


def test_stage2_cumulative_345_vs_346_and_gray_requirement():
    assert stage2_gate(counts(172), counts(173)) == "FAIL"
    assert stage2_gate(counts(172), counts(174)) == "PASS"
    assert stage2_gate(counts(164), counts(182)) == "PASS"
    assert stage2_gate(counts(172), counts(192, B_known=173)) == "FAIL"
    for first in (counts(163), counts(173)):
        with pytest.raises(ValueError):
            stage2_gate(first, counts(192))


def test_stage1_hard_failure_stops_all_without_using_gray_budget():
    decision = stage_decision(parents(b0=163, b1=172))
    assert decision["status"] == "SCREEN_FAIL_STOP"
    assert decision["parents"]["b1"]["status"] == "GRAY"
    assert decision["stage2_parents"] == []
    assert decision["budget"]["required_stage2_answers"] == 0
    with pytest.raises(ValueError):
        stage_decision({"b1": counts(192)}, stage=2, prior=decision)


def test_exact_gray_subset_and_prior_passes_carry_forward_without_mutation():
    initial = parents(b0=172, b1=173, b2=164)
    before = copy.deepcopy(initial)
    prior = stage_decision(initial)
    assert initial == before
    assert prior["status"] == "STAGE2_REQUIRED"
    assert prior["stage2_parents"] == ["b0", "b2"]
    assert prior["budget"]["required_stage2_answers"] == 384
    prior_before = copy.deepcopy(prior)
    second = {"b0": counts(174), "b2": counts(182)}
    second_before = copy.deepcopy(second)
    decision = stage_decision(second, stage=2, prior=prior)
    assert prior == prior_before and second == second_before
    assert decision["status"] == "SCREEN_PASS"
    assert decision["stage2_parents"] == []
    assert decision["completed_stage2_parents"] == ["b0", "b2"]
    assert decision["parents"]["b1"] == prior["parents"]["b1"]
    assert decision["parents"]["b0"]["combined_n"] == 384
    assert decision["parents"]["b1"]["combined_n"] == 192
    assert not decision["training_allowed"] and not decision["generation_allowed"]
    decision["parents"]["b0"]["stage1_counts"]["V"] = 0
    assert prior == prior_before


@pytest.mark.parametrize("bad_parent_keys", [[], ["b0"], ["b0", "b1", "b2"], ["b0", "other"]])
def test_stage2_rejects_missing_extra_or_unregistered_parents(bad_parent_keys):
    prior = stage_decision(parents(b0=172, b2=172))
    with pytest.raises(ValueError):
        stage_decision({key: counts(192) for key in bad_parent_keys}, stage=2, prior=prior)


def test_stage1_requires_all_parents_and_no_prior():
    for bad in ({}, {"b0": counts()}, {**parents(), "other": counts()}, []):
        with pytest.raises(ValueError):
            stage_decision(bad)
    with pytest.raises(ValueError):
        stage_decision(parents(), prior={})


def test_terminal_stage2_failure_never_allows_stage3_or_another_stage2():
    prior = stage_decision(parents(b0=172))
    result = stage_decision({"b0": counts(173)}, stage=2, prior=prior)
    assert result["status"] == "SCREEN_FAIL_STOP"
    assert result["stage2_parents"] == []
    with pytest.raises(ValueError):
        stage_decision({"b0": counts(192)}, stage=2, prior=result)
    with pytest.raises(ValueError):
        stage_decision({"b0": counts(192)}, stage=3, prior=result)


@pytest.mark.parametrize(
    "field", ["status", "parents", "stage2_parents", "budget", "generation_allowed"]
)
def test_tampered_prior_is_rejected(field):
    prior = stage_decision(parents(b0=172))
    prior[field] = None
    with pytest.raises(ValueError):
        stage_decision({"b0": counts(192)}, stage=2, prior=prior)


def test_stage1_pass_and_missing_prior_do_not_allow_stage2():
    for prior in (None, {}, stage_decision(parents())):
        with pytest.raises(ValueError):
            stage_decision({}, stage=2, prior=prior)


@pytest.mark.parametrize("mutation", ["boolean_stage", "float_budget", "integer_flag"])
def test_prior_rejects_python_equal_but_json_distinct_values(mutation):
    prior = stage_decision(parents(b0=172))
    if mutation == "boolean_stage":
        prior["stage"] = True
    elif mutation == "float_budget":
        prior["budget"]["max_total_answers"] = 1152.0
    else:
        prior["generation_allowed"] = 0
    with pytest.raises(ValueError):
        stage_decision({"b0": counts(192)}, stage=2, prior=prior)
