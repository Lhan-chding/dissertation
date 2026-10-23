"""Adversarial tests of the development grammar, not a human accuracy audit."""

import copy
import json

import pytest

from sr_f4 import task_contract
from sr_f4.verification import verify_semantics as verify_v1
from sr_f4.verification_v2 import VERIFIER_VERSION, verify_semantics

WORLD = dict(categories=["Jan", "Feb", "Mar"], domain=[-20, 100],
             series=dict(Alpha=[10, 13, 19], Beta=[1, 3, 5]))
QUERY = dict(family="INTERVAL", start=0, end=2, include_start=False,
             aggregate="mean", value="diff")


def evaluate(work, *, world=None, query=None, evidence=None, answer=12, **fields):
    world, query = copy.deepcopy(world or WORLD), copy.deepcopy(query or QUERY)
    task = dict(qid="development_q", world_id="development_world", world=world, query=query)
    if evidence is None:
        evidence = [dict(series=s, category=c, value=world["series"][s][i])
                    for s in ("Alpha", "Beta") for i, c in enumerate(world["categories"])]
    record = dict(full_text=json.dumps(dict(evidence=evidence, work=work, answer=answer)),
                  row_id="development_row", generation_status="COMPLETE")
    record.update(fields)
    score = task_contract.score(record["full_text"], world, query)
    return verify_semantics(task, record, score), task, record, score


def selection(work, **kwargs):
    return evaluate(work, **kwargs)[0]["dimensions"]["S"]


@pytest.mark.parametrize("work", [
    "final selection=Feb,Mar",
    "Feb and Mar are selected.",
    "The selected categories are Feb and Mar.",
    "Initially selected: Jan,Feb,Mar; final selection=Feb,Mar",
    "Hypothetical selection=Jan; Feb and Mar are selected.",
    "selected=Jan,Feb,Mar; correction: selected=Feb,Mar",
    "Selected months: [Feb, Mar]",
])
def test_explicit_and_natural_final_sets(work):
    assert selection(work)["status"] == "PASS"


@pytest.mark.parametrize("work", [
    "Initially selected: Feb,Mar",
    "Not selected: Feb,Mar",
    "hypothetical selection=Feb,Mar",
    "The answer is 12.",
    "Jan: 10-1=9; Feb: 13-3=10; Mar: 19-5=14",
    "13-3=10;19-5=14; Count=2",
    "Range = 19 - 13 = 6",
    "Count = 2",
    "The empty set was considered, but no final selection was stated.",
])
def test_unadopted_or_insufficient_observations_stay_unknown(work):
    assert selection(work)["status"] == "UNKNOWN"


def test_wrong_explicit_set_fails_even_with_correct_answer():
    assert selection("Jan and Mar are selected.", answer=12)["status"] == "FAIL"


@pytest.mark.parametrize("work", [
    "selected=Jan,Mar; final selection=Feb,Mar",
    "final selection=Feb,Feb,Mar",
    "selected=Feb,Mar and an unspecified month",
    "final selection=Feb,Mar (perhaps Jan)",
    "final selection=Feb,Mar; I actually used Jan and Mar",
])
def test_conflict_duplicate_or_unsupported_list_is_unknown(work):
    assert selection(work)["status"] == "UNKNOWN"


def test_annotated_natural_rank_set():
    query = dict(family="TOPK", k=2, direction="largest", aggregate="sum", value="Beta")
    label = selection("The two categories with the largest Alpha values are: Mar (19), Feb (13).",
                      query=query, answer=8)
    assert label["status"] == "PASS"
    assert selection("The top 2 Alpha values are Mar (19), Feb (13).",
                     query=query, answer=8)["status"] == "PASS"


def test_wrong_rank_series_is_failure_even_if_same_categories():
    query = dict(family="TOPK", k=2, direction="largest", aggregate="sum", value="Beta")
    assert selection("The two highest Beta values are Mar (5), Feb (3).",
                     query=query, answer=8)["status"] == "FAIL"


def test_only_finally_used_interval_differences_are_selected():
    label = selection("Jan: 10-1=9; Feb: 13-3=10; Mar: 19-5=14; Mean=(10+14)/2=12")
    assert label["status"] == "PASS"
    assert label["observed"]["categories"] == ["Feb", "Mar"]
    assert label["rule_id"] == "S_V2_EVIDENCE_BOUND_FINAL_REDUCTION"


def test_for_category_prefix_preserves_identity():
    assert selection("For Feb: 13-3=10; For Mar:19-5=14; Mean=(10+14)/2=12")["status"] == "PASS"
    assert selection("For Jan: 13-3=10; For Mar:19-5=14; Mean=(10+14)/2=12")["status"] == "UNKNOWN"


def test_final_reduction_accepts_spaces_and_category_intro():
    work = ("Calculate Alpha - Beta for each month: Feb: 13 - 3 = 10; "
            "Mar: 19 - 5 = 14; Mean = (10 + 14) / 2 = 12")
    assert selection(work)["status"] == "PASS"


@pytest.mark.parametrize("intro", [
    "Compute (Alpha - Beta) for each month:",
    "Calculate ( Alpha - Beta ) for each category:",
])
def test_balanced_parenthesized_difference_intro_preserves_first_category(intro):
    work = intro + " Feb:13-3=10; Mar:19-5=14; Mean=(10+14)/2=12"
    assert selection(work)["status"] == "PASS"


@pytest.mark.parametrize("intro", [
    "Compute (Alpha - Beta for each month:",
    "Compute Alpha - Beta) for each month:",
    "Compute (Alpha - Beta)) for each month:",
    "Compute range (Alpha - Beta) for each month:",
])
def test_malformed_or_range_intro_cannot_bind_first_category(intro):
    work = intro + " Feb:13-3=10; Mar:19-5=14; Mean=(10+14)/2=12"
    assert selection(work)["status"] == "UNKNOWN"


@pytest.mark.parametrize("work", [
    "Range = 13-3=10; Mar:19-5=14; Mean=(10+14)/2=12",
    "Example 13-3=10; Mar:19-5=14; Mean=(10+14)/2=12",
    "- Jan:13-3=10; Mar:19-5=14; Mean=(10+14)/2=12",
    "Mean=(10+14)/2=12; Feb:13-3=10; Mar:19-5=14",
])
def test_range_examples_wrong_named_pairs_and_later_equations_cannot_bind_final_sum(work):
    assert selection(work)["status"] == "UNKNOWN"


def test_mean_divisor_and_result_never_become_operands():
    world = copy.deepcopy(WORLD)
    world["series"]["Beta"][0] = 2
    query = dict(family="TOPK", k=2, direction="largest", aggregate="mean", value="Beta")
    label = selection("Beta mean=(3+5)/2=4", world=world, query=query, answer=4)
    assert label["status"] == "PASS"
    assert label["observed"]["categories"] == ["Feb", "Mar"]


def test_topk_sum_is_bound_to_response_evidence_without_using_k_to_choose():
    query = dict(family="TOPK", k=2, direction="largest", aggregate="sum", value="Beta")
    label = selection("Summing their Beta values: 1+3+5=9", query=query, answer=8)
    assert label["status"] == "FAIL"
    assert label["observed"]["categories"] == ["Feb", "Jan", "Mar"]


def test_arithmetic_range_is_not_entire_selection():
    query = dict(family="TOPK", k=2, direction="largest", aggregate="range", value="Beta")
    assert selection("Beta range=5-3=2", query=query, answer=2)["status"] == "UNKNOWN"


@pytest.mark.parametrize("expression", [
    "Mean=(10+14)/2+7=19",  # post-reduction offset is not a selected datum
    "Sum=10+14+999=1023",  # unknown constant cannot be discarded
    "Mean=(10+14)/0=12",
    "Sum=(10+14)/2=12",   # unsupported structural operation
    "Mean=(10+14)/2",     # no final equality
])
def test_reducer_constants_and_incomplete_expressions_are_not_selection(expression):
    assert selection("Feb: 13-3=10; Mar: 19-5=14; " + expression)["status"] == "UNKNOWN"


def test_declared_set_conflicting_with_adopted_operands_is_unknown():
    label = selection("selected=Feb,Mar; Jan:10-1=9; Mar:19-5=14; Mean=(9+14)/2=11.5")
    assert label["status"] == "UNKNOWN"


def test_repeated_numeric_mapping_is_never_arbitrarily_resolved():
    world = copy.deepcopy(WORLD)
    world["series"]["Beta"][0] = 3
    query = dict(family="TOPK", k=2, direction="largest", aggregate="sum", value="Beta")
    label = selection("Beta sum=3+5=8", world=world, query=query, answer=8)
    assert label["status"] == "UNKNOWN"
    assert label["ambiguous_candidates"]


def test_repeated_difference_source_is_ambiguous_despite_distinct_final_operands():
    world = copy.deepcopy(WORLD)
    world["series"]["Alpha"][0] = 11
    work = "Jan:11-1=10; Feb:13-3=10; Mar:19-5=14; Mean=(10+14)/2=12"
    assert selection(work, world=world)["status"] == "UNKNOWN"


def test_named_pair_resolves_only_that_source_not_other_names_elsewhere():
    world = copy.deepcopy(WORLD)
    world["series"]["Alpha"][0] = 13
    world["series"]["Beta"][0] = 3
    assert selection("Feb:13-3=10; Mar:19-5=14; Mean=(10+14)/2=12",
                     world=world)["status"] == "PASS"
    assert selection("Names Jan and Feb; 13-3=10;19-5=14;Mean=(10+14)/2=12",
                     world=world)["status"] == "UNKNOWN"


def test_missing_or_mismatched_evidence_cannot_support_numeric_trace():
    work = "Feb:13-3=10; Mar:19-5=14; Mean=(10+14)/2=12"
    assert selection(work, evidence=[])["status"] == "UNKNOWN"
    evidence = [dict(series="Beta", category="Feb", value=999)]
    assert selection(work, evidence=evidence)["status"] == "UNKNOWN"


def test_false_claimed_evidence_does_not_silently_become_world_truth():
    evidence = [dict(series="Beta", category="Feb", value=30),
                dict(series="Beta", category="Mar", value=50)]
    label, *_ = evaluate("Beta sum=30+50=80", evidence=evidence)
    assert label["dimensions"]["P"]["status"] == "FAIL"
    assert label["dimensions"]["S"]["status"] == "PASS"
    assert not label["normalization"]["sufficient"]


def test_answer_query_and_reward_do_not_choose_extraction():
    work = "Jan:10-1=9; Feb:13-3=10; Mar:19-5=14; Mean=(10+14)/2=12"
    a = selection(work, answer=12)
    b = selection(work, answer=999, query=dict(QUERY, include_start=True))
    assert a["observed"] == b["observed"]
    assert a["status"] == "PASS" and b["status"] == "FAIL"


@pytest.mark.parametrize("fields", [dict(truncated=True), dict(generation_status="ERROR")])
def test_technical_or_truncated_outputs_do_not_receive_positive_labels(fields):
    label, *_ = evaluate("Feb and Mar are selected.", **fields)
    assert [label["dimensions"][d]["status"] for d in "PSOE"] == ["UNKNOWN"] * 4


def test_invalid_json_and_invalid_protocol_are_not_upgraded():
    _, task, record, _ = evaluate("Feb and Mar are selected.")
    for text in (record["full_text"] + " trailing prose",
                 record["full_text"][:-1] + ',"answer":12}'):
        record["full_text"] = text
        score = task_contract.score(text, task["world"], task["query"])
        assert verify_semantics(task, record, score)["dimensions"]["S"]["status"] == "UNKNOWN"


def test_v1_dimensions_normalization_provenance_and_inputs_are_preserved():
    label, task, record, score = evaluate("Feb and Mar are selected.")
    snapshot = copy.deepcopy((task, record, score))
    base = verify_v1(task, record, score)
    assert label["verifier_version"] == VERIFIER_VERSION
    assert label["development_only"] and not label["frozen_gate_eligible"]
    assert not label["human_audited"]
    assert label["normalization"] == base["normalization"]
    assert all(label["dimensions"][d] == base["dimensions"][d] for d in "POE")
    assert (task, record, score) == snapshot
    assert base["dimensions"]["S"]["status"] == "UNKNOWN"
    lo, hi = label["dimensions"]["S"]["source_spans"][0]
    assert json.loads(record["full_text"][lo:hi]) == "Feb and Mar are selected."
    json.dumps(label, allow_nan=False)
