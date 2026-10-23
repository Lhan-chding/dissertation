"""Development regressions; these are not a population accuracy audit."""

import copy
import json

import pytest

from sr_f4 import task_contract
from sr_f4.verification_v2 import verify_semantics as verify_v2
from sr_f4.verification_v3 import VERIFIER_VERSION, _normalize_months, verify_semantics

WORLD = dict(
    categories=["January", "February", "March", "April"],
    domain=[-100, 100],
    series=dict(Alpha=[51, 84, 30, 20], Beta=[31, 30, 82, 88]),
)
QUERY = dict(
    family="INTERVAL",
    start=0,
    end=3,
    include_start=True,
    include_end=True,
    aggregate="mean",
    value="diff",
)
PAIRS = (
    "Compute Alpha - Beta for each month: Jan:51-31=20; Feb:84-30=54; "
    "Mar:30-82=-52; Apr:20-88=-68. "
)
CHAIN = PAIRS + "Sum = 20+54-52-68 = -46. Count=4. Mean=-46/4=-11.5"


def evaluate(work=CHAIN, *, world=None, query=None, answer=-11.5, evidence=None, **fields):
    world, query = copy.deepcopy(world or WORLD), copy.deepcopy(query or QUERY)
    if evidence is None:
        evidence = [
            dict(series=s, category=c, value=world["series"][s][i])
            for s in world["series"]
            for i, c in enumerate(world["categories"])
        ]
    task = dict(qid="dev", world_id="dev_world", world=world, query=query)
    record = dict(
        full_text=json.dumps(dict(evidence=evidence, work=work, answer=answer)),
        row_id="dev_row",
        generation_status="COMPLETE",
        **fields,
    )
    score = task_contract.score(record["full_text"], world, query)
    return verify_semantics(task, record, score), task, record, score


def selection(work=CHAIN, **kwargs):
    return evaluate(work, **kwargs)[0]["dimensions"]["S"]


def test_realistic_abbreviation_and_intermediate_chain():
    label, task, record, score = evaluate()
    assert label["dimensions"]["S"]["status"] == "PASS"
    assert verify_v2(task, record, score)["dimensions"]["S"]["status"] == "UNKNOWN"
    traces = label["dimensions"]["S"]["observed"]["final_reduction_claims"]
    chain = next(t for t in traces if t.get("rule") == "V3_LITERAL_SUM_TO_MEAN_CHAIN")
    assert chain["categories"] == sorted(WORLD["categories"])
    assert chain["divisor"] == "4"
    assert chain["intermediate_source"]["claimed_result"] == "-46"
    for trace in traces:
        lo, hi = trace["decoded_active_work_span"]
        assert CHAIN[lo:hi] == trace["source_text"]
    lo, hi = label["dimensions"]["S"]["source_spans"][0]
    assert json.loads(record["full_text"][lo:hi]) == CHAIN


def test_aliases_for_declarations_and_wrong_set():
    assert selection("Selected: Jan, Feb, Mar, Apr")["status"] == "PASS"
    assert selection("Selected: Jan, Feb, Mar")["status"] == "FAIL"
    assert selection("The question refers to Jan through Apr")["status"] == "UNKNOWN"


def test_ambiguous_month_vocabulary_is_not_collapsed():
    normalized, _, aliases = _normalize_months("Jan and January", ["Jan", "January"])
    assert normalized == "Jan and January"
    assert aliases == []


@pytest.mark.parametrize(
    "tail",
    [
        "Beta mean=-46/4=-11.5",  # source is diff, not Beta
        "Mean=-45/4=-11.25",  # wrong intermediate source reference
        "Mean=-46/0=-11.5",
        "Sum=20+54-52-68=-46; Mean=-46/4=-11.5",  # two possible Sum sources
    ],
)
def test_incomplete_or_conflicting_chain_never_guesses(tail):
    assert selection(PAIRS + "Sum=20+54-52-68=-46; " + tail)["status"] == "UNKNOWN"


def test_earlier_missing_source_and_later_source_not_reused():
    assert selection("Mean=-46/4=-11.5; " + PAIRS + "Sum=20+54-52-68=-46")["status"] == "UNKNOWN"
    assert selection("correction: Sum=20+54-52-68=-46; Mean=-46/4=-11.5")["status"] == "UNKNOWN"


def test_withdrawal_does_not_supply_pair_provenance():
    work = PAIRS + "correction: Sum=20+54-52-68=-46; Mean=-46/4=-11.5"
    assert selection(work)["status"] == "UNKNOWN"


def test_mixed_explicit_selection_stays_unknown():
    assert selection("Selected: Jan, Feb; " + CHAIN)["status"] == "UNKNOWN"


def test_no_answer_query_or_reward_used_to_choose_extraction():
    left = selection()
    _label, task, record, score = evaluate(answer=999, query=dict(QUERY, include_start=False))
    score["J"] = 0.0
    right = verify_semantics(task, record, score)["dimensions"]["S"]
    assert left["observed"] == right["observed"]
    assert left["status"] == "PASS" and right["status"] == "FAIL"


def test_rhs_cannot_become_a_world_datum():
    world = copy.deepcopy(WORLD)
    world["series"]["Beta"][0] = 112
    world["domain"] = [-100, 200]
    # Source sums Beta values; its RHS happens to equal January's datum.
    work = "Beta sum=30+82=112; Mean=112/2=56"
    q = dict(
        family="TOPK",
        k=2,
        direction="largest",
        selection_series="Alpha",
        value="Beta",
        aggregate="mean",
    )
    label = selection(work, world=world, query=q, answer=56)
    assert label["observed"]["categories"] == ["February", "March"]
    assert label["status"] == "FAIL"


def test_unrecognized_withdrawal_between_sum_and_mean_abstains():
    work = PAIRS + "Sum=20+54-52-68=-46; Discard that sum; Mean=-46/4=-11.5"
    assert selection(work)["status"] == "UNKNOWN"


def test_same_set_wrong_explicit_rank_series_still_fails():
    q = dict(family="TOPK", k=2, direction="largest", value="Beta", aggregate="sum")
    assert selection("The top 2 Beta values are Feb, Jan", query=q, answer=61)["status"] == "FAIL"


def test_false_sum_keeps_selection_and_original_execution_failure_separate():
    work = CHAIN.replace("= -46", "= -45").replace("Mean=-46/4=-11.5", "Mean=-45/4=-11.25")
    label, *_ = evaluate(work, answer=-11.25)
    assert label["dimensions"]["S"]["status"] == "PASS"
    assert label["dimensions"]["E"]["status"] == "FAIL"


@pytest.mark.parametrize("tail", [
    "The final average of January and February is (20+54)/2 = 37",
    "Final result = (20+54)/2 = 37",
    "For the answer, take January and February: (20+54)/2 = 37",
])
def test_unrecognized_final_replacement_cannot_keep_earlier_pass(tail):
    result = selection(CHAIN + "; " + tail, answer=37)
    assert result["status"] == "UNKNOWN"
    assert result["observed"]["unsupported_tail"]


@pytest.mark.parametrize("suffix", [
    "54.666... which is 164/3.",
    "54.666... \N{RIGHTWARDS ARROW} 164/3",
])
def test_repeating_decimal_display_does_not_break_observed_chain(suffix):
    world = dict(categories=["January", "February", "March"], domain=[0, 100],
                 series=dict(Alpha=[90, 80, 70], Beta=[68, 49, 47]))
    query = dict(family="TOPK", k=3, direction="largest", value="Beta", aggregate="mean")
    work = "Beta sum=68+49+47=164; Mean=164/3=" + suffix
    label, task, record, score = evaluate(work, world=world, query=query, answer="164/3")
    assert label["dimensions"]["S"]["status"] == "PASS"
    chain = label["dimensions"]["S"]["observed"]["final_reduction_claims"][-1]
    assert chain["claimed_result"] == "54.666..."
    assert chain["result_display_kind"] == "ELLIPSIS_NOT_EXACT"
    assert label["dimensions"]["E"] == verify_v2(task, record, score)["dimensions"]["E"]


def test_inconsistent_ellipsis_annotation_stays_unknown():
    work = PAIRS + "Sum=20+54-52=-46; Mean=-46/3=-15.333... which is -47/3."
    assert selection(work)["status"] == "UNKNOWN"


def test_single_sentence_period_keeps_finite_result():
    result = selection(CHAIN + ".")
    assert result["status"] == "PASS"
    chain = result["observed"]["final_reduction_claims"][-1]
    assert chain["claimed_result"] == "-11.5"
    assert chain["result_display_kind"] == "FINITE_LITERAL"


@pytest.mark.parametrize("count", ["Count = 3", "Number of categories = 4", "Count=999"])
def test_scalar_count_does_not_replace_observed_categories(count):
    result = selection("Selected: Jan, Feb, Mar, Apr; " + count)
    assert result["status"] == "PASS"
    assert selection(count)["status"] == "UNKNOWN"


def test_count_with_new_members_is_not_scalar_annotation():
    assert selection(CHAIN + "; Count = 2 for January and February")["status"] == "UNKNOWN"


@pytest.mark.parametrize("expression, expected", [("80+79=159", "PASS"), ("80+76=156", "UNKNOWN")])
def test_natural_beta_sum_uses_actual_operands_and_compares_declaration(expression, expected):
    world = dict(categories=["January", "February", "March"], domain=[0, 100],
                 series=dict(Alpha=[90, 80, 70], Beta=[80, 79, 76]))
    query = dict(family="TOPK", k=2, direction="largest", value="Beta", aggregate="sum")
    work = "Selected: Jan, Feb; Sum of Beta values for these categories: " + expression
    result = selection(work, world=world, query=query, answer=159)
    assert result["status"] == expected
    trace = result["observed"]["final_reduction_claims"][0]
    assert trace["source_text"].strip().startswith("Sum of Beta")
    if expected == "UNKNOWN":
        assert trace["categories"] == ["January", "March"]
        assert selection(work.split("; ", 1)[1], world=world, query=query,
                         answer=159)["status"] == "FAIL"


def test_natural_sum_without_operands_is_not_trusted_from_pronoun():
    work = "Selected: Jan, Feb, Mar, Apr; Sum of Beta values for these categories: 42"
    assert selection(work)["status"] == "UNKNOWN"


@pytest.mark.parametrize("rhs, matches", [("-46/4", True), ("-45 / 4", False)])
def test_exact_fraction_rhs_is_recorded_separately_from_adopted_set(rhs, matches):
    work = PAIRS + "Sum=20+54-52-68=-46; Mean=-46/4=" + rhs
    result = selection(work)
    assert result["status"] == "PASS"
    chain = result["observed"]["final_reduction_claims"][-1]
    assert chain["result_display_kind"] == "EXACT_FRACTION"
    assert chain["display_matches_computation"] is matches


def test_zero_denominator_rhs_is_unsupported():
    assert selection(PAIRS + "Sum=20+54-52-68=-46; Mean=-46/4=-46/0")["status"] == "UNKNOWN"


MEAN_WORLD = dict(
    categories=["January", "February", "March", "April"], domain=[0, 100],
    series=dict(Alpha=[12, 9, 34, 66], Beta=[19, 11, 73, 46]),
)
MEAN_QUERY = dict(family="TOPK", k=2, direction="largest", value="Beta", aggregate="mean")
MEAN_DECLARATION = (
    "The two categories with the largest Alpha values are April (66) and March (34). "
    "For these categories, the Beta values are 46 and 73. "
)


@pytest.mark.parametrize("calculation", [
    "The arithmetic mean is (46 + 73) / 2 = 119 / 2 = 59.5",
    "Arithmetic mean = (46 + 73) / 2 = 59.5",
    "Mean = (46+73)/2 = 119/2 = 59.5",
    "Beta sum=46+73=119; Arithmetic mean = 119/2 = 59.5",
])
def test_natural_mean_chain_binds_first_operands_and_keeps_source(calculation):
    work = MEAN_DECLARATION + calculation
    label, task, record, score = evaluate(work, world=MEAN_WORLD, query=MEAN_QUERY, answer=59.5)
    selected = label["dimensions"]["S"]
    assert selected["status"] == "PASS"
    assert selected["observed"]["categories"] == ["April", "March"]
    trace = selected["observed"]["final_reduction_claims"][-1]
    lo, hi = trace["decoded_active_work_span"]
    assert trace["source_text"] == work[lo:hi]
    assert trace["claimed_result"] == "59.5"
    assert all(stage["display_matches_computation"] for stage in trace["equality_stages"])
    assert all(label["dimensions"][d] == verify_v2(task, record, score)["dimensions"][d]
               for d in "POE")


def test_wrong_arithmetic_in_mean_chain_does_not_change_selection():
    work = MEAN_DECLARATION + "The arithmetic mean is (46+73)/3 = 110/3 = 40"
    label, task, record, score = evaluate(work, world=MEAN_WORLD, query=MEAN_QUERY, answer=40)
    assert label["dimensions"]["S"]["status"] == "PASS"
    trace = label["dimensions"]["S"]["observed"]["final_reduction_claims"][-1]
    assert not any(stage["display_matches_computation"] for stage in trace["equality_stages"])
    assert label["dimensions"]["E"] == verify_v2(task, record, score)["dimensions"]["E"]


@pytest.mark.parametrize("calculation", [
    "The arithmetic mean is (46+73)/2 = (46+19)/2 = 32.5",
    "The arithmetic mean is (46+73)/2 = 119/0 = 59.5",
    "The arithmetic mean is (46+73)/2 = 119/2, using January instead",
    "The arithmetic mean is (46+73)/2 = 59.5; Final result = (46+19)/2 = 32.5",
    "The Alpha arithmetic mean is (46+73)/2 = 59.5",
])
def test_natural_mean_chain_cannot_hide_replacement_or_unsupported_stage(calculation):
    result = selection(MEAN_DECLARATION + calculation, world=MEAN_WORLD,
                       query=MEAN_QUERY, answer=59.5)
    assert result["status"] == "UNKNOWN"


def test_mean_chain_does_not_infer_missing_evidence_or_resolve_duplicate_values():
    work = MEAN_DECLARATION + "The arithmetic mean is (46+73)/2 = 119/2 = 59.5"
    evidence = [dict(series="Alpha", category=c, value=v)
                for c, v in zip(MEAN_WORLD["categories"], MEAN_WORLD["series"]["Alpha"],
                                strict=True)]
    assert selection(work, world=MEAN_WORLD, query=MEAN_QUERY,
                     evidence=evidence, answer=59.5)["status"] == "UNKNOWN"
    duplicate = copy.deepcopy(MEAN_WORLD)
    duplicate["series"]["Beta"][0] = 46
    assert selection(work, world=duplicate, query=MEAN_QUERY, answer=59.5)["status"] == "UNKNOWN"


def test_mean_chain_extraction_is_independent_of_query_and_answer():
    work = "The arithmetic mean is (46+73)/2 = 119/2 = 59.5"
    left = selection(work, world=MEAN_WORLD, query=MEAN_QUERY, answer=59.5)
    right = selection(work, world=MEAN_WORLD, query=dict(MEAN_QUERY, direction="smallest"),
                      answer=999)
    assert left["observed"] == right["observed"]
    assert left["status"] == "PASS" and right["status"] == "FAIL"


def test_direct_mean_ellipsis_keeps_approximate_display():
    work = "Beta arithmetic mean=(31+30+82)/3=143/3=47.666... which is 143/3"
    result = selection(work, query=dict(QUERY, end=2), answer="143/3")
    assert result["status"] == "PASS"
    trace = result["observed"]["final_reduction_claims"][-1]
    assert trace["claimed_result"] == "47.666..."
    assert trace["result_display_kind"] == "ELLIPSIS_NOT_EXACT"
    assert "exact_display_value" not in trace


@pytest.mark.parametrize("work", [
    "Selected categories are February, March, April (excluding January, including April)",
    "Selected categories: February, March, April (excluding January)",
    "Categories selected: Feb, Mar, Apr (exclude Jan, include Apr)",
    "Selected categories from January to April, excluding start (January) "
    "and including end (April): February, March, April",
    "Selected categories from January to April excluding start and including end: "
    "February, March, April",
])
def test_complete_natural_selection_requires_consistent_annotations(work):
    label, *_ = evaluate(work, query=dict(QUERY, include_start=False))
    selected = label["dimensions"]["S"]
    assert selected["status"] == "PASS"
    claim = selected["observed"]["explicit_claims"][0]
    lo, hi = claim["decoded_active_work_span"]
    assert claim["source_text"] == work[lo:hi]
    assert selection(work)["status"] == "FAIL"


@pytest.mark.parametrize("work", [
    "Selected categories: January, February, March, April (excluding January)",
    "Selected categories: February, March (including April)",
    "Selected categories: February, March, April (excluding January, except March)",
    "Selected categories: February, March, April (excluding January) but use only February",
    "Selected categories from January to April, excluding start (February) "
    "and including end (April): February, March, April",
    "Selected categories from January to April, excluding start and including end: "
    "February, March",
    "Selected categories from January to April, excluding start and including end",
    "Selected categories: February, March, April (excluding January); Selected: February",
])
def test_incomplete_or_conflicting_natural_selection_stays_unknown(work):
    assert selection(work)["status"] == "UNKNOWN"


def test_range_extrema_alone_still_do_not_supply_a_selection():
    assert selection("Range = 88 - 30 = 58")["status"] == "UNKNOWN"
    assert selection("Selected: Jan, Feb, Mar, Apr; Range = 88 - 30 = 58")["status"] == "UNKNOWN"


@pytest.mark.parametrize("fields", [dict(truncated=True), dict(generation_status="ERROR")])
def test_technical_failures_not_upgraded(fields):
    if "generation_status" in fields:
        label, task, record, score = evaluate()
        record.update(fields)
        label = verify_semantics(task, record, score)
    else:
        label, *_ = evaluate(**fields)
    assert all(label["dimensions"][d]["status"] == "UNKNOWN" for d in "PSOE")


def test_versions_normalization_and_inputs_are_preserved():
    label, task, record, score = evaluate()
    snapshot = copy.deepcopy((task, record, score))
    old = verify_v2(task, record, score)
    assert label["verifier_version"] == VERIFIER_VERSION
    assert label["measurement_mode"] == "PROGRAMMATIC_ONLY"
    assert label["development_only"] and not label["frozen_gate_eligible"]
    assert not label["model_consensus_implemented"] and not label["human_audited"]
    assert label["normalization"] == old["normalization"]
    assert all(label["dimensions"][d] == old["dimensions"][d] for d in "POE")
    assert (task, record, score) == snapshot
    json.dumps(label, allow_nan=False)
