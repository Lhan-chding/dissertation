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


def test_complete_adoption_linked_values_are_independent_of_structured_facts():
    work = MEAN_DECLARATION + "The arithmetic mean is (46+73)/2 = 119/2 = 59.5"
    evidence = [dict(series="Alpha", category=c, value=v)
                for c, v in zip(MEAN_WORLD["categories"], MEAN_WORLD["series"]["Alpha"],
                                strict=True)]
    assert selection(work, world=MEAN_WORLD, query=MEAN_QUERY,
                         evidence=evidence, answer=59.5)["status"] == "PASS"
    duplicate = copy.deepcopy(MEAN_WORLD)
    duplicate["series"]["Beta"][0] = 46
    assert selection(work, world=duplicate, query=MEAN_QUERY, answer=59.5)["status"] == "PASS"
    unlinked = ("The two categories with the largest Alpha values are April and March. "
                "The arithmetic mean is (46+73)/2 = 119/2 = 59.5")
    assert selection(unlinked, world=MEAN_WORLD, query=MEAN_QUERY,
                     evidence=evidence, answer=59.5)["status"] == "UNKNOWN"
    assert selection(unlinked, world=duplicate, query=MEAN_QUERY,
                     answer=59.5)["status"] == "UNKNOWN"


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


@pytest.mark.parametrize("declaration", [
    "The top 2 categories with largest Alpha values are: April (66), March (34)",
    "The two categories with largest Alpha values are April (66) and March (34)",
    "The two largest Alpha values are in April (66) and March (34)",
    "The two largest Alpha values are at April (66) and March (34)",
    "Selected top 2 Alpha categories: April (66), March (34)",
    "Selected categories with largest Alpha values: April (66), March (34)",
    "Identify top 2 Alpha values: 66 (April), 34 (March)",
    "The 2 largest Alpha values are: April=66, March=34",
])
def test_r5_natural_named_ranks_preserve_actual_rank_series(declaration):
    assert selection(declaration, world=MEAN_WORLD, query=MEAN_QUERY,
                     answer=59.5)["status"] == "PASS"
    assert selection(declaration.replace("Alpha", "Beta"), world=MEAN_WORLD,
                     query=MEAN_QUERY, answer=59.5)["status"] == "FAIL"


def test_r5_beta_rank_and_eight_categories_are_supported():
    world = dict(categories=["January", "February", "March", "April", "May", "June",
                             "July", "August"], domain=[0, 100],
                 series=dict(Alpha=[80, 70, 60, 50, 40, 30, 20, 10],
                             Beta=[10, 20, 30, 40, 50, 60, 70, 80]))
    query = dict(family="TOPK", k=2, direction="largest", selection_series="Beta",
                 value="Alpha", aggregate="sum")
    work = ("The two largest Beta values are in August (80) and July (70). "
            "Their corresponding Alpha values are 10 and 20. Sum=10+20=30")
    selected = selection(work, world=world, query=query, answer=30)
    assert selected["status"] == "PASS"
    assert selected["observed"]["selection_series"] == ["Beta"]
    assert selection(work.replace("largest Beta", "largest Alpha"), world=world,
                     query=query, answer=30)["status"] == "FAIL"


def test_r5_contextual_rank_retains_observed_beta_series():
    work = ("The Beta values for each month are: January=31, February=30, March=82, April=88. "
            "The top two are April (88) and March (82)")
    query = dict(MEAN_QUERY, selection_series="Beta")
    assert selection(work, query=query)["status"] == "PASS"
    assert selection(work, query=MEAN_QUERY)["status"] == "FAIL"
    assert selection(work.replace("The Beta values", "The chart values"),
                     query=query)["status"] == "UNKNOWN"


@pytest.mark.parametrize("correspondence", [
    "Their corresponding Beta values are 35, 68, and 35",
    "The corresponding Beta values for these categories are February=35, March=68, June=35",
    "Now, sum their corresponding Beta values: February Beta=35, March Beta=68, June Beta=35",
])
def test_r5_real_duplicate_values_are_bound_to_complete_named_adoption(correspondence):
    world = dict(categories=["January", "February", "March", "April", "May", "June"],
                 domain=[0, 100], series=dict(Alpha=[20, 84, 79, 30, 40, 62],
                                            Beta=[35, 35, 68, 35, 10, 35]))
    query = dict(MEAN_QUERY, k=3)
    work = ("The three categories with largest Alpha values are February (84), March (79), "
            "and June (62). " + correspondence + ". Sum=35+68+35=138; Mean=138/3=46")
    evidence = [dict(series="Alpha", category="February", value=84)]
    label, task, record, score = evaluate(work, world=world, query=query,
                                        evidence=evidence, answer=46)
    assert label["dimensions"]["S"]["status"] == "PASS"
    assert label["dimensions"]["S"]["observed"]["categories"] == ["February", "June", "March"]
    assert all(label["dimensions"][d] == verify_v2(task, record, score)["dimensions"][d]
               for d in "POE")


@pytest.mark.parametrize("middle", [
    "Their corresponding Beta values are April=46, January=73",
    "Their corresponding Beta values are 46, 73, 19",
    "Their corresponding Beta values are 46",
    "Their corresponding Beta values are April=46, April=73",
    "Their corresponding Beta values are April=46, Unknown=73",
    "Their corresponding Beta values are 46 and 73; Use January instead of March",
    "Their corresponding Beta values are 46 and 73; Take only April",
    "Their corresponding Beta values are 46 and 73; Selected: April, January",
    "Their corresponding Beta values are 46 and 73; The top 2 Beta values are April, March",
])
def test_r5_linkage_cannot_hide_missing_extra_repeated_or_replaced_members(middle):
    work = ("The top 2 Alpha values are April and March; " + middle
            + "; Mean=(46+73)/2=59.5")
    assert selection(work, world=MEAN_WORLD, query=MEAN_QUERY,
                     answer=59.5)["status"] == "UNKNOWN"


@pytest.mark.parametrize("calculation", ["Alpha mean=(46+73)/2=59.5", "Mean=(46+19)/2=32.5"])
def test_r5_conflicting_series_or_operands_cannot_override_correspondence(calculation):
    assert selection(MEAN_DECLARATION + calculation, world=MEAN_WORLD,
                     query=MEAN_QUERY, answer=59.5)["status"] == "UNKNOWN"


def test_r5_linked_extraction_does_not_depend_on_target_or_answer():
    work = MEAN_DECLARATION + "Mean=(46+73)/2=59.5"
    left = selection(work, world=MEAN_WORLD, query=MEAN_QUERY)
    right = selection(work, world=MEAN_WORLD, query=dict(MEAN_QUERY, direction="smallest"),
                      answer=999)
    assert left["observed"] == right["observed"]
    assert left["status"] == "PASS" and right["status"] == "FAIL"


@pytest.mark.parametrize("pairs", [
    "Differences: Jan (51-31=20), Feb (84-30=54), Mar (30-82=-52), Apr (20-88=-68)",
    "Calculate (Alpha - Beta) for each month from January to April: "
    "Jan:51-31=20, Feb:84-30=54, Mar:30-82=-52, Apr:20-88=-68",
    "For January: Alpha=51, Beta=31 → difference=20; "
    "In February: Alpha=84, Beta=30 → difference=54; "
    "March: Alpha=30, Beta=82 → difference=30-82=-52; "
    "Alpha - Beta in April: 20-88=-68",
])
def test_r5_named_work_differences_support_complete_sum_without_fact_lookup(pairs):
    work = pairs + ". Sum of differences=20+54-52-68=-46; Mean=-46/4=-11.5"
    label, task, record, score = evaluate(work, evidence=[dict(series="Alpha", category="January",
                                                            value=51)])
    assert label["dimensions"]["S"]["status"] == "PASS"
    assert all(label["dimensions"][d] == verify_v2(task, record, score)["dimensions"][d]
               for d in "POE")


def test_r5_named_difference_cannot_change_series_or_hide_inconsistent_pair_assignments():
    work = ("January: Alpha=51, Beta=31 → difference=84-30=54; "
            "February:84-30=54; Sum=54+54=108")
    assert selection(work)["status"] == "UNKNOWN"
    assert selection(PAIRS + "Beta sum=20+54-52-68=-46")["status"] == "UNKNOWN"
    assert selection("January:51-31=20; February:50-30=20; Sum=20+20=40")["status"] == "UNKNOWN"


@pytest.mark.parametrize("suffix", ["Beta range=31-30=1", "Beta range=1", "Beta maximum=31"])
def test_r5_range_summary_requires_consistent_full_adopted_set(suffix):
    work = "Selected: Jan, Feb; " + suffix
    assert selection(work, query=dict(QUERY, end=1))["status"] == "PASS"
    assert selection(suffix, query=dict(QUERY, end=1))["status"] == "UNKNOWN"


@pytest.mark.parametrize("suffix", [
    "Beta range=88-30=58", "Beta range=31-30=2", "Range=31-30=1",
    "Beta range=31-30=1 for January and April", "Beta maximum=88",
])
def test_r5_inconsistent_or_unbound_range_summary_stays_unknown(suffix):
    assert selection("Selected: Jan, Feb; " + suffix,
                     query=dict(QUERY, end=1))["status"] == "UNKNOWN"


def test_r5_full_correspondence_supports_scalar_sum_and_range_without_world_completion():
    work = "Selected: Jan, Feb; Their corresponding Beta values are 31 and 30; Range=31-30=1"
    evidence = [dict(series="Alpha", category="January", value=51)]
    assert selection(work, query=dict(QUERY, end=1), evidence=evidence)["status"] == "PASS"
    assert selection(work.replace("Range=31-30=1", "Sum=61; Mean=61/2=30.5"),
                     query=dict(QUERY, end=1), evidence=evidence)["status"] == "PASS"
    assert selection(work.replace("Range=31-30=1", "Sum=62; Mean=62/2=31"),
                     query=dict(QUERY, end=1), evidence=evidence)["status"] == "UNKNOWN"


def test_r5_explicit_withdrawal_scopes_only_later_work():
    work = "Selected: January; Discard the previous selection. Selected: February, March, April"
    selected = selection(work, query=dict(QUERY, include_start=False))
    assert selected["status"] == "PASS"
    assert selected["observed"]["withdrawn_work"].endswith("selection. ")
    assert selected["observed"]["explicit_claims"][0]["source_text"].startswith("Selected:")


@pytest.mark.parametrize("withdrawal", [
    "Perhaps discard the previous selection", "Do not discard the previous selection",
    "Should we discard the previous selection?", "Discard the previous calculation only if needed",
])
def test_r5_speculative_or_negated_withdrawal_cannot_erase_conflict(withdrawal):
    work = "Selected: January; " + withdrawal + ". Selected: February, March, April"
    assert selection(work, query=dict(QUERY, include_start=False))["status"] == "UNKNOWN"


def test_r5_incompatible_exclusion_annotation_stays_unknown():
    for annotation in ["excluding start and including end", "excluding January",
                       "excluding start, including end"]:
        work = "Selected: January, February, March, April (" + annotation + "); " + CHAIN
        assert selection(work)["status"] == "UNKNOWN"


@pytest.mark.parametrize("replacement", [
    "Exclude February", "Include March instead of February", "January and March instead",
    "Remove February", "Omit February", "Drop February",
])
def test_r5_intermediate_replacement_cannot_be_hidden_by_a_later_supported_sum(replacement):
    work = "Selected: January, February. " + replacement + ". Alpha sum=51+84=135"
    assert selection(work, query=dict(QUERY, end=1))["status"] == "UNKNOWN"


@pytest.mark.parametrize("mode", ["Beta", "beta", "BETA", "Alpha", "alpha"])
def test_r5_named_difference_binding_respects_any_case_of_explicit_series(mode):
    assert selection(PAIRS + mode + " sum=20+54-52-68=-46")["status"] == "UNKNOWN"


@pytest.mark.parametrize("work", [
    "Selected categories excluding start (January) and end (April): February and March",
    "Selected categories excluding January (start) and April (end) are February and March",
    "Selected categories: February and March (excluding January and April)",
])
def test_r5_complete_boundary_annotated_list_is_observable_without_completing_it(work):
    assert selection(work, query=dict(QUERY, include_start=False,
                                     include_end=False))["status"] == "PASS"
    assert selection(work.replace("February and March", "March"),
                     query=dict(QUERY, include_start=False, include_end=False))["status"] == "FAIL"
    assert selection(work.replace("February and March", "January and March"))["status"] == "UNKNOWN"


def test_r5_endpoint_membership_annotations_cannot_contradict_complete_list():
    work = ("Selected categories: January, February, March, April "
            "(start=January included, end=April included)")
    assert selection(work)["status"] == "PASS"
    assert selection(work.replace("January included", "January excluded"))["status"] == "UNKNOWN"


@pytest.mark.parametrize("work", [
    "The Alpha values for each category are: January=51, February=84, March=30, April=20. "
    "Sorting in descending order: February (84), January (51). These are the top 2",
    "Sorting Alpha values in descending order: February (84), January (51), March (30), "
    "April (20). The top 2 are February, January",
])
def test_r5_local_sorting_context_retains_named_selection_and_series(work):
    assert selection(work, query=MEAN_QUERY)["status"] == "PASS"
    assert selection(work.replace("Alpha", "Beta"), query=MEAN_QUERY)["status"] == "FAIL"
    assert selection(work.replace("Alpha", "chart"), query=MEAN_QUERY)["status"] == "UNKNOWN"


@pytest.mark.parametrize("annotation", [
    "Using the printed category order for ties, January comes before February",
    "Since January and February are tied for largest, we use printed category order "
    "— so we take January first, then February",
])
def test_r5_same_set_tie_order_annotation_does_not_erase_selection(annotation):
    work = "The top 2 Alpha values are January (84), February (84). " + annotation
    assert selection(work, query=MEAN_QUERY)["status"] == "PASS"
    assert selection(work.replace(annotation, annotation.replace("February", "March")),
                     query=MEAN_QUERY)["status"] == "UNKNOWN"
    assert selection(work + "; Include March instead of January; Alpha sum=51+84=135",
                     query=MEAN_QUERY)["status"] == "UNKNOWN"


@pytest.mark.parametrize("declaration", [
    "Largest two Alpha values are April (66) and March (34)",
    "The largest two Alpha values are April (66) and March (34)",
    "The two Alpha values are highest in April (66) and March (34)",
])
def test_r5_rank_word_order_does_not_veto_a_complete_named_set(declaration):
    work = declaration + "; Beta mean=(46+73)/2=59.5"
    assert selection(work, world=MEAN_WORLD, query=MEAN_QUERY)["status"] == "PASS"
    assert selection(work.replace("Alpha", "Beta"), world=MEAN_WORLD,
                     query=MEAN_QUERY)["status"] == "FAIL"
    assert selection(work.replace("April (66)", "January (66)"), world=MEAN_WORLD,
                     query=MEAN_QUERY)["status"] == "UNKNOWN"


@pytest.mark.parametrize("header", [
    "The Beta values across categories are",
    "From the chart, Beta values are",
    "The Beta values (orange line with square markers) for each month are",
    "Beta values by category",
    "The Beta values per month are",
    "The Beta values per month (from bar heights) are",
])
@pytest.mark.parametrize("rank", [
    "The two largest are April (88) and March (82)",
    "Top 2: 88 (Apr), 82 (Mar)",
])
def test_r5_local_series_header_variants_keep_context_and_conflicts(header, rank):
    work = (header + ": January=31, February=30, March=82, April=88. "
            + rank + ". Mean=(88+82)/2=85")
    query = dict(MEAN_QUERY, selection_series="Beta")
    assert selection(work, query=query)["status"] == "PASS"
    assert selection(work, query=MEAN_QUERY)["status"] == "FAIL"
    assert selection(work.replace("Beta values", "Beta and Alpha values"),
                     query=query)["status"] == "UNKNOWN"


@pytest.mark.parametrize("note", ["not Beta", "use Alpha", "except January", "only 2 categories"])
def test_r5_chart_context_parenthesis_cannot_carry_a_hidden_replacement(note):
    work = ("The Beta values (" + note + ") for each month are: "
            "January=31, February=30, March=82, April=88. "
            "The two largest are April (88) and March (82). Mean=(88+82)/2=85")
    assert selection(work, query=dict(MEAN_QUERY, selection_series="Beta"))["status"] == "UNKNOWN"


@pytest.mark.parametrize("suffix", [" respectively", ", respectively"])
def test_r5_respectively_comma_is_only_a_complete_list_terminator(suffix):
    work = ("The top 2 Alpha values are April and March; "
            "Their corresponding Beta values are 46 and 73" + suffix + "; Mean=(46+73)/2=59.5")
    assert selection(work, world=MEAN_WORLD, query=MEAN_QUERY)["status"] == "PASS"
    assert selection(work.replace("46 and 73", "46"), world=MEAN_WORLD,
                     query=MEAN_QUERY)["status"] == "UNKNOWN"
    assert selection(work.replace(suffix, suffix + " using January instead"),
                     world=MEAN_WORLD, query=MEAN_QUERY)["status"] == "UNKNOWN"


@pytest.mark.parametrize("adoption", [
    "So we select February and April",
    "So we SELECT February and April",
    "So we select only February and April",
    "So we adopt February and April",
    "We adopted February and April",
])
def test_r5_later_active_adoption_cannot_bind_values_to_an_earlier_rank(adoption):
    world = dict(categories=["January", "February", "March", "April"], domain=[0, 100],
                 series=dict(Alpha=[57, 72, 20, 55], Beta=[12, 62, 35, 22]))
    work = ("The highest Alpha values are February (72) and January (57). "
            "April (55) is third. " + adoption + ". Their Beta values are 62 and 22. "
            "Mean = (62 + 22)/2 = 42")
    assert selection(work, world=world, query=MEAN_QUERY, answer=42)["status"] == "UNKNOWN"
