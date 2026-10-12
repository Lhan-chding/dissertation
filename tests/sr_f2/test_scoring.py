"""Exact syntax and sufficient-certificate acceptance without model execution."""

import copy
import json
import random
from decimal import Decimal
from fractions import Fraction

import pytest

from sr_f1.contract import _reference, gold_output, score
from sr_f1.evaluation import score_response
from sr_f2.scoring import (
    answer_fraction,
    classify_answer_syntax,
    last_balanced_json,
    score_record_v2,
    score_v2,
    sufficient_keys,
)

WORLD = {
    "categories": ["January", "February", "March", "April"],
    "series": {"Alpha": [30, 42, 26, 17], "Beta": [20, 51, 14, 21]},
    "unit": "count",
}


def certificate(world, query, *, full=False):
    output = gold_output(world, query)
    if not full:
        keys = set(sufficient_keys(world, query))
        output["evidence"] = [
            item for item in output["evidence"] if (item["series"], item["category"]) in keys
        ]
    return output


def test_1200_six_value_certificates_supersets_and_wrong_selected_categories():
    rng = random.Random(71001)
    for index in range(1200):
        world = copy.deepcopy(WORLD)
        world["series"] = {
            name: [rng.randrange(101) for _ in range(4)] for name in ("Alpha", "Beta")
        }
        query = _reference.base_query("TOPK")
        query.update(
            direction="largest" if index % 2 else "smallest",
            aggregate="mean" if index % 3 else "sum",
        )
        output = certificate(world, query)
        assert len(output["evidence"]) == 6
        result = score_v2(json.dumps(output), world, query)
        assert (result["E6"], result["P6"], result["J6"]) == (1, 1, 1)
        assert (result["E"], result["P"], result["J"], result["missing_count"]) == (0, 0, 0, 2)
        full = gold_output(world, query)
        extras = [item for item in full["evidence"] if item not in output["evidence"]]
        for count in (1, 2):
            superset = dict(output, evidence=output["evidence"] + extras[:count])
            scored = score_v2(json.dumps(superset), world, query)
            assert (scored["E6"], scored["P6"], scored["J6"]) == (1, 1, 1)
        wrong = copy.deepcopy(output)
        selected_beta = next(
            i for i, item in enumerate(wrong["evidence"]) if item["series"] == "Beta"
        )
        wrong["evidence"][selected_beta] = extras[0]
        assert score_v2(json.dumps(wrong), world, query)["E6"] == 0


def test_topk_tie_order_is_gold_order_not_reported_values():
    world = copy.deepcopy(WORLD)
    world["series"]["Alpha"] = [90, 90, 90, 20]
    query = _reference.base_query("TOPK")
    keys = sufficient_keys(world, query)
    assert keys[-2:] == [("Beta", "January"), ("Beta", "February")]
    output = certificate(world, query)
    output["evidence"][0]["value"] = 0
    scored = score_v2(json.dumps(output), world, query)
    assert (scored["E6"], scored["P6"], scored["J6"]) == (1, 0, 0)


@pytest.mark.parametrize(
    "defect", ["duplicate", "unknown", "illegal", "string", "missing_alpha", "optional_wrong"]
)
def test_sufficient_certificate_rejects_duplicates_illegal_and_wrong_supersets(defect):
    query = _reference.base_query("TOPK")
    output = certificate(WORLD, query)
    if defect == "duplicate":
        output["evidence"].append(copy.deepcopy(output["evidence"][0]))
    elif defect == "unknown":
        output["evidence"].append(dict(series="Gamma", category="January", value=5))
    elif defect == "illegal":
        output["evidence"][0]["value"] = 101
    elif defect == "string":
        output["evidence"][0]["value"] = "30.0"
    elif defect == "missing_alpha":
        output["evidence"] = [
            item
            for item in output["evidence"]
            if not (item["series"] == "Alpha" and item["category"] == "April")
        ]
    else:
        output["evidence"].append(dict(series="Beta", category="April", value=20))
    result = score_v2(json.dumps(output), WORLD, query)
    assert result["P6"] == result["J6"] == 0
    assert result["E6"] == int(defect == "optional_wrong")
    if defect == "duplicate":
        assert result["duplicate_count"] == 1
    if defect == "unknown":
        assert result["extra_count"] == 1


@pytest.mark.parametrize("family", _reference.FAMILIES)
def test_all_existing_fields_unchanged_outside_decimal_answer_extension(family):
    query = _reference.base_query(family)
    output = gold_output(WORLD, query)
    mutations = [output, {"answer": output["answer"]}, {"evidence": output["evidence"]}]
    mutations += [dict(output, answer=value) for value in (True, None, "1/2", "1e1", [], 1.25)]
    mutations += [
        dict(output, evidence=output["evidence"][:-1]),
        dict(output, evidence=output["evidence"] * 2),
    ]
    for item in mutations:
        text = json.dumps(item)
        expected, actual = score(text, WORLD, query), score_v2(text, WORLD, query)
        assert {key: actual[key] for key in expected} == expected
        if family != "TOPK":
            assert [actual[key] for key in ("E6", "P6", "J6")] == [
                expected[key] for key in ("E", "P", "J")
            ]


@pytest.mark.parametrize("value", ["35.5", "035.500", "-0.000", "0.00000000186264514923095703125"])
def test_decimal_strings_are_exact_and_do_not_mutate_v1(value):
    assert answer_fraction(value, text=True) == Fraction(Decimal(value))
    with pytest.raises(ValueError):
        _reference.fraction_of(value, text=True)
    query = dict(_reference.base_query("TOPK"), aggregate="mean")
    output = dict(gold_output(WORLD, query), answer=value)
    actual = score_v2(json.dumps(output), WORLD, query)
    assert actual["L_answer"] == 1
    assert actual["A"] == int(Fraction(value) == Fraction(71, 2))
    assert score(json.dumps(output), WORLD, query)["L_answer"] == 0


@pytest.mark.parametrize(
    "value",
    [
        "+1.5",
        ".5",
        "1.",
        " 1.5",
        "1.5 ",
        "1.5e1",
        "1/0",
        "NaN",
        "Infinity",
        "0.000000000931322574615478515625",
        "1000000000001.0",
    ],
)
def test_v2_does_not_broaden_other_strings_or_numeric_limits(value):
    query = _reference.base_query("TOPK")
    assert score_v2(json.dumps({"answer": value}), WORLD, query)["L_answer"] == 0


@pytest.mark.parametrize(
    "number", ["1e999999999", "1e-999999999", "1e99999999999999999999999999999999999999"]
)
def test_extreme_numeric_exponents_fail_bounded(number):
    query = _reference.base_query("TOPK")
    result = score_v2('{"answer":' + number + "}", WORLD, query)
    assert result["A"] == result["J6"] == 0


@pytest.mark.parametrize(
    "raw,kind,v1,v2",
    [
        ('{"answer":"12.5"}', "decimal_string", 0, 1),
        ('{"answer":"25/2"}', "fraction_string", 1, 1),
        ('{"answer":"12"}', "integer_string", 1, 1),
        ('{"answer":12}', "integer_number", 1, 1),
        ('{"answer":12.5}', "decimal_number", 1, 1),
        ('{"answer":null}', "null", 1, 1),
        ('{"answer":true}', "boolean", 0, 0),
        ('{"answer":"1.25e1"}', "exponent_string", 0, 0),
        ('{"answer":" 12.5"}', "whitespace_string", 0, 0),
        ('{"answer":"12.5 counts"}', "other_string", 0, 0),
        ('{"answer":[]}', "array", 0, 0),
        ('{"answer":{}}', "object", 0, 0),
        ("{}", "missing_answer", 0, 0),
        ('{"other":12.5}', "invalid_root", 0, 0),
        ('{"answer":1,"answer":2}', "invalid_json", 0, 0),
    ],
)
def test_blind_classifier_has_no_gold_inputs(raw, kind, v1, v2):
    result = classify_answer_syntax(raw)
    assert (result["answer_syntax"], result["v1_parseable"], result["v2_parseable"]) == (
        kind,
        v1,
        v2,
    )


def test_answer_only_preserves_v1_schema_and_rejects_evidence():
    query = _reference.base_query("TOPK")
    task = dict(world=WORLD, query=query)
    for raw in ('{"answer":71}', '{"answer":"71/1"}', "{}", '{"evidence":[],"answer":71}', "bad"):
        assert score_v2(raw, WORLD, query, "answer_only") == score_response(
            raw, task, "answer_only"
        )


@pytest.mark.parametrize("protocol", ["free", "FREE", "list", "LIST"])
def test_diagnostic_protocol_uses_last_object_only(protocol):
    query = _reference.base_query("TOPK")
    text = 'First {"answer":0}. Explanation then\n{"answer":"71.0"}'
    assert score_v2(text, WORLD, query, protocol)["A"] == 1
    assert classify_answer_syntax(text, protocol)["answer_syntax"] == "decimal_string"
    assert score_v2(text + '\n{"answer":false}', WORLD, query, protocol)["L_answer"] == 0
    assert score_v2(text + "\n{invalid}", WORLD, query, protocol)["L_json"] == 0
    assert score_v2(text, WORLD, query)["L_json"] == 0


def test_balanced_extractor_handles_nesting_strings_and_partial_suffix():
    text = 'prose {"nested":{"x":"} \\" {"}}\n{"answer":71}\n{"incomplete":'
    assert last_balanced_json(text) == '{"answer":71}'
    assert last_balanced_json("no object") is None
    assert last_balanced_json("x" * 65537) is None


def test_stored_format_error_is_preserved_even_when_raw_answer_is_correct():
    query = _reference.base_query("TOPK")
    record = dict(
        raw_text=json.dumps(gold_output(WORLD, query)),
        protocol="evidence_answer",
        format_protocol_error="trailing_after_close",
    )
    result = score_record_v2(record, dict(world=WORLD, query=query))
    assert result["reason"] == "trailing_after_close"
    assert result["L_answer"] == result["L_json"] == result["J6"] == 0


@pytest.mark.parametrize(
    "value,expected",
    [
        ("50+21", "expression_string"),
        ("1 / 2", "expression_string"),
        ("(59 + (-16) + 5)/3", "expression_string"),
        ("(59 + (\u221216) + 5)\u00f73", "expression_string"),
        ("3\u00d724-1", "expression_string"),
        ("NaN", "nonfinite_string"),
        ("nan", "nonfinite_string"),
        ("+INF", "nonfinite_string"),
        ("-Infinity", "nonfinite_string"),
        (" NaN", "whitespace_string"),
        ("50+21 ", "whitespace_string"),
        ("+.5", "other_string"),
        ("12.5 counts", "other_string"),
        ("float('inf')", "other_string"),
    ],
)
def test_blind_expression_nonfinite_strata_never_change_answer_acceptance(value, expected):
    query = _reference.base_query("TOPK")
    raw = json.dumps({"answer": value})
    classified = classify_answer_syntax(raw)
    assert classified["answer_syntax"] == expected
    assert classified["v1_parseable"] == classified["v2_parseable"] == 0
    assert score_v2(raw, WORLD, query)["L_answer"] == 0
    assert score(raw, WORLD, query)["L_answer"] == 0


@pytest.mark.parametrize(
    "value,expected",
    [
        ("71", "integer_string"),
        ("71/1", "fraction_string"),
        ("71.0", "decimal_string"),
        ("7.1e1", "exponent_string"),
    ],
)
def test_blind_expression_subclassification_preserves_numeric_grammar_priority(value, expected):
    assert classify_answer_syntax(json.dumps({"answer": value}))["answer_syntax"] == expected
