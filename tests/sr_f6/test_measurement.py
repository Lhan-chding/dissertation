"""Native parity and field-local measurement regressions; no model calls."""

import copy
import itertools
import json
from decimal import Decimal

import pytest

from sr_f5.measurement import score_response as old_score
from sr_f6.measurement import (
    answer_only,
    lossless_legacy_projection,
    parse,
    score_response,
)
from srf6_ref.core import FAMILIES, oracle_a
from srf6_ref.credit import credit
from srf6_ref.design import build_bank
from srf6_ref.measurement import full_truth_response


@pytest.fixture(scope="module")
def bank():
    return build_bank("DEV_CAL", 8, 2026102711)


@pytest.fixture
def example(bank):
    task = bank[0]["tasks"]["P0"][0]
    return task, json.loads(full_truth_response(task["world"], task["query"]))


def score(task, obj, **kwargs):
    return score_response(json.dumps(obj), task["world"], task["query"], **kwargs)


def test_300_native_j_fixtures_and_600_semantic_mutations(bank):
    """50 fixtures/family, positive and negative native J, two mutants each."""
    checked = mutated = 0
    native_outcomes = set()
    for family in FAMILIES:
        tasks = [
            task
            for root in bank
            for factor in ("P0", "P1")
            for task in root["tasks"][factor]
            if task["family"] == family
        ][:50]
        assert len(tasks) == 50
        for index, task in enumerate(tasks):
            obj = json.loads(full_truth_response(task["world"], task["query"]))
            if index % 4 == 1:
                obj["answer"] = "9999"
            elif index % 4 == 2:
                obj["evidence"] = []
            elif index % 4 == 3:
                fact = obj["evidence"][0]
                fact["value"] = "18" if fact["value"] == "12" else "12"
            projected = {
                key: value for key, value in obj.items() if key not in {"bindings", "aggregate"}
            }
            native = old_score(json.dumps(projected), task["world"], task["query"])
            got = score(task, obj)
            assert got["V"] == 1
            assert got["J"] == got["J_legacy_projection"] == native["J"]
            native_outcomes.add(native["J"])
            checked += 1
            changed_selection = copy.deepcopy(obj)
            changed_selection["selected"] = ["unlisted-category", "unlisted-category"]
            changed_binding = copy.deepcopy(obj)
            changed_binding["bindings"] = [{"category": "unlisted-category", "operand": 999}]
            for changed in (changed_selection, changed_binding):
                result = score(task, changed)
                assert result["V"] == 1
                assert result["J"] == native["J"]
                mutated += 1
    assert (checked, mutated, native_outcomes) == (300, 600, {0, 1})


def test_long_decimal_keeps_native_j_and_literal(example):
    task, obj = example
    obj["work"] = 'Literal {"bindings": [1]} and Chinese 文本 stay unchanged.'
    raw = json.dumps(obj, ensure_ascii=False)
    value = oracle_a(task["world"], task["query"])["answer"]
    assert value.denominator == 1
    literal = str(value.numerator) + "." + "0" * 80
    raw = raw.replace('"answer": "' + str(value.numerator) + '"', '"answer": ' + literal)
    projected = lossless_legacy_projection(raw)
    assert '"answer": ' + literal in projected
    assert "文本" in projected
    assert isinstance(json.loads(projected, parse_float=Decimal)["answer"], Decimal)
    assert old_score(projected, task["world"], task["query"])["J"] == 1
    result = score_response(raw, task["world"], task["query"])
    assert result["A"] == result["V"] == result["J"] == 1


def test_projection_all_top_level_orders(example):
    _, obj = example
    obj["work"] = 'The string "aggregate" is not a JSON key here.'
    expected = {key: value for key, value in obj.items() if key not in {"bindings", "aggregate"}}
    for order in itertools.permutations(obj):
        raw = json.dumps({key: obj[key] for key in order}, indent=2)
        assert json.loads(lossless_legacy_projection(" \n" + raw + " \t")) == expected


@pytest.mark.parametrize(
    ("field", "value", "s", "b", "p"),
    [
        ("bindings", None, 1, None, 1),
        ("selected", None, None, 1, 1),
        ("aggregate", {}, 1, 1, 1),
        ("evidence", None, 1, 1, None),
        ("work", 7, 1, 1, 1),
    ],
)
def test_field_error_does_not_erase_other_measurements(example, field, value, s, b, p):
    task, obj = example
    obj[field] = value
    result = score(task, obj)
    assert (result["S"], result["B"], result["P"]) == (s, b, p)
    assert (result["A"], result["V"], result["J"]) == (1, 0, 0)
    assert not result["credit_eligible"]
    assert not any(result["eligible_components"].values())
    assert credit([result] * 8)["applied"] == [0.0] * 8


def test_null_operand_unknown_b_preserves_s(example):
    task, obj = example
    obj["bindings"][0]["operand"] = None
    result = score(task, obj)
    assert (result["S"], result["B"], result["V"], result["J"], result["A"]) == (1, None, 0, 0, 1)
    assert result["E_raw"] is None
    assert "B" in result["unknown_reasons"]


def test_missing_binding_field_still_measures_s_and_a(example):
    task, obj = example
    del obj["bindings"]
    result = score(task, obj)
    assert result["A"] == result["S"] == 1
    assert result["B"] is None
    assert result["J"] == result["V"] == 0
    with pytest.raises(ValueError, match="Invalid SR-F6"):
        lossless_legacy_projection(json.dumps(obj))


@pytest.mark.parametrize("claim", [[], ["Unknown"], ["Mug", "Mug"]])
def test_readable_wrong_selection_is_fail(example, claim):
    task, obj = example
    obj["selected"] = claim
    result = score(task, obj)
    assert (result["V"], result["S"], result["B"], result["A"], result["J"]) == (1, 0, 1, 1, 1)


@pytest.mark.parametrize("claim", [[], [{"category": "Unknown", "operand": 12}]])
def test_readable_wrong_binding_scope_is_fail(example, claim):
    task, obj = example
    obj["bindings"] = claim
    result = score(task, obj)
    assert (result["V"], result["S"], result["B"], result["A"], result["J"]) == (1, 1, 0, 1, 1)


@pytest.mark.parametrize("suffix", [" trailing", " {}", '\n{"answer": 0}'])
def test_no_gold_driven_extraction_or_trailing_object(example, suffix):
    task, obj = example
    result = score_response(json.dumps(obj) + suffix, task["world"], task["query"])
    assert result["A"] == result["V"] == result["J"] == 0
    assert result["S"] is result["B"] is None


@pytest.mark.parametrize("finished,reason", [(False, None), (False, "eos"), (True, "length")])
def test_incomplete_generation_excludes_credit_but_preserves_a(example, finished, reason):
    task, obj = example
    result = score(task, obj, finished=finished, finish_reason=reason)
    assert result["A"] == result["V"] == 1
    assert result["J"] == result["finished"] == 0
    assert result["S"] is result["B"] is None
    assert not result["credit_eligible"]


def test_truncated_json_and_duplicate_keys_rejected(example):
    task, obj = example
    raw = json.dumps(obj)
    for text in (raw[:-1], raw[:-1] + ',"answer": 0}', '{"answer": NaN}', "[]"):
        result = score_response(text, task["world"], task["query"])
        assert result["A"] == result["V"] == result["J"] == 0
        assert result["S"] is result["B"] is None


def test_invalid_protocol_with_length_finish_cannot_gain_known_semantics(example):
    task, obj = example
    del obj["bindings"]
    result = score(task, obj, finished=True, finish_reason="length")
    assert result["A"] == 1
    assert result["V"] == result["J"] == result["finished"] == 0
    assert result["S"] is result["B"] is None


def test_a_ignores_new_semantic_field_requirements(example):
    task, obj = example
    raw = json.dumps({"answer": obj["answer"]})
    result = score_response(raw, task["world"], task["query"])
    assert result["A"] == 1
    assert result["J"] == result["V"] == 0
    assert answer_only(raw, oracle_a(task["world"], task["query"])["answer"]) == 1


def test_input_and_extraction_do_not_mutate(example):
    task, obj = example
    original = copy.deepcopy(task)
    raw = json.dumps(obj)
    before = parse(raw)
    score_response(raw, task["world"], task["query"])
    assert task == original
    assert parse(raw) == before


def test_unknown_task_extension_fails_at_boundary(example):
    task, obj = example
    query = {**task["query"], "secret_new_semantics": True}
    with pytest.raises(ValueError, match="unsupported query field"):
        score_response(json.dumps(obj), task["world"], query)
