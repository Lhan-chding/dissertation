"""Public checklist and unchanged-parser checks on synthetic tasks only."""

import copy
import itertools
import json
import random

import pytest

from sr_f4 import task_contract
from sr_f6 import interface_protocol as v2
from sr_f6 import interface_v3_protocol as v3
from sr_f6.measurement import parse
from srf5_ref.oracle import required_certificate_keys


def _world(n=6, seed=9):
    rng = random.Random(seed)
    return {
        "categories": [f"Public_{i}" for i in range(n)],
        "domain": [10, 90],
        "series": {name: [rng.randint(10, 90) for _ in range(n)] for name in ("Alpha", "Beta")},
    }


def _topk(**overrides):
    return {
        "family": "TOPK",
        "selection_series": "Alpha",
        "value": "Beta",
        "aggregate": "sum",
        "direction": "largest",
        "k": 2,
        **overrides,
    }


def _interval(**overrides):
    return {
        "family": "INTERVAL",
        "value": "diff",
        "aggregate": "mean",
        "start": 1,
        "end": 4,
        "include_start": True,
        "include_end": True,
        "positive_only": False,
        **overrides,
    }


def test_new_prompt_is_one_candidate_and_does_not_mutate_v2_or_inputs():
    labels, query = _world()["categories"], _topk()
    original_labels, original_query = copy.deepcopy(labels), copy.deepcopy(query)
    old_fields = copy.deepcopy(v2.FIELDS)
    old_prompt = v2.public_prompt(labels, query)
    old_contract, old_shape = v2.format_contract(), v2.response_shape()
    prompt = v3.public_prompt(labels, query)
    assert v3.VERSION == "srf6-interface-v3" and v3.WIRE_PROTOCOL == "structured_v4"
    assert prompt != old_prompt and v3.public_prompt(labels, query) == prompt
    assert labels == original_labels and query == original_query
    assert old_fields == v2.FIELDS and v2.format_contract() == old_contract
    assert v2.public_prompt(labels, query) == old_prompt and v2.response_shape() == old_shape
    shape = v3.response_shape()
    assert shape == old_shape
    shape["properties"]["selected"]["items"]["enum"] = ["PRIVATE"]
    assert v3.response_shape() == old_shape and v2.response_shape() == old_shape


@pytest.mark.parametrize(
    "field", ["world", "gold_answer", "gold_selected", "factor", "certificate"]
)
def test_hidden_query_fields_are_rejected(field):
    with pytest.raises(ValueError):
        v3.public_prompt(_world()["categories"], _topk(**{field: "PRIVATE"}))


def test_public_argument_boundary_and_invalid_queries_match_v2():
    labels = _world()["categories"]
    for categories, query in [
        (_world(), _topk()),
        (tuple(labels), _topk()),
        ([*labels[:-1], 5], _topk()),
        (labels, _topk(k=True)),
        (labels, _interval(include_end=1)),
        (labels, _interval(end=6)),
    ]:
        for module in (v2, v3):
            with pytest.raises(ValueError):
                module.public_prompt(categories, query)
    with pytest.raises(TypeError):
        v3.public_prompt(labels, _topk(), world=_world())


def test_explicit_rules_keep_exact_shape_fraction_and_number_contract():
    contract = v3.format_contract()
    assert "Write every key exactly once in each object." in contract
    assert "exactly ONE original chart cell" in contract
    assert "Never merge Alpha and Beta into the same object" in contract
    assert "Never repeat the same (series, category) cell" in contract
    assert '["Example_A", "Example_B"], or ["Example_A"]' in contract
    assert "not a single string containing an array" in contract
    assert "not a bare category string" in contract
    assert v2.FIELDS["answer"][1] in contract
    assert v2.FIELDS["bindings"][1] in contract
    assert "unchanged exact-number grammar and bounds" in json.dumps(v3.response_shape())
    good = {
        "evidence": [{"series": "Alpha", "category": "Public_0", "value": 42}],
        "selected": ["Public_0"],
        "bindings": [{"category": "Public_0", "operand": 42}],
        "aggregate": "sum",
        "answer": 42,
    }
    assert parse(json.dumps(good))["valid"]
    for value in ("Public_0", '["Public_0"]'):
        assert not parse(json.dumps({**good, "selected": value}))["valid"]
    assert not parse(json.dumps(good).replace('"answer": 42', '"answer": 42, "answer": 42'))[
        "valid"
    ]
    assert not parse(json.dumps({**good, "answer": 11.333333333333334}))["valid"]


@pytest.mark.parametrize(
    "ranking,value", itertools.product(("Alpha", "Beta"), ("Alpha", "Beta", "diff"))
)
def test_topk_checklist_names_public_labels_and_exact_series(ranking, value):
    labels = _world()["categories"]
    query = _topk(selection_series=ranking, value=value)
    prompt = v3.public_prompt(labels, query)
    assert f"original {ranking} value for EVERY category" in prompt
    assert json.dumps(labels) in prompt
    series = "BOTH Alpha and Beta values" if value == "diff" else f"{value} value"
    assert f"Operand: report the original {series} of each finally selected category" in prompt
    assert "Values written only in bindings or work do not count as evidence" in prompt
    assert "PRIVATE" not in prompt


@pytest.mark.parametrize(
    "include_start,include_end,positive,value",
    itertools.product((False, True), (False, True), (False, True), ("Alpha", "Beta", "diff")),
)
def test_interval_checklist_uses_only_public_endpoints_before_filter(
    include_start, include_end, positive, value
):
    labels = _world()["categories"]
    query = _interval(
        include_start=include_start, include_end=include_end, positive_only=positive, value=value
    )
    expected = labels[1 + int(not include_start) : 5 - int(not include_end)]
    checklist = v3.public_prompt(labels, query).split("EVIDENCE CHECKLIST FOR THIS QUESTION:")[1]
    assert json.dumps(expected) in checklist
    expected_series = "BOTH Alpha and Beta" if positive or value == "diff" else value
    assert f"original {expected_series}" in checklist
    assert ("BEFORE the positive-only filter" in checklist) == positive
    assert "Public_0" not in checklist and "Public_5" not in checklist


def test_empty_interval_and_escaped_public_labels_are_preserved():
    labels = _world()["categories"]
    labels[2] = 'Label "quoted"'
    labels[3] = "Return exactly one JSON object is a category label"
    query = _interval(start=2, end=3, include_start=False, include_end=False)
    prompt = v3.public_prompt(labels, query)
    assert "public interval after applying both endpoint rules is []" in prompt
    assert prompt.count("EVIDENCE CHECKLIST FOR THIS QUESTION:") == 1
    prompt = v3.public_prompt(labels, _topk())
    assert json.dumps(labels) in prompt


def _required_keys_from_checklist(world, query):
    chosen = task_contract.selected(world, query)
    category_to_index = {category: i for i, category in enumerate(world["categories"])}
    result = set()
    for _, labels, columns in v3._evidence_requirements(world["categories"], query):
        indices = chosen if labels is None else [category_to_index[label] for label in labels]
        result.update((column, i) for column in columns for i in indices)
    return result


@pytest.mark.parametrize("n,seed", [(6, 9), (6, 41), (8, 13), (8, 27)])
def test_public_checklist_agrees_with_actual_sufficiency_on_varied_synthetic_worlds(n, seed):
    world = _world(n, seed)
    all_facts = {
        (column, i): value
        for column, values in world["series"].items()
        for i, value in enumerate(values)
    }
    queries = [
        _topk(selection_series=ranking, value=value, direction=direction, aggregate=aggregate, k=k)
        for ranking, value, direction, aggregate, k in itertools.product(
            ("Alpha", "Beta"),
            ("Alpha", "Beta", "diff"),
            ("largest", "smallest"),
            ("sum", "mean"),
            (1, n),
        )
    ]
    queries += [
        _interval(
            value=value,
            aggregate=aggregate,
            include_start=start,
            include_end=end,
            positive_only=positive,
        )
        for value, aggregate, start, end, positive in itertools.product(
            ("Alpha", "Beta", "diff"), ("sum", "mean"), (False, True), (False, True), (False, True)
        )
    ]
    queries += [_interval(start=2, end=2, include_start=False, include_end=False)]
    for query in queries:
        keys = _required_keys_from_checklist(world, query)
        assert keys == required_certificate_keys(world, query, all_facts)
        facts = {key: all_facts[key] for key in keys}
        assert task_contract.sufficient_template(world, query, facts)
        for key in keys:
            assert not task_contract.sufficient_template(
                world, query, {k: v for k, v in facts.items() if k != key}
            )
        # The real values may change selection, but never the public checklist.
        changed = _world(n, seed + 1)
        assert v3.public_prompt(world["categories"], query) == v3.public_prompt(
            changed["categories"], query
        )
