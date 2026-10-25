"""Raw-output audits must detect label corruption without rescoring themselves."""

import copy
import json

import pytest

from sr_f5.calibration import independent_review
from sr_f5.measurement import map_token_mask, score_response
from srf5_ref.generator import FAMILIES, make_root
from srf5_ref.oracle import canonical_evidence, encode, evaluate_a


def scored_row(task, text, finish="balanced"):
    label = score_response(text, task["world"], task["query"], finish_reason=finish)
    ids = list(map(ord, text[1:]))
    mask, reason = map_token_mask(
        ids, text, label["selected_span"], lambda x: "".join(map(chr, x)), "{"
    )
    return dict(
        **label,
        full_text=text,
        tokens=ids,
        mask=mask,
        selected_mask=list(mask),
        mask_ok=reason == "MAPPED",
        token_prefix_character_ends=list(range(2, len(text) + 1)),
        prefix_character_offsets_stable=True,
        assistant_prefill="{",
        finish_reason=finish,
        row_id="b0/review/0",
        qid=task["qid"],
        cell=task["cell"],
        parent="b0",
    )


def valid_output(task):
    gold = evaluate_a(task["world"], task["query"])
    return dict(
        evidence=canonical_evidence(task["world"], task["query"]),
        selected=gold["selected"],
        answer=encode(gold["answer"]),
    )


@pytest.fixture
def example():
    task = make_root("dev", "BINDING", 0, 122)[0]
    return task, scored_row(task, json.dumps(valid_output(task)))


@pytest.mark.parametrize(
    "field",
    [
        "V",
        "A",
        "value_correct",
        "value_parseable",
        "J",
        "J_M3",
        "J_legacy_projection",
        "finish",
        "certificate_truth",
        "certificate_sufficient",
        "certificate",
    ],
)
def test_binary_label_flip_is_rejected(example, field):
    task, row = example
    row[field] = 1 - row[field]
    assert independent_review(row, task)["status"] == "FAIL"


@pytest.mark.parametrize(
    "field,value", [("S", "UNKNOWN"), ("S", "FAIL"), ("P", "UNKNOWN"), ("P", "FAIL")]
)
def test_semantic_label_flip_is_rejected(example, field, value):
    task, row = example
    row[field] = value
    assert independent_review(row, task)["status"] == "FAIL"


def test_wrong_unknown_and_hidden_mask_are_rejected(example):
    task, row = example
    row.update(S="UNKNOWN", mask=[False] * len(row["tokens"]), mask_ok=False)
    review = independent_review(row, task)
    assert "independent_selection_disagreement" in review["errors"]
    assert "native_prefix_mask_disagreement" in review["errors"]


@pytest.mark.parametrize("family", FAMILIES)
def test_original_valid_outputs_pass_all_families_and_variants(family):
    for task in make_root("dev", family, 0, 122):
        row = scored_row(task, json.dumps(valid_output(task)))
        assert independent_review(row, task)["errors"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "wrong_selection",
        "duplicate_selection",
        "bad_fact",
        "duplicate_fact",
        "unknown_category",
        "bad_answer",
        "wrong_answer",
        "length",
        "trailing",
        "malformed",
        "duplicate_key",
        "work_decoy",
    ],
)
def test_adverse_raw_outputs_are_not_automatically_review_failures(example, mutation):
    task, _ = example
    obj = copy.deepcopy(valid_output(task))
    finish = "balanced"
    if mutation == "empty":
        obj["evidence"], obj["selected"] = [], []
    elif mutation == "wrong_selection":
        obj["selected"] = ["absent"]
    elif mutation == "duplicate_selection":
        obj["selected"] *= 2
    elif mutation == "bad_fact":
        obj["evidence"][0]["value"] = 9999
    elif mutation == "duplicate_fact":
        obj["evidence"].append(obj["evidence"][0])
    elif mutation == "unknown_category":
        obj["evidence"][0]["category"] = "absent"
    elif mutation == "bad_answer":
        obj["answer"] = "5 people"
    elif mutation == "wrong_answer":
        obj["answer"] = 999999
    elif mutation == "length":
        finish = "length"
    elif mutation == "work_decoy":
        obj["work"] = 'selected=["absent"] before reconsideration'
    text = json.dumps(obj)
    if mutation == "trailing":
        text += " extra text"
    elif mutation == "malformed":
        text = '{"selected": [}'
    elif mutation == "duplicate_key":
        text = text[:-1] + ', "selected": []}'
    row = scored_row(task, text, finish)
    assert independent_review(row, task)["errors"] == []


def test_unparseable_raw_cannot_hide_wrong_valid_structure(example):
    task, _ = example
    row = scored_row(task, '{"selected": [}')
    row["V"] = 1
    assert "independent_structure_disagreement" in independent_review(row, task)["errors"]


def test_review_does_not_call_production_score(example, monkeypatch):
    import sr_f4.task_contract as legacy
    import sr_f5.measurement as measurement

    def forbidden(*args, **kwargs):
        raise AssertionError("production score must not be its own reviewer")

    task, row = example
    monkeypatch.setattr(legacy, "score", forbidden)
    monkeypatch.setattr(measurement, "score_response", forbidden)
    assert independent_review(row, task)["status"] == "PASS"


@pytest.mark.parametrize("family", FAMILIES)
def test_unparseable_answer_preserves_registered_certificate_diagnostics(family):
    for task in make_root("dev", family, 0, 122):
        obj = valid_output(task)
        obj["answer"] = "5 people"
        row = scored_row(task, json.dumps(obj))
        assert row["J"] == 0
        assert independent_review(row, task)["errors"] == []
