"""Additive diagnostics cannot repair text or change historical wire labels."""

import copy
import itertools
import json
from decimal import Decimal

import pytest

from sr_f6.interface_diagnostics import core_task_projection, diagnose_response
from sr_f6.measurement import _object, score_response
from srf6_ref.measurement import full_truth_response


@pytest.fixture
def task():
    world = {
        "categories": ["Mug", "Lamp", "Desk", "Vase", "Chair", "Clock"],
        "series": {"Alpha": [84, 78, 72, 48, 42, 36], "Beta": [24, 78, 42, 66, 48, 78]},
        "domain": [10, 90],
    }
    query = {
        "family": "TOPK",
        "selection_series": "Alpha",
        "direction": "largest",
        "k": 2,
        "value": "Beta",
        "aggregate": "sum",
    }
    return world, query


@pytest.fixture
def response(task):
    return json.loads(full_truth_response(*task))


def diagnose(obj, task, reason="balanced"):
    return diagnose_response(json.dumps(obj), *task, reason)


def test_wire_score_is_unchanged_and_inputs_are_not_mutated(task, response):
    original = copy.deepcopy((task, response))
    raw = json.dumps(response)
    result = diagnose_response(raw, *task, "balanced")
    assert result["wire_score"] == score_response(raw, *task, finish_reason="balanced")
    assert result["A"] == result["J_wire"] == result["V_aux"] == result["J_core_diag"] == 1
    assert result["S"] == result["B"] == result["E_raw"] == 1
    assert result["gate_status"] == "NOT_SIGNED"
    assert result["training_authorized"] is False
    assert result["errors"] == []
    assert (task, response) == original


def test_core_task_projection_preserves_decimal_and_work_bytes(task, response):
    response["work"] = 'Quoted {"selected":[],"bindings":[]} stays exactly here. 文本'
    raw = json.dumps(response, ensure_ascii=False)
    raw = raw.replace('"answer": "102"', '"answer": 102.' + "0" * 80)
    projected = core_task_projection(" \n" + raw + " \t")
    assert "102." + "0" * 80 in projected
    assert "文本" in projected
    assert isinstance(_object(projected)["answer"], Decimal)
    assert diagnose_response(raw, *task, "balanced")["J_core_diag"] == 1


def test_core_projection_handles_every_top_level_order(response):
    response["work"] = 'A string including "aggregate", not a member.'
    expected = {k: v for k, v in response.items() if k not in {"selected", "bindings", "aggregate"}}
    for keys in itertools.permutations(response):
        raw = json.dumps({k: response[k] for k in keys}, indent=2)
        assert json.loads(core_task_projection(raw)) == expected


def test_real_failure_shapes_do_not_turn_into_production_credit(task, response):
    response["selected"] = [{"category": "Mug", "operand": "Beta"}, {"category": "Lamp"}]
    response["bindings"][0]["operand"] = "Beta"
    response["bindings"][1]["operand"] = "84 - 18"
    result = diagnose(response, task)
    assert result["A"] == result["J_core_diag"] == 1
    assert result["V_aux"] == result["J_wire"] == 0
    assert result["S"] is result["B"] is None
    assert result["wire_score"]["credit_eligible"] is False
    assert result["certificate"]["truth"] == result["certificate"]["sufficient"] == 1
    assert {e["path"] for e in result["errors"]} == {
        "selected[0]",
        "selected[1]",
        "bindings[0].operand",
        "bindings[1].operand",
    }


def test_all_field_errors_survive_first_protocol_error(task, response):
    response["selected"] = 12
    response["bindings"] = [{"category": False, "operand": "Beta", "extra": 0}, {}]
    response["aggregate"] = "median"
    response["evidence"] = [{"series": 1, "category": False, "value": "Alpha"}, 4]
    response["answer"] = "7/0"
    response["work"] = 12
    response["extra"] = 13
    result = diagnose(response, task)
    assert {e["path"].split("[")[0] for e in result["errors"]} == {
        "selected",
        "bindings",
        "aggregate",
        "evidence",
        "answer",
        "work",
        "extra",
    }
    assert len(result["fields"]["bindings"]["errors"]) == 5
    assert len(result["fields"]["evidence"]["errors"]) == 4
    assert result["certificate"]["known"] is False
    assert result["certificate"]["truth"] is None
    assert result["A"] == result["J_wire"] == 0


@pytest.mark.parametrize("reason", ["eos", "length"])
def test_raw_finish_is_not_promoted_to_balanced(task, response, reason):
    result = diagnose(response, task, reason)
    expected = score_response(json.dumps(response), *task, finished=False, finish_reason=reason)
    assert result["wire_score"] == expected
    assert result["generation_finish_reason"] == reason
    assert result["scored_finish_reason"] == "length"
    assert result["A"] == result["V_aux"] == 1
    assert result["finished"] == result["J_wire"] == result["J_core_diag"] == 0
    assert result["S"] is result["B"] is None
    assert result["fields"]["selected"]["syntax"] == "VALID"
    assert result["fields"]["selected"]["known"] is False
    assert result["certificate"]["known"] is True
    assert result["certificate"]["scope"] == "content_only_not_credit_eligibility"


@pytest.mark.parametrize("reason", [None, "COMPLETE", "unknown", "final_tag", ""])
def test_unknown_finish_rejected(task, response, reason):
    with pytest.raises(ValueError, match="raw generation finish reason"):
        diagnose(response, task, reason)


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer":2,"answer":3}',
        '{"answer":84 - 18}',
        '{"answer":66,"work":"bare\nnewline"}',
        '{"answer":66',
        '{"answer":NaN}',
        '```json\n{"answer":66}\n```',
        '{"answer":66} {"answer":67}',
        'prefix {"answer":66}',
        "[]",
    ],
)
def test_ambiguous_invalid_json_is_not_repaired(task, raw):
    result = diagnose_response(raw, *task, "balanced")
    assert result["A"] == result["J_wire"] == result["V_aux"] == 0
    assert result["J_core_diag"] is None
    assert result["core_status"] == "UNAVAILABLE"
    assert all(f["syntax"] == "UNAVAILABLE" for f in result["fields"].values())
    assert result["certificate"]["known"] is False
    with pytest.raises((ValueError, TypeError)):
        core_task_projection(raw)


def test_extra_member_is_preserved_in_projection_and_fails_core(task, response):
    response["hidden_alternative_answer"] = 102
    projected = json.loads(core_task_projection(json.dumps(response)))
    assert projected["hidden_alternative_answer"] == 102
    assert diagnose(response, task)["J_core_diag"] == 0


def test_missing_auxiliary_fields_do_not_prevent_content_diagnosis(task, response):
    del response["selected"], response["bindings"], response["aggregate"]
    result = diagnose(response, task)
    assert result["A"] == result["J_core_diag"] == 1
    assert result["V_aux"] == result["J_wire"] == 0
    assert {e["path"] for e in result["errors"]} == {"selected", "bindings", "aggregate"}
    assert result["fields"]["selected"]["syntax"] == "ABSENT"
    assert result["fields"]["work"]["syntax"] == "ABSENT"
    assert result["fields"]["work"]["errors"] == []


def test_wrong_numbers_and_duplicates_are_known_semantic_failures(task, response):
    response["selected"] = ["Mug", "Mug"]
    response["bindings"][0]["operand"] = -12
    response["aggregate"] = "mean"
    response["answer"] = 17
    result = diagnose(response, task)
    assert result["V_aux"] == 1
    for field in ("selected", "bindings", "aggregate", "answer"):
        assert result["fields"][field]["syntax"] == "VALID"
        assert result["fields"][field]["known"] is True
        assert result["fields"][field]["truth"] == 0


def test_evidence_does_not_get_bindings_merged_or_truth_filled(task, response):
    response["evidence"] = [e for e in response["evidence"] if e["series"] == "Alpha"]
    result = diagnose(response, task)
    assert result["A"] == result["S"] == result["B"] == result["V_aux"] == 1
    assert result["J_wire"] == result["J_core_diag"] == 0
    assert result["certificate"]["truth"] == 1
    assert result["certificate"]["sufficient"] == 0


def test_answer_last_and_work_order_diagnostic_cannot_choose_answer(task, response):
    response["answer"] = 17
    response["work"] = "Correction: the answer should be 102."
    result = diagnose(response, task)
    assert result["A"] == result["J_wire"] == 0
    assert result["answer_last"] is result["work_before_answer"] is False
    work = response.pop("work")
    reordered = {"work": work, **response}
    after = diagnose(reordered, task)
    assert after["A"] == after["J_wire"] == 0
    assert after["work_before_answer"] is after["answer_last"] is True


def test_null_answer_shape_is_not_numeric_observability(task, response):
    response["answer"] = None
    result = diagnose(response, task)
    assert result["V_aux"] == 1
    assert result["fields"]["answer"]["syntax"] == "VALID"
    assert result["fields"]["answer"]["known"] is False
    assert result["fields"]["answer"]["truth"] is None
