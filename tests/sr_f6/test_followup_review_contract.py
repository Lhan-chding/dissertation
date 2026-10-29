"""Synthetic boundary checks, never real review judgments or scientific PASS.

All texts/worlds below are hand-authored fixtures. The frozen production entry
is exercised only to validate the published method, never to review real rows.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sr_f6 import census
from sr_f6_followup import review_contract


@pytest.fixture
def task():
    return {
        "qid": "SYNTHETIC_ONLY_REVIEW_CONTRACT",
        "world": {
            "categories": ["A", "B", "C", "D", "E", "F"],
            "series": {"Alpha": [70, 70, 60, 40, 20, 10], "Beta": [20, 40, 30, 50, 80, 90]},
            "domain": [10, 90],
        },
        "query": {
            "family": "TOPK",
            "selection_series": "Alpha",
            "direction": "largest",
            "k": 2,
            "value": "Beta",
            "aggregate": "mean",
        },
    }


@pytest.fixture
def answer():
    return {
        "evidence": [
            {"series": "Alpha", "category": category, "value": value}
            for category, value in zip("ABCDEF", [70, 70, 60, 40, 20, 10], strict=True)
        ]
        + [
            {"series": "Beta", "category": "A", "value": 20},
            {"series": "Beta", "category": "B", "value": 40},
        ],
        "selected": ["A", "B"],
        "bindings": [{"category": "A", "operand": 20}, {"category": "B", "operand": 40}],
        "aggregate": "mean",
        "answer": 30,
    }


def scored(task, answer, stop="balanced"):
    """Use the exact frozen G8 entrypoint, including its raw-stop mapping."""
    text = answer if isinstance(answer, str) else json.dumps(answer)
    model_input = {"fixture": "SYNTHETIC_ONLY"}
    spec = {
        "phase": "SCHEMA_CONFIRM_V3",
        "parent": "b0",
        "factor": "P0",
        "qid": task["qid"],
        "root_id": "SYNTHETIC_ONLY",
        "family": "BINDING",
        "group_id": "SYNTHETIC_ONLY_GROUP",
        "policy_hash": "a" * 64,
        "model_input_sha256": census.object_sha256(model_input),
        "frozen_spec_sha256": "b" * 64,
        "seeds": list(range(8)),
    }
    raw = [
        {
            "seed": seed,
            "generation_status": "COMPLETE",
            "technical_validation_errors": [],
            "tokens": [1, 2],
            "sampler_logprobs": [-1.0, -2.0],
            "model_input": model_input,
            "adapter_identity": {"parent": "b0", "step": 0},
            "parent_content_hash": spec["policy_hash"],
            "policy_tensor_hash": "c" * 64,
            "full_text": text,
            "finish_reason": stop,
        }
        for seed in spec["seeds"]
    ]
    result = census.annotate_group(raw, task, spec)
    assert all(row["score"] == result[0]["score"] for row in result)
    return result[0]["score"]


def labels(result):
    return tuple(result[key] for key in ("A", "J", "V", "finish", "S", "B"))


def test_payload_identity_is_deterministic_fresh_and_nonrecursive():
    files = review_contract.payloads()
    spec = review_contract.specification()
    assert set(files) == {"REVIEW_METHOD.md", "ADJUDICATION_METHOD.md", "CONTRACT.json"}
    assert spec["payload_sha256"] == {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }
    contract = json.loads(files["CONTRACT.json"])
    assert "payload_sha256" not in contract
    assert contract["method_sha256"] == {
        name: spec["payload_sha256"][name]
        for name in ("REVIEW_METHOD.md", "ADJUDICATION_METHOD.md")
    }
    spec["numbers"]["fields"].clear()
    files.clear()
    assert len(review_contract.specification()["numbers"]["fields"]) == 3
    assert len(review_contract.payloads()) == 3
    json.dumps(review_contract.specification(), allow_nan=False)


def test_behavior_test_sources_are_exact_frozen_producer_bytes():
    # The frozen producer and new consumer can have separate source deployments.
    # Resolve imported production code, never any original-answer/run directory.
    root = Path(census.__file__).resolve().parents[2]
    for pin in review_contract.specification()["source_anchors"]:
        assert hashlib.sha256((root / pin["path"]).read_bytes()).hexdigest() == pin["sha256"]


@pytest.mark.parametrize("stop", ["balanced", "eos", "length"])
def test_entrypoint_stop_is_balanced_only_and_preserves_answer_and_V(task, answer, stop):
    result = scored(task, answer, stop)
    expected = (1, 1, 1, 1, 1, 1) if stop == "balanced" else (1, 0, 1, 0, None, None)
    assert labels(result) == expected
    assert review_contract.specification()["completion"]["raw_eos_is_scored_complete"] is False


@pytest.mark.parametrize(
    "text", ['{"answer":30,"answer":30}', '{"answer":30,"x":{"a":1,"a":1}}', "[]"]
)
def test_whole_object_ambiguity_cannot_salvage_answer_or_semantics(task, text):
    assert labels(scored(task, text)) == (0, 0, 0, 1, None, None)


@pytest.mark.parametrize("suffix", [" prose", " {}", '\n{"answer":30}'])
def test_whole_text_not_a_complete_prefix(task, answer, suffix):
    assert labels(scored(task, json.dumps(answer) + suffix)) == (0, 0, 0, 1, None, None)


@pytest.mark.parametrize("value", [None, True, "Alpha - Beta", 10**12 + 1, "1/1000000001"])
def test_unreadable_operand_is_local_unknown_not_semantic_failure(task, answer, value):
    answer["bindings"][0]["operand"] = value
    assert labels(scored(task, answer)) == (1, 0, 0, 1, 1, None)


@pytest.mark.parametrize("field", ["evidence", "bindings", "answer"])
@pytest.mark.parametrize("value", [10**12 + 1, "1/1000000001", "1e003"])
def test_numeric_bounds_and_string_grammar_cover_every_exact_field(task, answer, field, value):
    if field == "answer":
        answer[field] = value
    else:
        answer[field][0]["operand" if field == "bindings" else "value"] = value
    result = scored(task, answer)
    assert result["V"] == result["J"] == 0
    assert result["S"] == 1
    assert result["A"] == int(field != "answer")
    assert result["B"] == (None if field == "bindings" else 1)


@pytest.mark.parametrize(
    "value", [10**12, "1/1000000000", "2/2000000000", "1000000000001/1000000000001"]
)
def test_bounds_are_inclusive_and_apply_after_reduction(task, answer, value):
    answer["bindings"][0]["operand"] = value
    result = scored(task, answer)
    assert (result["V"], result["B"], result["J"]) == (1, 0, 1)


def test_decimal_literal_and_quoted_number_have_distinct_exponent_rules(task, answer):
    answer["bindings"][0]["operand"] = "0e-25"
    quoted = scored(task, answer)
    literal = scored(task, json.dumps(answer).replace('"0e-25"', "0e-25"))
    assert (quoted["V"], quoted["B"]) == (1, 0)
    assert (literal["V"], literal["B"]) == (0, None)
    answer["answer"] = "30"
    text = json.dumps(answer).replace('"answer": "30"', '"answer": 30.' + "0" * 80)
    assert scored(task, text)["A"] == 1


@pytest.mark.parametrize("field", ["selected", "bindings"])
def test_empty_or_duplicate_readable_semantics_do_not_redefine_V_or_J(task, answer, field):
    original = copy.deepcopy(answer[field])
    for value in ([], [*original, original[0]]):
        answer[field] = value
        result = scored(task, answer)
        assert (result["A"], result["V"], result["J"]) == (1, 1, 1)
        assert result["S" if field == "selected" else "B"] == 0
        assert result["B" if field == "selected" else "S"] == 1


def test_selection_spellings_are_exact_and_B_uses_true_not_submitted_set(task, answer):
    answer["selected"] = ["a", " B "]
    assert labels(scored(task, answer)) == (1, 1, 1, 1, 0, 1)
    answer["selected"] = None
    assert labels(scored(task, answer)) == (1, 0, 0, 1, None, 1)
    answer["selected"] = ["B", "A"]
    assert labels(scored(task, answer)) == (1, 1, 1, 1, 1, 1)


def test_A_independence_null_syntax_and_aggregate_truth(task, answer):
    assert labels(scored(task, {"answer": 30})) == (1, 0, 0, 1, None, None)
    answer["aggregate"] = "sum"
    assert labels(scored(task, answer)) == (1, 1, 1, 1, 1, 1)
    answer["answer"] = None
    assert labels(scored(task, answer)) == (0, 0, 1, 1, 1, 1)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "unknown", "domain", "missing_rank", "missing_operand"]
)
def test_native_evidence_constraints_are_distinct_from_F6_V(task, answer, mutation):
    if mutation == "duplicate":
        answer["evidence"].append(copy.deepcopy(answer["evidence"][0]))
    elif mutation == "unknown":
        answer["evidence"][0]["category"] = "unknown"
    elif mutation == "domain":
        answer["evidence"][0]["value"] = 91
    elif mutation == "missing_rank":
        answer["evidence"].pop(5)
    else:
        answer["evidence"].pop()
    assert labels(scored(task, answer)) == (1, 0, 1, 1, 1, 1)


def test_true_extra_evidence_allowed_false_extra_evidence_rejected(task, answer):
    answer["evidence"].append({"series": "Beta", "category": "F", "value": 90})
    assert scored(task, answer)["J"] == 1
    answer["evidence"][-1]["value"] = 89
    assert labels(scored(task, answer)) == (1, 0, 1, 1, 1, 1)


def test_legacy_presentation_work_limit_is_not_an_extra_frozen_J_gate(task, answer):
    answer["work"] = "x" * 16_001
    result = scored(task, answer)
    assert result["presentation"] == 0
    assert labels(result) == (1, 1, 1, 1, 1, 1)
    answer["work"] = "x" * 64_001
    assert labels(scored(task, answer)) == (1, 0, 1, 1, 1, 1)


def test_native_numeric_string_raw_length_can_fail_J_while_F6_A_V_hold(task, answer):
    answer["answer"] = " " * 64 + "30"
    assert labels(scored(task, answer)) == (1, 0, 1, 1, 1, 1)


def test_topk_tie_uses_printed_order_and_signed_operand(task, answer):
    task["query"].update(k=1, value="diff")
    answer.update(selected=["A"], bindings=[{"category": "A", "operand": 50}], answer=50)
    assert labels(scored(task, answer)) == (1, 1, 1, 1, 1, 1)
    task["query"].update(selection_series="Beta", direction="largest")
    answer.update(selected=["F"], bindings=[{"category": "F", "operand": -80}], answer=-80)
    answer["evidence"] = [
        {"series": series, "category": category, "value": value}
        for series, values in task["world"]["series"].items()
        for category, value in zip(task["world"]["categories"], values, strict=True)
    ]
    assert labels(scored(task, answer)) == (1, 1, 1, 1, 1, 1)
    answer["bindings"][0]["operand"] = 80
    assert labels(scored(task, answer)) == (1, 1, 1, 1, 1, 0)


def test_interval_endpoints_and_prefilter_evidence(task, answer):
    task["query"] = {
        "family": "INTERVAL",
        "start": 0,
        "end": 4,
        "include_start": False,
        "include_end": False,
        "positive_only": True,
        "value": "diff",
        "aggregate": "sum",
    }
    answer.update(
        selected=["B", "C"],
        bindings=[{"category": "B", "operand": 30}, {"category": "C", "operand": 30}],
        aggregate="sum",
        answer=60,
    )
    answer["evidence"] = [
        {"series": series, "category": category, "value": task["world"]["series"][series][index]}
        for series in ("Alpha", "Beta")
        for index, category in ((1, "B"), (2, "C"), (3, "D"))
    ]
    assert labels(scored(task, answer)) == (1, 1, 1, 1, 1, 1)
    answer["evidence"] = [r for r in answer["evidence"] if r["category"] != "D"]
    assert labels(scored(task, answer)) == (1, 0, 1, 1, 1, 1)


def test_staged_contract_requires_full_PROGRAM_claims_without_implied_agreement():
    policy = review_contract.specification()
    assert (
        policy["exposure"]["program_claims"]
        == "adjudication only, after all 360 judgments are sealed"
    )
    assert policy["adjudication"]["program_labels_required_for_every_finding_row"] == [
        "A",
        "J",
        "V",
        "finish",
        "S",
        "B",
    ]
    assert policy["adjudication"]["program_oracle_required"] == ["answer", "selected", "bindings"]
    assert policy["adjudication"]["no_inferred_program_claims"] is True
    assert policy["adjudication"]["agreement_threshold"] is None
    method = review_contract.payloads()["ADJUDICATION_METHOD.md"].decode()
    assert "REVIEW_AMBIGUITY" in method
    assert "REVIEW_CONTRACT_VIOLATION" in method
    assert "do not issue PRODUCTION_CORRECT" in method
