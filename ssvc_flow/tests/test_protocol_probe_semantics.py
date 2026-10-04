"""Frozen protocol contracts; no model outputs or GPU claims are made here."""

import copy
import csv
import itertools
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from src.protocol_state_probes.cases import BundleValidationError, load_bundle
from src.protocol_state_probes.ptlc import dpe, external, permutation, run, transform_star
from src.protocol_state_probes.semantics import (
    event_label,
    parse_four_ints,
    relation_values,
    score_raw,
)
from src.protocol_state_probes.transforms import (
    FWD,
    REV,
    emit,
    predict,
    public_from_record,
    render,
    restore,
)

DESIGN = Path(__file__).resolve().parents[1] / "docs/protocol_state_probes/design"


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(DESIGN)


@pytest.fixture(scope="module")
def records():
    return [
        json.loads(line)
        for line in (DESIGN / "sources/selected_original_records.jsonl").read_text().splitlines()
    ]


def lookup(bundle, predicate):
    case = next(c for c in bundle["cases"] if predicate(c))
    audit = next(a for a in bundle["audits"] if a["case_id"] == case["case_id"])
    prediction = next((p for p in bundle["predictions"] if p["case_id"] == case["case_id"]), None)
    return case, audit, prediction


def test_all_frozen_manifests_recompiled_without_gpu(bundle):
    verification = bundle["verification"]
    assert verification["all_fields_match"] is True
    assert verification["source_snapshot_only"] is True
    assert verification["recompiled_from_prepared"] is False
    assert verification["counts"] == {
        "total_cases": 509,
        "aliases": 35,
        "unique_case_prompts": 474,
        "job_cells": 1944,
        "scheduled_completions_after_exact_prompt_alias_reuse": 70656,
    }
    assert verification["program_predictions_verified"] == 461


def test_prepared_roundtrip_and_changed_input_is_rejected(records, tmp_path):
    records = copy.deepcopy(records)
    prepared = {
        "panels": {
            "D": [r for r in records if r.get("panel") == "D"],
            "P": [r for r in records if r.get("panel") == "P"],
        },
        "continuation_prompts": [r for r in records if r.get("panel") is None],
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(prepared))
    assert load_bundle(DESIGN, path)["verification"]["recompiled_from_prepared"] is True
    prepared["panels"]["D"][0]["prompt"]["user"] += " Unexpected addition."
    path.write_text(json.dumps(prepared))
    with pytest.raises(BundleValidationError) as error:
        load_bundle(DESIGN, path)
    assert any("prompt.user" in diff["path"] for diff in error.value.differences)


def test_prepared_missing_record_is_reported(records, tmp_path):
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps({"panels": {"D": [], "P": []}, "continuation_prompts": []}))
    with pytest.raises(BundleValidationError) as error:
        load_bundle(DESIGN, path)
    assert len(error.value.differences) == 509


def test_all_permutation_roundtrips():
    for order in itertools.permutations(FWD):
        assert restore(emit([5, 10, 20, 40], order), order) == [5, 10, 20, 40]


@pytest.mark.parametrize("order", [[0, 1, 1, 3], [0, 1, 2], [0, 1, 2, True], [0, 1, 2, 3.0]])
def test_invalid_permutation_rejected(order):
    with pytest.raises(ValueError):
        permutation(order)


def test_reverse_difference_uses_canonical_world_only():
    truth = [5, 10, 20, 40]
    assert event_label("[40,20,11,4]", REV, truth, "difference_pairs")[0] == "S"
    assert event_label("[40,20,11,4]", FWD, truth, "difference_pairs")[0] == "W"
    assert event_label("[40,20,10,5]", REV, truth, "difference_pairs")[0] == "X"
    assert event_label("[5,10,20,40]", REV, truth, "difference_pairs")[0] == "W"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        123,
        "text [1,2,3,4]",
        "```json\n[1,2,3,4]\n```",
        "[1,2,3,4]\n[1,2,3,4]",
        "[1,2,3]",
        '{"a":1}',
        "[1.0,2,3,4]",
        "[true,2,3,4]",
        "[NaN,2,3,4]",
        "[1e0,2,3,4]",
    ],
)
def test_parser_accepts_only_one_complete_integer_array(raw):
    assert parse_four_ints(raw) is None


def test_parser_whitespace_and_domain_are_separate():
    assert parse_four_ints(" \n [-1,2,3,100] \t") == [-1, 2, 3, 100]
    assert event_label("[-1,2,3,100]", FWD, [1, 2, 3, 4], "sum4") == ("I", [-1, 2, 3, 100])


def test_main_renderer_rejects_privileged_fields_and_ignores_record_audit(records):
    record = copy.deepcopy(records[0])
    public = public_from_record(record)
    for key in ("truth", "j", "DPE", "historical_X", "outputs", "solver_output"):
        with pytest.raises(ValueError):
            render({**public, key: 1}, "O0")
    record["scene"]["truth_world"] = [99, 99, 99, 99]
    record["scene"]["changed_index"] = 3
    assert public_from_record(record) == public


def test_input_and_output_factors_are_independent(records):
    public = public_from_record(next(r for r in records if r["family"] == "trend"))
    texts = {p: render(public, p) for p in ("O0", "A1", "L00", "L01", "L10", "L11")}
    assert texts["A1"]["user"] == texts["O0"]["user"].replace(
        "Return [a,b,c,d]", "Return [d,c,b,a]"
    )
    for a, b in (("L00", "L01"), ("L10", "L11")):
        assert texts[b]["user"] == texts[a]["user"].replace("Return [a,b,c,d]", "Return [d,c,b,a]")
    assert texts["L10"]["output_order"] == list(FWD)
    assert texts["L11"]["output_order"] == list(REV)
    for p in ("L00", "L01", "L10", "L11"):
        for k, name in enumerate("abcd"):
            assert f"{name}={public['observed'][k]}" in texts[p]["user"]


def test_b1_exact_row_transform_and_literal_subtraction(records):
    for record in records:
        if record["family"] != "cross_series":
            continue
        public = public_from_record(record)
        H, b = public["H"], public["b"]
        HH, bb, meta = transform_star(H, b)
        T = meta["row_transform"]
        assert [[sum(t * H[i][j] for i, t in enumerate(tr)) for j in FWD] for tr in T] == HH
        assert [sum(t * b[i] for i, t in enumerate(tr)) for tr in T] == bb
        for y in (public["observed"], record["scene"]["truth_world"], [-1, 101, 0, 12], [0] * 4):
            assert all(relation_values(H, b, y)) == all(relation_values(HH, bb, y))
        equations = (
            render(public, "B1")["user"]
            .split("The following relationships are reliable:\n")[1]
            .split("\nThe downstream")[0]
            .splitlines()
        )
        assert all(" - " in row and not row.startswith("-") for row in equations[1:])
        for variant in ("V1", "V2"):
            assert (
                predict(public, "BR")[variant]["canonical"]
                == predict(public, "O0")[variant]["canonical"]
            )


def test_b1_partial_relation_score_is_not_invariant(bundle):
    differences = []
    for case in bundle["cases"]:
        if case["protocol"] != "B1" or case["kind"] != "repair":
            continue
        _, audit, prediction = lookup(bundle, lambda c, case=case: c["case_id"] == case["case_id"])
        features = score_raw(json.dumps(case["observed_world_public"]), case, audit, prediction)
        if features["C_orig"] != features["C_display"]:
            differences.append(case["case_id"])
    assert differences


def test_literal_ptlc_majority_conflict_domain_and_fraction():
    H = [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]
    y, trace = run(H, [10, 11, 12], [99, 2, 3, 0], (1, 2, 3, 0))
    assert y[0] == 8 and trace[-1]["reason"] == "STRICT_MAJORITY"
    y, trace = run(H, [10, 20, 30], [7, 2, 3, 4], (1, 2, 3, 0))
    assert y[0] == 7 and trace[-1]["reason"] == "CONFLICT_FALLBACK_COPY"
    H = [[1, -2, 1, 0], [0, 1, -2, 1]]
    assert external(run(H, [0, 0], [70, 4, 10, 20], variant="V1")[0]) == [70, 4, -62, -128]
    assert external(run(H, [0, 0], [70, 4, 10, 20], variant="V2")[0]) == [70, 4, 10, 16]
    y, _ = run([[2, 1, 0, 0]], [1], [3, 0, 4, 5], (1, 0, 2, 3))
    assert y[0] == Fraction(1, 2)
    assert external(y)[0] == "1/2"


def test_dpe_theorem_for_all_registered_metadata_and_permutations():
    rows = list(csv.DictReader((DESIGN / "sources/task_DPE_registry.csv").read_text().splitlines()))
    count = 0
    for row in rows:
        if row["interface"] != "SYMBOLIC_FRESH":
            continue
        H, b, obs, truth = (json.loads(row[k]) for k in ("H", "b", "observed", "truth"))
        for order in itertools.permutations(FWD):
            for variant in ("V1", "V2"):
                y, _ = run(H, b, obs, order, variant=variant)
                assert (y == list(map(Fraction, truth))) == dpe(H, int(row["j0"]), order)
                count += 1
    assert count == 19008


def test_all_truth_outputs_restore_and_score(bundle):
    audits = {a["case_id"]: a for a in bundle["audits"]}
    predictions = {p["case_id"]: p for p in bundle["predictions"]}
    for case in bundle["cases"]:
        audit = audits[case["case_id"]]
        expected = audit["control_expected_canonical"] or audit["truth_world"]
        raw = json.dumps(emit(expected, case["output_order"]))
        features = score_raw(raw, case, audit, predictions.get(case["case_id"]))
        if case["panel"].startswith("CONTROL_"):
            assert features["is_control"] is True
            assert features["control_success"] is True
        if case["kind"] == "repair":
            assert features["event"] == "X"
        else:
            assert features["event"] is None
            assert features["X"] is None
            assert features["F"] is None


def test_illegal_outputs_keep_diagnostics_and_null_audits(bundle):
    predictions = {p["case_id"]: p for p in bundle["predictions"]}
    case, audit, prediction = lookup(
        bundle,
        lambda c: (
            c["kind"] == "repair" and not predictions[c["case_id"]]["programs"]["V1"]["in_domain"]
        ),
    )
    features = score_raw(
        json.dumps(prediction["programs"]["V1"]["emitted"]), case, audit, prediction
    )
    assert features["event"] == "I"
    assert features["match_V1"] is True
    assert features["C_orig"] == features["C_display"] == 0
    assert features["C_alg_orig"] is not None
    assert features["F"] is features["B"] is features["M"] is None
    bad = score_raw("not an array", case, audit, prediction)
    assert bad["event"] == "I"
    assert bad["canonical_world"] is bad["match_mask"] is bad["anchor_mask"] is None
    assert bad["relation_flags_orig"] is bad["relation_flags_display"] is None
    assert bad["anchor_all"] is None and bad["anchor_all_parsed"] is False
    assert bad["C_orig"] == 0 and bad["C_alg_orig"] is None


def test_aliases_share_exact_evidence_and_joint_mask_does_not_double_count(bundle):
    cases = {c["case_id"]: c for c in bundle["cases"]}
    for alias in bundle["aliases"]:
        a, b = cases[alias["case_id"]], cases[alias["owner_case_id"]]
        assert a["prompt"] == b["prompt"] and a["output_order"] == b["output_order"]
    case, audit, prediction = lookup(bundle, lambda c: c["panel"] == "D48")
    features = score_raw(
        json.dumps(emit(case["observed_world_public"], case["output_order"])),
        case,
        audit,
        prediction,
    )
    assert features["match_mask"] % 2 == 1
    assert 0 <= features["match_mask"] <= 7


def test_registered_transition_and_b1_risk_counts(bundle):
    audits = {a["case_id"]: a for a in bundle["audits"]}
    for panel, n in (("D48", 6), ("U22", 2)):
        assert (
            sum(
                not audits[c["case_id"]]["DPE1_old"] and audits[c["case_id"]]["DPE1_new"]
                for c in bundle["cases"]
                if c["panel"] == panel and c["protocol"] == "B1"
            )
            == n
        )
    transitions = Counter(
        (audits[c["case_id"]]["DPE1_old"], audits[c["case_id"]]["DPE1_new"])
        for c in bundle["cases"]
        if c["panel"] == "D48" and c["protocol"] == "A1"
    )
    assert transitions == {(False, True): 28, (True, False): 19, (True, True): 1}
    counts = {"V1": 0, "V2": 0}
    for p in bundle["predictions"]:
        if (
            not p["case_id"].startswith("D48:")
            or not p["case_id"].endswith(":B1")
            or "b1_diagnostic" not in p
        ):
            continue
        diagnostic = p["b1_diagnostic"]
        if diagnostic["first_leaf_risk"]:
            for variant in counts:
                counts[variant] += diagnostic["changed"][variant]
    assert counts == {"V1": 2, "V2": 3}


def test_b1_old_and_new_vectors_measured_in_both_protocols(bundle):
    {c["case_id"]: c for c in bundle["cases"]}
    for prediction in bundle["predictions"]:
        if not prediction["case_id"].endswith(":B1"):
            continue
        _case, _audit, _ = lookup(
            bundle, lambda c, prediction=prediction: c["case_id"] == prediction["case_id"]
        )
        for protocol in ("B0", "B1"):
            other_id = prediction["case_id"].rsplit(":", 1)[0] + ":" + protocol
            other_case, other_audit, other_prediction = lookup(
                bundle, lambda c, other_id=other_id: c["case_id"] == other_id
            )
            for variant in ("V1", "V2"):
                for label in ("old", "new"):
                    vector = prediction["b1_diagnostic"][label][variant]
                    features = score_raw(
                        json.dumps(emit(vector["canonical"], other_case["output_order"])),
                        other_case,
                        other_audit,
                        other_prediction,
                    )
                    assert features[f"match_{variant}_{label}"] == vector["integer_prediction"]
                    if not prediction["b1_diagnostic"]["changed"][variant]:
                        assert features[f"match_{variant}_old"] == features[f"match_{variant}_new"]


def test_execution_control_failure_is_retained(bundle):
    case, audit, prediction = lookup(
        bundle, lambda c: c["panel"] == "CONTROL_COPY" and c["protocol"] == "A1"
    )
    features = score_raw(json.dumps(audit["control_expected_canonical"]), case, audit, prediction)
    assert features["parse_four_ints"] is True
    assert features["control_default_forward_output"] is True
    assert features["control_success"] is False
    assert features["event"] is None
