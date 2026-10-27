"""Synthetic CPU evidence fixtures: these tests do not certify a model census."""

from __future__ import annotations

import copy
import json
from collections import Counter

import pytest

from sr_f6 import census_support as support
from srf6_ref.census import summarize
from srf6_ref.core import FAMILIES


@pytest.fixture
def evidence():
    rows = []
    for parent in support.PARENTS:
        for factor in support.FACTORS:
            for family_index, family in enumerate(FAMILIES):
                for root_index in range(8):
                    root = f"DEV_CAL_{family}_{root_index:04d}"
                    for view in range(4):
                        qid = f"{root}_{factor}_v{view}"
                        active = (
                            parent in ("b0", "b1")
                            and family_index < 3
                            and root_index < 4
                            and view == 0
                        )
                        for draw in range(8):
                            rows.append(
                                dict(
                                    parent=parent,
                                    factor=factor,
                                    group_id=f"{parent}/{qid}",
                                    qid=qid,
                                    root_id=root,
                                    family=family,
                                    draw=draw,
                                    policy_hash="b" * 64,
                                    A=0,
                                    J=0,
                                    V=1,
                                    finished=1,
                                    S=int(active and draw == 7),
                                    B=0,
                                )
                            )
    summary = summarize(rows)
    summary.update(
        status="COMPLETE_NOT_GATE_SIGNED",
        source_rows=9216,
        frozen_spec_sha256="a" * 64,
        optimizer_updates=0,
        real_measurement_review="NOT_RUN",
    )
    audit = {
        "status": "PASS",
        "model_calls": 0,
        "independent_native_oracles": 2,
        "cross_split_world_duplicates": 0,
        "splits": {
            name: dict(
                roots=roots,
                questions=questions,
                factors=["P0", "P1"],
                family_questions={f: per_family for f in FAMILIES},
            )
            for name, roots, questions, per_family in (
                ("DEV_CAL", 48, 384, 64),
                ("TRAIN", 288, 2304, 384),
            )
        },
        "training_manipulation": dict(
            status="PASS",
            roots=288,
            mean_gap=0.1,
            strict_gain_fraction=0.5,
            model_outputs_used=False,
        ),
    }
    return summary, audit


def refresh_aggregates(summary):
    """Maintain a synthetic summary after deliberately changing fixture facts."""
    for factor in support.FACTORS:
        groups = [g for g in summary["groups"] if g["factor"] == factor]
        nz = [g for g in groups if g["nonzero"]]
        summary["factors"][factor] = dict(
            groups=len(groups),
            nonzero_groups=len(nz),
            roots=len({g["root_id"] for g in nz}),
            families=len({g["family"] for g in nz}),
            parents=len({g["parent"] for g in nz}),
            exchangeable_groups=sum(g["exchangeable"] for g in groups),
            actually_changed_groups=sum(g["changed"] for g in groups),
            nonzero_by_J_class=dict(Counter(g["J_class"] for g in nz)),
        )
    changed = [g for g in summary["groups"] if g["changed"]]
    summary["total_changed_groups"] = len(changed)
    summary["total_changed_roots"] = len({g["root_id"] for g in changed})


def deactivate(group):
    group.update(nonzero=False, exchangeable=False, changed=False, credit_L1=0)


def failure_names(report):
    return {item["name"] for item in report["failures"]}


def test_complete_fixture_support_is_not_stage_authorization(evidence):
    before = copy.deepcopy(evidence)
    report = support.assess_support(*evidence)
    assert report["status"] == "PASS"
    assert report["scope"] == "registered_feasibility_only"
    assert report["gate_status"] == "NOT_SIGNED"
    assert report["training_authorized"] is False
    assert report["model_calls"] == report["optimizer_updates"] == 0
    assert "NOT_VERIFIED_BY_THIS_CHECK" in report["pending_verification"].values()
    assert len(report["checks"]) == 32
    assert report["observed"]["factors"]["P0"]["nonzero_groups"] == 24
    assert report["observed"]["factors"]["P0"]["roots"] == 12
    assert evidence == before


@pytest.mark.parametrize("label", ["V", "S_known", "B_known"])
@pytest.mark.parametrize(
    "count,expected", [(1383, "PASS"), (1382, "INSUFFICIENT_REGISTERED_EXPOSURE")]
)
def test_90_percent_uses_all_1536_including_unknown(evidence, label, count, expected):
    summary, audit = evidence
    cell = summary["cells"]["b2/P0"]  # Inactive fixture cell has all S=B=0.
    cell[f"{label}_rate"] = count / 1536
    if label != "V":
        component = label[0]
        cell["A_J_S_B_cross"] = {
            "A0/J0/S0/B0": count,
            "A0/J0/SNone/B0" if component == "S" else "A0/J0/S0/BNone": 1536 - count,
        }
    report = support.assess_support(summary, audit)
    assert report["status"] == expected
    observed = report["observed"]["cells"]["b2/P0"]
    assert observed["n"] == 1536
    assert observed[label] == count
    if label != "V":
        assert observed[label[0] + "_unknown"] == 1536 - count


def test_three_observability_deficits_are_all_reported(evidence):
    summary, audit = evidence
    cell = summary["cells"]["b2/P0"]
    cell.update(V_rate=0, S_known_rate=0, B_known_rate=0, A_J_S_B_cross={"A0/J0/SNone/BNone": 1536})
    report = support.assess_support(summary, audit)
    assert report["status"] == "INSUFFICIENT_REGISTERED_EXPOSURE"
    assert failure_names(report) == {f"b2/P0.{key}_count" for key in ("V", "S_known", "B_known")}


@pytest.mark.parametrize("deficit", ["nonzero_groups", "roots", "families", "parents"])
def test_each_factor_cond_minimum_is_independent(evidence, deficit):
    summary, audit = evidence
    active = [g for g in summary["groups"] if g["factor"] == "P0" and g["nonzero"]]
    if deficit == "nonzero_groups":
        deactivate(active[0])
    else:
        value = {"roots": active[0]["root_id"], "families": FAMILIES[0], "parents": "b0"}[deficit]
        key = {"roots": "root_id", "families": "family", "parents": "parent"}[deficit]
        for group in active:
            if group[key] == value:
                deactivate(group)
    refresh_aggregates(summary)
    report = support.assess_support(summary, audit)
    assert report["status"] == "INSUFFICIENT_REGISTERED_EXPOSURE"
    assert f"P0.{deficit}" in failure_names(report)
    assert not any(name.startswith("P1.") for name in failure_names(report))


@pytest.mark.parametrize(
    "changed_roots,changed_groups,expected",
    [
        (6, 12, "PASS"),
        (6, 11, "INSUFFICIENT_REGISTERED_EXPOSURE"),
        (5, 12, "INSUFFICIENT_REGISTERED_EXPOSURE"),
        (0, 0, "INSUFFICIENT_REGISTERED_EXPOSURE"),
    ],
)
def test_shuf_counts_actual_change_not_exchangeable_or_factor_duplicates(
    evidence, changed_roots, changed_groups, expected
):
    summary, audit = evidence
    active = [g for g in summary["groups"] if g["nonzero"]]
    roots = list(dict.fromkeys(g["root_id"] for g in active))[:changed_roots]
    # Interleave roots to cover all requested roots before adding other parents/factors.
    eligible = [g for root in roots for g in active if g["root_id"] == root]
    chosen = []
    for offset in range(4):
        chosen.extend(eligible[root * 4 + offset] for root in range(len(roots)))
    changed_ids = {id(g) for g in chosen[:changed_groups]}
    for group in summary["groups"]:
        group["changed"] = id(group) in changed_ids
    refresh_aggregates(summary)
    report = support.assess_support(summary, audit)
    assert report["status"] == expected
    assert report["observed"]["total_changed_groups"] == changed_groups
    assert report["observed"]["total_changed_roots"] == changed_roots
    assert report["observed"]["factors"]["P0"]["exchangeable_groups"] == 24


@pytest.mark.parametrize(
    "mean,gains,expected",
    [
        (0.05, 96, "PASS"),
        (0.049999, 96, "INSUFFICIENT_REGISTERED_EXPOSURE"),
        (0.05, 95, "INSUFFICIENT_REGISTERED_EXPOSURE"),
    ],
)
def test_train_coverage_thresholds_are_inclusive_and_root_based(evidence, mean, gains, expected):
    summary, audit = evidence
    audit["training_manipulation"].update(
        mean_gap=mean, strict_gain_fraction=gains / 288, status=expected
    )
    report = support.assess_support(summary, audit)
    assert report["status"] == expected
    assert report["observed"]["training_manipulation"]["strict_gain_roots"] == gains


@pytest.mark.parametrize(
    "kind", ["factor_nonzero", "factor_roots", "factor_changed", "total_changed", "total_roots"]
)
def test_forged_aggregate_does_not_override_complete_groups(evidence, kind):
    summary, audit = evidence
    if kind.startswith("factor_"):
        name = {
            "factor_nonzero": "nonzero_groups",
            "factor_roots": "roots",
            "factor_changed": "actually_changed_groups",
        }[kind]
        summary["factors"]["P0"][name] += 1
    else:
        summary["total_changed_groups" if kind == "total_changed" else "total_changed_roots"] += 1
    assert support.assess_support(summary, audit)["status"] == "INVALID_SUPPORT_EVIDENCE"


@pytest.mark.parametrize(
    "kind",
    [
        "missing_group",
        "duplicate_group",
        "missing_cell",
        "cell_denominator",
        "total_denominator",
        "missing_unknown",
        "unknown_as_false",
        "nonfinite",
        "boolean_rate",
        "fractional_count",
        "changed_without_exchangeable",
        "missing_group_fact",
        "missing_L1",
        "mixed_root_family",
        "partial_coverage",
        "fractional_root_count",
        "model_conditioned",
        "missing_frozen_identity",
        "mismatched_J_classes",
        "integer_boolean",
        "forged_coverage_status",
        "incomplete_family_matrix",
    ],
)
def test_invalid_evidence_is_rejected_without_authorization(evidence, kind):
    summary, audit = evidence
    cell = summary["cells"]["b2/P0"]
    if kind == "missing_group":
        summary["groups"].pop()
    elif kind == "duplicate_group":
        summary["groups"][-1] = copy.deepcopy(summary["groups"][-2])
    elif kind == "missing_cell":
        del summary["cells"]["b2/P0"]
    elif kind == "cell_denominator":
        cell["n"] = 1383
    elif kind == "total_denominator":
        summary["rows"] = 9215
    elif kind == "missing_unknown":
        cell.update(S_known_rate=1383 / 1536, A_J_S_B_cross={"A0/J0/S0/B0": 1383})
    elif kind == "unknown_as_false":
        cell["A_J_S_B_cross"] = {"A0/J0/SFalse/B0": 1536}
    elif kind == "nonfinite":
        cell["V_rate"] = float("nan")
    elif kind == "boolean_rate":
        cell["V_rate"] = True
    elif kind == "fractional_count":
        cell["V_rate"] = 0.9
    elif kind == "changed_without_exchangeable":
        summary["groups"][-1]["changed"] = True
    elif kind == "missing_group_fact":
        del summary["groups"][-1]["changed"]
    elif kind == "missing_L1":
        del summary["groups"][-1]["credit_L1"]
    elif kind == "mixed_root_family":
        summary["groups"][-1]["family"] = FAMILIES[0]
    elif kind == "partial_coverage":
        audit["training_manipulation"]["roots"] = 96
    elif kind == "fractional_root_count":
        audit["training_manipulation"]["strict_gain_fraction"] = 0.51
    elif kind == "model_conditioned":
        audit["training_manipulation"]["model_outputs_used"] = True
    elif kind == "mismatched_J_classes":
        summary["groups"][-1]["J_class"] = "mixed"
    elif kind == "integer_boolean":
        summary["groups"][-1]["nonzero"] = 0
    elif kind == "forged_coverage_status":
        audit["training_manipulation"]["mean_gap"] = 0.04
    elif kind == "incomplete_family_matrix":
        audit["splits"]["TRAIN"]["family_questions"][FAMILIES[0]] -= 1
    else:
        del summary["frozen_spec_sha256"]
    report = support.assess_support(summary, audit)
    assert report["status"] == "INVALID_SUPPORT_EVIDENCE"
    assert report["training_authorized"] is False
    assert report["failures"][0]["detail"]


def test_unknowns_cannot_be_dropped_from_the_rate_denominator(evidence):
    summary, audit = evidence
    # Reporting 100% only among 1382 known labels would falsely pass 90%.
    summary["cells"]["b2/P0"].update(
        S_known_rate=1.0,
        A_J_S_B_cross={"A0/J0/S0/B0": 1382, "A0/J0/SNone/B0": 154},
    )
    report = support.assess_support(summary, audit)
    assert report["status"] == "INVALID_SUPPORT_EVIDENCE"
    assert "cross-table count mismatch" in report["failures"][0]["detail"]


def test_large_other_factor_exposure_cannot_fill_a_deficient_factor(evidence):
    summary, audit = evidence
    for group in summary["groups"]:
        if group["factor"] == "P0":
            deactivate(group)
    refresh_aggregates(summary)
    report = support.assess_support(summary, audit)
    assert report["status"] == "INSUFFICIENT_REGISTERED_EXPOSURE"
    assert {
        f"P0.{name}" for name in ("nonzero_groups", "roots", "families", "parents")
    } <= failure_names(report)
    assert report["observed"]["factors"]["P1"]["nonzero_groups"] == 24


def certificates():
    return [
        dict(
            root_id=f"TRAIN_fixture_{index:04d}",
            certificate=dict(
                root_id=f"TRAIN_fixture_{index:04d}",
                coverage_P0=0,
                coverage_P1=0.2 if index < 144 else 0,
            ),
        )
        for index in range(288)
    ]


def test_certificate_coverage_is_recomputed_but_not_claimed_independent_truth(evidence):
    roots = certificates()
    report = support.assess_support(*evidence, train_roots=roots)
    assert report["status"] == "PASS"
    assert report["observed"]["training_manipulation"]["certificate_recomputed"] is True
    assert (
        "not independently regenerated"
        in report["observed"]["training_manipulation"]["truth_dependency"]
    )
    roots[0]["certificate"]["coverage_P1"] = 0.8
    assert (
        support.assess_support(*evidence, train_roots=roots)["status"] == "INVALID_SUPPORT_EVIDENCE"
    )


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "unknown", "tiny_positive", "false_identity"]
)
def test_certificate_unknowns_and_duplicate_roots_cannot_be_zero_filled(evidence, kind):
    roots = certificates()
    if kind == "missing":
        roots.pop()
    elif kind == "duplicate":
        roots[-1] = roots[-2]
    elif kind == "unknown":
        roots[0]["certificate"]["coverage_P0"] = None
    elif kind == "tiny_positive":
        roots[-1]["certificate"]["coverage_P1"] = 1e-15  # Strictly positive, not silently ignored.
    else:
        roots[0]["certificate"]["root_id"] = "different"
    assert (
        support.assess_support(*evidence, train_roots=roots)["status"] == "INVALID_SUPPORT_EVIDENCE"
    )


def test_check_support_calls_actual_summary_entrypoint_and_is_read_only(
    evidence, tmp_path, monkeypatch
):
    from sr_f6 import census

    summary, audit = evidence
    calls = []

    def fixture_summary(root):
        calls.append(root)
        return summary

    # Only file-boundary fixture: pure tests above use the real census summarizer.
    monkeypatch.setattr(census, "complete_summary", fixture_summary)
    data = tmp_path / "DATA"
    data.mkdir()
    (data / "CPU_AUDIT.json").write_text(json.dumps(audit))
    certificate_path = data / "ROOT_CERTIFICATES_PRIVATE.jsonl"
    certificate_path.write_text(
        "\n".join(json.dumps(row) for row in [dict(root_id="DEV_CAL_ignored"), *certificates()])
    )
    before = {p: p.read_bytes() for p in data.iterdir()}
    report = support.check_support(tmp_path)
    assert calls == [tmp_path]
    assert report["status"] == "PASS"
    assert report["training_authorized"] is False
    assert "LOCAL_RECOMPUTATION" in report["evidence_origin"]
    assert before == {p: p.read_bytes() for p in data.iterdir()}
    certificate_path.unlink()
    assert support.check_support(tmp_path)["status"] == "INVALID_SUPPORT_EVIDENCE"


def test_check_support_propagates_failure_of_raw_census_validation(tmp_path, monkeypatch):
    from sr_f6 import census

    def rejected(root):
        raise PermissionError("Persisted labels/credits differ from frozen recomputation")

    monkeypatch.setattr(census, "complete_summary", rejected)
    report = support.check_support(tmp_path)
    assert report["status"] == "INVALID_SUPPORT_EVIDENCE"
    assert "Persisted labels/credits" in report["failures"][0]["detail"]
    assert not list(tmp_path.iterdir())
