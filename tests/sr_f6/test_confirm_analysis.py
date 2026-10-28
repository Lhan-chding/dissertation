"""Synthetic CPU fixtures never substitute for an actual confirmation."""

import copy
import hashlib
import json
from collections import Counter

import pytest

from sr_f6 import confirm_analysis as analysis
from sr_f6.group_diagnostics import census_diagnostics
from sr_f6.interface_diagnostics import diagnose_response
from srf6_ref.core import FAMILIES
from srf6_ref.design import build_bank
from srf6_ref.measurement import full_truth_response


def payload(value, *, lines=False):
    return ("".join(json.dumps(r) + "\n" for r in value) if lines else json.dumps(value)).encode()


@pytest.fixture(scope="module")
def matrices():
    roots = build_bank(analysis.SPLIT, 8, analysis.MASTER_SEED)
    tasks = [t for r in roots for v in r["tasks"].values() for t in v]
    train = build_bank("TRAIN", 48, 2026102714)
    records = []
    for task in tasks:
        correct = json.loads(full_truth_response(task["world"], task["query"]))
        for draw in range(8):
            value = copy.deepcopy(correct)
            if draw == 7:
                value["selected"] = ["unlisted-category"]
            text = json.dumps(value)
            if task["family"] == "BINDING" and task["root_id"].endswith("0000"):
                if task["variant"] == 0 and draw == 0:
                    text = "unfinished"
                if task["variant"] == 1 and draw == 0:
                    value["evidence"] = []
                    text = json.dumps(value)
            finish = "length" if text == "unfinished" else "balanced"
            diag = diagnose_response(text, task["world"], task["query"], finish)
            for parent in analysis.PARENTS:
                records.append(
                    dict(
                        diag["wire_score"],
                        parent=parent,
                        factor=task["factor"],
                        family=task["family"],
                        root_id=task["root_id"],
                        qid=task["qid"],
                        split=analysis.SPLIT,
                        variant=task["variant"],
                        policy_hash="a" * 64,
                        group_id=f"{parent}/{task['qid']}",
                        row_id=f"{parent}/{task['qid']}/{draw}",
                        draw=draw,
                        raw_text=text,
                        full_text=text,
                        finish_reason=finish,
                        wire_score=diag["wire_score"],
                        diagnostic=diag,
                        execution_origin="SYNTHETIC_TEST",
                    )
                )
    gaps = [r["certificate"]["coverage_P1"] - r["certificate"]["coverage_P0"] for r in train]
    audit = dict(
        status="PASS",
        model_calls=0,
        independent_native_oracles=2,
        cross_split_world_duplicates=0,
        splits={
            name: dict(
                roots=n,
                questions=q,
                factors=["P0", "P1"],
                family_questions={f: per for f in FAMILIES},
            )
            for name, n, q, per in [("DEV_CAL", 48, 384, 64), ("TRAIN", 288, 2304, 384)]
        },
        training_manipulation=dict(
            status="PASS",
            roots=288,
            mean_gap=sum(gaps) / 288,
            strict_gain_fraction=sum(g > 0 for g in gaps) / 288,
            model_outputs_used=False,
        ),
    )
    return roots, tasks, train, records, audit


@pytest.fixture
def case(tmp_path, monkeypatch, matrices):
    roots, tasks, train, records, old_audit = matrices
    root, old = tmp_path / "confirm", tmp_path / "original"
    root.mkdir()
    old.mkdir()
    monkeypatch.setattr(analysis.confirm_registration, "_root", lambda _: root)
    local, external = {}, {}

    def put(relative, value, *, lines=False, original=False):
        path = (old if original else root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = payload(value, lines=lines)
        path.write_bytes(raw)
        pin = hashlib.sha256(raw).hexdigest()
        (external if original else local)[str(path) if original else relative] = pin
        return dict(path=str(path) if original else relative, sha256=pin)

    audit_ref = put("DATA/CPU_AUDIT.json", old_audit, original=True)
    roots_ref = put("DATA/ROOT_CERTIFICATES_PRIVATE.jsonl", train, original=True, lines=True)
    put("DATA/ROOTS.jsonl", roots, lines=True)
    put(f"DATA/{analysis.SPLIT}/TASKS_PRIVATE.jsonl", tasks, lines=True)
    new_audit = dict(
        status="PASS",
        split=analysis.SPLIT,
        master_seed=analysis.MASTER_SEED,
        roots=48,
        questions=384,
        forced_anchors=18,
        historical_world_exclusion=dict(status="PASS", collisions=[]),
    )
    put("DATA/AUDIT.json", new_audit)
    plan = dict(
        source_split=analysis.SPLIT,
        master_seed=analysis.MASTER_SEED,
        gates=dict(review_seed=2026102701, new_independent_response_review=360),
        followup_requirements=dict(
            independent_response_review=dict(
                count=360,
                seed=2026102701,
                new_original_answers_only=True,
                authorization="SEPARATE_INDEPENDENT_REVIEW_REQUIRED",
            ),
            numerical_replay_selection=dict(
                qid=min(t["qid"] for t in tasks),
                parents=list(analysis.PARENTS),
                draws=list(range(8)),
                sequences=24,
                new_generated_responses=0,
                optimizer_updates=0,
                authorization="SEPARATE_ACTUAL_NUMERICAL_ADMISSION_REQUIRED",
            ),
        ),
    )
    plan_ref = put("CONFIRM_SPEC_FROZEN.json", plan)
    source = dict(path="SOURCE_MANIFEST.json", sha256="b" * 64)
    producer = dict(
        phase=analysis.SPLIT, deployment="/new/source", revision="a" * 40, source_manifest=source
    )
    predecessor = dict(deployment="/old/source", revision="b" * 40, status="SCREEN_PASS")
    registration = dict(
        source_root=str(old),
        original_identity_sha256="c" * 64,
        upstream_files=[audit_ref, roots_ref],
        source_manifest=source,
        revision=producer["revision"],
        deployment=producer["deployment"],
        predecessor_screen=predecessor,
        spec=plan_ref,
    )
    put("REGISTRATION.json", registration)
    evidence = dict(
        status="VERIFIED_STORED_CONFIRMATION_NOT_GATE_SIGNED",
        split=analysis.SPLIT,
        source_rows=9216,
        actual_recorded_calls=9216,
        records=records,
        raw_by_row={r["row_id"]: {"full_text": r["full_text"]} for r in records},
        input_producer=producer,
        consumer_source=copy.deepcopy(producer),
        predecessor_screen=predecessor,
        registration_sha256=local["REGISTRATION.json"],
        saved_token_validation=dict(
            status="PASS", checked_responses=9216, checked_tokens=9216, new_model_calls=0
        ),
        accounting=dict(
            completion_attempts=9216, generated_tokens=9216, extra_forward_sequences=48
        ),
        execution_origin="SYNTHETIC_TEST",
    )

    def refresh():
        evidence["inputs"] = [dict(path=k, sha256=v) for k, v in sorted(local.items())]
        evidence["external_inputs"] = [dict(path=k, sha256=v) for k, v in sorted(external.items())]
        evidence["registration_sha256"] = local["REGISTRATION.json"]

    refresh()
    monkeypatch.setattr(analysis.confirm_evidence, "validate_saved_confirm", lambda _: evidence)
    return dict(
        root=root,
        old=old,
        evidence=evidence,
        put=put,
        refresh=refresh,
        plan=plan,
        registration=registration,
        new_audit=new_audit,
    )


def saved(case, name):
    return json.loads((case["root"] / analysis.DIRECTORY / name).read_text())


def test_complete_real_algorithm_support_preserves_pending_review_and_all_diagnostics(case):
    report = analysis.assess(case["root"])
    assert report["status"] == "CONFIRM_SUPPORT_PASS_REVIEW_PENDING", report
    assert report["training_allowed"] is report["generation_allowed"] is False
    assert report["evaluation_allowed"] is False
    assert report["gate_status"] == "NOT_SIGNED"
    assert report["followup_requirements"]["independent_review"]["status"] == "REQUIRED_NOT_RUN"
    numerical = report["followup_requirements"]["actual_numerical_gate"]
    assert numerical["status"] == "REQUIRED_NOT_RUN"
    assert numerical["selection"]["sequences"] == 24
    support = saved(case, "SUPPORT.json")
    assert support["status"] == "PASS"
    counts = [c for c in support["checks"] if c["name"].endswith("_count")]
    assert len(counts) == 18 and all(c["minimum"] == 1383 for c in counts)
    assert all(c["denominator"] == 1536 for c in counts)
    assert support["data_audit_scope"]["historical_dev_cal_is_new_confirmation"] is False
    assert support["data_audit_scope"]["original"]["cpu_audit"]["path"].startswith(str(case["old"]))
    assert report["input_producer"] != report["predecessor_screen"]
    groups = saved(case, "GROUP_DIAGNOSTICS.json")
    assert len(groups["groups"]) == 1152
    assert all(g["credit_status"] == "COMPLETE_G8" for g in groups["groups"])
    interface = saved(case, "GROUP_INTERFACE.json")
    assert interface["roots"] == 48 and interface["forced_roots"] == 18
    assert len(interface["root_certificates"]) == 96
    assert len(interface["all_draw_pairs"]) == 864
    assert sum(r["cross_product_pairs"] for r in interface["all_draw_pairs"]) == 55296
    assert any(r["counts"]["relative_unknown"] for r in interface["all_draw_pairs"])
    assert any(r["at_least_one_unfinished_pairs"] for r in interface["all_draw_pairs"])
    gap = saved(case, "J_GAP.json")
    assert gap["all"]["exclusive_gap_partition"]["TRUE_BUT_INSUFFICIENT"] == 6
    assert gap["additional_admission_gate"] is False
    summary = saved(case, "CENSUS_SUMMARY.json")
    assert summary["rows"] == 9216 and summary["split"] == analysis.SPLIT
    assert summary["frozen_spec_sha256"] == analysis.digest(case["plan"])
    assert summary["frozen_spec_file_sha256"] == case["registration"]["spec"]["sha256"]
    assert Counter(report["raw_finish_counts"]) == Counter(balanced=9210, length=6)
    seal = saved(case, "SEAL.json")
    assert seal["training_allowed"] is False
    path = case["root"] / seal["assessment"]["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == seal["assessment"]["sha256"]
    assert all(
        hashlib.sha256((case["root"] / r["path"]).read_bytes()).hexdigest() == r["sha256"]
        for r in report["reports"].values()
    )


@pytest.fixture
def light_diagnostics(monkeypatch):
    """Boundary cases still run original summarize/support; skip repeated descriptive CPU work."""
    monkeypatch.setattr(
        analysis.group_diagnostics,
        "census_diagnostics",
        lambda *a, **k: dict(
            status="DERIVED_COMPLETE_SUPPLIED_MATRIX",
            full_registered_task_manifest=True,
            groups=[dict(credit_status="COMPLETE_G8")] * 1152,
        ),
    )
    monkeypatch.setattr(analysis.group_diagnostics, "mathematical_diagnostics", lambda *a: {})
    monkeypatch.setattr(analysis, "_interface", lambda *a: {})
    monkeypatch.setattr(analysis, "j_gap_breakdown", lambda *a: {})


def test_original_cond_support_failure_stops_without_new_gate_or_resampling(
    case, light_diagnostics
):
    case["evidence"]["records"] = [
        dict(r, J=1, V=1, S=1, B=1, A=1) for r in case["evidence"]["records"]
    ]
    report = analysis.assess(case["root"])
    assert report["status"] == "CONFIRM_FAIL_STOP"
    failed = {r["name"] for r in report["support_failures"]}
    assert "P0.nonzero_groups" in failed and "P1.nonzero_groups" in failed
    assert "SHUF.total_changed_groups" in failed
    assert report["training_allowed"] is False
    assert report["new_model_calls"] == 0


def test_original_observability_count_is_not_pooled_or_known_conditioned(case, light_diagnostics):
    bad = copy.deepcopy(case["evidence"]["records"])
    chosen = [r for r in bad if (r["parent"], r["factor"]) == ("b0", "P0")]
    for i, row in enumerate(chosen):
        row.update(V=int(i < 1382), S=None if i >= 1382 else 1, B=None if i >= 1382 else 1, J=0)
    case["evidence"]["records"] = bad
    report = analysis.assess(case["root"])
    assert report["status"] == "CONFIRM_FAIL_STOP"
    failures = {r["name"]: r for r in report["support_failures"]}
    for name in ("V", "S_known", "B_known"):
        item = failures[f"b0/P0.{name}_count"]
        assert item["observed"] == 1382 and item["minimum"] == 1383


def test_validator_failure_is_technical_without_analysis_or_fake_pass(case, monkeypatch):
    def fail(_):
        raise PermissionError("partial raw cannot be resampled")

    monkeypatch.setattr(analysis.confirm_evidence, "validate_saved_confirm", fail)
    monkeypatch.setattr(analysis, "summarize", lambda *a: pytest.fail("must validate raw first"))
    report = analysis.assess(case["root"])
    assert report["status"] == "TECHNICAL_INVALID"
    assert report["reports"] == {} and report["training_allowed"] is False
    assert saved(case, "SEAL.json")["gate_status"] == "NOT_SIGNED"


@pytest.mark.parametrize("kind", ["partial", "old_split", "old_source", "token_shortfall"])
def test_incomplete_or_old_identity_cannot_be_analysis_authority(case, kind):
    evidence = case["evidence"]
    if kind == "partial":
        evidence["records"] = evidence["records"][:-1]
    elif kind == "old_split":
        evidence["split"] = "DEV_CAL"
    elif kind == "old_source":
        evidence["input_producer"]["deployment"] = evidence["predecessor_screen"]["deployment"]
    else:
        evidence["saved_token_validation"]["checked_responses"] = 9215
    assert analysis.assess(case["root"])["status"] == "TECHNICAL_INVALID"


@pytest.mark.parametrize(
    "relative,original",
    [
        ("DATA/AUDIT.json", False),
        ("DATA/CPU_AUDIT.json", True),
        ("DATA/ROOT_CERTIFICATES_PRIVATE.jsonl", True),
    ],
)
def test_modified_bound_data_is_not_hidden_by_prior_verified_status(case, relative, original):
    path = (case["old"] if original else case["root"]) / relative
    path.write_text(path.read_text() + " ")
    report = analysis.assess(case["root"])
    assert report["status"] == "TECHNICAL_INVALID"
    assert "Bound analysis input changed" in report["error"]


@pytest.mark.parametrize(
    "kind", ["review_seed", "missing_numerical", "different_qid", "new_dev_cal"]
)
def test_followups_and_new_data_scope_cannot_be_relabelled(case, kind):
    plan = case["plan"]
    if kind == "new_dev_cal":
        case["new_audit"]["split"] = "DEV_CAL"
        case["put"]("DATA/AUDIT.json", case["new_audit"])
    else:
        if kind == "review_seed":
            plan["gates"]["review_seed"] += 1
        elif kind == "missing_numerical":
            del plan["followup_requirements"]
        else:
            plan["followup_requirements"]["numerical_replay_selection"]["qid"] = "better-result"
        case["put"]("CONFIRM_SPEC_FROZEN.json", plan)
    case["refresh"]()
    assert analysis.assess(case["root"])["status"] == "TECHNICAL_INVALID"


def test_reports_are_write_once_and_repeat_analysis_cannot_replace_outcome(
    case, light_diagnostics, monkeypatch
):
    first = analysis.assess(case["root"])
    assert first["status"] == "CONFIRM_SUPPORT_PASS_REVIEW_PENDING"
    original = (case["root"] / analysis.DIRECTORY / "ASSESSMENT.json").read_bytes()
    monkeypatch.setattr(
        analysis.confirm_evidence,
        "validate_saved_confirm",
        lambda _: pytest.fail("existing analysis must not rescan raw"),
    )
    with pytest.raises(FileExistsError):
        analysis.assess(case["root"])
    assert (case["root"] / analysis.DIRECTORY / "ASSESSMENT.json").read_bytes() == original


def test_partial_analysis_does_not_retry_expensive_validation(case, monkeypatch):
    output = case["root"] / analysis.DIRECTORY
    output.mkdir(parents=True)
    (output / "SUPPORT.json").write_text("partial")
    monkeypatch.setattr(
        analysis.confirm_evidence,
        "validate_saved_confirm",
        lambda _: pytest.fail("partial analysis must not rescan raw"),
    )
    with pytest.raises(FileExistsError, match="partial"):
        analysis.assess(case["root"])


def test_authorized_root_guard_rejects_before_any_technical_failure_write(tmp_path):
    with pytest.raises(PermissionError):
        analysis.assess(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_symlink_output_is_rejected_before_validation(case, tmp_path, monkeypatch):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (case["root"] / "CONFIRM").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        analysis.confirm_evidence,
        "validate_saved_confirm",
        lambda _: pytest.fail("unsafe output root"),
    )
    with pytest.raises(PermissionError, match="Symlink"):
        analysis.assess(case["root"])


def test_explicit_confirm_diagnostics_keep_default_dev_cal_boundary(matrices):
    roots, _, _, records, _ = matrices
    tasks = [t for views in roots[0]["tasks"].values() for t in views]
    selected = [r for r in records if r["root_id"] == roots[0]["root_id"] and r["parent"] == "b0"]
    with pytest.raises(ValueError, match="DEV_CAL"):
        census_diagnostics(selected, tasks)
    report = census_diagnostics(selected, tasks, master=analysis.MASTER_SEED, split=analysis.SPLIT)
    assert report["status"] == "DERIVED_PARTIAL_MATRIX"
    assert report["observed_rows"] == 64 and report["stage_gate_decision"] == "NOT_ISSUED"
    with pytest.raises(ValueError, match="registered development"):
        census_diagnostics(selected, tasks, split="TRAIN")
