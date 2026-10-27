"""CPU evidence-chain regression; these fixtures cannot certify actual PILOT."""

import importlib.util
import json
from pathlib import Path

import pytest

from sr_f6 import pilot_audit
from sr_f6.transactions import RunTransactions
from srf6_ref.design import make_root

_spec = importlib.util.spec_from_file_location(
    "srf6_audit_pilot_fixture", Path(__file__).with_name("test_pilot.py")
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


@pytest.fixture(scope="module")
def prototype():
    return make_root("DEV_PILOT", "BINDING", 0, 2026102712)["tasks"]["P1"][0]


@pytest.fixture
def complete(tmp_path, prototype):
    case = _fixture._case(tmp_path, prototype)
    _fixture._run(case, "original")
    return case


def _verify(case):
    with RunTransactions(case["run"], case["identity"]) as tx:
        return pilot_audit.verify_synthetic_run(tx, root=case["experiment"])


def test_actual_cpu_states_originals_credit_and_all_calls_are_replayed(complete):
    result = _verify(complete)
    assert result["status"] == "SYNTHETIC_ONLY"
    assert result["scope"] == "SYNTHETIC_ONLY"
    assert result["training_authorized"] is False
    assert result["updates"] == 12 and result["responses"] == 1536
    assert len(result["groups"]) == 192
    assert result["inventory"]["actual_generated_responses"] == 1536
    assert result["inventory"]["physical_preupdate_count"] == 12
    assert result["inventory"]["attempts"][0]["status"] == "SEGMENT_COMPLETE"


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_call",
        "missing_token",
        "unknown_kind",
        "orphan_task",
        "missing_end",
        "raw_tamper",
        "extra_group",
        "audit_failure",
    ],
)
def test_original_evidence_mutations_cannot_be_hidden_by_complete_commits(complete, mutation):
    run = complete["run"]
    accounting = run / "accounting/original.jsonl"
    if mutation in ("extra_call", "missing_token", "unknown_kind"):
        rows = accounting.read_text().splitlines()
        if mutation == "extra_call":
            rows.append(rows[0])
        elif mutation == "missing_token":
            rows.pop(1)
        else:
            row = json.loads(rows[0])
            row["kind"] = "unknown_call"
            rows.append(json.dumps(row))
        accounting.write_text("\n".join(rows) + "\n")
    elif mutation == "orphan_task":
        (run / "accounting/orphan.jsonl").write_text("")
    elif mutation == "missing_end":
        (run / "EXECUTIONS/original/END.json").unlink()
    elif mutation == "raw_tamper":
        raw = run / "rollouts/step-001/groups/00/raw.jsonl"
        raw.write_text(raw.read_text().replace('"full_text":', '"fake":', 1))
    elif mutation == "extra_group":
        (run / "rollouts/step-012/groups/16").mkdir()
    else:
        with (run / "NUMERICAL_TOKEN_AUDIT.jsonl").open("a") as stream:
            stream.write('{"status":"UNCOMMITTED_FAILURE_NO_OPTIMIZER_STEP"}\n')
    with pytest.raises((PermissionError, ValueError, KeyError)):
        _verify(complete)


@pytest.mark.parametrize("point", ["after_ready", "after_update", "after_rename"])
def test_interrupted_complete_originals_reconcile_without_resampling(tmp_path, prototype, point):
    case = _fixture._case(tmp_path, prototype)

    def fail(name, step):
        if name == point and step == 1:
            raise RuntimeError("saved interrupted process")

    with pytest.raises(RuntimeError):
        _fixture._run(case, "crashed", fail)
    _fixture._run(case, "restored")
    result = _verify(case)
    facts = {a["task_id"]: a for a in result["inventory"]["attempts"]}
    assert facts["crashed"]["status"] == "INTERRUPTED_RECONCILED_WITH_ORIGINALS"
    assert result["inventory"]["actual_generated_responses"] == 1536
    assert result["inventory"]["physical_preupdate_count"] == (
        13 if point == "after_update" else 12
    )


def test_production_run_core_cannot_upgrade_cpu_receipts(complete):
    with (
        RunTransactions(complete["run"], complete["identity"]) as tx,
        pytest.raises(PermissionError),
    ):
        pilot_audit._verify_run(tx, complete["experiment"], synthetic=False)
    with pytest.raises(PermissionError, match="CUDA"):
        pilot_audit._runtime(tx, complete["experiment"], "original", synthetic=False)


def test_cache_only_interrupted_update_has_its_own_actual_task_event(tmp_path, prototype):
    case = _fixture._case(tmp_path, prototype)

    def crash_at(point):
        def crash(name, step):
            if name == point and step == 1:
                raise RuntimeError("registered interruption")

        return crash

    for task, point in (("generated", "after_ready"), ("cache_update", "after_update")):
        with pytest.raises(RuntimeError):
            _fixture._run(case, task, crash_at(point))
    _fixture._run(case, "committed")
    result = _verify(case)
    facts = {row["task_id"]: row for row in result["inventory"]["attempts"]}
    assert facts["cache_update"]["counts"].get("completion_attempts", 0) == 0
    assert facts["cache_update"]["status"] == "INTERRUPTED_RECONCILED_WITH_ORIGINALS"
    old = [
        row
        for row in result["inventory"]["physical_preupdate_events"]
        if row["task_id"] == "cache_update"
    ]
    assert len(old) == 1 and old[0]["status"] == "UNCOMMITTED_PREUPDATE_RECONCILED"
    assert result["inventory"]["physical_preupdate_count"] == 13


def _group(parent, step, root, family, nonzero=True):
    return dict(
        step=step,
        qid=f"{parent}-{step}-{root}",
        root_id=root,
        family=family,
        nonzero=nonzero,
        exchangeable=False,
        changed=False,
        source_step=step - 1,
        commit_sha256=f"commit-{step}",
        original_reference={"path": f"step{step}/{root}"},
    )


def test_registered_exposure_is_applied_groups_and_real_later_generation_not_final_step():
    runs = [
        dict(
            parent=p,
            arm="U11",
            groups=[
                _group(p, s, f"root-{i}", f"family-{i % 3}")
                for i, s in enumerate([1, 2, 3, 4, 11, 12, 12, 12])
            ],
        )
        for p in ("b0", "b1", "b2")
    ]
    facts = pilot_audit._exposure(runs)
    assert facts["registered_exposure_met"] is True
    assert facts["nonzero_groups"] == 24 and len(facts["roots"]) == 8
    for run in runs:
        for group in run["groups"]:
            group.update(step=12, source_step=11)
    facts = pilot_audit._exposure(runs)
    assert facts["nonzero_groups"] == 24
    assert facts["registered_exposure_met"] is False
    assert facts["parents_with_feedback"] == []


def test_previous_cached_policy_cannot_prove_feedback():
    runs = [
        dict(parent=p, arm="U11", groups=[_group(p, 1, "r1", "F1"), _group(p, 4, "r2", "F2")])
        for p in ("b0", "b1", "b2")
    ]
    for run in runs:
        run["groups"][1]["source_step"] = 0
    assert pilot_audit._exposure(runs)["parents_with_feedback"] == []


def test_native_g8_accounting_rejects_independent_draw_reservation_shape():
    with pytest.raises(PermissionError, match="reservation"):
        pilot_audit._event_key(
            dict(
                kind="completion_attempts",
                count=1,
                metadata=dict(group_seed=1, protocol="structured"),
            )
        )


def test_production_entry_requires_real_admission_before_inspecting_caller_receipts(tmp_path):
    with pytest.raises((PermissionError, FileNotFoundError, ValueError)):
        pilot_audit.verify_runtime_pilot(
            tmp_path,
            phase_identity_sha256="a" * 64,
            pilot_spec_sha256="b" * 64,
            admission_ref={},
            recovery_binding_ref={},
            recovery_comparison_ref={},
            evaluation_summary_ref={},
        )


def test_comparison_inventory_retains_prior_technical_unknown(tmp_path):
    from sr_f6 import recovery

    directory = tmp_path / "recovery"
    folder = directory / "COMPARISONS"
    folder.mkdir(parents=True)
    facts = dict(
        format=recovery.FORMAT,
        status="SYNTHETIC_ONLY",
        scope="SYNTHETIC_ONLY",
        binding_ref={"path": "binding", "sha256": "a" * 64},
        exact=True,
    )
    selected = folder / ("a" * 32 + ".json")
    selected.write_text(json.dumps(facts))
    compared = {
        **facts,
        "receipt_reference": recovery._ref(tmp_path, selected),
        "verified_from_saved_facts": True,
    }
    failure = dict(
        format=recovery.FORMAT,
        status="INVALID_OR_INCOMPLETE_EVIDENCE",
        binding_ref=facts["binding_ref"],
        gate_status="NOT_SIGNED",
        training_authorized=False,
    )
    (folder / ("b" * 32 + "-FAILURE.json")).write_text(json.dumps(failure))
    report = pilot_audit._comparison_inventory(tmp_path, directory, compared)
    assert report["status"] == "HISTORICAL_TECHNICAL_UNKNOWN"
    assert len(report["receipts"]) == 2
    bad = dict(facts, exact=False, status="FAIL")
    (folder / ("c" * 32 + ".json")).write_text(json.dumps(bad))
    with pytest.raises(PermissionError, match="Historical recovery"):
        pilot_audit._comparison_inventory(tmp_path, directory, compared)


def test_unclosed_empty_task_is_not_inferred_to_have_zero_calls(complete):
    runtime, _, _ = _fixture._fresh(complete, "unknown_loaded")
    assert runtime.calls == 0  # Actual fixture fact is deliberately unavailable to the consumer.
    with pytest.raises(PermissionError, match="Unclosed attempt"):
        _verify(complete)
