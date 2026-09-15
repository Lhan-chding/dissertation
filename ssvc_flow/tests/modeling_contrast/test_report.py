"""Report generation cannot promote missing experiments to evidence."""

import json

import numpy as np
import pytest

from src.modeling_contrast.io import canonical_hash, sha256_file
from src.modeling_contrast.protocol import load_config
from src.modeling_contrast.report import (
    Evidence,
    _covariance_audit,
    _isolation_rows,
    _supplemental_audits,
    _validation_audit,
    run_report,
)
from src.modeling_qualification.math_contracts import helmert


@pytest.fixture
def report_inputs(tmp_path):
    root = tmp_path / "runs/modeling_contrast_v2"
    out = tmp_path / "docs/modeling_contrast/results"
    root.mkdir(parents=True)
    out.mkdir(parents=True)
    lock = {
        "status": "NO_ACCEPTABLE_MODEL",
        "allow_fresh_cpu": False,
        "selected": [],
        "best_nonzero_diagnostic": None,
        **{
            name: {"status": "FAIL", "reasons": ["fixture_not_qualified"]}
            for name in (
                "science_gate",
                "cost_gate",
                "resource_gate",
                "input_identity_gate",
                "data_gate",
            )
        },
    }
    lock["lock_sha256"] = canonical_hash(lock)
    (root / "N3").mkdir()
    (root / "N3/MODEL_SELECTION_LOCK.json").write_text(json.dumps(lock))
    return root, out


def test_failed_gates_never_read_stale_fresh_results_or_invent_test_pass(report_inputs, tmp_path):
    root, out = report_inputs
    (root / "N4").mkdir()
    (root / "N4/FRESH_COLLECTION_SUMMARY.json").write_text("unreadable stale result")
    result = run_report(load_config(), root, out, tmp_path / "parent", project_root=tmp_path)
    assert result["fresh_cpu_status"] == "NOT_RUN"
    readiness = json.loads((out / "machine_readable_readiness.json").read_text())
    assert readiness["new_module_tests"]["status"] == "UNKNOWN"
    assert readiness["new_module_tests"]["passed"] is None
    assert readiness["recommended_surrogates"] == []
    assert readiness["science_gate"]["status"] == "FAIL"


def test_outputs_preserve_existing_evidence_and_reject_report_overwrite(report_inputs, tmp_path):
    root, out = report_inputs
    (out / "existing_receipt.json").write_text("original bytes")
    run_report(load_config(), root, out, tmp_path / "parent", project_root=tmp_path)
    assert (out / "existing_receipt.json").read_text() == "original bytes"
    before = sha256_file(out / "MODELING_DECISION_V2_zh.md")
    with pytest.raises(FileExistsError):
        run_report(load_config(), root, out, tmp_path / "parent", project_root=tmp_path)
    assert sha256_file(out / "MODELING_DECISION_V2_zh.md") == before
    manifest = json.loads((out / "REPORT_MANIFEST.json").read_text())
    assert "existing_receipt.json" not in manifest["outputs"]
    assert all(sha256_file(out / name) == digest for name, digest in manifest["outputs"].items())


def test_corrupt_selection_lock_is_not_reported_as_authorization(report_inputs, tmp_path):
    root, out = report_inputs
    path = root / "N3/MODEL_SELECTION_LOCK.json"
    value = json.loads(path.read_text())
    value["allow_fresh_cpu"] = True
    path.write_text(json.dumps(value))
    run_report(load_config(), root, out, tmp_path / "parent", project_root=tmp_path)
    readiness = json.loads((out / "machine_readable_readiness.json").read_text())
    assert readiness["selection_lock_integrity"] == "FAIL"
    assert readiness["fresh_cpu_status"] == "NOT_RUN"
    assert readiness["overall_decision"] == "BLOCKED_LOCK_INTEGRITY"


def test_source_tables_are_hash_bound_and_missing_inputs_remain_explicit(report_inputs, tmp_path):
    root, out = report_inputs
    (root / "N1").mkdir()
    table = root / "N1/OBSERVATION_COMPARISON.csv"
    table.write_text("method,n,target,view,rmse\nO_IND,64,joint_1_minus_joint_0,helmert,0.5\n")
    run_report(load_config(), root, out, tmp_path / "parent", project_root=tmp_path)
    binding = json.loads((out / "REPORT_SOURCE_BINDING.json").read_text())
    assert binding["inputs"][str(table.resolve())] == sha256_file(table)
    assert any(path.endswith("N2/AGGREGATE.csv") for path in binding["missing_inputs"])


def test_report_path_stays_inside_authorized_new_result_roots(report_inputs, tmp_path):
    root, _ = report_inputs
    with pytest.raises(ValueError, match="output path"):
        run_report(
            load_config(),
            root,
            tmp_path / "old_results",
            tmp_path / "parent",
            project_root=tmp_path,
        )


def test_junit_execution_overrides_an_inconsistent_pass_total(report_inputs, tmp_path):
    root, out = report_inputs
    (root / "implementation").mkdir()
    xml = root / "implementation/final_tests.xml"
    xml.write_text(
        '<testsuite><testcase classname="tests" name="passed"/>'
        '<testcase classname="tests" name="errored"><error/></testcase></testsuite>'
    )
    (root / "implementation/final_tests.json").write_text(
        json.dumps({"status": "PASS", "passed": 999, "junit": str(xml)})
    )
    run_report(load_config(), root, out, tmp_path / "parent", project_root=tmp_path)
    ready = json.loads((out / "machine_readable_readiness.json").read_text())
    assert ready["new_module_tests"]["status"] == "FAIL"
    assert ready["new_module_tests"]["passed"] == 1
    assert ready["new_module_tests"]["errors"] == 1
    assert ready["engineering_status"] == "BLOCKED"


def test_n3_permission_and_completed_collection_are_not_independent_validation(
    report_inputs, tmp_path
):
    root, out = report_inputs
    lock_path = root / "N3/MODEL_SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text())
    lock.pop("lock_sha256")
    lock.update(allow_fresh_cpu=True, selected=[{"configuration_id": "candidate"}])
    for key in ("science_gate", "cost_gate", "resource_gate", "input_identity_gate", "data_gate"):
        lock[key] = {"status": "PASS"}
    lock["lock_sha256"] = canonical_hash(lock)
    lock_path.write_text(json.dumps(lock))
    (root / "N4").mkdir()
    (root / "N4/FRESH_COLLECTION_SUMMARY.json").write_text(json.dumps({"status": "PASS"}))
    run_report(load_config(), root, out, tmp_path / "parent", project_root=tmp_path)
    ready = json.loads((out / "machine_readable_readiness.json").read_text())
    assert ready["fresh_cpu_status"] == "COLLECTED_NOT_EVALUATED"
    assert ready["recommended_surrogates"] == []
    assert ready["frozen_development_candidates"] == [{"configuration_id": "candidate"}]


def test_isolation_uses_sealed_raw_arrays_and_separate_primary_metrics(tmp_path):
    root = tmp_path / "run"
    out = root / "N2/N2D/seed101/a0/O_LR_ORIGIN"
    parent = tmp_path / "parent"
    out.mkdir(parents=True)
    parent.mkdir()
    (parent / "probe_metadata.json").write_text(
        json.dumps([{"group": i // 12, "weight": 1 / 72} for i in range(72)])
    )
    truth = np.zeros((4, 2, 72, 4))
    truth[..., 0], truth[..., 3] = 0.01, -0.01
    predictions = (1.5 * truth @ helmert()).reshape(1, 8, 72, 3)
    path = out / "predictions.npz"
    np.savez(path, prediction=predictions, truth=truth)
    (out / "diagnostics.json").write_text(
        json.dumps(
            {
                "main_ranking_eligible": False,
                "prediction_sha256": sha256_file(path),
                "records": [
                    {
                        "array_index": 0,
                        "development_only": True,
                        "main_ranking_eligible": False,
                        "observation": "O_LR_ORIGIN",
                        "method": "C5",
                        "rank_cap": 1,
                        "diagnostic": "FINITE",
                        "covariance_source": "FINITE_PAID",
                        "seed": 101,
                        "subspace_to_exact": {"projector_frobenius_distance": 0.0},
                    }
                ],
            }
        )
    )
    rows = _isolation_rows(Evidence(), root, parent)
    assert rows[0]["pX_nrmse"] == pytest.approx(0.5)
    assert rows[0]["v_nrmse"] == pytest.approx(0.5)
    assert rows[0]["training_seed_count"] == 1
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="unbound"):
        _isolation_rows(Evidence(), root, parent)


def test_scope_receipt_requires_matching_real_evidence_bytes(tmp_path):
    root = tmp_path / "run"
    implementation = root / "implementation"
    implementation.mkdir(parents=True)
    evidence = implementation / "git_diff.txt"
    evidence.write_text("reviewed scoped diff")
    receipt = {
        "schema_version": "contrast-git-scope-v1",
        "status": "PASS",
        "evidence_files": {str(evidence): sha256_file(evidence)},
        "worktree": str(tmp_path),
        "branch": "codex/example",
        "parent_commit": load_config()["repository"]["parent_commit"],
        "independent_worktree": True,
        "unrelated_changes_preserved": True,
        "historical_sources_unchanged": True,
        "reviewed_paths": ["src/example.py"],
    }
    (implementation / "git_scope_review.json").write_text(json.dumps(receipt))
    assert _supplemental_audits(Evidence(), root, load_config())["A4"]["status"] == "PASS"
    evidence.write_text("changed after review")
    assert _supplemental_audits(Evidence(), root, load_config())["A4"]["status"] == "FAIL"


def test_cost_pass_without_complete_coverage_does_not_pass_acceptance(tmp_path):
    root = tmp_path / "run"
    implementation = root / "implementation"
    implementation.mkdir(parents=True)
    evidence = implementation / "cost_raw.json"
    evidence.write_text("{}")
    receipt = {
        "schema_version": "contrast-total-cost-v1",
        "status": "PASS",
        "evidence_files": {str(evidence): sha256_file(evidence)},
        "coverage": {"generation": True},
        "operation_counts": {"actions": 5},
        "wall_seconds_by_stage": {"N1": 1},
        "scenario_costs_are_conditional": True,
        "actual_cpu_saving_claim": True,
        "cpu_timing_comparison_status": "NOT_RUN",
    }
    (implementation / "TOTAL_COST_AUDIT.json").write_text(json.dumps(receipt))
    audit = _supplemental_audits(Evidence(), root, load_config())["F2"]
    assert audit["status"] == "BLOCKED"
    assert audit["actual_cpu_saving_claim_verified"] is False


def test_complete_cost_accounting_does_not_imply_cpu_savings(tmp_path):
    root = tmp_path / "run"
    implementation = root / "implementation"
    implementation.mkdir(parents=True)
    evidence = implementation / "cost_raw.json"
    evidence.write_text("{}")
    receipt = {
        "schema_version": "contrast-total-cost-v1",
        "status": "PASS",
        "evidence_files": {str(evidence): sha256_file(evidence)},
        "coverage": {
            name: True
            for name in (
                "generation",
                "labels",
                "scoring",
                "calibration",
                "oracle",
                "fit",
                "query",
                "candidate_updates",
            )
        },
        "operation_counts": {"actions": 5, "updates": 0},
        "wall_seconds_by_stage": {"N1": 1},
        "scenario_costs_are_conditional": True,
        "actual_cpu_saving_claim": False,
        "cpu_timing_comparison_status": "NOT_RUN",
    }
    (implementation / "TOTAL_COST_AUDIT.json").write_text(json.dumps(receipt))
    audit = _supplemental_audits(Evidence(), root, load_config())["F2"]
    assert audit["status"] == "PASS"
    assert audit["actual_cpu_saving_claim_verified"] is False


def test_projection_receipt_checks_sparse_bytes_and_all_materialized_units(tmp_path):
    root = tmp_path / "run"
    base = root / "recompute"
    unit = base / "projection/N2A/seed101_X_VALID/a0"
    unit.mkdir(parents=True)
    raw, oracle = tmp_path / "raw.npz", tmp_path / "oracle.npz"
    raw.write_bytes(b"raw")
    oracle.write_bytes(b"oracle")
    sparse, errors = unit / "projected_sparse.npz", unit / "raw_projected_errors.jsonl.gz"
    sparse.write_bytes(b"frozen sparse bytes")
    errors.write_bytes(b"errors")
    binding = {
        "raw_prediction_file": str(raw),
        "raw_prediction_file_sha256": sha256_file(raw),
        "oracle_baseline_file": str(oracle),
        "oracle_baseline_file_sha256": sha256_file(oracle),
        "sparse_sha256": sha256_file(sparse),
        "error_table_sha256": sha256_file(errors),
        "main_ranking_eligible": False,
        "all_projected_arrays_bitwise_reconstructed_from_sparse_archive": True,
    }
    (unit / "PROJECTION_BINDING.json").write_text(json.dumps(binding))
    summary = {
        "status": "PASS",
        "units": 1,
        "materialized_N2A_B_C_E_units": 1,
        "complete_for_materialized_N2A_B_C_E_units": True,
        "main_ranking_eligible": False,
    }
    (base / "PROJECTION_AUDIT_SUMMARY.json").write_text(json.dumps(summary))
    assert _supplemental_audits(Evidence(), root, load_config())["D10"]["status"] == "PASS"
    sparse.write_bytes(b"changed projection")
    assert _supplemental_audits(Evidence(), root, load_config())["D10"]["status"] != "PASS"


def test_c1_receipt_cannot_hide_modified_historical_counts(tmp_path):
    root, parent = tmp_path / "run", tmp_path / "parent"
    unit = root / "N2/C1_auxiliary/seed101_X_VALID/a0/O_IND_n64"
    unit.mkdir(parents=True)
    source = root / "N1/packet_arrays.npz"
    source.parent.mkdir(parents=True)
    origin, levels = np.zeros((2, 4)), np.zeros((8, 2, 4))
    np.savez(source, legacy_origin_counts=origin, legacy_total_levels=levels)
    (source.parent / "RUN_MANIFEST.json").write_text(
        json.dumps({"status": "COMPLETE", "outputs": {source.name: sha256_file(source)}})
    )
    changed = levels.copy()
    changed[0, 0, 0] = 1
    auxiliary = unit / "C1_auxiliary.npz"
    np.savez(auxiliary, legacy_origin_counts=origin, legacy_total_levels=changed)
    receipt = {
        "version": "C1_SAME_BANK_COUNTS_V1",
        "status": "REPAIRED_WITHOUT_MUTATING_N1",
        "source_packet_path": str(source),
        "source_packet_sha256": sha256_file(source),
        "auxiliary_sha256": sha256_file(auxiliary),
        "historical_replicas_preserved": [0, 1, 2, 3, 4],
        "origin_counts_preserved": True,
        "fit_evaluation_independent": True,
    }
    (unit / "receipt.json").write_text(json.dumps(receipt))
    before = sha256_file(source)
    result = _covariance_audit(Evidence(), root, parent)
    assert result["C1_auxiliary_repair"]["status"] == "FAIL"
    assert result["C1_auxiliary_repair"]["verified_overlay_count"] == 0
    assert sha256_file(source) == before
    changed = levels.copy()
    changed[5, 0, 0] = 1
    np.savez(auxiliary, legacy_origin_counts=origin, legacy_total_levels=changed)
    receipt["auxiliary_sha256"] = sha256_file(auxiliary)
    (unit / "receipt.json").write_text(json.dumps(receipt))
    assert _covariance_audit(Evidence(), root, parent)["C1_auxiliary_repair"]["status"] == "PASS"


def test_completed_n4_requires_bound_predictions_and_qualified_selected_results(tmp_path):
    selection, prediction, scoring = (
        tmp_path / name for name in ("lock.json", "prediction.npz", "metrics.json")
    )
    for path in (selection, prediction, scoring):
        path.write_bytes(b"frozen bytes")
    selected = [
        {
            "model": "C2",
            "method": "O_IND",
            "n": 64,
            "rank": 1,
            "alpha": 0.01,
            "eta": 0,
            "fit_banks": 8,
        }
    ]
    value = {
        "status": "INDEPENDENT_VALIDATION_COMPLETE",
        "science_status": "PREDICTION_PASS",
        "input_hashes": {str(selection): sha256_file(selection)},
        "prediction_hashes": {str(prediction): sha256_file(prediction)},
        "scoring_hashes": {str(scoring): sha256_file(scoring)},
        "selection_sha256": sha256_file(selection),
        "selected_results": [
            {
                "configuration_id": "C2|O_IND|64|1|0.01|0|8",
                "prediction_status": "PREDICTION_PASS",
                "resolution_status": "RESOLUTION_FAIL",
            }
        ],
    }
    output = tmp_path / "INDEPENDENT_VALIDATION.json"
    audit = _validation_audit(Evidence(), value, output, selection, selected)
    assert audit["status"] == "PASS"
    assert audit["recommendation_qualified"] is False
    value["selected_results"][0]["resolution_status"] = "RESOLUTION_PASS"
    assert (
        _validation_audit(Evidence(), value, output, selection, selected)[
            "recommendation_qualified"
        ]
        is True
    )
    prediction.write_bytes(b"changed after scoring")
    audit = _validation_audit(Evidence(), value, output, selection, selected)
    assert audit["status"] == "FAIL"
    assert audit["recommendation_qualified"] is False
