"""Statistical contracts for fixed protocol probes; no model construction."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.protocol_state_probes.statistics import (
    ContrastEngine,
    clopper_pearson,
    family_weights,
    read_committed_rows,
)


def test_cp_boundary_keeps_finite_uncertainty():
    interval = clopper_pearson(0, 32)
    assert interval[0] == 0
    assert interval[1] > 0.10
    assert pytest.approx(0.0893681989862648) == 1 - 0.05 ** (1 / 32)
    assert clopper_pearson(0, 0) == (None, None)


def test_alias_coefficients_cancel_before_uncertainty():
    engine = ContrastEngine({"same": [0.0] * 32}, posterior_draws=1000, bootstrap_draws=100)
    result = engine.estimate(
        [{"scene": "a", "family": "cross", "terms": [("same", 1), ("same", -1)]}]
    )
    assert result["estimate"] == result["mc_se"] == 0
    assert result["jeffreys_low"] == result["jeffreys_high"] == 0
    assert result["uniform_low"] == result["uniform_high"] == 0
    assert result["independent_cells"] == 0


def test_zero_observations_are_not_zero_uncertainty():
    engine = ContrastEngine(
        {"a": [0.0] * 32, "b": [0.0] * 32}, posterior_draws=3000, bootstrap_draws=100
    )
    result = engine.estimate([{"scene": "x", "family": "cross", "terms": [("a", 1), ("b", -1)]}])
    assert result["mc_se"] == 0
    assert result["jeffreys_low"] < 0 < result["jeffreys_high"]


def test_missing_cell_blocks_full_contrast_not_zero_filled():
    engine = ContrastEngine({"a": [1.0] * 32}, posterior_draws=100, bootstrap_draws=100)
    result = engine.estimate(
        [{"scene": "x", "family": "cross", "terms": [("a", 1), ("missing", -1)]}]
    )
    assert result["status"] == "MISSING_ENDPOINTS"
    assert result["estimate"] is None
    assert result["missing_cells"] == ["missing"]


def test_equal_family_weight_not_completion_or_scene_count_weight():
    assert family_weights(["cross", "cross", "trend"]) == pytest.approx([0.25, 0.25, 0.5])
    engine = ContrastEngine(
        {"a": [1.0] * 32, "b": [1.0] * 32, "c": [0.0] * 64},
        posterior_draws=100,
        bootstrap_draws=100,
    )
    result = engine.estimate(
        [
            {"scene": "1", "family": "cross", "terms": [("a", 1)]},
            {"scene": "2", "family": "cross", "terms": [("b", 1)]},
            {"scene": "3", "family": "trend", "terms": [("c", 1)]},
        ]
    )
    assert result["estimate"] == 0.5
    assert result["scene_bootstrap_low"] == result["scene_bootstrap_high"] == 0.5


def test_b1_signed_preference_preserves_old_new_covariance():
    engine = ContrastEngine(
        {"old": [-1.0] * 16 + [1.0] * 16, "new": [1.0] * 32},
        posterior_draws=2000,
        bootstrap_draws=100,
    )
    result = engine.estimate(
        [{"scene": "x", "family": "cross", "terms": [("new", 1), ("old", -1)]}],
        support=(-1.0, 0.0, 1.0),
    )
    assert result["estimate"] == 1
    assert result["mc_se"] == pytest.approx(np.sqrt(1 / 31))
    assert result["posterior_model"] == "independent_cell_Dirichlet_signed_preference"


def test_scene_bootstrap_pairs_protocols_and_checkpoints():
    samples = {
        f"{scene}-{cp}": [value] * 32
        for scene, value in enumerate((0.0, 0.25, 1.0))
        for cp in ("a", "b")
    }
    engine = ContrastEngine(samples, posterior_draws=100, bootstrap_draws=500)
    result = engine.estimate(
        [
            {
                "scene": str(scene),
                "family": "cross",
                "terms": [(f"{scene}-a", 1), (f"{scene}-b", -1)],
            }
            for scene in range(3)
        ],
        binary=False,
    )
    assert result["scene_bootstrap_low"] == result["scene_bootstrap_high"] == 0
    assert result["mc_se"] == 0


def test_commits_only_and_hash_validation(tmp_path):
    chunks = tmp_path / "raw" / "S96" / "D48" / "case" / "chunks"
    chunks.mkdir(parents=True)
    row = {
        "checkpoint": "S96",
        "canonical_case_owner": "case",
        "panel": "D48",
        "role": "frozen_probe",
        "draw_index": 0,
        "sample_key": "key",
        "features": {"event": "I"},
    }
    path = chunks / "frozen_probe-000000.json"
    path.write_text(json.dumps({"rows": [row]}))
    (chunks / "uncommitted.json").write_text('{"rows": []}')
    commit = chunks / "frozen_probe-000000.COMMIT.json"
    receipt = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_rows": 1,
        "sample_keys": ["key"],
    }
    commit.write_text(json.dumps(receipt))
    rows, receipt_info = read_committed_rows(tmp_path)
    assert rows == [row]
    assert receipt_info["committed_chunks"] == 1
    path.write_text('{"rows": []}')
    with pytest.raises(ValueError, match="hash"):
        read_committed_rows(tmp_path)


def _prepared_fixture(root):
    from src.protocol_state_probes.protocol import digest

    design = Path(__file__).resolve().parents[1] / "docs" / "protocol_state_probes" / "design"
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "protocol.json": design / "protocol.json",
        "cases.jsonl": design / "manifests/cases.jsonl",
        "audit_labels.jsonl": design / "manifests/audit_labels_NOT_FOR_MODEL.jsonl",
        "program_predictions.jsonl": design / "manifests/program_predictions.jsonl",
    }
    for name, source in files.items():
        (root / name).write_bytes(source.read_bytes())
    aliases = {
        row["case_id"]: row["owner_case_id"]
        for row in (
            json.loads(line)
            for line in (design / "manifests/prompt_aliases.jsonl").read_text().splitlines()
        )
    }
    (root / "alias_map.json").write_text(json.dumps(aliases))
    jobs = [
        json.loads(line)
        for line in (design / "manifests/logical_jobs.jsonl").read_text().splitlines()
    ]
    protocol = json.loads((root / "protocol.json").read_text())
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "jobs": jobs,
        "verification": {"recompiled_from_prepared": True, "all_fields_match": True},
        "runtime": {"path": "/nonexistent/remote/runtime.json", "sha256": "provenance-only"},
        "file_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (*files, "alias_map.json")
        },
    }
    manifest["identity"] = digest(manifest)
    (root / "execution_manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_full_manifest_empty_summary_keeps_controls_separate_and_missing(tmp_path):
    from src.protocol_state_probes.statistics import summarize

    _prepared_fixture(tmp_path)
    (tmp_path / "U22_EXPOSURE_AUDIT.json").write_text(
        json.dumps(
            {
                "status": "CONTAMINATED_DOWNGRADED",
                "audit_complete": True,
                "downgraded_to_development": True,
            }
        )
    )
    result = summarize(tmp_path, posterior_draws=32, bootstrap_draws=32)
    assert result["status"] == "INCOMPLETE"
    assert result["planned_outputs"] == 70656
    assert result["committed_outputs"] == 0
    assert result["stage_statuses"]["INITIAL_S96_REPORT_zh.md"] == "INCOMPLETE"
    import pandas as pd

    main = pd.read_csv(tmp_path / "tables/prompt_metrics.csv")
    controls = pd.read_csv(tmp_path / "tables/execution_controls.csv")
    assert set(main.panel) == {"D48", "U22"}
    assert set(controls.panel) == {"CONTROL_COPY", "CONTROL_DUPLICATE", "CONTROL_CLEAN_REL"}
    assert main.estimate.isna().all()
    assert controls.estimate.isna().all()
    assert set(main[main.panel == "U22"].effective_split_role) == {"development_diagnostic"}
    assert set(main[main.panel == "U22"].U22_exposure_audit_status) == {"CONTAMINATED_DOWNGRADED"}
    assert result["U22_effective_split_role"] == "development_diagnostic"
    assert "不能称为独立" in (tmp_path / "U22_REPLICATION_REPORT_zh.md").read_text()
    assert (tmp_path / "tables/joint_behavior_atoms.parquet").is_file()
    assert (
        "INCOMPLETE_NO_FINAL_SCIENTIFIC_DECISION"
        in (tmp_path / "FINAL_MODELING_DECISION_zh.md").read_text()
    )


def test_analysis_rejects_changed_frozen_file(tmp_path):
    from src.protocol_state_probes.statistics import RunAnalysis

    _prepared_fixture(tmp_path)
    with (tmp_path / "cases.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Frozen run artifact changed"):
        RunAnalysis(tmp_path, posterior_draws=10, bootstrap_draws=10, seed=1)


def test_committed_rows_reject_wrong_checkpoint_binding_and_seed(tmp_path):
    from src.protocol_state_probes.protocol import digest, sample_identity

    manifest = {"identity": "frozen-run", "protocol_id": "probe-test"}
    binding = {
        "run_identity": manifest["identity"],
        "checkpoint": "S96",
        "checkpoint_record": {"identity": "old-weights"},
        "runtime_sha256": "runtime",
    }
    binding["identity"] = digest(
        {
            "run_identity": binding["run_identity"],
            "checkpoint": binding["checkpoint_record"],
            "runtime_sha256": binding["runtime_sha256"],
        }
    )
    (tmp_path / "checkpoint_bindings").mkdir()
    (tmp_path / "checkpoint_bindings/S96.json").write_text(json.dumps(binding))
    chunks = tmp_path / "raw/S96/D48/case/chunks"
    chunks.mkdir(parents=True)
    sample_key, seed = sample_identity("probe-test", "S96", "case", "frozen_probe", 0)
    row = {
        "checkpoint": "S96",
        "canonical_case_owner": "case",
        "role": "frozen_probe",
        "draw_index": 0,
        "sample_key": sample_key,
        "seed": seed,
    }
    path = chunks / "frozen_probe-000000.json"

    def write(identity):
        path.write_text(json.dumps({"identity": identity, "rows": [row]}))
        path.with_name("frozen_probe-000000.COMMIT.json").write_text(
            json.dumps(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "n_rows": 1,
                    "sample_keys": [sample_key],
                }
            )
        )

    write("stale-other-checkpoint")
    with pytest.raises(ValueError, match="binding mismatch"):
        read_committed_rows(tmp_path, manifest)
    write(binding["identity"])
    assert read_committed_rows(tmp_path, manifest)[0] == [row]
    row["seed"] += 1
    write(binding["identity"])
    with pytest.raises(ValueError, match="seed/sample identity"):
        read_committed_rows(tmp_path, manifest)


def test_full_manifest_partial_outputs_keep_out_of_domain_and_control_errors(tmp_path):
    from src.protocol_state_probes.protocol import digest, sample_identity
    from src.protocol_state_probes.run import commit_chunk
    from src.protocol_state_probes.semantics import score_raw
    from src.protocol_state_probes.statistics import RunAnalysis

    manifest = _prepared_fixture(tmp_path)
    cases = {
        row["case_id"]: row
        for row in map(json.loads, (tmp_path / "cases.jsonl").read_text().splitlines())
    }
    audits = {
        row["case_id"]: row
        for row in map(json.loads, (tmp_path / "audit_labels.jsonl").read_text().splitlines())
    }
    predictions = {
        row["case_id"]: row
        for row in map(
            json.loads, (tmp_path / "program_predictions.jsonl").read_text().splitlines()
        )
    }
    binding = {
        "run_identity": manifest["identity"],
        "checkpoint": "S96",
        "checkpoint_record": {"fixture": True},
        "runtime_sha256": "runtime",
    }
    binding["identity"] = digest(
        {
            "run_identity": binding["run_identity"],
            "checkpoint": binding["checkpoint_record"],
            "runtime_sha256": binding["runtime_sha256"],
        }
    )
    (tmp_path / "checkpoint_bindings").mkdir()
    (tmp_path / "checkpoint_bindings/S96.json").write_text(json.dumps(binding))
    risk = next(
        case_id
        for case_id, prediction in predictions.items()
        if cases[case_id]["panel"] == "D48"
        and cases[case_id]["protocol"] == "B1"
        and not prediction["programs"]["V1"]["in_domain"]
        and prediction["programs"]["V1"]["canonical"]
        != predictions[case_id[:-2] + "B0"]["programs"]["V1"]["canonical"]
    )
    original = risk[:-2] + "O0"
    selected = [risk, original] + [
        next(case_id for case_id, case in cases.items() if case["panel"] == panel)
        for panel in ("CONTROL_COPY", "CONTROL_DUPLICATE", "CONTROL_CLEAN_REL")
    ]
    for case_id in selected:
        case = cases[case_id]
        rows = []
        for draw in range(8):
            if case_id in (risk, original):
                raw_text = json.dumps(predictions[case_id]["programs"]["V1"]["emitted"])
            else:
                raw_text = (
                    "bad-json"
                    if draw < 4
                    else json.dumps(
                        audits[case_id]["control_expected_canonical"]
                        or audits[case_id]["truth_world"]
                    )
                )
            key, seed = sample_identity(
                manifest["protocol_id"], "S96", case_id, "frozen_probe", draw
            )
            rows.append(
                {
                    "checkpoint": "S96",
                    "canonical_case_owner": case_id,
                    "case_id": case_id,
                    "panel": case["panel"],
                    "role": "frozen_probe",
                    "draw_index": draw,
                    "seed": seed,
                    "sample_key": key,
                    "raw_text": raw_text,
                    "features": score_raw(
                        raw_text, case, audits[case_id], predictions.get(case_id)
                    ),
                }
            )
        path = (
            tmp_path
            / "raw"
            / "S96"
            / case["panel"]
            / case_id.replace(":", "_")
            / "chunks/frozen_probe-000000.json"
        )
        commit_chunk(path, rows, binding["identity"])
    analysis = RunAnalysis(tmp_path, posterior_draws=64, bootstrap_draws=32, seed=1)
    prompt, controls, mass, _atoms = analysis.prompt_metrics()
    assert not any(row["status"] == "FAIL" for row in mass)
    assert all(not row["panel"].startswith("CONTROL_") for row in prompt)
    assert {row["panel"] for row in controls} == {
        "CONTROL_COPY",
        "CONTROL_DUPLICATE",
        "CONTROL_CLEAN_REL",
    }
    invalid = next(
        row
        for row in prompt
        if row["case_id"] == risk and row["checkpoint"] == "S96" and row["metric"] == "pI"
    )
    assert invalid["successes"] == invalid["n"] == 8
    matched = next(
        row
        for row in prompt
        if row["case_id"] == risk and row["checkpoint"] == "S96" and row["metric"] == "match_V1"
    )
    assert matched["estimate"] == 1
    result = []
    analysis._b1_rows("S96", "D48", result)
    shift = next(
        row
        for row in result
        if row.get("base_scene_id") == cases[risk]["base_scene_id"] and row["variant"] == "V1"
    )
    assert shift["estimate"] == 2
    assert shift["new_prediction_in_domain"] is False
    assert shift["status"] == "PARTIAL_FIXED_N"
    assert shift["B1_match_new_count"] == shift["B0_match_old_count"] == 8
