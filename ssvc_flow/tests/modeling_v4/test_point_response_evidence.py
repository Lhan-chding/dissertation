"""Tiny synthetic derived files test E arithmetic/gates; they are not GPU evidence."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.modeling_v3.io import finalize_run
from src.modeling_v4.analysis_rules import load_analysis_rules
from src.modeling_v4.config import digest
from src.modeling_v4.data_adapter import CONTRASTS
from src.modeling_v4.point_response_evidence import (
    WAITING,
    build_point_response_evidence,
    verify_point_response_evidence,
)


def bound(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return bound(path)


def campaign(
    tmp_path,
    monkeypatch,
    *,
    error=0.001,
    se=0.0001,
    signal=0.01,
    missing=False,
    independent=True,
    wrong_role=False,
    wrong_method=False,
):
    # The phase verifier has its own actual-original tests. Isolate this numerical
    # assessment here; production calls it normally and fixture output cannot open E.
    monkeypatch.setattr(
        "src.modeling_v4.point_response_evidence.verify_phase_evidence",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        "src.modeling_v4.point_response_evidence.validate_selection", lambda config, value: value
    )
    config = json.loads(
        (Path(__file__).parents[2] / "configs/modeling_v4/protocol.json").read_text()
    )
    selection = {
        "selection_hash": digest("tiny-selection"),
        "m": 32,
        "campaign_id": "fixture-point-only",
        "gradient_reference": "FULL_SCORE_JVP",
        "primary_models": [
            {"id": "raw", "model": "FULL_DUAL_RIDGE"},
            {"id": "rbf", "model": "FULL_RBF_RAW"},
        ],
    }
    tasks = [
        {
            "id": f"D_map_{seed}_X_BASE_{step}",
            "kind": "map",
            "stage": "D",
            "role": "calibration",
            "seed": seed,
            "arm": "X_BASE",
            "step": step,
            "calibration_banks": 32,
            "query_banks": 24,
            "prompts": 72,
            "draws": 1024,
        }
        for seed in (42001, 42002, 42003)
        for step in (32, 96)
    ]
    probes = [{"prompt_id": f"p{i}"} for i in range(72)]
    rules = load_analysis_rules()
    plan = {
        "config": config,
        "tasks": tasks,
        "source": {"sha256": digest("fixture")},
        "inputs": {"panels": {"observation": probes}},
        "analysis_rules": rules,
        "analysis_rules_hash": digest(rules),
    }
    tasks_binding = write(tmp_path / "tasks.json", plan)
    evidence = {
        "phase": "D_CALIBRATION",
        "selection_hash": selection["selection_hash"],
        "campaign_id": selection["campaign_id"],
        "test_opened": False,
        "tasks_binding": tasks_binding,
        "origins": [],
    }
    for task in tasks:
        root = tmp_path / task["id"]
        banks = [
            {
                "bank_id": f"query_{i:03d}",
                "role": "query",
                "contrasts": {
                    target: {"exact_inference_alias": False} for target, _, _ in CONTRASTS
                },
            }
            for i in range(24)
        ]
        forks = write(root / "forks.json", {"banks": banks})
        complete = write(
            root / "collection.json",
            {"task": task, "forks": forks, "execution_kind": "REAL_CUDA_MODEL"},
        )
        binding = {
            "selection_hash": selection["selection_hash"],
            "tasks": tasks_binding,
            "task": complete,
            "config_hash": digest(config),
            "source": plan["source"],
            "execution_kind": "REAL_CUDA_MODEL",
        }
        fit_dir = root / "fit"
        fit_dir.mkdir()
        rows = []
        for model in selection["primary_models"]:
            for bank in banks:
                for target, _, _ in CONTRASTS:
                    if missing and bank["bank_id"] == "query_000":
                        continue
                    rows.append(
                        {
                            "origin_id": task["id"],
                            "seed": task["seed"],
                            "role": "test" if wrong_role else "calibration",
                            "arm": "X_BASE",
                            "step": task["step"],
                            "n": 1024,
                            "bank_id": bank["bank_id"],
                            "target": target,
                            "method": "ZERO" if wrong_method else model["model"],
                            "design_id": model["id"],
                            "evaluation_unit": "prompt",
                            "query_response_used_for_fit": False,
                            "is_alias": False,
                            "prediction_status": "PREDICTED",
                            "prompt_ids": [p["prompt_id"] for p in probes],
                            "prediction": [
                                [signal + error, -signal - error, 0.0, 0.0] for _ in probes
                            ],
                        }
                    )
        pq.write_table(pa.Table.from_pylist(rows), fit_dir / "PREDICTIONS.parquet")
        prediction = bound(fit_dir / "PREDICTIONS.parquet")
        evdir = fit_dir / "evaluation"
        metrics = write(
            evdir / "METRICS.json",
            {
                "kind": "V4_EVALUATION",
                "binding": {**binding, "predictions": prediction},
                "selection_modified": False,
                "reference_read_scope": "EVALUATION_ONLY",
            },
        )
        with (evdir / "CORE_RESULTS.csv").open("w", newline="") as stream:
            core_rows = [
                {
                    **{k: v for k, v in row.items() if k != "prediction"},
                    "prompt_ids": json.dumps(row["prompt_ids"]),
                    "prediction_record_id": prediction["sha256"],
                    "reference_independent": independent,
                }
                for row in rows
            ]
            writer = csv.DictWriter(stream, fieldnames=list(core_rows[0]))
            writer.writeheader()
            writer.writerows(core_rows)
        write(fit_dir / "REFERENCE_DIAGNOSTICS.json", {"rules": rules, "coverage_guarantee": False})
        refs = {}
        for bank in banks:
            bid = bank["bank_id"]
            refs[f"{bid}__estimate"] = np.tile([signal, -signal, 0.0, 0.0], (72, 3, 1))
            refs[f"{bid}__se"] = np.full((72, 3, 4), se)
            refs[f"{bid}__resolved"] = np.full((72, 3, 4), se > 0 and 1.96 * se <= 0.0025)
        np.savez_compressed(fit_dir / "REFERENCE_STATISTICS.npz", **refs)
        fit = write(
            fit_dir / "RESULTS.json",
            {
                "kind": "V4_RESPONSE_MAP_FIT",
                "status": "COMPLETED",
                "binding": binding,
                "reference_read_after_predictions": True,
                "query_response_used_for_fit": False,
                "predictions": prediction,
            },
        )
        finalize_run(fit_dir, binding)
        evidence["origins"].append(
            {
                "task_id": task["id"],
                "seed": task["seed"],
                "arm": task["arm"],
                "step": task["step"],
                "fit_receipt": fit,
                "evaluation_receipt": metrics,
                "collection_receipt": complete,
            }
        )
    return config, selection, evidence


def test_actual_numbers_resolve_and_reproduce_but_fixture_cannot_enter_production(
    tmp_path, monkeypatch
):
    config, selection, evidence = campaign(tmp_path, monkeypatch)
    result = build_point_response_evidence(config, selection, evidence, fixture=True)
    assert result["status"] == "POINT_RESPONSE_RESOLVED"
    assert all(a["q95_absolute_error"] == pytest.approx(0.001) for a in result["assessments"])
    assert all(a["expected_cells"] == 6 * 24 * 3 * 72 * 4 for a in result["assessments"])
    assert result["cross_seed_95"] == "NOT_CERTIFIED"
    assert verify_point_response_evidence(config, selection, result, fixture=True) == result
    with pytest.raises(ValueError, match="derived point-response"):
        verify_point_response_evidence(config, selection, result)


@pytest.mark.parametrize(
    "arguments", [{"error": 0.01}, {"se": 0.01}, {"signal": 0.0, "error": 0.0}, {"se": 0.0}]
)
def test_complete_measured_failure_is_unresolved(tmp_path, monkeypatch, arguments):
    config, selection, evidence = campaign(tmp_path, monkeypatch, **arguments)
    result = build_point_response_evidence(config, selection, evidence, fixture=True)
    assert result["status"] == "POINT_RESPONSE_UNRESOLVED"


def test_missing_rows_wait_and_handwritten_rehashed_success_is_rejected(tmp_path, monkeypatch):
    config, selection, evidence = campaign(tmp_path, monkeypatch, missing=True)
    result = build_point_response_evidence(config, selection, evidence, fixture=True)
    assert result["status"] == WAITING
    forged = copy.deepcopy(result)
    forged["status"] = "POINT_RESPONSE_RESOLVED"
    forged["evidence_hash"] = digest({k: v for k, v in forged.items() if k != "evidence_hash"})
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_point_response_evidence(config, selection, forged, fixture=True)


@pytest.mark.parametrize(
    "arguments", [{"independent": False}, {"wrong_role": True}, {"wrong_method": True}]
)
def test_unrelated_or_nonindependent_rows_cannot_qualify(tmp_path, monkeypatch, arguments):
    config, selection, evidence = campaign(tmp_path, monkeypatch, **arguments)
    with pytest.raises(ValueError):
        build_point_response_evidence(config, selection, evidence, fixture=True)


def test_missing_inputs_wait_and_changed_original_bytes_reject(tmp_path, monkeypatch):
    config, selection, evidence = campaign(tmp_path, monkeypatch)
    assert build_point_response_evidence(config, selection, fixture=True)["status"] == WAITING
    incomplete = copy.deepcopy(evidence)
    incomplete["origins"].pop()
    assert (
        build_point_response_evidence(config, selection, incomplete, fixture=True)["status"]
        == WAITING
    )
    assert (
        build_point_response_evidence(config, selection, tmp_path / "missing.json", fixture=True)[
            "status"
        ]
        == WAITING
    )
    result = build_point_response_evidence(config, selection, evidence, fixture=True)
    reference = Path(result["input_bindings"][0]["reference_statistics"]["path"])
    reference.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        verify_point_response_evidence(config, selection, result, fixture=True)
