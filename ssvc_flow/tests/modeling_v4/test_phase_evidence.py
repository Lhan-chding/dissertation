"""Tiny synthetic receipt files verify metadata integrity, never GPU effectiveness."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.modeling_v3.io import finalize_run
from src.modeling_v4.config import digest
from src.modeling_v4.phase_evidence import (
    ReceiptReader,
    _verify_cv,
    build_phase_evidence,
    verify_phase_evidence,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))
    return bound(path)


def bound(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture_campaign(tmp_path, *, phase="C", reuse=False):
    config = json.loads(
        (Path(__file__).parents[2] / "configs/modeling_v4/protocol.json").read_text()
    )
    seeds = [41001, 41002] if phase == "C" else [41001, 41002, 41003]
    arms = ["X_BASE"] if phase == "C" else ["X_BASE", "X_VALID"]
    root = tmp_path / "campaign"
    tasks = []
    counts = (32, 16, 24, 256) if phase == "C" else (96, 24, 72, 1024)
    for seed in seeds:
        for arm in arms:
            task = {"id": f"source_{seed}_{arm}", "kind": "source", "seed": seed, "arm": arm}
            tasks.append(task)
            state = write(root / "tasks" / task["id"] / "dummy-checkpoint.bin", {"fixture": True})
            write(
                root / "tasks" / task["id"] / "COMPLETE.json",
                {
                    "kind": "source",
                    "status": "COMPLETED",
                    "seed": seed,
                    "arm": arm,
                    "steps": 128,
                    "origins": {"32": state, "96": state},
                    "execution_kind": "REAL_CUDA_MODEL",
                },
            )
        for step in (32, 96):
            task_id = f"{'D_' if phase != 'C' else ''}map_{seed}_X_BASE_{step}"
            task = {
                "id": task_id,
                "kind": "map",
                "seed": seed,
                "arm": "X_BASE",
                "step": step,
                **dict(
                    zip(
                        ("calibration_banks", "query_banks", "prompts", "draws"),
                        counts,
                        strict=True,
                    )
                ),
            }
            if reuse:
                task["reuse_prefix_task"] = f"map_{seed}_X_BASE_{step}"
                previous = {**task, "id": task["reuse_prefix_task"]}
                previous.pop("reuse_prefix_task")
                tasks.append(previous)
                write(
                    root / "tasks" / previous["id"] / "COMPLETE.json",
                    {
                        "status": "COMPLETED",
                        "kind": "map",
                        "task": previous,
                        "origin_id": previous["id"],
                        "execution_kind": "REAL_CUDA_MODEL",
                    },
                )
            tasks.append(task)
    plan = {
        "schema": "ssvc-v4-task-list-1",
        "config": config,
        "tasks": tasks,
        "campaign_id": "tiny-fixture-not-real-evidence",
        "source": {"sha256": digest("fixture")},
        "root": str(root),
        "workers": 2,
    }
    plan["task_list_hash"] = digest(plan)
    tasks_binding = write(tmp_path / "tasks.json", plan)
    write(
        root / "B_MERGED.json", {"status": "TECHNICAL_BRIDGE_COMPLETED", "parity": {"passed": True}}
    )
    fits = {}
    for task in tasks:
        if task["kind"] != "map" or (reuse and not task.get("reuse_prefix_task")):
            continue
        task_id = task["id"]
        origin = task.get("reuse_prefix_task", task_id)
        task_dir = root / "tasks" / task_id
        banks = [
            {"role": role, "bank_id": f"{role}_{i:03d}"}
            for role, count in (
                ("calibration", task["calibration_banks"]),
                ("query", task["query_banks"]),
            )
            for i in range(count)
        ]
        forks = write(task_dir / "forks.json", {"origin_id": origin, "banks": banks})
        response = {"origin_id": origin, "all_queries_retained": True}
        for role, draws in (("work", task["draws"]), ("reference", 4096)):
            chunk = write(task_dir / f"{role}-tiny.bin", {"fixture": True, "role": role})
            response[role] = {
                "identity": {
                    "origin_id": origin,
                    "role": role,
                    "draws": draws,
                    "rng_namespace": f"{origin}:{role}",
                },
                "count": task["prompts"] * draws,
                "chunks": [{**chunk, "start": 0, "stop": task["prompts"] * draws}],
            }
        response_binding = write(task_dir / "response.json", response)
        completed = {
            "status": "COMPLETED",
            "kind": "map",
            "task": task,
            "origin_id": origin,
            "forks": forks,
            "response": response_binding,
            "execution_kind": "REAL_CUDA_MODEL",
        }
        completion = write(task_dir / "COMPLETE.json", completed)
        binding = {
            "tasks": tasks_binding,
            "task": completion,
            "forks": forks,
            "response": response_binding,
            "config_hash": digest(config),
            "source": plan["source"],
            "execution_kind": "REAL_CUDA_MODEL",
            "task_id": task_id,
        }
        out = tmp_path / "fits" / task_id
        predictions = write(out / "PREDICTIONS.parquet", {"tiny_metadata_only": True})
        write(
            out / "MODELS.json",
            {
                "binding": binding,
                "models": [{"method": "FULL_DUAL_RIDGE"}],
                "calibration_bank_ids": [b["bank_id"] for b in banks if b["role"] == "calibration"],
                "query_labels_read_for_fit": False,
            },
        )
        eval_binding = {**binding, "predictions": predictions}
        eval_dir = out / "evaluation"
        write(
            eval_dir / "METRICS.json",
            {
                "kind": "V4_EVALUATION",
                "binding": eval_binding,
                "selection_modified": False,
                "reference_read_scope": "EVALUATION_ONLY",
                "summary": [{"status": "UNKNOWN"}],
                "core_results": [],
                "core_results_file": "CORE_RESULTS.csv",
            },
        )
        (eval_dir / "CORE_RESULTS.csv").write_text(
            "seed,origin_id,bank_id,prediction_status,prediction_record_id\n"
            f"{task['seed']},{task_id},query_000,UNKNOWN,{predictions['sha256']}\n"
        )
        finalize_run(eval_dir, {**eval_binding, "kind": "V4_EVALUATION"})
        result = {
            "kind": "V4_RESPONSE_MAP_FIT",
            "status": "COMPLETED",
            "task_id": task_id,
            "binding": binding,
            "reference_read_after_predictions": True,
            "query_response_used_for_fit": False,
            "prediction_records": 1,
            "calibration_banks": task["calibration_banks"],
            "query_banks": task["query_banks"],
            "predictions": predictions,
            "evaluation": {
                "status": "COMPLETE",
                "out": str(eval_dir),
                "metrics": str(eval_dir / "METRICS.json"),
                "core_results": str(eval_dir / "CORE_RESULTS.csv"),
            },
        }
        fits[task_id] = write(out / "RESULTS.json", result)
        finalize_run(out, {**binding, "kind": "V4_RESPONSE_MAP_FIT"})
    return tasks_binding, fits, root


def test_C_actual_bound_four_maps_and_streamed_evaluation(tmp_path):
    tasks, fits, _ = fixture_campaign(tmp_path)
    evidence = build_phase_evidence(tasks, fit_results=fits)
    assert len(evidence["sources"]) == 2 and len(evidence["origins"]) == 4
    assert evidence["scientific_status"] == "NOT_CERTIFIED"
    assert evidence["verification"]["arrays_deserialized"] is False
    assert verify_phase_evidence(evidence) == evidence


def test_missing_actual_fit_and_hash_mismatch_rejected(tmp_path):
    tasks, fits, _ = fixture_campaign(tmp_path)
    with pytest.raises(ValueError, match="every expected origin"):
        build_phase_evidence(tasks, fit_results=dict(list(fits.items())[:-1]))
    broken = copy.deepcopy(fits)
    next(iter(broken.values()))["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        build_phase_evidence(tasks, fit_results=broken)


def test_actual_raw_mutation_rejected_on_reverification(tmp_path):
    tasks, fits, root = fixture_campaign(tmp_path)
    evidence = build_phase_evidence(tasks, fit_results=fits)
    (root / "tasks" / "map_41001_X_BASE_32" / "reference-tiny.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        verify_phase_evidence(evidence)


def test_normalized_claim_cannot_be_edited_after_original_verification(tmp_path):
    tasks, fits, _ = fixture_campaign(tmp_path)
    evidence = build_phase_evidence(tasks, fit_results=fits)
    evidence["origins"][0]["fitted_model_ids"] = ["ZERO"]
    with pytest.raises(ValueError, match="no longer reproduces"):
        verify_phase_evidence(evidence)


def test_D_does_not_invent_whole_seed_cv_and_retains_C_origin(tmp_path):
    tasks, fits, _ = fixture_campaign(tmp_path, phase="D_DEVELOPMENT", reuse=True)
    evidence = build_phase_evidence(tasks, fit_results=fits, phase="D_DEVELOPMENT")
    assert evidence["missing_evidence"] == ["ACTUAL_WHOLE_SEED_DEVELOPMENT_CV"]
    assert evidence["whole_seed_cv"] == []
    assert all(row["task_id"] == "D_" + row["origin_id"] for row in evidence["origins"])
    assert verify_phase_evidence(evidence) == evidence


def test_reader_detects_file_mutation_even_within_one_verification(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(b"old")
    reader = ReceiptReader()
    reader.binding(path)
    path.write_bytes(b"new")
    with pytest.raises(ValueError, match="changed during"):
        reader.binding(path)


def cv_fixture(root, *, invalid_scope=False, invalid_indices=False):
    from src.modeling_v4.data_adapter import CONTRASTS

    config = json.loads(
        (Path(__file__).parents[2] / "configs/modeling_v4/protocol.json").read_text()
    )
    plan = {"config": config, "source": {"sha256": digest("fixture")}, "campaign_id": "cv-fixture"}
    tasks = write(root / "tasks.json", plan)
    records, originals = [], []
    for seed in (41001, 41002, 41003):
        for step in (32, 96):
            original = root / "originals" / f"{seed}_{step}"
            origin = f"D_map_{seed}_X_BASE_{step}"
            rows = [
                {
                    "origin_id": origin,
                    "seed": seed,
                    "arm": "X_BASE",
                    "step": step,
                    "role": "development",
                    "bank_role": "calibration",
                    "bank_id": f"calibration_{i:03d}",
                    "contrast_id": target,
                    "candidate_fingerprint": digest([origin, i, target, "candidate"]),
                    "baseline_fingerprint": digest([origin, i, target, "baseline"]),
                }
                for i in range(96)
                for target, _, _ in CONTRASTS
            ]
            labels = write(
                original / "labels" / "LABELS.json",
                {
                    "kind": "V4_CALIBRATION_LABELS",
                    "role": "development",
                    "bank_role_scope": "calibration",
                    "query_labels_read": False,
                    "reference_labels_read": False,
                    "config_hash": digest(config),
                    "source_hash": plan["source"]["sha256"],
                    "calibration_bank_count": 96,
                    "draws": 1024,
                    "prompt_ids": [f"p{i}" for i in range(72)],
                    "event_order": ["X", "S", "W", "I"],
                    "binding": {"tasks": tasks, "execution_kind": "REAL_CUDA_MODEL"},
                    "origin_id": origin,
                    "seed": seed,
                    "arm": "X_BASE",
                    "step": step,
                    "units": rows,
                },
            )
            finalize_run(original / "labels", {})
            signature = write(
                original / "signature" / "FEATURES.json",
                {
                    "kind": "V4_NATIVE_SIGNATURE_FEATURES",
                    "origin_id": origin,
                    "runtime_identity": {
                        "source_hash": plan["source"]["sha256"],
                        "config_hash": digest(config),
                    },
                },
            )
            finalize_run(original / "signature", {})
            originals.append({"labels": labels, "signature": signature})
            records.extend(rows)
    cvroot = root / "cv"
    dataset = write(
        cvroot / "DATASET.json",
        {
            "fixture": False,
            "config_hash": digest(config),
            "source_hash": plan["source"]["sha256"],
            "campaign_id": plan["campaign_id"],
            "original_bindings": originals,
            "records": records,
        },
    )
    foldroot = cvroot / "fold_41001"
    weights = write(foldroot / "model" / "COEFFICIENTS.npz", {"fixture_metadata_only": True})
    model = write(
        foldroot / "model" / "MODEL.json", {"kind": "V4_FROZEN_SIGNATURE_MODEL", "arrays": weights}
    )
    finalize_run(foldroot / "model", {})
    prediction = write(foldroot / "PREDICTIONS.npz", {"fixture_metadata_only": True})
    models = [
        {
            "id": "RAW4/full",
            "status": "AVAILABLE",
            "model": model,
            "predictions": prediction,
            "metrics": {"raw4_mse": 0.001},
            "fit_seconds": 0.01,
        }
    ]
    train = [i for i, row in enumerate(records) if row["seed"] != 41001]
    valid = [i for i, row in enumerate(records) if row["seed"] == 41001]
    shared = {
        "dataset": dataset,
        "train_seeds": [41002, 41003],
        "validation_seeds": [41001],
        "config_hash": digest(config),
        "campaign_id": plan["campaign_id"],
        "fixture": False,
        "query_labels_read": False,
        "reference_labels_read": False,
    }
    fit = write(
        foldroot / "FIT_RECEIPT.json",
        {
            **shared,
            "kind": "V4_SIGNATURE_CV_FITS",
            "training_indices": valid if invalid_indices else train,
            "models": [
                {k: v for k, v in row.items() if k not in {"predictions", "metrics"}}
                for row in models
            ],
        },
    )
    evaluation = write(
        foldroot / "EVALUATION_RECEIPT.json",
        {
            **shared,
            "kind": "V4_SIGNATURE_CV_EVALUATION",
            "validation_indices": valid,
            "evaluation_scope": "QUERY_REFERENCE"
            if invalid_scope
            else "HELDOUT_DEVELOPMENT_SEED_CALIBRATION_OBSERVATIONS",
            "models": [{k: v for k, v in row.items() if k != "model"} for row in models],
        },
    )
    fold = write(
        foldroot / "FOLD.json",
        {
            **shared,
            "schema": "ssvc-v4-whole-seed-cv-1",
            "kind": "V4_SIGNATURE_WHOLE_SEED_FOLD",
            "status": "COMPLETED",
            "training_rows": len(train),
            "validation_rows": len(valid),
            "fit_receipt": fit,
            "evaluation_receipt": evaluation,
            "models": models,
        },
    )
    finalize_run(foldroot, {})
    finalize_run(cvroot, {})
    return fold, plan, weights


def test_cv_receipt_verifies_original_calibration_rows_and_nested_weights(tmp_path):
    fold, plan, weights = cv_fixture(tmp_path)
    result = _verify_cv(ReceiptReader(), fold, plan)
    assert result["validation_seeds"] == [41001]
    Path(weights["path"]).write_bytes(b"changed actual coefficients")
    with pytest.raises(ValueError, match="SHA256"):
        _verify_cv(ReceiptReader(), fold, plan)


@pytest.mark.parametrize("kwargs", [{"invalid_scope": True}, {"invalid_indices": True}])
def test_cv_cannot_substitute_query_reference_or_training_on_heldout_seed(tmp_path, kwargs):
    fold, plan, _ = cv_fixture(tmp_path, **kwargs)
    with pytest.raises(ValueError, match="whole-seed split"):
        _verify_cv(ReceiptReader(), fold, plan)


def test_arbitrary_json_cannot_become_actual_cv_by_adding_bindings(tmp_path):
    fake = write(tmp_path / "unrelated.json", {"status": "COMPLETED"})
    with pytest.raises(ValueError, match="CV producer"):
        _verify_cv(ReceiptReader(), fake, {})
