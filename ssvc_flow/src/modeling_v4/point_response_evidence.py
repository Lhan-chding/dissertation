"""Actual frozen-calibration point evidence for the optional offline E stage.

This is a descriptive work-resolution check, never a reliability or coverage
certificate. It recomputes errors from frozen predictions and independently
measured reference statistics; a caller-supplied RESOLVED flag has no authority.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import platform
from pathlib import Path

import numpy as np

from .analysis_rules import load_analysis_rules, reference_precision
from .config import atomic_json, digest, validate_config
from .data_adapter import CONTRASTS
from .phase_evidence import ReceiptReader, verify_phase_evidence
from .phase_schedule import ROLES, validate_selection

WAITING = "WAITING_FOR_POINT_RESPONSE_EVIDENCE"


class _MissingOrigins(ValueError):
    pass


def _document(reader, value):
    return reader.json(value) if not isinstance(value, dict) or "path" in value else value


def _scope(row, task, prediction_hash):
    if (
        row.get("origin_id") != task["id"]
        or int(row.get("seed", -1)) != task["seed"]
        or row.get("role") != "calibration"
        or row.get("arm") != "X_BASE"
        or int(row.get("step", -1)) != task["step"]
        or int(row.get("n", -1)) != task["draws"]
    ):
        raise ValueError("Point evaluation row has a different source/role/origin/budget")
    if prediction_hash is not None and row.get("prediction_record_id") != prediction_hash:
        raise ValueError("Point evaluation does not bind its frozen prediction bytes")


def _method(row, selected):
    for model in selected["primary_models"]:
        if row.get("design_id") == model["id"]:
            if row.get("method") != model["model"]:
                raise ValueError("Frozen model ID was paired with another method")
            return model["id"]
        if (
            row.get("method") == model["model"] == "FULL_SCORE_JVP"
            and row.get("design_id") == "ORIGIN_FIXED_PROMPT_SUBSET"
        ):
            return model["id"]
    return None


def _bool(value):
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError("Actual boolean identity required")


def _origin(reader, plan, origin, selected, accumulators, rules):
    import pyarrow.parquet as pq

    task = next(t for t in plan["tasks"] if t["id"] == origin["task_id"])
    if (
        task.get("role") != "calibration"
        or task["seed"] not in ROLES["calibration"]
        or task["arm"] != "X_BASE"
        or task["step"] not in (32, 96)
        or task["calibration_banks"] != selected["m"]
        or task["query_banks"] != 24
        or task["prompts"] != 72
        or task["draws"] != 1024
    ):
        raise ValueError("Point qualification requires the complete frozen calibration design")
    fit = reader.json(origin["fit_receipt"])
    directory = Path(origin["fit_receipt"]["path"]).parent
    reader.completed_bundle(directory)
    binding = fit.get("binding", {})
    if (
        fit.get("kind") != "V4_RESPONSE_MAP_FIT"
        or fit.get("status") != "COMPLETED"
        or binding.get("selection_hash") != selected["selection_hash"]
        or binding.get("tasks") != reader.binding(origin["tasks_binding"])
        or binding.get("task") != origin["collection_receipt"]
        or binding.get("config_hash") != digest(plan["config"])
        or binding.get("source") != plan["source"]
        or binding.get("execution_kind") != "REAL_CUDA_MODEL"
        or fit.get("reference_read_after_predictions") is not True
        or fit.get("query_response_used_for_fit") is not False
    ):
        raise ValueError("Point evaluation must use the actual frozen model/config/source chain")
    complete = reader.json(origin["collection_receipt"])
    if complete.get("task") != task or complete.get("execution_kind") != "REAL_CUDA_MODEL":
        raise ValueError("Point source collection does not match the task")
    forks = reader.json(complete["forks"])
    banks = {b["bank_id"]: b for b in forks["banks"] if b["role"] == "query"}
    if len(banks) != 24:
        raise ValueError("Actual calibration evaluation must retain all 24 query banks")
    probes = plan["inputs"]["panels"]["observation"][:72]
    prompt_ids = [p["prompt_id"] for p in probes]
    if len(prompt_ids) != 72 or len(set(prompt_ids)) != 72:
        raise ValueError("Complete original 72-prompt identity required")
    pindex = {pid: i for i, pid in enumerate(prompt_ids)}
    prediction = reader.binding(fit["predictions"])
    metrics = reader.json(origin["evaluation_receipt"])
    if (
        metrics.get("kind") != "V4_EVALUATION"
        or metrics.get("binding") != {**binding, "predictions": prediction}
        or metrics.get("selection_modified") is not False
        or metrics.get("reference_read_scope") != "EVALUATION_ONLY"
    ):
        raise ValueError("Actual independent evaluation receipt required")
    diagnostic = reader.json(directory / "REFERENCE_DIAGNOSTICS.json")
    if diagnostic.get("rules") != rules or diagnostic.get("coverage_guarantee") is not False:
        raise ValueError("Reference precision was evaluated under different rules")
    reference_binding = reader.binding(directory / "REFERENCE_STATISTICS.npz")
    with np.load(reference_binding["path"], allow_pickle=False) as archive:
        references = {
            bid: {
                key: np.asarray(archive[f"{bid}__{key}"]) for key in ("estimate", "se", "resolved")
            }
            for bid in banks
        }
    aliases = {}
    for bid, bank in banks.items():
        values = references[bid]
        if any(array.shape != (72, len(CONTRASTS), 4) for array in values.values()):
            raise ValueError("Reference statistics have incomplete prompt/target/event axes")
        if values["resolved"].dtype.kind != "b":
            raise ValueError("Reference availability must remain explicit boolean data")
        for ti, (target, _, _) in enumerate(CONTRASTS):
            aliases[bid, target] = _bool(bank["contrasts"][target]["exact_inference_alias"])
            se = values["se"][:, ti]
            precision = reference_precision(se, exact_alias=aliases[bid, target], rules=rules)
            if np.any(values["resolved"][:, ti] & ~precision["reference_resolved"]):
                raise ValueError("Stored reference precision exceeds its frozen numerical rule")
    expected = {
        (bid, target, pid)
        for bid in banks
        for target, _, _ in CONTRASTS
        if not aliases[bid, target]
        for pid in prompt_ids
    }
    actual, evaluation = {k: {} for k in accumulators}, {k: {} for k in accumulators}
    core = reader.binding(directory / "evaluation" / "CORE_RESULTS.csv")
    with Path(core["path"]).open(newline="") as stream:
        for row in csv.DictReader(stream):
            model = _method(row, selected)
            if model is None or row.get("evaluation_unit") != "prompt":
                continue
            _scope(row, task, prediction["sha256"])
            if not _bool(row.get("reference_independent")):
                raise ValueError("Point reference is not actually independent")
            ids = json.loads(row["prompt_ids"])
            for pid in ids:
                key = row["bank_id"], row["target"], pid
                if key in evaluation[model]:
                    raise ValueError("Duplicated point evaluation identity")
                evaluation[model][key] = row["prediction_status"]
    for batch in pq.ParquetFile(prediction["path"]).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            model = _method(row, selected)
            if model is None or row.get("evaluation_unit") != "prompt":
                continue
            _scope(row, task, None)
            if row.get("query_response_used_for_fit") is not False:
                raise ValueError("Prediction used query response labels for fitting")
            bid, target = row["bank_id"], row["target"]
            if (bid, target) not in aliases:
                raise ValueError("Unregistered point query/target")
            if _bool(row["is_alias"]) != aliases[bid, target]:
                raise ValueError("Prediction and actual inference alias identity differ")
            ids = row["prompt_ids"]
            if len(ids) != len(set(ids)) or set(ids) - pindex.keys():
                raise ValueError("Point prediction prompt identity differs")
            values = row.get("prediction")
            if row["prediction_status"] == "UNKNOWN":
                if values is not None:
                    raise ValueError("Unknown prediction cannot be filled with zero")
                values = np.full((len(ids), 4), np.nan)
            elif row["prediction_status"] == "PREDICTED":
                values = np.asarray(values, dtype=float)
                if values.shape != (len(ids), 4):
                    raise ValueError("Point prediction must retain all four events")
            else:
                raise ValueError("Explicit actual prediction status required")
            ti = [v[0] for v in CONTRASTS].index(target)
            for pid, value in zip(ids, values, strict=True):
                key = bid, target, pid
                if key in actual[model]:
                    raise ValueError("Duplicated frozen point prediction")
                actual[model][key] = row["prediction_status"]
                if key not in expected:
                    continue
                stats, pi = references[bid], pindex[pid]
                truth, se, resolved = (stats[k][pi, ti] for k in ("estimate", "se", "resolved"))
                state = accumulators[model]
                finite = np.isfinite(value) & np.isfinite(truth) & np.isfinite(se)
                state["observed_cells"] += 4
                state["finite_cells"] += int(finite.sum())
                state["resolved_cells"] += int((resolved & finite).sum())
                state["errors"].extend(abs(value[finite] - truth[finite]).tolist())
                state["signal_cells"] += int(
                    (
                        finite
                        & resolved
                        & (
                            abs(truth)
                            > rules["offline_tracking_eligibility"][
                                "signal_reference_se_multiplier"
                            ]
                            * se
                        )
                    ).sum()
                )
    for model, state in accumulators.items():
        if any(actual[model].get(key) != value for key, value in evaluation[model].items()):
            raise ValueError("Frozen predictions and evaluated prediction statuses differ")
        missing = expected - actual[model].keys() | expected - evaluation[model].keys()
        state["missing_cells"] += len(missing) * 4
        state["expected_cells"] += len(expected) * 4
        state["nonalias_queries"] += len({key[:2] for key in expected})
        state["origins"].append(task["id"])
    return {
        "fit": reader.binding(origin["fit_receipt"]),
        "evaluation": reader.binding(origin["evaluation_receipt"]),
        "predictions": prediction,
        "reference_statistics": reference_binding,
        "core_results": core,
    }


def build_point_response_evidence(
    config, selection, calibration_evidence=None, *, out=None, fixture=False
):
    """Derive optional E eligibility solely from frozen calibration query originals."""
    config = validate_config(config)
    reader = ReceiptReader()
    selected = validate_selection(config, _document(reader, selection))
    rules = load_analysis_rules()
    criterion = rules["offline_tracking_eligibility"]
    result = {
        "schema": "ssvc-v4-point-response-evidence-1",
        "status": WAITING,
        "config_hash": digest(config),
        "selection_hash": selected["selection_hash"],
        "campaign_id": selected["campaign_id"],
        "execution_kind": "REAL_CUDA_MODEL",
        "fixture": bool(fixture),
        "analysis_rules": rules,
        "analysis_rules_hash": digest(rules),
        "criterion": criterion,
        "criterion_scope": criterion["criterion_scope"],
        "calibration_evidence": None,
        "assessments": [],
        "input_bindings": [],
        "test_labels_read": False,
        "online_control": False,
        "cross_seed_95": "NOT_CERTIFIED",
        "scientific_reliability_certified": False,
    }
    if calibration_evidence is None:
        result["missing_evidence"] = ["COMPLETE_FROZEN_CALIBRATION_POINT_EVALUATIONS"]
    else:
        if not fixture and (
            platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID", "").isdigit()
        ):
            raise RuntimeError("Actual point-response assessment must run on server Slurm CPU")
        result["calibration_evidence"] = (
            copy.deepcopy(calibration_evidence)
            if isinstance(calibration_evidence, dict)
            else str(calibration_evidence)
        )
        try:
            supplied = _document(reader, calibration_evidence)
            result["calibration_evidence"] = copy.deepcopy(supplied)
            expected_keys = {
                (seed, "X_BASE", step) for seed in ROLES["calibration"] for step in (32, 96)
            }
            supplied_keys = [(r["seed"], r["arm"], r["step"]) for r in supplied.get("origins", [])]
            if len(supplied_keys) != len(set(supplied_keys)) or set(supplied_keys) - expected_keys:
                raise ValueError(
                    "Duplicated or noncalibration origin cannot qualify offline tracking"
                )
            if set(supplied_keys) != expected_keys:
                raise _MissingOrigins("ALL_SIX_CALIBRATION_ORIGINS")
            evidence = verify_phase_evidence(supplied, selection=selected)
            if (
                evidence["phase"] != "D_CALIBRATION"
                or evidence["selection_hash"] != selected["selection_hash"]
                or evidence["campaign_id"] != selected["campaign_id"]
                or evidence.get("test_opened") is not False
            ):
                raise ValueError(
                    "Point qualification requires this campaign's frozen calibration evaluation"
                )
            plan = reader.json(evidence["tasks_binding"])
            if plan.get("analysis_rules") != rules or plan.get("analysis_rules_hash") != digest(
                rules
            ):
                raise ValueError(
                    "Operational E criterion was not frozen in the collection task manifest"
                )
            expected = {
                (seed, "X_BASE", step) for seed in ROLES["calibration"] for step in (32, 96)
            }
            origins = evidence["origins"]
            if len(origins) != 6 or {(r["seed"], r["arm"], r["step"]) for r in origins} != expected:
                result["missing_evidence"] = ["ALL_SIX_CALIBRATION_ORIGINS"]
            else:
                models = {m["id"]: m["model"] for m in selected["primary_models"]}
                states = {
                    k: {
                        "errors": [],
                        "observed_cells": 0,
                        "finite_cells": 0,
                        "resolved_cells": 0,
                        "missing_cells": 0,
                        "expected_cells": 0,
                        "signal_cells": 0,
                        "nonalias_queries": 0,
                        "origins": [],
                    }
                    for k in models
                }
                result["source"] = plan["source"]
                for origin in sorted(origins, key=lambda r: r["task_id"]):
                    result["input_bindings"].append(
                        _origin(
                            reader,
                            plan,
                            {**origin, "tasks_binding": evidence["tasks_binding"]},
                            selected,
                            states,
                            rules,
                        )
                    )
                for model, state in states.items():
                    errors = np.asarray(state.pop("errors"))
                    q95 = (
                        float(
                            np.quantile(
                                errors, criterion["pooled_four_event_absolute_error_quantile"]
                            )
                        )
                        if len(errors)
                        else None
                    )
                    complete = not state["missing_cells"]
                    resolved = (
                        complete
                        and state["expected_cells"] > 0
                        and state["finite_cells"] == state["expected_cells"]
                        and state["resolved_cells"] == state["expected_cells"]
                        and q95 is not None
                        and q95 <= criterion["maximum_absolute_error_quantile"]
                        and state["signal_cells"] > 0
                    )
                    result["assessments"].append(
                        {
                            "model_id": model,
                            "method": models[model],
                            **state,
                            "status": "RESOLVED"
                            if resolved
                            else "UNRESOLVED"
                            if complete
                            else "WAITING",
                            "reference_status": "RESOLVED"
                            if state["resolved_cells"] == state["expected_cells"]
                            and state["expected_cells"]
                            else "UNRESOLVED",
                            "working_resolution": rules["reference_precision"][
                                "primary_resolved_scale"
                            ],
                            "q95_absolute_error": q95,
                            "criterion": criterion["criterion_version"],
                        }
                    )
                statuses = {a["status"] for a in result["assessments"]}
                result["status"] = (
                    "POINT_RESPONSE_RESOLVED"
                    if "RESOLVED" in statuses
                    else WAITING
                    if "WAITING" in statuses
                    else "POINT_RESPONSE_UNRESOLVED"
                )
        except FileNotFoundError as exc:
            result["missing_evidence"] = ["ORIGINAL_FILE_MISSING:" + str(exc.filename)]
        except _MissingOrigins as exc:
            result["missing_evidence"] = [str(exc)]
    result["evidence_hash"] = digest(result)
    if out is not None:
        path = Path(out)
        if path.suffix != ".json":
            path /= "POINT_RESPONSE_EVIDENCE.json"
        if path.exists() and json.loads(path.read_text()) != result:
            raise FileExistsError("Preserve the first point-response assessment; use a new output")
        if not path.exists():
            atomic_json(path, result)
    return result


def verify_point_response_evidence(config, selection, evidence, *, fixture=False):
    """Reject handwritten flags by rebuilding every numerical eligibility field."""
    value = _document(ReceiptReader(), evidence)
    if (
        value.get("schema") != "ssvc-v4-point-response-evidence-1"
        or value.get("fixture") is not bool(fixture)
        or value.get("evidence_hash")
        != digest({k: v for k, v in value.items() if k != "evidence_hash"})
    ):
        raise ValueError("Actual derived point-response evidence with unchanged identity required")
    rebuilt = build_point_response_evidence(
        config, selection, value.get("calibration_evidence"), fixture=fixture
    )
    if rebuilt != value:
        raise ValueError(
            "Point-response eligibility does not reproduce actual independent evaluations"
        )
    return rebuilt
