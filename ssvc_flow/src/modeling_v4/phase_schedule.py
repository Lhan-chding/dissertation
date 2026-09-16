"""Pure D/E scheduling contracts; no model, filesystem, or scientific inference.

The runtime supplies normalized metadata from verified original receipts. A
``{path, sha256}`` here identifies that original; this module does not read or
verify its bytes. ``ssvc-v4-phase-evidence-1`` carries the sources, origin
collection/fit/evaluation bindings, and actual completed whole-seed CV records.
Passing raw collection alone cannot open D. No error threshold opens C -> D.
The caller publishes returned selection/calibration records once, without
overwriting an earlier freeze. Missing scientific evidence never becomes PASS.
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import PurePosixPath

from .config import digest, validate_config

ROLES = {
    "development": [41001, 41002, 41003],
    "calibration": [42001, 42002, 42003],
    "test": [43001, 43002, 43003, 43004, 43005, 43006],
}
BASELINES = ["ZERO", "DIRECT_MEASURE", "FULL_DUAL_RIDGE", "LOW_RANK_COMPRESSION"]
GRADIENTS = {"FULL_SCORE_JVP", "BASE_SCORE_JVP", "MIDPOINT_DIRECTIONAL"}


def _config(value):
    config = validate_config(value)
    q = config["qwen"]
    if q["seed_roles"] != ROLES:
        raise ValueError("D requires the twelve exact registered source seeds and roles")
    if q["steps"] != 128 or q["B"] != 4 or q["K"] != 8 or q["arms"] != ["X_BASE", "X_VALID"]:
        raise ValueError("D source matrix must remain two arms, 128 steps, B4/K8")
    if q["primary_origins"] != {"arm": "X_BASE", "steps": [32, 96]}:
        raise ValueError("Registered primary origins changed")
    if q["shift_origins"] != {"arm": "X_VALID", "step": 96, "seeds": "test"}:
        raise ValueError("Registered shift origins changed")
    if q["calibration_banks_development"] != 96 or q["query_banks"] != 24:
        raise ValueError("D requires 96 development banks and 24 query banks")
    if (
        q["observation_panel"]["prompts"] != 72
        or config["observations"]["primary_n"] != 1024
        or q["continuous_track"]["anchor"] != 64
        or q["continuous_track"]["horizons"] != [1, 2, 4, 8, 16]
    ):
        raise ValueError("Registered D observation or E tracking scope changed")
    return config


def _binding(value):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("path"), str)
        or not PurePosixPath(value["path"]).is_absolute()
        or re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", "")) is None
    ):
        raise ValueError("Original receipt needs an absolute path and SHA256 binding")
    return copy.deepcopy(value)


def _source_id(seed, arm):
    return f"source_{seed}_{arm}"


def _origin_key(row):
    return row["seed"], row["arm"], row["step"]


def registered_matrix(config):
    config = _config(config)
    sources, origins = [], []
    for role, seeds in ROLES.items():
        for seed in seeds:
            for arm in ("X_BASE", "X_VALID"):
                sources.append(
                    {
                        "id": _source_id(seed, arm),
                        "seed": seed,
                        "arm": arm,
                        "role": role,
                        "steps": 128,
                        "B": 4,
                        "K": 8,
                        "paired_initialization_key": f"seed_{seed}",
                        "rollout_policy": "own_arm",
                        "retain_full_steps": [32, 64, 96]
                        if role == "development" and arm == "X_BASE"
                        else [32, 96],
                        "retain_latest_resume": True,
                        "retain_light_steps": list(range(64, 81)) if role == "development" else [],
                    }
                )
            for arm, step in [("X_BASE", 32), ("X_BASE", 96)] + (
                [("X_VALID", 96)] if role == "test" else []
            ):
                origins.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "step": step,
                        "role": role,
                        "source_task": _source_id(seed, arm),
                        "report_scope": "shift" if arm == "X_VALID" else "primary",
                    }
                )
    return {
        "sources": sources,
        "origins": origins,
        "source_optimizer_updates": 24 * 128,
        "source_training_outputs": 24 * 128 * 4 * 8,
    }


def _evidence(config, receipt, phase, sources, origins, counts, *, campaign_id=None):
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "ssvc-v4-phase-evidence-1"
        or receipt.get("phase") != phase
        or receipt.get("status") != "COMPLETE"
        or receipt.get("execution_kind") != "REAL_CUDA_MODEL"
        or receipt.get("fixture", False)
        or receipt.get("config_hash") != digest(config)
    ):
        raise ValueError("Complete actual V4 phase evidence with matching config required")
    if not receipt.get("campaign_id") or (campaign_id and receipt["campaign_id"] != campaign_id):
        raise ValueError("Phase evidence campaign identity differs")
    if receipt.get("test_opened") is not False:
        raise ValueError("Selection/calibration evidence cannot have opened the locked test")
    actual_sources = receipt.get("sources", [])
    if len(actual_sources) != len(sources) or {
        (r.get("seed"), r.get("arm")) for r in actual_sources
    } != set(sources):
        raise ValueError("Completed source matrix is incomplete or duplicated")
    for row in actual_sources:
        if row.get("steps") != 128 or row.get("status") != "COMPLETED":
            raise ValueError("All source trajectories must finish 128 actual steps")
        _binding(row.get("receipt"))
    actual_origins = receipt.get("origins", [])
    keys = [(r.get("seed"), r.get("arm"), r.get("step")) for r in actual_origins]
    if len(keys) != len(origins) or set(keys) != set(origins):
        raise ValueError("Completed fitted/evaluated origin matrix is incomplete or duplicated")
    allowed = set(config["models"]["primary"] + config["models"]["secondary"])
    for row in actual_origins:
        if (
            tuple(row.get(k) for k in ("calibration_banks", "query_banks", "prompts", "draws"))
            != counts
        ):
            raise ValueError("Origin bank/panel/draw scope differs from the registered phase")
        if (
            row.get("collection_status") != "COMPLETED"
            or row.get("evaluation_status") != "COMPLETE"
            or row.get("all_queries_retained") is not True
            or row.get("fit_scope") != "calibration_banks"
            or row.get("evaluation_scope") != "heldout_query_banks"
        ):
            raise ValueError("Actual fits and heldout evaluations are required for every origin")
        models = row.get("fitted_model_ids", [])
        if (
            not models
            or any(not isinstance(m, str) or not m for m in models)
            or not set(models) & (allowed - {"ZERO", "DIRECT_MEASURE"})
        ):
            raise ValueError("Raw acquisition/direct counts do not replace fitted model evaluation")
        for field in ("collection_receipt", "fit_receipt", "evaluation_receipt"):
            _binding(row.get(field))
    return copy.deepcopy(receipt)


def validate_first_map_completion(config, receipt):
    config = _config(config)
    seeds = ROLES["development"][:2]
    result = _evidence(
        config,
        receipt,
        "C",
        [(s, "X_BASE") for s in seeds],
        [(s, "X_BASE", step) for s in seeds for step in (32, 96)],
        (32, 16, 24, 256),
    )
    _binding(result.get("bridge_receipt"))
    return result


def _workers(config, workers):
    if workers not in (2, 3) or workers > config["operations"]["max_concurrent_gpus"]:
        raise ValueError("Use two or three registered independent workers")


def _extension(config, first_map, role, workers, *, m, selection=None, calibration=None):
    _workers(config, workers)
    matrix = registered_matrix(config)
    tasks = []
    initial_sources = {(row["seed"], row["arm"]): row for row in first_map["sources"]}
    barrier = {
        "development": "C_FIRST_MAP_COMPLETE",
        "calibration": "D_SELECTION_FROZEN",
        "test": "D_CALIBRATION_FROZEN",
    }[role]
    for i, source in enumerate(r for r in matrix["sources"] if r["role"] == role):
        task = {
            **source,
            "stage": "D",
            "kind": "source",
            "worker": i % workers,
            "depends_on": [barrier],
        }
        if (source["seed"], source["arm"]) in initial_sources:
            task["reuse_completed_receipt"] = initial_sources[source["seed"], source["arm"]][
                "receipt"
            ]
        tasks.append(task)
    by_id = {task["id"]: task for task in tasks}
    for origin in (row for row in matrix["origins"] if row["role"] == role):
        seed, arm, step = _origin_key(origin)
        task = {
            **origin,
            "id": f"D_map_{seed}_{arm}_{step}",
            "stage": "D",
            "kind": "map",
            "worker": by_id[origin["source_task"]]["worker"],
            "depends_on": [barrier, origin["source_task"]],
            "calibration_banks": m,
            "query_banks": 24,
            "prompts": 72,
            "draws": 1024,
            "nested_draw_counts": [256, 1024, 4096],
            "reference_draws": 4096,
            "reference_rng_independent": True,
            "selection_uses_query_labels": False,
            "bank_order_information": "training_metadata_and_update_geometry",
            "nested_bank_prefixes": [24, 32, 48, 96] if role == "development" else [m],
            "evaluation_only": role == "test",
            "shared_raw_across_models": True,
            "mandatory_baselines": copy.deepcopy(BASELINES),
        }
        if role == "development":
            task["n_curve_max_draws"] = 4096
            if seed == 41001 and arm == "X_BASE" and step == 32:
                task["reference_validation"] = {
                    "draws": 8192,
                    "bank_ids": ["query_000", "query_001"],
                    "prompt_scope": "all_frozen_observation_prompts",
                }
            task["directional_bank_ids"] = [f"calibration_{i:03d}" for i in range(4)]
            task["finite_difference_steps"] = [0.25, 0.5, 1.0]
            task["finite_difference_scope"] = "central_2h_relative_to_actual_update_e"
            task["finite_difference_purpose"] = "engineering_diagnostic_not_AD_replacement"
        if selection:
            task["bank_selector"] = selection["bank_selector"]
            task["draws"] = selection["draw_count"]
        if role == "development" and seed in ROLES["development"][:2]:
            task["reuse_prefix_task"] = f"map_{seed}_{arm}_{step}"
            task["reuse_prefix_scope"] = {
                "calibration_banks": 32,
                "query_banks": 16,
                "prompts": 24,
                "draws": 256,
            }
            task["prefix_reuse_requires_original_identity_verification"] = True
        tasks.append(task)
    result = {
        "schema": "ssvc-v4-phase-extension-1",
        "phase": "D",
        "role": role,
        "status": "PLANNED_NOT_EXECUTED",
        "version": config["version"],
        "config_hash": digest(config),
        "campaign_id": first_map["campaign_id"],
        "first_map_evidence_hash": digest(first_map),
        "first_map_evidence": copy.deepcopy(first_map),
        "tasks": tasks,
        "registered_matrix": matrix,
        "workers": workers,
        "barrier": {"id": barrier, "requires_runtime_evidence_verification": True},
        "calibration_used_as_model_training_data": False,
        "scientific_qualification": "NOT_INFERRED",
        "online_control": False,
        "planning_is_not_an_execution_receipt": True,
    }
    if selection:
        result["selection_hash"] = selection["selection_hash"]
        result["selection"] = copy.deepcopy(selection)
    if calibration:
        result["calibration_hash"] = calibration["calibration_hash"]
        result["calibration"] = copy.deepcopy(calibration)
    result["extension_hash"] = digest(result)
    return result


def build_development_extension(config, first_map_receipt, *, workers=2):
    config = _config(config)
    first = validate_first_map_completion(config, first_map_receipt)
    return _extension(config, first, "development", workers, m=96)


def _whole_seed_cv(config, receipt):
    dev = set(ROLES["development"])
    folds = receipt.get("whole_seed_cv", [])
    heldout = []
    if len(folds) != 3:
        raise ValueError("Three actual whole-seed development validation folds required")
    for fold in folds:
        train, validation = fold.get("train_seeds", []), fold.get("validation_seeds", [])
        if (
            len(validation) != 1
            or len(train) != 2
            or len(set(train)) != 2
            or set(train) & set(validation)
            or set(train) | set(validation) != dev
        ):
            raise ValueError("Whole-seed CV cannot mix heldout/test/calibration seeds into fitting")
        heldout += validation
        _binding(fold.get("receipt"))
    if set(heldout) != dev:
        raise ValueError("Each development seed must be held out exactly once")
    return copy.deepcopy(folds)


def _models(config, primary_models, gradient_reference):
    if not isinstance(primary_models, list) or len(primary_models) != 2:
        raise ValueError("Freeze exactly two primary representation/model specifications")
    allowed = set(config["models"]["primary"] + config["models"]["secondary"])
    for model in primary_models:
        if (
            not isinstance(model, dict)
            or not isinstance(model.get("id"), str)
            or not model["id"]
            or model.get("model") not in allowed - {"ZERO", "DIRECT_MEASURE"}
            or model.get("representation") not in config["representations"]
            or not isinstance(model.get("settings"), dict)
        ):
            raise ValueError("Complete frozen model ID/representation/settings required")
    if (
        len({row["id"] for row in primary_models}) != 2
        or len({digest({k: v for k, v in row.items() if k != "id"}) for row in primary_models}) != 2
    ):
        raise ValueError("The two frozen models must be distinct")
    if gradient_reference not in GRADIENTS:
        raise ValueError("Freeze one registered full directional gradient reference")


def freeze_selection(
    config,
    first_map_receipt,
    development_receipt,
    *,
    m,
    primary_models,
    gradient_reference,
    bank_selector="BLOCK_PIVOT_QR",
):
    config = _config(config)
    first = validate_first_map_completion(config, first_map_receipt)
    if type(m) is not int or m not in (24, 32, 48, 96):
        raise ValueError("Freeze exactly one m from 24/32/48/96")
    if bank_selector not in {"STRATIFIED_RANDOM", "BLOCK_PIVOT_QR"}:
        raise ValueError("Freeze one registered calibration bank selector")
    dev = ROLES["development"]
    evidence = _evidence(
        config,
        development_receipt,
        "D_DEVELOPMENT",
        [(s, arm) for s in dev for arm in ("X_BASE", "X_VALID")],
        [(s, "X_BASE", step) for s in dev for step in (32, 96)],
        (96, 24, 72, 1024),
        campaign_id=first["campaign_id"],
    )
    folds = _whole_seed_cv(config, evidence)
    _models(config, primary_models, gradient_reference)
    result = {
        "schema": "ssvc-v4-selection-1",
        "status": "FROZEN",
        "version": config["version"],
        "config_hash": digest(config),
        "campaign_id": first["campaign_id"],
        "m": m,
        "bank_selector": bank_selector,
        "draw_count": config["observations"]["primary_n"],
        "primary_models": copy.deepcopy(primary_models),
        "gradient_reference": gradient_reference,
        "mandatory_baselines": copy.deepcopy(BASELINES),
        "compression_ranks": [2, 8, 32, "FULL"],
        "fit_seed_role": "development",
        "whole_seed_cv": folds,
        "development_evidence_hash": digest(evidence),
        "development_evidence": copy.deepcopy(evidence),
        "first_map_evidence_hash": digest(first),
        "test_opened": False,
        "model_parameters_use_calibration_seeds": False,
        "cross_seed_95": "NOT_CERTIFIED",
        "online_control": False,
    }
    result["selection_hash"] = digest(result)
    return result


def _frozen(config, value, schema, hash_field):
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("status") != "FROZEN"
        or value.get("config_hash") != digest(config)
        or value.get("version") != config["version"]
    ):
        raise ValueError("Frozen record config/version/status changed")
    if value.get(hash_field) != digest({k: v for k, v in value.items() if k != hash_field}):
        raise ValueError("Frozen record content hash changed")
    return copy.deepcopy(value)


def validate_selection(config, selection):
    config = _config(config)
    result = _frozen(config, selection, "ssvc-v4-selection-1", "selection_hash")
    if (
        type(result.get("m")) is not int
        or result["m"] not in (24, 32, 48, 96)
        or result.get("bank_selector") not in {"STRATIFIED_RANDOM", "BLOCK_PIVOT_QR"}
        or result.get("draw_count") != config["observations"]["primary_n"]
        or result.get("test_opened") is not False
        or result.get("model_parameters_use_calibration_seeds") is not False
        or result.get("mandatory_baselines") != BASELINES
        or result.get("compression_ranks") != [2, 8, 32, "FULL"]
        or result.get("fit_seed_role") != "development"
        or result.get("online_control") is not False
    ):
        raise ValueError("Frozen selection lost independent test or nondegenerate baselines")
    _whole_seed_cv(config, result)
    _models(config, result.get("primary_models"), result.get("gradient_reference"))
    evidence = result.get("development_evidence")
    if not isinstance(evidence, dict) or digest(evidence) != result.get(
        "development_evidence_hash"
    ):
        raise ValueError("Frozen selection lost its actual development evidence")
    dev = ROLES["development"]
    _evidence(
        config,
        evidence,
        "D_DEVELOPMENT",
        [(s, arm) for s in dev for arm in ("X_BASE", "X_VALID")],
        [(s, "X_BASE", step) for s in dev for step in (32, 96)],
        (96, 24, 72, 1024),
        campaign_id=result["campaign_id"],
    )
    return result


def freeze_calibration(config, selection, calibration_receipt, *, thresholds):
    config = _config(config)
    selected = validate_selection(config, selection)
    cal = ROLES["calibration"]
    evidence = _evidence(
        config,
        calibration_receipt,
        "D_CALIBRATION",
        [(s, arm) for s in cal for arm in ("X_BASE", "X_VALID")],
        [(s, "X_BASE", step) for s in cal for step in (32, 96)],
        (selected["m"], 24, 72, 1024),
        campaign_id=selected["campaign_id"],
    )
    if evidence.get("selection_hash") != selected["selection_hash"]:
        raise ValueError("Calibration must evaluate the previously frozen selection")
    if (
        not isinstance(thresholds, dict)
        or not thresholds
        or any(
            not isinstance(k, str)
            or not k
            or isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v < 0
            for k, v in thresholds.items()
        )
    ):
        raise ValueError("Calibration thresholds must be named finite nonnegative values")
    result = {
        "schema": "ssvc-v4-calibration-freeze-1",
        "status": "FROZEN",
        "version": config["version"],
        "config_hash": digest(config),
        "campaign_id": selected["campaign_id"],
        "selection_hash": selected["selection_hash"],
        "calibration_evidence_hash": digest(evidence),
        "calibration_evidence": copy.deepcopy(evidence),
        "calibration_seeds": cal.copy(),
        "thresholds": copy.deepcopy(thresholds),
        "threshold_fit_role": "calibration",
        "model_parameters_refitted": False,
        "test_opened": False,
        "cross_seed_95": "NOT_CERTIFIED",
    }
    result["calibration_hash"] = digest(result)
    return result


def build_confirmation_extension(
    config, first_map_receipt, selection, *, role="calibration", workers=2, calibration=None
):
    config = _config(config)
    first = validate_first_map_completion(config, first_map_receipt)
    selected = validate_selection(config, selection)
    if selected["campaign_id"] != first["campaign_id"] or selected[
        "first_map_evidence_hash"
    ] != digest(first):
        raise ValueError("Selection/first-map campaign identity differs")
    if role not in ("calibration", "test"):
        raise ValueError("Confirmation role must be calibration or test")
    checked = None
    if role == "test":
        if calibration is None:
            raise ValueError("Test requires completed calibration and frozen empirical rules")
        checked = _frozen(config, calibration, "ssvc-v4-calibration-freeze-1", "calibration_hash")
        if (
            checked.get("selection_hash") != selected["selection_hash"]
            or checked.get("campaign_id") != first["campaign_id"]
            or checked.get("calibration_seeds") != ROLES["calibration"]
            or checked.get("test_opened") is not False
            or checked.get("model_parameters_refitted") is not False
            or checked.get("cross_seed_95") != "NOT_CERTIFIED"
        ):
            raise ValueError("Calibration freeze differs from the fixed empirical-only protocol")
        reconstructed = freeze_calibration(
            config,
            selected,
            checked.get("calibration_evidence"),
            thresholds=checked.get("thresholds"),
        )
        if reconstructed != checked:
            raise ValueError("Calibration freeze no longer reproduces its original evidence")
    return _extension(
        config, first, role, workers, m=selected["m"], selection=selected, calibration=checked
    )


def build_tracking_extension(config, selection, point_response_receipt, *, workers=2):
    config = _config(config)
    selected = validate_selection(config, selection)
    _workers(config, workers)
    result = {
        "schema": "ssvc-v4-phase-extension-1",
        "phase": "E",
        "role": "development",
        "config_hash": digest(config),
        "version": config["version"],
        "campaign_id": selected["campaign_id"],
        "selection_hash": selected["selection_hash"],
        "selection": copy.deepcopy(selected),
        "status": "SKIPPED_POINT_RESPONSE_UNRESOLVED",
        "tasks": [],
        "preserve_source_checkpoints": list(range(64, 81)),
        "online_control": False,
        "planning_is_not_an_execution_receipt": True,
        "workers": workers,
    }
    if point_response_receipt is None:
        result["status"] = "WAITING_FOR_POINT_RESPONSE_EVIDENCE"
        result["reason"] = "No actual point-response resolution evidence supplied"
    else:
        from .point_response_evidence import verify_point_response_evidence

        receipt = verify_point_response_evidence(config, selected, point_response_receipt)
        if (
            receipt.get("schema") != "ssvc-v4-point-response-evidence-1"
            or receipt.get("config_hash") != digest(config)
            or receipt.get("selection_hash") != selected["selection_hash"]
            or receipt.get("campaign_id") != selected["campaign_id"]
            or receipt.get("execution_kind") != "REAL_CUDA_MODEL"
            or receipt.get("fixture", False)
        ):
            raise ValueError("Actual point-response evidence must match the frozen campaign")
        result["point_response_evidence_hash"] = digest(receipt)
        result["point_response_evidence"] = copy.deepcopy(receipt)
        if receipt.get("status") == "WAITING_FOR_POINT_RESPONSE_EVIDENCE":
            result["status"] = "WAITING_FOR_POINT_RESPONSE_EVIDENCE"
            result["reason"] = "Actual frozen calibration point evaluation is incomplete"
        elif receipt.get("status") == "POINT_RESPONSE_UNRESOLVED":
            result["reason"] = (
                "Actual point responses remain unresolved; retain paths without E measurement"
            )
        elif receipt.get("status") != "POINT_RESPONSE_RESOLVED":
            raise ValueError("Explicit resolved or unresolved point-response status required")
        else:
            allowed = {m["id"] for m in selected["primary_models"]} | {
                selected["gradient_reference"]
            }
            resolved = []
            for assessment in receipt.get("assessments", []):
                if assessment.get("status") != "RESOLVED":
                    continue
                scale = assessment.get("working_resolution")
                if (
                    assessment.get("model_id") not in allowed
                    or assessment.get("reference_status") != "RESOLVED"
                    or type(assessment.get("nonalias_queries")) is not int
                    or assessment["nonalias_queries"] <= 0
                    or isinstance(scale, bool)
                    or not isinstance(scale, (int, float))
                    or not math.isfinite(scale)
                    or scale <= 0
                    or not isinstance(assessment.get("criterion"), str)
                    or not assessment["criterion"]
                ):
                    raise ValueError(
                        "Resolution needs an actual nonalias model/reference assessment"
                    )
                resolved.append(assessment["model_id"])
            if not resolved:
                raise ValueError("No supported actual resolved point-response assessment")
            result["status"] = "PLANNED_NOT_EXECUTED"
            result["resolved_model_ids"] = sorted(set(resolved))
            for i, seed in enumerate(ROLES["development"]):
                result["tasks"].append(
                    {
                        "id": f"E_track_{seed}_X_BASE_64_80",
                        "stage": "E",
                        "kind": "offline_track",
                        "seed": seed,
                        "arm": "X_BASE",
                        "role": "development",
                        "worker": i % workers,
                        "source_task": _source_id(seed, "X_BASE"),
                        "depends_on": [_source_id(seed, "X_BASE"), "D_POINT_RESPONSE_RESOLVED"],
                        "anchor": 64,
                        "checkpoints": list(range(64, 81)),
                        "horizons": [1, 2, 4, 8, 16],
                        "required_anchor_state": "full_parameters_Adam_RNG_at_step64",
                        "fixed_anchor_basis": True,
                        "model_frozen_at_anchor": True,
                        "reference_rng_independent": True,
                        "online_control": False,
                        "derivative_refresh_steps": [68, 72, 76, 80],
                        "refresh_uses_intermediate_measurements": True,
                        "report_endpoint_and_window_max_error": True,
                        "report_unknown_count": True,
                        "unmeasured_extrapolation_claim_for_refreshed_method": False,
                        "selection_hash": selected["selection_hash"],
                    }
                )
    result["extension_hash"] = digest(result)
    return result


def append_phase_extension(task_list, extension):
    """Return an immutable child manifest; runtime verifies original evidence once.

    Existing task dictionaries, campaign ID and worker assignments stay intact.
    Publication and active-job checks belong to the runtime/Slurm submitter.
    """
    if task_list.get("schema") != "ssvc-v4-task-list-1" or task_list.get(
        "task_list_hash"
    ) != digest({k: v for k, v in task_list.items() if k != "task_list_hash"}):
        raise ValueError("Parent task manifest identity changed")
    if extension.get("schema") != "ssvc-v4-phase-extension-1" or extension.get(
        "extension_hash"
    ) != digest({k: v for k, v in extension.items() if k != "extension_hash"}):
        raise ValueError("Phase extension identity changed")
    if (
        extension.get("config_hash") != digest(task_list["config"])
        or extension.get("campaign_id") != task_list["campaign_id"]
        or extension.get("workers") != task_list["workers"]
    ):
        raise ValueError("Phase extension config/campaign/workers identity differs")
    extension = copy.deepcopy(extension)
    maps = [task for task in extension["tasks"] if task.get("kind") == "map"]
    if maps:
        from ..r4_inputs import STRATA

        prompts = task_list.get("inputs", {}).get("panels", {}).get("observation", [])
        for task in maps:
            groups = {}
            for prompt in prompts[: task["prompts"]]:
                groups.setdefault((prompt["family"], prompt["interface"]), prompt["prompt_id"])
            if set(groups) != set(STRATA):
                raise ValueError("Derivative probes require the six actual frozen prompt strata")
            task["derivative_prompt_subset"] = [groups[group] for group in sorted(groups)]
        extension.pop("extension_hash")
        extension["extension_hash"] = digest(extension)
    existing_extensions = task_list.get("phase_extensions", [])
    if any(e["extension_hash"] == extension["extension_hash"] for e in existing_extensions):
        raise ValueError("Phase extension is already registered")
    result = copy.deepcopy(task_list)
    existing = {row["id"]: row for row in result["tasks"]}
    if len(existing) != len(result["tasks"]):
        raise ValueError("Parent task IDs are duplicated")
    for task in extension["tasks"]:
        previous = existing.get(task["id"])
        if previous:
            if (
                task.get("kind") != "source"
                or previous.get("kind") != "source"
                or not task.get("reuse_completed_receipt")
                or any(task.get(k) != previous.get(k) for k in ("seed", "arm"))
            ):
                raise ValueError("Extension would rewrite an existing task")
            continue
        result["tasks"].append(copy.deepcopy(task))
        existing[task["id"]] = task
    result["phase_extensions"] = [*copy.deepcopy(existing_extensions), copy.deepcopy(extension)]
    result["stages"] = list(dict.fromkeys([*result["stages"], extension["phase"]]))
    result["parent_task_list_hash"] = task_list["task_list_hash"]
    result.pop("task_list_hash")
    result["task_list_hash"] = digest(result)
    return result
