"""Artifact-level select/fit/evaluate commands and reproducible CPU receipts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .io import (
    atomic_bytes,
    atomic_json,
    atomic_npz,
    canonical_hash,
    finalize_run,
    sha256_file,
    source_identity,
)
from .statistics import error_metrics, reference_corrected_mse, risk_coverage


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_artifact(path):
    path = Path(path)
    spec = json.loads(path.read_text())
    if spec.get("requires_server_cpu") and (
        platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID")
    ):
        raise ValueError("production response artifacts require server CPU through Slurm")
    arrays_path = Path(spec["arrays"]["path"])
    if not arrays_path.is_absolute():
        arrays_path = path.parent / arrays_path
    if sha256_file(arrays_path) != spec["arrays"]["sha256"]:
        raise ValueError("artifact arrays hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as handle:
        arrays = {key: handle[key].copy() for key in handle.files}
    return (
        spec,
        arrays,
        {"spec_sha256": sha256_file(path), "arrays_sha256": sha256_file(arrays_path)},
    )


def array_identity(value):
    value = np.ascontiguousarray(value)
    return canonical_hash(
        {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    )


def _verify_acceptance_receipt(config, spec, prediction, accepted, spec_path):
    binding = spec.get("acceptance_receipt")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("accepted predictions require a bound acceptance receipt")
    path = Path(binding["path"])
    if not path.is_absolute():
        path = Path(spec_path).parent / path
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("acceptance receipt hash mismatch")
    receipt = json.loads(path.read_text())
    expected = {
        "schema": "ssvc-v3-acceptance-receipt-1",
        "status": "ACCEPTANCE_FROZEN",
        "config_sha256": canonical_hash(config),
        "source_sha256": source_identity()["sha256"],
        "predictions_sha256": array_identity(prediction),
        "accepted_sha256": array_identity(accepted),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("acceptance receipt prediction/mask/config/source binding mismatch")
    if receipt.get("fixture") or not receipt.get("calibration_receipt"):
        raise ValueError("acceptance receipt needs nonfixture calibration provenance")
    calibration_binding = receipt["calibration_receipt"]
    calibration_path = Path(calibration_binding["path"])
    if not calibration_path.is_absolute():
        calibration_path = path.parent / calibration_path
    if sha256_file(calibration_path) != calibration_binding["sha256"]:
        raise ValueError("acceptance calibration receipt hash mismatch")
    calibration = json.loads(calibration_path.read_text())
    if (
        calibration.get("schema") != "ssvc-v3-cpu-calibration-1"
        or calibration.get("status") != "CALIBRATION_ANALYZED"
        or calibration.get("fixture")
    ) or any(calibration.get(key) != expected[key] for key in ("config_sha256", "source_sha256")):
        raise ValueError("acceptance calibration provenance mismatch")
    selection_binding = receipt.get("selection_lock")
    if not isinstance(selection_binding, dict):
        raise ValueError("acceptance receipt needs its frozen selection lock")
    selection_path = Path(selection_binding["path"])
    if not selection_path.is_absolute():
        selection_path = path.parent / selection_path
    if sha256_file(selection_path) != selection_binding["sha256"]:
        raise ValueError("acceptance selection lock hash mismatch")
    from .cpu_results import _verify_calibration
    from .schema import verify_selection_lock

    lock = verify_selection_lock(config, selection_path)
    _verify_calibration(config, lock, calibration_path, fixture=False)
    envelope = calibration["designs"].get(receipt.get("design_id"), {})
    radius, tolerance = envelope.get("radius"), envelope.get("tolerance")
    if (
        type(radius) not in (int, float)
        or type(tolerance) not in (int, float)
        or not 0 <= radius <= tolerance
    ):
        raise ValueError("accepted predictions lack a resolved calibration tolerance")
    return {
        "receipt": {"path": str(path.resolve()), "sha256": binding["sha256"]},
        "validation_scope": "HASH_BOUND_UPSTREAM_ACCEPTANCE; no recalibration performed here",
    }


def run_tests(out, selectors):
    if platform.system() != "Linux":
        raise ValueError("full V3 CPU acceptance is assigned to the server")
    out = Path(out).resolve()
    root = Path(__file__).resolve().parents[2]
    if out.exists():
        raise FileExistsError(out)
    command = [
        sys.executable,
        str(root / "scripts/verify_modeling_v3_cpu.py"),
        "--out",
        str(out),
        "--",
        *selectors,
    ]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=root, check=False)
    return {
        "status": "TESTS_PASSED" if result.returncode == 0 else "TESTS_FAILED",
        "exit_code": result.returncode,
        "wall_seconds": time.perf_counter() - started,
        "scientific_status": "NOT_EVALUATED",
        "out": str(out),
    }


def run_artifact_command(command, config, inputs, out):
    if command == "summarize":
        return summarize(config, inputs, out)
    spec, arrays, input_binding = _load_artifact(inputs[0])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if command == "select":
        from .bank_selection import select_banks

        result = select_banks(
            arrays["updates"],
            spec["n_banks"],
            spec["method"],
            bank_ids=spec.get("bank_ids"),
            strata=spec.get("strata"),
            seed=spec.get("seed", 2026091500),
            logdet_ridge=spec.get("logdet_ridge", 1e-8),
        )
        atomic_npz(out / "SELECTED_UPDATES.npz", {"selected_updates": result["selected_updates"]})
        atomic_json(
            out / "SELECTION.json",
            _jsonable(
                {
                    **{key: value for key, value in result.items() if key != "selected_updates"},
                    "selected_updates_payload": {
                        "path": "SELECTED_UPDATES.npz",
                        "sha256": sha256_file(out / "SELECTED_UPDATES.npz"),
                    },
                }
            ),
        )
        status = "GEOMETRY_SELECTED_NO_QUERY_LABELS"
    elif command == "fit":
        from .response_models import fit_response_model

        model = fit_response_model(
            arrays["updates"],
            arrays["responses"],
            spec["method"],
            rank_cap=spec.get("rank_cap", "FULL"),
            alpha=spec.get("alpha", 1e-5),
            covariance=arrays.get("covariance"),
            output_policy=spec.get("output_policy", "RAW4"),
            regression=spec.get("regression", "RIDGE"),
            group_map=arrays.get("group_map"),
            random_seed=spec.get("seed", 0),
        )
        # Keep the fitted basis/coefficients, input calibration, specification,
        # and optional predictions. No arbitrary Python objects are serialized.
        model_arrays = {key: value for key, value in model.items() if isinstance(value, np.ndarray)}
        if "query_updates" in arrays:
            model_arrays["predictions"] = model["predict"](arrays["query_updates"])
        atomic_npz(out / "MODEL_ARRAYS.npz", model_arrays)
        atomic_json(
            out / "MODEL.json",
            _jsonable(
                {
                    **{
                        key: value
                        for key, value in model.items()
                        if not isinstance(value, np.ndarray) and not callable(value)
                    },
                    "fit_specification": spec,
                    "input_binding": input_binding,
                    "array_payload": {
                        "path": "MODEL_ARRAYS.npz",
                        "sha256": sha256_file(out / "MODEL_ARRAYS.npz"),
                        "members": {
                            key: {"shape": list(value.shape), "dtype": value.dtype.str}
                            for key, value in model_arrays.items()
                        },
                    },
                }
            ),
        )
        status = model["status"]
    elif command == "evaluate":
        prediction, truth = arrays["predictions"], arrays["reference"]
        if prediction.shape != truth.shape or prediction.ndim < 2 or prediction.shape[-1] != 4:
            raise ValueError("aligned raw four-event prediction/reference arrays required")
        resolved = np.isfinite(prediction).all(axis=-1) & np.isfinite(truth).all(axis=-1)
        known = int(resolved.sum())
        result = {
            "all_cases": int(resolved.size),
            "resolved_cases": known,
            "unknown_cases": int(resolved.size - known),
            "unknown_is_safe": False,
            "reference_kind": spec.get("reference_kind", "ESTIMATED"),
            "events": {},
        }
        if known:
            for index, event in enumerate(("X", "S", "W", "I")):
                result["events"][event] = error_metrics(
                    prediction[..., index][resolved], truth[..., index][resolved]
                )
            result["delta_v"] = error_metrics(
                -prediction[resolved][..., 3], -truth[resolved][..., 3]
            )
            result["delta_v_definition"] = "-delta_pI"
            result["raw_valid_event_sum_diagnostic"] = error_metrics(
                prediction[resolved][..., :3].sum(-1), truth[resolved][..., :3].sum(-1)
            )
            if "reference_variance" in arrays:
                result["reference_corrected_mse"] = reference_corrected_mse(
                    prediction[resolved], truth[resolved], arrays["reference_variance"][resolved]
                )
        accepted = arrays.get("accepted", np.zeros(resolved.shape, dtype=bool))
        if (
            accepted.dtype.kind != "b"
            or accepted.shape != resolved.shape
            or np.any(accepted & ~resolved)
        ):
            raise ValueError("explicit acceptance mask must be boolean, aligned, and resolved")
        if accepted.any():
            result["acceptance_provenance"] = _verify_acceptance_receipt(
                config, spec, prediction, accepted, inputs[0]
            )
        loss = np.mean((prediction - truth) ** 2, axis=-1)
        result["risk_coverage"] = {
            "all_cases": int(resolved.size),
            "accepted_cases": int(accepted.sum()),
            "unknown_cases": int((~accepted).sum()),
            "coverage": float(accepted.mean()),
            "accepted_risk": float(loss[accepted].mean()) if accepted.any() else None,
            "all_case_model_loss": float(loss.mean()) if resolved.all() else None,
        }
        if "all_case_baseline_loss" in arrays:
            result["all_case_baseline"] = risk_coverage(
                arrays["all_case_baseline_loss"], np.zeros(resolved.shape, bool)
            )
        result["unknown_cases"] = int((~accepted).sum())
        atomic_json(out / "EVALUATION.json", result)
        status = "EVALUATED" if resolved.all() else "PARTIALLY_UNRESOLVED"
    else:
        raise ValueError("unsupported artifact command")
    receipt = {
        "command": command,
        "status": status,
        "wall_seconds": time.perf_counter() - started,
        "input_binding": input_binding,
        "config_sha256": canonical_hash(config),
    }
    atomic_json(out / "RECEIPT.json", receipt)
    finalize_run(
        out,
        {**input_binding, "source": source_identity()["sha256"], "config": canonical_hash(config)},
    )
    return receipt


def summarize(config, inputs, out):
    """Index actual stage receipts; missing stages remain NOT_RUN.

    This draft decision does not replace a reviewed analysis of Q1-Q6 tables.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    evidence = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            candidates = sorted(path.glob("*SUMMARY.json")) + sorted(path.glob("*RESULTS.json"))
        else:
            candidates = [path]
        for p in candidates:
            value = json.loads(p.read_text())
            evidence.append(
                {
                    "path": str(p.resolve()),
                    "sha256": sha256_file(p),
                    "stage": value.get("stage", p.stem),
                    "status": value.get("status", "RECORDED"),
                }
            )
    receipt = {
        "status": "EVIDENCE_INDEXED_REVIEW_REQUIRED",
        "evidence": evidence,
        "online_ssvc": "NOT_CERTIFIED",
        "config_sha256": canonical_hash(config),
    }
    atomic_json(out / "EVIDENCE_INDEX.json", receipt)
    lines = [
        "# V3 建模决定（待逐阶段证据审阅）",
        "",
        "此文件仅索引已提供原件；未提供的阶段记为 NOT_RUN。",
        "工程测试通过不等于观测精度、预测能力或在线 SSVC 有效。",
        "",
        "| 阶段 | 原件状态 |",
        "|---|---|",
    ]
    lines.extend(f"| {row['stage']} | {row['status']} |" for row in evidence)
    lines += [
        "",
        "观测方法、校准要求、维数、误差范围及失效条件需由完整阶段表审阅后填写；当前不作在线有效性结论。",
    ]
    atomic_bytes(out / "MODELING_DECISION_V3_zh.md", ("\n".join(lines) + "\n").encode())
    finalize_run(out, {"source": source_identity()["sha256"], "config": canonical_hash(config)})
    return receipt
