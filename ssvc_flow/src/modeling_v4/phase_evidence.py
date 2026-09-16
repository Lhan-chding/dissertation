"""Build phase metadata from actual bound CUDA collection and CPU fit receipts.

Read-only: SHA256 streams originals once per immutable file version, without
deserializing tensor/Parquet arrays or loading a model. Numerical/semantic
correctness remains the producers' responsibility; this verifies completion,
identity, registered coverage and fit-before-reference provenance. It does not
invent cross-seed CV or point-response resolution evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .config import digest, validate_config
from .data_adapter import CONTRASTS
from .phase_schedule import ROLES, validate_first_map_completion, validate_selection


class ReceiptReader:
    def __init__(self):
        self.files = {}
        self.documents = {}

    @staticmethod
    def _version(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def binding(self, value):
        path = Path(value["path"] if isinstance(value, dict) else value).resolve()
        before = self._version(path)
        if path in self.files:
            saved, version = self.files[path]
            if version != before:
                raise ValueError("Original changed during phase-evidence verification")
        else:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if self._version(path) != before:
                raise ValueError("Original changed while hashing")
            saved = {"path": str(path), "sha256": hasher.hexdigest()}
            self.files[path] = saved, before
        if isinstance(value, dict) and value.get("sha256") != saved["sha256"]:
            raise ValueError("Original receipt/artifact SHA256 mismatch: " + str(path))
        return saved.copy()

    def json(self, value):
        binding = self.binding(value)
        path = binding["path"]
        if path not in self.documents:
            self.documents[path] = json.loads(Path(path).read_text())
            self.binding(binding)
        return self.documents[path]

    def graph(self, value):
        """Verify nested explicit file bindings; do not follow arbitrary path strings."""
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.binding(value)
            for child in value.values():
                self.graph(child)
        elif isinstance(value, list):
            for child in value:
                self.graph(child)

    def completed_bundle(self, root):
        root = Path(root).resolve()
        complete = self.json(root / "COMPLETE.json")
        manifest_binding = {
            "path": str(root / "RUN_MANIFEST.json"),
            "sha256": complete["manifest_sha256"],
        }
        manifest = self.json(manifest_binding)
        if manifest.get("status") != "COMPLETE" or not manifest.get("files"):
            raise ValueError("Fit/evaluation bundle is not complete")
        for relative, sha in manifest["files"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or path == root:
                raise ValueError("Fit/evaluation manifest path escapes its root")
            self.binding({"path": str(path), "sha256": sha})
        actual = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file()
            and path.name != ".writer.lock"
            and path not in {root / "COMPLETE.json", root / "RUN_MANIFEST.json"}
        }
        if actual != set(manifest["files"]):
            raise ValueError("Fit/evaluation bundle file inventory changed")
        return manifest


def _packet(receipt, *, origin, role, prompts, draws, reader):
    identity = receipt.get("identity", {})
    if (
        identity.get("origin_id") != origin
        or identity.get("role") != role
        or identity.get("draws") != draws
        or receipt.get("count") != prompts * draws
    ):
        raise ValueError("Actual action packet scope differs from its task")
    chunks = receipt.get("chunks", [])
    expected = 0
    for chunk in chunks:
        if (
            chunk.get("start") != expected
            or type(chunk.get("stop")) is not int
            or chunk["stop"] <= expected
        ):
            raise ValueError("Action packet chunk coverage is incomplete or duplicated")
        reader.binding(chunk)
        expected = chunk["stop"]
    if expected != receipt["count"]:
        raise ValueError("Action packet lacks its complete original chunks")
    return identity


def _fit(reader, result_path, *, plan, tasks_binding, task, completed, task_binding, forks):
    supplied_binding = result_path if isinstance(result_path, dict) else None
    result_path = Path(result_path["path"] if supplied_binding else result_path)
    if result_path.is_dir():
        result_path /= "RESULTS.json"
    fit_manifest = reader.completed_bundle(result_path.parent)
    result = reader.json(supplied_binding or result_path)
    binding = result.get("binding", {})
    expected = {
        "tasks": tasks_binding,
        "task": task_binding,
        "forks": completed["forks"],
        "response": completed["response"],
        "config_hash": digest(plan["config"]),
        "source": plan["source"],
        "execution_kind": "REAL_CUDA_MODEL",
        "task_id": task["id"],
    }
    if (
        result.get("kind") != "V4_RESPONSE_MAP_FIT"
        or result.get("status") != "COMPLETED"
        or result.get("task_id") != task["id"]
        or any(binding.get(k) != v for k, v in expected.items())
        or fit_manifest.get("binding") != {**binding, "kind": "V4_RESPONSE_MAP_FIT"}
        or result.get("reference_read_after_predictions") is not True
        or result.get("query_response_used_for_fit") is not False
        or type(result.get("prediction_records")) is not int
        or result["prediction_records"] <= 0
    ):
        raise ValueError("Actual fitted evaluation identity/order differs from the collected map")
    if (
        result.get("calibration_banks") != task["calibration_banks"]
        or result.get("query_banks") != task["query_banks"]
    ):
        raise ValueError("Fitted bank coverage differs from the completed map")
    prediction = reader.binding(result["predictions"])
    if Path(prediction["path"]).parent != result_path.parent.resolve():
        raise ValueError("Prediction must belong to this completed fit bundle")
    models = reader.json(result_path.parent / "MODELS.json")
    calibration = [bank["bank_id"] for bank in forks["banks"] if bank["role"] == "calibration"]
    if (
        models.get("binding") != binding
        or models.get("query_labels_read_for_fit") is not False
        or models.get("calibration_bank_ids") != calibration
    ):
        raise ValueError("Actual model fits used a different calibration-bank scope")
    methods = sorted({row["method"] for row in models.get("models", []) if row.get("method")})
    if not set(methods) & (set(plan["config"]["models"]["primary"]) - {"ZERO", "DIRECT_MEASURE"}):
        raise ValueError("Direct/ZERO-only or missing fits cannot complete the first response maps")
    evaluation = result.get("evaluation", {})
    if evaluation.get("status") != "COMPLETE":
        raise ValueError("Actual evaluation completion receipt required")
    evaluation_root = Path(evaluation["out"]).resolve()
    if evaluation_root != (result_path.parent / "evaluation").resolve():
        raise ValueError("Evaluation belongs to another fit")
    eval_manifest = reader.completed_bundle(evaluation_root)
    metric_path = Path(evaluation["metrics"]).resolve()
    if metric_path != evaluation_root / "METRICS.json":
        raise ValueError("Evaluation metrics path differs")
    metrics = reader.json(metric_path)
    expected_evaluation_binding = {**binding, "predictions": prediction}
    if (
        metrics.get("kind") != "V4_EVALUATION"
        or metrics.get("binding") != expected_evaluation_binding
        or eval_manifest.get("binding") != {**expected_evaluation_binding, "kind": "V4_EVALUATION"}
        or metrics.get("selection_modified") is not False
        or metrics.get("reference_read_scope") != "EVALUATION_ONLY"
        or not metrics.get("summary")
    ):
        raise ValueError("Completed evaluation must bind predictions and disclose actual metrics")
    core_path = Path(evaluation["core_results"]).resolve()
    if (
        core_path != evaluation_root / "CORE_RESULTS.csv"
        or metrics.get("core_results_file") != "CORE_RESULTS.csv"
    ):
        raise ValueError("Evaluation must name its original streamed core results")
    core_binding = reader.binding(core_path)
    with core_path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        required = {"seed", "origin_id", "bank_id", "prediction_status", "prediction_record_id"}
        if not required <= set(rows.fieldnames or []):
            raise ValueError("Streamed evaluation lacks required identity columns")
        count = 0
        for row in rows:
            if (
                row["prediction_record_id"] != prediction["sha256"]
                or row["origin_id"] != task["id"]
                or row["seed"] != str(task["seed"])
            ):
                raise ValueError("Streamed evaluation identity differs from frozen predictions")
            count += 1
    reader.binding(core_binding)
    if count == 0:
        raise ValueError("Completed evaluation contains no actual query results")
    return {
        "fit_receipt": reader.binding(result_path),
        "evaluation_receipt": reader.binding(metric_path),
        "fitted_model_ids": methods,
    }


def _verify_cv(reader, binding, plan):
    """Verify the actual signature CV producer and its whole-seed index split."""
    cv = reader.json(binding)
    if (
        cv.get("schema") != "ssvc-v4-whole-seed-cv-1"
        or cv.get("kind") != "V4_SIGNATURE_WHOLE_SEED_FOLD"
        or cv.get("status") != "COMPLETED"
        or cv.get("fixture") is not False
        or cv.get("config_hash") != digest(plan["config"])
        or cv.get("campaign_id") != plan["campaign_id"]
        or cv.get("query_labels_read") is not False
        or cv.get("reference_labels_read") is not False
    ):
        raise ValueError("Actual matching raw-calibration-only whole-seed CV producer required")
    heldout = cv.get("validation_seeds", [])
    dev = set(ROLES["development"])
    if (
        len(heldout) != 1
        or heldout[0] not in dev
        or cv.get("train_seeds") != sorted(dev - set(heldout))
    ):
        raise ValueError("CV must hold out exactly one entire development seed")
    reader.completed_bundle(Path(reader.binding(binding)["path"]).parent)
    dataset = reader.json(cv["dataset"])
    reader.completed_bundle(Path(cv["dataset"]["path"]).parent)
    if (
        dataset.get("fixture") is not False
        or dataset.get("config_hash") != digest(plan["config"])
        or dataset.get("campaign_id") != plan["campaign_id"]
        or dataset.get("source_hash") != plan["source"]["sha256"]
    ):
        raise ValueError("CV dataset uses another source/config/campaign")
    reader.graph(dataset)
    original_rows = {}
    origin_keys = set()
    for item in dataset.get("original_bindings", []):
        labels = reader.json(item["labels"])
        signature = reader.json(item["signature"])
        reader.completed_bundle(Path(item["labels"]["path"]).parent)
        reader.completed_bundle(Path(item["signature"]["path"]).parent)
        reader.graph(labels)
        reader.graph(signature)
        if (
            labels.get("kind") != "V4_CALIBRATION_LABELS"
            or labels.get("role") != "development"
            or labels.get("bank_role_scope") != "calibration"
            or labels.get("query_labels_read") is not False
            or labels.get("reference_labels_read") is not False
            or labels.get("config_hash") != digest(plan["config"])
            or labels.get("source_hash") != plan["source"]["sha256"]
            or labels.get("calibration_bank_count") != 96
            or labels.get("draws") != 1024
            or len(labels.get("prompt_ids", [])) != 72
            or labels.get("event_order") != ["X", "S", "W", "I"]
            or labels.get("binding", {}).get("execution_kind") != "REAL_CUDA_MODEL"
            or signature.get("kind") != "V4_NATIVE_SIGNATURE_FEATURES"
            or signature.get("origin_id") != labels.get("origin_id")
            or signature.get("runtime_identity", {}).get("source_hash") != plan["source"]["sha256"]
            or signature.get("runtime_identity", {}).get("config_hash") != digest(plan["config"])
        ):
            raise ValueError(
                "CV originals must be actual matching development calibration observations"
            )
        original_plan = reader.json(labels["binding"]["tasks"])
        if (
            original_plan.get("campaign_id") != plan["campaign_id"]
            or original_plan.get("source") != plan["source"]
            or digest(original_plan["config"]) != digest(plan["config"])
        ):
            raise ValueError("CV label originals belong to a different campaign")
        key = (labels["seed"], labels["arm"], labels["step"])
        if key in origin_keys:
            raise ValueError("CV repeats a development origin")
        origin_keys.add(key)
        units = labels.get("units", [])
        if (
            len(units) != 96 * 3
            or len({r["bank_id"] for r in units}) != 96
            or {r["contrast_id"] for r in units} != {c[0] for c in CONTRASTS}
            or len({(r["bank_id"], r["contrast_id"]) for r in units}) != len(units)
        ):
            raise ValueError("CV calibration label bank/contrast matrix is incomplete")
        for row in units:
            if row.get("bank_role") != "calibration" or any(
                row.get(k) != labels[k] for k in ("origin_id", "seed", "arm", "step", "role")
            ):
                raise ValueError("Query/test/reference rows cannot enter development CV")
            original_rows[digest(row)] = row
    if origin_keys != {(s, "X_BASE", step) for s in dev for step in (32, 96)}:
        raise ValueError("CV dataset is missing its six actual development origins")
    records = dataset.get("records", [])
    if not records or any(digest(row) not in original_rows for row in records):
        raise ValueError("CV rows do not reproduce original calibration label identities")
    pairs = [
        tuple(sorted((r["baseline_fingerprint"], r["candidate_fingerprint"]))) for r in records
    ]
    if len(set(pairs)) != len(pairs):
        raise ValueError("CV counts duplicated endpoint pairs as independent rows")
    train = [i for i, row in enumerate(records) if row["seed"] != heldout[0]]
    valid = [i for i, row in enumerate(records) if row["seed"] == heldout[0]]
    if (
        not train
        or not valid
        or cv.get("training_rows") != len(train)
        or cv.get("validation_rows") != len(valid)
    ):
        raise ValueError("CV actual row counts do not match heldout-seed identities")
    fit, evaluation = reader.json(cv["fit_receipt"]), reader.json(cv["evaluation_receipt"])
    shared = {
        k: cv[k]
        for k in (
            "dataset",
            "train_seeds",
            "validation_seeds",
            "config_hash",
            "campaign_id",
            "fixture",
            "query_labels_read",
            "reference_labels_read",
        )
    }
    if (
        fit.get("kind") != "V4_SIGNATURE_CV_FITS"
        or evaluation.get("kind") != "V4_SIGNATURE_CV_EVALUATION"
        or any(fit.get(k) != v or evaluation.get(k) != v for k, v in shared.items())
        or fit.get("training_indices") != train
        or evaluation.get("validation_indices") != valid
        or evaluation.get("evaluation_scope") != "HELDOUT_DEVELOPMENT_SEED_CALIBRATION_OBSERVATIONS"
        or fit.get("models")
        != [
            {k: v for k, v in r.items() if k not in {"predictions", "metrics"}}
            for r in cv["models"]
        ]
        or evaluation.get("models")
        != [{k: v for k, v in r.items() if k != "model"} for r in cv["models"]]
    ):
        raise ValueError(
            "Actual CV fit/evaluation receipts do not match the disjoint whole-seed split"
        )
    available = 0
    for row in cv["models"]:
        if row["status"] == "NOT_ELIGIBLE":
            if "model" in row or "predictions" in row:
                raise ValueError("Ineligible CV model cannot claim fitted predictions")
            continue
        if row["status"] != "AVAILABLE" or not row.get("metrics"):
            raise ValueError("CV model lacks actual evaluated fitted output")
        model = reader.json(row["model"])
        if model.get("kind") != "V4_FROZEN_SIGNATURE_MODEL":
            raise ValueError("CV fit does not bind an actual serialized predictor")
        reader.completed_bundle(Path(row["model"]["path"]).parent)
        reader.graph(model)
        reader.binding(row["predictions"])
        available += 1
    if not available:
        raise ValueError("No actually fitted/evaluated model in this whole-seed CV fold")
    return {
        "train_seeds": cv["train_seeds"],
        "validation_seeds": heldout,
        "receipt": reader.binding(binding),
    }


def build_phase_evidence(tasks, *, fit_results, phase="C", selection=None, cv_receipts=None):
    """Read actual phase originals, returning the normalized scheduling evidence.

    ``fit_results`` maps each expected map task ID to its fit directory or
    RESULTS.json. Optional CV receipts must be actual independent producer
    artifacts; absence is reported and does not manufacture selection readiness.
    No files are created. The caller may publish this result once.
    """
    reader = ReceiptReader()
    tasks_binding = reader.binding(tasks)
    plan = reader.json(tasks_binding)
    config = validate_config(plan["config"])
    if plan.get("schema") != "ssvc-v4-task-list-1" or plan.get("task_list_hash") != digest(
        {k: v for k, v in plan.items() if k != "task_list_hash"}
    ):
        raise ValueError("Actual task manifest identity changed")
    if phase not in ("C", "D_DEVELOPMENT", "D_CALIBRATION"):
        raise ValueError("Only completed C/development/calibration metadata can enter this reader")
    if not isinstance(fit_results, dict) or {"path", "sha256"} <= set(fit_results):
        fit_results = reader.json(fit_results)
    role = "calibration" if phase == "D_CALIBRATION" else "development"
    seeds = ROLES[role][:2] if phase == "C" else ROLES[role]
    source_arms = ["X_BASE"] if phase == "C" else ["X_BASE", "X_VALID"]
    root = Path(plan["root"])
    bridge = reader.json(root / "B_MERGED.json")
    reader.graph(bridge)
    if (
        bridge.get("status") != "TECHNICAL_BRIDGE_COMPLETED"
        or bridge.get("parity", {}).get("passed") is not True
    ):
        raise ValueError("Actual two-GPU technical bridge has not completed")
    result = {
        "schema": "ssvc-v4-phase-evidence-1",
        "phase": phase,
        "status": "COMPLETE",
        "config_hash": digest(config),
        "campaign_id": plan["campaign_id"],
        "execution_kind": "REAL_CUDA_MODEL",
        "tasks_binding": tasks_binding,
        "bridge_receipt": reader.binding(root / "B_MERGED.json"),
        "sources": [],
        "origins": [],
        "test_opened": False,
        "scientific_status": "NOT_CERTIFIED",
        "missing_evidence": [],
    }
    by_id = {t["id"]: t for t in plan["tasks"]}
    if len(by_id) != len(plan["tasks"]):
        raise ValueError("Task manifest contains duplicate identities")
    # Do not open test content. Any completed registered test task means this
    # campaign can no longer generate a pre-test model/calibration freeze.
    for task in plan["tasks"]:
        if (
            task.get("seed") in ROLES["test"]
            and (root / "tasks" / task["id"] / "COMPLETE.json").exists()
        ):
            raise ValueError("Locked test was already opened; cannot create selection evidence")
    for seed in seeds:
        for arm in source_arms:
            task_id = f"source_{seed}_{arm}"
            task = by_id.get(task_id, {})
            path = root / "tasks" / task_id / "COMPLETE.json"
            source = reader.json(path)
            if (
                task.get("kind") != "source"
                or task.get("seed") != seed
                or task.get("arm") != arm
                or source.get("status") != "COMPLETED"
                or source.get("kind") != "source"
                or source.get("seed") != seed
                or source.get("arm") != arm
                or source.get("steps") != 128
                or source.get("execution_kind") != "REAL_CUDA_MODEL"
                or not {"32", "96"} <= set(source.get("origins", {}))
            ):
                raise ValueError("Complete actual 128-step source with both origins required")
            reader.graph(source)
            result["sources"].append(
                {
                    "seed": seed,
                    "arm": arm,
                    "steps": 128,
                    "status": "COMPLETED",
                    "receipt": reader.binding(path),
                }
            )
    expected_ids = {
        f"{'D_' if phase != 'C' else ''}map_{seed}_X_BASE_{step}"
        for seed in seeds
        for step in (32, 96)
    }
    if set(fit_results) != expected_ids:
        raise ValueError("Fit-result mapping must cover every expected origin exactly once")
    for task_id in sorted(expected_ids):
        task = by_id.get(task_id, {})
        if task.get("kind") != "map":
            raise ValueError("Missing actual response-map task")
        task_path = root / "tasks" / task_id / "COMPLETE.json"
        completed = reader.json(task_path)
        if (
            completed.get("status") != "COMPLETED"
            or completed.get("kind") != "map"
            or completed.get("task") != task
            or completed.get("execution_kind") != "REAL_CUDA_MODEL"
        ):
            raise ValueError("Actual CUDA map completion does not match its task")
        forks, response = reader.json(completed["forks"]), reader.json(completed["response"])
        reader.graph(forks)
        reader.graph(response)
        origin_id = completed.get("origin_id")
        expected_origin = task.get("reuse_prefix_task", task_id)
        if origin_id != expected_origin:
            raise ValueError("Stable map origin does not match its registered prefix identity")
        if task.get("reuse_prefix_task"):
            previous = reader.json(root / "tasks" / expected_origin / "COMPLETE.json")
            if (
                previous.get("status") != "COMPLETED"
                or previous.get("origin_id") != origin_id
                or previous.get("task") != by_id.get(expected_origin)
                or previous.get("execution_kind") != "REAL_CUDA_MODEL"
            ):
                raise ValueError("Nested map lacks its actual completed original prefix")
            reader.graph(previous)
        if (
            forks.get("origin_id") != origin_id
            or response.get("origin_id") != origin_id
            or response.get("all_queries_retained") is not True
        ):
            raise ValueError("Map origin/query provenance differs")
        banks = forks.get("banks", [])
        if Counter(bank["role"] for bank in banks) != {
            "calibration": task["calibration_banks"],
            "query": task["query_banks"],
        } or len({bank["bank_id"] for bank in banks}) != len(banks):
            raise ValueError("Actual candidate banks are incomplete or duplicated")
        work = _packet(
            response["work"],
            origin=origin_id,
            role="work",
            prompts=task["prompts"],
            draws=task["draws"],
            reader=reader,
        )
        reference = _packet(
            response["reference"],
            origin=origin_id,
            role="reference",
            prompts=task["prompts"],
            draws=config["observations"]["reference_primary_n"],
            reader=reader,
        )
        if (
            not work.get("rng_namespace")
            or not reference.get("rng_namespace")
            or work["rng_namespace"] == reference["rng_namespace"]
        ):
            raise ValueError("Work and reference RNG namespaces must be independently bound")
        fitted = _fit(
            reader,
            fit_results[task_id],
            plan=plan,
            tasks_binding=tasks_binding,
            task=task,
            completed=completed,
            task_binding=reader.binding(task_path),
            forks=forks,
        )
        result["origins"].append(
            {
                **{
                    key: task[key]
                    for key in (
                        "seed",
                        "arm",
                        "step",
                        "calibration_banks",
                        "query_banks",
                        "prompts",
                        "draws",
                    )
                },
                "task_id": task_id,
                "origin_id": origin_id,
                "collection_status": "COMPLETED",
                "evaluation_status": "COMPLETE",
                "collection_receipt": reader.binding(task_path),
                **fitted,
                "fit_scope": "calibration_banks",
                "evaluation_scope": "heldout_query_banks",
                "all_queries_retained": True,
            }
        )
    if phase == "D_DEVELOPMENT":
        result["whole_seed_cv"] = []
        for binding in cv_receipts or []:
            result["whole_seed_cv"].append(_verify_cv(reader, binding, plan))
        if cv_receipts and (
            len(result["whole_seed_cv"]) != 3
            or {tuple(row["validation_seeds"]) for row in result["whole_seed_cv"]}
            != {(s,) for s in ROLES["development"]}
        ):
            raise ValueError("Actual development CV requires all three distinct heldout-seed folds")
        if not cv_receipts:
            result["missing_evidence"].append("ACTUAL_WHOLE_SEED_DEVELOPMENT_CV")
    if phase == "D_CALIBRATION":
        if selection is None:
            raise ValueError("Calibration evidence requires the existing frozen selection")
        selected = validate_selection(
            config,
            reader.json(selection)
            if not isinstance(selection, dict) or "path" in selection
            else selection,
        )
        if selected["campaign_id"] != plan["campaign_id"]:
            raise ValueError("Calibration selection belongs to another campaign")
        result["selection_hash"] = selected["selection_hash"]
    result["verification"] = {
        "files_sha256_verified": len(reader.files),
        "arrays_deserialized": False,
        "model_loaded": False,
        "scientific_resolution_inferred": False,
    }
    if phase == "C":
        validate_first_map_completion(config, result)
    return result


def verify_phase_evidence(evidence, *, selection=None):
    """Rebuild normalized evidence from bound originals before a phase starts.

    This is read-only and excludes only a hashing-cache counter from equality;
    every scientific field and original file binding must reproduce exactly.
    """
    if not isinstance(evidence, dict) or "path" in evidence:
        evidence = ReceiptReader().json(evidence)
    fits = {row["task_id"]: row["fit_receipt"] for row in evidence["origins"]}
    if len(fits) != len(evidence["origins"]):
        raise ValueError("Evidence repeats a fitted origin")
    rebuilt = build_phase_evidence(
        evidence["tasks_binding"],
        fit_results=fits,
        phase=evidence["phase"],
        selection=selection,
        cv_receipts=[row["receipt"] for row in evidence.get("whole_seed_cv", [])],
    )

    def substantive(value):
        return {
            **value,
            "verification": {
                k: v for k, v in value["verification"].items() if k != "files_sha256_verified"
            },
        }

    if substantive(rebuilt) != substantive(evidence):
        raise ValueError("Phase evidence no longer reproduces its actual original receipts")
    return rebuilt
