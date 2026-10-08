"""Read-only, pre-freeze qualification of historical trajectory endpoints.

Historical observations keep their original seed and complete source record.
Only explicit aliases refer to the new development panel. No model is called,
missing observations are never replaced, and RERUN_24 never imports old students.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

from ..exposure_substitution import evaluate as legacy_evaluate
from ..exposure_substitution import schema as legacy_schema
from ..model_adapters.base import pure_generation_options
from ..modeling_v3.io import canonical_hash
from .schema import (
    ARMS,
    PARENTS,
    PHASE_ID,
    digest,
    file_digest,
    public_prompt,
    read_json,
    read_jsonl,
)

ROWS_RELATIVE_PATH = "manifests/historical_trajectory_reused_rows.jsonl"
RECEIPT_RELATIVE_PATH = "HISTORICAL_TRAJECTORY_REUSE.json"
SOURCE_PHASE_ID = "SER_J2_20261007"
AUDIT_KEYS = (
    "model_id",
    "model_revision",
    "base_dtype",
    "probability_execution",
    "processor_hash",
    "tokenizer_hash",
    "chat_template_hash",
    "frozen_parameter_hash",
    "eos_token_ids",
    "lora_modules",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
)
CODEC_KEYS = ("eos_token_ids", "pad_token_id", "bos_token_id", "forward_config")
OLD_MANIFEST_GENERATION = {
    "temperature": 1,
    "top_p": 1,
    "top_k": 0,
    "max_new_tokens": 64,
    "enable_thinking": False,
    "use_cache": False,
    "extra_rescoring": False,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _candidates(mode):
    if mode not in {"REUSE_12", "RERUN_24"}:
        raise ValueError("Historical trajectory qualification requires a selected execution mode")
    jobs = []
    for parent in PARENTS:
        jobs.append(dict(parent=parent, block=None, logical_arm_id="PARENT", arm="PARENT", step=0))
        if mode == "REUSE_12":
            for block in range(3):
                for arm in tuple(ARMS)[:2]:
                    jobs.append(
                        dict(
                            parent=parent,
                            block=block,
                            logical_arm_id=arm,
                            arm=ARMS[arm]["legacy_arm"],
                            step=256,
                        )
                    )
    return jobs


def _legacy_job_id(candidate):
    block = "baseline" if candidate["block"] is None else str(candidate["block"])
    return (
        f"eval.{candidate['parent']}.{block}.{candidate['arm']}.step{candidate['step']}.E_CONFIRM"
    )


def _checkpoint_binding(bindings, candidate):
    if candidate["block"] is None:
        key = "PARENT." + candidate["parent"]
    else:
        key = (
            f"LEGACY.{candidate['parent']}.block{candidate['block']}."
            f"{candidate['logical_arm_id']}.step256"
        )
    binding = bindings.get("checkpoint_lookup", {}).get(key)
    if candidate["block"] is None and binding is None:
        binding = bindings.get("checkpoint_lookup", {}).get(
            f"PARENT.{candidate['parent']}.baseline.PARENT.step0"
        )
    _require(
        isinstance(binding, dict) and binding.get("status") == "VERIFIED_FULL_CPU_STATE",
        "Historical checkpoint lacks full-state qualification",
    )
    _require(
        file_digest(binding["path"]) == binding["checkpoint_sha256"],
        "Historical checkpoint bytes changed after qualification",
    )
    _require(
        bool(binding.get("source_experiment_id")) and bool(binding.get("source_checkpoint_id")),
        "Historical checkpoint source identity missing",
    )
    if candidate["block"] is None:
        _require(
            binding["source_checkpoint_id"] == candidate["parent"], "Parent source alias changed"
        )
    else:
        expected = {
            "experiment_id": SOURCE_PHASE_ID,
            "parent": candidate["parent"],
            "block": candidate["block"],
            "arm": candidate["arm"],
            "technical_only": False,
        }
        source_id = f"{candidate['parent']}.block{candidate['block']}.{candidate['arm']}.step256"
        _require(
            binding["source_experiment_id"] == SOURCE_PHASE_ID
            and binding["source_checkpoint_id"] == source_id
            and all(
                binding.get("expected_identity", {}).get(key) == value
                for key, value in expected.items()
            ),
            "Student source alias changed historical identity",
        )
    return binding


def _reference_backend(bridge, bindings, parent):
    _require(
        bridge.get("execution_kind") == "REAL_CUDA_BRIDGE"
        and bridge.get("status") in {"PASS", "PASS_NEW_RUNTIME_REUSE_INCOMPATIBLE"},
        "A completed real bridge is required to compare current inference/codec",
    )
    expected_parent = bindings["parents"][parent]
    expected_fingerprint = expected_parent.get("inference_fingerprint")
    _require(bool(expected_fingerprint), "Current bound parent inference fingerprint missing")
    receipts = [
        row["receipt"]
        for row in bridge.get("runtime_receipts", [])
        if row.get("receipt", {}).get("checkpoint_id") == parent
    ]
    _require(bool(receipts), "Current bridge has no parent codec/runtime receipt")
    for backend in receipts:
        _require(
            backend.get("inference_fingerprint") == expected_fingerprint
            and backend.get("checkpoint") == expected_parent["checkpoint"]
            and backend.get("checkpoint_full_cpu_state_verified") is True,
            "Current bridge parent identity differs from bound CPU state",
        )
        _validate_backend_common(backend, bindings, parent)
    reference = receipts[0]
    for backend in receipts:
        _require(
            all(key in backend and backend[key] == reference.get(key) for key in CODEC_KEYS),
            "Current bridge codec/forward configuration is inconsistent",
        )
    return reference


def _validate_backend_common(backend, bindings, parent):
    expected = bindings["parents"][parent]
    _require(
        backend.get("execution_kind") == "REAL_CUDA_FROZEN_INFERENCE",
        "Historical backend is not real frozen inference",
    )
    _require(
        backend.get("checkpoint_full_cpu_state_verified") is True
        and backend.get("checkpoint") == expected["checkpoint"],
        "Historical backend parent checkpoint differs",
    )
    environment = bindings.get("runtime_environment")
    _require(
        isinstance(environment, dict)
        and bool(environment)
        and backend.get("environment") == environment,
        "Historical/current runtime environment differs or is unverified",
    )
    history = expected.get("historical_runtime_identity")
    _require(
        isinstance(history, dict)
        and bool(history)
        and backend.get("historical_runtime_identity") == history,
        "Historical base snapshot/runtime identity differs",
    )
    audit, wanted = backend.get("model_audit", {}), expected.get("historical_model_audit", {})
    for key in AUDIT_KEYS:
        _require(
            key in audit and key in wanted and audit[key] == wanted[key],
            "Historical/current model or codec audit differs: " + key,
        )
    _require(
        audit["probability_execution"] == "uncached_prefix_recompute"
        and audit["model_id"] == "Qwen/Qwen3.5-9B"
        and audit["model_revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        and audit["base_dtype"] == "bfloat16",
        "Historical inference engine or immutable base changed",
    )
    _require(
        backend.get("generation_options") == pure_generation_options(64),
        "Historical decode options differ from frozen generation",
    )
    _require(
        backend.get("eos_token_ids") == audit["eos_token_ids"],
        "Historical EOS codec differs from certified audit",
    )
    for key, value in {
        "optimizer_constructed": False,
        "backward_calls": 0,
        "optimizer_updates": 0,
        "likelihood_rescoring_calls": 0,
        "all_parameters_frozen": True,
        "dropout_disabled": True,
        "eval_mode": True,
    }.items():
        _require(backend.get(key) == value, "Historical frozen backend control differs: " + key)


def _panel_mapping(run):
    root = Path(run)
    tasks_path = root / "manifests/DEV_TRAJECTORY/tasks_public.jsonl"
    mapping_path = root / "manifests/development_source_mapping.jsonl"
    tasks = read_jsonl(tasks_path)
    mapping = [row for row in read_jsonl(mapping_path) if row.get("panel") == "DEV_TRAJECTORY"]
    _require(
        len(tasks) == len(mapping) == 128, "Trajectory reuse requires exactly 128 mapped tasks"
    )
    by_id = {task["task_id"]: task for task in tasks}
    _require(
        len(by_id) == 128
        and {row["task_id"] for row in mapping} == set(by_id)
        and len({row["source_task_id"] for row in mapping}) == 128,
        "Trajectory source mapping is incomplete or duplicates source tasks",
    )
    for row in mapping:
        task = by_id[row["task_id"]]
        _require(
            task["split"] == "DEV_TRAJECTORY" and task["root_id"] == row["root_id"],
            "Trajectory mapping changed target task/root identity",
        )
        public_prompt(task)  # Strict public validation; never render audit fields.
    _require(
        len({row["source_root_id"] for row in mapping}) == 32
        and len({row["root_id"] for row in mapping}) == 32,
        "Trajectory reuse requires the same 32 source roots",
    )
    root_pairs = Counter((row["source_root_id"], row["root_id"]) for row in mapping)
    _require(
        len(root_pairs) == 32 and set(root_pairs.values()) == {4},
        "Each historical root must map to one complete four-task target root",
    )
    return (
        by_id,
        mapping,
        {str(path.relative_to(root)): file_digest(path) for path in (tasks_path, mapping_path)},
    )


def _qualify_candidate(legacy_run, bindings, candidate, bridge, tasks, mapping):
    legacy_id = _legacy_job_id(candidate)
    job_rows = legacy_schema.read_bound(legacy_run, "manifests/evaluation_jobs.jsonl")
    selected_jobs = [job for job in job_rows if legacy_evaluate.evaluation_job_id(job) == legacy_id]
    _require(len(selected_jobs) == 1, "Historical evaluation was not registered exactly once")
    job = selected_jobs[0]
    _require(
        job.get("tasks") == 800 and job.get("draws") == 8 and job.get("panel") == "E_CONFIRM",
        "Historical endpoint must retain its entire 800-task, K8 matrix",
    )
    binding = _checkpoint_binding(bindings, candidate)
    current = _reference_backend(bridge, bindings, candidate["parent"])
    parent_fingerprint = bindings["parents"][candidate["parent"]]["inference_fingerprint"]
    expected_fingerprint = (
        parent_fingerprint
        if candidate["block"] is None
        else canonical_hash(
            {"parent": parent_fingerprint, "student_sha256": binding["checkpoint_sha256"]}
        )
    )
    directory = legacy_run / "evaluations" / legacy_id
    manifest = read_json(directory / "manifest.json")
    _require(
        manifest.get("generation") == OLD_MANIFEST_GENERATION and manifest.get("protocol") == "O0",
        "Historical public protocol/manifest decoding differs",
    )
    model = read_json(directory / "MODEL_IDENTITY.json")
    parent_checkpoint = bindings["parents"][candidate["parent"]]["checkpoint"]
    _require(
        model.get("execution_kind") == "REAL_FROZEN_GPU"
        and model.get("inference_fingerprint") == expected_fingerprint
        and model.get("checkpoint_id") == legacy_evaluate.checkpoint_id(job)
        and model.get("parent_checkpoint") == parent_checkpoint
        and model.get("all_lora_adapters_disabled") is False,
        "Historical MODEL_IDENTITY is incompatible with current frozen inference",
    )
    if candidate["block"] is not None:
        _require(
            model.get("student_checkpoint_sha256") == binding["checkpoint_sha256"]
            and model.get("student_identity") == binding["expected_identity"],
            "Historical student SHA/identity differs from read-only binding",
        )
    else:
        _require(
            model.get("student_checkpoint_sha256") is None
            and model.get("student_identity") is None,
            "Historical parent observation was generated with a student",
        )
    backend_paths = sorted((directory / "attempts").glob("*/BACKEND.json"))
    _require(bool(backend_paths), "Historical per-attempt backend evidence is unavailable")
    for path in backend_paths:
        provenance = read_json(path)
        _require(
            provenance.get("execution_kind") == "REAL_FROZEN_GPU",
            "CPU or unverified historical attempt cannot be reused",
        )
        backend = provenance["backend"]
        _validate_backend_common(backend, bindings, candidate["parent"])
        _require(
            backend.get("inference_fingerprint") == expected_fingerprint,
            "Historical attempt inference fingerprint differs",
        )
        _require(
            all(key in backend and backend[key] == current.get(key) for key in CODEC_KEYS),
            "Historical/current tokenizer boundaries or forward configuration differs",
        )
        if candidate["block"] is not None:
            _require(
                backend.get("student_checkpoint_sha256") == binding["checkpoint_sha256"]
                and backend.get("student_identity") == binding["expected_identity"]
                and backend.get("student_step") == 256,
                "Historical attempt used a different student",
            )
    # Native validation rechecks ALL 800x8 records, original request seeds,
    # prompt hashes, accepted-record hashes, weight bytes and completion receipt.
    original_rows, completion = legacy_evaluate.validate_completed_evaluation(legacy_run, job)
    _require(
        len(original_rows) == 6400 and completion.get("samples") == 6400,
        "Historical endpoint is incomplete; partial results are not reusable",
    )
    original_tasks = {
        task["task_id"]: task for task in legacy_evaluate.evaluation_tasks(legacy_run, job)
    }
    by_source = {row["source_task_id"]: row for row in mapping}
    for source_id, alias in by_source.items():
        _require(source_id in original_tasks, "Mapped source task absent from old public panel")
        source = original_tasks[source_id]
        _require(
            source["root_id"] == alias["source_root_id"]
            and public_prompt(tasks[alias["task_id"]]) == legacy_schema.public_prompt(source),
            "New development prompt/root differs from historical source",
        )
    rows, seen = [], set()
    for original in original_rows:
        if original["task_id"] not in by_source:
            continue
        alias = by_source[original["task_id"]]
        draw = original.get("draw_index")
        _require(type(draw) is int and 0 <= draw < 8, "Historical draw index differs")
        key = original["task_id"], draw
        _require(key not in seen, "Historical source draw duplicated")
        seen.add(key)
        _require(
            original.get("experiment_id") == SOURCE_PHASE_ID
            and original.get("execution_kind") == "REAL_FROZEN_GPU"
            and original.get("inference_fingerprint") == expected_fingerprint
            and original.get("parent") == candidate["parent"]
            and original.get("block") == candidate["block"]
            and original.get("arm") == candidate["arm"]
            and original.get("step") == candidate["step"]
            and original.get("root_id") == alias["source_root_id"],
            "Historical observation source identity differs",
        )
        expected_identity = legacy_evaluate.sample_identity(
            job, original_tasks[original["task_id"]], draw
        )
        _require(
            original.get("sample_seed") == legacy_evaluate.sample_seed(expected_identity)
            and original.get("request_id") == legacy_evaluate.digest(expected_identity),
            "Historical observation seed/request identity differs",
        )
        row = {
            key: copy.deepcopy(value)
            for key, value in original.items()
            if key not in {"arm", "experiment_id", "task_id", "root_id", "panel"}
        }
        row.update(
            phase_id=SOURCE_PHASE_ID,
            target_phase_id=PHASE_ID,
            panel="DEV_TRAJECTORY",
            task_id=alias["task_id"],
            root_id=alias["root_id"],
            logical_arm_id=candidate["logical_arm_id"],
            origin="historical_reuse",
            sample_id=original["request_id"],
            source_arm=original["arm"],
            source_panel="E_CONFIRM",
            source_task_id=original["task_id"],
            source_root_id=original["root_id"],
            source_phase_id=SOURCE_PHASE_ID,
            source_experiment_id=binding["source_experiment_id"],
            source_checkpoint_id=binding["source_checkpoint_id"],
            checkpoint_sha256=binding["checkpoint_sha256"],
            original_record=copy.deepcopy(original),
            source_record_sha256=digest(original),
            compatibility_verified=True,
        )
        rows.append(row)
    _require(
        len(rows) == 1024 and len(seen) == 1024,
        "Historical trajectory selection must include exactly 128 complete K8 tasks",
    )
    evidence_paths = [
        directory / "manifest.json",
        directory / "MODEL_IDENTITY.json",
        directory / "completion.json",
        *backend_paths,
    ]
    receipt = {
        **{key: candidate[key] for key in ("parent", "block", "logical_arm_id", "step")},
        "legacy_job_id": legacy_id,
        "compatible": True,
        "samples": len(rows),
        "expected_current_inference_fingerprint": expected_fingerprint,
        "checkpoint_binding": copy.deepcopy(binding),
        "source_completion": completion,
        "source_records_hash": digest(original_rows),
        "alias_records_hash": digest(rows),
        "source_file_hashes": {str(path): file_digest(path) for path in evidence_paths},
    }
    return receipt, rows


def audit_historical_trajectory(run, bindings, *, mode):
    """Return ``(receipt, rows)`` for the freeze owner to persist without mutation.

    Missing/incompatible model evidence becomes an explicit unavailable point.
    Invalid target mapping is a structural error and raises instead of silently
    dropping data. This API must run before FROZEN_PLAN exists.
    """
    root = Path(run).resolve()
    if (root / "FROZEN_PLAN.json").exists():
        raise PermissionError("Historical trajectory availability must be registered before freeze")
    candidates = _candidates(mode)
    tasks, mapping, panel_files = _panel_mapping(root)
    legacy_run = Path(bindings["legacy_run"]).resolve()
    _require(
        legacy_run != root and not root.is_relative_to(legacy_run),
        "Historical source must remain separate from the new run",
    )
    bridge_path = root / "CONTINUITY_BRIDGE.json"
    bridge = read_json(bridge_path) if bridge_path.exists() else {}
    compatible, unavailable, rows = [], [], []
    for candidate in candidates:
        try:
            receipt, selected = _qualify_candidate(
                legacy_run, bindings, candidate, bridge, tasks, mapping
            )
        except (OSError, ValueError, KeyError, TypeError, PermissionError) as exc:
            unavailable.append(
                {
                    **{
                        key: candidate[key] for key in ("parent", "block", "logical_arm_id", "step")
                    },
                    "legacy_job_id": _legacy_job_id(candidate),
                    "compatible": False,
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                    "replacement_generation_permitted": False,
                }
            )
        else:
            compatible.append(receipt)
            rows.extend(selected)
    _require(
        len({row["sample_id"] for row in rows}) == len(rows),
        "Historical source sample identity duplicated across reused models",
    )
    expected_models = 14 if mode == "REUSE_12" else 2
    _require(
        len(compatible) + len(unavailable) == expected_models,
        "Historical trajectory qualification matrix changed",
    )
    receipt = {
        "schema": "ser-j23-historical-trajectory-reuse-v1",
        "status": "AUDITED",
        "phase_id": PHASE_ID,
        "source_phase_id": SOURCE_PHASE_ID,
        "mode": mode,
        "legacy_run": str(legacy_run),
        "read_only": True,
        "compatible_models": compatible,
        "unavailable": unavailable,
        "candidate_models": expected_models,
        "rows_count": len(rows),
        "rows_hash": digest(rows),
        "rows_relative_path": ROWS_RELATIVE_PATH,
        "new_model_calls": 0,
        "new_logical_generations": 0,
        "no_budget_transfer": True,
        "legacy_student_step256_permitted": mode == "REUSE_12",
        "source_binding_hash": digest(bindings),
        "target_panel_files": panel_files,
        "bridge_sha256": file_digest(bridge_path) if bridge_path.exists() else None,
        "not_comparable_to_E_CONFIRM2": True,
    }
    return receipt, rows
