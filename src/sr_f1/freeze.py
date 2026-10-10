"""Fail-closed S0/S1 identity gates, separate from GPU engine certification."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
from pathlib import Path

from mm_core.execution import atomic_json, object_hash, read_json, utc_now

from .amendment import (
    EFFECTIVE_CONFIG,
    ORIGINAL_FREEZE_SHA256,
    REGISTERED_DOCUMENTS,
    approved_changes,
    effective_plan,
    protocol_amendment,
    validate_plan,
    verify_amendment,
    verify_protocol_route,
)
from .contract import PACKAGE, PLAN_ID, file_hash
from .data import bounded_path, read_jsonl

TECHNICAL_REPAIR_ID = "SR_F1_FORMAT_GUARD_REPAIR_20261009"
TECHNICAL_REPAIR_ALLOWED_FILES = frozenset(
    {
        "src/sr_f1/freeze.py",
        "src/sr_f1/runtime.py",
        "src/sr_f1/format_review.py",
        "src/sr_f1/orchestration.py",
        "scripts/sr_f1/submit_matrix.py",
    }
)
QOS_SCOPE_REPAIR_ID = "SR_F1_1_QOS_SCOPE_20261010"
QOS_SCOPE_ALLOWED_FILES = frozenset({"src/sr_f1/freeze.py", "src/sr_f1/orchestration.py"})
ENGINE_MEMORY_REPAIR_ID = "SR_F1_1_ENGINE_MEMORY_20261010"
ENGINE_MEMORY_ALLOWED_FILES = frozenset(
    {
        "src/sr_f1/freeze.py",
        "src/sr_f1/runtime.py",
        "src/sr_f1/engine.py",
        "src/sr_f1/orchestration.py",
        "scripts/sr_f1/submit_matrix.py",
    }
)
ENGINE_STORAGE_REPAIR_ID = "SR_F1_1_ENGINE_STORAGE_20261010"
ENGINE_STORAGE_ALLOWED_FILES = frozenset(
    {
        "src/sr_f1/freeze.py",
        "src/sr_f1/runtime.py",
        "src/sr_f1/orchestration.py",
        "scripts/sr_f1/submit_matrix.py",
    }
)

ENGINE_IO_REPAIR_ID = "SR_F1_1_ENGINE_IO_20261010"
ENGINE_IO_ALLOWED_FILES = frozenset(
    {
        "src/sr_f1/freeze.py",
        "src/sr_f1/runtime.py",
        "src/sr_f1/orchestration.py",
        "scripts/sr_f1/submit_matrix.py",
    }
)

ENGINE_MULTIGPU_REPAIR_ID = "SR_F1_1_ENGINE_MULTIGPU_20261010"
ENGINE_MULTIGPU_REQUIRED_FILES = frozenset(
    {
        "src/sr_f1/freeze.py",
        "src/sr_f1/runtime.py",
        "src/sr_f1/orchestration.py",
        "scripts/sr_f1/submit_matrix.py",
        "scripts/sr_f1/run_worker.py",
    }
)
ENGINE_MULTIGPU_ALLOWED_FILES = ENGINE_MULTIGPU_REQUIRED_FILES | {
    "scripts/sr_f1/validate_multigpu.py",
}

BASELINE_PARALLEL_REPAIR_ID = "SR_F1_1_BASELINE_PARALLEL_20261010"
BASELINE_PARALLEL_REQUIRED_FILES = frozenset(
    {
        "src/sr_f1/freeze.py",
        "src/sr_f1/orchestration.py",
        "src/sr_f1/baseline_parallel.py",
        "scripts/sr_f1/run_worker.py",
        "scripts/sr_f1/baseline_handoff.py",
    }
)
BASELINE_PARALLEL_ALLOWED_FILES = BASELINE_PARALLEL_REQUIRED_FILES | {
    "scripts/sr_f1/validate_baseline_parallel.py",
}


ENGINE_COMPUTE_PARALLEL_REPAIR_ID = "SR_F1_1_ENGINE_COMPUTE_PARALLEL_20261010"
ENGINE_COMPUTE_PARALLEL_REQUIRED_FILES = frozenset(
    {
        "src/sr_f1/freeze.py",
        "src/sr_f1/orchestration.py",
        "src/sr_f1/training.py",
        "src/sr_f1/runtime.py",
        "src/sr_f1/compute_parallel.py",
        "src/sr_f1/compute_workers.py",
        "src/sr_f1/recompute.py",
        "src/sr_f1/engine.py",
        "scripts/sr_f1/run_worker.py",
        "scripts/sr_f1/submit_matrix.py",
        "src/sr_f1/baseline_parallel.py",
        "scripts/sr_f1/baseline_handoff.py",
    }
)
ENGINE_COMPUTE_PARALLEL_ALLOWED_FILES = ENGINE_COMPUTE_PARALLEL_REQUIRED_FILES | {
    "scripts/sr_f1/validate_compute_parallel.py",
    "scripts/sr_f1/validate_compute_parallel_9b.py",
    "scripts/sr_f1/compute_parallel_maintenance.py",
    "scripts/sr_f1/validate_baseline_parallel.py",
}
COMPUTE_PARALLEL_PREFIX = "technical_incidents/compute_parallel_20261010/"


def _compute_parallel_source_context(root):
    root = Path(root).resolve(strict=True)
    declared = root / "code_before_engine_compute_parallel_20261010"
    if declared.is_symlink():
        raise PermissionError("Pre-compute source must be a real directory")
    saved = bounded_path(root, declared.name)
    before = read_json(saved / "SOURCE_DEPLOYMENT.json")
    _required(
        _preserved_source_files(saved, include_amendments=True),
        before.get("source_file_hashes"),
        "Pre-compute source changed",
    )
    parent = _verify_original_engine_multigpu_repair(root, actual_source=before)
    return root, saved, before, parent


def build_engine_compute_parallel_repair(
    root, *, actual_source, gpu_count, authorized_at, authorized_user_message
):
    root, saved, before, parent = _compute_parallel_source_context(root)
    if type(gpu_count) is not int or not 2 <= gpu_count <= 5:
        raise PermissionError("Compute GPU count must be from two through five")
    names = [
        "PRESERVATION.json",
        "TERMINAL_JOBS.json",
        "CPU_STATE_REVIEW.json",
        "RECOVERY_REVIEW.json",
        "GPU_PROBE.json",
    ]
    previous, current = before["source_file_hashes"], actual_source["source_file_hashes"]
    return {
        "schema_version": 1,
        "repair_id": ENGINE_COMPUTE_PARALLEL_REPAIR_ID,
        "status": "AUTHORIZED_TECHNICAL_COMPUTE_PARALLEL_REPAIR",
        "plan_id": PLAN_ID,
        "run_root": str(root),
        "original_execution_freeze_sha256": file_hash(root / "EXECUTION_FREEZE.json"),
        "previous_engine_multigpu_repair_sha256": parent["repair_sha256"],
        "previous_source_commit": before["source_commit"],
        "previous_source_tree_sha256": before["source_tree_sha256"],
        "preserved_source_relative_path": saved.name,
        "previous_worker_source_sha256": previous["scripts/sr_f1/run_worker.py"],
        "registration_sha256": file_hash(root / "orchestration/REGISTRATION.json"),
        "scientific_protocol_unchanged": True,
        "original_once_consumed_preserved": True,
        "engine_restart_mode": "FULL_REGISTERED_CONTINUOUS4_SPLIT2_PLUS2",
        "gradient_reduction": "ORIGINAL_SEQUENCE_ORDER",
        "gpu_count": gpu_count,
        "recompute_policy": "native_pure_delta_rule_nonreentrant_recompute_v1",
        "activation_storage": "LOCAL_GPU48G_CPU48G_EXACT_EXTERNAL_DISK",
        "gpu_activation_budget_bytes": 48 * 1024**3,
        "cpu_activation_budget_bytes": 48 * 1024**3,
        **{
            key: parent[key]
            for key in ("activation_spill_directory", "quota_root", "quota_reserve_bytes")
        },
        "compute_device_indices": list(range(gpu_count)),
        "qos": "soujanya-poria-startfund-2026-03",
        "gpu_worker_constraint": "highmem",
        "cpus_per_gpu": 4,
        "minimum_gpu_host_memory_gb_per_gpu": 80,
        "resource_operations": ["engine", "train", "baseline"],
        "baseline_partition_mode": "STABLE_SLOT_INDEX_MODULO",
        "baseline_slot_count": 6784,
        "baseline_requires_engine_acceptance": True,
        "authorized_at": authorized_at,
        "authorized_user_message": authorized_user_message,
        **actual_source,
        "worker_source_sha256": current["scripts/sr_f1/run_worker.py"],
        "changed_files": sorted(
            n for n in previous.keys() | current.keys() if previous.get(n) != current.get(n)
        ),
        "historical_artifact_hashes": {
            COMPUTE_PARALLEL_PREFIX + n: file_hash(root / (COMPUTE_PARALLEL_PREFIX + n))
            for n in names
        },
    }


def verify_engine_compute_parallel_repair(root, actual_source=None):
    """Authenticate new compute and baseline capabilities without reinterpreting old attempts."""
    from .prepare import source_identity

    root, _saved, before, parent = _compute_parallel_source_context(root)
    if (root / "BASELINE_PARALLEL_REPAIR.json").exists():
        raise PermissionError("Compute revision cannot replace an activated baseline source")
    path = root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json"
    repair = read_json(path)
    count = repair.get("gpu_count")
    if type(count) is not int or not 2 <= count <= 5:
        raise PermissionError("Compute GPU count must be from two through five")
    actual = source_identity() if actual_source is None else actual_source
    expected = build_engine_compute_parallel_repair(
        root,
        actual_source=actual,
        gpu_count=count,
        authorized_at=repair.get("authorized_at"),
        authorized_user_message=repair.get("authorized_user_message"),
    )
    _required(repair, expected, "Compute repair/source/history identity differs")
    if not repair["authorized_at"] or not repair["authorized_user_message"]:
        raise PermissionError("Compute repair lacks user authorization")
    previous = _checked_source_files(before.get("source_file_hashes"), "Pre-compute")
    current = _checked_source_files(actual.get("source_file_hashes"), "Compute")
    changed = set(repair["changed_files"])
    if (
        not ENGINE_COMPUTE_PARALLEL_REQUIRED_FILES.issubset(changed)
        or not changed.issubset(ENGINE_COMPUTE_PARALLEL_ALLOWED_FILES)
        or set(previous) - set(current)
    ):
        raise PermissionError("Compute repair changed unapproved source or omitted implementation")
    _required(actual.get("source_dirty_files"), [], "Compute source must be committed")
    _required(actual.get("source_tree_sha256"), object_hash(current), "Compute tree invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", actual.get("source_commit", "")):
        raise PermissionError("Compute source commit missing")
    prefix = root / COMPUTE_PARALLEL_PREFIX
    preservation = read_json(prefix / "PRESERVATION.json")
    entries = preservation.get("artifact_hashes", {})
    if not entries:
        raise PermissionError("Compute preservation empty")
    hashes = _checked_source_files(
        {n: v.get("sha256") for n, v in entries.items()}, "Compute evidence"
    )
    total = 0
    for name, checksum in hashes.items():
        preserved = bounded_path(prefix / "evidence", name)
        if (prefix / "evidence" / name).is_symlink():
            raise PermissionError("Compute evidence symlink")
        _required(file_hash(preserved), checksum, "Compute evidence changed")
        _required(
            preserved.stat().st_size, entries[name].get("bytes"), "Compute evidence size changed"
        )
        total += preserved.stat().st_size
    _required(preservation.get("files"), len(hashes), "Compute preservation count changed")
    _required(preservation.get("bytes"), total, "Compute preservation bytes changed")
    mandatory = {
        "orchestration/STATE.json",
        "orchestration/REGISTRATION.json",
        "orchestration/journal.jsonl",
        "accounting/ENGINE.jsonl",
        "technical_incidents/baseline_parallel_20261010/HANDOFF_AUTHORIZATION.json",
        "technical_incidents/baseline_parallel_20261010/HANDOFF_SUBMISSION.json",
        "technical_incidents/engine_memory_20261010/ZERO_UPDATE_REUSE_ACTIVATED.json",
        "technical_incidents/engine_memory_20261010/ZERO_UPDATE_REUSE_PROCESS.json",
    }
    track = "engineering/engine/natural/continuous"
    mandatory |= {track + "/RUN_MANIFEST.json", track + "/checkpoints/LATEST.json"}
    if not mandatory.issubset(hashes):
        raise PermissionError("Compute preservation lacks history/checkpoint/handoff")
    _required(
        hashes["orchestration/REGISTRATION.json"],
        repair["registration_sha256"],
        "Compute registration changed",
    )
    state = read_json(prefix / "evidence/orchestration/STATE.json")
    if state.get("test_sealed") is not True or any(
        t.get("attempts")
        for n, t in state.get("tasks", {}).items()
        if n not in {"COMMON_START", "ENGINE"}
    ):
        raise PermissionError("Compute repair requires sealed pre-science history")
    attempts = state["tasks"]["ENGINE"]["attempts"]
    maintained = attempts[-1]
    terminal = read_json(prefix / "TERMINAL_JOBS.json")
    for key, value in {
        "controller_terminal": True,
        "gpu_terminal": True,
        "controller_queue_empty": True,
        "gpu_queue_empty": True,
        "gpu_job_id": str(maintained["job_id"]),
    }.items():
        _required(terminal.get(key), value, "Compute maintenance not terminal: " + key)
    for key in ("controller_terminal_state", "gpu_terminal_state"):
        if terminal.get(key) not in {
            "CANCELLED",
            "COMPLETED",
            "FAILED",
            "TIMEOUT",
            "PREEMPTED",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
        }:
            raise PermissionError("Compute maintenance terminal state unknown")
    handoff = read_json(
        prefix / "evidence/technical_incidents/baseline_parallel_20261010/HANDOFF_SUBMISSION.json"
    )
    if handoff.get("returncode") != 0 or not re.fullmatch(
        r"[0-9]+(?:;[^\s;]+)?", handoff.get("stdout", "").strip()
    ):
        raise PermissionError("Superseded baseline handoff job ID unknown")
    _required(
        terminal.get("controller_job_id"),
        handoff["stdout"].strip().split(";")[0],
        "Wrong baseline handoff controller terminated",
    )
    for name in ("ZERO_UPDATE_REUSE_ACTIVATED.json", "ZERO_UPDATE_REUSE_PROCESS.json"):
        relative = "technical_incidents/engine_memory_20261010/" + name
        _required(file_hash(root / relative), hashes[relative], "Consumed recovery marker changed")
    old_auth = read_json(
        prefix
        / "evidence/technical_incidents/baseline_parallel_20261010/HANDOFF_AUTHORIZATION.json"
    )
    candidate = root / "code_baseline_parallel_candidate_20261010"
    if candidate.is_symlink():
        raise PermissionError("Superseded candidate cannot be a symlink")
    _required(
        file_hash(candidate / "SOURCE_DEPLOYMENT.json"),
        old_auth.get("candidate_source_deployment_sha256"),
        "Superseded candidate identity changed",
    )
    candidate_source = read_json(candidate / "SOURCE_DEPLOYMENT.json")
    _required(
        _preserved_source_files(candidate, include_amendments=True),
        candidate_source.get("source_file_hashes"),
        "Superseded candidate source changed",
    )
    latest = read_json(prefix / "evidence" / track / "checkpoints/LATEST.json")
    step = latest.get("step")
    if type(step) is not int or not 0 <= step <= 4:
        raise PermissionError("Compute maintenance checkpoint step invalid")
    commit_name = track + f"/checkpoints/commit-{step:02d}.json"
    if commit_name not in hashes:
        raise PermissionError("Compute checkpoint commit missing")
    _required(
        latest, read_json(prefix / "evidence" / commit_name), "Compute checkpoint commit differs"
    )
    checkpoint = bounded_path(prefix / "evidence" / track / "checkpoints", latest.get("path", ""))
    _required(
        hashes.get(str(checkpoint.relative_to(prefix / "evidence"))),
        latest.get("sha256"),
        "Compute checkpoint not preserved",
    )
    if not re.fullmatch(r"[0-9a-f]{64}", latest.get("state_hash", "")):
        raise PermissionError("Compute checkpoint full state hash missing")
    raw_names = {name for name in hashes if name.startswith(track + "/rollouts/")}
    completed_raw = {
        track + f"/rollouts/{n:02d}-{slot:02d}-{sample}.json"
        for n in range(1, step + 1)
        for slot in range(16)
        for sample in range(8)
    }
    if not completed_raw.issubset(raw_names):
        raise PermissionError("Compute preserved completed-step raw slots incomplete")
    review = read_json(prefix / "CPU_STATE_REVIEW.json")
    for key, value in {
        "status": "PASS_PRESERVED_FULL_STATE",
        "committed_logical_step": step,
        "checkpoint_state_hash": latest["state_hash"],
        "checkpoint_file_sha256": latest["sha256"],
        "raw_record_hashes_verified": True,
        "optimizer_state_verified": True,
        "rng_state_verified": True,
        "learning_rate": 1e-4,
        "controller_identity_verified": True,
        "journal_integrity": True,
    }.items():
        _required(review.get(key), value, "Compute CPU state review differs: " + key)
    recovery = read_json(prefix / "RECOVERY_REVIEW.json")
    for key, value in {
        "status": "PASS_FULL_ENGINE_RESTART",
        "science_attempts": 0,
        "test_sealed": True,
        "full_engine_restart": True,
        "no_partial_gradient_reuse": True,
        "no_old_once_reuse": True,
        "old_handoff_superseded": True,
        "old_candidate_preserved": True,
    }.items():
        _required(recovery.get(key), value, "Compute recovery differs: " + key)
    probe = read_json(prefix / "GPU_PROBE.json")
    for key, value in {
        "status": "PASS",
        "numerical_equivalence": True,
        "exact_ordered_gradient_accumulation": True,
        "actual_compute_gpu_count": count,
    }.items():
        _required(probe.get(key), value, "Compute CUDA probe differs: " + key)
    numerical = {
        n
        for n in current
        if n
        in {
            "src/sr_f1/runtime.py",
            "src/sr_f1/training.py",
            "src/sr_f1/compute_parallel.py",
            "src/sr_f1/compute_workers.py",
            "src/sr_f1/recompute.py",
        }
    }
    _required(
        probe.get("source_file_hashes"),
        {n: current[n] for n in sorted(numerical)},
        "Compute CUDA probe source differs",
    )
    real_binding = probe.get("real_model_probe", {})
    real_name = COMPUTE_PARALLEL_PREFIX + "REAL_MODEL_PROBE.json"
    _required(set(real_binding), {"path", "sha256"}, "Real-model probe binding missing")
    _required(real_binding.get("path"), real_name, "Real-model probe path changed")
    real_path = bounded_path(root, real_name)
    _required(file_hash(real_path), real_binding.get("sha256"), "Real-model probe bytes changed")
    real_probe = read_json(real_path)
    for key, value in {
        "status": "PASS",
        "numerical_equivalence": True,
        "all_four_actual_compute": True,
        "actual_compute_gpu_count": count,
    }.items():
        _required(real_probe.get(key), value, "Real-model probe acceptance differs: " + key)
    _required(
        real_probe.get("source_hashes"),
        {n: current[n] for n in sorted(numerical)},
        "Real-model probe source differs",
    )
    _required(
        real_probe.get("synthetic_768_smoke", {}).get("status"),
        "PASS",
        "Real-model 768-token memory probe not passed",
    )
    return {
        **repair,
        "repair_sha256": file_hash(path),
        "engine_multigpu_repair": parent,
        "maintained_attempt_id": maintained["attempt_id"],
        "maintained_job_id": str(maintained["job_id"]),
    }


def _required(value, expected, message):
    if value != expected:
        raise PermissionError(message)


def _checked_source_files(value, name):
    if not isinstance(value, dict) or not value:
        raise PermissionError(name + " source-file inventory is missing")
    for relative, expected in value.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or str(Path(relative)) != relative
            or not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise PermissionError(name + " source-file inventory is invalid")
    return value


def _preserved_source_files(snapshot, *, include_amendments=False):
    """Apply the original source_identity inventory policy to the preserved copy."""
    files = {}
    for folder in ("src/sr_f1", "scripts/sr_f1", "src/mm_core", "src/mm_dev"):
        for path in sorted((snapshot / folder).rglob("*")):
            if path.is_symlink():
                raise PermissionError("Preserved original source contains unsafe links")
            if not path.is_file() or path.suffix not in (".py", ".sh", ".sbatch"):
                continue
            if not path.resolve().is_relative_to(snapshot):
                raise PermissionError("Preserved original source contains unsafe links")
            files[str(path.relative_to(snapshot))] = file_hash(path)
    for name in ("pyproject.toml", "uv.lock"):
        path = snapshot / name
        if path.is_file():
            if path.is_symlink():
                raise PermissionError("Preserved original dependency identity is a symlink")
            files[name] = file_hash(path)
    if include_amendments:
        for path in sorted((snapshot / "docs/sr_f1/amendments").rglob("*")):
            if path.is_symlink():
                raise PermissionError("Preserved amendment source contains unsafe links")
            if path.is_file() and path.suffix in (".md", ".json"):
                files[str(path.relative_to(snapshot))] = file_hash(path)
    return files


def _verify_original_qos_scope_repair(root, actual_source=None):
    """Authenticate the user's scheduler-only correction without rewriting a freeze."""
    from .prepare import source_identity

    root = Path(root).resolve(strict=True)
    receipt_path = root / "QOS_SCOPE_REPAIR.json"
    repair = read_json(receipt_path)
    freeze = read_json(root / "EXECUTION_FREEZE.json")
    original = read_json(root / "SOURCE_AND_RENDER_MANIFEST.json")
    permission = read_json(root / "manifests/ALLOCATION_PERMISSION.json")
    _required(freeze.get("experiment"), "SR-F1.1-20261010", "Wrong QoS repair experiment")
    _required(freeze.get("status"), "FROZEN", "QoS repair requires a frozen execution")
    _required(
        freeze.get("artifact_hashes", {}).get("SOURCE_AND_RENDER_MANIFEST.json"),
        file_hash(root / "SOURCE_AND_RENDER_MANIFEST.json"),
        "Original frozen source manifest changed",
    )
    expected = {
        "repair_id": QOS_SCOPE_REPAIR_ID,
        "status": "USER_AUTHORIZED_QOS_SCHEDULER_ONLY",
        "plan_id": PLAN_ID,
        "run_root": str(root),
        "original_execution_freeze_sha256": file_hash(root / "EXECUTION_FREEZE.json"),
        "original_source_tree_sha256": original["source_tree_sha256"],
        "original_source_commit": original["source_commit"],
        "permission_sha256": file_hash(root / "manifests/ALLOCATION_PERMISSION.json"),
        "capacity_mode": "TEACHER_QOS_SCHEDULER_ONLY",
        "qos": "soujanya-poria-startfund-2026-03",
        "gpus_per_worker": 1,
        "scientific_protocol_unchanged": True,
        "new_generation_count_before_activation": 0,
        "no_gpu_attempts_before_activation": True,
        "preserved_source_relative_path": "code_before_qos_scope_20261010",
    }
    for key, value in expected.items():
        _required(repair.get(key), value, "QoS scope repair differs: " + key)
    _required(permission.get("qos"), expected["qos"], "Teacher GPU QoS changed")
    _required(permission.get("gpus_per_worker"), 1, "GPU workers must remain single-card")
    if not repair.get("authorized_user_message") or not repair.get("authorized_at"):
        raise PermissionError("Explicit user QoS-scope authorization is absent")
    artifacts = repair.get("pre_activation_artifact_hashes", {})
    prefix = "technical_incidents/qos_scope_20261010/"
    required = {prefix + name for name in ("STATE_BEFORE.json", "CONTROLLER_TERMINAL.json")}
    if not required.issubset(artifacts):
        raise PermissionError("QoS repair lacks pre-activation evidence")
    for relative, expected_hash in artifacts.items():
        _required(
            file_hash(bounded_path(root, relative)),
            expected_hash,
            "QoS pre-activation evidence changed",
        )
    prior_state = read_json(root / (prefix + "STATE_BEFORE.json"))
    if (
        prior_state.get("test_sealed") is not True
        or not prior_state.get("tasks")
        or any(task.get("attempts") for task in prior_state["tasks"].values())
    ):
        raise PermissionError("QoS correction was not registered before GPU attempts")
    terminal = read_json(root / (prefix + "CONTROLLER_TERMINAL.json"))
    if terminal.get("old_controller_terminal") is not True:
        raise PermissionError("Previous controller termination was not verified")
    before = _checked_source_files(original.get("source_file_hashes"), "Original")
    snapshot = bounded_path(root, repair["preserved_source_relative_path"])
    _required(
        _preserved_source_files(snapshot, include_amendments=True),
        before,
        "Preserved pre-QoS source differs from the original freeze",
    )
    actual = source_identity() if actual_source is None else actual_source
    after = _checked_source_files(actual.get("source_file_hashes"), "Actual")
    if actual.get("source_dirty_files"):
        raise PermissionError("QoS repair requires committed source")
    _required(actual.get("source_tree_sha256"), object_hash(after), "Invalid actual source hash")
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    _required(set(changed), QOS_SCOPE_ALLOWED_FILES, "QoS repair changed scientific source")
    _required(repair.get("changed_files"), changed, "QoS source change inventory differs")
    for key in ("source_commit", "source_tree_sha256", "source_file_hashes"):
        _required(repair.get(key), actual.get(key), "QoS deployed source differs: " + key)
    if not re.fullmatch(r"[0-9a-f]{40}", actual.get("source_commit") or ""):
        raise PermissionError("QoS repair source commit is missing")
    return {
        "capacity_mode": expected["capacity_mode"],
        "qos": expected["qos"],
        "repair_sha256": file_hash(receipt_path),
        "source_commit": actual["source_commit"],
        "original_execution_freeze_sha256": expected["original_execution_freeze_sha256"],
    }


def verify_qos_scope_repair(root, actual_source=None):
    """Keep the original QoS receipt valid through an authenticated later repair."""
    if (Path(root) / "ENGINE_MEMORY_REPAIR.json").exists():
        repair = verify_engine_memory_repair(root, actual_source=actual_source)
        return {
            "capacity_mode": "TEACHER_QOS_SCHEDULER_ONLY",
            "qos": "soujanya-poria-startfund-2026-03",
            "repair_sha256": repair["previous_qos_repair_sha256"],
            "source_commit": repair["source_commit"],
            "original_execution_freeze_sha256": repair["original_execution_freeze_sha256"],
            "engine_memory_repair_sha256": repair["repair_sha256"],
        }
    return _verify_original_qos_scope_repair(root, actual_source=actual_source)


def verify_engine_memory_repair(root, actual_source=None):
    """Authenticate the original memory repair through a later bounded-storage repair."""
    if (Path(root) / "ENGINE_STORAGE_REPAIR.json").exists():
        storage = verify_engine_storage_repair(root, actual_source=actual_source)
        return {
            "repair_id": ENGINE_MEMORY_REPAIR_ID,
            "repair_sha256": file_hash(Path(root) / "ENGINE_MEMORY_REPAIR.json"),
            "source_commit": storage["source_commit"],
            "original_execution_freeze_sha256": storage["original_execution_freeze_sha256"],
            "previous_qos_repair_sha256": file_hash(Path(root) / "QOS_SCOPE_REPAIR.json"),
            "gpu_worker_host_memory_gb": storage["minimum_gpu_host_memory_gb"],
            "engine_storage_repair_sha256": storage["repair_sha256"],
        }
    return _verify_original_engine_memory_repair(root, actual_source=actual_source)


def _verify_original_engine_memory_repair(root, actual_source=None):
    """Bind activation storage and step-zero recovery to preserved failure evidence.

    Only historical copies are permanent identity inputs. Current STATE/LATEST
    legitimately advance after the one-time scheduler and ENGINE activation.
    """
    from .prepare import source_identity

    root = Path(root).resolve(strict=True)
    repair_path = root / "ENGINE_MEMORY_REPAIR.json"
    repair = read_json(repair_path)
    historical_root = bounded_path(root, "code_before_engine_memory_20261010")
    historical = read_json(historical_root / "SOURCE_DEPLOYMENT.json")
    _required(
        _preserved_source_files(historical_root, include_amendments=True),
        historical.get("source_file_hashes"),
        "Preserved pre-memory source changed",
    )
    parent = _verify_original_qos_scope_repair(root, actual_source=historical)
    expected = {
        "repair_id": ENGINE_MEMORY_REPAIR_ID,
        "status": "AUTHORIZED_TECHNICAL_MEMORY_REPAIR",
        "plan_id": PLAN_ID,
        "run_root": str(root),
        "original_execution_freeze_sha256": parent["original_execution_freeze_sha256"],
        "previous_qos_repair_sha256": file_hash(root / "QOS_SCOPE_REPAIR.json"),
        "previous_source_commit": historical["source_commit"],
        "previous_source_tree_sha256": historical["source_tree_sha256"],
        "preserved_source_relative_path": historical_root.name,
        "scientific_protocol_unchanged": True,
        "optimizer_updates_before_repair": 0,
        "gpu_worker_host_memory_gb": 384,
        "activation_storage": "CPU_SAVED_NONPARAMETER_TENSORS_EXACT_DTYPE",
    }
    for key, value in expected.items():
        _required(repair.get(key), value, "ENGINE memory repair differs: " + key)
    if not repair.get("authorized_user_message") or not repair.get("authorized_at"):
        raise PermissionError("ENGINE memory repair lacks user authorization provenance")
    prefix = "technical_incidents/engine_memory_20261010/"
    required = {
        prefix + n
        for n in ("EVIDENCE_MANIFEST.json", "TERMINAL_JOBS.json", "CPU_STATE_REVIEW.json")
    }
    artifacts = _checked_source_files(repair.get("historical_artifact_hashes"), "Historical")
    if not required.issubset(artifacts) or any(not n.startswith(prefix) for n in artifacts):
        raise PermissionError("ENGINE memory repair lacks its bounded historical evidence")
    for name, expected_hash in artifacts.items():
        _required(file_hash(bounded_path(root, name)), expected_hash, "Repair history changed")
    manifest = read_json(root / (prefix + "EVIDENCE_MANIFEST.json"))
    _required(manifest.get("status"), "BYTE_VERIFIED_PRESERVED", "Failure preservation absent")
    files = manifest.get("files", {})
    if not files:
        raise PermissionError("Failure evidence inventory is empty")
    for name, entry in files.items():
        saved = bounded_path(root / (prefix + "evidence"), name)
        _required(file_hash(saved), entry["sha256"], "Preserved failure artifact changed")
        _required(saved.stat().st_size, entry["bytes"], "Preserved artifact size changed")
    terminal = read_json(root / (prefix + "TERMINAL_JOBS.json"))
    if not (
        terminal.get("old_controller_terminal") is True
        and terminal.get("old_engine_terminal") is True
    ):
        raise PermissionError("Previous allocations are not verified terminal")
    state = read_json(root / (prefix + "evidence/orchestration/STATE.json"))
    if state.get("test_sealed") is not True or state["tasks"]["ENGINE"]["status"] != "BLOCKED":
        raise PermissionError("ENGINE repair does not preserve a sealed technical failure")
    if any(
        t.get("attempts") for k, t in state["tasks"].items() if k not in {"COMMON_START", "ENGINE"}
    ):
        raise PermissionError("ENGINE repair was not established before downstream work")
    review = read_json(root / (prefix + "CPU_STATE_REVIEW.json"))
    for key, value in {
        "status": "PASS_ZERO_UPDATE_FULL_STATE",
        "committed_logical_step": 0,
        "physical_optimizer_updates": 0,
        "optimizer_empty": True,
        "raw_rollouts": 128,
        "raw_record_hashes_verified": True,
        "learning_rate": 1e-4,
    }.items():
        _required(review.get(key), value, "Invalid zero-update state review: " + key)
    recovery = repair.get("zero_update_recovery", {})
    track = "engineering/engine/natural/continuous"
    for key, value in {
        "staged_track_relative_path": track,
        "archived_track_relative_path": prefix + "evidence/" + track,
        "original_checkpoint_state_hash": review.get("checkpoint_state_hash"),
        "original_raw_count": 128,
        "engine_mode": "natural",
        "resume_step": 0,
    }.items():
        _required(recovery.get(key), value, "ENGINE recovery boundary differs: " + key)
    restore = _checked_source_files(recovery.get("artifact_hashes"), "Recovery")
    expected_restore = {
        name: entry["sha256"]
        for name, entry in files.items()
        if name == track + "/RUN_MANIFEST.json"
        or name.startswith(track + "/checkpoints/")
        or name.startswith(track + "/rollouts/")
    }
    _required(restore, expected_restore, "Recovery inventory does not match preserved state")
    actual = source_identity() if actual_source is None else actual_source
    before = _checked_source_files(historical.get("source_file_hashes"), "Previous")
    after = _checked_source_files(actual.get("source_file_hashes"), "Actual")
    if actual.get("source_dirty_files"):
        raise PermissionError("ENGINE memory repair requires committed source")
    _required(actual.get("source_tree_sha256"), object_hash(after), "Actual source hash invalid")
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    _required(set(changed), ENGINE_MEMORY_ALLOWED_FILES, "Memory repair changed scientific source")
    _required(repair.get("changed_files"), changed, "Memory repair source inventory differs")
    for key in ("source_commit", "source_tree_sha256", "source_file_hashes"):
        _required(repair.get(key), actual.get(key), "Memory repair deployed source differs: " + key)
    if not re.fullmatch(r"[0-9a-f]{40}", actual.get("source_commit") or ""):
        raise PermissionError("Memory repair source commit is missing")
    return {
        "repair_id": ENGINE_MEMORY_REPAIR_ID,
        "repair_sha256": file_hash(repair_path),
        "source_commit": actual["source_commit"],
        "original_execution_freeze_sha256": expected["original_execution_freeze_sha256"],
        "previous_qos_repair_sha256": expected["previous_qos_repair_sha256"],
        "gpu_worker_host_memory_gb": expected["gpu_worker_host_memory_gb"],
    }


def verify_engine_storage_repair(root, actual_source=None):
    """Preserve the storage receipt's identity through an authenticated I/O repair."""
    if (Path(root) / "ENGINE_IO_REPAIR.json").exists():
        io = verify_engine_io_repair(root, actual_source=actual_source)
        return {
            "repair_id": ENGINE_STORAGE_REPAIR_ID,
            "repair_sha256": file_hash(Path(root) / "ENGINE_STORAGE_REPAIR.json"),
            "source_commit": io["source_commit"],
            "original_execution_freeze_sha256": io["original_execution_freeze_sha256"],
            "gpu_worker_constraint": io["gpu_worker_constraint"],
            "minimum_gpu_host_memory_gb": io["minimum_gpu_host_memory_gb"],
            "engine_io_repair_sha256": io["repair_sha256"],
        }
    return _verify_original_engine_storage_repair(root, actual_source=actual_source)


def _verify_original_engine_storage_repair(root, actual_source=None):
    """Bind bounded exact storage and supported Slurm resources to unstarted history."""
    from .prepare import source_identity

    root = Path(root).resolve(strict=True)
    repair_path = root / "ENGINE_STORAGE_REPAIR.json"
    repair = read_json(repair_path)
    saved_source = bounded_path(root, "code_before_engine_storage_20261010")
    before = read_json(saved_source / "SOURCE_DEPLOYMENT.json")
    _required(
        _preserved_source_files(saved_source, include_amendments=True),
        before.get("source_file_hashes"),
        "Pre-storage source changed",
    )
    parent = _verify_original_engine_memory_repair(root, actual_source=before)
    expected = {
        "repair_id": ENGINE_STORAGE_REPAIR_ID,
        "status": "AUTHORIZED_TECHNICAL_STORAGE_REPAIR",
        "plan_id": PLAN_ID,
        "run_root": str(root),
        "original_execution_freeze_sha256": parent["original_execution_freeze_sha256"],
        "previous_engine_memory_repair_sha256": parent["repair_sha256"],
        "previous_source_commit": before["source_commit"],
        "previous_source_tree_sha256": before["source_tree_sha256"],
        "preserved_source_relative_path": saved_source.name,
        "scientific_protocol_unchanged": True,
        "gpu_worker_constraint": "highmem",
        "minimum_gpu_host_memory_gb": 80,
        "cpu_activation_budget_bytes": 48 * 1024**3,
        "activation_storage": "BOUNDED_CPU_EXACT_DTYPE_DISK_SPILL",
        "superseded_unstarted_attempt_id": "ENGINE_attempt0001",
        "superseded_unstarted_job_id": "196139",
    }
    for key, value in expected.items():
        _required(repair.get(key), value, "ENGINE storage repair differs: " + key)
    if not repair.get("authorized_user_message") or not repair.get("authorized_at"):
        raise PermissionError("Storage repair lacks authorization provenance")
    prefix = "technical_incidents/host_memory_20261010/"
    required = {
        prefix + n
        for n in ("EVIDENCE_MANIFEST.json", "TERMINAL_JOBS.json", "HOST_STATE_REVIEW.json")
    }
    artifacts = _checked_source_files(repair.get("historical_artifact_hashes"), "Storage history")
    if not required.issubset(artifacts) or any(not n.startswith(prefix) for n in artifacts):
        raise PermissionError("Storage history inventory is missing or unbounded")
    for name, expected_hash in artifacts.items():
        _required(file_hash(bounded_path(root, name)), expected_hash, "Storage history changed")
    manifest = read_json(root / (prefix + "EVIDENCE_MANIFEST.json"))
    _required(manifest.get("status"), "BYTE_VERIFIED_PRESERVED", "Storage preservation missing")
    if not manifest.get("files"):
        raise PermissionError("Storage preservation inventory empty")
    for name, entry in manifest["files"].items():
        p = bounded_path(root / (prefix + "evidence"), name)
        _required(file_hash(p), entry["sha256"], "Preserved storage artifact changed")
        _required(p.stat().st_size, entry["bytes"], "Preserved storage artifact size changed")
    terminal = read_json(root / (prefix + "TERMINAL_JOBS.json"))
    for key in ("controller_terminal", "gpu_terminal", "gpu_never_started"):
        _required(terminal.get(key), True, "Unstarted storage replacement is not terminal")
    review = read_json(root / (prefix + "HOST_STATE_REVIEW.json"))
    for key, value in {
        "status": "PASS_UNCONSUMED_ZERO_UPDATE_STATE",
        "committed_logical_step": 0,
        "physical_optimizer_updates": 0,
        "raw_rollouts": 128,
        "staged_artifact_hashes_unchanged": True,
        "recovery_activation_exists": False,
        "recovery_process_exists": False,
        "physical_process_files": 0,
        "superseded_gpu_job_id": "196139",
        "superseded_gpu_elapsed_seconds": 0,
        "science_attempts": 0,
        "test_sealed": True,
    }.items():
        _required(review.get(key), value, "Invalid unstarted state review: " + key)
    original_recovery = read_json(root / "ENGINE_MEMORY_REPAIR.json")["zero_update_recovery"]
    _required(
        review.get("staged_artifact_hashes"),
        original_recovery["artifact_hashes"],
        "Storage retry does not preserve original step-zero state and raw records",
    )
    for name, expected_hash in original_recovery["artifact_hashes"].items():
        _required(
            manifest["files"].get(name, {}).get("sha256"),
            expected_hash,
            "Storage history does not match authenticated recovery bytes",
        )
    actual = source_identity() if actual_source is None else actual_source
    previous_files = _checked_source_files(before.get("source_file_hashes"), "Pre-storage")
    current_files = _checked_source_files(actual.get("source_file_hashes"), "Storage actual")
    if actual.get("source_dirty_files"):
        raise PermissionError("Storage repair requires committed source")
    _required(actual.get("source_tree_sha256"), object_hash(current_files), "Storage tree invalid")
    changed = sorted(
        k
        for k in previous_files.keys() | current_files.keys()
        if previous_files.get(k) != current_files.get(k)
    )
    _required(
        set(changed), ENGINE_STORAGE_ALLOWED_FILES, "Storage repair changed scientific source"
    )
    _required(repair.get("changed_files"), changed, "Storage repair file inventory differs")
    for key in ("source_commit", "source_tree_sha256", "source_file_hashes"):
        _required(repair.get(key), actual.get(key), "Storage source identity differs: " + key)
    if not re.fullmatch(r"[0-9a-f]{40}", actual.get("source_commit") or ""):
        raise PermissionError("Storage repair source commit missing")
    return {
        "repair_id": ENGINE_STORAGE_REPAIR_ID,
        "repair_sha256": file_hash(repair_path),
        "source_commit": actual["source_commit"],
        "original_execution_freeze_sha256": expected["original_execution_freeze_sha256"],
        "gpu_worker_constraint": expected["gpu_worker_constraint"],
        "minimum_gpu_host_memory_gb": expected["minimum_gpu_host_memory_gb"],
    }


def verify_engine_io_repair(root, actual_source=None):
    """Keep the original I/O receipt's identity through later GPU storage repair."""
    if (Path(root) / "ENGINE_MULTIGPU_REPAIR.json").exists():
        multi = verify_engine_multigpu_repair(root, actual_source=actual_source)
        original = read_json(Path(root) / "ENGINE_IO_REPAIR.json")
        return {
            "repair_id": ENGINE_IO_REPAIR_ID,
            "repair_sha256": file_hash(Path(root) / "ENGINE_IO_REPAIR.json"),
            "source_commit": multi["source_commit"],
            **{
                key: original[key]
                for key in (
                    "original_execution_freeze_sha256",
                    "previous_engine_storage_repair_sha256",
                    "failed_attempt_id",
                    "failed_job_id",
                    "gpu_worker_constraint",
                    "minimum_gpu_host_memory_gb",
                    "activation_spill_directory",
                    "quota_root",
                    "quota_reserve_bytes",
                    "cpu_activation_budget_bytes",
                )
            },
            "engine_multigpu_repair_sha256": multi["repair_sha256"],
        }
    return _verify_original_engine_io_repair(root, actual_source=actual_source)


def _verify_original_engine_io_repair(root, actual_source=None):
    """Authenticate external derived storage and an interrupted ENGINE restart.

    Original receipts, recovery bytes and consumed markers remain immutable.
    Only preserved STATE/LATEST participate: a subsequent valid trial may advance.
    """
    from .prepare import source_identity

    root = Path(root).resolve(strict=True)
    repair_path = root / "ENGINE_IO_REPAIR.json"
    repair = read_json(repair_path)
    declared_source = root / "code_before_engine_io_20261010"
    if declared_source.is_symlink():
        raise PermissionError("Pre-I/O source snapshot must be a real directory")
    saved_source = bounded_path(root, declared_source.name)
    before = read_json(saved_source / "SOURCE_DEPLOYMENT.json")
    _required(
        _preserved_source_files(saved_source, include_amendments=True),
        before.get("source_file_hashes"),
        "Pre-I/O source changed",
    )
    parent = _verify_original_engine_storage_repair(root, actual_source=before)
    quota_root = "/projects/_hdd/varunhdd"
    expected = {
        "repair_id": ENGINE_IO_REPAIR_ID,
        "status": "AUTHORIZED_TECHNICAL_IO_REPAIR",
        "plan_id": PLAN_ID,
        "run_root": str(root),
        "original_execution_freeze_sha256": parent["original_execution_freeze_sha256"],
        "previous_engine_storage_repair_sha256": parent["repair_sha256"],
        "previous_source_commit": before["source_commit"],
        "previous_source_tree_sha256": before["source_tree_sha256"],
        "preserved_source_relative_path": saved_source.name,
        "scientific_protocol_unchanged": True,
        "old_onetime_consumed_preserved": True,
        "failed_attempt_id": "ENGINE_attempt0002",
        "failed_job_id": "196156",
        "gpu_worker_constraint": "highmem",
        "minimum_gpu_host_memory_gb": 80,
        "cpu_activation_budget_bytes": 48 * 1024**3,
        "activation_storage": "BOUNDED_CPU_EXACT_DTYPE_EXTERNAL_DISK_SPILL",
        "quota_root": quota_root,
        "activation_spill_directory": f"{quota_root}/louis-ssvc/{root.name}_activation_offload",
        "quota_reserve_bytes": 10 * 1024**3,
    }
    for key, value in expected.items():
        _required(repair.get(key), value, "ENGINE I/O repair differs: " + key)
    if not repair.get("authorized_user_message") or not repair.get("authorized_at"):
        raise PermissionError("I/O repair lacks authorization provenance")
    prefix = "technical_incidents/io_exit120_20261010/"
    required = {
        prefix + name
        for name in (
            "PRESERVATION.json",
            "TERMINAL_JOBS.json",
            "CPU_STATE_REVIEW.json",
            "RECOVERY_REVIEW.json",
        )
    }
    artifacts = _checked_source_files(repair.get("historical_artifact_hashes"), "I/O history")
    if not required.issubset(artifacts) or any(not name.startswith(prefix) for name in artifacts):
        raise PermissionError("I/O history inventory is missing or unbounded")
    for name, expected_hash in artifacts.items():
        _required(file_hash(bounded_path(root, name)), expected_hash, "I/O history changed")
    preservation = read_json(root / (prefix + "PRESERVATION.json"))
    files = preservation.get("artifact_hashes")
    if not isinstance(files, dict) or not files:
        raise PermissionError("I/O preservation inventory empty")
    hashes = _checked_source_files(
        {
            name: entry.get("sha256") if isinstance(entry, dict) else None
            for name, entry in files.items()
        },
        "I/O preserved",
    )
    _required(preservation.get("files"), len(hashes), "I/O preservation file count differs")
    preserved_bytes = 0
    evidence = root / (prefix + "evidence")
    for name, expected_hash in hashes.items():
        saved = bounded_path(evidence, name)
        if (evidence / name).is_symlink():
            raise PermissionError("Preserved I/O artifact is a symlink")
        _required(file_hash(saved), expected_hash, "Preserved I/O artifact changed")
        size = files[name].get("bytes")
        if type(size) is not int or size < 0:
            raise PermissionError("Preserved I/O artifact size invalid")
        _required(saved.stat().st_size, size, "Preserved I/O artifact size changed")
        preserved_bytes += size
    _required(preservation.get("bytes"), preserved_bytes, "I/O preservation byte count differs")
    terminal = read_json(root / (prefix + "TERMINAL_JOBS.json"))
    for key, value in {
        "controller_terminal": True,
        "gpu_terminal": True,
        "controller_job_id": "196155",
        "gpu_job_id": "196156",
        "gpu_exit_code": "120:0",
    }.items():
        _required(
            terminal.get(key), value, "I/O failed allocation is not verified terminal: " + key
        )
    original_recovery = read_json(root / "ENGINE_MEMORY_REPAIR.json")["zero_update_recovery"]
    original_hashes = _checked_source_files(original_recovery.get("artifact_hashes"), "Original")
    for name, expected_hash in original_hashes.items():
        _required(hashes.get(name), expected_hash, "I/O history changed original recovery bytes")
    review = read_json(root / (prefix + "CPU_STATE_REVIEW.json"))
    for key, value in {
        "status": "PASS_ZERO_UPDATE_FULL_STATE",
        "committed_logical_step": 0,
        "physical_optimizer_updates": 0,
        "optimizer_empty": True,
        "raw_rollouts": 128,
        "raw_record_hashes_verified": True,
        "learning_rate": 1e-4,
        "checkpoint_state_hash": original_recovery["original_checkpoint_state_hash"],
    }.items():
        _required(review.get(key), value, "Invalid I/O zero-update review: " + key)
    recovery = read_json(root / (prefix + "RECOVERY_REVIEW.json"))
    for key, value in {
        "status": "PASS_INTERRUPTED_ENGINE_RESTART",
        "science_attempts": 0,
        "test_sealed": True,
        "once_activation_consumed": True,
        "once_process_consumed": True,
        "journal_integrity": True,
        "original_artifact_hashes_unchanged": True,
    }.items():
        _required(recovery.get(key), value, "Invalid I/O recovery review: " + key)
    state_name = "orchestration/STATE.json"
    if state_name not in hashes:
        raise PermissionError("I/O history lacks preserved orchestration state")
    state = read_json(evidence / state_name)
    if state.get("test_sealed") is not True or any(
        task.get("attempts")
        for name, task in state.get("tasks", {}).items()
        if name not in {"COMMON_START", "ENGINE"}
    ):
        raise PermissionError("I/O repair does not preserve sealed pre-science history")
    attempts = state.get("tasks", {}).get("ENGINE", {}).get("attempts", [])
    if not any(
        attempt.get("attempt_id") == expected["failed_attempt_id"]
        and str(attempt.get("job_id")) == expected["failed_job_id"]
        for attempt in attempts
    ):
        raise PermissionError("I/O history does not include the failed ENGINE allocation")
    marker_prefix = "technical_incidents/engine_memory_20261010/"
    marker_names = [
        marker_prefix + name
        for name in ("ZERO_UPDATE_REUSE_ACTIVATED.json", "ZERO_UPDATE_REUSE_PROCESS.json")
    ]
    for name in marker_names:
        if name not in hashes:
            raise PermissionError("I/O history lacks consumed recovery markers")
        _required(file_hash(bounded_path(root, name)), hashes[name], "Consumed marker changed")
    activation, process = [read_json(evidence / name) for name in marker_names]
    memory_sha = file_hash(root / "ENGINE_MEMORY_REPAIR.json")
    for key, value in {
        "plan_id": PLAN_ID,
        "repair_sha256": memory_sha,
        "staged_track_relative_path": original_recovery["staged_track_relative_path"],
        "archived_track_relative_path": original_recovery["archived_track_relative_path"],
        "original_raw_count": 128,
        "engine_mode": "natural",
        "resume_step": 0,
        "original_checkpoint_state_hash": original_recovery["original_checkpoint_state_hash"],
        "staged_inventory_sha256": object_hash(original_hashes),
    }.items():
        _required(activation.get(key), value, "Consumed activation identity changed: " + key)
    for key, value in {
        "repair_sha256": memory_sha,
        "activation_sha256": hashes[marker_names[0]],
        "resume_step": 0,
    }.items():
        _required(process.get(key), value, "Consumed process identity changed: " + key)
    if any(
        type(pid) is not int or pid <= 0
        for pid in (
            activation.get("activating_pid"),
            process.get("pid"),
        )
    ):
        raise PermissionError("Consumed ENGINE process identity missing")
    actual = source_identity() if actual_source is None else actual_source
    previous_files = _checked_source_files(before.get("source_file_hashes"), "Pre-I/O")
    current_files = _checked_source_files(actual.get("source_file_hashes"), "I/O actual")
    _required(actual.get("source_dirty_files"), [], "I/O repair requires committed source")
    _required(actual.get("source_tree_sha256"), object_hash(current_files), "I/O tree invalid")
    changed = sorted(
        name
        for name in previous_files.keys() | current_files.keys()
        if previous_files.get(name) != current_files.get(name)
    )
    _required(set(changed), ENGINE_IO_ALLOWED_FILES, "I/O repair changed scientific source")
    _required(repair.get("changed_files"), changed, "I/O repair file inventory differs")
    for key in ("source_commit", "source_tree_sha256", "source_file_hashes"):
        _required(repair.get(key), actual.get(key), "I/O source identity differs: " + key)
    if not re.fullmatch(r"[0-9a-f]{40}", actual.get("source_commit") or ""):
        raise PermissionError("I/O repair source commit missing")
    return {
        "repair_id": ENGINE_IO_REPAIR_ID,
        "repair_sha256": file_hash(repair_path),
        "source_commit": actual["source_commit"],
        **{
            key: expected[key]
            for key in (
                "original_execution_freeze_sha256",
                "previous_engine_storage_repair_sha256",
                "failed_attempt_id",
                "failed_job_id",
                "gpu_worker_constraint",
                "minimum_gpu_host_memory_gb",
                "activation_spill_directory",
                "quota_root",
                "quota_reserve_bytes",
                "cpu_activation_budget_bytes",
            )
        },
    }


def verify_engine_multigpu_repair(root, actual_source=None):
    """Keep existing ENGINE attempt resource and worker identities immutable."""
    if (Path(root) / "ENGINE_COMPUTE_PARALLEL_REPAIR.json").exists():
        return verify_engine_compute_parallel_repair(root, actual_source=actual_source)[
            "engine_multigpu_repair"
        ]
    if (Path(root) / "BASELINE_PARALLEL_REPAIR.json").exists():
        latest = verify_baseline_parallel_repair(root, actual_source=actual_source)
        return latest["engine_multigpu_repair"]
    return _verify_original_engine_multigpu_repair(root, actual_source=actual_source)


def _verify_original_engine_multigpu_repair(root, actual_source=None):
    """Verify a fixed GPU activation-storage override and complete ENGINE restart.

    The original single-card registration and every prior receipt remain intact.
    This only authenticates resources and technical source; ENGINE still certifies
    its entire fresh four-step versus two-plus-two trace before science starts.
    """
    from .prepare import source_identity

    root = Path(root).resolve(strict=True)
    repair_path = root / "ENGINE_MULTIGPU_REPAIR.json"
    repair = read_json(repair_path)
    declared_source = root / "code_before_engine_multigpu_20261010"
    if declared_source.is_symlink():
        raise PermissionError("Pre-multiGPU source snapshot must be a real directory")
    saved_source = bounded_path(root, declared_source.name)
    before = read_json(saved_source / "SOURCE_DEPLOYMENT.json")
    _required(
        _preserved_source_files(saved_source, include_amendments=True),
        before.get("source_file_hashes"),
        "Pre-multiGPU source changed",
    )
    parent = _verify_original_engine_io_repair(root, actual_source=before)
    count = repair.get("gpu_count")
    if type(count) is not int or not 2 <= count <= 5:
        raise PermissionError("MultiGPU count must be an immutable integer from two through five")
    registration_path = root / "orchestration/REGISTRATION.json"
    registration = read_json(registration_path)
    worker_name = "scripts/sr_f1/run_worker.py"
    previous_worker = before["source_file_hashes"].get(worker_name)
    if not isinstance(previous_worker, str) or not re.fullmatch(r"[0-9a-f]{64}", previous_worker):
        raise PermissionError("MultiGPU source lacks the prior worker identity")
    _required(
        registration.get("freeze_sha256"),
        parent["original_execution_freeze_sha256"],
        "MultiGPU registration belongs to another freeze",
    )
    _required(
        registration.get("source_hashes", {}).get(worker_name),
        previous_worker,
        "MultiGPU registration does not preserve the original worker source",
    )
    _required(
        registration.get("permission", {}).get("qos"),
        "soujanya-poria-startfund-2026-03",
        "MultiGPU teacher QoS changed",
    )
    expected = {
        "repair_id": ENGINE_MULTIGPU_REPAIR_ID,
        "status": "AUTHORIZED_TECHNICAL_MULTIGPU_REPAIR",
        "plan_id": PLAN_ID,
        "run_root": str(root),
        "original_execution_freeze_sha256": parent["original_execution_freeze_sha256"],
        "previous_engine_io_repair_sha256": parent["repair_sha256"],
        "previous_source_commit": before["source_commit"],
        "previous_source_tree_sha256": before["source_tree_sha256"],
        "preserved_source_relative_path": saved_source.name,
        "registration_sha256": file_hash(registration_path),
        "previous_worker_source_sha256": previous_worker,
        "scientific_protocol_unchanged": True,
        "original_once_consumed_preserved": True,
        "engine_restart_mode": "FULL_REGISTERED_CONTINUOUS4_SPLIT2_PLUS2",
        "maintained_attempt_id": "ENGINE_attempt0003",
        "maintained_job_id": "196227",
        "gpu_count": count,
        "compute_device_index": 0,
        "storage_device_indices": list(range(1, count)),
        "gpu_activation_budget_bytes": 80 * 1024**3,
        "cpu_activation_budget_bytes": parent["cpu_activation_budget_bytes"],
        "activation_storage": "AUXILIARY_GPU_THEN_BOUNDED_CPU_EXACT_EXTERNAL_DISK",
        "gpu_worker_constraint": "highmem",
        "minimum_gpu_host_memory_gb_per_gpu": 80,
        "minimum_gpu_host_memory_gb": 80 * count,
        "cpus_per_gpu": 4,
        "qos": "soujanya-poria-startfund-2026-03",
        "resource_operations": ["engine", "train"],
        **{
            key: parent[key]
            for key in (
                "activation_spill_directory",
                "quota_root",
                "quota_reserve_bytes",
            )
        },
    }
    for key, value in expected.items():
        _required(repair.get(key), value, "ENGINE multiGPU repair differs: " + key)
    if not repair.get("authorized_user_message") or not repair.get("authorized_at"):
        raise PermissionError("MultiGPU repair lacks user authorization provenance")
    prefix = "technical_incidents/multigpu_20261010/"
    required = {
        prefix + name
        for name in (
            "PRESERVATION.json",
            "TERMINAL_JOBS.json",
            "CPU_STATE_REVIEW.json",
            "RECOVERY_REVIEW.json",
        )
    }
    artifacts = _checked_source_files(repair.get("historical_artifact_hashes"), "MultiGPU history")
    if not required.issubset(artifacts) or any(not name.startswith(prefix) for name in artifacts):
        raise PermissionError("MultiGPU history inventory is missing or unbounded")
    for name, expected_hash in artifacts.items():
        _required(file_hash(bounded_path(root, name)), expected_hash, "MultiGPU history changed")
    preservation = read_json(root / (prefix + "PRESERVATION.json"))
    files = preservation.get("artifact_hashes")
    if not isinstance(files, dict) or not files:
        raise PermissionError("MultiGPU preservation inventory empty")
    hashes = _checked_source_files(
        {
            name: entry.get("sha256") if isinstance(entry, dict) else None
            for name, entry in files.items()
        },
        "MultiGPU preserved",
    )
    _required(preservation.get("files"), len(hashes), "MultiGPU preservation file count differs")
    evidence = root / (prefix + "evidence")
    preserved_bytes = 0
    for name, expected_hash in hashes.items():
        declared = evidence / name
        if declared.is_symlink():
            raise PermissionError("Preserved multiGPU artifact is a symlink")
        saved = bounded_path(evidence, name)
        _required(file_hash(saved), expected_hash, "Preserved multiGPU artifact changed")
        size = files[name].get("bytes")
        if type(size) is not int or size < 0:
            raise PermissionError("Preserved multiGPU artifact size invalid")
        _required(saved.stat().st_size, size, "Preserved multiGPU artifact size changed")
        preserved_bytes += size
    _required(
        preservation.get("bytes"), preserved_bytes, "MultiGPU preservation byte count differs"
    )
    terminal = read_json(root / (prefix + "TERMINAL_JOBS.json"))
    for key, value in {
        "controller_terminal": True,
        "gpu_terminal": True,
        "controller_job_id": "196225",
        "gpu_job_id": expected["maintained_job_id"],
        "controller_terminal_state": "CANCELLED",
        "gpu_terminal_state": "CANCELLED",
    }.items():
        _required(terminal.get(key), value, "MultiGPU maintenance allocations not terminal: " + key)
    mandatory = {"orchestration/STATE.json", "orchestration/REGISTRATION.json"}
    track = "engineering/engine/natural/continuous"
    mandatory |= {
        track + "/" + name
        for name in (
            "RUN_MANIFEST.json",
            "checkpoints/LATEST.json",
            "checkpoints/commit-00.json",
        )
    }
    if not mandatory.issubset(hashes):
        raise PermissionError("MultiGPU preservation lacks complete state identities")
    _required(
        hashes["orchestration/REGISTRATION.json"],
        expected["registration_sha256"],
        "MultiGPU history registration changed",
    )
    state = read_json(evidence / "orchestration/STATE.json")
    if state.get("test_sealed") is not True or any(
        task.get("attempts")
        for name, task in state.get("tasks", {}).items()
        if name not in {"COMMON_START", "ENGINE"}
    ):
        raise PermissionError("MultiGPU repair does not preserve sealed pre-science history")
    attempts = state.get("tasks", {}).get("ENGINE", {}).get("attempts", [])
    if not any(
        attempt.get("attempt_id") == expected["maintained_attempt_id"]
        and str(attempt.get("job_id")) == expected["maintained_job_id"]
        for attempt in attempts
    ):
        raise PermissionError("MultiGPU history lacks the maintained ENGINE allocation")
    latest = read_json(evidence / track / "checkpoints/LATEST.json")
    _required(
        latest,
        read_json(evidence / track / "checkpoints/commit-00.json"),
        "MultiGPU preserved checkpoint commit differs",
    )
    _required(latest.get("step"), 0, "MultiGPU maintenance requires preserved committed step zero")
    checkpoint = bounded_path(evidence / track / "checkpoints", latest.get("path", ""))
    checkpoint_name = str(checkpoint.relative_to(evidence))
    _required(
        hashes.get(checkpoint_name),
        latest.get("sha256"),
        "MultiGPU checkpoint bytes are not authenticated",
    )
    if checkpoint_name not in hashes or not re.fullmatch(
        r"[0-9a-f]{64}", latest.get("state_hash", "")
    ):
        raise PermissionError("MultiGPU full checkpoint identity missing")
    raw_names = {f"01-{slot:02d}-{index}.json" for slot in range(16) for index in range(8)}
    raw_hashes = {
        Path(name).name: sha
        for name, sha in hashes.items()
        if name.startswith(track + "/rollouts/")
    }
    _required(set(raw_hashes), raw_names, "MultiGPU current raw sample slots are incomplete")
    if any(track + "/rollouts/" + name not in hashes for name in raw_names):
        raise PermissionError("MultiGPU current raw sample paths differ")
    review = read_json(root / (prefix + "CPU_STATE_REVIEW.json"))
    for key, value in {
        "status": "PASS_ZERO_UPDATE_FULL_STATE",
        "committed_logical_step": 0,
        "physical_optimizer_updates": 0,
        "optimizer_empty": True,
        "raw_rollouts": 128,
        "raw_record_hashes_verified": True,
        "learning_rate": 1e-4,
        "checkpoint_state_hash": latest["state_hash"],
        "checkpoint_file_sha256": latest["sha256"],
        "raw_file_hashes": raw_hashes,
        "source_commit": before["source_commit"],
    }.items():
        _required(review.get(key), value, "Invalid multiGPU full-state review: " + key)
    recovery = read_json(root / (prefix + "RECOVERY_REVIEW.json"))
    for key, value in {
        "status": "PASS_INTERRUPTED_ENGINE_RESTART",
        "science_attempts": 0,
        "test_sealed": True,
        "once_activation_consumed": True,
        "once_process_consumed": True,
        "journal_integrity": True,
        "original_artifact_hashes_unchanged": True,
        "full_engine_restart": True,
        "no_partial_gradient_reuse": True,
        "no_old_once_reuse": True,
    }.items():
        _required(recovery.get(key), value, "Invalid multiGPU recovery review: " + key)
    old_prefix = "technical_incidents/engine_memory_20261010/"
    for name in ("ZERO_UPDATE_REUSE_ACTIVATED.json", "ZERO_UPDATE_REUSE_PROCESS.json"):
        relative = old_prefix + name
        if relative not in hashes:
            raise PermissionError("MultiGPU history lacks consumed recovery markers")
        _required(
            file_hash(bounded_path(root, relative)),
            hashes[relative],
            "MultiGPU consumed marker changed",
        )
    actual = source_identity() if actual_source is None else actual_source
    previous_files = _checked_source_files(before.get("source_file_hashes"), "Pre-multiGPU")
    current_files = _checked_source_files(actual.get("source_file_hashes"), "MultiGPU actual")
    _required(actual.get("source_dirty_files"), [], "MultiGPU repair requires committed source")
    _required(actual.get("source_tree_sha256"), object_hash(current_files), "MultiGPU tree invalid")
    changed = sorted(
        name
        for name in previous_files.keys() | current_files.keys()
        if previous_files.get(name) != current_files.get(name)
    )
    if not ENGINE_MULTIGPU_REQUIRED_FILES.issubset(changed) or not set(changed).issubset(
        ENGINE_MULTIGPU_ALLOWED_FILES
    ):
        raise PermissionError("MultiGPU repair changed scientific source or omitted required files")
    if set(previous_files) - set(current_files):
        raise PermissionError("MultiGPU repair cannot remove frozen source files")
    _required(repair.get("changed_files"), changed, "MultiGPU repair file inventory differs")
    _required(
        repair.get("worker_source_sha256"),
        current_files[worker_name],
        "MultiGPU current worker revision differs",
    )
    for key in ("source_commit", "source_tree_sha256", "source_file_hashes"):
        _required(repair.get(key), actual.get(key), "MultiGPU source identity differs: " + key)
    if not re.fullmatch(r"[0-9a-f]{40}", actual.get("source_commit") or ""):
        raise PermissionError("MultiGPU repair source commit missing")
    return {
        **expected,
        "repair_sha256": file_hash(repair_path),
        "source_commit": actual["source_commit"],
        "worker_source_sha256": current_files[worker_name],
    }


def _baseline_parallel_history_names():
    prefix = "technical_incidents/baseline_parallel_20261010/activation/"
    return [
        prefix + "TERMINAL_JOBS.json",
        *[
            prefix + "evidence/orchestration/" + name
            for name in (
                "STATE.json",
                "journal.jsonl",
                "REGISTRATION.json",
                "completions/ENGINE.json",
            )
        ],
    ]


def _baseline_parallel_source_context(root):
    declared = root / "code_before_baseline_parallel_20261010"
    if declared.is_symlink():
        raise PermissionError("Pre-baseline source snapshot must be a real directory")
    saved = bounded_path(root, declared.name)
    before = read_json(saved / "SOURCE_DEPLOYMENT.json")
    _required(
        _preserved_source_files(saved, include_amendments=True),
        before.get("source_file_hashes"),
        "Pre-baseline source changed",
    )
    parent = _verify_original_engine_multigpu_repair(root, actual_source=before)
    expected = {
        "repair_id": BASELINE_PARALLEL_REPAIR_ID,
        "status": "AUTHORIZED_TECHNICAL_BASELINE_PARALLEL_REPAIR",
        "plan_id": PLAN_ID,
        "run_root": str(root),
        "original_execution_freeze_sha256": parent["original_execution_freeze_sha256"],
        "previous_engine_multigpu_repair_sha256": parent["repair_sha256"],
        "previous_source_commit": before["source_commit"],
        "previous_source_tree_sha256": before["source_tree_sha256"],
        "preserved_source_relative_path": saved.name,
        "registration_sha256": file_hash(root / "orchestration/REGISTRATION.json"),
        "previous_worker_source_sha256": before["source_file_hashes"][
            "scripts/sr_f1/run_worker.py"
        ],
        "scientific_protocol_unchanged": True,
        "qos": "soujanya-poria-startfund-2026-03",
        "max_gpu_count": 5,
        "resource_operations": ["baseline"],
        "partition_mode": "STABLE_SLOT_INDEX_MODULO",
        "baseline_slot_count": 6784,
    }
    return before, parent, expected


def build_baseline_parallel_repair(root, *, actual_source, authorized_at, authorized_user_message):
    """Construct the append-only receipt; callers must verify after exclusive write.

    This does not grant prospective permission to change an active ENGINE. The
    verifier requires the preserved, successful ENGINE handoff before execution.
    """
    root = Path(root).resolve(strict=True)
    before, _, expected = _baseline_parallel_source_context(root)
    previous = _checked_source_files(before.get("source_file_hashes"), "Pre-baseline")
    current = _checked_source_files(actual_source.get("source_file_hashes"), "Baseline actual")
    return {
        **expected,
        **{
            key: actual_source.get(key)
            for key in (
                "source_commit",
                "source_tree_sha256",
                "source_file_hashes",
                "source_dirty_files",
            )
        },
        "worker_source_sha256": current.get("scripts/sr_f1/run_worker.py"),
        "authorized_at": authorized_at,
        "authorized_user_message": authorized_user_message,
        "changed_files": sorted(
            name
            for name in previous.keys() | current.keys()
            if previous.get(name) != current.get(name)
        ),
        "historical_artifact_hashes": {
            name: file_hash(bounded_path(root, name)) for name in _baseline_parallel_history_names()
        },
    }


def _verify_baseline_parallel_handoff(root, repair):
    from mm_dev.orchestration import digest

    prefix = "technical_incidents/baseline_parallel_20261010/activation/"
    expected_names = set(_baseline_parallel_history_names())
    artifacts = _checked_source_files(repair.get("historical_artifact_hashes"), "Baseline handoff")
    if not expected_names.issubset(artifacts) or any(
        not name.startswith(prefix) for name in artifacts
    ):
        raise PermissionError("Baseline handoff history is missing or unbounded")
    for relative, expected_hash in artifacts.items():
        path = root / relative
        if path.is_symlink():
            raise PermissionError("Baseline handoff evidence is a symlink")
        _required(
            file_hash(bounded_path(root, relative)),
            expected_hash,
            "Baseline handoff evidence changed",
        )
    evidence = root / prefix / "evidence"
    registration = read_json(root / "orchestration/REGISTRATION.json")
    for relative in ("orchestration/REGISTRATION.json", "orchestration/completions/ENGINE.json"):
        _required(
            file_hash(evidence / relative),
            file_hash(root / relative),
            "Baseline handoff identity changed",
        )
    registration_hash = digest(registration)
    journal_path = evidence / "orchestration/journal.jsonl"
    journal = journal_path.read_bytes()
    with (root / "orchestration/journal.jsonl").open("rb") as stream:
        _required(
            stream.read(len(journal)),
            journal,
            "Baseline handoff is not the preserved journal prefix",
        )
    sequence, previous, state = 0, None, None
    for line in journal.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise PermissionError("Baseline handoff journal is truncated")
        row = json.loads(line)
        recorded = row.pop("sha256")
        if (
            recorded != digest(row)
            or row.get("previous") != previous
            or row.get("sequence") != sequence + 1
            or row.get("registration_hash") != registration_hash
        ):
            raise PermissionError("Baseline handoff journal identity or chain changed")
        sequence, previous, state = row["sequence"], recorded, row.get("state")
    snapshot = read_json(evidence / "orchestration/STATE.json")
    if state is None or state != snapshot:
        raise PermissionError("Baseline handoff state does not match its journal")
    tasks = snapshot.get("tasks", {})
    if (
        snapshot.get("test_sealed") is not True
        or tasks.get("COMMON_START", {}).get("status") != "COMPLETE"
        or tasks.get("ENGINE", {}).get("status") != "COMPLETE"
        or any(
            task.get("attempts")
            for name, task in tasks.items()
            if name not in {"COMMON_START", "ENGINE"}
        )
        or set(tasks) != set(registration.get("tasks", {}))
        or "BASELINE" not in tasks
    ):
        raise PermissionError(
            "Baseline handoff requires successful ENGINE before any downstream attempt"
        )
    attempts = tasks["ENGINE"].get("attempts", [])
    if not attempts:
        raise PermissionError("Baseline handoff lacks a completed ENGINE allocation")
    attempt = attempts[-1]
    accounting = attempt.get("accounting", {})
    if (
        attempt.get("status") != "COMPLETED"
        or accounting.get("terminal_state") != "COMPLETED"
        or accounting.get("exit_code") != "0:0"
    ):
        raise PermissionError("Baseline handoff ENGINE allocation has not completed successfully")
    marker_path = root / "orchestration/completions/ENGINE.json"
    marker = read_json(marker_path)
    for key, value in {
        "plan_id": PLAN_ID,
        "task_id": "ENGINE",
        "status": "COMPLETE",
        "attempt_id": attempt.get("attempt_id"),
        "registration_hash": registration_hash,
    }.items():
        _required(marker.get(key), value, "Baseline ENGINE completion identity differs: " + key)
    _required(
        tasks["ENGINE"].get("completion_marker_sha256"),
        file_hash(marker_path),
        "Baseline ENGINE completion is not journal authenticated",
    )
    descriptors = marker.get("artifacts", {})
    if "ENGINE_PROBABILITY_GRADIENT_RESUME.json" not in descriptors:
        raise PermissionError("Baseline ENGINE completion omits the actual acceptance receipt")
    for relative, descriptor in descriptors.items():
        path = bounded_path(root, relative)
        if not isinstance(descriptor, dict):
            raise PermissionError("Baseline ENGINE artifact descriptor is invalid")
        _required(
            file_hash(path), descriptor.get("sha256"), "Baseline ENGINE completion artifact changed"
        )
        _required(
            path.stat().st_size,
            descriptor.get("bytes"),
            "Baseline ENGINE completion artifact size changed",
        )
    engine = verify_engine_receipt(root)
    terminal = read_json(root / prefix / "TERMINAL_JOBS.json")
    for key, value in {
        "engine_job_id": str(attempt.get("job_id")),
        "engine_terminal": True,
        "engine_terminal_state": "COMPLETED",
        "engine_exit_code": "0:0",
        "engine_queue_empty": True,
        "controller_terminal": True,
        "controller_queue_empty": True,
    }.items():
        _required(terminal.get(key), value, "Baseline handoff allocation is not terminal: " + key)
    if (
        terminal.get("controller_terminal_state")
        not in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
        or not re.fullmatch(r"[0-9]+", terminal.get("controller_job_id", ""))
        or not re.fullmatch(r"[0-9]+", str(attempt.get("job_id", "")))
        or terminal["controller_job_id"] == str(attempt["job_id"])
    ):
        raise PermissionError("Baseline handoff lacks the old controller terminal identity")
    return {
        "engine_receipt_sha256": file_hash(root / "ENGINE_PROBABILITY_GRADIENT_RESUME.json"),
        "engine_status": engine["status"],
        "engine_attempt_id": attempt["attempt_id"],
        "engine_job_id": str(attempt["job_id"]),
        "handoff_journal_sha256": file_hash(journal_path),
        "handoff_journal_sequence": sequence,
    }


def verify_baseline_parallel_repair(root, actual_source=None):
    """Authenticate baseline-only parallelism after the complete ENGINE boundary."""
    from .prepare import source_identity

    root = Path(root).resolve(strict=True)
    path = root / "BASELINE_PARALLEL_REPAIR.json"
    repair = read_json(path)
    before, parent, expected = _baseline_parallel_source_context(root)
    for key, value in expected.items():
        _required(repair.get(key), value, "Baseline parallel repair differs: " + key)
    if not repair.get("authorized_at") or not repair.get("authorized_user_message"):
        raise PermissionError("Baseline parallel repair lacks user authorization provenance")
    history = _verify_baseline_parallel_handoff(root, repair)
    actual = source_identity() if actual_source is None else actual_source
    previous = _checked_source_files(before.get("source_file_hashes"), "Pre-baseline")
    current = _checked_source_files(actual.get("source_file_hashes"), "Baseline actual")
    _required(actual.get("source_dirty_files"), [], "Baseline repair requires committed source")
    _required(repair.get("source_dirty_files"), [], "Baseline repair source is dirty")
    _required(
        actual.get("source_tree_sha256"), object_hash(current), "Baseline source tree invalid"
    )
    changed = sorted(
        name for name in previous.keys() | current.keys() if previous.get(name) != current.get(name)
    )
    if (
        not BASELINE_PARALLEL_REQUIRED_FILES.issubset(changed)
        or not set(changed).issubset(BASELINE_PARALLEL_ALLOWED_FILES)
        or set(previous) - set(current)
    ):
        raise PermissionError("Baseline repair changed scientific source or omitted required files")
    _required(repair.get("changed_files"), changed, "Baseline repair source inventory differs")
    worker = current["scripts/sr_f1/run_worker.py"]
    _required(repair.get("worker_source_sha256"), worker, "Baseline worker revision differs")
    for key in ("source_commit", "source_tree_sha256", "source_file_hashes"):
        _required(repair.get(key), actual.get(key), "Baseline source identity differs: " + key)
    if not re.fullmatch(r"[0-9a-f]{40}", actual.get("source_commit") or ""):
        raise PermissionError("Baseline repair source commit missing")
    return {
        **expected,
        **history,
        "repair_sha256": file_hash(path),
        "source_commit": actual["source_commit"],
        "worker_source_sha256": worker,
        "engine_multigpu_repair": parent,
    }


def verify_technical_repair(root, actual_source=None):
    """Authenticate the one scoped source amendment without rewriting the freeze.

    The original model/data/processor gates remain in verify_execution. This
    additional source gate stays valid after the one-time resume has progressed;
    the scheduler separately enforces the before-bridge/science resume boundary.
    """
    from .prepare import source_identity, verify_package

    root = Path(root).resolve(strict=True)
    verify_package()
    freeze_path = root / "EXECUTION_FREEZE.json"
    source_path = root / "SOURCE_AND_RENDER_MANIFEST.json"
    repair_path = root / "TECHNICAL_REPAIR.json"
    freeze, original, repair = map(read_json, (freeze_path, source_path, repair_path))
    _required(freeze.get("plan_id"), PLAN_ID, "Technical repair has the wrong frozen plan")
    _required(freeze.get("status"), "FROZEN", "Technical repair requires the original freeze")
    _required(freeze.get("run_root"), str(root), "Technical repair root differs from the freeze")
    for key, expected in {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "repair_id": TECHNICAL_REPAIR_ID,
        "status": "AUTHORIZED_TECHNICAL_REPAIR",
        "original_freeze_sha256": file_hash(freeze_path),
        "original_source_manifest_sha256": file_hash(source_path),
        "original_source_tree_sha256": freeze.get("source_tree_sha256"),
        "original_source_commit": freeze.get("source_commit"),
        "preserved_source_root": "code_before_format_guard_repair",
        "scientific_contract_changes": [],
        "source_dirty_files": [],
    }.items():
        _required(repair.get(key), expected, "Technical repair identity differs: " + key)
    if not isinstance(repair.get("reason"), str) or not repair["reason"].strip():
        raise PermissionError("Technical repair requires an explicit implementation-only reason")
    if not re.fullmatch(r"[0-9a-f]{40}", freeze.get("source_commit") or ""):
        raise PermissionError("Technical repair lacks the original committed source identity")
    for relative, expected in freeze.get("artifact_hashes", {}).items():
        _required(
            file_hash(bounded_path(root, relative)),
            expected,
            "Technical repair changed an original frozen artifact: " + relative,
        )
    _required(
        freeze.get("artifact_hashes", {}).get("SOURCE_AND_RENDER_MANIFEST.json"),
        file_hash(source_path),
        "Original source manifest is not bound to the execution freeze",
    )
    before = _checked_source_files(original.get("source_file_hashes"), "Original")
    for candidate in (freeze.get("source_file_hashes"), repair.get("original_source_file_hashes")):
        _required(candidate, before, "Technical repair substituted the original file inventory")
    for candidate in (original.get("source_tree_sha256"), freeze.get("source_tree_sha256")):
        _required(candidate, object_hash(before), "Original source aggregate is inconsistent")
    for relative, expected in original["package_input_hashes"].items():
        _required(
            file_hash(bounded_path(root, relative)),
            expected,
            "Technical repair changed a registered input: " + relative,
        )
        _required(
            file_hash(bounded_path(PACKAGE, relative)),
            expected,
            "Technical repair substituted the uploaded input contract: " + relative,
        )
    declared_snapshot = root / repair["preserved_source_root"]
    snapshot = bounded_path(root, repair["preserved_source_root"])
    if not snapshot.is_dir() or declared_snapshot.is_symlink():
        raise PermissionError("Preserve the actual original source directory before repair")
    _required(
        _preserved_source_files(snapshot),
        before,
        "Preserved source differs from the original freeze",
    )
    actual = source_identity() if actual_source is None else actual_source
    after = _checked_source_files(actual.get("source_file_hashes"), "Actual repaired")
    _required(
        actual.get("source_tree_sha256"), object_hash(after), "Repaired source hash is invalid"
    )
    _required(actual.get("source_dirty_files"), [], "Commit the technical repair before deployment")
    if not re.fullmatch(r"[0-9a-f]{40}", actual.get("source_commit") or ""):
        raise PermissionError("Technical repair requires its actual full clean commit identity")
    if actual["source_commit"] == freeze["source_commit"]:
        raise PermissionError("Changed technical source cannot retain the original commit identity")
    for key in ("source_commit", "source_tree_sha256", "source_file_hashes"):
        _required(repair.get(key), actual[key], "Actual repaired source differs: " + key)
    if set(before) - set(after):
        raise PermissionError("Technical repair cannot remove frozen source files")
    changed = [
        {"path": name, "before_sha256": before.get(name), "after_sha256": after[name]}
        for name in sorted(after)
        if before.get(name) != after[name]
    ]
    if not changed or any(row["path"] not in TECHNICAL_REPAIR_ALLOWED_FILES for row in changed):
        raise PermissionError("Technical repair exceeds the explicit source-file allowlist")
    _required(repair.get("changed_files"), changed, "Technical repair file changes are not exact")
    review_path = root / "FORMAT_TECHNICAL_REVIEW.json"
    _required(
        repair.get("format_review_sha256"),
        file_hash(review_path),
        "Technical repair references a different FORMAT technical review",
    )
    review = read_json(review_path)
    for key, expected in {
        "plan_id": PLAN_ID,
        "freeze_sha256": file_hash(freeze_path),
        "status": "VERIFIED_PROTOCOL_NONADHERENCE",
    }.items():
        _required(review.get(key), expected, "FORMAT technical review identity differs: " + key)
    # Imported only after the repair module's bytes, original freeze and review
    # identity have been checked; no circular module import or scientific state read.
    from .format_review import verify_format_review

    verify_format_review(root)
    return {
        "status": "VERIFIED_TECHNICAL_REPAIR",
        "repair_id": TECHNICAL_REPAIR_ID,
        "repair_sha256": file_hash(repair_path),
        "review_sha256": file_hash(review_path),
        "original_freeze_sha256": file_hash(freeze_path),
        "source_commit": actual["source_commit"],
        "source_tree_sha256": actual["source_tree_sha256"],
        "changed_files": changed,
    }


def _verify_execution_source(root, original_source, actual_source=None):
    from .prepare import source_identity

    actual = source_identity() if actual_source is None else actual_source
    files = _checked_source_files(actual.get("source_file_hashes"), "Actual")
    _required(actual.get("source_tree_sha256"), object_hash(files), "Actual source hash is invalid")
    if (Path(root) / "ENGINE_COMPUTE_PARALLEL_REPAIR.json").exists():
        return verify_engine_compute_parallel_repair(root, actual_source=actual)
    if (Path(root) / "BASELINE_PARALLEL_REPAIR.json").exists():
        return verify_baseline_parallel_repair(root, actual_source=actual)
    if original_source.get("source_tree_sha256") == actual.get("source_tree_sha256"):
        _required(original_source.get("source_file_hashes"), files, "Original source files differ")
        return None
    if (Path(root) / "QOS_SCOPE_REPAIR.json").exists():
        return verify_qos_scope_repair(root, actual_source=actual)
    return verify_technical_repair(root, actual_source=actual)


def verify_engine_receipt(root):
    root = Path(root).resolve(strict=True)
    engine = read_json(root / "ENGINE_PROBABILITY_GRADIENT_RESUME.json")
    _required(engine.get("plan_id"), PLAN_ID, "ENGINE plan differs")
    _required(
        engine.get("freeze_sha256"),
        file_hash(root / "EXECUTION_FREEZE.json"),
        "ENGINE belongs to a different execution freeze",
    )
    _required(
        engine.get("common_start_sha256"),
        file_hash(root / "COMMON_START.json"),
        "ENGINE belongs to a different common start",
    )
    common = read_json(root / "COMMON_START.json")
    for key, expected in dict(
        plan_id=PLAN_ID,
        status="VERIFIED",
        model_id="SRF1_COMMON_START",
        freeze_sha256=engine["freeze_sha256"],
    ).items():
        _required(common.get(key), expected, "ENGINE common start is not verified")
    if not common.get("adapter_file_hashes"):
        raise PermissionError("ENGINE common start lacks actual adapter files")
    adapter = bounded_path(root, common["adapter_path"])
    for relative, expected in common["adapter_file_hashes"].items():
        _required(
            file_hash(bounded_path(adapter, relative)), expected, "Common adapter bytes changed"
        )
    status = engine.get("status")
    if status not in ("PASS_NATURAL_GRPO_RESUME", "PASS_NONZERO_SURROGATE_KERNEL_RESUME"):
        raise PermissionError("GPU probability/gradient/resume certification missing")
    natural, stress = engine.get("natural"), engine.get("stress")
    if not isinstance(natural, dict):
        raise PermissionError("Actual natural ENGINE trace is missing")
    expected_natural = (
        "EXACT_NONZERO_TRACE"
        if status == "PASS_NATURAL_GRPO_RESUME"
        else "EXACT_CONSTANT_REWARD_TRACE"
    )
    _required(natural.get("status"), expected_natural, "Natural ENGINE result differs")
    if status == "PASS_NATURAL_GRPO_RESUME" and stress is not None:
        raise PermissionError("Unexpected surrogate trace for nonconstant natural ENGINE")
    if status == "PASS_NONZERO_SURROGATE_KERNEL_RESUME":
        if not natural.get("stress_allowed") or not isinstance(stress, dict):
            raise PermissionError("Missing permitted nonzero surrogate ENGINE qualification")
        _required(stress.get("status"), "EXACT_NONZERO_TRACE", "Surrogate ENGINE gradient failed")
    artifacts = engine.get("artifact_hashes", {})
    if not artifacts:
        raise PermissionError("ENGINE is missing its actual trace artifact identities")
    for mode, trace in (("natural", natural), ("stress", stress)):
        if trace is None:
            continue
        for key, expected in dict(
            plan_id=PLAN_ID,
            mode=mode,
            logical_steps=4,
            valid_trace_updates=8,
            valid_trace_completions=1024,
            exact_state_and_rng=True,
            exact_token_and_slot_paths=True,
            all_probability_gates_passed=True,
        ).items():
            _required(trace.get(key), expected, "Incomplete ENGINE evidence: " + key)
        if trace.get("physical_updates", 0) < 8 or trace.get("completion_attempts", 0) < 1024:
            raise PermissionError("ENGINE physical attempts omit the valid trace cost")
        if not trace.get("artifact_hashes"):
            raise PermissionError("ENGINE trace has no underlying evidence")
        comparison = f"engineering/engine/{mode}/COMPARISON.json"
        if (
            artifacts.get(comparison) != file_hash(root / comparison)
            or read_json(root / comparison) != trace
        ):
            raise PermissionError("ENGINE comparison receipt does not match its trace")
        for relative, expected in trace["artifact_hashes"].items():
            _required(artifacts.get(relative), expected, "ENGINE omitted underlying trace evidence")
    for relative, expected in artifacts.items():
        _required(
            file_hash(bounded_path(root, relative)), expected, "ENGINE trace artifact changed"
        )
    return engine


def verify_execution(plan, root, require_engine=False):
    """Verify actual artifacts before GPU work, not just a self-reported PASS."""
    from .prepare import BOLD_HASH, FONT_HASH, verify_package

    root = Path(root).resolve(strict=True)
    plan = validate_plan(plan)
    amendment = protocol_amendment(plan)
    verify_package()
    freeze = read_json(root / "EXECUTION_FREEZE.json")
    _required(freeze.get("plan_id"), PLAN_ID, "Wrong execution freeze plan")
    _required(freeze.get("status"), "FROZEN", "CPU preparation has not been frozen")
    _required(freeze.get("run_root"), str(root), "Execution root identity differs")
    amendment_receipt = verify_amendment(root, plan)
    _required(
        freeze.get("config_sha256"),
        file_hash(root / EFFECTIVE_CONFIG)
        if amendment
        else file_hash(PACKAGE / "config/SR_F1.json"),
        "Frozen effective configuration identity differs",
    )
    if amendment is not None:
        for key, expected in {
            "amendment_sha256": file_hash(root / "AMENDMENT.json"),
            "protocol_amendment": amendment,
            "original_freeze_sha256": ORIGINAL_FREEZE_SHA256,
            "inherited_or_prior_run_results_used_to_modify_protocol": True,
            "uploaded_config_sha256": file_hash(PACKAGE / "config/SR_F1.json"),
        }.items():
            _required(freeze.get(key), expected, "Frozen amendment identity differs: " + key)
    _required(
        freeze.get("allocation_permission_sha256"),
        file_hash(root / "manifests/ALLOCATION_PERMISSION.json"),
        "Allocation permission differs from the execution freeze",
    )
    _required(
        freeze.get("formal_test_results_seen_before_freeze"),
        False,
        "Prereveal declaration is missing or invalid",
    )
    _required(
        freeze.get("inherited_or_prior_run_results_used_to_select_parameters"),
        amendment is not None,
        "Prior-result declaration is missing or invalid",
    )
    _required(
        freeze.get("changes_from_uploaded_contract"),
        approved_changes(plan),
        "Unregistered scientific changes",
    )
    for relative, expected in freeze.get("artifact_hashes", {}).items():
        if file_hash(bounded_path(root, relative)) != expected:
            raise PermissionError("Frozen artifact changed: " + relative)
    required = {
        "MODEL_ENVIRONMENT_IDENTITY.json",
        "SOURCE_AND_RENDER_MANIFEST.json",
        "RENDER_RECEIPT.json",
        "PROCESSOR_PREFLIGHT.json",
        "PROCESSOR_INPUT_ROUTES.jsonl",
        "PROCESSED_IMAGES.jsonl",
    }
    if amendment_receipt is not None:
        required.update({"AMENDMENT.json", EFFECTIVE_CONFIG, *REGISTERED_DOCUMENTS})
    if not required.issubset(freeze.get("artifact_hashes", {})):
        raise PermissionError("Freeze lacks the complete CPU evidence chain")
    source = read_json(root / "SOURCE_AND_RENDER_MANIFEST.json")
    _required(source.get("status"), "CPU_VERIFIED", "Source/render preparation incomplete")
    _verify_execution_source(root, source)
    for relative, expected in source["package_input_hashes"].items():
        if file_hash(bounded_path(root, relative)) != expected:
            raise PermissionError("Registered copied input changed: " + relative)
        if file_hash(bounded_path(PACKAGE, relative)) != expected:
            raise PermissionError("Copied input differs from authenticated package")
    render = read_json(root / "RENDER_RECEIPT.json")
    from PIL import ImageFont
    from PIL import __version__ as pillow_version

    _required(
        render.get("renderer_environment"),
        {
            "Pillow": pillow_version,
            "FreeType": ImageFont.core.freetype2_version,
            "renderer_sha256": file_hash(PACKAGE / "reference/render_charts.py"),
        },
        "Actual renderer/Pillow/FreeType environment changed",
    )
    _required(render.get("chart_count"), 2890, "Incomplete chart materialization")
    _required(render.get("image_count"), 2891, "Incomplete blank/chart materialization")
    for key, expected in (("normal", FONT_HASH), ("bold", BOLD_HASH)):
        entry = render["fonts"][key]
        _required(entry.get("sha256"), expected, "Wrong renderer font")
        _required(file_hash(entry["path"]), expected, "Renderer font bytes changed")
    images = {row["file"]: row for row in render["images"]}
    if len(images) != 2891:
        raise PermissionError("Missing or duplicated rendered image identity")
    for relative, row in images.items():
        _required(file_hash(bounded_path(root, relative)), row["sha256"], "Rendered image changed")
    processor = read_json(root / "PROCESSOR_PREFLIGHT.json")
    _required(processor.get("status"), "PASS", "Actual CPU processor preflight did not pass")
    _required(
        processor.get("processor_status"),
        "ACTUAL_CPU_PROCESSOR_VERIFIED",
        "Processor evidence is not native execution",
    )
    _required(processor.get("question_count"), 4800, "Incomplete processor question coverage")
    _required(processor.get("chart_image_count"), 2890, "Incomplete processor image coverage")
    _required(processor.get("blank_image_verified"), True, "Blank visual channel was not checked")
    _required(
        processor.get("no_crop_pixel_reconstruction_verified"),
        True,
        "Native pixel reconstruction was not verified",
    )
    checks = processor.get("isolation", {}).get("checks", {})
    if len(checks) != 7 or not all(value is True for value in checks.values()):
        raise PermissionError("Model-input isolation controls are incomplete")
    if processor.get("readability", {}).get("minimum_label_height_pixels", 0) < 9:
        raise PermissionError("Native label readability failed")
    routes = read_jsonl(root / "PROCESSOR_INPUT_ROUTES.jsonl")
    processed = read_jsonl(root / "PROCESSED_IMAGES.jsonl")
    _required(len(routes), 4800, "Input route records are incomplete")
    _required(len({r["qid"] for r in routes}), 4800, "Duplicate processor questions")
    _required(len(processed), 2890, "Processed image records are incomplete")
    for row in routes:
        verify_protocol_route(row, plan)
        if (
            row.get("processed_size") != [1024, 768]
            or row.get("pixels_per_count", 0) < 3
            or row.get("image_token_count", 0) <= 0
            or not row.get("input_tensor_hash")
            or row.get("source_image_sha256") != images[row["image_file"]]["sha256"]
        ):
            raise PermissionError("Processor route violates image identity or readability")
    for row in processed:
        if (
            row.get("processed_size") != [1024, 768]
            or row.get("pixels_per_count", 0) < 3
            or row.get("minimum_label_height_pixels", 0) < 9
            or row.get("max_inverse_rgb_error", 999) > 1
        ):
            raise PermissionError("Processed image readability evidence failed")
    for row in processor["qa_samples"] + processor["qa_contact_sheets"]:
        _required(file_hash(bounded_path(root, row["path"])), row["sha256"], "QA artifact changed")
    expected_qa = {
        (r["pool"], r["family"], r["chart"], r["style"], f"{r['series_count']}series")
        for r in render["images"]
        if "pool" in r
    }
    if not expected_qa or {tuple(r["key"]) for r in processor["qa_samples"]} != expected_qa:
        raise PermissionError(
            "Fixed pool/family/chart/style/series visual QA coverage is incomplete"
        )
    if not processor["qa_contact_sheets"]:
        raise PermissionError("Actual native processor QA contact sheets are missing")
    identity = read_json(root / "MODEL_ENVIRONMENT_IDENTITY.json")
    for owner in (identity, processor.get("processor_identity", {})):
        _required(
            owner.get("protocol_amendment"),
            amendment,
            "Processor/model amendment identity conflict",
        )
    verify_protocol_route(processor.get("blank_image_routing", {}), plan)
    _required(freeze.get("model_identity"), identity, "Frozen model identity evidence changed")
    _required(identity.get("status"), "ACTUAL_CPU_VERIFIED", "Actual model inspection missing")
    _required(identity.get("model_id"), plan["model"]["id"], "Model family/size differs")
    _required(identity.get("model_revision"), plan["model"]["revision"], "Model revision differs")
    _required(
        identity.get("model_weights_hash"),
        plan["model"]["prior_composite_weight_hash"],
        "Frozen model aggregate differs",
    )
    _required(identity.get("python"), platform.python_version(), "Python environment changed")
    for name, version in identity.get("packages", {}).items():
        _required(importlib.metadata.version(name), version, "Runtime package changed: " + name)
    from mm_core.vl_runtime import model_file_path

    weights = {}
    for row in identity.get("model_files", []):
        path = model_file_path(identity["model_path"], row["name"])
        if path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise PermissionError("Actual model/processor bytes changed: " + row["name"])
        if row["name"].endswith((".safetensors", "index.json")) or row["name"] == "config.json":
            weights[row["name"]] = row["sha256"]
    _required(
        object_hash(weights), identity["model_weights_hash"], "Actual model aggregate differs"
    )
    for key in (
        "processor_hash",
        "tokenizer_hash",
        "chat_template_hash",
        "chat_template_kwargs_hash",
        "linear_kernel_identity_hash",
        "generation_config_expanded",
    ):
        if not identity.get(key) or identity[key] != processor["processor_identity"].get(key):
            raise PermissionError("Processor/model environment identity conflict: " + key)
    _required(
        identity.get("chat_template_kwargs"), {"enable_thinking": False}, "Thinking mode changed"
    )
    _required(identity.get("canvas_pixels"), [1024, 768], "Synthetic canvas identity changed")
    _required(identity.get("processor_target_pixels"), 786432, "Native pixel budget changed")
    if require_engine:
        verify_engine_receipt(root)
    if (root / "ENGINE_COMPUTE_PARALLEL_REPAIR.json").exists():
        compute = verify_engine_compute_parallel_repair(root)
        return {
            **freeze,
            "engine_compute_parallel_repair": compute,
            "engine_multigpu_repair": compute["engine_multigpu_repair"],
        }
    if (root / "BASELINE_PARALLEL_REPAIR.json").exists():
        baseline = verify_baseline_parallel_repair(root)
        return {
            **freeze,
            "baseline_parallel_repair": baseline,
            "engine_multigpu_repair": baseline["engine_multigpu_repair"],
        }
    if (root / "ENGINE_MULTIGPU_REPAIR.json").exists():
        return {**freeze, "engine_multigpu_repair": verify_engine_multigpu_repair(root)}
    return freeze


def freeze_execution(root, *, operator, plan=None):
    from .prepare import source_identity

    root = Path(root).resolve(strict=True)
    plan = effective_plan(root) if plan is None else validate_plan(plan)
    amendment = protocol_amendment(plan)
    verify_amendment(root, plan)
    if not isinstance(operator, str) or not operator.strip():
        raise ValueError("Actual execution operator is required")
    destination = root / "EXECUTION_FREEZE.json"
    if destination.exists() and read_json(destination).get("status") == "FROZEN":
        raise FileExistsError("Preserve existing execution freeze")
    source = source_identity()
    if not re.fullmatch(r"[0-9a-f]{40}", source.get("source_commit") or ""):
        raise PermissionError("A full actual source commit/deployment receipt is required")
    if source.get("source_dirty_files"):
        raise PermissionError("Commit the scoped source before the execution freeze")
    receipt = read_json(root / "SOURCE_AND_RENDER_MANIFEST.json")
    if source["source_tree_sha256"] != receipt["source_tree_sha256"]:
        raise PermissionError("Source changed during CPU preparation")
    names = [
        "MODEL_ENVIRONMENT_IDENTITY.json",
        "SOURCE_AND_RENDER_MANIFEST.json",
        "RENDER_RECEIPT.json",
        "PROCESSOR_PREFLIGHT.json",
        "PROCESSOR_INPUT_ROUTES.jsonl",
        "PROCESSED_IMAGES.jsonl",
    ]
    if amendment is not None:
        names.extend(["AMENDMENT.json", EFFECTIVE_CONFIG, *REGISTERED_DOCUMENTS])
    identity = read_json(root / names[0])
    external = root / "EXTERNAL_DATASET_AVAILABILITY.json"
    if not external.exists():
        external = root / "external/chartqa/IDENTITY.json"
    allocation = root / "manifests/ALLOCATION_PERMISSION.json"
    if not allocation.exists():
        raise PermissionError("Current allocation permission receipt is required before freezing")
    permission = read_json(allocation)
    for key, value in dict(
        plan_id=PLAN_ID,
        authorized=True,
        run_root=str(root),
        max_gpus_concurrent=5,
        gpus_per_worker=1,
    ).items():
        _required(permission.get(key), value, "Allocation permission is missing or differs")
    inventory = root / "manifests/ACTIVE_PROJECT_JOBS.json"
    payload = dict(
        plan_id=PLAN_ID,
        experiment=amendment["id"] if amendment else PLAN_ID,
        status="FROZEN",
        run_root=str(root),
        server_utc_time=utc_now(),
        operator=operator,
        plan_sha256=file_hash(PACKAGE / "CODEX_NEXT_PLAN_SR_F1_zh.md"),
        config_sha256=file_hash(root / EFFECTIVE_CONFIG)
        if amendment
        else file_hash(PACKAGE / "config/SR_F1.json"),
        **source,
        artifact_hashes={name: file_hash(root / name) for name in names},
        allocation_permission_sha256=file_hash(allocation),
        model_revision_verified=identity["model_revision"],
        model_identity=identity,
        model_file_hashes=identity["model_files"],
        model_path=identity["model_path"],
        processor_and_template_hashes={
            k: identity[k] for k in ("processor_hash", "tokenizer_hash", "chat_template_hash")
        },
        renderer_font_hash=read_json(root / "RENDER_RECEIPT.json")["fonts"],
        rendered_dataset_manifest="RENDER_RECEIPT.json",
        runtime_environment=identity["packages"],
        adapter_initialisation_sha256=None,
        common_start_identity=None,
        bridge_used=None,
        common_start_status="PENDING_GPU_FORMAT_AND_ENGINE",
        formal_test_results_seen_before_freeze=False,
        inherited_or_prior_run_results_used_to_select_parameters=amendment is not None,
        active_F2_jobs_read_only_inventory=read_json(inventory) if inventory.exists() else [],
        total_authorised_project_workers=5,
        research_gpu_hours_limit=None,
        external_dataset_availability_and_revision=read_json(external)
        if external.exists()
        else {"status": "PENDING", "main_experiment_may_continue": True},
        changes_from_uploaded_contract=approved_changes(plan),
    )
    if amendment is not None:
        payload.update(
            amendment_sha256=file_hash(root / "AMENDMENT.json"),
            protocol_amendment=amendment,
            original_freeze_sha256=ORIGINAL_FREEZE_SHA256,
            uploaded_config_sha256=file_hash(PACKAGE / "config/SR_F1.json"),
            inherited_or_prior_run_results_used_to_modify_protocol=True,
        )
    # Validate against a candidate, deleting only this unaccepted candidate on error.
    atomic_json(destination, payload)
    try:
        return verify_execution(plan, root)
    except Exception:
        payload["status"] = "CPU_FREEZE_REJECTED"
        atomic_json(destination, payload)
        raise
