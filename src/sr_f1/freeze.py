"""Fail-closed S0/S1 identity gates, separate from GPU engine certification."""

from __future__ import annotations

import importlib.metadata
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
