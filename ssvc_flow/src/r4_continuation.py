"""Reviewed continuation contracts and immutable snapshots of stopped R4 evidence.

The caller must first run ``audit_stopped_r4``. This module binds that audit to
the explicit user decision and current source before any model is loaded. A
snapshot is copied byte for byte; old identities, alarms and checkpoints are
never rewritten. Current execution identities belong to the caller, while the
returned logical identity keeps the original sample keys and random seeds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath

from .core import canonical_hash

REVIEWED_POLICY = {
    "reviewed_warning_policy": "sequence_p99_only_two_arms_to_step64",
    "arms": ["X_BASE", "X_VALID"],
    "steps": 64,
    "thresholds": {
        "mean_token_kl": 0.1,
        "sequence_log_ratio_p99_abs": 2.0,
        "comparison": "strict_greater_than",
    },
    "sequence_p99": "preserve_warning_and_continue",
    "hard_stops": [
        "mean_token_kl",
        "measurement_fault",
        "policy_contamination",
        "hash_mismatch",
        "parity_failure",
    ],
}


def _json(value):
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if path.is_symlink():
            raise ValueError("Continuation decision must not be a symbolic link")
        value = json.loads(path.read_text())
    # Enforce JSON serializability and reject NaN even in otherwise unused fields.
    canonical_hash(value)
    return copy.deepcopy(value)


def _same(left, right):
    # JSON hashes preserve numeric types, unlike Python equality (True == 1).
    return canonical_hash(left) == canonical_hash(right)


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid continuation artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in (".", "..") for p in value.split("/")):
        raise ValueError("Continuation artifact path escapes its root")
    if path.as_posix() != value:
        raise ValueError("Continuation artifact path is not canonical")
    return path


def _validate_binding(binding):
    if not isinstance(binding, dict) or binding.get("status") != "PASS_STOPPED_AUDIT":
        raise ValueError("A passed stopped-R4 audit is required")
    stable = {k: v for k, v in binding.items() if k not in ("root", "audit_hash")}
    if binding.get("audit_hash") != canonical_hash(stable):
        raise ValueError("Stopped-R4 audit hash mismatch")
    files = binding.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Stopped-R4 audited files are missing")
    for name, digest in files.items():
        _relative(name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Invalid stopped-R4 artifact hash")
    if not isinstance(binding.get("root"), str) or not Path(binding["root"]).is_absolute():
        raise ValueError("Stopped-R4 recorded root is missing")
    identity, source = binding.get("identity", {}), binding.get("source", {})
    if identity.get("phase") != "R4" or identity.get("source_hash") != canonical_hash(source):
        raise ValueError("Stopped-R4 source identity mismatch")
    stop = binding.get("stop", {})
    if stop.get("arm") not in REVIEWED_POLICY["arms"] or type(stop.get("step")) is not int:
        raise ValueError("Stopped-R4 arm or step is invalid")
    if not 0 < stop["step"] < 64:
        raise ValueError("Continuation requires a stopped update before step 64")
    diagnostic = stop.get("diagnostic", {})
    if not _same(diagnostic.get("thresholds"), REVIEWED_POLICY["thresholds"]):
        raise ValueError("Stopped-R4 diagnostic thresholds changed")
    if diagnostic.get("should_stop") is not True or not _same(
        diagnostic.get("alarms"),
        {"mean_token_kl": False, "sequence_log_ratio_p99_abs": True},
    ):
        raise ValueError("Only an audited sequence-p99 warning can continue")
    mean, p99 = diagnostic.get("mean_token_kl"), diagnostic.get("sequence_log_ratio_p99_abs")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (mean, p99)):
        raise ValueError("Nonfinite stopped-R4 diagnostic cannot continue")
    if not 0 <= mean <= 0.1 or p99 <= 2.0:
        raise ValueError("Stopped-R4 warning values contradict the reviewed policy")


def validate_continuation_decision(decision, parent_binding, execution_source):
    """Validate a decision dict or JSON path against the exact audit and source.

    The returned JSON dict is detached from caller-owned inputs. No timestamps,
    machine paths or local floating-point recomputation enter its identity.
    Explicit ``allow_training`` is additionally required by preparation/runtime.
    """
    decision, binding, source = map(_json, (decision, parent_binding, execution_source))
    _validate_binding(binding)
    if not isinstance(decision, dict) or set(decision) != {
        "schema_version",
        "authorization",
        "parent_audit_hash",
        "execution_source_hash",
        "policy",
    }:
        raise ValueError("Continuation decision schema changed")
    if type(decision["schema_version"]) is not int or decision["schema_version"] != 1:
        raise ValueError("Unsupported continuation decision schema")
    authorization = decision["authorization"]
    if not isinstance(authorization, dict) or set(authorization) != {"user_request", "reason"}:
        raise ValueError("Continuation user authorization and diagnostic reason are required")
    if any(not isinstance(v, str) or not v.strip() for v in authorization.values()):
        raise ValueError("Continuation user authorization and diagnostic reason must be nonempty")
    if decision["parent_audit_hash"] != binding["audit_hash"]:
        raise ValueError("Continuation parent audit binding changed")
    if decision["execution_source_hash"] != canonical_hash(source):
        raise ValueError("Continuation execution source changed")
    if not _same(decision["policy"], REVIEWED_POLICY):
        raise ValueError("Continuation policy, two arms, steps or thresholds changed")
    return decision


def _open_regular(root, relative):
    """Open only beneath root, refusing symlinks at every path component."""
    parts = _relative(relative).parts
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError("Continuation evidence contains a nonregular file")
        return os.fdopen(descriptor, "rb")
    except OSError as exc:
        raise ValueError("Continuation evidence path contains a symlink or is unavailable") from exc
    finally:
        os.close(directory)


def _inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Continuation evidence root must be a real directory")
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            entry = Path(directory) / name
            mode = entry.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError("Continuation evidence contains a symlink or nonregular entry")
        for name in sorted(names):
            relative = (Path(directory) / name).relative_to(root).as_posix()
            digest, size = hashlib.sha256(), 0
            with _open_regular(root, relative) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            files[relative] = {"sha256": digest.hexdigest(), "bytes": size}
    return dict(sorted(files.items()))


def _check_audited_files(parent, binding, inventory):
    for name, digest in binding["files"].items():
        if inventory.get(name, {}).get("sha256") != digest:
            raise ValueError(f"Stopped-R4 audited artifact changed: {name}")
    for name, key in (
        ("manifest.json", "manifest_sha256"),
        ("runtime_lock.json", "runtime_lock_sha256"),
        ("alarm_stop.json", "alarm_stop_sha256"),
    ):
        if inventory.get(name, {}).get("sha256") != binding.get(key):
            raise ValueError(f"Stopped-R4 binding changed: {name}")
    for relative in inventory:
        if PurePosixPath(relative).name in ("measurement_fault.json", "policy_contamination.json"):
            raise ValueError("A preserved measurement or contamination fault cannot continue")
    runtime = _json(parent / "runtime_lock.json")
    if not _same(runtime.get("identity"), binding["identity"]) or not _same(
        runtime.get("source"), binding["source"]
    ):
        raise ValueError("Stopped-R4 runtime identity changed")
    if not _same(_json(parent / "identity.json"), binding["identity"]):
        raise ValueError("Stopped-R4 logical sampling identity changed")
    if not _same(_json(parent / "checkpoint_manifest.json"), binding["checkpoint_manifest"]):
        raise ValueError("Stopped-R4 checkpoint manifest changed")
    checkpoint = binding["stop"]["checkpoint"]
    checkpoint_path = Path(checkpoint["checkpoint_path"])
    if checkpoint_path.is_absolute():
        try:
            relative = checkpoint_path.relative_to(binding["root"]).as_posix()
        except ValueError as exc:
            raise ValueError("Stopped-R4 checkpoint escapes its recorded root") from exc
    else:
        relative = checkpoint_path.as_posix()
    _relative(relative)
    if inventory.get(relative, {}).get("sha256") != checkpoint["checkpoint_sha256"]:
        raise ValueError("Stopped-R4 diagnostic checkpoint changed")
    return runtime


def _exclusive_json(path, value):
    if path.is_symlink():
        raise ValueError("Continuation metadata must not be a symbolic link")
    if path.exists():
        if not _same(_json(path), value):
            raise ValueError(f"Existing continuation metadata changed: {path.name}")
        return
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    # Exclusive creation refuses to overwrite evidence, including after interruption.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _records(decision, binding, inventory):
    logical = copy.deepcopy(binding["identity"])
    stable = {
        "schema_version": 1,
        "parent_audit_hash": binding["audit_hash"],
        "decision_sha256": canonical_hash(decision),
        "parent_snapshot_sha256": canonical_hash(inventory),
        "logical_sampling_identity_sha256": canonical_hash(logical),
        "reviewed_warning_policy": REVIEWED_POLICY["reviewed_warning_policy"],
    }
    contract = {
        **stable,
        "continuation_hash": canonical_hash(stable),
        "parent_dir": "inherited_parent",
        "parent_binding": binding,
    }
    return logical, contract, {"runtime_binding": contract, "snapshot_files": inventory}


def _context(snapshot, binding, runtime, logical, decision, contract):
    return {
        "parent_root": snapshot,
        "parent_binding": binding,
        "parent_runtime": runtime,
        "logical_identity": logical,
        "sampling_identity": logical,
        "decision": decision,
        "contract": contract,
        "runtime_binding": contract,
        "continuation_hash": contract["continuation_hash"],
    }


def verify_continuation(out, parent_binding, execution_source):
    """Read-only acceptance of a saved continuation, returning its full context.

    The caller first re-audits inherited_parent with the original recorded root.
    This verifier writes no files and never requires the external parent path.
    It is suitable for completed-stage gates and pre-GPU resume checks.
    """
    destination = Path(out)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("Continuation output must be an existing real directory")
    destination = destination.resolve()
    decision = validate_continuation_decision(
        destination / "continuation_decision.json", parent_binding, execution_source
    )
    binding = _json(parent_binding)
    snapshot = destination / "inherited_parent"
    inventory = _inventory(snapshot)
    runtime = _check_audited_files(snapshot, binding, inventory)
    logical, contract, record = _records(decision, binding, inventory)
    if not _same(_json(destination / "logical_sampling_identity.json"), logical):
        raise ValueError("Saved logical sampling identity changed")
    if not _same(_json(destination / "continuation_binding.json"), record):
        raise ValueError("Saved continuation binding or snapshot inventory changed")
    return _context(snapshot, binding, runtime, logical, decision, contract)


def _partial_inventory(copying, inventory):
    """Only completely copied, hash-matching files may be reused."""
    present = _inventory(copying)
    if any(name not in inventory or value != inventory[name] for name, value in present.items()):
        raise ValueError("Partial parent snapshot contains changed or unexpected complete files")
    directories = {
        str(parent)
        for name in inventory
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    for directory, dirs, _ in os.walk(copying, followlinks=False):
        for name in dirs:
            relative = (Path(directory) / name).relative_to(copying).as_posix()
            if relative not in directories:
                raise ValueError("Partial parent snapshot contains an unexpected directory")
    return present


def _copy_snapshot(parent, destination, snapshot, inventory):
    """Resume private copying scratch; publish the verified directory atomically."""
    copying = destination / ".inherited_parent.copying"
    scratch = destination / ".inherited_parent.file.part"
    if copying.is_symlink():
        raise ValueError("Partial parent snapshot must not be a symbolic link")
    copying.mkdir(exist_ok=True)
    present = _partial_inventory(copying, inventory)
    for relative, expected in inventory.items():
        if relative in present:
            continue
        target = copying / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Only this unpublished transfer scratch can be truncated after a crash.
        # Reject aliases before truncating, so retry cannot mutate parent bytes.
        descriptor = os.open(scratch, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            mode = os.fstat(descriptor)
            if not stat.S_ISREG(mode.st_mode) or mode.st_nlink != 1:
                raise ValueError("Partial transfer scratch must be an unaliased regular file")
            os.ftruncate(descriptor, 0)
            digest, size = hashlib.sha256(), 0
            with os.fdopen(descriptor, "wb", closefd=False) as copied:
                with _open_regular(parent, relative) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        copied.write(block)
                        digest.update(block)
                        size += len(block)
                copied.flush()
                os.fsync(copied.fileno())
            if {"sha256": digest.hexdigest(), "bytes": size} != expected:
                raise ValueError("Parent evidence changed during its partial snapshot copy")
        finally:
            os.close(descriptor)
        if target.exists() or target.is_symlink():
            raise ValueError("Partial parent snapshot target appeared during copying")
        os.replace(scratch, target)
    if _inventory(copying) != inventory or _inventory(parent) != inventory:
        raise ValueError("Parent evidence changed while its immutable snapshot was copied")
    if snapshot.exists() or snapshot.is_symlink():
        raise ValueError("Inherited parent appeared before atomic snapshot publication")
    os.replace(copying, snapshot)


def prepare_continuation(
    parent_dir,
    out,
    decision,
    parent_binding,
    execution_source,
    *,
    allow_training=False,
    resume=False,
):
    """Snapshot an audited stop and return a GPU-independent continuation context.

    Returns ``parent_root`` (the current snapshot Path), ``parent_binding``,
    ``parent_runtime``, ``logical_identity`` (also ``sampling_identity``), the
    validated ``decision``, and ``continuation_hash``. ``contract`` (also
    ``runtime_binding``) is the exact value to place in runtime["continuation"].
    The current root identity must include continuation_hash. For resume the
    caller re-audits inherited_parent with its original recorded root, then calls
    this function with that snapshot as parent_dir. An interrupted initial copy
    instead requires the original parent until its verified snapshot is published.
    Completed copied files and existing snapshot files are never overwritten.
    """
    if allow_training is not True:
        raise ValueError("Explicit allow_training=True is required for continuation")
    decision = validate_continuation_decision(decision, parent_binding, execution_source)
    binding = _json(parent_binding)
    parent, destination = Path(parent_dir), Path(out)
    if parent.is_symlink() or destination.is_symlink():
        raise ValueError("Continuation parent and output must not be symbolic links")
    parent, destination = parent.resolve(), destination.resolve()
    snapshot = destination / "inherited_parent"
    if parent == destination or (parent != snapshot and parent in destination.parents):
        raise ValueError("Continuation output must not be inside its parent evidence")
    if destination in parent.parents and parent != snapshot:
        raise ValueError("Continuation parent has an unexpected location beneath output")
    inventory = _inventory(parent)
    runtime = _check_audited_files(parent, binding, inventory)
    logical, contract, record = _records(decision, binding, inventory)
    if destination.exists():
        files = {p.name for p in destination.iterdir()} - {".writer.lock"}
        if files and not resume:
            raise FileExistsError("Continuation exists; resume with identical decision and source")
    if resume and not (destination / "continuation_binding.json").is_file():
        raise ValueError("Resume requires the saved continuation binding")
    if snapshot.is_symlink():
        raise ValueError("Inherited parent must not be a symbolic link")
    if snapshot.exists() and _inventory(snapshot) != inventory:
        raise ValueError("Existing inherited parent changed; snapshot must never be overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    _exclusive_json(destination / "continuation_decision.json", decision)
    _exclusive_json(destination / "logical_sampling_identity.json", logical)
    _exclusive_json(destination / "continuation_binding.json", record)
    if not snapshot.exists():
        _copy_snapshot(parent, destination, snapshot, inventory)
    if _inventory(snapshot) != inventory or _inventory(parent) != inventory:
        raise ValueError("Parent evidence changed while its immutable snapshot was copied")
    return _context(snapshot, binding, runtime, logical, decision, contract)
