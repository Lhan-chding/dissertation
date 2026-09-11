"""Reviewed continuation contracts and immutable snapshots of stopped R4 evidence.

The caller must first run ``audit_stopped_r4``. This module binds that audit to
the explicit user decision and current source before any model is loaded. A
snapshot is copied byte for byte; old identities, alarms and checkpoints are
never rewritten. Current execution identities belong to the caller, while the
returned logical identity keeps the original sample keys and random seeds.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import math
import os
import stat
import time
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

# Version 1 remains byte-for-byte compatible with the first reviewed decision.
# This distinct, source-bound amendment never changes the measured thresholds.
REVIEWED_CUMULATIVE_POLICY = {
    "reviewed_warning_policy": "finite_cumulative_kl_or_sequence_p99_two_arms_to_step64",
    "arms": ["X_BASE", "X_VALID"],
    "steps": 64,
    "thresholds": copy.deepcopy(REVIEWED_POLICY["thresholds"]),
    "sequence_p99": "preserve_warning_and_continue",
    "cumulative_mean_kl": "preserve_warning_and_continue",
    "hard_stops": [
        "nonfinite_diagnostic",
        "measurement_fault",
        "policy_contamination",
        "hash_mismatch",
        "parity_failure",
    ],
}


def policy_for_name(name):
    """Return a detached supported policy; names cannot select arbitrary overrides."""
    if isinstance(name, str):
        for policy in (REVIEWED_POLICY, REVIEWED_CUMULATIVE_POLICY):
            if name == policy["reviewed_warning_policy"]:
                return copy.deepcopy(policy)
    raise ValueError("Unsupported reviewed warning policy")


def _consistent_alarm(diagnostic, *, legacy_missing_status=False):
    if not isinstance(diagnostic, dict):
        return False
    try:
        # Reject nonfinite values even in detailed group/prompt measurements.
        canonical_hash(diagnostic)
        if not _same(diagnostic.get("thresholds"), REVIEWED_POLICY["thresholds"]):
            return False
        mean = diagnostic.get("mean_token_kl")
        sequence = diagnostic.get("sequence_log_ratio_p99_abs")
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (mean, sequence)
        ):
            return False
        alarms = {"mean_token_kl": mean > 0.1, "sequence_log_ratio_p99_abs": sequence > 2.0}
        return (
            any(alarms.values())
            and diagnostic.get("should_stop") is True
            and _same(diagnostic.get("alarms"), alarms)
            and (
                diagnostic.get("status") == "STOP_DIAGNOSE"
                or (legacy_missing_status and "status" not in diagnostic)
            )
        )
    except (ValueError, TypeError, OverflowError):
        return False


def allows_reviewed_warning(diagnostic, policy):
    """Whether this exact supported decision permits the original measured alarm.

    This only classifies a diagnostic; callers still enforce arm/step scope and
    reject preserved integrity, measurement and parity faults independently.
    """
    if not isinstance(policy, dict):
        return False
    try:
        supported = policy_for_name(policy.get("reviewed_warning_policy"))
        if not _same(policy, supported) or not _consistent_alarm(diagnostic):
            return False
        return (
            supported["reviewed_warning_policy"]
            == REVIEWED_CUMULATIVE_POLICY["reviewed_warning_policy"]
            or diagnostic["mean_token_kl"] <= 0.1
        )
    except (ValueError, TypeError):
        return False


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
    if not _consistent_alarm(diagnostic, legacy_missing_status=True):
        raise ValueError("Stopped-R4 diagnostic must retain a consistent finite measured alarm")


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
    if type(decision["schema_version"]) is not int or decision["schema_version"] not in (1, 2):
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
    policy = REVIEWED_POLICY if decision["schema_version"] == 1 else REVIEWED_CUMULATIVE_POLICY
    if not _same(decision["policy"], policy):
        raise ValueError("Continuation policy, two arms, steps or thresholds changed")
    diagnostic = copy.deepcopy(binding["stop"]["diagnostic"])
    if decision["schema_version"] == 1:
        # The original contract accepted audit fixtures without a status field.
        # Real stored diagnoses still carry their exact STOP_DIAGNOSE status.
        diagnostic.setdefault("status", "STOP_DIAGNOSE")
        if "continuation" in binding:
            raise ValueError("Version 1 requires an original stopped R4 parent")
    if not allows_reviewed_warning(diagnostic, policy):
        raise ValueError("Stopped-R4 alarm is outside the bound reviewed policy")
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


def _inventory_once(root):
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


def _inventory(root):
    """Retry only transient ENOENT inside a complete, read-only inventory pass.

    A failed pass contributes no hashes. Every successful result still undergoes
    the caller's exact inventory/hash checks; absent roots and other errors fail.
    """
    delays = (0.1, 0.5, 1.0, 2.0)
    for attempt in range(len(delays) + 1):
        try:
            return _inventory_once(root)
        except (FileNotFoundError, ValueError) as exc:
            underlying = exc if isinstance(exc, OSError) else exc.__cause__
            missing_entry = isinstance(underlying, OSError) and underlying.errno == errno.ENOENT
            if not missing_entry or attempt == len(delays):
                raise
            if root.is_symlink() or not root.is_dir():
                raise ValueError("Continuation evidence root must be a real directory") from exc
            time.sleep(delays[attempt])


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
    inherited_contract = runtime.get("continuation")
    if inherited_contract is not None:
        logical = binding.get("logical_sampling_identity")
        if (
            not isinstance(logical, dict)
            or not _same(binding.get("continuation"), inherited_contract)
            or not _same(_json(parent / "logical_sampling_identity.json"), logical)
            or inherited_contract.get("logical_sampling_identity_sha256") != canonical_hash(logical)
        ):
            raise ValueError("Stopped-R4 inherited logical sampling identity changed")
        excluded = {"source_hash", "continuation_hash"}
        if not _same(
            {key: value for key, value in logical.items() if key not in excluded},
            {key: value for key, value in binding["identity"].items() if key not in excluded},
        ):
            raise ValueError("Stopped-R4 logical identity changed the experiment")
    elif "continuation" in binding or (
        "logical_sampling_identity" in binding
        and not _same(binding["logical_sampling_identity"], binding["identity"])
    ):
        raise ValueError("Stopped-R4 logical sampling identity lacks its inherited contract")
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
    version = decision["schema_version"]
    logical = copy.deepcopy(
        binding.get("logical_sampling_identity", binding["identity"])
        if version == 2
        else binding["identity"]
    )
    stable = {
        "schema_version": version,
        "parent_audit_hash": binding["audit_hash"],
        "decision_sha256": canonical_hash(decision),
        "parent_snapshot_sha256": canonical_hash(inventory),
        "logical_sampling_identity_sha256": canonical_hash(logical),
        "reviewed_warning_policy": decision["policy"]["reviewed_warning_policy"],
    }
    contract = {
        **stable,
        "continuation_hash": canonical_hash(stable),
        "parent_dir": "inherited_parent",
        "parent_binding": binding,
        **({"logical_sampling_identity": logical} if version == 2 else {}),
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


def _evidence_path(root, relative):
    """Inspect an existing path beneath a verified root without following links."""
    current = root
    for index, part in enumerate(relative.parts):
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("Inherited evidence contains a symlink or nonregular entry")
        if index + 1 < len(relative.parts) and not stat.S_ISDIR(mode):
            raise ValueError("Inherited evidence path crosses a non-directory")
    return current


def resolve_evidence(root, relative):
    """Locate an artifact in an already audited, explicitly declared parent chain.

    Logical paths fall back only through runtime.continuation.parent_dir.
    Explicit inherited_parent prefixes are consumed as ownership transitions,
    never returned as part of another execution's path. This does not replace
    recursive manifest/source auditing. The returned relative path is relative
    to the actual owning execution and can identify either a ledger or a file.
    """
    relative = _relative(str(relative) if isinstance(relative, PurePosixPath) else relative)
    current = Path(root)
    if current.is_symlink() or not current.is_dir():
        raise ValueError("Inherited evidence root must be a real directory")
    current = current.resolve()
    seen = set()
    for depth in range(33):
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ValueError("Inherited evidence root must be a real directory")
        marker = (info.st_dev, info.st_ino)
        if marker in seen:
            raise ValueError("Inherited evidence ownership cycle")
        seen.add(marker)
        with _open_regular(current, "runtime_lock.json") as stream:
            runtime = json.load(stream)
        if not isinstance(runtime, dict) or not isinstance(runtime.get("identity"), dict):
            raise ValueError("Inherited evidence lacks its execution runtime identity")
        explicit = relative.parts[0] == "inherited_parent"
        candidate = None if explicit else _evidence_path(current, relative)
        if candidate is not None:
            return {"root": current, "runtime": runtime, "path": candidate, "relative": relative}
        contract = runtime.get("continuation")
        if not isinstance(contract, dict) or contract.get("parent_dir") != "inherited_parent":
            raise FileNotFoundError("Artifact is absent from the declared inherited evidence chain")
        if depth == 32:
            raise ValueError("Inherited evidence exceeds the supported depth of 32")
        if explicit:
            if len(relative.parts) == 1:
                raise ValueError("Inherited evidence path must identify an artifact")
            relative = PurePosixPath(*relative.parts[1:])
        parent = _evidence_path(current, PurePosixPath("inherited_parent"))
        if parent is None or not parent.is_dir():
            raise ValueError("Declared inherited evidence directory is missing")
        current = parent
    raise ValueError("Inherited evidence exceeds the supported depth of 32")


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
