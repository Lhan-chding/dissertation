#!/usr/bin/env python3
"""Preserve one known interrupted candidate06 unit, then resume its CPU request.

Run manually inside a Slurm CPU allocation with --phase Q1 or --phase Q2.
This operator helper is outside the immutable checkout. It never submits jobs,
deletes originals, changes source/configuration, or follows partial-unit links.
Every recovery directory is single-use; interruption requires operator review.
Attempt03 preserves the first two failed attempts and accepts inherited setgid
with private 0700 access permissions. It never chmods a server directory.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

BASE = Path("/projects/varunssd/louis-ssvc/modeling_v3_20260915/candidate06")
CANONICAL_BASE = Path("/projects/_ssd/varunssd/louis-ssvc/modeling_v3_20260915/candidate06")
PYTHON = Path("/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python")
SNAPSHOT_SHA256 = "cd703f28725212aa17e86ac8d0be82890f253c955aba7293fc9d84728d34b52d"
CONFIG = "configs/modeling_v3/protocol.json"
ATTEMPT = "03"
PREVIOUS_HELPER_SHA256 = "f20943c7c916ed8f13ad1f9e6cd37b33819040217de2e385b97cadbe51bcd621"
ATTEMPT02_HELPER_SHA256 = "8023ed7ea8cb8e976615697aad43e59e352ca93f408bede6efc355b729d001bb"
PHASES = {
    "Q1": {
        "tag": "Q1_OBSERVATION",
        "command": "observe-toy",
        "out": "Q1_observation_full",
        "collection": "Q1_historical_collection",
        "job": "155942",
        "previous_recovery_job": "156492",
        "attempt02_recovery_job": "156527",
        "partial": "historical_seed501_X_BASE_a24_b6_a24_b0_IDENTICAL_POLICY_origin_n4096",
        "completed_count": 27,
        "request_sha256": "3bdb1b69a704e7815d3175243fd0a29aed7ac800b30a6ea26cc10b6987f949ab",
        "binding_sha256": "f3968bc603d302290baceb422acfe5b7fc459deb47d2584ba2dd6c42048d08df",
        "collection_sha256": "473cfcb9aa527536b5752834b3438499555f79b89adfec0b46e6dfda7642a7e1",
    },
    "Q2": {
        "tag": "Q2_COVERAGE",
        "command": "coverage-study",
        "out": "Q2_coverage_full",
        "collection": "Q2_development_collection",
        "job": "155943",
        "previous_recovery_job": "156493",
        "attempt02_recovery_job": "156528",
        "partial": "seed201_X_BASE_init7001_a8_repeat00",
        "completed_count": 6,
        "request_sha256": "0e64fa7764c5e4058b5c556d348ce1a4e7d8f8e5a894589b11e562924ffac32d",
        "binding_sha256": "f6af19cd7ef84930f9705a52897763f1fa5997e95ed7ba2b5e6873869fbe26b1",
        "collection_sha256": "a1cbcda7d14bb927112779f691b94d6fed1db336ef84bc727d6ae3c8990408cc",
    },
}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("invalid relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or str(relative) != value
        or any(part in {".", "..", ".git"} for part in relative.parts)
        or value == "."
    ):
        raise ValueError("unsafe relative path: " + value)
    return relative


def checked_path(root, relative, *, exists=True):
    """Only the explicitly resolved base may use the known upstream alias."""
    path = root / safe_relative(relative)
    cursor = root
    for component in path.relative_to(root).parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("symlink on protected path: " + str(cursor))
    resolved = path.resolve(strict=exists)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("path escapes fixed base: " + str(path))
    return path


def sha256_file(path):
    """Read regular files only; reject a replaced file or a final symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("not a regular file: " + str(path))
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
        after.st_dev,
    ):
        raise ValueError("file changed while hashing: " + str(path))
    return digest.hexdigest()


def write_new(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)
    return hashlib.sha256(data).hexdigest()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def tree_entries(root, *, hash_files):
    """Inventory all names, including .pending files; never traverse symlinks."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("plain directory required: " + str(root))
    entries = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        files.sort()
        for name in [*dirs, *files]:
            path = Path(directory) / name
            relative = str(path.relative_to(root))
            safe_relative(relative)
            info = path.lstat()
            row = {"mode": stat.S_IMODE(info.st_mode)}
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                target_bytes = os.fsencode(target)
                row.update(
                    type="symlink",
                    target=target,
                    size=len(target_bytes),
                    link_text_sha256=hashlib.sha256(target_bytes).hexdigest(),
                )
            elif stat.S_ISDIR(info.st_mode):
                row.update(type="directory")
            elif stat.S_ISREG(info.st_mode):
                row.update(type="file", size=info.st_size)
                if hash_files:
                    row["sha256"] = sha256_file(path)
            else:
                raise ValueError("special file cannot be preserved by this helper: " + str(path))
            entries[relative] = row
    return entries


def verify_complete_tree(root, campaign, expected_binding=None):
    """Validate all existing completion markers, including nested units."""
    inventory = tree_entries(root, hash_files=False)
    if any(row["type"] == "symlink" for row in inventory.values()):
        raise ValueError("symlink in completed scientific originals: " + str(root))
    markers = sorted(name for name in inventory if PurePosixPath(name).name == "COMPLETE.json")
    if "COMPLETE.json" not in markers:
        raise ValueError("completed unit has no marker: " + str(root))
    # Leaf-first checks retain the complete dependency chain without loading arrays.
    for name in sorted(markers, key=lambda value: (-len(PurePosixPath(value).parts), value)):
        target = checked_path(root, name)
        campaign._verify_complete(
            target.parent, expected_binding if name == "COMPLETE.json" else None
        )
    return {name: sha256_file(root / name) for name in markers}


def validate_original_request(request, spec):
    expected = [
        spec["command"],
        "--collection",
        str(BASE / spec["collection"]),
        "--config",
        CONFIG,
        "--out",
        str(BASE / spec["out"]),
    ]
    if request != {"argv": expected}:
        raise ValueError("request differs from the original authorized CPU argv")
    return {"argv": [*expected, "--resume"]}


def enforce_cpu_environment():
    if sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("recovery requires Linux inside a Slurm CPU allocation")
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS"):
        raise RuntimeError("GPU-allocated jobs cannot run this recovery helper")
    if Path(sys.executable).resolve() != PYTHON.resolve(strict=True):
        raise RuntimeError("use the original verified production Python: " + str(PYTHON))
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    sys.dont_write_bytecode = True


def verify_snapshot(base):
    checkout = checked_path(base, "checkout")
    manifest_path = checked_path(checkout, "CODE_SNAPSHOT_MANIFEST.json")
    if sha256_file(manifest_path) != SNAPSHOT_SHA256:
        raise ValueError("candidate06 snapshot manifest hash differs")
    inventory = json.loads(manifest_path.read_bytes())
    if not isinstance(inventory, dict) or len(inventory) != 1025:
        raise ValueError("candidate06 must contain exactly 1,025 bound files")
    for relative, digest in inventory.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid snapshot digest")
        if sha256_file(checked_path(checkout, relative)) != digest:
            raise ValueError("immutable source changed: " + relative)
    return checkout, inventory


def expected_units(phase, config, collection_root, collection, campaign, binding):
    result = {}
    for trajectory in collection["summary"]["trajectories"]:
        unit = checked_path(collection_root, trajectory["path"])
        identity = json.loads(checked_path(unit, "identity.json").read_bytes())
        for anchor in identity["anchors"]:
            if phase == "Q1":
                bank = min(
                    range(identity["bank_counts"]["calibration"]),
                    key=lambda b: campaign._digest([trajectory["id"], anchor, b, "Q1-fixed-pair"]),
                )
                for case in ("IDENTITY_HASH_SELECTED", "IDENTICAL_POLICY"):
                    for proposal in config["observation"]["proposals"]:
                        for n in config["observation"]["total_draws_grid"]:
                            name = f"{trajectory['id']}_a{anchor}_b{bank}_{case}_{proposal}_n{n}"
                            if name in result:
                                raise ValueError("duplicate Q1 identity")
                            result[name] = {
                                **binding,
                                "measurement_id": name,
                                "repeats": config["cpu"]["repeat_measurements_dev"],
                            }
            else:
                name = f"{trajectory['id']}_a{anchor}_repeat00"
                if name in result:
                    raise ValueError("duplicate Q2 identity")
                result[name] = {**binding, "origin": name}
    if len(result) != (192 if phase == "Q1" else 12):
        raise ValueError("original phase identity matrix differs")
    return result


def verify_attempt01_failure(base, phase, spec):
    """Keep attempt01 and its empty destination unchanged; bind their bytes."""
    previous = checked_path(base, "RECOVERY_" + phase + "_01")
    expected_files = {
        "INTENT.json",
        "PARTIAL_INVENTORY.json",
        "VERIFIED_BEFORE_MOVE.json",
        "FAILED.json",
    }
    if {path.name for path in previous.iterdir()} != expected_files:
        raise ValueError("attempt01 files differ or a move/execution was recorded")
    old_entries = tree_entries(previous, hash_files=True)
    if any(row["type"] != "file" for row in old_entries.values()):
        raise ValueError("attempt01 must contain only its original regular receipt files")
    intent = json.loads(checked_path(previous, "INTENT.json").read_bytes())
    failed = json.loads(checked_path(previous, "FAILED.json").read_bytes())
    if (
        str(intent.get("slurm_job_id")) != spec["previous_recovery_job"]
        or intent.get("helper_sha256") != PREVIOUS_HELPER_SHA256
        or intent.get("snapshot_sha256") != SNAPSHOT_SHA256
        or intent.get("phase") != phase
        or failed.get("exception_type") != "OSError"
        or "[Errno 22] Invalid argument" not in failed.get("message", "")
        or failed.get("status") != "STOPPED"
    ):
        raise ValueError("attempt01 is not the known unsupported-rename failure")
    old_backup = checked_path(base, "INTERRUPTED_ORIGINALS_" + spec["job"])
    if not old_backup.is_dir() or list(old_backup.iterdir()):
        raise ValueError("attempt01 backup is not an unchanged empty directory")
    return {
        "record": str(previous),
        "entries": old_entries,
        "old_backup": str(old_backup),
        "old_backup_empty": True,
        "failure_job": spec["previous_recovery_job"],
        "observed_errno": 22,
    }


def verify_previous_failure(base, phase, spec):
    """Bind both unchanged failures and their still-empty backup directories."""
    first = verify_attempt01_failure(base, phase, spec)
    previous = checked_path(base, "RECOVERY_" + phase + "_02")
    expected_files = {
        "INTENT.json",
        "PARTIAL_INVENTORY.json",
        "VERIFIED_BEFORE_MOVE.json",
        "FAILED.json",
        "PREVIOUS_ATTEMPT_PRESERVED.json",
    }
    if {path.name for path in previous.iterdir()} != expected_files:
        raise ValueError("attempt02 files differ or a move/execution was recorded")
    entries = tree_entries(previous, hash_files=True)
    if any(row["type"] != "file" for row in entries.values()):
        raise ValueError("attempt02 must contain only its original regular receipt files")
    intent = json.loads(checked_path(previous, "INTENT.json").read_bytes())
    failed = json.loads(checked_path(previous, "FAILED.json").read_bytes())
    if (
        str(intent.get("slurm_job_id")) != spec["attempt02_recovery_job"]
        or intent.get("helper_sha256") != ATTEMPT02_HELPER_SHA256
        or intent.get("snapshot_sha256") != SNAPSHOT_SHA256
        or intent.get("phase") != phase
        or intent.get("recovery_attempt") != "02"
        or failed.get("exception_type") != "ValueError"
        or failed.get("message") != "backup parent must be owned by this operator with mode 0700"
        or failed.get("status") != "STOPPED"
    ):
        raise ValueError("attempt02 is not the known inherited-setgid permission failure")
    if json.loads(checked_path(previous, "PREVIOUS_ATTEMPT_PRESERVED.json").read_bytes()) != first:
        raise ValueError("attempt02's preserved attempt01 binding differs")
    first_inventory = json.loads((Path(first["record"]) / "PARTIAL_INVENTORY.json").read_bytes())
    second_inventory = json.loads(checked_path(previous, "PARTIAL_INVENTORY.json").read_bytes())
    if any(first_inventory.get(key) != second_inventory.get(key) for key in ("source", "entries")):
        raise ValueError("attempt01/02 partial inventories differ")
    old_backup = checked_path(base, "INTERRUPTED_ORIGINALS_" + spec["job"] + "_ATTEMPT02")
    if not old_backup.is_dir() or list(old_backup.iterdir()):
        raise ValueError("attempt02 backup is not an unchanged empty directory")
    return {
        "record": str(previous),
        "entries": entries,
        "old_backup": str(old_backup),
        "old_backup_empty": True,
        "failure_job": spec["attempt02_recovery_job"],
        "observed_error": failed["message"],
        "previous_attempt01": first,
    }


def private_ceph_rename(source, destination, inventory, owner):
    """One ordinary rename under a private, exclusive, persistent owner lock.

    The caller must exclusively create the destination parent with access mode
    0700. Inherited setgid is retained and recorded, never cleared with chmod.
    Plain rename has no no-replace flag: its safety here requires this dedicated
    parent, one cooperative operator owner, and no concurrent external writer.
    The lock is never removed, even on failure. Any error needs a new reviewed
    attempt; there is no automatic syscall retry or copy/delete fallback.
    """
    if source.is_symlink() or not source.is_dir() or destination.parent.is_symlink():
        raise ValueError("plain source and destination-parent directories required")
    parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    source_parent_fd = os.open(source.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    owner_fd = None
    try:
        parent_stat = os.fstat(parent_fd)
        if parent_stat.st_uid != os.geteuid() or parent_stat.st_mode & 0o777 != 0o700:
            raise ValueError("backup parent must be owned by this operator with mode 0700")
        parent_modes = {
            "destination_parent_mode": oct(stat.S_IMODE(parent_stat.st_mode)),
            "destination_parent_permissions": f"{parent_stat.st_mode & 0o777:04o}",
            "destination_parent_setgid": bool(parent_stat.st_mode & stat.S_ISGID),
            "destination_parent_gid": parent_stat.st_gid,
        }
        if source.lstat().st_dev != parent_stat.st_dev:
            raise ValueError("preservation rename must stay on the same filesystem")
        if os.listdir(parent_fd):
            raise FileExistsError("new private backup parent is not empty; do not reuse")
        owner_fd = os.open(
            "OWNER_LOCK.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        owner_data = (
            json.dumps(
                {
                    **owner,
                    "pid": os.getpid(),
                    "uid": os.geteuid(),
                    "utc": utc(),
                    "exclusive": True,
                    **parent_modes,
                    "mechanism": f"EXPLICIT_ATTEMPT{ATTEMPT}_CEPH_PLAIN_RENAME",
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode()
        with os.fdopen(os.dup(owner_fd), "wb") as stream:
            stream.write(owner_data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(parent_fd)
        if tree_entries(source, hash_files=True) != inventory:
            raise ValueError("partial changed after inventory; no move performed")
        source_stat = source.lstat()
        # Pinned directory descriptors and the persistent owner prevent a second
        # helper from claiming or replacing this attempt's destination.
        if (destination.parent.lstat().st_dev, destination.parent.lstat().st_ino) != (
            parent_stat.st_dev,
            parent_stat.st_ino,
        ):
            raise ValueError("backup parent path changed during verification")
        owner_stat = os.fstat(owner_fd)
        visible_owner = os.stat("OWNER_LOCK.json", dir_fd=parent_fd, follow_symlinks=False)
        if (owner_stat.st_dev, owner_stat.st_ino) != (
            visible_owner.st_dev,
            visible_owner.st_ino,
        ) or set(os.listdir(parent_fd)) != {"OWNER_LOCK.json"}:
            raise ValueError("exclusive backup ownership changed")
        try:
            os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("destination already exists; rename is forbidden")
        visible_source = os.stat(source.name, dir_fd=source_parent_fd, follow_symlinks=False)
        if (visible_source.st_dev, visible_source.st_ino) != (
            source_stat.st_dev,
            source_stat.st_ino,
        ) or not stat.S_ISDIR(visible_source.st_mode):
            raise ValueError("source path changed before rename")
        os.rename(source.name, destination.name, src_dir_fd=source_parent_fd, dst_dir_fd=parent_fd)
        os.fsync(source_parent_fd)
        os.fsync(parent_fd)
        if tree_entries(destination, hash_files=True) != inventory or os.path.lexists(source):
            raise ValueError("post-rename preservation verification failed; no CPU request run")
        return {
            "owner_lock": str(destination.parent / "OWNER_LOCK.json"),
            "owner_lock_sha256": hashlib.sha256(owner_data).hexdigest(),
            **parent_modes,
            "uid": os.geteuid(),
            "mechanism": f"EXPLICIT_ATTEMPT{ATTEMPT}_CEPH_PLAIN_RENAME",
            "rename_calls": 1,
            "automatic_retry": False,
        }
    finally:
        if owner_fd is not None:
            os.close(owner_fd)
        os.close(source_parent_fd)
        os.close(parent_fd)


def recover(phase, base, record, spec):
    previous_failure = verify_previous_failure(base, phase, spec)
    write_new(record / "PREVIOUS_ATTEMPT_PRESERVED.json", previous_failure)
    checkout, snapshot = verify_snapshot(base)
    flow = checked_path(checkout, "ssvc_flow")
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise RuntimeError("frozen CPU dependencies must not have been pre-imported")
    os.chdir(flow)
    sys.path.insert(0, str(flow))
    from src.modeling_v3 import cpu_campaign as campaign
    from src.modeling_v3.schema import load_config

    config_path = checked_path(flow, CONFIG)
    config = load_config(config_path)
    source_hashes = campaign._sources()
    if any(snapshot.get("ssvc_flow/" + name) != digest for name, digest in source_hashes.items()):
        raise ValueError("CPU dependencies do not match frozen snapshot")
    for name, module in list(sys.modules.items()):
        if name == "src" or name.startswith("src."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(flow):
                raise ValueError("import escaped frozen checkout: " + name)
            relative = str(path.relative_to(checkout))
            if sha256_file(path) != snapshot.get(relative):
                raise ValueError("imported source absent from frozen manifest: " + name)

    request_path = checked_path(base, spec["tag"] + "_REQUEST.json")
    if sha256_file(request_path) != spec["request_sha256"]:
        raise ValueError("original request hash differs")
    resumed_request = validate_original_request(json.loads(request_path.read_bytes()), spec)
    collection_root = checked_path(base, spec["collection"])
    if sha256_file(checked_path(collection_root, "COMPLETE.json")) != spec["collection_sha256"]:
        raise ValueError("original collection completion hash differs")
    collection_markers = verify_complete_tree(collection_root, campaign)
    collection = json.loads((collection_root / "COMPLETE.json").read_bytes())
    if (
        collection["summary"]["role"] != "development"
        or collection["binding"]["source_hashes"] != source_hashes
        or collection["binding"]["config_sha256"] != campaign._digest(config)
    ):
        raise ValueError("collection source/config/development identity differs")
    binding = campaign._binding(
        config, stage=phase, pilot=False, collection_sha256=spec["collection_sha256"]
    )
    if phase == "Q2":
        binding.update(
            role="development",
            designs=campaign.development_designs(config),
            calibration_receipt=None,
        )
    out = checked_path(base, spec["out"])
    binding_path = checked_path(out, "BINDING.json")
    if (
        sha256_file(binding_path) != spec["binding_sha256"]
        or json.loads(binding_path.read_bytes()) != binding
    ):
        raise ValueError("original output binding differs")
    allowed_top = {
        "BINDING.json",
        "units",
        "analytic_witnesses" if phase == "Q1" else "DESIGNS.json",
    }
    if {path.name for path in out.iterdir()} != allowed_top:
        raise ValueError("unexpected top-level output or completed phase; operator review required")
    auxiliary_markers = {}
    if phase == "Q1":
        witness = checked_path(out, "analytic_witnesses")
        auxiliary_markers = verify_complete_tree(
            witness, campaign, campaign._binding(config, stage="ANALYTIC_WITNESS")
        )
    elif json.loads(checked_path(out, "DESIGNS.json").read_bytes()) != binding["designs"]:
        raise ValueError("frozen development design matrix changed")

    units = checked_path(out, "units")
    expected = expected_units(phase, config, collection_root, collection, campaign, binding)
    completed, partials = {}, []
    for path in sorted(units.iterdir()):
        checked_path(units, path.name)
        if not path.is_dir() or path.name not in expected:
            raise ValueError("unexpected unit path: " + str(path))
        if (path / "COMPLETE.json").exists():
            print(json.dumps({"utc": utc(), "verifying_complete_unit": path.name}), flush=True)
            completed[path.name] = verify_complete_tree(path, campaign, expected[path.name])
        else:
            partials.append(path.name)
    if partials != [spec["partial"]] or len(completed) != spec["completed_count"]:
        raise ValueError(
            "interrupted attempt topology differs; unexpected/missing partial or complete unit"
        )
    partial = checked_path(units, spec["partial"])
    if json.loads(checked_path(partial, "BINDING.json").read_bytes()) != expected[partial.name]:
        raise ValueError("known partial binding differs")
    inventory = tree_entries(partial, hash_files=True)
    nested_completed = {}
    for relative, row in inventory.items():
        if PurePosixPath(relative).name == "COMPLETE.json":
            if row["type"] != "file":
                raise ValueError("invalid nested completion marker in partial")
            target = checked_path(partial, relative)
            nested_completed[relative] = verify_complete_tree(target.parent, campaign)
    previous_inventory = json.loads(
        (Path(previous_failure["record"]) / "PARTIAL_INVENTORY.json").read_bytes()
    )
    if (
        previous_inventory.get("source") != str(partial)
        or previous_inventory.get("entries") != inventory
    ):
        raise ValueError("partial inventory differs from the preserved attempt02 inventory")
    backup_parent = checked_path(
        base, "INTERRUPTED_ORIGINALS_" + spec["job"] + "_ATTEMPT" + ATTEMPT, exists=False
    )
    if os.path.lexists(backup_parent):
        raise FileExistsError("existing backup will not be overwritten: " + str(backup_parent))
    saved_unit = backup_parent / partial.name
    manifest = {
        "schema": "ssvc-v3-interrupted-unit-preservation-v1",
        "utc": utc(),
        "phase": phase,
        "recovery_attempt": ATTEMPT,
        "interrupted_job": spec["job"],
        "source": str(partial),
        "destination": str(saved_unit),
        "pending_files_included": True,
        "symlinks_followed": False,
        "entries": inventory,
        "nested_complete_markers_verified": nested_completed,
        "cleanup_status": "PRESERVED; REVIEW_AFTER_REDO; NO_DELETION_BY_HELPER",
    }
    manifest_sha = write_new(record / "PARTIAL_INVENTORY.json", manifest)
    verification = {
        "snapshot_sha256": SNAPSHOT_SHA256,
        "snapshot_file_count": len(snapshot),
        "config_file_sha256": sha256_file(config_path),
        "config_sha256": campaign._digest(config),
        "source_hashes": source_hashes,
        "original_request_sha256": spec["request_sha256"],
        "original_binding_sha256": spec["binding_sha256"],
        "collection_complete_markers": collection_markers,
        "auxiliary_complete_markers": auxiliary_markers,
        "reusable_complete_units": completed,
        "partial_inventory_sha256": manifest_sha,
    }
    write_new(record / "VERIFIED_BEFORE_MOVE.json", verification)
    # This private parent is never shared with or reused from either earlier attempt.
    backup_parent.mkdir(mode=0o700)
    fsync_directory(base)
    move = private_ceph_rename(
        partial,
        saved_unit,
        inventory,
        {
            "phase": phase,
            "recovery_attempt": ATTEMPT,
            "record": str(record),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "inventory_sha256": manifest_sha,
        },
    )
    write_new(
        record / "MOVED_AND_VERIFIED.json",
        {
            "utc": utc(),
            "source": str(partial),
            "destination": str(saved_unit),
            "same_filesystem_rename": True,
            "all_entries_equal": True,
            "partial_inventory_sha256": manifest_sha,
            "deleted_files": 0,
            **move,
        },
    )
    recovery_request_path = record / "RESUME_REQUEST.json"
    request_sha = write_new(recovery_request_path, resumed_request)
    wrapper = checked_path(flow, "scripts/run_modeling_v3_cpu_request.py")
    argv = [
        str(PYTHON),
        "-B",
        str(wrapper),
        "--request",
        str(recovery_request_path),
        "--sha256",
        request_sha,
    ]
    write_new(
        record / "EXECUTION_INTENT.json",
        {
            "utc": utc(),
            "argv": argv,
            "cwd": str(flow),
            "request_sha256": request_sha,
            "original_request_sha256": spec["request_sha256"],
            "only_cli_change": "APPEND --resume",
            "new_gpu_calls": 0,
            "jobs_submitted": 0,
        },
    )
    started = time.monotonic()
    result = subprocess.run(argv, cwd=flow, check=False)
    execution = {
        "utc": utc(),
        "returncode": result.returncode,
        "wall_seconds": time.monotonic() - started,
        "request_sha256": request_sha,
    }
    write_new(record / "EXECUTION_RESULT.json", execution)
    if result.returncode:
        raise RuntimeError(
            "original CPU wrapper failed; preserve all current outputs and inspect receipts"
        )
    campaign._verify_complete(out, binding)
    if sha256_file(binding_path) != spec["binding_sha256"]:
        raise ValueError("original top-level binding changed during resume")
    for name, markers in completed.items():
        campaign._verify_complete(out / "units" / name, expected[name])
        if any(
            sha256_file(out / "units" / name / relative) != digest
            for relative, digest in markers.items()
        ):
            raise ValueError("reused completion marker changed: " + name)
    if tree_entries(saved_unit, hash_files=True) != inventory:
        raise ValueError("preserved interrupted originals changed during resume")
    write_new(
        record / "RECOVERY_COMPLETE.json",
        {
            **execution,
            "status": "COMPLETE",
            "scientific_confirmation": False,
            "phase_complete_sha256": sha256_file(out / "COMPLETE.json"),
            "reused_complete_unit_count": len(completed),
            "preserved_partial": str(saved_unit),
            "partial_inventory_sha256": manifest_sha,
            "cleanup_status": "REDO_COMPLETE; BACKUP_FOR_LATER_OPERATOR_REVIEW; NOT_DELETED",
        },
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    args = parser.parse_args()
    enforce_cpu_environment()
    base = BASE.resolve(strict=True)
    if base != CANONICAL_BASE or not base.is_dir():
        raise ValueError("candidate06 base resolves outside its fixed canonical location")
    record = checked_path(base, "RECOVERY_" + args.phase + "_" + ATTEMPT, exists=False)
    record.mkdir(mode=0o700)
    fsync_directory(base)
    write_new(
        record / "INTENT.json",
        {
            "schema": "ssvc-v3-candidate06-cpu-recovery-v1",
            "utc": utc(),
            "phase": args.phase,
            "recovery_attempt": ATTEMPT,
            "previous_attempts": ["01_PRESERVED_UNCHANGED", "02_PRESERVED_UNCHANGED"],
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "helper_sha256": sha256_file(Path(__file__)),
            "snapshot_sha256": SNAPSHOT_SHA256,
            "gpu_allocation": False,
            "scope": "VERIFY_COMPLETE; PRESERVE_KNOWN_PARTIAL; RESUME_ORIGINAL_CPU_REQUEST",
        },
    )
    try:
        return recover(args.phase, base, record, PHASES[args.phase])
    except BaseException as error:
        failure = {
            "utc": utc(),
            "status": "STOPPED",
            "exception_type": type(error).__name__,
            "message": str(error),
            "automatic_retry": False,
            "all_existing_files_preserved": True,
        }
        print(json.dumps(failure), flush=True)
        try:
            write_new(record / "FAILED.json", failure)
        except OSError as receipt_error:
            print(json.dumps({"failure_receipt_write_error": str(receipt_error)}), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
