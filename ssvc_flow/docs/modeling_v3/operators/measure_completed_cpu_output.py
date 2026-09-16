#!/usr/bin/env python3
"""Measure one completed candidate06 Q2/Q3 output inside a Slurm CPU job.

Within the measured tree only COMPLETE.json is read, for its supplied SHA256;
scientific data files are never opened. Pending names contribute metadata too.
This is a resource receipt, not scientific integrity validation or a snapshot.
The output directory must be new and outside the measured tree, within SSVC.
No scheduler commands, network calls, deletion, chmod, or source edits occur.
Ceph attributes are read separately at SSVC and its fixed physical parent;
shared quota headroom comes from the parent's attributes, never an SSVC zero.

Usage: python -B measure_completed_cpu_output.py --root ROOT \
    --complete-sha256 SHA256 --out NEW_RECEIPT_DIRECTORY
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import stat
import sys
import time
from contextlib import suppress
from pathlib import Path

PHYSICAL_SSVC_ROOT = Path("/projects/_ssd/varunssd/louis-ssvc")
LOGICAL_SSVC_ROOT = Path("/projects/varunssd/louis-ssvc")
CANDIDATE_RELATIVE = Path("modeling_v3_20260915/candidate06")
ROLES = {"interval_calibration", "locked_test", "orthogonal_generalization"}
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def enforce_cpu_environment():
    if (
        sys.platform != "linux"
        or re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")) is None
    ):
        raise RuntimeError("measurement requires Linux inside a numeric Slurm CPU allocation")
    # SLURM_JOB_GPUS='0' names GPU index zero, rather than zero allocated GPUs.
    for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        if os.environ.get(name):
            raise RuntimeError("GPU-allocated jobs cannot run this CPU measurement: " + name)
    for name in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK"):
        if os.environ.get(name, "") not in ("", "0"):
            raise RuntimeError("GPU allocation is not allowed: " + name)
    for name in ("SLURM_JOB_GRES", "SLURM_TRES_PER_TASK", "SLURM_TRES_PER_NODE"):
        if "gpu" in os.environ.get(name, "").lower():
            raise RuntimeError("GPU allocation is not allowed: " + name)
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if os.environ.get(name, "") not in ("", "-1"):
            raise RuntimeError("GPU visibility is not allowed: " + name)
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        os.environ[name] = ""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "PYTHONDONTWRITEBYTECODE",
    ):
        os.environ[name] = "1"
    sys.dont_write_bytecode = True


def physical_path(value):
    """Only the known upstream alias is allowed; no user-controlled resolve()."""
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute SSVC path without parent traversal required")
    if PHYSICAL_SSVC_ROOT.resolve(strict=True) != PHYSICAL_SSVC_ROOT:
        raise ValueError("configured physical SSVC root is not physical")
    if path.is_relative_to(PHYSICAL_SSVC_ROOT):
        return path
    if path.is_relative_to(LOGICAL_SSVC_ROOT):
        if LOGICAL_SSVC_ROOT.resolve(strict=True) != PHYSICAL_SSVC_ROOT:
            raise ValueError("known SSVC alias does not resolve to the fixed physical root")
        return PHYSICAL_SSVC_ROOT / path.relative_to(LOGICAL_SSVC_ROOT)
    raise ValueError("path is outside the fixed physical SSVC root")


def open_directory(path):
    """Open each component relative to its pinned parent, never following links."""
    fd = os.open(PHYSICAL_SSVC_ROOT, DIRECTORY_FLAGS)
    try:
        for component in path.relative_to(PHYSICAL_SSVC_ROOT).parts:
            child = os.open(component, DIRECTORY_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_quota_ancestor():
    """Read-only exception for the one fixed physical parent, not arbitrary paths."""
    ancestor = PHYSICAL_SSVC_ROOT.parent
    if ancestor.resolve(strict=True) != ancestor:
        raise ValueError("fixed physical quota ancestor must not contain symlinks")
    return os.open(ancestor, DIRECTORY_FLAGS)


def validate_root_scope(root):
    candidate = PHYSICAL_SSVC_ROOT / CANDIDATE_RELATIVE
    if not root.is_relative_to(candidate):
        raise ValueError("only candidate06 Q2/Q3 completed outputs are allowed")
    parts = root.relative_to(candidate).parts
    if parts == ("Q2_coverage_full",):
        return "Q2"
    if len(parts) in (2, 4) and parts[0] == "Q3_campaign":
        match = re.fullmatch(
            r"(interval_calibration|locked_test|orthogonal_generalization)_seed([0-9]+)", parts[1]
        )
        if match and (len(parts) == 2 or parts[2:] == (match.group(1), "response")):
            return "Q3"
    raise ValueError("only candidate06 Q2 full or one Q3 seed output/response root is allowed")


def identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def complete_identity(root_fd):
    fd = os.open("COMPLETE.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("COMPLETE.json must be a regular original file")
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if identity(before) != identity(after):
        raise ValueError("COMPLETE changed while hashing")
    return {"sha256": digest.hexdigest(), "stat_identity": list(identity(after))}


def empty_totals():
    return {
        name: 0
        for name in (
            "entry_count",
            "regular_file_count",
            "directory_count",
            "symlink_count",
            "other_entry_count",
            "regular_apparent_bytes",
            "regular_allocated_bytes",
            "unique_inode_count",
            "unique_regular_inode_count",
            "unique_regular_apparent_bytes",
            "unique_regular_allocated_bytes",
        )
    }


def partition(relative):
    parts = relative.parts
    if parts and parts[0] in {"collection", "response", "units"}:
        return parts[0]
    if len(parts) >= 2 and parts[0] in ROLES and parts[1] in {"collection", "response"}:
        return parts[1]
    return "other"


def scan_tree(root_fd):
    """One metadata traversal; open directory descriptors only, never data files."""
    total, seen = empty_totals(), set()
    partitions, partition_seen = {}, {}
    known_regular, roots = {}, {}
    pending_count = 0
    fingerprint = hashlib.sha256()
    device = os.fstat(root_fd).st_dev

    def record(relative, info):
        nonlocal pending_count
        group = partition(relative)
        aggregate = partitions.setdefault(group, empty_totals())
        local_seen = partition_seen.setdefault(group, set())
        key = (info.st_dev, info.st_ino)
        if not hasattr(info, "st_blocks"):
            raise ValueError("filesystem did not supply st_blocks")
        regular = stat.S_ISREG(info.st_mode)
        if regular:
            signature = (identity(info), info.st_blocks)
            if key in known_regular and known_regular[key] != signature:
                raise ValueError("regular inode metadata changed during traversal")
            known_regular[key] = signature
        for counts, visited in ((total, seen), (aggregate, local_seen)):
            counts["entry_count"] += 1
            if regular:
                counts["regular_file_count"] += 1
                counts["regular_apparent_bytes"] += info.st_size
                counts["regular_allocated_bytes"] += info.st_blocks * 512
            elif stat.S_ISDIR(info.st_mode):
                counts["directory_count"] += 1
            elif stat.S_ISLNK(info.st_mode):
                counts["symlink_count"] += 1
            else:
                counts["other_entry_count"] += 1
            if key not in visited:
                visited.add(key)
                counts["unique_inode_count"] += 1
                if regular:
                    counts["unique_regular_inode_count"] += 1
                    counts["unique_regular_apparent_bytes"] += info.st_size
                    counts["unique_regular_allocated_bytes"] += info.st_blocks * 512
        if relative != Path(".") and "pending" in relative.name.lower():
            pending_count += 1
        if regular and len(relative.parts) == 1:
            roots[str(relative)] = {
                "apparent_bytes": info.st_size,
                "allocated_bytes": info.st_blocks * 512,
            }
        fingerprint.update(
            json.dumps(
                [str(relative), *identity(info), info.st_blocks], separators=(",", ":")
            ).encode()
            + b"\n"
        )

    def walk(fd, relative):
        before = os.fstat(fd)
        if before.st_dev != device:
            raise ValueError("nested mount/device is outside this single-filesystem measurement")
        record(relative, before)
        with os.scandir(fd) as entries:
            # Sorting names gives reproducible metadata ordering, not data reads.
            names = sorted(entry.name for entry in entries)
        for name in names:
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            child_relative = relative / name
            if stat.S_ISDIR(info.st_mode):
                child_fd = os.open(name, DIRECTORY_FLAGS, dir_fd=fd)
                try:
                    if identity(os.fstat(child_fd)) != identity(info):
                        raise ValueError("directory changed before traversal")
                    walk(child_fd, child_relative)
                finally:
                    os.close(child_fd)
            else:
                record(child_relative, info)
        if identity(os.fstat(fd)) != identity(before):
            raise ValueError("directory changed during traversal")

    walk(root_fd, Path("."))
    return {
        "total": total,
        "partitions": partitions,
        "partition_rule": (
            "Disjoint paths: direct collection/response/units, or role/collection and "
            "role/response; all remaining paths including root in other."
        ),
        "directory_count_includes_measured_root": True,
        "partition_unique_counts_are_additive": False,
        "partition_unique_regular_apparent_overlap_bytes": sum(
            p["unique_regular_apparent_bytes"] for p in partitions.values()
        )
        - total["unique_regular_apparent_bytes"],
        "partition_unique_regular_allocated_overlap_bytes": sum(
            p["unique_regular_allocated_bytes"] for p in partitions.values()
        )
        - total["unique_regular_allocated_bytes"],
        "root_regular_files": roots,
        "pending_name_count": pending_count,
        "metadata_inventory_sha256": fingerprint.hexdigest(),
        "symlinks_followed": False,
        "pending_names_excluded": False,
        "atomic_filesystem_snapshot": False,
    }


def ceph_snapshot(parent_fd, path):
    result = {"timestamp_utc": utc(), "path": str(path), "attributes": {}}
    for name in ("ceph.dir.rbytes", "ceph.quota.max_bytes"):
        try:
            raw = os.getxattr(parent_fd, name)
            value = int(raw.decode("ascii"))
            if value < 0:
                raise ValueError("negative Ceph byte count")
            result["attributes"][name] = {"status": "AVAILABLE", "value": value}
        except (OSError, AttributeError, UnicodeError, ValueError) as exc:
            result["attributes"][name] = {
                "status": "UNAVAILABLE",
                "value": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "errno": getattr(exc, "errno", None),
            }
    used = result["attributes"]["ceph.dir.rbytes"]["value"]
    quota = result["attributes"]["ceph.quota.max_bytes"]["value"]
    result["headroom_at_this_directory_bytes"] = (
        quota - used if quota and used is not None else None
    )
    result["zero_quota_semantics"] = (
        "No quota set at this directory; not proof of unlimited space or absent ancestor quotas."
    )
    return result


def ceph_layer(before, after, layer):
    delta = {}
    for name in before["attributes"]:
        a, b = before["attributes"][name]["value"], after["attributes"][name]["value"]
        delta[name] = b - a if a is not None and b is not None else None
    return {
        "layer": layer,
        "before": before,
        "after": after,
        "shared_attribute_delta": delta,
        "delta_is_attributable_to_measured_task": False,
        "scope": (
            "Shared, asynchronously updated directory attributes, read only. Other jobs and "
            "retained files contribute; unavailable is not zero. INTENT exists before first "
            "reading; final RESOURCE_MEASUREMENT is written after last reading. Layer "
            "readings are sequential, not an atomic shared-filesystem snapshot."
        ),
    }


def write_new(out_fd, name, value):
    data = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=out_fd)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.fsync(out_fd)
    return hashlib.sha256(data).hexdigest()


def measure_completed_output(root, complete_sha256, out):
    enforce_cpu_environment()
    if re.fullmatch(r"[0-9a-f]{64}", complete_sha256) is None:
        raise ValueError("lowercase COMPLETE SHA256 required")
    root, out = physical_path(root), physical_path(out)
    stage = validate_root_scope(root)
    if out == PHYSICAL_SSVC_ROOT or out.is_relative_to(root):
        raise ValueError("new receipt directory must be outside the measured tree")
    root_fd = open_directory(root)
    parent_fd = ancestor_fd = out_parent_fd = out_fd = None
    try:
        initial = complete_identity(root_fd)
        if initial["sha256"] != complete_sha256:
            raise ValueError("initial COMPLETE SHA256 mismatch")
        parent_fd = open_directory(PHYSICAL_SSVC_ROOT)
        ancestor_fd = open_quota_ancestor()
        out_parent_fd = open_directory(out.parent)
        # mkdir is exclusive even when an existing destination is an empty dir
        # or dangling link. No retry, replacement, chmod, or mkdir(parents=True).
        os.mkdir(out.name, mode=0o700, dir_fd=out_parent_fd)
        out_fd = os.open(out.name, DIRECTORY_FLAGS, dir_fd=out_parent_fd)
        intent = {
            "schema": "ssvc-candidate06-cpu-output-resource-intent-1",
            "status": "MEASUREMENT_STARTED",
            "started_utc": utc(),
            "stage": stage,
            "root": str(root),
            "out": str(out),
            "expected_complete_sha256": complete_sha256,
            "initial_complete_identity": initial,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        write_new(out_fd, "INTENT.json", intent)
        before = ceph_snapshot(parent_fd, PHYSICAL_SSVC_ROOT)
        ancestor_before = ceph_snapshot(ancestor_fd, PHYSICAL_SSVC_ROOT.parent)
        started = time.monotonic()
        tree = scan_tree(root_fd)
        final = complete_identity(root_fd)
        if final != initial:
            raise ValueError("COMPLETE changed during measurement")
        # Verify the protected pathname still denotes our pinned directory.
        fresh_fd = open_directory(root)
        try:
            if (os.fstat(fresh_fd).st_dev, os.fstat(fresh_fd).st_ino) != (
                os.fstat(root_fd).st_dev,
                os.fstat(root_fd).st_ino,
            ):
                raise ValueError("measured root pathname changed during measurement")
        finally:
            os.close(fresh_fd)
        after = ceph_snapshot(parent_fd, PHYSICAL_SSVC_ROOT)
        ancestor_after = ceph_snapshot(ancestor_fd, PHYSICAL_SSVC_ROOT.parent)
        report = {
            **intent,
            "schema": "ssvc-candidate06-cpu-output-resource-measurement-1",
            "status": "MEASURED",
            "finished_utc": utc(),
            "measurement_elapsed_seconds": time.monotonic() - started,
            "elapsed_scope": (
                "Metadata scan, final COMPLETE hashing, path and Ceph checks; "
                "not original experiment wall time or peak RSS."
            ),
            "final_complete_identity": final,
            "complete_hash_unchanged": True,
            "tree": tree,
            "summary_bytes_used": False,
            "closing_files_scope": (
                "Full measured tree includes top-level COMPLETE, summaries, bindings and "
                "all pending names. Earlier summary output_bytes is not substituted or read."
            ),
            "ceph_parent": ceph_layer(before, after, "SSVC_DIRECTORY"),
            "ceph_quota_ancestor": ceph_layer(
                ancestor_before, ancestor_after, "FIXED_PHYSICAL_SSVC_PARENT"
            ),
            "shared_quota_headroom": {
                "source_layer": "ceph_quota_ancestor",
                "path": str(PHYSICAL_SSVC_ROOT.parent),
                "before_bytes": ancestor_before["headroom_at_this_directory_bytes"],
                "after_bytes": ancestor_after["headroom_at_this_directory_bytes"],
                "is_space_reserved_for_this_task": False,
                "scope": (
                    "Shared ancestor max_bytes minus ancestor rbytes only; SSVC quota=0 "
                    "does not override it. Unavailable or zero ancestor quota yields null. "
                    "Does not establish exclusive capacity or account for unqueried limits."
                ),
            },
            "space_semantics": {
                "regular_apparent_bytes": (
                    "Sum st_size for every regular-file pathname; hardlink paths count repeatedly."
                ),
                "regular_allocated_bytes": (
                    "Sum st_blocks * 512 for every regular-file pathname; excludes "
                    "directory/symlink blocks and repeats hardlinked inode blocks."
                ),
                "unique_inode_count": (
                    "Distinct (st_dev, st_ino) across all entry types including the root; "
                    "unique_regular_* is regular files only."
                ),
                "unique_regular_bytes": (
                    "Regular-file size/blocks once per (st_dev,st_ino) within the reported "
                    "scope; cross-partition hardlinks make partition unique totals nonadditive."
                ),
                "unique_allocated_bytes_are_exclusively_owned_physical_bytes": False,
                "limitation": (
                    "Inode deduplication does not account for outside hardlinks, shared/reflink "
                    "extents, Ceph replication/compression or physical exclusive ownership."
                ),
            },
            "data_arrays_opened": False,
            "scientific_data_values_read": False,
            "all_original_hashes_verified": False,
            "completion_scope": (
                "Supplied top-level COMPLETE SHA256 identity only; its manifest/data contents "
                "are not parsed and do not establish scientific correctness or prevent "
                "unrelated file writes."
            ),
            "scientific_originals_modified": False,
            "scheduler_commands_run": False,
        }
        write_new(out_fd, "RESOURCE_MEASUREMENT.json", report)
        return report
    except BaseException as exc:
        if out_fd is not None:
            with suppress(OSError):
                write_new(
                    out_fd,
                    "FAILED.json",
                    {
                        "status": "FAILED_NO_VALID_MEASUREMENT",
                        "timestamp_utc": utc(),
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
        raise
    finally:
        for fd in (out_fd, out_parent_fd, ancestor_fd, parent_fd, root_fd):
            if fd is not None:
                os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--complete-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = measure_completed_output(args.root, args.complete_sha256, args.out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt": str(Path(result["out"]) / "RESOURCE_MEASUREMENT.json"),
                "regular_apparent_bytes": result["tree"]["total"]["regular_apparent_bytes"],
                "unique_regular_allocated_bytes": result["tree"]["total"][
                    "unique_regular_allocated_bytes"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            json.dumps({"status": "FAILED", "type": type(error).__name__, "message": str(error)}),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
