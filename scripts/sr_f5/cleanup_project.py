"""Delete only archived F3 files and an explicitly opted-in obsolete model cache.

The canonical project boundary is fixed. No directory, current model, parent,
environment, or unlisted experiment file is removed. Default is a metadata-only
dry run; apply checks the retained archive once and each F3 candidate once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc")
SOURCE = "sr_f3_ce_protocol_fix_20261016/selected_v2"
OLD_CACHE = "cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct"
MIN_BYTES = 65536
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def relative_parts(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid relative path")
    parts = value.split("/")
    if PurePosixPath(value).is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise ValueError("Absolute paths and traversal are forbidden")
    return parts


def inside_root(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("CLI paths must be absolute inside the fixed project root")
    relative = path.relative_to(ROOT)
    parts = relative_parts(str(relative))
    return "/".join(parts)


def validate_member(member):
    if not isinstance(member, dict) or set(member) != {"relative", "bytes", "sha256"}:
        raise ValueError("Unexpected member schema")
    parts = relative_parts(member["relative"])
    permitted = (
        (len(parts) >= 4 and parts[0] == "runs" and parts[2] in {"rollouts", "support"})
        or (len(parts) >= 3 and parts[0] in {"monitor", "engine"})
        or member["relative"] in {"release/ENDPOINT_RAW.jsonl.gz", "release/ENDPOINT_SCORES.jsonl"}
    )
    if not permitted or any(p in {"parents", "checkpoints"} for p in parts):
        raise PermissionError("Protected or unregistered candidate path")
    if type(member["bytes"]) is not int or member["bytes"] < MIN_BYTES:
        raise ValueError("Only registered files of at least 64 KiB are candidates")
    if not re.fullmatch(r"[0-9a-f]{64}", member["sha256"]):
        raise ValueError("Invalid member digest")
    return member


@contextmanager
def project_fd():
    # Reject alias roots as well as symlinks anywhere above the fixed boundary.
    for path in (*reversed(ROOT.parents), ROOT):
        if stat.S_ISLNK(path.lstat().st_mode):
            raise PermissionError("Project path contains a symlink")
    fd = os.open(ROOT, DIR_FLAGS)
    try:
        yield fd
    finally:
        os.close(fd)


@contextmanager
def parent_fd(root_fd, relative, *, create=False):
    parts = relative_parts(relative)
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            if create:
                with suppress(FileExistsError):
                    os.mkdir(part, dir_fd=current)
            following = os.open(part, DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = following
        yield current, parts[-1]
    finally:
        os.close(current)


def metadata(value):
    return {
        key: getattr(value, "st_" + key)
        for key in ("dev", "ino", "mode", "size", "mtime_ns", "ctime_ns", "nlink", "blocks")
    }


def read_json(root_fd, relative):
    with parent_fd(root_fd, relative) as (directory, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        with os.fdopen(fd, "r") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise PermissionError("Expected a regular JSON file")
            return json.load(stream)


def sha256_fd(fd):
    digest = hashlib.sha256()
    while chunk := os.read(fd, 8 << 20):
        digest.update(chunk)
    return digest.hexdigest()


def verify_archive(root_fd, relative, allowlist, *, apply):
    with parent_fd(root_fd, relative) as (directory, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        try:
            before = metadata(os.fstat(fd))
            if not stat.S_ISREG(before["mode"]) or before["size"] != allowlist["archive_bytes"]:
                raise PermissionError("Retained compact archive size/type differs")
            if not apply:
                return {"status": "SIZE_CHECKED_HASH_NOT_RUN_DRY_RUN", "stat": before}
            digest = sha256_fd(fd)
            if metadata(os.fstat(fd)) != before or digest != allowlist["archive_sha256"]:
                raise PermissionError("Retained compact archive changed or digest differs")
            if metadata(os.stat(name, dir_fd=directory, follow_symlinks=False)) != before:
                raise PermissionError("Retained archive pathname changed")
            return {
                "status": "SHA256_AND_SIZE_PASS",
                "sha256": digest,
                "stat": before,
                "full_reconstruction": "REUSED_EXISTING_BYTE_EXACT_RECEIPT",
            }
        finally:
            os.close(fd)


def validate_preflight(preflight, required_roots):
    evidence = preflight.get("active_job_evidence", {})
    if (
        preflight.get("status") != "PASS"
        or preflight.get("project_root") != str(ROOT)
        or preflight.get("open_handles") != []
        or evidence.get("status") != "PASS"
        or evidence.get("project_root") != str(ROOT)
        or evidence.get("conflicting_jobs") != []
        or not evidence.get("source")
    ):
        raise PermissionError("Missing actual project-scoped open-handle/job preflight")
    checked = preflight.get("checked_roots", [])
    if not set(required_roots) <= set(checked):
        raise PermissionError("Preflight does not cover every deletion root")
    for path in checked:
        inside_root(path)


def process_member(root_fd, member, *, apply):
    relative = SOURCE + "/" + member["relative"]
    result = dict(
        path=relative,
        reason="byte_exact_compact_archive_duplicate",
        expected_bytes=member["bytes"],
        expected_sha256=member["sha256"],
    )
    try:
        with parent_fd(root_fd, relative) as (directory, name):
            before = metadata(os.stat(name, dir_fd=directory, follow_symlinks=False))
            result["stat"] = before
            if stat.S_ISLNK(before["mode"]):
                return dict(result, status="PRESERVED_SOURCE_SYMLINK")
            if not stat.S_ISREG(before["mode"]):
                raise PermissionError("Nonregular F3 candidate; stop: " + relative)
            if before["size"] != member["bytes"]:
                return dict(result, status="PRESERVED_SIZE_MISMATCH")
            if not apply:
                return dict(result, status="DRY_RUN_CANDIDATE", hash_verified=False)
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            try:
                if metadata(os.fstat(fd)) != before:
                    return dict(result, status="PRESERVED_CHANGED_BEFORE_HASH")
                digest = sha256_fd(fd)
                if metadata(os.fstat(fd)) != before:
                    return dict(result, status="PRESERVED_CHANGED_DURING_HASH")
                result.update(actual_sha256=digest, hash_verified=digest == member["sha256"])
                if digest != member["sha256"]:
                    return dict(result, status="PRESERVED_HASH_MISMATCH")
                if metadata(os.stat(name, dir_fd=directory, follow_symlinks=False)) != before:
                    return dict(result, status="PRESERVED_CHANGED_BEFORE_UNLINK")
                os.unlink(name, dir_fd=directory)
                return dict(
                    result,
                    status="DELETED",
                    logical_bytes_removed=before["size"],
                    allocated_bytes_released=(
                        before["blocks"] * 512 if before["nlink"] == 1 else 0
                    ),
                )
            finally:
                os.close(fd)
    except FileNotFoundError:
        return dict(result, status="PRESERVED_MISSING")


def cache_entries(root_fd):
    """Inspect only this named obsolete cache, without following file symlinks."""

    def walk(directory, prefix):
        for name in sorted(os.listdir(directory)):
            value = os.stat(name, dir_fd=directory, follow_symlinks=False)
            relative = prefix + "/" + name
            if stat.S_ISDIR(value.st_mode):
                child = os.open(name, DIR_FLAGS, dir_fd=directory)
                try:
                    yield from walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode):
                yield relative, metadata(value)
            else:
                raise PermissionError("Nonregular obsolete-cache entry; stop: " + relative)

    with parent_fd(root_fd, OLD_CACHE + "/entry") as (directory, _):
        yield from walk(directory, OLD_CACHE)


def process_cache_entry(root_fd, relative, expected, *, apply):
    if not relative.startswith(OLD_CACHE + "/"):
        raise PermissionError("Cache candidate is outside the sole permitted model cache")
    relative_parts(relative)
    with parent_fd(root_fd, relative) as (directory, name):
        current = metadata(os.stat(name, dir_fd=directory, follow_symlinks=False))
        result = dict(
            path=relative,
            stat=current,
            reason="obsolete_regenerable_model_cache",
            hash_verified=False,
            hash_policy="metadata_only_regenerable_cache",
        )
        if current != expected:
            return dict(result, status="PRESERVED_CHANGED_CACHE_ENTRY")
        if not (stat.S_ISREG(current["mode"]) or stat.S_ISLNK(current["mode"])):
            raise PermissionError("Unexpected cache file type")
        if not apply:
            return dict(result, status="DRY_RUN_CACHE_CANDIDATE")
        os.unlink(name, dir_fd=directory)
        return dict(
            result,
            status="DELETED",
            logical_bytes_removed=current["size"],
            allocated_bytes_released=current["blocks"] * 512 if current["nlink"] == 1 else 0,
        )


def run(args):
    audit = inside_root(args.audit_dir)
    archive = inside_root(args.archive)
    allowlist_path = inside_root(args.allowlist)
    if any(
        p in (SOURCE, OLD_CACHE) or p.startswith(SOURCE + "/") or p.startswith(OLD_CACHE + "/")
        for p in (audit, archive, allowlist_path)
    ):
        raise PermissionError("Receipts, archive and allowlist must be outside deletion roots")
    mode = "APPLY" if args.apply else "DRY_RUN"
    with project_fd() as root_fd:
        allowlist = read_json(root_fd, allowlist_path)
        if (
            allowlist.get("schema") != "SRF3_COMPACT_BACKED_CLEANUP_V1"
            or allowlist.get("source_root") != SOURCE
            or allowlist.get("existing_reconstruction_receipt", {}).get("status")
            != "PASS_ALL_ORIGINAL_MEMBERS_BYTE_EXACT"
        ):
            raise PermissionError("Unregistered source or missing reconstruction receipt")
        if (
            type(allowlist.get("archive_bytes")) is not int
            or allowlist["archive_bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", allowlist.get("archive_sha256", ""))
        ):
            raise ValueError("Invalid retained archive identity")
        members = [validate_member(member) for member in allowlist["members"]]
        if not members or len({m["relative"] for m in members}) != len(members):
            raise ValueError("Empty or duplicate candidate inventory")
        required = [str(ROOT / SOURCE)]
        if args.remove_old_model_cache:
            required.append(str(ROOT / OLD_CACHE))
        if args.apply:
            validate_preflight(read_json(root_fd, audit + "/OPEN_HANDLES.json"), required)
        receipt = dict(
            status=mode + "_STARTED",
            project_root=str(ROOT),
            source_root=SOURCE,
            archive=archive,
            allowed_files=len(members),
            directories_removed=0,
            no_current_model_or_environment_scanned=True,
            archive_check=verify_archive(root_fd, archive, allowlist, apply=args.apply),
        )
        cache_plan = list(cache_entries(root_fd)) if args.remove_old_model_cache else []
        counts = Counter()
        totals = dict(logical_bytes_removed=0, allocated_bytes_released=0)
        with parent_fd(root_fd, audit + f"/{mode}_JOURNAL.jsonl", create=True) as (directory, name):
            fd = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
            )
            with os.fdopen(fd, "w") as journal:

                def record(item):
                    counts[item["status"]] += 1
                    for key in totals:
                        totals[key] += item.get(key, 0)
                    journal.write(json.dumps(item, sort_keys=True) + "\n")
                    if sum(counts.values()) % 500 == 0:
                        journal.flush()
                        os.fsync(journal.fileno())
                        print(
                            json.dumps(dict(processed=sum(counts.values()), **totals)), flush=True
                        )

                try:
                    for member in members:
                        record(process_member(root_fd, member, apply=args.apply))
                    for relative, expected in cache_plan:
                        record(process_cache_entry(root_fd, relative, expected, apply=args.apply))
                    receipt["status"] = mode + "_COMPLETE"
                except BaseException as exc:
                    receipt.update(status=mode + "_STOPPED", error=str(exc))
                    raise
                finally:
                    journal.flush()
                    os.fsync(journal.fileno())
                    receipt.update(counts=dict(counts), **totals)
                    receipt["allocated_bytes_note"] = (
                        "st_blocks*512 only when unlinking the last hardlink; filesystem snapshots "
                        "or compression may prevent equivalent free-space increase"
                    )
                    with parent_fd(root_fd, audit + f"/{mode}_RECEIPT.json") as (outdir, output):
                        outfd = os.open(
                            output,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o600,
                            dir_fd=outdir,
                        )
                        with os.fdopen(outfd, "w") as stream:
                            json.dump(receipt, stream, indent=2)
                            stream.write("\n")
                            stream.flush()
                            os.fsync(stream.fileno())
        print(json.dumps(receipt, indent=2))
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--allowlist", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--remove-old-model-cache", action="store_true")
    run(parser.parse_args())
