#!/usr/bin/env python3
"""Read-only byte-preservation audit for eight fixed SSVC transfer archives.

No extraction, deletion, rename, source mutation or scheduler call is performed.
Production execution requires a CPU Slurm allocation. Output is a new directory
inside the fixed SSVC root; each archive gets a complete member JSONL ledger.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import stat
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

SSVC_ROOT = Path("/projects/_ssd/varunssd/louis-ssvc")
RELATIVE_ROOT = Path("runs/NEXT_20260909")
ARCHIVE_NAMES = (
    "R1_server_146571_attempt_0",
    "R2_server_146853_attempt_0",
    "R3_cold_server_146949_attempt_0",
    "R3_warm_server_151678_attempt_0",
    "R4_server_147871_attempt_0",
    "R4_server_149841_attempt_0",
    "R4_server_149847_attempt_0",
    "R4_server_150317_attempt_0",
)
CHUNK = 1024 * 1024


def progress(**values):
    print(json.dumps({"time_unix": time.time(), **values}, sort_keys=True), flush=True)


def file_identity(value):
    return {
        name: getattr(value, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_mode",
            "st_nlink",
        )
    }


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def safe_parts(name):
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ValueError("unsafe member name")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("absolute or traversing member path")
    return tuple(part for part in relative.parts if part != ".")


def safe_target(root, parts):
    """Refuse every symlink component; never follow archive links."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("counterpart root missing or symlinked")
    path = root
    for part in parts:
        path /= part
        if path.is_symlink():
            raise ValueError("symlink in counterpart path")
    if path.resolve() != root.resolve() and root.resolve() not in path.resolve().parents:
        raise ValueError("counterpart escapes its protected root")
    return path


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data

    def tell(self):
        return self.count


def compare_regular(source, counterpart, *, expected_size, pulse=None):
    """Hash both streams fully, even on a size mismatch, with bounded memory."""
    archive_hash, target_hash = hashlib.sha256(), hashlib.sha256()
    source_bytes, target_bytes = 0, 0
    target_stream, target_before = None, None
    problems = []
    try:
        descriptor = os.open(counterpart, os.O_RDONLY | os.O_NOFOLLOW)
        target_before = os.fstat(descriptor)
        if not stat.S_ISREG(target_before.st_mode):
            os.close(descriptor)
            raise ValueError("counterpart is not a regular file")
        target_stream = os.fdopen(descriptor, "rb")
    except (OSError, ValueError) as exc:
        target_before = None
        problems.append(f"COUNTERPART_UNAVAILABLE: {type(exc).__name__}: {exc}")
    last_pulse = time.monotonic()
    try:
        while True:
            left = source.read(CHUNK)
            right = target_stream.read(CHUNK) if target_stream else b""
            if not left and not right:
                break
            archive_hash.update(left)
            target_hash.update(right)
            source_bytes += len(left)
            target_bytes += len(right)
            if pulse and time.monotonic() - last_pulse >= 30:
                pulse(source_bytes=source_bytes, counterpart_bytes=target_bytes)
                last_pulse = time.monotonic()
        if source_bytes != expected_size:
            problems.append("SOURCE_SIZE_DIFFERS_FROM_DECLARED_SIZE")
        if target_stream:
            target_after = os.fstat(target_stream.fileno())
            if file_identity(target_after) != file_identity(target_before):
                problems.append("COUNTERPART_CHANGED_DURING_READ")
            if file_identity(Path(counterpart).lstat()) != file_identity(target_before):
                problems.append("COUNTERPART_PATH_CHANGED_DURING_READ")
            if target_bytes != expected_size:
                problems.append("COUNTERPART_SIZE_MISMATCH")
            if archive_hash.digest() != target_hash.digest():
                problems.append("SHA256_MISMATCH")
    finally:
        if target_stream:
            target_stream.close()
    return {
        "status": "PRESERVED" if not problems else "DIFFERENT",
        "source_bytes": source_bytes,
        "source_sha256": archive_hash.hexdigest(),
        "counterpart": str(counterpart),
        "counterpart_bytes": target_bytes,
        "counterpart_sha256": target_hash.hexdigest() if target_before is not None else None,
        "counterpart_identity": file_identity(target_before) if target_before is not None else None,
        "problems": problems,
    }


def overall_status(statuses, errors):
    if errors or any(status not in {"PRESERVED", "DIFFERENT"} for status in statuses):
        return "ERROR"
    if any(status == "DIFFERENT" for status in statuses):
        return "DIFFERENT"
    return "ALL_MEMBERS_PRESERVED"


def audit_archive(archive, extracted, out):
    """Audit one archive against a consistent contents-or-named-root mapping."""
    archive, extracted, out = Path(archive), Path(extracted), Path(out)
    ledger_path = out / (archive.name + ".members.jsonl")
    statuses, identities, errors = [], [], []
    mapping, compressed_hash, compressed_bytes, before = None, None, 0, None
    started = time.monotonic()
    progress(event="archive_started", archive=str(archive), counterpart=str(extracted))
    with ledger_path.open("x") as ledger:
        try:
            if archive.is_symlink():
                raise ValueError("archive itself is a symlink")
            with archive.open("rb") as raw:
                before = file_identity(os.fstat(raw.fileno()))
                if not stat.S_ISREG(before["st_mode"]):
                    raise ValueError("archive is not a regular file")
                hashed = HashedReader(raw)
                try:
                    with gzip.GzipFile(fileobj=hashed, mode="rb") as uncompressed:
                        # 512-byte stream buffering leaves no hidden extra tar
                        # block when the first end marker is encountered.
                        with tarfile.open(fileobj=uncompressed, mode="r|", bufsize=512) as reader:
                            for index, member in enumerate(reader):
                                row = {
                                    "index": index,
                                    "name": member.name,
                                    "type": member.type.decode("ascii", errors="backslashreplace"),
                                    "declared_size": member.size,
                                    "mode": member.mode,
                                    "mtime": member.mtime,
                                    "uid": member.uid,
                                    "gid": member.gid,
                                    "uname": member.uname,
                                    "gname": member.gname,
                                    "linkname": member.linkname,
                                    "pax_headers": member.pax_headers,
                                }
                                try:
                                    parts = safe_parts(member.name)
                                    if mapping is None and parts:
                                        mapping = (
                                            "NAMED_DIRECTORY"
                                            if parts[0] == extracted.name
                                            else "CONTENTS"
                                        )
                                    if mapping == "NAMED_DIRECTORY" and parts:
                                        if parts[0] != extracted.name:
                                            raise ValueError(
                                                "member violates named-directory mapping"
                                            )
                                        parts = parts[1:]
                                    target = safe_target(extracted, parts)
                                    row["counterpart"] = str(target)
                                    if member.issym() or member.islnk():
                                        row.update(
                                            status="UNSUPPORTED_LINK",
                                            problems=["Link requires manual review."],
                                        )
                                    elif member.isdir():
                                        row.update(
                                            status="PRESERVED" if target.is_dir() else "DIFFERENT"
                                        )
                                        if target.is_dir():
                                            identities.append(
                                                (str(target), file_identity(target.lstat()))
                                            )
                                    elif member.isfile() and member.sparse is None:
                                        if not parts:
                                            raise ValueError(
                                                "regular member cannot map to counterpart root"
                                            )
                                        with reader.extractfile(member) as contents:
                                            row.update(
                                                compare_regular(
                                                    contents,
                                                    target,
                                                    expected_size=member.size,
                                                    pulse=lambda member_name=member.name, **v: (
                                                        progress(
                                                            event="member_progress",
                                                            archive=archive.name,
                                                            member=member_name,
                                                            **v,
                                                        )
                                                    ),
                                                )
                                            )
                                        if row["counterpart_identity"] is not None:
                                            identities.append(
                                                (str(target), row["counterpart_identity"])
                                            )
                                    else:
                                        row.update(
                                            status="UNSUPPORTED_MEMBER",
                                            problems=["special/sparse member"],
                                        )
                                except (OSError, ValueError, tarfile.TarError) as exc:
                                    row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
                                statuses.append(row["status"])
                                ledger.write(
                                    json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                )
                                ledger.flush()
                                progress(
                                    event="member_done",
                                    archive=archive.name,
                                    index=index,
                                    name=member.name,
                                    status=row["status"],
                                    compressed_bytes_read=hashed.count,
                                )
                        # Validate gzip CRC and all concatenated gzip members.
                        # Extra nonzero tar data after the end marker blocks deletion.
                        while tail := uncompressed.read(CHUNK):
                            if any(tail):
                                errors.append("TRAILING_NONZERO_DATA_AFTER_TAR_END_MARKER")
                                break
                except (OSError, ValueError, EOFError, tarfile.TarError) as exc:
                    errors.append(f"ARCHIVE_READ_ERROR: {type(exc).__name__}: {exc}")
                finally:
                    # A corrupt archive still gets its entire compressed-file hash.
                    while hashed.read(CHUNK):
                        pass
                    compressed_hash, compressed_bytes = hashed.digest.hexdigest(), hashed.count
                if (
                    file_identity(os.fstat(raw.fileno())) != before
                    or file_identity(archive.lstat()) != before
                ):
                    errors.append("ARCHIVE_CHANGED_DURING_AUDIT")
        except (OSError, ValueError) as exc:
            errors.append(f"ARCHIVE_UNAVAILABLE: {type(exc).__name__}: {exc}")
        for target, identity in identities:
            try:
                if file_identity(Path(target).lstat()) != identity:
                    errors.append("COUNTERPART_CHANGED_BEFORE_AUDIT_COMPLETED: " + target)
            except OSError as exc:
                errors.append(f"COUNTERPART_RECHECK_FAILED: {target}: {exc}")
        ledger.flush()
        os.fsync(ledger.fileno())
    if not statuses:
        errors.append("NO_AUDITABLE_MEMBERS")
    status = overall_status(statuses, errors)
    summary = {
        "archive": str(archive),
        "extracted_counterpart": str(extracted),
        "archive_sha256": compressed_hash,
        "archive_bytes": compressed_bytes,
        "archive_identity_before": before,
        "mapping": mapping,
        "member_count": len(statuses),
        "member_status_counts": {s: statuses.count(s) for s in set(statuses)},
        "member_ledger": str(ledger_path),
        "member_ledger_sha256": hash_path(ledger_path),
        "status": status,
        "eligible_duplicate_transfer_archive": status == "ALL_MEMBERS_PRESERVED",
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
        "scope": (
            "Member bytes and directory presence; original tar headers retained in ledger; "
            "no filesystem metadata equivalence claim."
        ),
        "deletion_performed": False,
        "before_any_later_deletion": (
            "Operator must review ledger and recheck unchanged archive/counterpart identities."
        ),
    }
    write_json(out / (archive.name + ".summary.json"), summary)
    progress(
        event="archive_done", archive=str(archive), status=status, archive_sha256=compressed_hash
    )
    return summary


def hash_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_copying(copying, protected, out):
    copying, protected, out = Path(copying), Path(protected), Path(out)
    ledger_path = out / "inherited_parent_copying.members.jsonl"
    statuses, errors, identities = [], [], []
    with ledger_path.open("x") as ledger:
        if copying.is_symlink() or not copying.is_dir():
            errors.append("COPYING_ROOT_MISSING_OR_SYMLINKED")
        else:
            identities.append((str(copying), file_identity(copying.lstat())))
            for directory, subdirs, filenames in os.walk(copying, followlinks=False):
                subdirs.sort()
                filenames.sort()
                for name in [*subdirs, *filenames]:
                    source = Path(directory) / name
                    relative = source.relative_to(copying)
                    row = {"relative_path": str(relative), "copying_source": str(source)}
                    try:
                        target = safe_target(protected, safe_parts(relative.as_posix()))
                        if source.is_symlink():
                            row.update(status="UNSUPPORTED_LINK", linkname=os.readlink(source))
                        elif source.is_dir():
                            row.update(status="PRESERVED" if target.is_dir() else "DIFFERENT")
                            identities.append((str(source), file_identity(source.lstat())))
                            if target.is_dir():
                                identities.append((str(target), file_identity(target.lstat())))
                        elif source.is_file():
                            before = file_identity(source.lstat())
                            source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
                            with os.fdopen(source_fd, "rb") as stream:
                                if file_identity(os.fstat(stream.fileno())) != before:
                                    raise ValueError("copying source changed before read")
                                row.update(
                                    compare_regular(
                                        stream,
                                        target,
                                        expected_size=before["st_size"],
                                        pulse=lambda relative_name=str(relative), **v: progress(
                                            event="copying_progress",
                                            relative_path=relative_name,
                                            **v,
                                        ),
                                    )
                                )
                            if before != file_identity(source.lstat()):
                                raise ValueError("copying source changed during read")
                            identities.append((str(source), before))
                            if row["counterpart_identity"] is not None:
                                identities.append((str(target), row["counterpart_identity"]))
                        else:
                            row.update(status="UNSUPPORTED_MEMBER")
                    except (OSError, ValueError) as exc:
                        row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
                    statuses.append(row["status"])
                    ledger.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    ledger.flush()
                    progress(
                        event="copying_member_done",
                        relative_path=str(relative),
                        status=row["status"],
                    )
        for path, identity in identities:
            try:
                if file_identity(Path(path).lstat()) != identity:
                    errors.append("FILE_CHANGED_BEFORE_COPY_AUDIT_COMPLETED: " + path)
            except OSError as exc:
                errors.append(f"COPY_RECHECK_FAILED: {path}: {exc}")
        ledger.flush()
        os.fsync(ledger.fileno())
    if not statuses:
        errors.append("NO_AUDITABLE_COPYING_MEMBERS")
    status = overall_status(statuses, errors)
    summary = {
        "copying_root": str(copying),
        "protected_original_root": str(protected),
        "protected_original_policy": "MUST_NOT_DELETE_OR_MODIFY_OLD_35_STEP_TRAJECTORY",
        "status": status,
        "eligible_redundant_copying_candidate": status == "ALL_MEMBERS_PRESERVED",
        "member_count": len(statuses),
        "member_ledger": str(ledger_path),
        "member_ledger_sha256": hash_path(ledger_path),
        "errors": errors,
        "deletion_performed": False,
    }
    write_json(out / "inherited_parent_copying.summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New audit directory inside fixed SSVC root")
    parser.add_argument("--include-copying", action="store_true")
    args = parser.parse_args(argv)
    if (
        sys.platform != "linux"
        or re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")) is None
    ):
        raise ValueError("full archive audit requires a server CPU Slurm allocation")
    if any(os.environ.get(k) != "" for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")):
        raise ValueError("CPU-only audit requires empty CUDA/HIP visibility")
    root = SSVC_ROOT.resolve()
    requested = Path(args.out)
    if not requested.is_absolute() or requested.is_symlink():
        raise ValueError("absolute nonsymlink output directory required")
    out = requested.resolve()
    if root not in out.parents:
        raise ValueError("audit output must be strictly inside the fixed SSVC root")
    base = root / RELATIVE_ROOT
    cursor = root
    for part in RELATIVE_ROOT.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("symlink in fixed audit input path")
    if args.include_copying:
        for relative in (
            "R4_server_149841_attempt_0/R4/.inherited_parent.copying",
            "R4_server_147871_attempt_0/R4",
        ):
            cursor = base
            for part in Path(relative).parts:
                cursor /= part
                if cursor.is_symlink():
                    raise ValueError("symlink in protected/copying audit input path")
    for name in ARCHIVE_NAMES:
        source = base / name
        if out == source or source in out.parents:
            raise ValueError("audit output cannot modify an audited original tree")
    cursor = root
    for part in requested.relative_to(root).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("symlink in requested output path")
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "AUDIT_INTENT.json",
        {
            "script_sha256": hash_path(__file__),
            "fixed_ssvc_root": str(root),
            "archives": [str(base / (name + ".tar.gz")) for name in ARCHIVE_NAMES],
            "include_copying": args.include_copying,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "read_only_originals": True,
            "deletion_performed": False,
        },
    )
    results = [audit_archive(base / (name + ".tar.gz"), base / name, out) for name in ARCHIVE_NAMES]
    copying = None
    if args.include_copying:
        copying = audit_copying(
            base / "R4_server_149841_attempt_0/R4/.inherited_parent.copying",
            base / "R4_server_147871_attempt_0/R4",
            out,
        )
    summary = {
        "schema": "ssvc-redundancy-audit-1",
        "archives": results,
        "copying": copying,
        "candidate_archive_bytes": sum(
            r["archive_bytes"] for r in results if r["eligible_duplicate_transfer_archive"]
        ),
        "deletion_performed": False,
        "read_only_originals": True,
        "scientific_results_validated": False,
    }
    write_json(out / "AUDIT_SUMMARY.json", summary)
    progress(
        event="audit_complete",
        output=str(out),
        candidate_archive_bytes=summary["candidate_archive_bytes"],
    )
    return (
        0
        if all(r["status"] == "ALL_MEMBERS_PRESERVED" for r in results)
        and (copying is None or copying["status"] == "ALL_MEMBERS_PRESERVED")
        else 2
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        progress(
            event="audit_error", error=f"{type(exc).__name__}: {exc}", deletion_performed=False
        )
        raise SystemExit(2) from None
