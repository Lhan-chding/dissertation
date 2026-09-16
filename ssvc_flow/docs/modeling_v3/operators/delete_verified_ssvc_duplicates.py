#!/usr/bin/env python3
"""Delete explicitly selected byte-verified duplicates inside the fixed SSVC root."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path("/projects/_ssd/varunssd/louis-ssvc")
NEXT = ROOT / "runs/NEXT_20260909"
AUDIT = ROOT / "storage_audits/next_20260909_01"
OUT = ROOT / "storage_audits/duplicate_cleanup_20260916_01"
NAMES = (
    "R1_server_146571_attempt_0",
    "R2_server_146853_attempt_0",
    "R3_cold_server_146949_attempt_0",
    "R3_warm_server_151678_attempt_0",
    "R4_server_147871_attempt_0",
    "R4_server_149841_attempt_0",
    "R4_server_149847_attempt_0",
    "R4_server_150317_attempt_0",
)
COPYING = NEXT / "R4_server_149841_attempt_0/R4/.inherited_parent.copying"
PROTECTED = NEXT / "R4_server_147871_attempt_0/R4"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def checked(path):
    path = Path(path)
    if not path.is_relative_to(ROOT) or ".." in path.parts:
        raise ValueError("outside fixed SSVC root")
    cursor = ROOT
    for part in path.relative_to(ROOT).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("symlink in scoped path: " + str(cursor))
    if path.resolve() != path:
        raise ValueError("noncanonical path")
    return path


def identity(path):
    info = checked(path).lstat()
    return {
        name: getattr(info, name)
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


def digest(path):
    path = checked(path)
    before = identity(path)
    if not stat.S_ISREG(before["st_mode"]):
        raise ValueError("regular file required")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    if identity(path) != before:
        raise ValueError("file changed during hashing")
    return value.hexdigest()


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def quota():
    result = {}
    for key in ("ceph.quota.max_bytes", "ceph.dir.rbytes"):
        try:
            result[key] = int(os.getxattr(ROOT.parent, key))
        except OSError as error:
            result[key] = {"error": str(error)}
    return result


def read_ledger(summary):
    path = checked(Path(summary["member_ledger"]))
    if path.parent != AUDIT or digest(path) != summary["member_ledger_sha256"]:
        raise ValueError("ledger path/hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != summary["member_count"] or any(r["status"] != "PRESERVED" for r in rows):
        raise ValueError("ledger not fully preserved")
    return rows


def verify_retained_files(files):
    for path, expected in files.items():
        if identity(Path(path)) != expected:
            raise ValueError("protected original identity changed before deletion: " + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-summary-sha256", required=True)
    parser.add_argument("--archives", nargs="*", choices=NAMES, default=[])
    parser.add_argument("--delete-copying", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux" or not re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")):
        raise ValueError("server Slurm CPU allocation required")
    if any(os.environ.get(key) for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")):
        raise ValueError("GPU allocation refused")
    if any(os.environ.get(key) != "" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")):
        raise ValueError("empty CUDA and HIP visibility required")
    if len(set(args.archives)) != len(args.archives) or not (args.archives or args.delete_copying):
        raise ValueError("nonempty unique explicit selection required")
    summary_path = checked(AUDIT / "AUDIT_SUMMARY.json")
    if digest(summary_path) != args.audit_summary_sha256:
        raise ValueError("reviewed audit summary hash mismatch")
    summary = json.loads(summary_path.read_text())
    if summary["schema"] != "ssvc-redundancy-audit-1" or summary["deletion_performed"]:
        raise ValueError("unexpected audit")
    index = {Path(item["archive"]).name: item for item in summary["archives"]}
    protected_files = {}
    archives = []
    for name in args.archives:
        path = checked(NEXT / (name + ".tar.gz"))
        item = index[path.name]
        if (
            item["status"] != "ALL_MEMBERS_PRESERVED"
            or item["archive"] != str(path)
            or item["extracted_counterpart"] != str(NEXT / name)
            or item["errors"]
        ):
            raise ValueError("unverified archive selected")
        if identity(path) != item["archive_identity_before"]:
            raise ValueError("archive changed since audit")
        for row in read_ledger(item):
            counterpart = checked(Path(row["counterpart"]))
            if not counterpart.is_relative_to(NEXT / name):
                raise ValueError("counterpart escapes retained tree")
            if row["type"] == "5":
                if not counterpart.is_dir():
                    raise ValueError("retained directory missing")
            elif row["type"] in ("0", "\x00"):
                if identity(counterpart) != row["counterpart_identity"]:
                    raise ValueError("retained original changed since audit")
                protected_files[str(counterpart)] = row["counterpart_identity"]
            else:
                raise ValueError("unsupported member type")
        archives.append(
            {
                "path": str(path),
                "bytes": item["archive_bytes"],
                "sha256": item["archive_sha256"],
                "identity": item["archive_identity_before"],
                "preserved_root": str(NEXT / name),
                "ledger_sha256": item["member_ledger_sha256"],
            }
        )
    copying_files = {}
    copying_entries = set()
    if args.delete_copying:
        item = summary["copying"]
        if (
            item["status"] != "ALL_MEMBERS_PRESERVED"
            or item["errors"]
            or item["copying_root"] != str(COPYING)
            or item["protected_original_root"] != str(PROTECTED)
        ):
            raise ValueError("copying residue not fully verified")
        if not shutil.rmtree.avoids_symlink_attacks:
            raise ValueError("symlink-safe rmtree unavailable")
        for row in read_ledger(item):
            source = checked(Path(row["copying_source"]))
            relative = source.relative_to(COPYING)
            if str(relative) != row["relative_path"] or source == COPYING:
                raise ValueError("unexpected copying member")
            copying_entries.add(str(relative))
            counterpart = checked(PROTECTED / relative)
            if source.is_dir():
                if not counterpart.is_dir():
                    raise ValueError("protected directory missing")
            elif source.is_file():
                if digest(source) != row["source_sha256"]:
                    raise ValueError("copying bytes changed since audit")
                if identity(counterpart) != row["counterpart_identity"]:
                    raise ValueError("protected original changed since audit")
                copying_files[str(source)] = identity(source)
                protected_files[str(counterpart)] = row["counterpart_identity"]
            else:
                raise ValueError("unsupported copying entry")
        current = {str(p.relative_to(COPYING)) for p in checked(COPYING).rglob("*")}
        if current != copying_entries:
            raise ValueError("copying inventory changed")
    # Archive 149841 was verified against its copying residue, whose real originals
    # are independently verified above. Remove those intermediate paths only after
    # deleting their archive; the independent original identities stay protected.
    final_protected = {
        p: info
        for p, info in protected_files.items()
        if not (args.delete_copying and Path(p).is_relative_to(COPYING))
    }
    checked(OUT).mkdir(exist_ok=False)
    plan = {
        "utc": utc(),
        "scope": str(ROOT),
        "job_id": os.environ["SLURM_JOB_ID"],
        "script_sha256": digest(Path(__file__)),
        "audit_summary_sha256": args.audit_summary_sha256,
        "archives": archives,
        "delete_copying": args.delete_copying,
        "copying_root": str(COPYING) if args.delete_copying else None,
        "copying_files": copying_files,
        "protected_files": final_protected,
        "bytes_planned": sum(a["bytes"] for a in archives)
        + sum(v["st_size"] for v in copying_files.values()),
        "quota_before": quota(),
    }
    write_new(OUT / "DELETE_PLAN.json", plan)
    deleted = []
    try:
        with (OUT / "DELETE_LEDGER.jsonl").open("x") as ledger:
            for item in archives:
                verify_retained_files(protected_files)
                path = Path(item["path"])
                if identity(path) != item["identity"]:
                    raise ValueError("archive changed before unlink")
                path.unlink()
                row = {
                    "utc": utc(),
                    "deleted": str(path),
                    "bytes": item["bytes"],
                    "absent": not path.exists(),
                }
                deleted.append(row)
                ledger.write(json.dumps(row) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
            if args.delete_copying:
                verify_retained_files(final_protected)
                for source, expected in copying_files.items():
                    if identity(Path(source)) != expected:
                        raise ValueError("copying residue changed before deletion")
                if {
                    str(p.relative_to(COPYING)) for p in checked(COPYING).rglob("*")
                } != copying_entries:
                    raise ValueError("copying inventory changed before deletion")
                shutil.rmtree(COPYING)
                row = {
                    "utc": utc(),
                    "deleted": str(COPYING),
                    "bytes": sum(v["st_size"] for v in copying_files.values()),
                    "absent": not COPYING.exists(),
                }
                deleted.append(row)
                ledger.write(json.dumps(row) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
        for path, expected in final_protected.items():
            if identity(Path(path)) != expected:
                raise ValueError("protected original identity changed during deletion")
        write_new(
            OUT / "DELETE_RECEIPT.json",
            {
                "utc": utc(),
                "status": "PASS",
                "deleted": deleted,
                "deleted_bytes": sum(v["bytes"] for v in deleted),
                "protected_regular_files_verified_unchanged": len(final_protected),
                "all_targets_absent": all(v["absent"] for v in deleted),
                "quota_after": quota(),
                "scientific_originals_deleted": False,
                "outside_ssvc_modified": False,
            },
        )
        print((OUT / "DELETE_RECEIPT.json").read_text(), flush=True)
    except BaseException as error:
        write_new(
            OUT / "DELETE_FAILED.json", {"utc": utc(), "deleted": deleted, "error": str(error)}
        )
        raise


if __name__ == "__main__":
    main()
