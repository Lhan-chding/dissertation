"""Delete old, reproducible test data after inventory and live-job checks.

Research checkpoints, raw research answers, test logs and test failure summaries
are outside the deletion allowlist. Plan and execution are separate commands.
"""

import argparse
import hashlib
import json
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/projects/varunssd/louis-ssvc").resolve()
RELATIVE_ROOTS = (
    "modeling_v4_20260916/validation/legacy_isolated/legacy_fixtures",
    "modeling_v4_20260916/validation/legacy_isolated/r1_fixtures",
    "modeling_v4_20260916/validation/cbc114a/tmp/157046/pytest-of-varun024",
)
SUFFIXES = {
    ".pt",
    ".pth",
    ".npz",
    ".npy",
    ".safetensors",
    ".json",
    ".jsonl",
    ".csv",
    ".parquet",
    ".png",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(path):
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise ValueError("Expected one regular unshared file: " + str(path))
    return {
        "bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "inode": st.st_ino,
        "allocated_bytes": st.st_blocks * 512,
    }


def active_jobs():
    ids = subprocess.check_output(["squeue", "-h", "-u", "varun024", "-o", "%i"], text=True)
    result = []
    for job in ids.split():
        detail = subprocess.check_output(["scontrol", "show", "job", job], text=True)
        if any(str(ROOT / rel) in detail for rel in RELATIVE_ROOTS):
            raise RuntimeError("Active job references a cleanup root")
        result.append({"job_id": job, "detail": detail})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["plan", "execute"])
    p.add_argument("--plan", required=True)
    args = p.parse_args()
    plan_path = Path(args.plan)
    roots = [(ROOT / rel).resolve() for rel in RELATIVE_ROOTS]
    if any(not path.is_relative_to(ROOT) for path in roots):
        raise ValueError("Escaping cleanup root")
    jobs = active_jobs()
    if args.action == "plan":
        if plan_path.exists():
            raise FileExistsError(plan_path)
        files, skipped = [], []
        for root in roots:
            for current, dirs, names in os.walk(root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not (Path(current) / d).is_symlink())
                for name in sorted(names):
                    path = Path(current) / name
                    if path.suffix not in SUFFIXES or path.is_symlink():
                        continue
                    if path.lstat().st_nlink != 1:
                        skipped.append({"path": str(path), "reason": "shared_inode_preserved"})
                        continue
                    sig = signature(path)
                    files.append({"path": str(path), **sig, "sha256": digest(path)})
        plan = {
            "schema": "vdt-disposable-test-tensors-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "roots": list(map(str, roots)),
            "files": files,
            "skipped": skipped,
            "active_jobs": jobs,
            "logical_bytes": sum(row["bytes"] for row in files),
            "allocated_bytes": sum(row["allocated_bytes"] for row in files),
            "preserved": (
                "All raw research data, real checkpoints, validation summaries/logs/XML "
                "and source; shared fixture inodes"
            ),
        }
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps({"files": len(files), "bytes": plan["logical_bytes"]}))
        return
    plan = json.loads(plan_path.read_text())
    if plan["roots"] != list(map(str, roots)):
        raise ValueError("Allowlist differs")
    # Validate the whole inventory before the first unlink.
    for row in plan["files"]:
        path = Path(row["path"])
        if not any(path.is_relative_to(root) for root in roots) or path.suffix not in SUFFIXES:
            raise ValueError("Non-allowlisted path")
        if path.resolve() != path or signature(path) != {k: row[k] for k in signature(path)}:
            raise ValueError("File changed since inventory")
        if digest(path) != row["sha256"]:
            raise ValueError("Content changed since inventory")
    receipt = plan_path.with_name("CLEANUP_RECEIPT.json")
    if receipt.exists():
        raise FileExistsError(receipt)
    journal = plan_path.with_name("CLEANUP_DELETED.jsonl")
    deleted = 0
    with journal.open("x") as stream:
        for row in plan["files"]:
            Path(row["path"]).unlink()
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            deleted += 1
    result = {
        "status": "CLEANUP_COMPLETE",
        "plan_sha256": digest(plan_path),
        "deleted_files": deleted,
        "deleted_logical_bytes": plan["logical_bytes"],
        "deleted_allocated_bytes_inventory": plan["allocated_bytes"],
        "remaining_targets": sum(Path(row["path"]).exists() for row in plan["files"]),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "preserved": plan["preserved"],
        "active_jobs_at_execution": jobs,
    }
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
