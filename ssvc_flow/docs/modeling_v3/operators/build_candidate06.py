#!/usr/bin/env python3
"""Prepare candidate06 only after the operator commits and freezes local code.

Usage from any directory: python build_candidate06.py --expected-commit FULL_SHA
This creates a new local package. It does not upload, deploy, or submit jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

BASE_CHECKOUT = "/projects/varunssd/louis-ssvc/modeling_v3_20260915/candidate05/checkout"
BASE_FLAT_SHA = "3716af3aedb715022574fac1c6e562b2c69e0fba5fe5773df952f96ca734805a"
CODE_PREFIXES = (
    "src/",
    "ssvc_flow/src/",
    "ssvc_flow/tests/",
    "ssvc_flow/scripts/",
    "ssvc_flow/configs/",
    "ssvc_flow/reference/",
)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("invalid relative path")
    path = PurePosixPath(value)
    if (
        value == "."
        or path.is_absolute()
        or str(path) != value
        or any(p in {"..", ".git"} for p in path.parts)
    ):
        raise ValueError("unsafe relative path: " + value)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        raise ValueError("an explicit full committed Git SHA is required")
    flow = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "configs/modeling_v3/protocol.json").is_file()
        and (parent / "src/modeling_v3").is_dir()
    )
    wt = flow.parent
    folder = flow / "runs/modeling_v3_dev/deployment/candidate06"

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=wt)

    def verify_frozen():
        if git("rev-parse", "HEAD").decode().strip() != args.expected_commit:
            raise ValueError("HEAD differs from the operator's frozen commit")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("commit all tracked changes before packaging")
        untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
        pending = [p for p in untracked if p.startswith(CODE_PREFIXES)]
        if pending:
            raise ValueError("uncommitted source files are not packageable: " + repr(pending))

    verify_frozen()
    base_path = flow / "runs/modeling_v3_dev/deployment/candidate05/manifest.json"
    base_bytes = base_path.read_bytes()
    base = json.loads(base_bytes)["files"]
    if not base or sha_bytes(json_bytes(base)) != BASE_FLAT_SHA:
        raise ValueError("candidate05 local flat inventory differs from the frozen base")
    for relative, digest in base.items():
        safe_relative(relative)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid base file digest")

    tree = {}
    for entry in git("ls-tree", "-r", "-z", args.expected_commit).split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        tree[raw_path.decode()] = (mode, kind, oid)
    paths = (set(base) & set(tree)) | {p for p in tree if p.startswith(CODE_PREFIXES)}
    files, sizes, modes, payloads = {}, {}, {}, {}
    for relative in sorted(paths):
        safe_relative(relative)
        mode, kind, oid = tree[relative]
        path = wt / relative
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError("only committed regular files are supported: " + relative)
        if any(p.is_symlink() for p in [path, *path.parents] if p != wt.parent):
            raise ValueError("symlink in package source: " + relative)
        if not path.resolve().is_relative_to(wt.resolve()) or not path.is_file():
            raise ValueError("missing or unsafe working file: " + relative)
        data = path.read_bytes()
        committed = git("cat-file", "blob", oid)
        if data != committed:
            raise ValueError("working bytes differ from frozen Git object: " + relative)
        files[relative], sizes[relative], modes[relative] = sha_bytes(data), len(data), mode
        if base.get(relative) != files[relative]:
            payloads[relative] = data

    manifest = {
        "schema": "ssvc-v3-candidate06-code-delta-1",
        "base_checkout": BASE_CHECKOUT,
        "base_manifest_sha256": BASE_FLAT_SHA,
        "base_local_manifest_sha256": sha_bytes(base_bytes),
        "upstream_commit": args.expected_commit,
        "scope": (
            "candidate05 inventoried files plus committed relevant "
            "source/tests/scripts/config/reference"
        ),
        "added_data_files": [],
        "delta_files": {p: files[p] for p in sorted(payloads)},
        "removed_files": sorted(set(base) - paths),
        "files": files,
        "file_sizes": sizes,
        "git_modes": modes,
    }
    manifest_bytes = json_bytes(manifest)
    verify_frozen()
    folder.mkdir(parents=True, exist_ok=False)
    manifest_path = folder / "manifest.json"
    with manifest_path.open("xb") as stream:
        stream.write(manifest_bytes)
    archive = folder / "candidate06-code-delta.tar.gz"
    with archive.open("xb") as output, tarfile.open(fileobj=output, mode="w:gz") as tar:
        for name, data in [
            ("manifest.json", manifest_bytes),
            *[("files/" + p, payloads[p]) for p in sorted(payloads)],
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644 if name == "manifest.json" else int(modes[name[6:]], 8) & 0o777
            tar.addfile(member, io.BytesIO(data))
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        expected_names = {"manifest.json", *("files/" + p for p in payloads)}
        if len(members) != len(expected_names) or {m.name for m in members} != expected_names:
            raise ValueError("archive inventory has omissions, duplicates, or extras")
        for member in members:
            safe_relative(member.name)
            if not member.isfile():
                raise ValueError("non-regular archive entry")
            data = tar.extractfile(member).read()
            expected = (
                manifest_bytes if member.name == "manifest.json" else payloads[member.name[6:]]
            )
            if data != expected:
                raise ValueError("archive bytes differ from verified input")
    verify_frozen()
    for relative, digest in files.items():
        if sha_bytes((wt / relative).read_bytes()) != digest:
            raise ValueError("source changed during package construction: " + relative)
    receipt = {
        "schema": "ssvc-v3-candidate06-package-receipt-1",
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": sha_bytes(archive.read_bytes()),
        "manifest_sha256": sha_bytes(manifest_bytes),
        "snapshot_manifest_sha256": sha_bytes(json_bytes(files)),
        "base_manifest_sha256": BASE_FLAT_SHA,
        "delta_file_count": len(payloads),
        "full_file_count": len(files),
        "removed_file_count": len(manifest["removed_files"]),
        "upstream_commit": args.expected_commit,
        "working_bytes_equal_committed_git_blobs": True,
        "path_and_payload_checks": "PASS",
        "uploaded": False,
    }
    for name in ("package_receipt.json", "BUILD_STDOUT.json"):
        with (folder / name).open("xb") as stream:
            stream.write(json_bytes(receipt))
    print(json_bytes(receipt).decode(), end="")


if __name__ == "__main__":
    main()
