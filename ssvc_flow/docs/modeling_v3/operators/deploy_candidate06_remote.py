#!/usr/bin/env python3
"""On the server, deploy the uploaded candidate06 archive into a new directory.

Usage: python deploy_candidate06_remote.py ARCHIVE_SHA256
No SSH, data acquisition, experiment execution, or scheduler submission occurs.
Existing candidate05 and all experiment output directories remain read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

BASE = Path("/projects/varunssd/louis-ssvc/modeling_v3_20260915")
OLD = BASE / "candidate05/checkout"
BASE_FLAT_SHA = "3716af3aedb715022574fac1c6e562b2c69e0fba5fe5773df952f96ca734805a"
PARENT = Path("/projects/varunssd/louis-ssvc/modeling_contrast_v2_20260915/parent_M2")
PYTHON = "/projects/varunssd/louis-ssvc/envs/ssvc-py312/bin/python"
SKIP = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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


def regular_path(root, relative):
    safe_relative(relative)
    path = root / relative
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError("symlink in code destination: " + relative)
        cursor = cursor.parent
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("path escapes checkout")
    return path


def write_new(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_sha256")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.archive_sha256):
        raise ValueError("explicit archive SHA256 required")
    archive = BASE / "candidate06-code-delta.tar.gz"
    if archive.is_symlink() or sha(archive) != args.archive_sha256:
        raise ValueError("archive checksum mismatch or symlink")
    old_manifest = OLD / "CODE_SNAPSHOT_MANIFEST.json"
    if old_manifest.is_symlink() or sha(old_manifest) != BASE_FLAT_SHA:
        raise ValueError("candidate05 flat code inventory changed")
    old_files = json.loads(old_manifest.read_bytes())
    for relative, digest in old_files.items():
        path = regular_path(OLD, relative)
        if not path.is_file() or sha(path) != digest:
            raise ValueError("candidate05 code original changed: " + relative)

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError("duplicate archive member")
        for member in members:
            safe_relative(member.name)
            if not member.isfile():
                raise ValueError("archive links/directories are forbidden")
        manifest_bytes = tar.extractfile("manifest.json").read()
        manifest = json.loads(manifest_bytes)
        if (
            manifest.get("schema") != "ssvc-v3-candidate06-code-delta-1"
            or manifest.get("base_checkout") != str(OLD)
            or manifest.get("base_manifest_sha256") != BASE_FLAT_SHA
            or manifest.get("added_data_files") != []
            or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("upstream_commit", ""))
        ):
            raise ValueError("wrong candidate06 base, scope, or committed identity")
        files, sizes, modes = manifest["files"], manifest["file_sizes"], manifest["git_modes"]
        delta, removed = manifest["delta_files"], manifest["removed_files"]
        if not files or set(files) != set(sizes) or set(files) != set(modes):
            raise ValueError("incomplete full code inventory")
        if delta != {p: h for p, h in files.items() if old_files.get(p) != h}:
            raise ValueError("delta does not cover every changed file")
        if removed != sorted(set(old_files) - set(files)):
            raise ValueError("removed-file inventory mismatch")
        expected_names = {"manifest.json", *("files/" + p for p in delta)}
        if {m.name for m in members} != expected_names:
            raise ValueError("archive contains missing or unexpected payloads")
        payloads = {}
        for relative, digest in files.items():
            safe_relative(relative)
            if (
                not re.fullmatch(r"[0-9a-f]{64}", digest)
                or modes[relative] not in {"100644", "100755"}
                or type(sizes[relative]) is not int
                or sizes[relative] < 0
            ):
                raise ValueError("invalid code hash, size, or mode")
            if relative in delta:
                payload = tar.extractfile("files/" + relative).read()
                if len(payload) != sizes[relative] or hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError("delta payload mismatch: " + relative)
                payloads[relative] = payload

    view_path = BASE / "candidate05/TEST_DATA_VIEW.json"
    view_bytes = view_path.read_bytes()
    view = json.loads(view_bytes)
    if view.get("sealed_confirm_copied_or_opened") is not False:
        raise ValueError("candidate05 does not declare the sealed-confirm exclusion")
    parent_binding = view["external_readonly_legacy_parent"]
    if Path(parent_binding["path"]).resolve() != PARENT.resolve():
        raise ValueError("unexpected external legacy parent")
    if sha(PARENT / "manifest.json") != parent_binding["manifest_sha256"]:
        raise ValueError("read-only M2 parent manifest changed")
    packet_relative = "ssvc_flow/runs/modeling_contrast_v2/N1/packets/seed101_X_VALID/a1/O_IND_n64"
    packet_names = {"packets.json", "packet_metadata.json.gz", "packet_arrays.npz"}
    packet_entries = view["additional_packet_originals"]
    if (
        len(packet_entries) != 3
        or {Path(p["destination"]).name for p in packet_entries} != packet_names
    ):
        raise ValueError("candidate05 three-file N1 view is missing")
    for entry in packet_entries:
        original = regular_path(OLD, packet_relative + "/" + Path(entry["destination"]).name)
        if sha(original) != entry["sha256"] or original.stat().st_size != entry["bytes"]:
            raise ValueError("candidate05 retained N1 bytes changed")

    # Inventory/copy exactly the existing view. Never follow a directory symlink.
    inherited, symlinks = {}, {}
    for directory, dirnames, filenames in os.walk(OLD, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
        for name in [*dirnames, *sorted(filenames)]:
            if name in SKIP:
                continue
            path = Path(directory) / name
            relative = path.relative_to(OLD).as_posix()
            safe_relative(relative)
            if relative.startswith(("ssvc_flow/data/", "ssvc_flow/runs/")) and any(
                "confirm" in p.lower() for p in Path(relative).parts
            ):
                raise ValueError("sealed-confirm data path is outside this deployment scope")
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                # Preserve an existing link verbatim, without reading/copying its
                # target. All code access separately rejects symlink ancestors.
                target = os.readlink(path)
                if any("confirm" in p.lower() for p in Path(target).parts):
                    raise ValueError("sealed-confirm link is outside deployment scope")
                symlinks[relative] = target
            elif stat.S_ISREG(mode):
                inherited[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
            elif not stat.S_ISDIR(mode):
                raise ValueError("unsupported inherited special file: " + relative)

    out = BASE / "candidate06"
    out.mkdir(exist_ok=False)
    checkout = out / "checkout"
    shutil.copytree(OLD, checkout, symlinks=True, ignore=shutil.ignore_patterns(*sorted(SKIP)))
    if (checkout / ".git").exists() or (checkout / ".git").is_symlink():
        raise ValueError("old Git metadata must not be copied")
    for relative, metadata in inherited.items():
        copied = checkout / relative
        if sha(copied) != metadata["sha256"] or copied.stat().st_size != metadata["bytes"]:
            raise ValueError("inherited file copy mismatch: " + relative)
    for relative, target in symlinks.items():
        if not (checkout / relative).is_symlink() or os.readlink(checkout / relative) != target:
            raise ValueError("inherited symlink changed")
    for relative in removed:
        regular_path(checkout, relative).unlink()
    for relative, payload in payloads.items():
        dest = regular_path(checkout, relative)
        if relative not in old_files and dest.exists():
            raise FileExistsError("new code would overwrite an inherited data view: " + relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb" if relative in old_files else "xb") as stream:
            stream.write(payload)
    flat = checkout / "CODE_SNAPSHOT_MANIFEST.json"
    flat.write_bytes(json_bytes(files))
    for relative, digest in files.items():
        path = regular_path(checkout, relative)
        path.chmod(int(modes[relative], 8) & 0o777)
        if not path.is_file() or sha(path) != digest or path.stat().st_size != sizes[relative]:
            raise ValueError("full deployed code verification failed: " + relative)
    for relative, metadata in inherited.items():
        if sha(OLD / relative) != metadata["sha256"]:
            raise ValueError("candidate05 changed during deployment: " + relative)
    if view_path.read_bytes() != view_bytes:
        raise ValueError("candidate05 data-view receipt changed during deployment")
    write_new(out / "CANDIDATE05_TEST_DATA_VIEW.json", view_bytes)
    write_new(
        out / "INHERITED_FILE_VERIFICATION.json",
        json_bytes(
            {
                "source_checkout": str(OLD),
                "files": inherited,
                "symlinks": symlinks,
                "ignored_metadata_and_caches": sorted(SKIP),
                "copy_bytes_verified": True,
            }
        ),
    )
    write_new(
        out / "TEST_DATA_VIEW.json",
        json_bytes(
            {
                "inherited_view": view,
                "inherited_view_sha256": hashlib.sha256(view_bytes).hexdigest(),
                "inherited_view_path": str(out / "CANDIDATE05_TEST_DATA_VIEW.json"),
                "rebase_from_checkout": str(OLD),
                "rebase_to_checkout": str(checkout),
                "additional_packet_originals": [],
                "new_data_files": [],
                "inherited_n1_packet_files_verified": 3,
                "external_readonly_legacy_parent": parent_binding,
                "sealed_confirm_copied_or_opened": False,
            }
        ),
    )
    write_new(out / "PACKAGE_MANIFEST.json", manifest_bytes)
    process = subprocess.run(
        [
            PYTHON,
            str(checkout / "ssvc_flow/scripts/initialize_modeling_v3_snapshot_git.py"),
            "--checkout",
            str(checkout),
            "--manifest-sha256",
            sha(flat),
            "--receipt",
            str(out / "SNAPSHOT_GIT_RECEIPT.json"),
        ],
        capture_output=True,
    )
    write_new(out / "SNAPSHOT_GIT_STDOUT.txt", process.stdout)
    write_new(out / "SNAPSHOT_GIT_STDERR.txt", process.stderr)
    process.check_returncode()
    receipt = {
        "schema": "ssvc-v3-candidate06-deployment-receipt-1",
        "archive_sha256": args.archive_sha256,
        "upstream_commit": manifest["upstream_commit"],
        "base_manifest_sha256": BASE_FLAT_SHA,
        "full_code_files": len(files),
        "delta_files": len(delta),
        "removed_files": len(removed),
        "snapshot_manifest_sha256": sha(flat),
        "code_hashes_verified": True,
        "inherited_regular_files_verified": len(inherited),
        "inherited_symlinks_verified": len(symlinks),
        "inherited_n1_packet_files_verified": 3,
        "new_data_files": 0,
        "old_git_copied": False,
        "sealed_confirm_copied_or_opened": False,
        "snapshot_git": json.loads((out / "SNAPSHOT_GIT_RECEIPT.json").read_bytes()),
        "jobs_submitted": False,
    }
    for name in ("DEPLOYMENT_RECEIPT.json", "DEPLOYMENT_STDOUT.json"):
        write_new(out / name, json_bytes(receipt))
    print(json_bytes(receipt).decode(), end="")


if __name__ == "__main__":
    main()
