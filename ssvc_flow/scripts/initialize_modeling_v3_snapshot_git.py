#!/usr/bin/env python3
"""Give a verified server snapshot real Git objects for legacy provenance tests.

This creates a new local snapshot commit, not an upstream history or revision.
Only the explicitly inventoried code files are staged. Data views stay untracked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def initialize_snapshot(checkout, expected_manifest_sha256, receipt_path):
    root, receipt_path = Path(checkout).resolve(), Path(receipt_path).resolve()
    manifest_path = root / "CODE_SNAPSHOT_MANIFEST.json"
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError("snapshot inventory hash mismatch")
    if (root / ".git").exists() or receipt_path.exists():
        raise FileExistsError("new isolated snapshot Git metadata and receipt required")
    manifest = json.loads(manifest_bytes)
    files = manifest.get("files", manifest)
    if not files or not isinstance(files, dict):
        raise ValueError("explicit nonempty file hash inventory required")
    for relative, digest in files.items():
        path = root / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or ".git" in Path(relative).parts
            or root not in path.resolve().parents
            or path.is_symlink()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("snapshot original changed or unsafe path: " + relative)

    def git(*args):
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", *args],
            cwd=root,
            capture_output=True,
            check=True,
        )
        return result.stdout

    git("init", "--template=")
    git("add", "--force", "--", *sorted(files), "CODE_SNAPSHOT_MANIFEST.json")
    git(
        "-c",
        "user.name=Codex snapshot builder",
        "-c",
        "user.email=codex@local.invalid",
        "commit",
        "--no-gpg-sign",
        "-m",
        "build: record verified SSVC server code snapshot",
    )
    commit = git("rev-parse", "HEAD").decode().strip()
    for relative, digest in files.items():
        if hashlib.sha256(git("show", f"{commit}:{relative}")).hexdigest() != digest:
            raise ValueError("Git snapshot content differs from code inventory")
    receipt = {
        "schema": "ssvc-v3-server-snapshot-git-1",
        "checkout": str(root),
        "snapshot_manifest_sha256": expected_manifest_sha256,
        "snapshot_git_commit": commit,
        "commit_scope": "NEW_LOCAL_SNAPSHOT_COMMIT; NOT_UPSTREAM_HISTORY",
        "verified_file_count": len(files),
        "data_views_staged": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    print(json.dumps(initialize_snapshot(args.checkout, args.manifest_sha256, args.receipt)))


if __name__ == "__main__":
    main()
