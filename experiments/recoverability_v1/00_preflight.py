#!/usr/bin/env python3
"""Run metadata-only Recoverability v1 server checks before model loading."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from compbias.recoverability.evidence import verify_server_package_lock
from compbias.recoverability.preflight import load_runtime_spec, run_metadata_preflight


def _metadata_subprocess_environment() -> dict[str, str]:
    """Return the installed-package environment without repository source metadata."""

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def _pip_check() -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        env=_metadata_subprocess_environment(),
    )
    return completed.returncode, completed.stdout + completed.stderr


def _pip_inventory() -> dict[str, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
        env=_metadata_subprocess_environment(),
    )
    rows = json.loads(completed.stdout)
    return {row["name"]: row["version"] for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--server-package-lock", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite preflight report: {args.output}")
    if args.output.parent.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("preflight output parent must be an existing regular directory")
    package_lock = verify_server_package_lock(args.server_package_lock, repository_root=root)
    report = run_metadata_preflight(
        load_runtime_spec(args.runtime),
        repository_root=root,
        version_lookup=importlib.metadata.version,
        inventory_lookup=_pip_inventory,
        pip_check=_pip_check,
        environ=os.environ,
    )
    payload = asdict(report)
    payload.update(
        {
            "artifact_type": "recoverability_v1_metadata_preflight",
            "schema_version": 1,
            "server_package_lock_verified": package_lock.verified,
            "server_package_lock_sha256": __import__("hashlib")
            .sha256(args.server_package_lock.read_bytes())
            .hexdigest(),
            "server_package_files": [item.relative_path for item in package_lock.files],
        }
    )
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
