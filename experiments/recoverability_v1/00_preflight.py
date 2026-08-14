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

from compbias.recoverability.evidence import verify_protocol_lock
from compbias.recoverability.preflight import load_runtime_spec, run_metadata_preflight


def _pip_check() -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout + completed.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--server-package-lock", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    package_lock = verify_protocol_lock(args.server_package_lock, repository_root=root)
    report = run_metadata_preflight(
        load_runtime_spec(args.runtime),
        repository_root=root,
        version_lookup=importlib.metadata.version,
        pip_check=_pip_check,
        environ=os.environ,
    )
    payload = asdict(report)
    payload.update(
        {
            "artifact_type": "recoverability_v1_metadata_preflight",
            "schema_version": 1,
            "server_package_lock_verified": package_lock.verified,
            "server_package_files": [item.relative_path for item in package_lock.files],
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
