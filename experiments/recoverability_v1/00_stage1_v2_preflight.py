#!/usr/bin/env python3
"""Run metadata-only checks for the Stage-1 v2 development probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

_BOOTSTRAP_SERVER_PATHS = frozenset(
    {
        "configs/data/cva_chart_pilot_v0_3.yaml",
        "configs/paths.example.yaml",
        "configs/recoverability/bridge_v1_failure.yaml",
        "configs/recoverability/power_plan_v1.json",
        "configs/recoverability/recoverability_v1.yaml",
        "configs/recoverability/server_runtime_v1.yaml",
        "configs/recoverability/stage1_v2_probe.yaml",
        "configs/recoverability/v0_3_negative_pilot.yaml",
        "experiments/recoverability_v1/00_preflight.py",
        "experiments/recoverability_v1/00_stage1_v2_preflight.py",
        "experiments/recoverability_v1/02_capture_v03_evidence.py",
        "experiments/recoverability_v1/03_bridge.py",
        "experiments/recoverability_v1/04_stage1_v2_probe.py",
        "requirements-gpu.lock.txt",
        "src/compbias/gpu_pilot/chart_data.py",
        "src/compbias/gpu_pilot/collection.py",
        "src/compbias/gpu_pilot/config.py",
        "src/compbias/gpu_pilot/execution_gate.py",
        "src/compbias/gpu_pilot/preflight.py",
        "src/compbias/gpu_pilot/qwen_smoke.py",
        "src/compbias/gpu_pilot/safe_io.py",
        "src/compbias/gpu_pilot/structured_generation.py",
        "src/compbias/gpu_pilot/taxonomy.py",
        "src/compbias/io/strict_json.py",
        "src/compbias/io/yaml_config.py",
        "src/compbias/models/structured_parser.py",
        "src/compbias/recoverability/bridge.py",
        "src/compbias/recoverability/bridge_v1_failure.py",
        "src/compbias/recoverability/config.py",
        "src/compbias/recoverability/dsl/executor.py",
        "src/compbias/recoverability/dsl/parser.py",
        "src/compbias/recoverability/dsl/schema.py",
        "src/compbias/recoverability/evidence.py",
        "src/compbias/recoverability/evidence_capture.py",
        "src/compbias/recoverability/preflight.py",
        "src/compbias/recoverability/stage1_v2.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--help" in sys.argv or "-h" in sys.argv:
        return
    try:
        raw = sys.argv[sys.argv.index("--server-package-lock") + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical Stage-1 v2 server package lock is required") from error
    root = Path(__file__).resolve().parents[2]
    lock_path = Path(raw).resolve()
    canonical = root / "configs/recoverability/server_package_lock_stage1_v2.yaml"
    if lock_path != canonical:
        raise SystemExit("BLOCKED: Stage-1 v2 server package lock path is not canonical")
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths:
        raise SystemExit("BLOCKED: Stage-1 v2 server package lock is malformed")
    if frozenset(paths) != _BOOTSTRAP_SERVER_PATHS or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: Stage-1 v2 server package lock closure is incomplete")
    for relative, expected in zip(paths, digests, strict=True):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: Stage-1 v2 package mismatch for {relative}")


_bootstrap_server_lock()

import importlib.metadata  # noqa: E402

from compbias.recoverability.preflight import (  # noqa: E402
    load_runtime_spec,
    run_metadata_preflight,
)
from compbias.recoverability.stage1_v2 import (  # noqa: E402
    verify_stage1_v2_server_package_lock,
)


def _metadata_subprocess_environment() -> dict[str, str]:
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
        raise FileExistsError(f"refusing to overwrite Stage-1 v2 preflight: {args.output}")
    if args.output.parent.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("Stage-1 v2 preflight output parent must be a regular directory")
    package_lock = verify_stage1_v2_server_package_lock(
        args.server_package_lock,
        repository_root=root,
    )
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
            "artifact_type": "recoverability_stage1_v2_metadata_preflight",
            "schema_version": 1,
            "server_package_lock_verified": package_lock.verified,
            "server_package_lock_sha256": _sha256(args.server_package_lock),
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
