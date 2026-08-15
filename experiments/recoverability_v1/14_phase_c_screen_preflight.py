#!/usr/bin/env python3
"""Metadata-only preflight for the amended confirmatory Phase-C v2 screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

_LOCK_RELATIVE = "configs/recoverability/server_package_lock_phase_c_screen_v2.yaml"
_INHERITED_LOCK_RELATIVE = "configs/recoverability/server_package_lock_phase_n.yaml"
_BOOTSTRAP_ADDITIONS = frozenset(
    {
        "configs/recoverability/phase_n_frozen_result.yaml",
        "configs/recoverability/recoverability_phase_c_v2_amendment.yaml",
        _INHERITED_LOCK_RELATIVE,
        "experiments/recoverability_v1/14_phase_c_screen_preflight.py",
        "experiments/recoverability_v1/15_run_phase_c_screen.py",
        "src/compbias/recoverability/compatibility.py",
        "src/compbias/recoverability/operators.py",
        "src/compbias/recoverability/phase_c_amendment.py",
        "src/compbias/recoverability/phase_c_screen.py",
        "src/compbias/recoverability/phase_c_screen_execution.py",
        "src/compbias/recoverability/phase_n_result.py",
        "src/compbias/recoverability/selection.py",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lock_rows(path: Path) -> tuple[tuple[str, str], ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: Phase C screen package lock is malformed")
    return tuple(zip(paths, digests, strict=True))


def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--help" in sys.argv or "-h" in sys.argv:
        return
    try:
        supplied = Path(sys.argv[sys.argv.index("--server-package-lock") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical Phase C screen lock is required") from error
    root = Path(__file__).resolve().parents[2]
    canonical = root / _LOCK_RELATIVE
    inherited = root / _INHERITED_LOCK_RELATIVE
    if supplied != canonical or canonical.is_symlink() or inherited.is_symlink():
        raise SystemExit("BLOCKED: Phase C screen package lock path is not canonical")
    inherited_paths = frozenset(relative for relative, _ in _lock_rows(inherited))
    rows = _lock_rows(canonical)
    if frozenset(relative for relative, _ in rows) != inherited_paths | _BOOTSTRAP_ADDITIONS:
        raise SystemExit("BLOCKED: Phase C screen package lock closure is incomplete")
    for relative, expected in rows:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: Phase C screen package mismatch for {relative}")


_bootstrap_server_lock()


def _metadata_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def _pip_check() -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        env=_metadata_environment(),
    )
    return completed.returncode, completed.stdout + completed.stderr


def _pip_inventory() -> dict[str, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
        env=_metadata_environment(),
    )
    return {row["name"]: row["version"] for row in json.loads(completed.stdout)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--server-package-lock", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    runtime = root / "configs/recoverability/server_runtime_v1.yaml"
    lock = root / _LOCK_RELATIVE
    if args.runtime.resolve() != runtime or args.runtime.is_symlink():
        raise ValueError("Phase C screen runtime path is not canonical")
    if args.server_package_lock.resolve() != lock or args.server_package_lock.is_symlink():
        raise ValueError("Phase C screen package lock path is not canonical")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Phase C screen preflight: {args.output}")
    if args.output.parent.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("Phase C screen preflight parent must be a regular directory")

    from compbias.recoverability.phase_c_screen_execution import (
        verify_phase_c_screen_execution_package_lock,
    )
    from compbias.recoverability.preflight import load_runtime_spec, run_metadata_preflight

    package = verify_phase_c_screen_execution_package_lock(lock, repository_root=root)
    report = run_metadata_preflight(
        load_runtime_spec(runtime),
        repository_root=root,
        version_lookup=importlib.metadata.version,
        inventory_lookup=_pip_inventory,
        pip_check=_pip_check,
        environ=os.environ,
    )
    if not report.ready or report.large_gpu_started or report.model_loaded:
        raise RuntimeError("Phase C screen metadata preflight did not remain inert")
    payload = {
        **asdict(report),
        "artifact_type": "recoverability_phase_c_v2_screen_metadata_preflight",
        "schema_version": 1,
        "server_package_lock_verified": True,
        "server_package_lock_sha256": _sha256(lock),
        "server_package_files": [item.relative_path for item in package.files],
        "model_loaded": False,
        "confirmatory_screen_authorized": True,
        "training_authorized": False,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
