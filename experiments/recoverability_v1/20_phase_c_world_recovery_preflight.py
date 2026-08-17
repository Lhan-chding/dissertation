#!/usr/bin/env python3
"""Metadata-only preflight for frozen world-recovery diagnostics."""

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

_INHERITED_LOCK_RELATIVE = "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml"
_PROFILE_LOCKS = {
    "configs/recoverability/phase_c_world_recovery_v1.yaml": (
        "configs/recoverability/server_package_lock_phase_c_world_recovery_v1.yaml"
    ),
    "configs/recoverability/phase_c_world_recovery_100_v1.yaml": (
        "configs/recoverability/server_package_lock_phase_c_world_recovery_100_v1.yaml"
    ),
}
_COMMON_ADDITIONS = frozenset(
    {
        _INHERITED_LOCK_RELATIVE,
        "experiments/recoverability_v1/20_phase_c_world_recovery_preflight.py",
        "experiments/recoverability_v1/21_run_phase_c_world_recovery.py",
        "prompts/no_cue.user.template.txt",
        "prompts/valid_cue.user.template.txt",
        "prompts/world_recovery_v1_ablation_no_examples.system.txt",
        "prompts/world_recovery_v1_main.system.txt",
        "src/compbias/recoverability/phase_c_world_recovery.py",
        "src/compbias/recoverability/phase_c_world_recovery_execution.py",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lock_rows(path: Path) -> tuple[tuple[str, str], ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    paths = [line.split(":", 1)[1].strip() for line in lines if line.startswith("  - path:")]
    digests = [line.split(":", 1)[1].strip() for line in lines if line.startswith("    sha256:")]
    if len(paths) != len(digests) or not paths or len(paths) != len(set(paths)):
        raise SystemExit("BLOCKED: world recovery package lock is malformed")
    return tuple(zip(paths, digests, strict=True))


def _bootstrap_server_lock() -> None:
    if __name__ != "__main__" or "--help" in sys.argv or "-h" in sys.argv:
        return
    try:
        supplied_lock = Path(sys.argv[sys.argv.index("--server-package-lock") + 1]).resolve()
        supplied_config = Path(sys.argv[sys.argv.index("--qualification-config") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("BLOCKED: canonical world recovery profile is required") from error
    root = Path(__file__).resolve().parents[2]
    inherited = root / _INHERITED_LOCK_RELATIVE
    matched = next(
        (
            (config_relative, lock_relative)
            for config_relative, lock_relative in _PROFILE_LOCKS.items()
            if supplied_config == root / config_relative and supplied_lock == root / lock_relative
        ),
        None,
    )
    if matched is None or supplied_config.is_symlink() or supplied_lock.is_symlink():
        raise SystemExit("BLOCKED: world recovery profile paths are not canonical")
    config_relative, _lock_relative = matched
    inherited_rows = _lock_rows(inherited)
    rows = _lock_rows(supplied_lock)
    expected = _COMMON_ADDITIONS | {config_relative}
    if frozenset(relative for relative, _digest in rows) != expected:
        raise SystemExit("BLOCKED: world recovery package closure is incomplete")
    for relative, expected in inherited_rows + rows:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256(candidate) != expected:
            raise SystemExit(f"BLOCKED: world recovery package mismatch for {relative}")


_bootstrap_server_lock()


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def _pip_check() -> tuple[int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(),
    )
    return completed.returncode, completed.stdout + completed.stderr


def _pip_inventory() -> dict[str, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(),
    )
    return {row["name"]: row["version"] for row in json.loads(completed.stdout)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--qualification-config", type=Path, required=True)
    parser.add_argument("--server-package-lock", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    runtime = root / "configs/recoverability/server_runtime_v1.yaml"
    matched = next(
        (
            (root / config_relative, root / lock_relative)
            for config_relative, lock_relative in _PROFILE_LOCKS.items()
            if args.qualification_config.resolve() == root / config_relative
            and args.server_package_lock.resolve() == root / lock_relative
        ),
        None,
    )
    if matched is None:
        raise ValueError("world recovery profile paths are not canonical")
    config_path, lock = matched
    if args.runtime.resolve() != runtime or args.runtime.is_symlink():
        raise ValueError("world recovery runtime path is not canonical")
    if args.qualification_config.is_symlink() or args.server_package_lock.is_symlink():
        raise ValueError("world recovery profile paths must not be symlinks")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite world recovery preflight: {args.output}")
    if args.output.parent.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("world recovery preflight parent must be a regular directory")

    from compbias.recoverability.phase_c_world_recovery import (
        load_phase_c_world_recovery_config,
    )
    from compbias.recoverability.phase_c_world_recovery_execution import (
        verify_phase_c_world_recovery_package_lock,
    )
    from compbias.recoverability.preflight import load_runtime_spec, run_metadata_preflight

    config = load_phase_c_world_recovery_config(config_path)
    package = verify_phase_c_world_recovery_package_lock(lock, repository_root=root)
    report = run_metadata_preflight(
        load_runtime_spec(runtime),
        repository_root=root,
        version_lookup=importlib.metadata.version,
        inventory_lookup=_pip_inventory,
        pip_check=_pip_check,
        environ=os.environ,
    )
    if not report.ready or report.large_gpu_started or report.model_loaded:
        raise RuntimeError("world recovery metadata preflight did not remain inert")
    payload = {
        **asdict(report),
        "artifact_type": "recoverability_phase_c_world_recovery_metadata_preflight",
        "schema_version": 1,
        "qualification_id": config.qualification_id,
        "server_package_lock_verified": True,
        "server_package_lock_sha256": _sha256(lock),
        "server_package_files": [item.relative_path for item in package.files],
        "model_loaded": False,
        "model_call_cap": config.model_call_cap,
        "hypothesis_tested": config.hypothesis_tested,
        "scale_authorized": config.scale_authorized,
        "training_authorized": config.training_authorized,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
