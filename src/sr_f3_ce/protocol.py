"""Immutable SR-F3-CE identities and fail-closed stage boundaries."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sr_f12.evaluation import durable_json

EXPERIMENT_ID = "SR-F3-CE-20261016"
PARENTS = tuple(f"SRF2_J6_s{s}" for s in (71001, 71002, 71003))
ARMS = ("RL", "GOLD", "PREFIX", "EXEC")
SEEDS = (73101, 73102, 73103)
ENGINE_TASKS = ("engine--b0", "engine--b1", "engine--b2", "engine--resume")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def contained(root, relative, *, must_exist=True):
    root = Path(root).resolve(strict=True)
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise PermissionError("Relative path escapes its registered root")
    path = (root / rel).resolve(strict=must_exist)
    if not path.is_relative_to(root):
        raise PermissionError("Symlink escapes its registered root")
    return path


def stable_seed(*parts):
    return int(digest([EXPERIMENT_ID, *parts])[:16], 16) % (2**31 - 1)


def validate_config(config):
    if config["experiment_id"] != EXPERIMENT_ID:
        raise PermissionError("Wrong experiment identity")
    if config["parents"] != list(PARENTS) or config["arms"] != list(ARMS):
        raise PermissionError("Registered parent/arm matrix differs")
    if config["future_seeds"] != list(SEEDS):
        raise PermissionError("Registered random streams differ")
    expected = dict(
        updates=64,
        prompts_per_update=16,
        samples_per_prompt=8,
        support_examples_per_update=4,
        lr=1e-5,
        aux_gamma=0.1,
        max_new_tokens=1024,
        kl_beta=0.02,
        tis_cap=2,
        recovery_every=8,
        monitor_steps=[0, 16, 32, 48, 64],
    )
    if any(config["training"].get(key) != val for key, val in expected.items()):
        raise PermissionError("Frozen scientific training settings differ")
    if config["resources"] != dict(
        max_project_GPU_workers=5,
        GPU_hours_limit=None,
        research_wallclock_limit=None,
        stop_on_elapsed_time=False,
        respect_scheduler_lease=True,
    ):
        raise PermissionError("Frozen resource boundary differs")
    return config


def verify_package(plan_dir):
    plan_dir = Path(plan_dir).resolve(strict=True)
    hashes = read_json(plan_dir / "PACKAGE_SHA256.json")
    if not isinstance(hashes, dict) or not hashes:
        raise PermissionError("Missing package file identities")
    for relative, expected in hashes.items():
        if file_hash(contained(plan_dir, relative)) != expected:
            raise PermissionError(f"Frozen package changed: {relative}")
    config = validate_config(read_json(plan_dir / "config/SR_F3_CE.json"))
    return dict(
        files=len(hashes),
        config=config,
        package_manifest_sha256=file_hash(plan_dir / "PACKAGE_SHA256.json"),
    )


def verify_source(manifest_path, code_root):
    manifest = read_json(manifest_path)
    for item in manifest["files"]:
        path = contained(code_root, item["path"])
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise PermissionError(f"Deployed source differs: {item['path']}")
    return manifest


def verify_run(root, *, science=False):
    root = Path(root).resolve(strict=True)
    if (root / "STOP").exists():
        raise PermissionError("This run has a STOP marker")
    identity = read_json(root / "RUN_IDENTITY.json")
    if identity["experiment_id"] != EXPERIMENT_ID or Path(identity["root"]).resolve() != root:
        raise PermissionError("Run identity/root mismatch")
    for relative, expected in identity["bound_files"].items():
        if file_hash(contained(root, relative)) != expected:
            raise PermissionError(f"Run-bound input changed: {relative}")
    package = verify_package(root / "plan")
    if package["package_manifest_sha256"] != identity["package_manifest_sha256"]:
        raise PermissionError("Package identity differs")
    verify_source(root / "deployment/SOURCE_MANIFEST.json", root / "code")
    if science:
        freeze = read_json(root / "EXECUTION_FREEZE.json")
        if (
            freeze.get("status") != "FROZEN_BEFORE_SCIENCE"
            or freeze.get("run_identity_sha256") != file_hash(root / "RUN_IDENTITY.json")
            or freeze.get("operator_has_seen_new_test_results") is not False
        ):
            raise PermissionError("No prospective scientific freeze")
        for relative, expected in freeze["engineering_files"].items():
            if file_hash(contained(root, relative)) != expected:
                raise PermissionError("Engineering receipt changed after scientific freeze")
    return identity, package["config"]


def record_incident(root, stage, error, *, extra=None):
    """Retain evidence; never turn an exception into successful task coverage."""
    import time

    value = dict(
        experiment_id=EXPERIMENT_ID,
        at=now(),
        stage=stage,
        error_type=type(error).__name__,
        error=str(error),
        job_id=os.environ.get("SLURM_JOB_ID"),
        extra=extra or {},
    )
    path = Path(root) / "technical_incidents" / f"{time.time_ns()}-{os.getpid()}.json"
    durable_json(path, value)
    return path
