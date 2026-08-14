"""Metadata-only, fail-closed server preflight for Recoverability v1."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

_RELATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PACKAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    schema_version: int
    status: str
    requirements_lock_path: str
    requirements_lock_sha256: str
    exact_packages: Mapping[str, str]
    offline_required: bool
    downloads_allowed: bool
    model_loading_allowed: bool
    training_authorized: bool


@dataclass(frozen=True, slots=True)
class MetadataPreflightReport:
    ready: bool
    requirements_lock_sha256: str
    installed_packages: tuple[tuple[str, str], ...]
    pip_check_passed: bool
    offline_verified: bool
    large_gpu_started: bool
    model_loaded: bool
    training_authorized: bool


def load_runtime_spec(path: Path) -> RuntimeSpec:
    """Load the exact runtime contract without importing any ML package."""

    mapping = load_yaml_mapping(path, label="recoverability server runtime")
    fields = {
        "schema_version",
        "status",
        "requirements_lock_path",
        "requirements_lock_sha256",
        "exact_packages",
        "offline_required",
        "downloads_allowed",
        "model_loading_allowed",
        "training_authorized",
    }
    reject_unknown_fields(mapping, fields, label="recoverability server runtime")
    if set(mapping) != fields or mapping["schema_version"] != 1:
        raise ValueError("recoverability server runtime schema is invalid")
    if mapping["status"] != "PREREGISTERED_NOT_RUN":
        raise ValueError("recoverability server runtime status is not preregistered")
    relative = mapping["requirements_lock_path"]
    digest = mapping["requirements_lock_sha256"]
    if not isinstance(relative, str) or _RELATIVE.fullmatch(relative) is None:
        raise ValueError("requirements lock path is invalid")
    if relative != "requirements-gpu.lock.txt":
        raise ValueError("requirements lock path differs from the registered path")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("requirements lock digest is invalid")
    packages = mapping["exact_packages"]
    if not isinstance(packages, Mapping) or not packages:
        raise ValueError("exact_packages must be a non-empty mapping")
    parsed_packages: dict[str, str] = {}
    for package, version in packages.items():
        if not isinstance(package, str) or _PACKAGE.fullmatch(package) is None:
            raise ValueError("package name is invalid")
        if not isinstance(version, str) or _PACKAGE.fullmatch(version) is None:
            raise ValueError("package version is invalid")
        parsed_packages[package] = version
    expected_packages = {
        "accelerate": "1.14.0",
        "numpy": "2.5.2",
        "peft": "0.19.1",
        "qwen-vl-utils": "0.0.14",
        "scipy": "1.18.0",
        "torch": "2.8.0+cu128",
        "transformers": "5.14.1",
    }
    if parsed_packages != expected_packages:
        raise ValueError("exact_packages differ from the registered GPU runtime")
    for key in (
        "offline_required",
        "downloads_allowed",
        "model_loading_allowed",
        "training_authorized",
    ):
        if type(mapping[key]) is not bool:
            raise TypeError(f"{key} must be boolean")
    if mapping["offline_required"] is not True:
        raise ValueError("offline_required must remain true")
    if any(
        mapping[key] is not False
        for key in ("downloads_allowed", "model_loading_allowed", "training_authorized")
    ):
        raise ValueError("downloads, model loading, and training must remain disabled in preflight")
    return RuntimeSpec(
        schema_version=1,
        status="PREREGISTERED_NOT_RUN",
        requirements_lock_path=relative,
        requirements_lock_sha256=digest,
        exact_packages=MappingProxyType(dict(sorted(parsed_packages.items()))),
        offline_required=True,
        downloads_allowed=False,
        model_loading_allowed=False,
        training_authorized=False,
    )


def run_metadata_preflight(
    spec: RuntimeSpec,
    *,
    repository_root: Path,
    version_lookup: Callable[[str], str],
    inventory_lookup: Callable[[], Mapping[str, str]],
    pip_check: Callable[[], tuple[int, str]],
    environ: Mapping[str, str],
) -> MetadataPreflightReport:
    """Verify bytes and package metadata before any model or GPU import."""

    if not isinstance(spec, RuntimeSpec):
        raise TypeError("spec must be RuntimeSpec")
    if environ.get("HF_HUB_OFFLINE") != "1" or environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("offline environment is required before preflight")
    root = repository_root.resolve()
    lock_path = root / spec.requirements_lock_path
    resolved = lock_path.resolve()
    if root not in resolved.parents or lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError("requirements lock must be a regular repository file")
    actual_lock_digest = _sha256(lock_path)
    if actual_lock_digest != spec.requirements_lock_sha256:
        raise RuntimeError("requirements lock SHA-256 mismatch")
    expected_inventory: dict[str, str] = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if "==" not in line:
            raise RuntimeError("requirements lock must contain only exact name==version entries")
        name, version = line.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if not normalized or not version or normalized in expected_inventory:
            raise RuntimeError("requirements lock package inventory is invalid")
        expected_inventory[normalized] = version
    observed_inventory = {
        re.sub(r"[-_.]+", "-", name).lower(): version
        for name, version in inventory_lookup().items()
    }
    if observed_inventory != expected_inventory:
        raise RuntimeError("installed package inventory differs from the exact requirements lock")
    installed: list[tuple[str, str]] = []
    for package, expected in spec.exact_packages.items():
        try:
            actual = version_lookup(package)
        except (KeyError, LookupError) as error:
            raise RuntimeError(f"missing package: {package}") from error
        if actual != expected:
            raise RuntimeError(
                f"version mismatch for {package}: expected {expected}, found {actual}"
            )
        installed.append((package, actual))
    return_code, pip_output = pip_check()
    if return_code != 0:
        bounded = pip_output.strip().replace("\n", " ")[:500]
        raise RuntimeError(f"pip check failed: {bounded}")
    return MetadataPreflightReport(
        ready=True,
        requirements_lock_sha256=actual_lock_digest,
        installed_packages=tuple(installed),
        pip_check_passed=True,
        offline_verified=True,
        large_gpu_started=False,
        model_loaded=False,
        training_authorized=False,
    )
