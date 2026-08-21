"""Fail-closed boundary for the server-only v5 support-LoRA runs.

This module intentionally has no dependency on torch, transformers, PEFT, or
any other GPU runtime.  It validates immutable inputs before a server runner is
allowed to import those packages.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
SUPPORT_LORA_ACK = "I_UNDERSTAND_THIS_STARTS_V5_BUDGET_MATCHED_LORA"
REQUIRED_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


class ServerExecutionBlocked(RuntimeError):
    """Raised before any accelerator import when an execution gate fails."""


@dataclass(frozen=True, slots=True)
class ValidatedExecution:
    phase: str
    config_sha256: str
    package_lock_sha256: str
    input_sha256: tuple[tuple[str, str], ...]
    output: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase": self.phase,
            "config_sha256": self.config_sha256,
            "package_lock_sha256": self.package_lock_sha256,
            "input_sha256": dict(self.input_sha256),
            "output": self.output,
        }


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 of a regular, non-symlink file."""

    if path.is_symlink() or not path.is_file():
        raise ServerExecutionBlocked(f"unsafe or missing input file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON-compatible value with deterministic encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ServerExecutionBlocked(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_file(path: Path, expected: Path, expected_sha256: str, label: str) -> str:
    if path.is_symlink() or not path.is_file() or path.resolve() != expected.resolve():
        raise ServerExecutionBlocked(f"{label} must be canonical repository file: {expected}")
    expected_digest = _require_sha256(expected_sha256, f"{label} expected hash")
    actual = sha256_file(path)
    if actual != expected_digest:
        raise ServerExecutionBlocked(f"{label} SHA-256 mismatch")
    return actual


def _load_yaml_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ServerExecutionBlocked(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ServerExecutionBlocked(f"{label} must be a schema_version 1 mapping")
    return value


def _offline_config_enabled(config: Mapping[str, object]) -> bool:
    value = config.get("offline")
    if value is True:
        return True
    if isinstance(value, Mapping):
        required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        return all(_offline_flag_enabled(value.get(key)) for key in required)
    return config.get("offline_only") is True


def _offline_flag_enabled(value: object) -> bool:
    if value is True:
        return True
    if value == "1":
        return True
    return isinstance(value, int) and not isinstance(value, bool) and value == 1


def _require_authorization(config: Mapping[str, object], required: Mapping[str, bool]) -> None:
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ServerExecutionBlocked("config authorization mapping is missing")
    for key, expected in required.items():
        if authorization.get(key) is not expected:
            raise ServerExecutionBlocked(f"config authorization gate {key!r} must be {expected}")


def _require_hash_fields_bound(value: object, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            label = f"{path}.{key}"
            if isinstance(key, str) and key.endswith("_sha256"):
                _require_sha256(item, label)
            else:
                _require_hash_fields_bound(item, label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _require_hash_fields_bound(item, f"{path}[{index}]")


def require_offline_environment(environment: Mapping[str, str] | None = None) -> None:
    """Require the three offline flags used by the Hugging Face stack."""

    current = os.environ if environment is None else environment
    missing = [key for key, value in REQUIRED_OFFLINE_ENV.items() if current.get(key) != value]
    if missing:
        raise ServerExecutionBlocked(f"offline environment is incomplete: {', '.join(missing)}")


def validate_new_output(output: Path, allowed_root: Path) -> Path:
    """Validate a new output path without creating or overwriting anything."""

    if output.exists() or output.is_symlink():
        raise ServerExecutionBlocked(f"output already exists; overwrite forbidden: {output}")
    root = allowed_root.resolve(strict=False)
    target = output.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ServerExecutionBlocked(f"output must remain under {allowed_root}") from error
    cursor = output.parent
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ServerExecutionBlocked(f"output parent must not be a symlink: {cursor}")
        if cursor.resolve(strict=False) == root:
            break
        cursor = cursor.parent
    return target


def validate_server_execution(
    *,
    phase: str,
    execute: bool,
    acknowledgement: str | None,
    required_acknowledgement: str,
    config: Path,
    canonical_config: Path,
    config_sha256: str,
    package_lock: Path,
    canonical_package_lock: Path,
    package_lock_sha256: str,
    inputs: Sequence[Path],
    input_sha256: Sequence[str],
    output: Path,
    allowed_output_root: Path,
    required_authorization: Mapping[str, bool],
    expected_config_phase: str | None = None,
    expected_seed_count: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> ValidatedExecution:
    """Apply all non-GPU execution gates in a fixed fail-closed order."""

    if not execute:
        raise ServerExecutionBlocked("explicit --execute is required")
    if acknowledgement != required_acknowledgement:
        raise ServerExecutionBlocked("exact execution acknowledgement is required")
    require_offline_environment(environment)
    config_digest = _canonical_file(config, canonical_config, config_sha256, f"{phase} config")
    lock_digest = _canonical_file(
        package_lock,
        canonical_package_lock,
        package_lock_sha256,
        "v5 server package lock",
    )
    config_payload = _load_yaml_mapping(config, f"{phase} config")
    _load_yaml_mapping(package_lock, "v5 server package lock")
    _require_hash_fields_bound(config_payload)
    if expected_config_phase is not None and config_payload.get("phase") != expected_config_phase:
        raise ServerExecutionBlocked(f"config phase must be exactly {expected_config_phase!r}")
    if not _offline_config_enabled(config_payload):
        raise ServerExecutionBlocked("config must enforce offline execution")
    _require_authorization(config_payload, required_authorization)
    if expected_seed_count is not None:
        seeds = config_payload.get("seeds")
        if (
            not isinstance(seeds, Sequence)
            or isinstance(seeds, (str, bytes))
            or len(seeds) != expected_seed_count
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        ):
            raise ServerExecutionBlocked(
                f"config must register exactly {expected_seed_count} integer pilot seed(s)"
            )
    if len(inputs) != len(input_sha256):
        raise ServerExecutionBlocked("each input requires exactly one --input-sha256")
    if not inputs:
        raise ServerExecutionBlocked("at least one hash-bound server input is required")
    validated_inputs: list[tuple[str, str]] = []
    for index, (path, expected) in enumerate(zip(inputs, input_sha256, strict=True)):
        expected_digest = _require_sha256(expected, f"input[{index}] expected hash")
        actual = sha256_file(path)
        if actual != expected_digest:
            raise ServerExecutionBlocked(f"input[{index}] SHA-256 mismatch: {path}")
        validated_inputs.append((str(path.resolve()), actual))
    target = validate_new_output(output, allowed_output_root)
    return ValidatedExecution(
        phase=phase,
        config_sha256=config_digest,
        package_lock_sha256=lock_digest,
        input_sha256=tuple(validated_inputs),
        output=str(target),
    )


def execute_support_lora(
    validation: ValidatedExecution,
    *,
    runner: Callable[[Mapping[str, object]], Any] | None = None,
) -> Any:
    """Cross the GPU boundary only through an injected server runner.

    Keeping the runner injected ensures that importing or dry-running this
    repository never imports an accelerator package or loads a model.
    """

    if runner is None:
        raise ServerExecutionBlocked(
            "server runner is not installed in the local package; transfer the frozen bundle"
        )
    return runner(validation.to_mapping())


def load_server_runner(
    specification: str | None,
    *,
    allowed_module_prefix: str = "compensability_v5.server_runtime.",
) -> Callable[..., Any]:
    """Load an explicitly registered runtime callback after all gates pass."""

    if not isinstance(specification, str) or specification.count(":") != 1:
        raise ServerExecutionBlocked("a server --runtime module:function callback is required")
    module_name, attribute = specification.split(":", 1)
    if not module_name.startswith(allowed_module_prefix) or not attribute.isidentifier():
        raise ServerExecutionBlocked(
            f"server runtime must be callable below {allowed_module_prefix}"
        )
    try:
        module = importlib.import_module(module_name)
        runner = getattr(module, attribute)
    except (ImportError, AttributeError) as error:
        raise ServerExecutionBlocked(f"server runtime callback is unavailable: {error}") from error
    if not callable(runner):
        raise ServerExecutionBlocked("server runtime attribute must be callable")
    return runner


__all__ = [
    "REQUIRED_OFFLINE_ENV",
    "SUPPORT_LORA_ACK",
    "ServerExecutionBlocked",
    "ValidatedExecution",
    "canonical_json_sha256",
    "execute_support_lora",
    "load_server_runner",
    "require_offline_environment",
    "sha256_file",
    "validate_new_output",
    "validate_server_execution",
]
