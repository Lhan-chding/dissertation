"""Atomic, provenance-complete run logging utilities."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

_RUN_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_PACKAGES = (
    "compbias",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "Pillow",
    "PyYAML",
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "qwen-vl-utils",
    "trl",
    "verl",
    "vllm",
)
_REQUIRED_RUN_FILES = (
    "config.yaml",
    "environment.json",
    "metrics.jsonl",
    "rollouts.jsonl",
    "predictions.npz",
    "report.md",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_output(worktree: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(worktree), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _cuda_environment() -> tuple[bool, list[str]]:
    try:
        import torch
    except (ImportError, OSError):
        return False, []
    try:
        available = bool(torch.cuda.is_available())
        devices = (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if available
            else []
        )
    except (AssertionError, RuntimeError, OSError):
        return False, []
    return available, devices


def publishable_path(path: Path | str, *, worktree: Path | str) -> str:
    """Render a path without embedding a machine-specific workspace prefix."""

    root = Path(worktree).expanduser().resolve()
    candidate = Path(path).expanduser()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        name = resolved.name or "path"
        return f"<external>/{name}"


def publishable_command(command: Sequence[str], *, worktree: Path | str) -> tuple[str, ...]:
    """Return an exact-argument command with local absolute prefixes redacted."""

    if isinstance(command, (str, bytes)):
        raise TypeError("command must be a sequence of arguments, not text")
    root = Path(worktree).expanduser().resolve()
    rendered: list[str] = []
    for index, raw_part in enumerate(command):
        if not isinstance(raw_part, (str, os.PathLike)):
            raise TypeError("command arguments must be strings or path-like values")
        part = os.fspath(raw_part)
        if not part:
            raise ValueError("command arguments must not be empty")
        if index == 0 and Path(part).is_absolute():
            rendered.append(Path(part).name)
            continue
        if "=" in part and part.startswith("-"):
            option, value = part.split("=", 1)
            if Path(value).is_absolute():
                rendered.append(f"{option}={publishable_path(value, worktree=root)}")
                continue
        if Path(part).is_absolute():
            rendered.append(publishable_path(part, worktree=root))
        else:
            rendered.append(part)
    return tuple(rendered)


def publishable_config_snapshot(
    config: Mapping[str, object],
    *,
    path_fields: Sequence[Sequence[str]],
    worktree: Path | str,
) -> dict[str, object]:
    """Return an immutable-style config copy with registered path fields redacted."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    snapshot = copy.deepcopy(dict(config))
    for raw_field in path_fields:
        field = tuple(raw_field)
        if not field or any(not isinstance(component, str) or not component for component in field):
            raise ValueError("path field selectors must be non-empty string sequences")
        cursor: dict[str, object] = snapshot
        for component in field[:-1]:
            child = cursor.get(component)
            if not isinstance(child, dict):
                raise ValueError(f"config path field is missing: {'.'.join(field)}")
            cursor = child
        leaf = field[-1]
        value = cursor.get(leaf)
        if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
            raise ValueError(f"config path field must be path-like: {'.'.join(field)}")
        cursor[leaf] = publishable_path(value, worktree=worktree)
    return snapshot


def capture_environment(
    *,
    worktree: Path | str,
    dataset_manifest_hash: str | None,
    seed: int,
    model_revision: str | None,
    verl_revision: str | None,
    command: Sequence[str],
) -> dict[str, object]:
    """Capture a fresh, JSON-serializable provenance snapshot."""

    root = Path(worktree).expanduser().resolve()
    git_commit = _git_output(root, "rev-parse", "HEAD")
    dirty_output = _git_output(root, "status", "--porcelain")
    cuda_available, gpu_devices = _cuda_environment()
    return {
        "git_commit": git_commit,
        "git_dirty": None if dirty_output is None else bool(dirty_output),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "cuda_available": cuda_available,
        "gpu_devices": gpu_devices,
        "dataset_manifest_hash": dataset_manifest_hash,
        "seed": seed,
        "model_revision": model_revision,
        "verl_revision": verl_revision,
        "command": list(publishable_command(command, worktree=root)),
        "start_timestamp": _timestamp(),
    }


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize value of type {type(value).__name__}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class RunLogger:
    """Own one non-overwritable run directory and its canonical artifacts."""

    def __init__(
        self,
        *,
        root: Path | str,
        experiment: str,
        run_id: str,
        config: Mapping[str, object],
        environment: Mapping[str, object],
    ) -> None:
        self._validate_component("experiment", experiment)
        self._validate_component("run_id", run_id)
        self.root = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
        self.experiment = experiment
        self.run_id = run_id
        self.run_dir = self.root / experiment / run_id
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self._config = copy.deepcopy(dict(config))
        self._environment = copy.deepcopy(dict(environment))
        self._finalized = False

        if self.root.is_symlink():
            raise ValueError("run root must not be a symbolic link")
        self.root.mkdir(parents=True, exist_ok=True)
        experiment_dir = self.root / experiment
        if experiment_dir.is_symlink():
            raise ValueError("run experiment directory must not be a symbolic link")
        experiment_dir.mkdir(exist_ok=True)
        self.run_dir.mkdir()
        self.checkpoint_dir.mkdir()
        self._validate_run_tree()
        _atomic_write_text(
            self.run_dir / "config.yaml",
            yaml.safe_dump(self._config, sort_keys=True),
        )
        self._write_environment()

    def _validate_run_tree(self) -> None:
        """Recheck that every live run directory is the original non-symlink tree."""

        experiment_dir = self.root / self.experiment
        for label, path in (
            ("run root", self.root),
            ("run experiment directory", experiment_dir),
            ("run directory", self.run_dir),
            ("checkpoint directory", self.checkpoint_dir),
        ):
            if path.is_symlink():
                raise ValueError(f"{label} must not be a symbolic link")
            if not path.is_dir():
                raise RuntimeError(f"{label} is missing or is not a directory")
        resolved_root = self.root.resolve(strict=True)
        for path in (experiment_dir, self.run_dir, self.checkpoint_dir):
            try:
                path.resolve(strict=True).relative_to(resolved_root)
            except ValueError as error:
                raise ValueError("run directory escaped the configured root") from error

    @staticmethod
    def _validate_component(name: str, value: object) -> None:
        if not isinstance(value, str) or _RUN_COMPONENT.fullmatch(value) is None:
            raise ValueError(f"{name} must be a safe non-empty run-directory component")

    def _ensure_active(self) -> None:
        if self._finalized:
            raise RuntimeError("run has already been finalized")
        self._validate_run_tree()

    def _write_environment(self) -> None:
        self._validate_run_tree()
        _atomic_write_text(
            self.run_dir / "environment.json",
            json.dumps(
                self._environment,
                default=_json_default,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )

    def _append_jsonl(self, filename: str, record: Mapping[str, object]) -> None:
        self._ensure_active()
        payload = copy.deepcopy(dict(record))
        line = (
            json.dumps(
                payload,
                default=_json_default,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        descriptor = os.open(
            self.run_dir / filename,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            remaining = memoryview(line)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("could not append run log record")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def log_metrics(self, metrics: Mapping[str, object]) -> None:
        self._append_jsonl("metrics.jsonl", metrics)

    def log_rollout(self, rollout: Mapping[str, object]) -> None:
        self._append_jsonl("rollouts.jsonl", rollout)

    def save_predictions(self, predictions: Mapping[str, object]) -> None:
        self._ensure_active()
        invalid_names = tuple(
            name
            for name in predictions
            if not isinstance(name, str) or _RUN_COMPONENT.fullmatch(name) is None
        )
        if invalid_names:
            raise ValueError("every prediction name must be a safe archive-member component")
        arrays = {name: np.asarray(value) for name, value in predictions.items()}
        if not arrays:
            raise ValueError("predictions must contain at least one named array")
        if any(array.dtype.hasobject for array in arrays.values()):
            raise ValueError("object arrays are not permitted in predictions.npz")
        target = self.run_dir / "predictions.npz"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=self.run_dir,
                prefix=".predictions.",
                suffix=".npz",
                delete=False,
            ) as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def write_report(self, report: str) -> None:
        self._ensure_active()
        if not isinstance(report, str):
            raise TypeError("report must be text")
        _atomic_write_text(self.run_dir / "report.md", report)

    def _finalize(
        self,
        *,
        checkpoint_hash: str | None,
        require_complete: bool,
    ) -> None:
        self._ensure_active()
        if require_complete:
            self._require_complete_bundle()
        self._environment = {
            **self._environment,
            "checkpoint_hash": checkpoint_hash,
            "end_timestamp": _timestamp(),
        }
        self._write_environment()
        self._finalized = True

    def finalize(self, *, checkpoint_hash: str | None = None) -> None:
        self._finalize(checkpoint_hash=checkpoint_hash, require_complete=True)

    def _require_complete_bundle(self) -> None:
        self._validate_run_tree()
        missing = tuple(
            filename for filename in _REQUIRED_RUN_FILES if not (self.run_dir / filename).is_file()
        )
        if missing:
            raise RuntimeError("missing required run artifacts: " + ", ".join(missing))

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        if not self._finalized:
            if exc_type is not None:
                self._environment = {
                    **self._environment,
                    "status": "failed",
                    "error_type": getattr(exc_type, "__name__", str(exc_type)),
                }
                self._finalize(checkpoint_hash=None, require_complete=False)
            else:
                self.finalize()
        return False


__all__ = [
    "RunLogger",
    "capture_environment",
    "publishable_command",
    "publishable_path",
]
