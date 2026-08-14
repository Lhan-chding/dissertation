"""Metadata-only safety gates for the large Qwen-VL experiment stage.

This module deliberately imports no training framework.  It can validate the
operator's acknowledgement, pinned provenance, audited veRL API surface, and
externally observed GPU metadata before a caller chooses to import ``torch``,
``transformers``, ``verl``, or ``vllm``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
PINNED_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
PINNED_VERL_REVISION = "7aed6b230776f963fa09509c10d9c3a767d1102c"
PINNED_TRANSFORMERS_REVISION = "5.3.0"
PINNED_VLLM_REVISION = "0.20.2"
PINNED_VERL_DOCKERFILE_SHA256 = "be8bd117fc415690c2d433e2e3c8832e6a96dd6de4e799be6a4be05c9eb4f300"

_MOVING_REVISIONS = frozenset({"head", "latest", "main", "master", "stable"})
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_GPU_UUID = re.compile(r"(?:GPU|MIG-GPU)-[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
_SAFE_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_SNAPSHOT_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000
_SAFE_SNAPSHOT_SUFFIXES = frozenset(
    {".json", ".safetensors", ".txt", ".model", ".tiktoken", ".jinja", ".md"}
)
_SAFE_SUFFIXLESS_SNAPSHOT_FILES = frozenset({"LICENSE", "NOTICE"})
_FORBIDDEN_CUSTOM_CODE_KEYS = frozenset({"auto_map", "custom_pipelines", "trust_remote_code"})


def _validate_revision(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} revision must be a non-empty pinned revision")
    if value.strip().lower() in _MOVING_REVISIONS:
        raise ValueError(f"{name} revision must be pinned, not {value!r}")


@dataclass(frozen=True, slots=True)
class VLMPreflightConfig:
    """Immutable provenance and operator gates for one proposed GPU run."""

    model_name: str
    model_revision: str
    transformers_revision: str
    verl_revision: str
    vllm_revision: str
    acknowledge_large_gpu_run: bool
    verl_api_audited: bool

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        for name in ("model", "transformers", "verl", "vllm"):
            _validate_revision(name, getattr(self, f"{name}_revision"))
        if not isinstance(self.acknowledge_large_gpu_run, bool):
            raise TypeError("acknowledge_large_gpu_run must be a boolean")
        if not isinstance(self.verl_api_audited, bool):
            raise TypeError("verl_api_audited must be a boolean")


@dataclass(frozen=True, slots=True, init=False)
class VLMPreflightReport:
    """Validated, serializable metadata; it does not imply training occurred."""

    model_name: str
    model_revision: str
    transformers_revision: str
    verl_revision: str
    vllm_revision: str
    gpu_devices: tuple[str, ...]
    verl_api_audited: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use validate_preflight to create a preflight report")

    @classmethod
    def _from_validation(
        cls,
        *,
        model_name: str,
        model_revision: str,
        transformers_revision: str,
        verl_revision: str,
        vllm_revision: str,
        gpu_devices: tuple[str, ...],
        verl_api_audited: bool,
    ) -> VLMPreflightReport:
        instance = object.__new__(cls)
        for field_name, value in locals().items():
            if field_name not in {"cls", "instance"}:
                object.__setattr__(instance, field_name, value)
        object.__setattr__(instance, "verl_api_audited", verl_api_audited)
        return instance

    def to_mapping(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "transformers_revision": self.transformers_revision,
            "verl_revision": self.verl_revision,
            "vllm_revision": self.vllm_revision,
            "gpu_devices": list(self.gpu_devices),
            "verl_api_audited": self.verl_api_audited,
        }


@dataclass(frozen=True, slots=True, init=False)
class ModelSnapshotEvidence:
    """A locally self-consistent snapshot record created only by the verifier."""

    path: str
    revision: str
    manifest_path: str
    manifest_sha256: str
    verified_file_count: int
    authenticity: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use verify_model_snapshot to create snapshot evidence")

    @classmethod
    def _from_self_consistency_check(
        cls,
        *,
        path: str,
        revision: str,
        manifest_path: str,
        manifest_sha256: str,
        verified_file_count: int,
    ) -> ModelSnapshotEvidence:
        instance = object.__new__(cls)
        object.__setattr__(instance, "path", path)
        object.__setattr__(instance, "revision", revision)
        object.__setattr__(instance, "manifest_path", manifest_path)
        object.__setattr__(instance, "manifest_sha256", manifest_sha256)
        object.__setattr__(instance, "verified_file_count", verified_file_count)
        object.__setattr__(instance, "authenticity", "self_consistency_only")
        instance._validate()
        return instance

    def _validate(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip() or "\x00" in self.path:
            raise ValueError("model snapshot path must be a non-empty path without NUL bytes")
        snapshot_path = Path(self.path)
        if not snapshot_path.is_absolute():
            raise ValueError("model snapshot path must be absolute")
        expected_repository = f"models--{DEFAULT_MODEL_NAME.replace('/', '--')}"
        parts = snapshot_path.parts
        if len(parts) < 3 or parts[-2:] != ("snapshots", self.revision):
            raise RuntimeError("model path must name the exact pinned snapshot revision")
        if expected_repository not in parts:
            raise RuntimeError("model snapshot path does not match the frozen model repository")
        if self.revision != PINNED_MODEL_REVISION:
            raise RuntimeError("verified model snapshot revision differs from the frozen revision")
        if (
            not isinstance(self.manifest_path, str)
            or not Path(self.manifest_path).is_absolute()
            or "\x00" in self.manifest_path
        ):
            raise ValueError("model snapshot manifest path must be absolute")
        if (
            not isinstance(self.manifest_sha256, str)
            or _SHA256.fullmatch(self.manifest_sha256) is None
        ):
            raise ValueError("model snapshot manifest SHA-256 must be 64 hexadecimal characters")
        if (
            isinstance(self.verified_file_count, bool)
            or not isinstance(self.verified_file_count, int)
            or self.verified_file_count < 4
        ):
            raise ValueError("verified model snapshot must contain at least four audited files")
        if self.authenticity != "self_consistency_only":
            raise RuntimeError("no trusted upstream snapshot allowlist is configured")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON numbers are forbidden")
    return parsed


def _validate_json_shape(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    visited = 0
    while pending:
        current, depth = pending.pop()
        visited += 1
        if depth > _MAX_JSON_DEPTH or visited > _MAX_JSON_NODES:
            raise ValueError("JSON artifact exceeds the permitted depth or complexity")
        if isinstance(current, dict):
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)


def _validated_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


def _validated_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_run_name(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _SAFE_RUN_NAME.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be 1-128 ASCII letters, digits, dots, hyphens, or underscores "
            "and begin with a letter or digit"
        )
    return value


def _read_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected snapshot manifest SHA-256 must be 64 hexadecimal characters")
    if not path.is_file():
        raise RuntimeError(f"model snapshot manifest is missing: {path}")
    size = path.stat().st_size
    if size > _MAX_MANIFEST_BYTES:
        raise RuntimeError("model snapshot manifest exceeds the 8 MiB safety limit")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(digest, expected_sha256.lower()):
        raise RuntimeError("model snapshot manifest SHA-256 does not match audited evidence")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise ValueError("model snapshot manifest must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("model snapshot manifest must contain a JSON object")
    _validate_json_shape(payload)
    return payload


def _validated_manifest_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("snapshot manifest file paths must be non-empty strings")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError("snapshot manifest file paths must stay inside the snapshot")
    return relative


def _validate_snapshot_member_type(relative: PurePosixPath, candidate: Path) -> None:
    if any(part.startswith(".") for part in relative.parts):
        raise RuntimeError(f"model snapshot contains a hidden member: {relative.as_posix()}")
    if (
        relative.suffix.lower() not in _SAFE_SNAPSHOT_SUFFIXES
        and relative.name not in _SAFE_SUFFIXLESS_SNAPSHOT_FILES
    ):
        raise RuntimeError(
            f"model snapshot contains an unsafe file extension: {relative.as_posix()}"
        )
    if candidate.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
        raise RuntimeError(f"model snapshot contains an executable file: {relative.as_posix()}")


def _prescan_snapshot_tree(snapshot: Path) -> tuple[Path, ...]:
    """Reject every unreviewable tree node before opening snapshot members."""

    nodes = tuple(snapshot.rglob("*"))
    for candidate in nodes:
        relative = PurePosixPath(candidate.relative_to(snapshot).as_posix())
        if any(part.startswith(".") for part in relative.parts):
            raise RuntimeError(f"model snapshot contains a hidden member: {relative.as_posix()}")
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"model snapshot tree contains a symlink: {relative.as_posix()}")
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise RuntimeError(
                f"model snapshot tree contains a non-regular filesystem node: {relative.as_posix()}"
            )
    return nodes


def _read_snapshot_json(path: Path, *, relative: PurePosixPath) -> object:
    if path.stat().st_size > _MAX_SNAPSHOT_JSON_BYTES:
        raise RuntimeError(f"snapshot JSON exceeds the 64 MiB limit: {relative.as_posix()}")
    try:
        payload = json.loads(
            path.read_bytes(),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise ValueError(f"snapshot JSON is invalid: {relative.as_posix()}") from error
    _validate_json_shape(payload)
    pending = [payload]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            forbidden = _FORBIDDEN_CUSTOM_CODE_KEYS.intersection(current)
            if forbidden:
                raise RuntimeError(
                    "model snapshot JSON requests custom code via "
                    f"{sorted(forbidden)}: {relative.as_posix()}"
                )
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return payload


def verify_model_snapshot(
    snapshot_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
) -> ModelSnapshotEvidence:
    """Verify a complete local snapshot without contacting the model hub."""

    raw_snapshot = Path(snapshot_path).expanduser()
    if raw_snapshot.is_symlink():
        raise RuntimeError("model snapshot directory may not be a symlink")
    snapshot = raw_snapshot.resolve(strict=True)
    if not snapshot.is_dir():
        raise RuntimeError("model snapshot path must be an existing directory")
    snapshot_nodes = _prescan_snapshot_tree(snapshot)
    raw_manifest = Path(manifest_path).expanduser()
    if raw_manifest.is_symlink():
        raise RuntimeError("model snapshot manifest may not be a symlink")
    resolved_manifest = raw_manifest.resolve(strict=True)
    payload = _read_manifest(resolved_manifest, expected_manifest_sha256)
    expected_manifest_fields = {
        "schema_version",
        "complete_snapshot",
        "model_name",
        "revision",
        "files",
    }
    if set(payload) != expected_manifest_fields:
        raise RuntimeError("model snapshot manifest must match the closed schema-v1 contract")
    if payload.get("schema_version") != 1 or payload.get("complete_snapshot") is not True:
        raise RuntimeError("model snapshot manifest is not a complete schema-v1 audit")
    if payload.get("model_name") != DEFAULT_MODEL_NAME:
        raise RuntimeError("model snapshot manifest names a different model")
    if payload.get("revision") != PINNED_MODEL_REVISION:
        raise RuntimeError("model snapshot manifest names a different revision")
    entries = payload.get("files")
    if not isinstance(entries, list) or len(entries) < 4:
        raise RuntimeError("model snapshot manifest must audit at least four files")

    audited_paths: set[str] = set()
    parsed_json: dict[str, object] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("every snapshot manifest file entry must be an object")
        if set(entry) != {"path", "size_bytes", "sha256"}:
            raise RuntimeError("snapshot manifest file entries must match the closed schema")
        relative = _validated_manifest_relative_path(entry.get("path"))
        relative_text = relative.as_posix()
        if relative_text in audited_paths:
            raise ValueError(f"duplicate snapshot manifest path: {relative_text}")
        audited_paths.add(relative_text)
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("snapshot manifest size_bytes must be a non-negative integer")
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or _SHA256.fullmatch(expected_hash) is None:
            raise ValueError("snapshot file SHA-256 must be 64 hexadecimal characters")
        candidate = snapshot.joinpath(*relative.parts)
        if candidate.is_symlink():
            raise RuntimeError(f"model snapshot may not contain symlinks: {relative_text}")
        if not candidate.is_file():
            raise RuntimeError(f"model snapshot file is missing: {relative_text}")
        _validate_snapshot_member_type(relative, candidate)
        if candidate.stat().st_size != size:
            raise RuntimeError(f"model snapshot size mismatch: {relative_text}")
        if not hmac.compare_digest(_sha256_file(candidate), expected_hash.lower()):
            raise RuntimeError(f"model snapshot SHA-256 mismatch: {relative_text}")
        if relative.suffix.lower() == ".json":
            parsed_json[relative_text] = _read_snapshot_json(candidate, relative=relative)

    required = {"config.json", "preprocessor_config.json", "tokenizer_config.json"}
    if not required.issubset(audited_paths):
        missing = sorted(required - audited_paths)
        raise RuntimeError(f"model snapshot manifest omits required files: {missing}")
    if any(not isinstance(parsed_json.get(path), dict) for path in required):
        raise RuntimeError("required snapshot configuration JSON files must contain objects")
    weight_paths = {path for path in audited_paths if path.endswith(".safetensors")}
    index_paths = {path for path in audited_paths if path.endswith(".safetensors.index.json")}
    if index_paths:
        if index_paths != {"model.safetensors.index.json"}:
            raise RuntimeError("model snapshot must use the canonical safetensors index path")
        index_payload = parsed_json["model.safetensors.index.json"]
        weight_map = index_payload.get("weight_map") if isinstance(index_payload, dict) else None
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(not isinstance(name, str) or not name for name in weight_map)
        ):
            raise RuntimeError("safetensors index must contain a non-empty weight_map")
        referenced_shards = {
            _validated_manifest_relative_path(name).as_posix() for name in weight_map.values()
        }
        if any(not name.endswith(".safetensors") for name in referenced_shards):
            raise RuntimeError("safetensors index may reference only safetensors shards")
        if referenced_shards != weight_paths:
            raise RuntimeError("safetensors index does not cover the exact audited weight shards")
    elif weight_paths != {"model.safetensors"}:
        raise RuntimeError(
            "model snapshot must contain model.safetensors or a complete canonical shard index"
        )
    actual_paths = {
        path.relative_to(snapshot).as_posix() for path in snapshot_nodes if path.is_file()
    }
    if actual_paths != audited_paths:
        missing = sorted(actual_paths - audited_paths)
        unexpected = sorted(audited_paths - actual_paths)
        raise RuntimeError(
            f"model snapshot manifest is not complete: unlisted={missing}, missing={unexpected}"
        )
    return ModelSnapshotEvidence._from_self_consistency_check(
        path=str(snapshot),
        revision=PINNED_MODEL_REVISION,
        manifest_path=str(resolved_manifest),
        manifest_sha256=expected_manifest_sha256.lower(),
        verified_file_count=len(audited_paths),
    )


def revalidate_model_snapshot(snapshot: ModelSnapshotEvidence) -> ModelSnapshotEvidence:
    """Repeat every file check immediately before constructing an execution plan."""

    if not isinstance(snapshot, ModelSnapshotEvidence):
        raise TypeError("snapshot must come from verify_model_snapshot")
    verified = verify_model_snapshot(
        snapshot.path,
        snapshot.manifest_path,
        expected_manifest_sha256=snapshot.manifest_sha256,
    )
    if verified != snapshot:
        raise RuntimeError("model snapshot changed after its evidence was loaded")
    return verified


def probe_local_cuda_devices() -> tuple[str, ...]:
    """Read GPU UUIDs from local ``nvidia-smi`` without importing CUDA libraries."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ()
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    devices: list[str] = []
    for raw_line in completed.stdout.splitlines():
        device = raw_line.strip()
        if _GPU_UUID.fullmatch(device) is not None and device not in devices:
            devices.append(device)
    return tuple(devices)


def require_large_gpu_acknowledgement(config: VLMPreflightConfig) -> VLMPreflightConfig:
    """Stop an unacknowledged run without importing any heavyweight package."""

    if not isinstance(config, VLMPreflightConfig):
        raise TypeError("config must be a VLMPreflightConfig")
    if not config.acknowledge_large_gpu_run:
        raise RuntimeError(
            "large GPU run is not acknowledged; pass an explicit acknowledgement "
            "only after reviewing cost, revisions, and the experiment gate"
        )
    return config


def require_frozen_qwen_stack(config: VLMPreflightConfig) -> VLMPreflightConfig:
    """Require the exact preregistered model and runtime revisions."""

    if not isinstance(config, VLMPreflightConfig):
        raise TypeError("config must be a VLMPreflightConfig")
    expected = {
        "model_name": DEFAULT_MODEL_NAME,
        "model_revision": PINNED_MODEL_REVISION,
        "transformers_revision": PINNED_TRANSFORMERS_REVISION,
        "verl_revision": PINNED_VERL_REVISION,
        "vllm_revision": PINNED_VLLM_REVISION,
    }
    for field, pinned in expected.items():
        actual = getattr(config, field)
        if actual != pinned:
            label = field.replace("_", " ")
            raise RuntimeError(
                f"{label} differs from the frozen preregistration: "
                f"expected {pinned!r}, received {actual!r}"
            )
    return config


def validate_preflight(
    config: VLMPreflightConfig,
    *,
    cuda_available: bool,
    gpu_devices: tuple[str, ...] | list[str],
    require_verl_api_audit: bool = True,
) -> VLMPreflightReport:
    """Validate supplied hardware metadata and return a frozen preflight report.

    Hardware facts are arguments on purpose.  The function remains safe to run
    before importing a training framework, including in metadata-only audits.
    """

    require_large_gpu_acknowledgement(config)
    require_frozen_qwen_stack(config)
    if not isinstance(cuda_available, bool):
        raise TypeError("cuda_available must be a boolean")
    devices = tuple(gpu_devices)
    if not cuda_available or not devices:
        raise RuntimeError("a CUDA GPU is mandatory for the large-model stage")
    if any(not isinstance(device, str) or not device.strip() for device in devices):
        raise ValueError("GPU device names must be non-empty strings")
    if not isinstance(require_verl_api_audit, bool):
        raise TypeError("require_verl_api_audit must be a boolean")
    if require_verl_api_audit and not config.verl_api_audited:
        raise RuntimeError(
            "the veRL API keys have not been audited against the pinned veRL revision"
        )
    return VLMPreflightReport._from_validation(
        model_name=config.model_name,
        model_revision=config.model_revision,
        transformers_revision=config.transformers_revision,
        verl_revision=config.verl_revision,
        vllm_revision=config.vllm_revision,
        gpu_devices=devices,
        verl_api_audited=config.verl_api_audited,
    )


def _build_verl_grpo_config(
    config: VLMPreflightConfig,
    *,
    model_snapshot: ModelSnapshotEvidence,
    learning_rate: float = 1.0e-6,
    mini_batch_size: int = 16,
    rollout_samples: int = 8,
    project_name: str = "compbias",
    experiment_name: str = "qwen25_vl_grpo",
) -> dict[str, Any]:
    """Build only keys audited in the pinned official Qwen2.5-VL example.

    The returned mapping is an execution plan, not an executed training job.
    Callers must run :func:`validate_preflight` first.
    """

    require_large_gpu_acknowledgement(config)
    require_frozen_qwen_stack(config)
    model_snapshot = revalidate_model_snapshot(model_snapshot)
    if model_snapshot.revision != config.model_revision:
        raise RuntimeError("verified model snapshot does not match the configured revision")
    if not config.verl_api_audited:
        raise RuntimeError("veRL API audit is required before generating configuration")
    rate = _validated_positive_float(learning_rate, name="learning_rate")
    batch_size = _validated_positive_int(mini_batch_size, name="mini_batch_size")
    samples = _validated_positive_int(rollout_samples, name="rollout_samples")
    project = _validated_run_name(project_name, name="project_name")
    experiment = _validated_run_name(experiment_name, name="experiment_name")
    return {
        "algorithm": {"adv_estimator": "grpo"},
        "data": {"image_key": "images"},
        "actor_rollout_ref": {
            "model": {"path": model_snapshot.path},
            "actor": {
                "optim": {"lr": rate},
                "ppo_mini_batch_size": batch_size,
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
            },
            "rollout": {"name": "vllm", "n": samples},
        },
        "trainer": {
            "project_name": project,
            "experiment_name": experiment,
            "nnodes": 1,
            "n_gpus_per_node": 1,
            "save_freq": 50,
            "test_freq": 50,
            "total_epochs": 1,
        },
    }


__all__ = [
    "DEFAULT_MODEL_NAME",
    "PINNED_MODEL_REVISION",
    "PINNED_TRANSFORMERS_REVISION",
    "PINNED_VERL_DOCKERFILE_SHA256",
    "PINNED_VERL_REVISION",
    "PINNED_VLLM_REVISION",
    "ModelSnapshotEvidence",
    "VLMPreflightConfig",
    "VLMPreflightReport",
    "probe_local_cuda_devices",
    "require_frozen_qwen_stack",
    "require_large_gpu_acknowledgement",
    "revalidate_model_snapshot",
    "validate_preflight",
    "verify_model_snapshot",
]
