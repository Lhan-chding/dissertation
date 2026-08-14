"""Fail-fast server audit for the RTX 4090 Qwen pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from compbias.io.strict_json import load_strict_json_mapping

from .config import PilotPaths, load_pilot_paths
from .safe_io import atomic_write_json_text

_MINIMUM_VRAM_GIB = 45.0
_MINIMUM_FREE_DISK_GIB = 150.0
_MODEL_FILES = (
    "chat_template.json",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
)


@dataclass(frozen=True, slots=True)
class HardwareSnapshot:
    cuda_available: bool
    device_name: str
    total_vram_gib: float
    bf16_supported: bool
    torch_version: str
    torch_cuda_runtime: str


def probe_hardware() -> HardwareSnapshot:
    import torch

    available = bool(torch.cuda.is_available())
    if not available:
        return HardwareSnapshot(
            cuda_available=False,
            device_name="",
            total_vram_gib=0.0,
            bf16_supported=False,
            torch_version=str(torch.__version__),
            torch_cuda_runtime=str(torch.version.cuda),
        )
    properties = torch.cuda.get_device_properties(0)
    return HardwareSnapshot(
        cuda_available=True,
        device_name=str(properties.name),
        total_vram_gib=float(properties.total_memory / 1024**3),
        bf16_supported=bool(torch.cuda.is_bf16_supported()),
        torch_version=str(torch.__version__),
        torch_cuda_runtime=str(torch.version.cuda),
    )


def _ensure_writable_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"storage path is not a safe directory: {path}")
    descriptor, name = tempfile.mkstemp(dir=path, prefix=".compbias-write-test-")
    os.close(descriptor)
    Path(name).unlink(missing_ok=True)


def _validate_model_tree(model_path: Path) -> None:
    if model_path.is_symlink() or not model_path.is_dir():
        raise RuntimeError(f"model path is not a regular directory: {model_path}")
    for name in _MODEL_FILES:
        candidate = model_path / name
        if candidate.is_symlink() or not candidate.is_file() or not os.access(candidate, os.R_OK):
            raise RuntimeError(f"required readable model file is missing: {candidate}")
    index = load_strict_json_mapping(
        model_path / "model.safetensors.index.json",
        label="model safetensors index",
        max_bytes=1024 * 1024,
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise RuntimeError("model safetensors index has an invalid weight_map")
    expected_shards = {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    if not weight_map or set(weight_map.values()) != expected_shards:
        raise RuntimeError("model safetensors index does not reference the exact two shards")


def model_snapshot_sha256(model_path: Path) -> str:
    """Hash every model byte that the offline pilot is allowed to load."""

    _validate_model_tree(model_path)
    digest = hashlib.sha256()
    file_count = 0
    for directory, directory_names, file_names in os.walk(model_path, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current = Path(directory)
        for name in directory_names:
            child = current / name
            if child.is_symlink():
                raise RuntimeError(f"model snapshot contains a symlink directory: {child}")
        for name in file_names:
            path = current / name
            metadata = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"model snapshot contains a non-regular file: {path}")
            file_count += 1
            if file_count > 10_000:
                raise RuntimeError("model snapshot contains more than 10000 files")
            relative = path.relative_to(model_path).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\n")
    if file_count < len(_MODEL_FILES):
        raise RuntimeError("model snapshot file closure is incomplete")
    return digest.hexdigest()


def audit_server(
    paths: PilotPaths,
    *,
    hardware: HardwareSnapshot | None = None,
    free_disk_gib: float | None = None,
) -> dict[str, object]:
    snapshot = probe_hardware() if hardware is None else hardware
    if not snapshot.cuda_available:
        raise RuntimeError("CUDA GPU is unavailable")
    if snapshot.total_vram_gib < _MINIMUM_VRAM_GIB:
        raise RuntimeError(
            f"GPU VRAM {snapshot.total_vram_gib:.2f} GiB is below {_MINIMUM_VRAM_GIB:.0f} GiB"
        )
    if not snapshot.bf16_supported:
        raise RuntimeError("GPU does not report BF16 support")
    if not snapshot.torch_version.startswith("2.8.0"):
        raise RuntimeError("the pilot requires the validated torch 2.8.0 runtime")
    if snapshot.torch_cuda_runtime != "12.8":
        raise RuntimeError("the pilot requires PyTorch CUDA runtime 12.8")

    _validate_model_tree(paths.model_path)
    for path in (
        paths.data,
        paths.outputs,
        paths.checkpoints,
        paths.trajectories,
        paths.cache,
    ):
        _ensure_writable_directory(path)
    available = (
        float(shutil.disk_usage(paths.data).free / 1024**3)
        if free_disk_gib is None
        else float(free_disk_gib)
    )
    if available < _MINIMUM_FREE_DISK_GIB:
        raise RuntimeError(
            f"data disk has {available:.2f} GiB free; at least "
            f"{_MINIMUM_FREE_DISK_GIB:.0f} GiB required"
        )
    return {
        "schema_version": 1,
        "artifact_type": "compbias_gpu_pilot_preflight",
        "ready": True,
        "large_gpu_started": False,
        "hardware": asdict(snapshot),
        "free_disk_gib": available,
        "model_path": str(paths.model_path),
        "model_snapshot_sha256": model_snapshot_sha256(paths.model_path),
        "storage": paths.to_mapping(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        paths = load_pilot_paths(args.paths)
        report = audit_server(paths)
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            atomic_write_json_text(paths.outputs, args.output, payload)
        print(payload, end="")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
