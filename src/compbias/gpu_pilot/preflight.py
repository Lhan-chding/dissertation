"""Fail-fast server audit for the RTX 4090 Qwen pilot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import PilotPaths, load_pilot_paths

_MINIMUM_VRAM_GIB = 45.0
_MINIMUM_FREE_DISK_GIB = 150.0
_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
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
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
