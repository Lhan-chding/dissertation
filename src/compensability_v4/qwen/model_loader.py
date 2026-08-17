"""Fail-closed, offline loader for the frozen Qwen2.5-VL snapshot.

Importing this module is intentionally cheap. ``torch`` and ``transformers``
are imported only after the on-server snapshot has passed the provenance gate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from compbias.gpu_pilot.preflight import model_snapshot_sha256

MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_SNAPSHOT_SHA256 = "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"


def snapshot_sha256(path: Path) -> str:
    """Use the repository's strict model-tree validator and exact hash."""

    return model_snapshot_sha256(path)


def require_server_model(
    path: Path | str = Path(MODEL_PATH),
    expected_sha256: str = MODEL_SNAPSHOT_SHA256,
) -> Path:
    """Return the exact pinned server directory, or fail before model import."""

    candidate = Path(path)
    if expected_sha256 != MODEL_SNAPSHOT_SHA256:
        raise RuntimeError("model snapshot expectation differs from the frozen v4 hash")
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(f"model snapshot is unavailable: {candidate}")
    if candidate.absolute() != Path(MODEL_PATH):
        raise RuntimeError(
            f"model snapshot path must be the frozen server path {MODEL_PATH}, got {candidate}"
        )
    actual = snapshot_sha256(candidate)
    if actual != expected_sha256:
        raise RuntimeError(
            f"model snapshot hash mismatch: expected {expected_sha256}, observed {actual}"
        )
    return candidate


def load_pinned_qwen(
    *,
    model_path: Path | str = Path(MODEL_PATH),
    device_map: str | dict[str, Any] = "cuda:0",
    torch_dtype: object | None = None,
    model_class: Any | None = None,
    processor_class: Any | None = None,
) -> tuple[object, object]:
    """Load Qwen only from verified local bytes without repository code."""

    verified = require_server_model(model_path, MODEL_SNAPSHOT_SHA256)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if model_class is None or processor_class is None or torch_dtype is None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_class = model_class or Qwen2_5_VLForConditionalGeneration
        processor_class = processor_class or AutoProcessor
        torch_dtype = torch_dtype or torch.bfloat16
    load_options = {"local_files_only": True, "trust_remote_code": False}
    model = model_class.from_pretrained(
        str(verified),
        device_map=device_map,
        torch_dtype=torch_dtype,
        **load_options,
    )
    processor = processor_class.from_pretrained(str(verified), **load_options)
    eval_mode = getattr(model, "eval", None)
    if not callable(eval_mode):
        raise RuntimeError("loaded model does not expose eval()")
    eval_mode()
    return model, processor


__all__ = [
    "MODEL_PATH",
    "MODEL_SNAPSHOT_SHA256",
    "load_pinned_qwen",
    "require_server_model",
    "snapshot_sha256",
]
