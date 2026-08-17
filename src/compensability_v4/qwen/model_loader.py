"""Pinned server-side model loader guards for v4."""

from __future__ import annotations

import hashlib
from pathlib import Path


MODEL_PATH = "/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_SNAPSHOT_SHA256 = "e104df572eab7267bc2a63c11d70f7c8b1ebf8f85aa835d17e2c2641447bca87"


def require_server_model(path: Path, expected_sha256: str) -> Path:
    if not path.exists():
        raise RuntimeError("model snapshot is unavailable in the local workspace")
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    if digest != expected_sha256:
        raise RuntimeError("model snapshot hash mismatch")
    return path
