"""Independent CPU element comparisons for this run's trusted ENGINE checkpoints.

Pickled checkpoints are accepted only from the fixed local ENGINE directories,
after their bytes match a committed SHA-256. This is not an untrusted-pickle
reader. The caller must first establish the run's freeze/source authorization.
No model, runtime, training module, or state-hash comparator is imported here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path

from .contract import PLAN_ID

FIELDS = frozenset(
    {
        "plan_id",
        "run_identity",
        "committed_logical_step",
        "parameters",
        "optimizer",
        "scheduler",
        "rng",
        "reference",
        "reference_hash",
        "input_stream_hash",
        "sampling_hash",
        "cursor",
        "diagnostics_hash",
        "token_path_hash",
    }
)
COUNT_KEYS = (
    "tensors",
    "tensor_elements",
    "numpy_arrays",
    "numpy_elements",
    "numpy_scalars",
    "scalars",
    "containers",
)


def _path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise PermissionError("Checkpoint path must remain within its registered directory")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise PermissionError("Symlink in checkpoint evidence path")
    if not current.resolve(strict=True).is_relative_to(root):
        raise PermissionError("Checkpoint path escapes its registered directory")
    if not current.is_file():
        raise PermissionError("Checkpoint evidence must be a regular file")
    return current


def _sha(handle):
    result = hashlib.sha256()
    for block in iter(lambda: handle.read(8 << 20), b""):
        result.update(block)
    return result.hexdigest()


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in checkpoint evidence")
        result[key] = value
    return result


def _json(root, relative, artifacts):
    path = _path(root, relative)
    payload = path.read_bytes()
    artifacts[str(path.relative_to(root))] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload, object_pairs_hook=_json_pairs)


def _equal(left, right, label, counts, torch, np):
    """Compare every supported leaf by value; hashes never establish equality."""
    if type(left) is not type(right):
        raise ValueError("Checkpoint type mismatch at " + label)
    if isinstance(left, torch.Tensor):
        if (
            left.device.type != "cpu"
            or right.device.type != "cpu"
            or left.dtype != right.dtype
            or left.shape != right.shape
            or left.layout != right.layout
            or not torch.equal(left, right)
        ):
            raise ValueError("Checkpoint tensor mismatch at " + label)
        counts["tensors"] += 1
        counts["tensor_elements"] += left.numel()
    elif isinstance(left, np.ndarray):
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left, right)
        ):
            raise ValueError("Checkpoint NumPy array mismatch at " + label)
        counts["numpy_arrays"] += 1
        counts["numpy_elements"] += left.size
    elif isinstance(left, np.generic):
        if left.dtype != right.dtype or not np.array_equal(left, right):
            raise ValueError("Checkpoint NumPy scalar mismatch at " + label)
        counts["numpy_scalars"] += 1
    elif type(left) in (dict, OrderedDict):
        if left.keys() != right.keys():
            raise ValueError("Checkpoint container keys mismatch at " + label)
        # Python considers True and 1 equal as dict keys; checkpoint types must agree too.
        right_keys = {key: key for key in right}
        if any(type(key) is not type(right_keys[key]) for key in left):
            raise ValueError("Checkpoint container key types mismatch at " + label)
        counts["containers"] += 1
        for key in left:
            _equal(left[key], right[key], f"{label}[{key!r}]", counts, torch, np)
    elif type(left) in (tuple, list):
        if len(left) != len(right):
            raise ValueError("Checkpoint container length mismatch at " + label)
        counts["containers"] += 1
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _equal(a, b, f"{label}[{index}]", counts, torch, np)
    elif type(left) in (str, int, float, bool, bytes, complex, type(None)):
        if left != right:
            raise ValueError("Checkpoint scalar mismatch at " + label)
        counts["scalars"] += 1
    else:
        raise TypeError("Unsupported checkpoint value type at " + label)


def verify_engine_checkpoints(root, mode):
    """Compare full step-2/4 states in a fresh CPU process and return stable evidence."""
    import numpy as np
    import torch

    if mode not in {"natural", "stress"}:
        raise ValueError("Unregistered ENGINE checkpoint mode")
    if torch.cuda.is_initialized():
        raise RuntimeError("Independent checkpoint recount requires CUDA uninitialized")
    root = Path(root).resolve(strict=True)
    artifacts = {}
    freeze = _json(root, "manifests/F2_FREEZE.json", artifacts)
    if freeze.get("plan_id") != PLAN_ID:
        raise PermissionError("Checkpoint recount freeze belongs to a different plan")
    freeze_sha = artifacts["manifests/F2_FREEZE.json"]
    base = f"engineering/{mode}"
    manifests, markers, commits = {}, {}, {}
    expected_run = "ENGINE_F2_" + mode.upper()
    for track in ("continuous", "split"):
        manifest = _json(root, f"{base}/{track}/RUN_MANIFEST.json", artifacts)
        identity = manifest["run_identity"]
        if (
            manifest.get("plan_id") != PLAN_ID
            or identity.get("run_id") != expected_run
            or identity.get("phase") != "ENGINE_F2"
            or identity.get("stress") is not (mode == "stress")
        ):
            raise PermissionError("Checkpoint recount run identity differs")
        manifests[track] = manifest
        segments = ("continuous",) if track == "continuous" else ("first", "resume")
        for segment in segments:
            marker = _json(root, f"{base}/{track}/{segment.upper()}_COMPLETE.json", artifacts)
            if (
                marker.get("freeze_sha256") != freeze_sha
                or marker.get("mode") != mode
                or marker.get("segment") != segment
                or marker.get("run_identity") != identity
                or marker.get("final_step") != (2 if segment == "first" else 4)
            ):
                raise PermissionError("Checkpoint segment is not bound to this frozen run")
            markers[segment] = marker
        for step in (2, 4):
            commit = _json(root, f"{base}/{track}/checkpoints/commit-{step:02d}.json", artifacts)
            if (
                type(commit.get("step")) is not int
                or commit["step"] != step
                or not re.fullmatch(r"[0-9a-f]{64}", str(commit.get("sha256", "")))
                or set(commit.get("field_hashes", {})) != FIELDS
            ):
                raise PermissionError("Invalid registered checkpoint commit")
            # The production writer publishes a filename, never a relative traversal.
            name = commit["path"]
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt"):
                raise PermissionError("Checkpoint commit must name one local .pt file")
            commits[track, step] = commit
        marker = markers["continuous" if track == "continuous" else "resume"]
        if marker["checkpoint"] != commits[track, 4]:
            raise PermissionError("Final segment checkpoint binding differs")
        if track == "split" and markers["first"]["checkpoint"] != commits[track, 2]:
            raise PermissionError("First segment checkpoint binding differs")

    comparisons = []
    totals = dict.fromkeys(COUNT_KEYS, 0)
    for step in (2, 4):
        states, files = {}, {}
        for track in ("continuous", "split"):
            commit = commits[track, step]
            relative = f"{base}/{track}/checkpoints/{commit['path']}"
            path = _path(root, relative)
            with path.open("rb") as handle:
                actual = _sha(handle)
                if actual != commit["sha256"]:
                    raise PermissionError("Checkpoint bytes differ from committed SHA-256")
                handle.seek(0)
                # These are hash-bound local files from the caller's trusted run only.
                state = torch.load(handle, map_location="cpu", weights_only=False)
                handle.seek(0)
                if _sha(handle) != actual:
                    raise PermissionError("Checkpoint changed during CPU recount")
            artifacts[relative] = actual
            manifest = manifests[track]
            if (
                type(state) is not dict
                or set(state) != FIELDS
                or state["plan_id"] != PLAN_ID
                or type(state["committed_logical_step"]) is not int
                or state["committed_logical_step"] != step
                or state["run_identity"] != manifest["run_identity"]
                or state["input_stream_hash"] != manifest["stream_hash"]
                or state["sampling_hash"] != manifest["sampling_hash"]
            ):
                raise PermissionError("Loaded checkpoint differs from registered run identity")
            if not state["parameters"] or not state["optimizer"]["state"] or not state["reference"]:
                raise ValueError("Checkpoint lacks actual policy, Adam or reference state")
            states[track] = state
            files[track] = {"path": relative, "sha256": actual}
        fields = {}
        for field in sorted(FIELDS):
            counts = dict.fromkeys(COUNT_KEYS, 0)
            _equal(states["continuous"][field], states["split"][field], field, counts, torch, np)
            fields[field] = {"equal": True, "counts": counts}
            for key, value in counts.items():
                totals[key] += value
        comparisons.append({"step": step, "files": files, "fields": fields})
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA unexpectedly initialized during independent checkpoint recount")
    # Recheck manifests/markers and files after comparison, before reporting a stable receipt.
    for relative, expected in artifacts.items():
        with _path(root, relative).open("rb") as handle:
            if _sha(handle) != expected:
                raise PermissionError("Checkpoint evidence changed during independent recount")
    return {
        "schema_version": 1,
        "status": "VERIFIED_ELEMENTWISE_EQUAL",
        "plan_id": PLAN_ID,
        "mode": mode,
        "freeze_sha256": freeze_sha,
        "cuda_initialized": False,
        "comparison_method": "torch.equal / numpy.array_equal / typed recursive scalar equality",
        "comparisons": comparisons,
        "total_counts": totals,
        "artifact_hashes": dict(sorted(artifacts.items())),
    }


def publish_recount(root, result):
    """Atomically publish once; an identical completed CPU recount is safely reusable."""
    root = Path(root).resolve(strict=True)
    mode = result["mode"]
    if mode not in {"natural", "stress"}:
        raise ValueError("Unregistered ENGINE checkpoint mode")
    directory = _path(root, f"engineering/{mode}/continuous/RUN_MANIFEST.json").parent.parent
    target = directory / "INDEPENDENT_CHECKPOINT_RECOUNT.json"
    payload = (
        json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    fd, temporary = tempfile.mkstemp(prefix=".pending-recount-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if _path(root, target.relative_to(root)).read_bytes() != payload:
                raise PermissionError("Independent recount receipt already differs") from None
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)
    return target
