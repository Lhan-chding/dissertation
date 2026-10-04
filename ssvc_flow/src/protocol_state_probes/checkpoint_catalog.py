"""Resolve registered historical checkpoints from COMMIT pointers, without Torch.

Availability is a filesystem/committed-identity receipt. Tensor and file-content
verification is performed once by the frozen loader, never implied by this check.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from ..modeling_v3.io import canonical_hash, sha256_file


def map_path(path, path_mappings=None):
    """Explicit whole-prefix remapping; preserve the original binding separately."""
    value = Path(path)
    if not value.is_absolute():
        raise ValueError("Historical paths must be absolute")
    mappings = path_mappings or {}
    if not isinstance(mappings, dict):
        raise ValueError("path_mappings must map absolute directory prefixes")
    for old, new in sorted(mappings.items(), key=lambda item: len(item[0]), reverse=True):
        old, new = Path(old), Path(new)
        if not old.is_absolute() or not new.is_absolute():
            raise ValueError("Path mapping prefixes must be absolute")
        try:
            return new / value.relative_to(old)
        except ValueError:
            continue
    return value


def read_bound_json(binding, path_mappings=None):
    path = map_path(binding["path"], path_mappings)
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("Bound JSON file digest mismatch: " + str(path))
    return json.loads(path.read_text())


def _json_binding(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_manifest_identity(record, manifest, commit):
    """Bind source/branch semantics, including R0 ancestry and branch origin."""
    if manifest.get("kind") != "PROSPECTIVE_TRAINING" or manifest.get("fixture") is not False:
        raise ValueError("Checkpoint must come from real prospective training")
    identity = manifest["identity"]
    if identity.get("lineage_id") != record["lineage"]:
        raise ValueError("Checkpoint lineage differs from registered lineage")
    expected_step = record["source_step"]
    if record["kind"] == "source":
        if identity.get("role") != "source" or identity.get("source_recipe") != "R0":
            raise ValueError("Source checkpoint must be the registered R0 lineage")
        if manifest.get("recipe") != "R0":
            raise ValueError("Source reward recipe differs")
    elif record["kind"] == "branch":
        expected_step = record["continuation_steps"]
        expected = {
            "origin_id": f"{record['lineage']}_t{record['source_step']}",
            "recipe_id": record["recipe"],
            "repeat": record["repeat"],
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError("Branch origin, recipe or repeat differs from registration")
        if manifest.get("recipe") != record["recipe"]:
            raise ValueError("Branch reward recipe differs")
    else:
        raise ValueError("Unknown registered checkpoint kind")
    if commit.get("stop") != expected_step or manifest.get("steps", -1) < expected_step:
        raise ValueError("Checkpoint step differs from registration")
    manifest_hash = canonical_hash(manifest)
    expected_identity = {"manifest_hash": manifest_hash, "step": expected_step}
    if commit.get("manifest_hash") != manifest_hash:
        raise ValueError("COMMIT manifest digest mismatch")
    checkpoint = commit["checkpoint"]
    if checkpoint.get("identity") != expected_identity:
        raise ValueError("Checkpoint identity differs from COMMIT and manifest")
    for key in ("sha256", "state_hash"):
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoint.get(key, "")):
            raise ValueError("Checkpoint requires committed " + key)
    if manifest.get("runtime_identity", {}).get("execution_kind") != "REAL_CUDA_MODEL":
        raise ValueError("Historical runtime was not REAL_CUDA_MODEL")


def _check_candidate(record, candidate, mappings):
    exported = map_path(candidate["path"], mappings)
    if exported.name != "state.pt" or not exported.parent.name.startswith("attempt_"):
        raise ValueError("Expected historical segment attempt/state.pt path")
    segment = exported.parents[1]
    if segment.parent.name != "segments":
        raise ValueError("Expected a committed historical segment")
    commit_path, manifest_path = segment / "COMMIT.json", segment.parents[1] / "MANIFEST.json"
    commit, manifest = json.loads(commit_path.read_text()), json.loads(manifest_path.read_text())
    validate_manifest_identity(record, manifest, commit)
    original_checkpoint = copy.deepcopy(commit["checkpoint"])
    checkpoint_path = map_path(original_checkpoint["path"], mappings)
    if checkpoint_path.resolve().parent.parent != segment.resolve():
        raise ValueError("COMMIT points outside its own segment")
    if checkpoint_path.name != "state.pt":
        raise ValueError("COMMIT does not point to a complete state.pt")
    size = checkpoint_path.stat().st_size
    if not checkpoint_path.is_file() or not os.access(checkpoint_path, os.R_OK) or size <= 0:
        raise OSError("Committed checkpoint is unreadable or empty")
    # The exported attempt is only a locator; COMMIT is authoritative if a
    # later committed attempt was retained for the same registered segment.
    if checkpoint_path.resolve() == exported.resolve() and candidate.get("bytes") != size:
        raise ValueError("Checkpoint size differs from exported identity")
    with checkpoint_path.open("rb") as stream:
        stream.read(1)
    return {
        "status": "AVAILABLE",
        "checkpoint": {**original_checkpoint, "path": str(checkpoint_path)},
        "checkpoint_original_binding": original_checkpoint,
        "checkpoint_bytes": size,
        "commit": _json_binding(commit_path),
        "manifest": _json_binding(manifest_path),
        "manifest_identity": manifest["identity"],
        "runtime_identity": manifest["runtime_identity"],
        "historical_config_hash": manifest.get("config_hash"),
        "checkpoint_content_verified": False,
        "authoritative_attempt_matches_export": checkpoint_path.resolve() == exported.resolve(),
    }


def check_checkpoints(checkpoint_manifest, path_mappings=None):
    """CPU-only availability, preserving every missing/invalid registered arm."""
    records = (
        json.loads(Path(checkpoint_manifest).read_text())
        if isinstance(checkpoint_manifest, (str, Path))
        else copy.deepcopy(checkpoint_manifest)
    )
    if not isinstance(records, list) or len({row["id"] for row in records}) != len(records):
        raise ValueError("Checkpoint manifest must have unique registered IDs")
    mappings = dict(path_mappings or {})
    results = []
    for record in records:
        base = {
            "id": record["id"],
            "checkpoint_id": record["id"],
            "source": record,
            "path_mappings": mappings,
            "allow_retraining_if_missing": False,
        }
        valid, errors = [], []
        for candidate in record.get("export_time_candidates", []):
            try:
                valid.append(_check_candidate(record, candidate, mappings))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(
                    {
                        "path": candidate.get("path"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        signatures = {canonical_hash(row["checkpoint"]) for row in valid}
        if len(signatures) > 1:
            result = {
                "status": "INVALID",
                "errors": [*errors, {"error": "Conflicting committed candidates"}],
            }
        elif valid:
            result = {**valid[0], "candidate_errors": errors}
        else:
            missing = not errors or all(
                error["error_type"] in ("FileNotFoundError", "PermissionError", "OSError")
                for error in errors
            )
            result = {"status": "MISSING" if missing else "INVALID", "errors": errors}
        results.append({**base, **result})
    available = sum(row["status"] == "AVAILABLE" for row in results)
    return {
        "schema": "protocol-probe-checkpoint-availability-v1",
        "status": "ALL_AVAILABLE" if available == len(results) else "PARTIAL_AVAILABILITY",
        "checkpoints": results,
        "available_count": available,
        "registered_count": len(results),
        "path_mappings": mappings,
        "model_loaded": False,
        "checkpoint_tensors_loaded": False,
        "new_training_allowed": False,
    }
