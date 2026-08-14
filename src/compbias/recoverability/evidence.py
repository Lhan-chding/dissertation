"""External anchors for the failed v0.3 pilot and its frozen source protocol."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RELATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,255}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NegativePilotRecord:
    status: str
    server_revision_observed: str
    model_snapshot_sha256: str
    records: int
    answer_accuracy: float
    parse_rate: float
    natural_perception_error_rate: float
    error_counts: Mapping[str, int]
    gate_passed: bool
    calibration_exit: int
    original_pilot_a: str
    original_pilot_b: str
    required_server_artifacts: tuple[str, ...]


def load_negative_pilot_record(path: Path) -> NegativePilotRecord:
    mapping = load_yaml_mapping(path, label="v0.3 negative pilot record")
    fields = {
        "status",
        "server_revision_observed",
        "model_snapshot_sha256",
        "records",
        "answer_accuracy",
        "parse_rate",
        "natural_perception_error_rate",
        "error_counts",
        "gate_passed",
        "calibration_exit",
        "original_pilot_a",
        "original_pilot_b",
        "required_server_artifacts",
        "dataset_manifest_sha256",
        "dataset_records_sha256",
        "dataset_images_sha256",
        "counterfactual_sha256",
    }
    reject_unknown_fields(mapping, fields, label="v0.3 negative pilot record")
    if set(mapping) != fields:
        raise ValueError("v0.3 negative pilot record is incomplete")
    if mapping["status"] != "final_failed_preregistered_pilot":
        raise ValueError("v0.3 negative pilot status cannot be changed")
    if mapping["gate_passed"] is not False or mapping["calibration_exit"] != 3:
        raise ValueError("v0.3 negative pilot must remain failed")
    if (
        mapping["original_pilot_a"] != "terminated_not_run"
        or mapping["original_pilot_b"] != "terminated_not_run"
    ):
        raise ValueError("original Pilot A/B must remain terminated_not_run")
    if mapping["records"] != 200 or type(mapping["records"]) is not int:
        raise ValueError("v0.3 records must equal 200")
    counts_value = mapping["error_counts"]
    if not isinstance(counts_value, Mapping) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in counts_value.items()
    ):
        raise ValueError("v0.3 error_counts are invalid")
    counts = dict(sorted(counts_value.items()))
    if sum(counts.values()) != 200:
        raise ValueError("v0.3 error_counts must sum to 200")
    for field in (
        "model_snapshot_sha256",
        "dataset_manifest_sha256",
        "dataset_records_sha256",
        "dataset_images_sha256",
        "counterfactual_sha256",
    ):
        if not isinstance(mapping[field], str) or _SHA256.fullmatch(mapping[field]) is None:
            raise ValueError(f"{field} must be a SHA-256 digest")
    artifacts = mapping["required_server_artifacts"]
    if not isinstance(artifacts, list) or any(
        not isinstance(item, str) or "/" in item or item in {".", ".."} for item in artifacts
    ):
        raise ValueError("required_server_artifacts must be safe basenames")
    return NegativePilotRecord(
        status=mapping["status"],
        server_revision_observed=mapping["server_revision_observed"],
        model_snapshot_sha256=mapping["model_snapshot_sha256"],
        records=200,
        answer_accuracy=float(mapping["answer_accuracy"]),
        parse_rate=float(mapping["parse_rate"]),
        natural_perception_error_rate=float(mapping["natural_perception_error_rate"]),
        error_counts=MappingProxyType(counts),
        gate_passed=False,
        calibration_exit=3,
        original_pilot_a="terminated_not_run",
        original_pilot_b="terminated_not_run",
        required_server_artifacts=tuple(artifacts),
    )


@dataclass(frozen=True, slots=True)
class LockedFile:
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ProtocolLockResult:
    verified: bool
    files: tuple[LockedFile, ...]


def verify_protocol_lock(path: Path, *, repository_root: Path) -> ProtocolLockResult:
    mapping = load_yaml_mapping(path, label="recoverability protocol lock")
    reject_unknown_fields(mapping, {"schema_version", "files"}, label="protocol lock")
    if mapping.get("schema_version") != 1 or set(mapping) != {"schema_version", "files"}:
        raise ValueError("protocol lock schema is invalid")
    raw_files = mapping["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("protocol lock files must be a non-empty list")
    root = repository_root.resolve()
    locked: list[LockedFile] = []
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError("each protocol lock entry must contain path and sha256")
        relative = item["path"]
        expected = item["sha256"]
        if not isinstance(relative, str) or _RELATIVE.fullmatch(relative) is None:
            raise ValueError("protocol lock path is invalid")
        candidate = root / relative
        resolved = candidate.resolve()
        if root not in resolved.parents or candidate.is_symlink() or not candidate.is_file():
            raise ValueError("protocol lock path must be a regular repository file")
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            raise ValueError("protocol lock digest is invalid")
        actual = _sha256(candidate)
        if actual != expected:
            raise ValueError(f"protocol lock mismatch for {relative}")
        locked.append(LockedFile(relative_path=relative, sha256=actual))
    if len({item.relative_path for item in locked}) != len(locked):
        raise ValueError("protocol lock contains duplicate paths")
    return ProtocolLockResult(verified=True, files=tuple(locked))
