"""External anchors for the failed v0.3 pilot and its frozen source protocol."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RELATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,255}\Z")
SERVER_PACKAGE_LOCK_PATH = "configs/recoverability/server_package_lock_v1.yaml"
SERVER_PACKAGE_PATHS = frozenset(
    {
        "configs/data/cva_chart_pilot_v0_3.yaml",
        "configs/recoverability/recoverability_v1.yaml",
        "configs/recoverability/power_plan_v1.json",
        "configs/recoverability/server_runtime_v1.yaml",
        "configs/recoverability/v0_3_negative_pilot.yaml",
        "experiments/recoverability_v1/00_preflight.py",
        "experiments/recoverability_v1/02_capture_v03_evidence.py",
        "experiments/recoverability_v1/03_bridge.py",
        "requirements-gpu.lock.txt",
        "src/compbias/gpu_pilot/chart_data.py",
        "src/compbias/gpu_pilot/collection.py",
        "src/compbias/gpu_pilot/config.py",
        "src/compbias/gpu_pilot/execution_gate.py",
        "src/compbias/gpu_pilot/preflight.py",
        "src/compbias/gpu_pilot/qwen_smoke.py",
        "src/compbias/gpu_pilot/safe_io.py",
        "src/compbias/gpu_pilot/structured_generation.py",
        "src/compbias/gpu_pilot/taxonomy.py",
        "src/compbias/io/strict_json.py",
        "src/compbias/io/yaml_config.py",
        "src/compbias/models/structured_parser.py",
        "src/compbias/recoverability/bridge.py",
        "src/compbias/recoverability/config.py",
        "src/compbias/recoverability/dsl/executor.py",
        "src/compbias/recoverability/dsl/parser.py",
        "src/compbias/recoverability/dsl/schema.py",
        "src/compbias/recoverability/evidence.py",
        "src/compbias/recoverability/evidence_capture.py",
        "src/compbias/recoverability/preflight.py",
    }
)


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
    dataset_manifest_sha256: str
    dataset_records_sha256: str
    dataset_images_sha256: str
    counterfactual_sha256: str
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
        dataset_manifest_sha256=mapping["dataset_manifest_sha256"],
        dataset_records_sha256=mapping["dataset_records_sha256"],
        dataset_images_sha256=mapping["dataset_images_sha256"],
        counterfactual_sha256=mapping["counterfactual_sha256"],
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


@dataclass(frozen=True, slots=True)
class PowerArtifactResult:
    verified: bool
    sha256: str
    required_eligible_scenes: int
    registered_intake_scenes: int
    independent_unit: str
    forks_per_arm: int
    target_power: float
    scenarios: tuple[str, ...]


def verify_power_artifact(path: Path, *, expected_sha256: str) -> PowerArtifactResult:
    """Verify the externally anchored power artifact and its closed schema."""

    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected power artifact SHA-256 is invalid")
    if path.is_symlink() or not path.is_file():
        raise ValueError("power artifact must be a regular file")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError("power artifact SHA-256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("power artifact must be valid UTF-8 JSON") from error
    fields = {
        "schema_version",
        "artifact_type",
        "status",
        "independent_unit",
        "forks_per_arm",
        "target_power",
        "alpha",
        "equivalence_margin",
        "repetitions",
        "seed",
        "registered_intake_scenes",
        "required_eligible_scenes",
        "eligibility_rate_lower",
        "required_intake_scenes",
        "feasible",
        "family_quotas",
        "scenarios",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("power artifact schema is invalid")
    fixed = {
        "schema_version": 1,
        "artifact_type": "recoverability_v1_phase_c_power",
        "status": "PREREGISTERED_NOT_RUN",
        "independent_unit": "semantic_scene",
        "forks_per_arm": 8,
        "target_power": 0.90,
        "alpha": 0.05,
        "equivalence_margin": 0.02,
        "repetitions": 2_000,
        "seed": 2026081605,
        "registered_intake_scenes": 8000,
        "required_eligible_scenes": 1066,
        "feasible": True,
        "family_quotas": {
            "cross_series": 400,
            "duplicate_encoding": 266,
            "trend": 400,
        },
    }
    if any(payload[key] != value for key, value in fixed.items()):
        raise ValueError("power artifact differs from the preregistration")
    for key in ("eligibility_rate_lower", "required_intake_scenes"):
        if isinstance(payload[key], bool) or not isinstance(payload[key], (int, float)):
            raise ValueError(f"power artifact {key} is invalid")
    scenarios = payload["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("power artifact scenarios must be a non-empty list")
    names: list[str] = []
    for item in scenarios:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "test",
            "discordance",
            "scene_icc",
            "true_effect",
            "alpha",
            "baseline_rate",
            "curve",
        }:
            raise ValueError("power scenario schema is invalid")
        name = item["name"]
        if not isinstance(name, str) or name in names:
            raise ValueError("power scenario names must be unique strings")
        if item["test"] not in {"one_sided_positive", "paired_tost"}:
            raise ValueError("power scenario test is invalid")
        for key in ("discordance", "scene_icc", "true_effect", "alpha", "baseline_rate"):
            value = item[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("power scenario numeric setting is invalid")
        curve = item["curve"]
        scene_grid = [point.get("scenes") for point in curve] if isinstance(curve, list) else []
        if scene_grid != [400, 600, 800, 1066]:
            raise ValueError("power scenario curve uses an invalid scene grid")
        if any(
            not isinstance(point, dict)
            or set(point) != {"estimated_power", "scenes"}
            or isinstance(point["estimated_power"], bool)
            or not 0 <= point["estimated_power"] <= 1
            for point in curve
        ):
            raise ValueError("power scenario curve is invalid")
        names.append(name)
    return PowerArtifactResult(
        verified=True,
        sha256=actual,
        required_eligible_scenes=1066,
        registered_intake_scenes=8000,
        independent_unit="semantic_scene",
        forks_per_arm=8,
        target_power=0.90,
        scenarios=tuple(names),
    )


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


def verify_server_package_lock(path: Path, *, repository_root: Path) -> ProtocolLockResult:
    """Require the canonical, closed server import package rather than a caller subset."""

    root = repository_root.resolve()
    expected_path = root / SERVER_PACKAGE_LOCK_PATH
    if path.resolve() != expected_path or path.is_symlink():
        raise ValueError("server package lock must use the canonical repository path")
    result = verify_protocol_lock(path, repository_root=root)
    observed = frozenset(item.relative_path for item in result.files)
    if observed != SERVER_PACKAGE_PATHS:
        missing = sorted(SERVER_PACKAGE_PATHS - observed)
        extra = sorted(observed - SERVER_PACKAGE_PATHS)
        raise ValueError(f"server package lock closure mismatch; missing={missing}, extra={extra}")
    return result
