"""Immutable anchor for the server-generated measurement qualification dataset."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

from compbias.io.strict_json import load_strict_json_mapping
from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .measurement_qualification import MeasurementQualificationConfig
from .measurement_qualification_data import verify_measurement_qualification_dataset

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True, slots=True)
class MeasurementQualificationDataAnchor:
    schema_version: int
    status: str
    generation_code_commit: str
    generation_server_package_lock_sha256: str
    attempt_marker_sha256: str
    manifest_sha256: str
    records_sha256: str
    images_sha256: str
    console_sha256: str
    dataset_id: str
    seed: int
    source_dataset_id: str
    source_dataset_records_sha256: str
    source_stage2_v2_external_evidence_sha256: str
    records: int
    per_stratum: int
    model_calls: int
    hypothesis_tested: bool
    confirmatory_execution_authorized: bool
    training_invoked: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _exact_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative exact integer")
    return value


def _expected_anchor() -> MeasurementQualificationDataAnchor:
    return MeasurementQualificationDataAnchor(
        schema_version=1,
        status="FINAL_GENERATED_DATASET_DO_NOT_RERUN",
        generation_code_commit="93bd5f509728a0465e4891672b3dbff0cfd2f568",
        generation_server_package_lock_sha256=(
            "25808fffdf62981163550084c36c4b37428fada7c85bc8b3a5e286b4bc75ec4c"
        ),
        attempt_marker_sha256=("29667a9866e2f969aa05a10737abec1fe59f657076e603d8be6328802e5cfd97"),
        manifest_sha256=("6c85db5a4bb6dd11f798f7bb5ccce777954dd8e699463e45cb71503e1521d091"),
        records_sha256=("98c1ab1228480b58dc4309f7c64280c347e87ac44547d79e36ab6ceb52adff6d"),
        images_sha256=("e01ea67f4b5ace4cec3201018ceed9cb68a5699470711e4d233ce64b5263d760"),
        console_sha256=("1f9ef2a6382dccea5f9de78bdd3b2ed78cc30450d0be548bde505ee23546c7ee"),
        dataset_id="CVA-Recoverability-Measurement-Qualification-v1",
        seed=20260817,
        source_dataset_id="CVA-Chart-Pilot-v0.3",
        source_dataset_records_sha256=(
            "92ccdf54b11e2a6c12e12ef5273137824c6f3b94f38224abeb32d8319b83a62b"
        ),
        source_stage2_v2_external_evidence_sha256=(
            "3a9e521cfe718cc3dea9aee4f1591aac761fa47f893c986eb1ba722a44374577"
        ),
        records=300,
        per_stratum=50,
        model_calls=0,
        hypothesis_tested=False,
        confirmatory_execution_authorized=False,
        training_invoked=False,
    )


def load_measurement_qualification_data_anchor(
    path: Path,
) -> MeasurementQualificationDataAnchor:
    """Load the exact externally supplied dataset-generation evidence anchor."""

    mapping = load_yaml_mapping(path, label="measurement qualification data anchor")
    fields = {
        field.name for field in MeasurementQualificationDataAnchor.__dataclass_fields__.values()
    }
    reject_unknown_fields(mapping, fields, label="measurement qualification data anchor")
    if set(mapping) != fields:
        raise ValueError("measurement qualification data anchor is incomplete")
    boolean_fields = (
        "hypothesis_tested",
        "confirmatory_execution_authorized",
        "training_invoked",
    )
    if any(type(mapping[field]) is not bool for field in boolean_fields):
        raise TypeError("measurement qualification data anchor flags must be exact booleans")
    commit = mapping["generation_code_commit"]
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError("generation code commit must be a full Git digest")
    anchor = MeasurementQualificationDataAnchor(
        schema_version=_exact_int(mapping["schema_version"], "schema_version"),
        status=str(mapping["status"]),
        generation_code_commit=commit,
        generation_server_package_lock_sha256=_digest(
            mapping["generation_server_package_lock_sha256"], "generation package lock"
        ),
        attempt_marker_sha256=_digest(mapping["attempt_marker_sha256"], "attempt marker"),
        manifest_sha256=_digest(mapping["manifest_sha256"], "manifest"),
        records_sha256=_digest(mapping["records_sha256"], "records"),
        images_sha256=_digest(mapping["images_sha256"], "images"),
        console_sha256=_digest(mapping["console_sha256"], "console"),
        dataset_id=str(mapping["dataset_id"]),
        seed=_exact_int(mapping["seed"], "seed"),
        source_dataset_id=str(mapping["source_dataset_id"]),
        source_dataset_records_sha256=_digest(
            mapping["source_dataset_records_sha256"], "source dataset records"
        ),
        source_stage2_v2_external_evidence_sha256=_digest(
            mapping["source_stage2_v2_external_evidence_sha256"],
            "Stage-2 v2 external evidence",
        ),
        records=_exact_int(mapping["records"], "records"),
        per_stratum=_exact_int(mapping["per_stratum"], "per_stratum"),
        model_calls=_exact_int(mapping["model_calls"], "model_calls"),
        hypothesis_tested=mapping["hypothesis_tested"],
        confirmatory_execution_authorized=mapping["confirmatory_execution_authorized"],
        training_invoked=mapping["training_invoked"],
    )
    if anchor != _expected_anchor():
        raise ValueError("measurement qualification data anchor differs from server evidence")
    return anchor


def _regular_file(path: Path, *, label: str, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")


def _expected_marker(
    anchor: MeasurementQualificationDataAnchor,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "MEASUREMENT_QUALIFICATION_DATA_GENERATION_STARTED",
        "dataset_id": anchor.dataset_id,
        "seed": anchor.seed,
        "server_package_lock_sha256": anchor.generation_server_package_lock_sha256,
        "source_dataset_records_sha256": anchor.source_dataset_records_sha256,
        "source_stage2_v2_external_evidence_sha256": (
            anchor.source_stage2_v2_external_evidence_sha256
        ),
        "model_calls": 0,
        "hypothesis_tested": False,
        "confirmatory_execution_authorized": False,
    }


def verify_measurement_qualification_data_evidence(
    anchor: MeasurementQualificationDataAnchor,
    *,
    dataset_root: Path,
    attempt_marker: Path,
    console_log: Path,
    config: MeasurementQualificationConfig,
    reserved_numeric_tables: Set[tuple[int, int, int, int]],
) -> MeasurementQualificationDataAnchor:
    """Bind server artifacts and replay the complete model-free dataset."""

    if not isinstance(anchor, MeasurementQualificationDataAnchor):
        raise TypeError("anchor must be a MeasurementQualificationDataAnchor")
    _regular_file(
        attempt_marker,
        label="qualification attempt marker",
        expected_sha256=anchor.attempt_marker_sha256,
    )
    _regular_file(
        dataset_root / "manifest.json",
        label="qualification manifest",
        expected_sha256=anchor.manifest_sha256,
    )
    _regular_file(
        dataset_root / "records.jsonl",
        label="qualification records",
        expected_sha256=anchor.records_sha256,
    )
    _regular_file(
        console_log,
        label="qualification console",
        expected_sha256=anchor.console_sha256,
    )
    marker = load_strict_json_mapping(
        attempt_marker,
        label="qualification attempt marker",
        max_bytes=16 * 1024,
    )
    if marker != _expected_marker(anchor):
        raise ValueError("qualification attempt marker payload differs from anchor")
    try:
        console = console_log.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("qualification console must be UTF-8") from error
    if console.count("qualification_data_exit=0") < 1:
        raise ValueError("qualification console does not preserve a successful exit")
    verification = verify_measurement_qualification_dataset(
        dataset_root,
        config=config,
        reserved_numeric_tables=reserved_numeric_tables,
    )
    if verification.records_sha256 != anchor.records_sha256:
        raise ValueError("qualification replay records SHA-256 differs from anchor")
    if verification.images_sha256 != anchor.images_sha256:
        raise ValueError("qualification image bundle SHA-256 differs from anchor")
    if len(verification.scenes) != anchor.records:
        raise ValueError("qualification replay record count differs from anchor")
    return anchor
