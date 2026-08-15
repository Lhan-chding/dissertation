"""Immutable public anchor for the externally captured Stage-2 v2 evidence."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from compbias.io.strict_json import load_strict_json_mapping
from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_LABELS = frozenset(
    {"attempt_marker", "console", "preflight", "probe_records", "probe_report"}
)


@dataclass(frozen=True, slots=True)
class Stage2V2ExternalEvidenceAnchor:
    schema_version: int
    status: str
    capture_code_commit: str
    evidence_package_lock_sha256: str
    external_evidence_sha256: str
    artifact_type: str
    records: int
    replayed_program_parse_successes: int
    replayed_program_execution_successes: int
    replayed_executor_answer_correct: int
    model_calls: int
    verified: bool
    hypothesis_tested: bool
    confirmatory_execution_authorized: bool
    training_invoked: bool
    source_stage2_v2_records_sha256: str
    source_sha256: tuple[tuple[str, str], ...]


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


def _source_hashes(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or set(value) != _SOURCE_LABELS:
        raise ValueError("Stage-2 v2 external source registry is incomplete")
    return tuple(sorted((label, _digest(digest, label)) for label, digest in value.items()))


def load_stage2_v2_external_evidence_anchor(
    path: Path,
) -> Stage2V2ExternalEvidenceAnchor:
    """Load the exact server-supplied external evidence anchor."""

    mapping = load_yaml_mapping(path, label="Stage-2 v2 external evidence anchor")
    fields = {field.name for field in Stage2V2ExternalEvidenceAnchor.__dataclass_fields__.values()}
    reject_unknown_fields(
        mapping,
        fields,
        label="Stage-2 v2 external evidence anchor",
    )
    if set(mapping) != fields:
        raise ValueError("Stage-2 v2 external evidence anchor is incomplete")
    boolean_fields = (
        "verified",
        "hypothesis_tested",
        "confirmatory_execution_authorized",
        "training_invoked",
    )
    if any(type(mapping[field]) is not bool for field in boolean_fields):
        raise TypeError("Stage-2 v2 external evidence flags must be exact booleans")
    anchor = Stage2V2ExternalEvidenceAnchor(
        schema_version=_exact_int(mapping["schema_version"], "schema_version"),
        status=str(mapping["status"]),
        capture_code_commit=str(mapping["capture_code_commit"]),
        evidence_package_lock_sha256=_digest(
            mapping["evidence_package_lock_sha256"], "evidence package lock"
        ),
        external_evidence_sha256=_digest(mapping["external_evidence_sha256"], "external evidence"),
        artifact_type=str(mapping["artifact_type"]),
        records=_exact_int(mapping["records"], "records"),
        replayed_program_parse_successes=_exact_int(
            mapping["replayed_program_parse_successes"],
            "replayed_program_parse_successes",
        ),
        replayed_program_execution_successes=_exact_int(
            mapping["replayed_program_execution_successes"],
            "replayed_program_execution_successes",
        ),
        replayed_executor_answer_correct=_exact_int(
            mapping["replayed_executor_answer_correct"],
            "replayed_executor_answer_correct",
        ),
        model_calls=_exact_int(mapping["model_calls"], "model_calls"),
        verified=mapping["verified"],
        hypothesis_tested=mapping["hypothesis_tested"],
        confirmatory_execution_authorized=mapping["confirmatory_execution_authorized"],
        training_invoked=mapping["training_invoked"],
        source_stage2_v2_records_sha256=_digest(
            mapping["source_stage2_v2_records_sha256"], "Stage-2 v2 records"
        ),
        source_sha256=_source_hashes(mapping["source_sha256"]),
    )
    expected = Stage2V2ExternalEvidenceAnchor(
        schema_version=1,
        status="FINAL_VERIFIED_EXTERNAL_EVIDENCE_DO_NOT_RERUN",
        capture_code_commit="943a73a27d51848805b4fbafb360fb881ea631a7",
        evidence_package_lock_sha256=(
            "9c1d0e8da34140a3a13332c64127896709390f2446392f93ddfacc2ad5f39e2f"
        ),
        external_evidence_sha256=(
            "3a9e521cfe718cc3dea9aee4f1591aac761fa47f893c986eb1ba722a44374577"
        ),
        artifact_type="recoverability_stage2_v2_external_evidence",
        records=24,
        replayed_program_parse_successes=24,
        replayed_program_execution_successes=24,
        replayed_executor_answer_correct=24,
        model_calls=0,
        verified=True,
        hypothesis_tested=False,
        confirmatory_execution_authorized=False,
        training_invoked=False,
        source_stage2_v2_records_sha256=(
            "6b0604a08ebbf4611c62b7fe9f1d9e03954385b1bcfaedf613f0b70b32f1d2f8"
        ),
        source_sha256=(
            (
                "attempt_marker",
                "5cf97d5cb67c5e9830ac7455b0759b47b6d1fe1e76ee5ce42f365f4059e51c7c",
            ),
            (
                "console",
                "3649a50b21a5482ab0e20bfe07ed63f4d49f72c1f1b783791b852044253eed81",
            ),
            (
                "preflight",
                "c3b8949f03ae7ba2947ad5632bfd68dc822f3aead33adc218e90619a0957fe0c",
            ),
            (
                "probe_records",
                "6b0604a08ebbf4611c62b7fe9f1d9e03954385b1bcfaedf613f0b70b32f1d2f8",
            ),
            (
                "probe_report",
                "d207cff9f6bdcb48142e3f1bb8a3d8676d7aa5abdbcd0c226666dbb58ac587b7",
            ),
        ),
    )
    if anchor != expected:
        raise ValueError("Stage-2 v2 external evidence anchor differs from server evidence")
    return anchor


def _expected_payload(anchor: Stage2V2ExternalEvidenceAnchor) -> dict[str, object]:
    return {
        "artifact_type": anchor.artifact_type,
        "confirmatory_execution_authorized": anchor.confirmatory_execution_authorized,
        "hypothesis_tested": anchor.hypothesis_tested,
        "model_calls": anchor.model_calls,
        "records": anchor.records,
        "replayed_executor_answer_correct": anchor.replayed_executor_answer_correct,
        "replayed_program_execution_successes": (anchor.replayed_program_execution_successes),
        "replayed_program_parse_successes": anchor.replayed_program_parse_successes,
        "schema_version": anchor.schema_version,
        "source_sha256": [list(item) for item in anchor.source_sha256],
        "source_stage2_v2_records_sha256": anchor.source_stage2_v2_records_sha256,
        "training_invoked": anchor.training_invoked,
        "verified": anchor.verified,
    }


def verify_stage2_v2_external_evidence(
    anchor: Stage2V2ExternalEvidenceAnchor,
    path: Path,
) -> Stage2V2ExternalEvidenceAnchor:
    """Verify both the exact bytes and fail-closed semantics of the manifest."""

    if not isinstance(anchor, Stage2V2ExternalEvidenceAnchor):
        raise TypeError("anchor must be a Stage2V2ExternalEvidenceAnchor")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Stage-2 v2 external evidence must be a regular file")
    if _sha256(path) != anchor.external_evidence_sha256:
        raise ValueError("Stage-2 v2 external evidence SHA-256 mismatch")
    payload = load_strict_json_mapping(
        path,
        label="Stage-2 v2 external evidence",
        max_bytes=64 * 1024,
    )
    if payload != _expected_payload(anchor):
        raise ValueError("Stage-2 v2 external evidence payload differs from anchor")
    return anchor
