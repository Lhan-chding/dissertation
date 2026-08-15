from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from compbias.recoverability.stage2_v2_anchor import (
    Stage2V2ExternalEvidenceAnchor,
    load_stage2_v2_external_evidence_anchor,
    verify_stage2_v2_external_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "configs" / "recoverability" / "stage2_v2_external_evidence_anchor.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload() -> dict[str, object]:
    return {
        "artifact_type": "recoverability_stage2_v2_external_evidence",
        "confirmatory_execution_authorized": False,
        "hypothesis_tested": False,
        "model_calls": 0,
        "records": 24,
        "replayed_executor_answer_correct": 24,
        "replayed_program_execution_successes": 24,
        "replayed_program_parse_successes": 24,
        "schema_version": 1,
        "source_sha256": [
            [
                "attempt_marker",
                "5cf97d5cb67c5e9830ac7455b0759b47b6d1fe1e76ee5ce42f365f4059e51c7c",
            ],
            [
                "console",
                "3649a50b21a5482ab0e20bfe07ed63f4d49f72c1f1b783791b852044253eed81",
            ],
            [
                "preflight",
                "c3b8949f03ae7ba2947ad5632bfd68dc822f3aead33adc218e90619a0957fe0c",
            ],
            [
                "probe_records",
                "6b0604a08ebbf4611c62b7fe9f1d9e03954385b1bcfaedf613f0b70b32f1d2f8",
            ],
            [
                "probe_report",
                "d207cff9f6bdcb48142e3f1bb8a3d8676d7aa5abdbcd0c226666dbb58ac587b7",
            ],
        ],
        "source_stage2_v2_records_sha256": (
            "6b0604a08ebbf4611c62b7fe9f1d9e03954385b1bcfaedf613f0b70b32f1d2f8"
        ),
        "training_invoked": False,
        "verified": True,
    }


def _synthetic_anchor(path: Path) -> Stage2V2ExternalEvidenceAnchor:
    payload = _payload()
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    frozen = load_stage2_v2_external_evidence_anchor(ANCHOR)
    return replace(frozen, external_evidence_sha256=_sha256(path))


def test_stage2_v2_external_evidence_is_frozen_as_non_hypothesis_evidence() -> None:
    anchor = load_stage2_v2_external_evidence_anchor(ANCHOR)

    assert anchor.status == "FINAL_VERIFIED_EXTERNAL_EVIDENCE_DO_NOT_RERUN"
    assert anchor.capture_code_commit == "943a73a27d51848805b4fbafb360fb881ea631a7"
    assert anchor.external_evidence_sha256 == (
        "3a9e521cfe718cc3dea9aee4f1591aac761fa47f893c986eb1ba722a44374577"
    )
    assert anchor.records == 24
    assert anchor.replayed_program_parse_successes == 24
    assert anchor.replayed_program_execution_successes == 24
    assert anchor.replayed_executor_answer_correct == 24
    assert anchor.model_calls == 0
    assert anchor.verified is True
    assert anchor.hypothesis_tested is False
    assert anchor.confirmatory_execution_authorized is False
    assert anchor.training_invoked is False


def test_stage2_v2_external_evidence_verifier_binds_bytes_and_payload(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "stage2-v2-external-evidence.json"
    anchor = _synthetic_anchor(evidence)

    verification = verify_stage2_v2_external_evidence(anchor, evidence)

    assert verification == anchor
    assert asdict(verification)["model_calls"] == 0

    evidence.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_stage2_v2_external_evidence(anchor, evidence)


def test_stage2_v2_external_evidence_rejects_semantic_tampering_with_new_hash(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "stage2-v2-external-evidence.json"
    anchor = _synthetic_anchor(evidence)
    payload = _payload()
    payload["confirmatory_execution_authorized"] = True
    evidence.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tampered_anchor = replace(anchor, external_evidence_sha256=_sha256(evidence))

    with pytest.raises(ValueError, match="payload"):
        verify_stage2_v2_external_evidence(tampered_anchor, evidence)
