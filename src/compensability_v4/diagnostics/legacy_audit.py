"""Phase-0 legacy audit builder for v4."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LegacyAudit:
    registry_rows: tuple[dict[str, str], ...]
    claim_rows: tuple[dict[str, str], ...]
    scoring_contract: str
    hash_manifest: dict[str, object]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_legacy_audit(root: Path) -> LegacyAudit:
    registry_rows = (
        {
            "experiment_id": "phase_c_v3_dsl",
            "interface_family": "strict_result_program",
            "registered_interpretation": "measurement_interface_failure",
            "true_world_recoveries": "0",
            "evidence_status": "frozen_legacy_result",
        },
        {
            "experiment_id": "qwen_world_only_valid_cue_50",
            "interface_family": "text_replay",
            "registered_interpretation": "hard_text_symbolic_recovery_failure",
            "true_world_recoveries": "0",
            "evidence_status": "frozen_local_world_only_summary",
        },
        {
            "experiment_id": "qwen_world_only_no_cue_100",
            "interface_family": "text_replay",
            "registered_interpretation": "legacy_no_cue_summary_recovered",
            "true_world_recoveries": "",
            "evidence_status": "awaiting_hash_bound_server_evidence",
        },
    )
    claim_rows = (
        {
            "claim_id": "qwen_world_only_copying",
            "status": "allowed",
        },
        {
            "claim_id": "qwen_natural_visual_state_absence",
            "status": "forbidden",
        },
    )
    scoring_contract = (
        "# V4 scoring contract\n\n"
        "- observed-world consistency is separate from exact world recovery.\n"
        "- Phase-C v3 is recorded as measurement_interface_failure.\n"
        "- two-call textual replay conditions are renamed text_replay.\n"
    )
    inputs = tuple(
        root / relative
        for relative in (
            "README.md",
            "docs/RESEARCH_QUESTION.md",
            "docs/CPU_EVIDENCE.md",
            "docs/GPU_PILOT_PROTOCOL.md",
        )
    )
    hash_manifest = {
        "schema_version": 1,
        "inputs": [
            {"relative_path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in inputs
        ],
    }
    return LegacyAudit(
        registry_rows=registry_rows,
        claim_rows=claim_rows,
        scoring_contract=scoring_contract,
        hash_manifest=hash_manifest,
    )


def write_legacy_audit(root: Path, artifact_root: Path) -> None:
    audit = build_legacy_audit(root)
    output = artifact_root / "v4" / "audit"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "legacy_experiment_registry.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(audit.registry_rows[0]))
        writer.writeheader()
        writer.writerows(audit.registry_rows)
    with (output / "claim_evidence_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(audit.claim_rows[0]))
        writer.writeheader()
        writer.writerows(audit.claim_rows)
    (output / "scoring_contract.md").write_text(audit.scoring_contract, encoding="utf-8")
    (output / "legacy_hash_manifest.json").write_text(
        json.dumps(audit.hash_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
