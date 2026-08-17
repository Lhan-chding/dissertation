"""Build the Phase-0 audit from repository-frozen evidence only.

The hundred-call run has no hash-bound raw summary in this checkout.  Its missing no-cue
result therefore remains an evidence request rather than an inferred result.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

_ARTIFACT_NAMES = frozenset(
    {
        "legacy_experiment_registry.csv",
        "claim_evidence_matrix.csv",
        "scoring_contract.md",
        "legacy_hash_manifest.json",
    }
)
_FROZEN_INPUTS = (
    "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
    "README.md",
    "docs/RECOVERABILITY_V1_PROTOCOL.md",
    "configs/recoverability/stage2_v2_frozen_result.yaml",
    "configs/recoverability/phase_c_screen_v2_frozen_result.yaml",
    "configs/recoverability/recoverability_phase_c_v3_postscreen_amendment.yaml",
    "configs/recoverability/phase_c_world_recovery_v1.yaml",
    "configs/recoverability/server_package_lock_phase_c_world_recovery_v1.yaml",
    "configs/recoverability/phase_c_world_recovery_100_v1.yaml",
    "configs/recoverability/server_package_lock_phase_c_world_recovery_100_v1.yaml",
)


@dataclass(frozen=True, slots=True)
class LegacyAudit:
    registry_rows: tuple[dict[str, str], ...]
    claim_rows: tuple[dict[str, str], ...]
    scoring_contract: str
    hash_manifest: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_input_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    resolved_root = root.resolve()
    for relative in _FROZEN_INPUTS:
        path = root / relative
        resolved = path.resolve()
        if resolved_root not in resolved.parents or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"frozen audit input is unavailable or unsafe: {relative}")
        rows.append({"relative_path": relative, "sha256": _sha256(path)})
    return rows


def build_legacy_audit(root: Path) -> LegacyAudit:
    """Return only claims supported by frozen repository files."""

    registry_rows = (
        {
            "experiment_id": "small_neural_natural_replay_v2",
            "interface_family": "controlled_exact_natural_fork",
            "registered_interpretation": "small_model_natural_and_synthetic_mediators_differ",
            "model_calls": "",
            "true_world_recoveries": "",
            "observation_copies": "",
            "evidence_status": "frozen_plan_summary",
            "evidence_reference": (
                "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md"
            ),
        },
        {
            "experiment_id": "stage2_v2_forward_dsl",
            "interface_family": "trusted_state_forward_program",
            "registered_interpretation": "forward_chain_operational_not_inverse_recovery",
            "model_calls": "24",
            "true_world_recoveries": "",
            "observation_copies": "",
            "evidence_status": "frozen_legacy_result",
            "evidence_reference": "configs/recoverability/stage2_v2_frozen_result.yaml",
        },
        {
            "experiment_id": "phase_c_v3_dsl",
            "interface_family": "strict_result_program",
            "registered_interpretation": "measurement_interface_failure",
            "model_calls": "27840",
            "true_world_recoveries": "",
            "observation_copies": "",
            "evidence_status": "frozen_legacy_result_interface_invalid",
            "evidence_reference": "README.md;docs/RECOVERABILITY_V1_PROTOCOL.md",
        },
        {
            "experiment_id": "qwen_world_only_v1r1_12",
            "interface_family": "text_replay",
            "registered_interpretation": "descriptive_world_only_diagnostic",
            "model_calls": "12",
            "true_world_recoveries": "1",
            "observation_copies": "",
            "evidence_status": "frozen_repository_summary",
            "evidence_reference": "README.md;docs/RECOVERABILITY_V1_PROTOCOL.md",
        },
        {
            "experiment_id": "qwen_world_only_valid_cue_50",
            "interface_family": "text_replay",
            "registered_interpretation": "hard_text_symbolic_recovery_failure",
            "model_calls": "50",
            "true_world_recoveries": "0",
            "observation_copies": "41",
            "evidence_status": "frozen_plan_summary_raw_summary_not_local",
            "evidence_reference": (
                "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md"
            ),
        },
        {
            "experiment_id": "qwen_world_only_no_cue_100",
            "interface_family": "text_replay",
            "registered_interpretation": "no_result_until_raw_summary_is_hash_bound",
            "model_calls": "100",
            "true_world_recoveries": "",
            "observation_copies": "",
            "evidence_status": "awaiting_hash_bound_server_evidence",
            "evidence_reference": "",
        },
    )
    claim_rows = (
        {
            "claim_id": "small_model_natural_synthetic_mediator_difference",
            "status": "allowed",
            "scope": "controlled_small_model_only",
            "evidence": "small_neural_natural_replay_v2",
        },
        {
            "claim_id": "qwen_world_only_copying",
            "status": "allowed",
            "scope": "valid_cue_hard_text_zero_shot_free_generation",
            "evidence": "qwen_world_only_valid_cue_50",
        },
        {
            "claim_id": "qwen_reliable_hard_text_recovery",
            "status": "forbidden",
            "scope": "current_frozen_evidence_does_not_support_reliability",
            "evidence": "qwen_world_only_valid_cue_50",
        },
        {
            "claim_id": "qwen_natural_visual_state_absence",
            "status": "forbidden",
            "scope": "natural_visual_state_not_tested_by_text_replay",
            "evidence": "",
        },
        {
            "claim_id": "qwen_cannot_repair_under_any_interface",
            "status": "forbidden",
            "scope": "interface_ladder_not_yet_run",
            "evidence": "",
        },
        {
            "claim_id": "rl_failed",
            "status": "forbidden",
            "scope": "no_recovery_sft_lora_or_rl_has_run",
            "evidence": "",
        },
        {
            "claim_id": "phase_c_v3_zero_is_semantic_recovery_null",
            "status": "forbidden",
            "scope": "strict_measurement_interface_failed",
            "evidence": "phase_c_v3_dsl",
        },
        {
            "claim_id": "stage2_forward_program_proves_inverse_recovery",
            "status": "forbidden",
            "scope": "forward_control_only",
            "evidence": "stage2_v2_forward_dsl",
        },
    )
    scoring_contract = """# V4 legacy scoring contract

## Hidden-truth primary score

Every no-cue and valid-cue world output is scored against the same immutable hidden true
world. Exact world recovery means that all four parsed integers equal that hidden world.

## Separate measurements

- `exact_world_recovery`: parsed output equals the hidden true world.
- `observed_world_consistency`: parsed output equals the Stage-1 observed world.
- `observation_copy`: the complete observed world is reproduced.
- `single_edit_wrong`, `over_edit`, and `parse_failure` are disjoint outcomes.
- Strict ResultProgram parsing, post-hoc semantic extraction, and world-only CSV recovery are
  different measurement interfaces and must never be pooled.

## Interface and claim boundary

Legacy two-call Qwen conditions are `text_replay`, never `c_fork`. Phase-C v3 is a
`measurement_interface_failure`; its 0/27,840 strict parses do not establish zero semantic
recovery. Candidate selection cannot substitute for free recovery. Image-retained correction
is `natural_visual_revision`, not pure symbolic reasoning repair.

## Missing hundred-call evidence

The archived plan records the valid-cue aggregate (0/50 true recoveries, 41/50 complete copies,
9/50 non-recovering edits), but this checkout has no hash-bound raw no-cue/valid-cue summary.
The no-cue result is blank and marked `awaiting_hash_bound_server_evidence`; it must not be
reconstructed from prose or fabricated.
"""
    hash_manifest = {
        "schema_version": 1,
        "artifact_type": "v4_legacy_evidence_hash_manifest",
        "evidence_scope": "frozen_repository_files_only",
        "inputs": _frozen_input_rows(root),
        "missing_evidence": [
            {
                "experiment_id": "qwen_world_only_no_cue_100",
                "status": "awaiting_hash_bound_server_evidence",
                "required_payload": "raw no-cue/valid-cue records plus aggregate summary",
            }
        ],
    }
    return LegacyAudit(registry_rows, claim_rows, scoring_contract, hash_manifest)


def _write_csv(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_legacy_audit(root: Path, artifact_root: Path) -> None:
    """Write exactly the four registered Phase-0 artifacts."""

    audit = build_legacy_audit(root)
    output = artifact_root / "v4" / "audit"
    output.mkdir(parents=True, exist_ok=True)
    unexpected = {path.name for path in output.iterdir()} - _ARTIFACT_NAMES
    if unexpected:
        raise RuntimeError(f"refusing to mix audit with unexpected artifacts: {unexpected}")
    _write_csv(output / "legacy_experiment_registry.csv", audit.registry_rows)
    _write_csv(output / "claim_evidence_matrix.csv", audit.claim_rows)
    (output / "scoring_contract.md").write_text(audit.scoring_contract, encoding="utf-8")
    (output / "legacy_hash_manifest.json").write_text(
        json.dumps(audit.hash_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
