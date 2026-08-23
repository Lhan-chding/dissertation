"""Canonical Study C2 inputs and Study C3 output paths."""

from pathlib import Path

CONFIG = Path("configs/v5/study_c3_resolution_validity.yaml")
PACKAGE_LOCK = Path("configs/v5/server_package_lock.yaml")
C2_FIBER_ROWS = Path("artifacts/v5/study_c2/data/reward_fibers.jsonl")
C2_TRAINING_ROOT = Path("artifacts/v5/study_c2/training")
C2_TRAINING_MANIFEST = C2_TRAINING_ROOT / "manifest.json"
C2_EVALUATION_ROOT = Path("artifacts/v5/study_c2/evaluation")
C2_EVALUATION_RAW = C2_EVALUATION_ROOT / "raw_rows.jsonl"
C2_EVALUATION_SUMMARY = C2_EVALUATION_ROOT / "summary.json"
C2_EVALUATION_MANIFEST = C2_EVALUATION_ROOT / "manifest.json"

ROOT = Path("artifacts/v5/study_c3")
ACTION_AUDIT_ROOT = ROOT / "action_channel_audit"
ACTION_AUDIT_ROWS = ACTION_AUDIT_ROOT / "per_row.jsonl"
ACTION_AUDIT_SUMMARY = ACTION_AUDIT_ROOT / "summary.json"
ACTION_AUDIT_EXAMPLES = ACTION_AUDIT_ROOT / "failure_examples.md"
ACTION_AUDIT_MANIFEST = ACTION_AUDIT_ROOT / "manifest.json"

TRIE_ROOT = ROOT / "valid_world_trie"
TRIE_PAYLOAD = TRIE_ROOT / "trie.json"
TRIE_MANIFEST = TRIE_ROOT / "manifest.json"
TRIE_VALIDATION_SUMMARY = TRIE_ROOT / "validation_summary.json"
TRIE_VALIDATION_MANIFEST = TRIE_ROOT / "validation_manifest.json"

EXISTING_EVAL_ROOT = ROOT / "existing_checkpoint_decoder_intervention"
EXISTING_EVAL_RAW = EXISTING_EVAL_ROOT / "raw_rows.jsonl"
EXISTING_EVAL_MASKS = EXISTING_EVAL_ROOT / "logit_masks.jsonl"
EXISTING_EVAL_SCENES = EXISTING_EVAL_ROOT / "per_scene.jsonl"
EXISTING_EVAL_SUMMARY = EXISTING_EVAL_ROOT / "summary.json"
EXISTING_EVAL_MANIFEST = EXISTING_EVAL_ROOT / "manifest.json"

EXECUTION_CONTRACT = ROOT / "factorial_execution_contract.json"
TRAINING_ROOT = ROOT / "training"
TRAINING_MANIFEST = TRAINING_ROOT / "manifest.json"

FACTORIAL_EVAL_ROOT = ROOT / "factorial_evaluation"
FACTORIAL_EVAL_RAW = FACTORIAL_EVAL_ROOT / "raw_rows.jsonl"
FACTORIAL_EVAL_MASKS = FACTORIAL_EVAL_ROOT / "logit_masks.jsonl"
FACTORIAL_EVAL_SCENES = FACTORIAL_EVAL_ROOT / "per_scene.jsonl"
FACTORIAL_EVAL_SUMMARY = FACTORIAL_EVAL_ROOT / "summary.json"
FACTORIAL_EVAL_MANIFEST = FACTORIAL_EVAL_ROOT / "manifest.json"

GRADIENT_ROOT = ROOT / "shared_gradient_validity_audit"
GRADIENT_BUFFER = GRADIENT_ROOT / "shared_buffer.jsonl"
GRADIENT_ROWS = GRADIENT_ROOT / "per_group.jsonl"
GRADIENT_SUMMARY = GRADIENT_ROOT / "summary.json"
GRADIENT_MANIFEST = GRADIENT_ROOT / "manifest.json"

ANALYSIS_ROOT = ROOT / "analysis"
ANALYSIS_SUMMARY = ANALYSIS_ROOT / "summary.json"
ANALYSIS_TABLES = ANALYSIS_ROOT / "summary_tables.json"
ANALYSIS_INTERVALS = ANALYSIS_ROOT / "confidence_intervals.json"
ANALYSIS_MANIFEST = ANALYSIS_ROOT / "manifest.json"

REPORT_ROOT = ROOT / "report"
FACT_REPORT = REPORT_ROOT / "STUDY_C3_RESULT_FACTS.md"
REPORT_SHA_MANIFEST = REPORT_ROOT / "sha256_manifest.json"

__all__ = [name for name in globals() if name.isupper()]
