"""Canonical Study C2 paths, kept separate from the completed Study C run."""

from __future__ import annotations

from pathlib import Path

ROOT = Path("artifacts/v5/study_c2")
DATA_ROOT = ROOT / "data"
PAIR_ROWS = DATA_ROOT / "reward_identifiability_pairs.jsonl"
PAIR_MANIFEST = DATA_ROOT / "reward_identifiability_pairs_manifest.json"
FIBER_ROWS = DATA_ROOT / "reward_fibers.jsonl"
FIBER_MANIFEST = DATA_ROOT / "reward_fibers_manifest.json"
LEGACY_ROOT = ROOT / "legacy_parser_audit"
SUPPORT_ROOT = ROOT / "frozen_policy_support"
SUPPORT_RAW_ROWS = SUPPORT_ROOT / "raw_rows.jsonl"
SUPPORT_SUMMARY = SUPPORT_ROOT / "summary.json"
SUPPORT_MANIFEST = SUPPORT_ROOT / "manifest.json"
SHARED_GRADIENT_ROOT = ROOT / "shared_gradient_audit"
SHARED_GRADIENT_ROWS = SHARED_GRADIENT_ROOT / "per_group.jsonl"
SHARED_GRADIENT_SUMMARY = SHARED_GRADIENT_ROOT / "summary.json"
SHARED_GRADIENT_MANIFEST = SHARED_GRADIENT_ROOT / "manifest.json"
STAGE24_EXECUTION_CONTRACT = ROOT / "stage24_execution_contract.json"
STAGE25_EXECUTION_CONTRACT = ROOT / "stage25_execution_contract.json"
TRAINING_ROOT = ROOT / "training"
TRAINING_PAIR_MANIFEST = TRAINING_ROOT / "manifest.json"
EVALUATION_ROOT = ROOT / "evaluation"
EVALUATION_RAW_ROWS = EVALUATION_ROOT / "raw_rows.jsonl"
EVALUATION_SUMMARY = EVALUATION_ROOT / "summary.json"
EVALUATION_MANIFEST = EVALUATION_ROOT / "manifest.json"
REPORT_ROOT = ROOT / "report"

__all__ = [
    "DATA_ROOT",
    "EVALUATION_MANIFEST",
    "EVALUATION_RAW_ROWS",
    "EVALUATION_ROOT",
    "EVALUATION_SUMMARY",
    "FIBER_MANIFEST",
    "FIBER_ROWS",
    "LEGACY_ROOT",
    "PAIR_MANIFEST",
    "PAIR_ROWS",
    "REPORT_ROOT",
    "ROOT",
    "SHARED_GRADIENT_MANIFEST",
    "SHARED_GRADIENT_ROOT",
    "SHARED_GRADIENT_ROWS",
    "SHARED_GRADIENT_SUMMARY",
    "STAGE24_EXECUTION_CONTRACT",
    "STAGE25_EXECUTION_CONTRACT",
    "SUPPORT_MANIFEST",
    "SUPPORT_RAW_ROWS",
    "SUPPORT_ROOT",
    "SUPPORT_SUMMARY",
    "TRAINING_PAIR_MANIFEST",
    "TRAINING_ROOT",
]
