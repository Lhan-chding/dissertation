from __future__ import annotations

from pathlib import Path


def test_registered_study_c3_surface_exists() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / "configs/v5/study_c3_resolution_validity.yaml").is_file()
    scripts = root / "scripts/v5/study_c3"
    assert [path.name for path in sorted(scripts.glob("[0-9][0-9]_*.py"))] == [
        "00_audit_existing_action_channel.py",
        "01_build_valid_world_trie.py",
        "02_validate_constrained_decoder.py",
        "03_evaluate_existing_c2_checkpoints.py",
        "04_freeze_factorial_execution_contract.py",
        "05_train_factorial_grpo.py",
        "06_evaluate_factorial_checkpoints.py",
        "07_shared_gradient_validity_audit.py",
        "08_analyze_resolution_validity.py",
        "09_package_study_c3_evidence.py",
    ]
