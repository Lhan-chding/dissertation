from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from compbias.recoverability.phase_c_screen_execution import (
    PHASE_C_SCREEN_EXECUTION_LOCK_PATH,
    PHASE_C_SCREEN_EXECUTION_PACKAGE_PATHS,
    verify_phase_c_screen_execution_package_lock,
)


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "experiments/recoverability_v1/14_phase_c_screen_preflight.py"
EXECUTE = ROOT / "experiments/recoverability_v1/15_run_phase_c_screen.py"
SERVER_LOCK = ROOT / PHASE_C_SCREEN_EXECUTION_LOCK_PATH


def test_phase_c_screen_package_lock_binds_exact_complete_surface() -> None:
    verification = verify_phase_c_screen_execution_package_lock(
        SERVER_LOCK, repository_root=ROOT
    )
    observed = frozenset(item.relative_path for item in verification.files)
    assert verification.verified is True
    assert observed == PHASE_C_SCREEN_EXECUTION_PACKAGE_PATHS
    assert observed >= {
        "configs/recoverability/phase_n_frozen_result.yaml",
        "configs/recoverability/recoverability_phase_c_v2_amendment.yaml",
        "experiments/recoverability_v1/14_phase_c_screen_preflight.py",
        "experiments/recoverability_v1/15_run_phase_c_screen.py",
        "src/compbias/recoverability/phase_c_screen.py",
        "src/compbias/recoverability/phase_n_result.py",
    }
    assert all(not path.startswith("tests/") for path in observed)


def test_phase_c_screen_lock_rejects_noncanonical_subset(tmp_path: Path) -> None:
    subset = tmp_path / "subset.yaml"
    subset.write_text(
        "schema_version: 1\nfiles:\n"
        "  - path: configs/recoverability/recoverability_v1.yaml\n"
        "    sha256: " + "0" * 64 + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical"):
        verify_phase_c_screen_execution_package_lock(subset, repository_root=ROOT)


def test_phase_c_screen_preflight_is_metadata_only_and_exclusive() -> None:
    result = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for argument in ("--runtime", "--server-package-lock", "--project-root", "--output"):
        assert argument in result.stdout
    source = PREFLIGHT.read_text(encoding="utf-8")
    assert "recoverability_phase_c_v2_screen_metadata_preflight" in source
    assert 'open("x"' in source
    assert "load_local_qwen" not in source
    assert "decode_qwen_once" not in source
    assert "training_authorized" in source


def test_phase_c_screen_runner_is_one_shot_and_stops_before_arm_calls() -> None:
    result = subprocess.run(
        [sys.executable, str(EXECUTE), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for argument in (
        "--execute",
        "--paths",
        "--runtime",
        "--amendment",
        "--phase-n-result",
        "--server-package-lock",
        "--preflight-report",
        "--phase-n-preflight",
        "--phase-n-attempt-marker",
        "--phase-n-dataset-root",
        "--phase-n-output-root",
        "--phase-n-console-log",
        "--source-records",
        "--qualification-records",
    ):
        assert argument in result.stdout
    blocked = subprocess.run(
        [sys.executable, str(EXECUTE)], cwd=ROOT, capture_output=True, text=True
    )
    assert blocked.returncode == 2
    assert "BLOCKED" in blocked.stdout
    source = EXECUTE.read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE" in source
    assert "TRANSFORMERS_OFFLINE" in source
    assert "COMPBIAS_" in source
    assert "8000" in source
    assert "max_format_retries=0" in source
    assert "attempted.json" in source
    assert 'open("x"' in source
    assert source.index("verify_phase_n_result_artifacts(") < source.index("load_local_qwen(")
    assert source.index('open("x"') < source.index("load_local_qwen(")
    assert "run_phase_c_arms" not in source
    assert "train(" not in source
