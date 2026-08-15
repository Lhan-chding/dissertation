from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from compbias.recoverability.phase_n_execution import (
    PHASE_N_EXECUTION_LOCK_PATH,
    PHASE_N_EXECUTION_PACKAGE_PATHS,
    verify_phase_n_execution_package_lock,
)

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "experiments/recoverability_v1/12_phase_n_preflight.py"
EXECUTE = ROOT / "experiments/recoverability_v1/13_run_phase_n.py"
SERVER_LOCK = ROOT / PHASE_N_EXECUTION_LOCK_PATH


def test_phase_n_package_lock_binds_exact_complete_surface() -> None:
    verification = verify_phase_n_execution_package_lock(SERVER_LOCK, repository_root=ROOT)
    observed = frozenset(item.relative_path for item in verification.files)

    assert verification.verified is True
    assert observed == PHASE_N_EXECUTION_PACKAGE_PATHS
    assert observed >= {
        "configs/recoverability/measurement_qualification_frozen_result.yaml",
        "configs/recoverability/recoverability_v1.yaml",
        "experiments/recoverability_v1/12_phase_n_preflight.py",
        "experiments/recoverability_v1/13_run_phase_n.py",
        "src/compbias/recoverability/measurement_qualification_result.py",
        "src/compbias/recoverability/natural_inference.py",
        "src/compbias/recoverability/phase_n.py",
        "src/compbias/recoverability/phase_n_execution.py",
    }
    assert all(not path.startswith("tests/") for path in observed)


def test_phase_n_lock_rejects_noncanonical_subset(tmp_path: Path) -> None:
    subset = tmp_path / "subset.yaml"
    subset.write_text(
        "schema_version: 1\nfiles:\n"
        "  - path: configs/recoverability/recoverability_v1.yaml\n"
        "    sha256: " + "0" * 64 + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical"):
        verify_phase_n_execution_package_lock(subset, repository_root=ROOT)


def test_phase_n_preflight_is_metadata_only_and_exclusive() -> None:
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
    assert "recoverability_phase_n_metadata_preflight" in source
    assert 'open("x"' in source
    assert "load_local_qwen" not in source
    assert "decode_qwen_once" not in source
    assert "import torch" not in source
    assert "model_loaded" in source
    assert "training_authorized" in source


def test_phase_n_runner_is_original_protocol_one_shot_and_fail_closed() -> None:
    help_result = subprocess.run(
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
        "--protocol",
        "--server-package-lock",
        "--preflight-report",
        "--qualification-preflight",
        "--qualification-attempt-marker",
        "--qualification-report",
        "--qualification-records",
        "--qualification-console-log",
        "--source-records",
        "--qualification-dataset-root",
        "--qualification-dataset-attempt-marker",
        "--qualification-dataset-console-log",
    ):
        assert argument in help_result.stdout

    blocked = subprocess.run(
        [sys.executable, str(EXECUTE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 2
    assert "BLOCKED" in blocked.stdout

    source = EXECUTE.read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE" in source
    assert "TRANSFORMERS_OFFLINE" in source
    assert "COMPBIAS_" in source
    assert "generate_with_format_retries" in source
    assert "max_format_retries=0" in source
    assert "decode_qwen_once" in source
    assert "decode_text_qwen_once" not in source
    assert "4000" in source
    assert "attempted.json" in source
    assert 'open("x"' in source
    assert "training_invoked" in source
    assert source.index("verify_measurement_qualification_result_artifacts(") < source.index(
        "load_local_qwen("
    )
    assert source.index('open("x"') < source.index("load_local_qwen(")
    assert "train(" not in source
