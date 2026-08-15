from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from compbias.recoverability.measurement_qualification_execution import (
    MEASUREMENT_QUALIFICATION_EXECUTION_LOCK_PATH,
    MEASUREMENT_QUALIFICATION_EXECUTION_PACKAGE_PATHS,
    verify_measurement_qualification_execution_package_lock,
)

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = (
    ROOT
    / "experiments/recoverability_v1/10_measurement_qualification_preflight.py"
)
EXECUTE = ROOT / "experiments/recoverability_v1/11_run_measurement_qualification.py"
SERVER_LOCK = ROOT / MEASUREMENT_QUALIFICATION_EXECUTION_LOCK_PATH


def test_qualification_execution_package_lock_binds_exact_complete_surface() -> None:
    verification = verify_measurement_qualification_execution_package_lock(
        SERVER_LOCK,
        repository_root=ROOT,
    )
    observed = frozenset(item.relative_path for item in verification.files)

    assert verification.verified is True
    assert observed == MEASUREMENT_QUALIFICATION_EXECUTION_PACKAGE_PATHS
    assert observed >= {
        "configs/recoverability/measurement_qualification_data_anchor.yaml",
        "configs/recoverability/measurement_qualification_v1.yaml",
        "experiments/recoverability_v1/10_measurement_qualification_preflight.py",
        "experiments/recoverability_v1/11_run_measurement_qualification.py",
        "src/compbias/recoverability/measurement_qualification.py",
        "src/compbias/recoverability/measurement_qualification_anchor.py",
        "src/compbias/recoverability/measurement_qualification_data.py",
        "src/compbias/recoverability/measurement_qualification_execution.py",
    }
    assert all(not path.startswith("tests/") for path in observed)


def test_qualification_execution_lock_rejects_noncanonical_subset(tmp_path: Path) -> None:
    subset = tmp_path / "subset.yaml"
    subset.write_text(
        "schema_version: 1\nfiles:\n"
        "  - path: configs/recoverability/measurement_qualification_v1.yaml\n"
        "    sha256: "
        + "0" * 64
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical"):
        verify_measurement_qualification_execution_package_lock(
            subset,
            repository_root=ROOT,
        )


def test_qualification_preflight_is_metadata_only_and_exclusive() -> None:
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
    assert "recoverability_measurement_qualification_metadata_preflight" in source
    assert 'open("x"' in source
    assert "load_local_qwen" not in source
    assert "decode_qwen_once" not in source
    assert "import torch" not in source
    assert "model_loaded" in source
    assert "training_authorized" in source


def test_qualification_runner_is_explicit_one_shot_and_fail_closed() -> None:
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
        "--config",
        "--data-anchor",
        "--server-package-lock",
        "--preflight-report",
        "--dataset-root",
        "--dataset-attempt-marker",
        "--dataset-console-log",
        "--source-records",
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
    assert "COMPBIAS_" in source
    assert "HF_HUB_OFFLINE" in source
    assert "TRANSFORMERS_OFFLINE" in source
    assert "format_retries" in source
    assert "attempted.json" in source
    assert 'open("x"' in source
    assert "decode_qwen_once" in source
    assert "decode_text_qwen_once" in source
    assert "training_invoked" in source
    assert "hypothesis_tested" in source
    assert "confirmatory_execution_authorized" in source
    assert source.index("verify_measurement_qualification_data_evidence(") < source.index(
        "load_local_qwen("
    )
    assert source.index('open("x"') < source.index("load_local_qwen(")


def test_qualification_runner_output_is_exclusive_and_bounded() -> None:
    source = EXECUTE.read_text(encoding="utf-8")

    assert "qualification_records.jsonl" in source
    assert "qualification_report.json" in source
    assert "model_calls" in source
    assert "600" in source
    assert "tempfile.TemporaryDirectory" in source
    assert "return 0 if report.qualification_passed else 3" in source
    assert "generate_with_format_retries" not in source
    assert "train(" not in source
