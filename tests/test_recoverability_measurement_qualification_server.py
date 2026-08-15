from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from compbias.recoverability.measurement_qualification_server import (
    MEASUREMENT_QUALIFICATION_DATA_LOCK_PATH,
    MEASUREMENT_QUALIFICATION_DATA_PACKAGE_PATHS,
    verify_measurement_qualification_data_package_lock,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments"
    / "recoverability_v1"
    / "09_generate_measurement_qualification_data.py"
)
SERVER_LOCK = ROOT / MEASUREMENT_QUALIFICATION_DATA_LOCK_PATH


def test_measurement_qualification_data_cli_is_explicit_model_free_one_shot() -> None:
    help_result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for argument in (
        "--execute",
        "--paths",
        "--config",
        "--server-package-lock",
        "--stage2-v2-external-evidence",
    ):
        assert argument in help_result.stdout

    blocked = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 2
    assert "BLOCKED" in blocked.stdout

    source = SCRIPT.read_text(encoding="utf-8")
    assert "load_local_qwen" not in source
    assert "decode_text_qwen_once" not in source
    assert "import torch" not in source
    assert "from_pretrained" not in source
    assert "training" not in source.lower()
    assert source.index("verify_stage2_v2_external_evidence(") < source.index(
        "write_measurement_qualification_dataset("
    )


def test_measurement_qualification_data_lock_binds_complete_surface() -> None:
    verification = verify_measurement_qualification_data_package_lock(
        SERVER_LOCK,
        repository_root=ROOT,
    )
    observed = frozenset(item.relative_path for item in verification.files)

    assert verification.verified is True
    assert observed == MEASUREMENT_QUALIFICATION_DATA_PACKAGE_PATHS
    assert observed >= {
        "configs/paths.example.yaml",
        "configs/recoverability/measurement_qualification_v1.yaml",
        "configs/recoverability/stage2_v2_external_evidence_anchor.yaml",
        "experiments/recoverability_v1/09_generate_measurement_qualification_data.py",
        "src/compbias/recoverability/measurement_qualification.py",
        "src/compbias/recoverability/measurement_qualification_data.py",
        "src/compbias/recoverability/measurement_qualification_server.py",
        "src/compbias/recoverability/stage2_v2_anchor.py",
    }
    assert all(not path.startswith("tests/") for path in observed)


def test_measurement_qualification_data_lock_rejects_subsets_and_noncanonical_paths(
    tmp_path: Path,
) -> None:
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
        verify_measurement_qualification_data_package_lock(
            subset,
            repository_root=ROOT,
        )


def test_measurement_qualification_data_script_has_canonical_fail_closed_paths() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "configs/paths.yaml" in source
    assert "configs/paths.example.yaml" in source
    assert "measurement_qualification_v1" in source
    assert "stage2-v2-external-evidence.json" in source
    assert "COMPBIAS_" in source
    assert "attempted.json" in source
    assert "open(\"x\"" in source
    assert "model_calls" in source
    assert "hypothesis_tested" in source
    assert "confirmatory_execution_authorized" in source
