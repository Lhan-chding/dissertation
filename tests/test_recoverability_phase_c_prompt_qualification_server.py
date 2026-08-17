from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from compbias.recoverability.phase_c_prompt_qualification_execution import (
    PHASE_C_PROMPT_QUALIFICATION_LOCK_PATH,
    PHASE_C_PROMPT_QUALIFICATION_PACKAGE_PATHS,
    verify_phase_c_prompt_qualification_package_lock,
)

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "experiments/recoverability_v1/18_phase_c_prompt_qualification_preflight.py"
EXECUTE = ROOT / "experiments/recoverability_v1/19_run_phase_c_prompt_qualification.py"
SERVER_LOCK = ROOT / PHASE_C_PROMPT_QUALIFICATION_LOCK_PATH


def test_prompt_qualification_lock_binds_the_complete_low_cost_surface() -> None:
    verification = verify_phase_c_prompt_qualification_package_lock(
        SERVER_LOCK, repository_root=ROOT
    )
    observed = frozenset(item.relative_path for item in verification.files)
    assert verification.verified is True
    assert observed == PHASE_C_PROMPT_QUALIFICATION_PACKAGE_PATHS
    assert observed >= {
        "configs/recoverability/phase_c_prompt_qualification_v1.yaml",
        "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml",
        "experiments/recoverability_v1/18_phase_c_prompt_qualification_preflight.py",
        "experiments/recoverability_v1/19_run_phase_c_prompt_qualification.py",
        "src/compbias/recoverability/phase_c_prompt_qualification.py",
        "src/compbias/recoverability/phase_c_prompt_qualification_execution.py",
    }
    assert all(not path.startswith("tests/") for path in observed)


def test_prompt_qualification_lock_rejects_noncanonical_subset(tmp_path: Path) -> None:
    subset = tmp_path / "subset.yaml"
    subset.write_text(
        "schema_version: 1\nfiles:\n"
        "  - path: configs/recoverability/recoverability_v1.yaml\n"
        "    sha256: " + "0" * 64 + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical"):
        verify_phase_c_prompt_qualification_package_lock(subset, repository_root=ROOT)


def test_prompt_qualification_preflight_is_metadata_only() -> None:
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
    assert "recoverability_phase_c_prompt_qualification_metadata_preflight" in source
    assert 'open("x"' in source
    assert "load_local_qwen" not in source
    assert '"model_call_cap": 36' in source
    assert '"scale_authorized": False' in source


def test_prompt_qualification_runner_is_explicit_text_only_and_capped_at_36() -> None:
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
        "--qualification-config",
        "--screen-result",
        "--server-package-lock",
        "--preflight-report",
        "--screen-preflight",
        "--screen-attempt-marker",
        "--screen-dataset-root",
        "--screen-output-root",
        "--screen-console-log",
    ):
        assert argument in result.stdout
    blocked = subprocess.run(
        [sys.executable, str(EXECUTE)], cwd=ROOT, capture_output=True, text=True
    )
    assert blocked.returncode == 2
    assert "BLOCKED" in blocked.stdout
    source = EXECUTE.read_text(encoding="utf-8")
    assert "decode_text_qwen_seeded_once" in source
    assert "len(calls) != 36" in source
    assert "max_format_retries = 0" in source
    assert "attempted.json" in source
    assert 'open("x"' in source
    assert source.index("verify_phase_c_screen_artifacts(") < source.index("load_local_qwen(")
    assert source.index('open("x"') < source.index("load_local_qwen(")
    assert "train(" not in source
    assert '"hypothesis_tested": False' in source
    assert '"scale_authorized": False' in source
