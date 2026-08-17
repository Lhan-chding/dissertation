from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from compbias.recoverability.phase_c_world_recovery_execution import (
    PHASE_C_WORLD_RECOVERY_LOCK_PATH,
    PHASE_C_WORLD_RECOVERY_PACKAGE_PATHS,
    verify_phase_c_world_recovery_package_lock,
)

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "experiments/recoverability_v1/20_phase_c_world_recovery_preflight.py"
EXECUTE = ROOT / "experiments/recoverability_v1/21_run_phase_c_world_recovery.py"
SERVER_LOCK = ROOT / PHASE_C_WORLD_RECOVERY_LOCK_PATH


def test_world_recovery_lock_binds_the_complete_twelve_call_surface() -> None:
    verification = verify_phase_c_world_recovery_package_lock(SERVER_LOCK, repository_root=ROOT)
    observed = frozenset(item.relative_path for item in verification.files)
    assert verification.verified is True
    assert observed == PHASE_C_WORLD_RECOVERY_PACKAGE_PATHS
    assert observed >= {
        "configs/recoverability/phase_c_world_recovery_v1.yaml",
        "configs/recoverability/server_package_lock_phase_c_arms_v3.yaml",
        "experiments/recoverability_v1/20_phase_c_world_recovery_preflight.py",
        "experiments/recoverability_v1/21_run_phase_c_world_recovery.py",
        "prompts/world_recovery_v1_main.system.txt",
        "prompts/no_cue.user.template.txt",
        "prompts/valid_cue.user.template.txt",
        "src/compbias/recoverability/phase_c_world_recovery.py",
        "src/compbias/recoverability/phase_c_world_recovery_execution.py",
    }
    assert all(not path.startswith("tests/") for path in observed)


def test_world_recovery_lock_rejects_noncanonical_subset(tmp_path: Path) -> None:
    subset = tmp_path / "subset.yaml"
    subset.write_text(
        "schema_version: 1\nfiles:\n"
        "  - path: configs/recoverability/recoverability_v1.yaml\n"
        "    sha256: " + "0" * 64 + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical"):
        verify_phase_c_world_recovery_package_lock(subset, repository_root=ROOT)


def test_world_recovery_preflight_is_metadata_only_and_capped_at_twelve() -> None:
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
    assert "recoverability_phase_c_world_recovery_metadata_preflight" in source
    assert 'open("x"' in source
    assert "load_local_qwen" not in source
    assert '"model_call_cap": 12' in source
    assert '"scale_authorized": False' in source


def test_world_recovery_runner_is_text_only_greedy_and_capped_at_twelve() -> None:
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
        "--system-prompt",
        "--server-package-lock",
        "--preflight-report",
        "--screen-result",
    ):
        assert argument in result.stdout
    blocked = subprocess.run(
        [sys.executable, str(EXECUTE)], cwd=ROOT, capture_output=True, text=True
    )
    assert blocked.returncode == 2
    assert "BLOCKED" in blocked.stdout
    source = EXECUTE.read_text(encoding="utf-8")
    assert "decode_text_qwen_greedy_once" in source
    assert "len(calls) != 12" in source
    assert "do_sample=False" in source
    assert "max_format_retries = 0" in source
    assert "attempted.json" in source
    assert "manifest.hidden.jsonl" in source
    assert "manifest.public.jsonl" in source
    assert 'open("x"' in source
    assert source.index("verify_phase_c_screen_artifacts(") < source.index("load_local_qwen(")
    assert source.index('open("x"') < source.index("load_local_qwen(")
    assert "train(" not in source
    assert '"hypothesis_tested": False' in source
    assert '"scale_authorized": False' in source
