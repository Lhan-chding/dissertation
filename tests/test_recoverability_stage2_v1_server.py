from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from compbias.recoverability.stage2_v1 import verify_stage2_v1_server_package_lock


ROOT = Path(__file__).resolve().parents[1]
SERVER_LOCK = ROOT / "configs" / "recoverability" / "server_package_lock_stage2_v1.yaml"


def test_stage2_v1_probe_cli_is_explicit_one_shot_and_development_only() -> None:
    script = ROOT / "experiments" / "recoverability_v1" / "05_stage2_v1_probe.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    for argument in (
        "--execute",
        "--paths",
        "--runtime",
        "--probe-config",
        "--stage1-result",
        "--server-package-lock",
        "--preflight-report",
        "--external-evidence",
        "--v03-records",
        "--stage1-preflight",
        "--stage1-console-log",
        "--stage1-report",
        "--stage1-records",
    ):
        assert argument in completed.stdout

    blocked = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 2
    assert "BLOCKED" in blocked.stdout

    source = script.read_text(encoding="utf-8")
    assert source.index("verify_stage1_v2_frozen_artifacts(") < source.index(
        "load_local_qwen("
    )
    assert "decode_qwen_once(" not in source
    assert "decode_text_qwen_once(" in source


def test_stage2_v1_server_lock_binds_complete_execution_surface() -> None:
    verification = verify_stage2_v1_server_package_lock(SERVER_LOCK, repository_root=ROOT)
    paths = {item.relative_path for item in verification.files}

    assert paths >= {
        "configs/recoverability/stage1_v2_frozen_result.yaml",
        "configs/recoverability/stage2_v1_probe.yaml",
        "configs/recoverability/server_package_lock_stage2_v1.yaml",
        "experiments/recoverability_v1/00_stage2_v1_preflight.py",
        "experiments/recoverability_v1/05_stage2_v1_probe.py",
        "src/compbias/recoverability/stage2_v1.py",
    }


def test_stage2_v1_preflight_is_metadata_only_and_uses_new_lock() -> None:
    script = ROOT / "experiments" / "recoverability_v1" / "00_stage2_v1_preflight.py"
    source = script.read_text(encoding="utf-8")

    assert "server_package_lock_stage2_v1.yaml" in source
    assert "recoverability_stage2_v1_metadata_preflight" in source
    assert "load_local_qwen" not in source
    assert "torch" not in source
    assert "cuda" not in source.lower()
