from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = ROOT / "scripts/v5/study_c2"


@pytest.mark.parametrize("number", range(20, 41))
def test_all_registered_scripts_expose_cpu_fixture_surface(number: int) -> None:
    scripts = tuple(sorted(SCRIPT_ROOT.glob(f"{number}_*.py")))
    assert len(scripts) == 1
    completed = subprocess.run(
        [sys.executable, str(scripts[0]), "--fixture-dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"].endswith("_FIXTURE_OK")
    assert payload["gpu_invoked"] is False


def test_gpu_scripts_require_explicit_execution_or_preflight() -> None:
    for number in (23, 24, 25, 26, 31, 32, 33, 36, 37, 39):
        script = next(iter(SCRIPT_ROOT.glob(f"{number}_*.py")))
        completed = subprocess.run(
            [sys.executable, str(script)], cwd=ROOT, check=False, capture_output=True, text=True
        )
        assert completed.returncode == 2
        assert "BLOCKED" in completed.stdout
