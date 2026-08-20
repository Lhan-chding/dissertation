from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

from compensability_v4.qwen.phase6_runtime import verify_phase6_package_lock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/v4/11_prepare_phase6_rl.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_phase6_manifest_script_is_importable_and_execute_gated() -> None:
    module = _load("test_phase6_prepare_manifest", SCRIPT_PATH)

    assert callable(module.main)

    source = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_constants = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "BLOCKED: Phase 6 RL manifest preparation requires explicit --execute." in string_constants
    assert "--policy-support-summary-sha256" in source


def test_phase6_manifest_script_does_not_construct_rl_training() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "Trainer(" not in source
    assert "GRPOTrainer" not in source
    assert "optimizer.step" not in source
    assert "torch.optim" not in source


def test_phase6_package_lock_closes_manifest_freeze_surface() -> None:
    paths = (
        "configs/recoverability/v4_phase_6.yaml",
        "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
        "docs/QWEN_V4_SERVER_HANDOFF.md",
        "pyproject.toml",
        "requirements-gpu.lock.txt",
        "scripts/v4/11_prepare_phase6_rl.py",
        "src/compensability_v4/qwen/phase6_runtime.py",
    )
    digest = verify_phase6_package_lock(
        lock_path=ROOT / "configs/recoverability/v4/server_package_lock_phase_6.yaml",
        repository_root=ROOT,
        expected_paths=paths,
    )

    assert len(digest) == 64
