from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

from compensability_v4.training.phase6 import verify_phase6_package_lock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/v4"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_phase6_server_scripts_expose_data_preflight_training_and_evaluation() -> None:
    prepare = _load("test_phase6_prepare", "11_prepare_phase6_rl_data.py")
    train = _load("test_phase6_train", "12_train_phase6_grpo.py")
    evaluate = _load("test_phase6_evaluate", "13_evaluate_phase6_rl.py")

    assert callable(prepare.main)
    assert callable(train.main)
    assert callable(evaluate.main)
    source = (SCRIPT_DIR / "12_train_phase6_grpo.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert any(node.id == "GRPOTrainer" for node in ast.walk(tree) if isinstance(node, ast.Name))
    assert "--preflight-only" in source
    assert "COMPBIAS_V4_PHASE6_RL_ACK" in source
    assert "resume_from_checkpoint" in source


def test_phase6_package_lock_closes_all_entrypoints() -> None:
    paths = (
        "configs/recoverability/v4_phase_6.yaml",
        "docs/Qwen25VL_Constraint_Assimilation_Natural_State_RL_Codex_Plan_v4.md",
        "docs/QWEN_V4_SERVER_HANDOFF.md",
        "pyproject.toml",
        "requirements-gpu.lock.txt",
        "scripts/v4/11_prepare_phase6_rl_data.py",
        "scripts/v4/12_train_phase6_grpo.py",
        "scripts/v4/13_evaluate_phase6_rl.py",
        "src/compensability_v4/training/phase6.py",
    )
    digest = verify_phase6_package_lock(
        lock_path=ROOT / "configs/recoverability/v4/server_package_lock_phase_6.yaml",
        repository_root=ROOT,
        expected_paths=paths,
    )
    assert len(digest) == 64
