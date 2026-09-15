import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src.modeling_v3 import cli

CONFIG = Path(__file__).resolve().parents[2] / "configs/modeling_v3/protocol.json"


def test_primary_family_dispatch_keeps_both_original_analysis_paths(tmp_path, monkeypatch):
    seen = {}

    def finalize(config, cpu_lock, cpu_analysis, vlm_lock, vlm_analysis, out):
        seen.update(
            cpu_lock=cpu_lock,
            cpu_analysis=cpu_analysis,
            vlm_lock=vlm_lock,
            vlm_analysis=vlm_analysis,
            out=out,
        )
        return {"status": "MOCK_DISPATCH_ONLY"}

    monkeypatch.setitem(
        sys.modules,
        "src.modeling_v3.primary_family",
        types.SimpleNamespace(finalize_primary_family=finalize),
    )
    assert (
        cli.main(
            [
                "finalize-primary-family",
                "--config",
                str(CONFIG),
                "--out",
                str(tmp_path / "out"),
                "--cpu-lock",
                "cpu-lock",
                "--cpu-analysis",
                "cpu-analysis",
                "--vlm-lock",
                "vlm-lock",
                "--vlm-analysis",
                "vlm-analysis",
            ]
        )
        == 0
    )
    assert seen["cpu_analysis"] == "cpu-analysis" and seen["vlm_analysis"] == "vlm-analysis"


def test_plan_imports_no_model_and_submits_no_jobs(tmp_path):
    script = (
        "from src.modeling_v3.cli import main; import sys; "
        "assert main(sys.argv[1:]) == 0; assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "plan-gpu",
            "--config",
            str(CONFIG),
            "--out",
            str(tmp_path / "plan"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "plan/PLAN.json").read_text())["status"] == "PLANNED_NOT_EXECUTED"


@pytest.mark.parametrize("training", [False, True])
def test_train_and_forks_dispatch_keyword_contract(tmp_path, monkeypatch, training):
    seen = {}

    def run(config, bindings, **kwargs):
        seen.update(kwargs)
        return {"status": "MOCK_DISPATCH_ONLY"}

    monkeypatch.setitem(
        sys.modules,
        "src.modeling_v3.vlm_campaign",
        types.SimpleNamespace(train_source=run, make_forks=run),
    )
    bindings = tmp_path / "bindings.json"
    bindings.write_text("{}")
    bank = tmp_path / "bank.json"
    bank.write_text("{}")
    base = [
        "--config",
        str(CONFIG),
        "--out",
        str(tmp_path / "out"),
        "--bindings",
        str(bindings),
        "--allow-gpu",
        "--acknowledge-new-experiment",
    ]
    if training:
        base += ["--allow-training"]
    assert cli.main(["train-source", *base, "--seed", "41001", "--arm", "X_BASE"]) == 0
    assert seen["allow_training"] is training and seen["seed"] == 41001
    assert (
        cli.main(
            ["make-forks", *base, "--checkpoint", "bound_checkpoint", "--bank-plan", str(bank)]
        )
        == 0
    )
    assert seen["allow_training"] is training


def test_gpu_without_flags_is_rejected_before_runtime(tmp_path):
    assert (
        cli.main(
            [
                "train-source",
                "--config",
                str(CONFIG),
                "--out",
                str(tmp_path / "out"),
                "--bindings",
                str(tmp_path / "missing"),
                "--seed",
                "41001",
                "--arm",
                "X_BASE",
            ]
        )
        == 2
    )


def test_failed_test_run_returns_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "dispatch", lambda args: {"status": "TESTS_FAILED", "exit_code": 7})
    assert cli.main(["test", "--config", str(CONFIG), "--out", str(tmp_path)]) == 7
