import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def module():
    return importlib.import_module("src.followup_protocol")


def test_frozen_design_budget_and_no_execution():
    p = module()
    design = p.load_design(Path("configs/mechanism_followup.yaml"))
    plan = p.build_plan(design)
    assert plan["budgets"]["candidate_updates"] == 41
    assert plan["budgets"]["total_adam_calls"] == 47
    assert plan["budgets"]["backward_sequences"] == 1504
    assert plan["budgets"]["direct_logical_candidates"] == 18
    assert plan["budgets"]["direct_outputs"] == 13824
    assert plan["budgets"]["s2_new_runs"] == 7
    assert plan["budgets"]["s2_training_outputs"] == 14336
    assert plan["budgets"]["s2_evaluation_outputs"] == 62272
    assert plan["status"] == "PLANNED_NOT_EXECUTED"
    assert not plan["gpu_started"] and not plan["training_started"]
    assert plan["design"] == design


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("execution", "allow_gpu", True),
        ("execution", "allow_training", True),
        ("optimization", "B", True),
        ("optimization", "lr", float("nan")),
        ("reward", "epsilon", 1e-8),
        ("model", "revision", "main"),
        ("S1", "maximum_direct_outputs_before_dedup", 1),
    ],
)
def test_design_cannot_silently_change_frozen_contract(section, key, value):
    p = module()
    d = p.load_design("configs/mechanism_followup.yaml")
    d[section][key] = value
    with pytest.raises(ValueError):
        p.validate_design(d)


def test_candidate_identity_rejects_bool_unknown_and_alias_paths():
    p = module()
    for candidate in [
        dict(id="../joint", policy="joint", auxiliary_weight=0),
        dict(id="joint_0", policy="other", auxiliary_weight=0),
        dict(id="joint_0", policy="joint", auxiliary_weight=True),
    ]:
        with pytest.raises(ValueError):
            p.CandidateSpec.from_mapping(candidate)


def test_safe_json_no_clobber_nonfinite_and_symlink(tmp_path):
    p = module()
    target = tmp_path / "result.json"
    p.write_new_json(target, {"status": "CHECKED"})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        p.write_new_json(target, {"status": "PASS"})
    assert target.read_bytes() == before
    with pytest.raises(ValueError):
        p.write_new_json(tmp_path / "nan.json", {"x": float("nan")})
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        p.write_new_json(link / "escape.json", {})
    assert not (tmp_path / "escape.json").exists()


def test_cli_plan_and_import_do_not_import_torch_or_adapter(tmp_path):
    code = (
        "import sys;from src.followup_cli import main;"
        "main(['plan','--design','configs/mechanism_followup.yaml','--out',sys.argv[1]]);"
        "assert 'torch' not in sys.modules;assert 'src.model_adapters.base' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "plan.json")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "plan.json").read_text())["gpu_started"] is False


@pytest.mark.parametrize(
    "command,args",
    [
        ("run-s1", ["--bank", "bank03"]),
        ("run-s2", ["--seed", "29", "--arm", "X_BASE"]),
        ("smoke", []),
        ("eval-historical-r2", ["--arm", "X_BASE"]),
    ],
)
def test_real_commands_reject_without_permission_before_reading_plan(tmp_path, command, args):
    from src.followup_cli import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                command,
                "--validated-plan",
                str(tmp_path / "missing.json"),
                "--out",
                str(tmp_path / "out"),
                *args,
            ]
        )
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()


def test_s2_requires_separate_training_flag(tmp_path):
    from src.followup_cli import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run-s2",
                "--validated-plan",
                str(tmp_path / "absent.json"),
                "--out",
                str(tmp_path / "out"),
                "--seed",
                "29",
                "--arm",
                "X_BASE",
                "--allow-gpu-execution",
            ]
        )
    assert exc.value.code == 2


def test_slurm_script_has_no_resource_guesses_and_valid_syntax(tmp_path):
    p = module()
    script = p.render_slurm(stage="S1")
    path = tmp_path / "followup.sbatch"
    path.write_text(script)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    assert "#SBATCH --partition" not in script and "#SBATCH --gres" not in script
    assert "--allow-gpu-execution" in script and "SLURM_JOB_ID" in script
    assert "sbatch " not in script
