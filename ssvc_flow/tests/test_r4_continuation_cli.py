"""Continuation entry points retain explicit authorization and warning provenance."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _training_args(tmp_path):
    result = ["--allow-training", "--out", str(tmp_path / "new-stage")]
    for flag in ("data-root", "r0-dir", "r1-run", "supplement-dir", "r2-dir", "r3-dir"):
        result.extend(["--" + flag, str(tmp_path / flag)])
    return result


def test_cli_forwards_continuation_and_resume_preserving_warning_status(
    tmp_path, monkeypatch, capsys
):
    from src import r4_runtime, train_two_arm_pilot

    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "COMPLETED_WITH_DIAGNOSTIC_WARNINGS", "warning_count": 1}

    monkeypatch.setattr(r4_runtime, "run_r4", capture)
    command = [
        *_training_args(tmp_path),
        "--resume",
        "--continuation-parent",
        str(tmp_path / "parent"),
        "--continuation-decision",
        str(tmp_path / "decision.json"),
    ]
    assert train_two_arm_pilot.main(command) == 0
    args, kwargs = calls[0]
    assert kwargs == {
        "allow_training": True,
        "resume": True,
        "continuation_parent": tmp_path / "parent",
        "continuation_decision": tmp_path / "decision.json",
    }
    assert args[0]["R4"]["allow_training"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETED_WITH_DIAGNOSTIC_WARNINGS"


@pytest.mark.parametrize("provided", ["--continuation-parent", "--continuation-decision"])
def test_cli_rejects_unpaired_continuation_arguments_before_runner(
    tmp_path, monkeypatch, capsys, provided
):
    from src import r4_runtime, train_two_arm_pilot

    monkeypatch.setattr(r4_runtime, "run_r4", lambda *a, **k: pytest.fail("runner must not start"))
    with pytest.raises(SystemExit) as error:
        train_two_arm_pilot.main([*_training_args(tmp_path), provided, str(tmp_path / "value")])
    assert error.value.code == 2
    assert "must be supplied together" in capsys.readouterr().err
    assert not (tmp_path / "new-stage").exists()


def test_continuation_does_not_imply_allow_training(tmp_path, monkeypatch, capsys):
    from src import r4_runtime, train_two_arm_pilot

    monkeypatch.setattr(r4_runtime, "run_r4", lambda *a, **k: pytest.fail("runner must not start"))
    with pytest.raises(SystemExit) as error:
        train_two_arm_pilot.main(
            [
                "--continuation-parent",
                str(tmp_path / "parent"),
                "--continuation-decision",
                str(tmp_path / "decision.json"),
                "--out",
                str(tmp_path / "new-stage"),
            ]
        )
    assert error.value.code == 2
    assert "--allow-training" in capsys.readouterr().err
    assert not (tmp_path / "new-stage").exists()


def test_continuation_preflight_audits_parent_and_reports_inherited_budget(tmp_path, monkeypatch):
    from src import next_stage_preflight as module
    from src import r4_continuation, r4_gate, r4_runtime

    environment = {"packages": "fixture"}
    gate = {"config": {"data_root": "/data"}, "binding": {"certified": True}}
    r2 = {"environment": environment}
    r3 = {"environment": environment, "plan_hash": "cold-plan"}
    source = {"source_commit": "reviewed-source"}
    parent = {
        "status": "PASS_STOPPED_AUDIT",
        "environment": environment,
        "audit_hash": "verified-parent",
        "plan_hash": "r4-plan",
        "inherited_counts": {
            "new_outputs": 2272,
            "optimizer_updates": 35,
            "training_rollouts": 1120,
            "shared_step0_outputs": 576,
            "step32_outputs": 576,
            "step64_outputs": 0,
        },
    }
    decision = tmp_path / "decision.json"
    calls = []
    monkeypatch.setattr(module, "validate_prerequisites", lambda *args: gate)
    monkeypatch.setattr(module, "validate_config_against_gate", lambda c, g: g["config"])
    monkeypatch.setattr(module, "validate_r2_gate", lambda *args: r2)
    monkeypatch.setattr(module, "validate_r3_cold_gate", lambda *args: r3)
    monkeypatch.setattr(module, "validate_runtime_environment", lambda *args: environment)
    monkeypatch.setattr(module, "_source", lambda: source)
    monkeypatch.setattr(
        r4_runtime,
        "_load_plan",
        lambda *args: ({"plan_hash": "r4-plan", "counts": {"updates": 128}}, {}, {}),
    )

    def audit(*args):
        calls.append(("audit", args))
        return parent

    def validate(*args):
        calls.append(("decision", args))
        return {"validated": True}

    monkeypatch.setattr(r4_gate, "audit_stopped_r4", audit, raising=False)
    monkeypatch.setattr(r4_continuation, "validate_continuation_decision", validate)
    kwargs = {
        "phase": "R4-continuation",
        "r3_dir": "cold",
        "r4_dir": "parent",
        "continuation_decision": decision,
    }
    result = module.preflight({}, "/data", "r0", "r1", "supp", "r2", **kwargs)
    assert calls == [("audit", ("parent", gate, r2, r3)), ("decision", (decision, parent, source))]
    assert result["budgets"]["new_outputs"] == 14400
    assert result["budgets"]["optimizer_updates"] == 128
    assert result["inherited_budgets"] == parent["inherited_counts"]
    assert result["remaining_budgets"] == {
        "new_outputs": 12128,
        "optimizer_updates": 93,
        "training_rollouts": 2976,
        "shared_step0_outputs": 0,
        "step32_outputs": 576,
        "step64_outputs": 8576,
    }
    assert result["gate_binding"]["r4_parent"] == parent
    assert result["gate_binding"]["continuation_decision"] == {"validated": True}
    assert result["phase"] == "R4-continuation_PREFLIGHT"
    assert result["source"] == source
    assert not result["model_loaded"] and not result["gpu_work_requested"]
    assert not decision.exists()
    parent["environment"] = {"packages": "changed"}
    with pytest.raises(ValueError, match="environment"):
        module.preflight({}, "/data", "r0", "r1", "supp", "r2", **kwargs)
    parent["environment"] = environment
    parent["plan_hash"] = "changed-plan"
    with pytest.raises(ValueError, match="plan"):
        module.preflight({}, "/data", "r0", "r1", "supp", "r2", **kwargs)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"r4_dir": "parent", "continuation_decision": "decision"}, "R3-cold"),
        ({"r3_dir": "cold", "continuation_decision": "decision"}, "parent"),
        ({"r3_dir": "cold", "r4_dir": "parent"}, "decision"),
    ],
)
def test_continuation_preflight_requires_complete_inputs(arguments, message):
    from src.next_stage_preflight import preflight

    with pytest.raises(ValueError, match=message):
        preflight({}, "/data", "r0", "r1", "supp", "r2", phase="R4-continuation", **arguments)


def test_preflight_cli_forwards_decision_and_preserves_existing_output(tmp_path, monkeypatch):
    from src import next_stage_preflight as module

    calls = []
    monkeypatch.setattr(module, "load_yaml", lambda path: {"loaded": True})

    def preflight(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "status": "PASS",
            "phase": "R4-continuation_PREFLIGHT",
            "execution_kind": "CPU_AUDIT",
        }

    monkeypatch.setattr(module, "preflight", preflight)
    arguments = ["--phase", "R4-continuation"]
    for name in (
        "data-root",
        "r0-dir",
        "r1-run",
        "supplement-dir",
        "r2-dir",
        "r3-dir",
        "r4-dir",
        "continuation-decision",
        "out",
    ):
        arguments.extend(["--" + name, str(tmp_path / name)])
    assert module.main(arguments) == 0
    assert calls[0][1] == {
        "phase": "R4-continuation",
        "r3_dir": tmp_path / "r3-dir",
        "r4_dir": tmp_path / "r4-dir",
        "continuation_decision": tmp_path / "continuation-decision",
    }
    before = (tmp_path / "out").read_bytes()
    with pytest.raises(FileExistsError):
        module.main(arguments)
    assert (tmp_path / "out").read_bytes() == before
    assert len(calls) == 1


def _slurm_fixture(tmp_path):
    project = tmp_path / "project"
    checkout = project / "dissertation-ssvc/ssvc_flow"
    checkout.mkdir(parents=True)
    parent = project / "runs/NEXT_20260909/R4_server_123_attempt_0/R4"
    parent.mkdir(parents=True)
    (parent / "identity.json").write_text('{"parent":true}\n')
    decision = project / "continuation-decision.json"
    decision.write_text('{"reviewed":true}\n')
    cold = project / "runs/NEXT_20260909/R3_cold_server_100_attempt_0/R3_cold"
    cold.mkdir(parents=True)
    (cold / "status.json").write_text('{"status":"PASS"}\n')
    binary = tmp_path / "bin"
    binary.mkdir()
    git = binary / "git"
    git.write_text("#!/bin/sh\nprintf '%s\\n' fixture-source\n")
    git.chmod(0o755)
    realpath = binary / "realpath"
    realpath.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
        "print(Path(sys.argv[-1]).resolve(strict=True))\n"
    )
    realpath.chmod(0o755)
    python = binary / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == '-m':\n"
        "    args=sys.argv[3:]\n"
        "    out=Path(args[args.index('--out')+1])\n"
        "    out.mkdir(parents=True,exist_ok=True)\n"
        "    (out/'received_args.json').write_text(json.dumps(args))\n"
        "    if os.environ.get('SSVC_TEST_STAGE_STATUS') != 'MISSING':\n"
        "        status={'status':os.environ['SSVC_TEST_STAGE_STATUS']}\n"
        "        (out/'status.json').write_text(json.dumps(status))\n"
        "elif sys.argv[1] == '-':\n"
        "    sys.argv=sys.argv[1:]\n"
        "    exec(sys.stdin.read())\n"
        "else:\n"
        "    raise SystemExit('unexpected Python invocation')\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "SSVC_WORK": str(project),
        "SSVC_PYTHON": str(python),
        "SLURM_JOB_ID": "456",
        "SLURM_RESTART_COUNT": "0",
        "SSVC_R3_COLD_STAGE": str(cold),
        "SSVC_R4_CONTINUATION_PARENT": str(parent),
        "SSVC_R4_CONTINUATION_DECISION": str(decision),
        "SSVC_TEST_STAGE_STATUS": "COMPLETED_WITH_DIAGNOSTIC_WARNINGS",
    }
    env.pop("SSVC_R4_RESUME_STAGE", None)
    return project, parent, decision, env


def test_slurm_continuation_uses_new_stage_and_keeps_warning_status(tmp_path):
    project, parent, decision, env = _slurm_fixture(tmp_path)
    result = subprocess.run(
        ["bash", str(Path("scripts/ntu_r4.sbatch").resolve())],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wrapper = project / "runs/NEXT_20260909/R4_server_456_attempt_0"
    new_stage = wrapper / "R4"
    args = json.loads((new_stage / "received_args.json").read_text())
    assert args[args.index("--continuation-parent") + 1] == str(parent.resolve())
    assert args[args.index("--continuation-decision") + 1] == str(decision.resolve())
    assert args[args.index("--out") + 1] == str(new_stage)
    assert set(p.name for p in parent.iterdir()) == {"identity.json"}
    state = dict(line.split("=", 1) for line in (wrapper / "result.txt").read_text().splitlines())
    assert state["state"] == "COMPLETED_WITH_DIAGNOSTIC_WARNINGS"
    assert state["stage_status"] == "COMPLETED_WITH_DIAGNOSTIC_WARNINGS"
    assert state["exit_code"] == state["archive_exit_code"] == "0"


@pytest.mark.parametrize("status", ["MISSING", "FAIL", "DIAGNOSTIC_STOP"])
def test_slurm_zero_process_exit_does_not_hide_missing_or_failed_stage(tmp_path, status):
    project, _, _, env = _slurm_fixture(tmp_path)
    env["SSVC_TEST_STAGE_STATUS"] = status
    result = subprocess.run(
        ["bash", str(Path("scripts/ntu_r4.sbatch").resolve())],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    record = project / "runs/NEXT_20260909/R4_server_456_attempt_0/result.txt"
    state = dict(line.split("=", 1) for line in record.read_text().splitlines())
    assert state["state"] == "FAILED"
    assert state["stage_status"] == ("UNAVAILABLE" if status == "MISSING" else status)


def test_slurm_resume_continuation_keeps_parent_read_only(tmp_path):
    project, parent, _, env = _slurm_fixture(tmp_path)
    child = project / "runs/NEXT_20260909/R4_server_455_attempt_0/R4"
    child.mkdir(parents=True)
    (child / "identity.json").write_text('{"continuation":true}\n')
    env["SSVC_R4_RESUME_STAGE"] = str(child)
    result = subprocess.run(
        ["bash", str(Path("scripts/ntu_r4.sbatch").resolve())],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads((child / "received_args.json").read_text())
    assert "--resume" in args
    assert args[args.index("--out") + 1] == str(child.resolve())
    assert args[args.index("--continuation-parent") + 1] == str(parent.resolve())
    assert set(p.name for p in parent.iterdir()) == {"identity.json"}


@pytest.mark.parametrize("started", [False, True])
def test_slurm_resumes_verified_preparation_before_identity_only(tmp_path, started):
    project, _, _, env = _slurm_fixture(tmp_path)
    child = project / "runs/NEXT_20260909/R4_server_455_attempt_0/R4"
    child.mkdir(parents=True)
    for name in (
        "continuation_binding.json",
        "continuation_decision.json",
        "logical_sampling_identity.json",
    ):
        (child / name).write_text("{}\n")
    if started:
        (child / "runtime_lock.json").write_text("{}\n")
    env["SSVC_R4_RESUME_STAGE"] = str(child)
    result = subprocess.run(
        ["bash", str(Path("scripts/ntu_r4.sbatch").resolve())],
        env=env,
        capture_output=True,
        text=True,
    )
    if started:
        assert result.returncode == 2
        assert not (child / "received_args.json").exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        args = json.loads((child / "received_args.json").read_text())
        assert "--resume" in args
        assert args[args.index("--out") + 1] == str(child.resolve())


@pytest.mark.parametrize("bad_case", ["resume-parent", "unpaired"])
def test_slurm_rejects_parent_as_output_or_unpaired_decision(tmp_path, bad_case):
    _, parent, _, env = _slurm_fixture(tmp_path)
    if bad_case == "resume-parent":
        env["SSVC_R4_RESUME_STAGE"] = str(parent)
    else:
        env.pop("SSVC_R4_CONTINUATION_DECISION")
    before = (parent / "identity.json").read_bytes()
    result = subprocess.run(
        ["bash", str(Path("scripts/ntu_r4.sbatch").resolve())],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert set(p.name for p in parent.iterdir()) == {"identity.json"}
    assert (parent / "identity.json").read_bytes() == before
