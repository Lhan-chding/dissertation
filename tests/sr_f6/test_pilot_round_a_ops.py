"""Synthetic outer-operation tests; never invoke Slurm or model production."""

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6.interface_screen_dispatch import capacity


def data(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def load():
    path = Path(__file__).resolve().parents[2] / "scripts/sr_f6_pilot_ops/round_a.py"
    spec = importlib.util.spec_from_file_location("round_a_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path, monkeypatch):
    module = load()
    project = tmp_path.resolve()
    for name, relative in (
        ("PROJECT", "."),
        ("ROOT", "run"),
        ("OPS", "operations"),
        ("CODE", "sr_f6_code_b7e7b06"),
    ):
        monkeypatch.setattr(module, name, project / relative)
    module.CODE.mkdir()
    monkeypatch.setattr(module, "PREPARE", "prepare")
    paths = []
    for parent in ("b0", "b1", "b2"):
        for arm in ("U11", "C_SHUF", "U10"):
            relative = f"PHASE_RUNS/PILOT_V3/PILOT_FROZEN/runs/{parent}/r0/{arm}"
            directory = module.ROOT / relative
            directory.mkdir(parents=True)
            for name in ("ROLLOUT_REGISTRATION.json", "RUN_IDENTITY.json"):
                (directory / name).write_text('{"scope":"SYNTHETIC_TEST_ONLY"}')
            paths.append(dict(parent=parent, arm=arm, audit_root=relative))
    launcher = module.CODE / "scripts/sr_f6/run_pilot_v3_pair.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# SYNTHETIC_TEST_ONLY: never executed\n")
    values = {
        "PHASES/PILOT_V3/IDENTITY.json": dict(
            revision=module.PRODUCER,
            deployment=str(module.CODE),
            root=str(module.ROOT),
            phase="PILOT_V3",
        ),
        "PHASES/PILOT_V3/SOURCE_MANIFEST.json": {
            "files": [
                dict(
                    path="scripts/sr_f6/run_pilot_v3_pair.sh",
                    bytes=launcher.stat().st_size,
                    sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
                )
            ]
        },
        "PHASES/PILOT_V3/ADMISSION.json": dict(
            status="PASS",
            scope="PILOT_V3_REGISTERED_ONLY",
            phase_identity_sha256=module.PHASE,
            pilot_spec_sha256=module.SPEC,
            training_authorized=True,
            formal_authorized=False,
            test_authorized=False,
        ),
        "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/PILOT_SPEC_FROZEN.json": dict(
            phase="PILOT_V3", paths=paths
        ),
        "PHASE_RUNS/PILOT_V3/JOBS/PLAN.json": dict(
            phase="PILOT_V3", queues={"A": module.QUEUES}, workers=2, gpus_per_worker=1
        ),
        "PHASE_RUNS/PILOT_V3/INPUTS_V3/OVERLAY_MANIFEST.json": {"scope": "SYNTHETIC_TEST_ONLY"},
    }
    root_pins = {}
    for relative, value in values.items():
        target = module.ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data(value))
        root_pins[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "ROOT_PINS", root_pins)
    prepared = {
        "PILOT_PREPARE_RESULT.json": dict(status="READY_FOR_REGISTERED_PILOT_ONLY"),
        "CPU_PREPARE_SUBMISSION.json": dict(status="SUBMITTED", job_id="233935"),
    }
    prepare_pins = {}
    for name, value in prepared.items():
        target = project / module.PREPARE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data(value))
        prepare_pins[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "PREPARE_PINS", prepare_pins)
    for key in list(os.environ):
        if key.startswith(("SBATCH_", "SLURM_", "SRF6_")) or key in (
            "BASH_ENV",
            "ENV",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "PYTHONSTARTUP",
            "PYTHONINSPECT",
            "PYTHONOPTIMIZE",
        ):
            monkeypatch.delenv(key)
    monkeypatch.setattr(module.getpass, "getuser", lambda: module.USER)
    snapshot = dict(
        gpu_limit=8,
        submit_limit=5,
        can_submit_two_gpu_job=True,
        reserved_gpus=1,
        submitted_by_user=1,
    )
    states = {"233935": ("COMPLETED", "0:0")}
    monkeypatch.setattr(
        module,
        "scheduler",
        lambda facts: (
            SimpleNamespace(capacity=lambda **kw: dict(snapshot)),
            SimpleNamespace(_job_states=lambda jobs: dict(states)),
        ),
    )
    monkeypatch.setattr(module, "query", lambda argv: "")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="234001\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    identity = dict(
        revision="d" * 40,
        deployment=str(project / "sr_f6_pilot_ops_code_ddddddd"),
        binding_sha256="e" * 64,
    )
    return module, identity, calls, snapshot, states


def test_bound_metadata_and_single_correct_submission(setup):
    module, identity, calls, _, _ = setup
    result = module.submit(identity)
    assert result["status"] == "SUBMITTED" and result["job_id"] == "234001"
    argv = calls[0]
    assert len(calls) == 1 and argv == module.argv_for(identity)
    for value in (
        "--gres=gpu:pro6000:2",
        "--cpus-per-task=8",
        "--mem=66G",
        "--time=23:00:00",
        "--no-requeue",
    ):
        assert value in argv
    assert "--job-name=LH-SRF6-PilotV3-A" in argv
    assert not any("ALL," in item or "RESUME" in item for item in argv)
    intent = json.loads((module.OPS / "A_ATTEMPT.json").read_text())
    assert intent["worker_attempt_template"] == "A-{job_id}" and intent["resume_tasks"] == []
    with pytest.raises(PermissionError, match="forbids retry"):
        module.submit(identity)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [("submitted_by_user", 5), ("reserved_gpus", 7), ("can_submit_two_gpu_job", False)],
)
def test_full_slots_or_gpu_quota_blocks_without_intent(setup, field, value):
    module, identity, calls, snapshot, _ = setup
    snapshot[field] = value
    assert module.submit(identity)["status"] == "NO_LEGAL_CAPACITY"
    assert calls == [] and not (module.OPS / "A_ATTEMPT.json").exists()


def test_unknown_capacity_blocks_without_intent(setup, monkeypatch):
    module, identity, calls, _, states = setup

    def unknown(**kwargs):
        raise PermissionError("Unknown reservation")

    monkeypatch.setattr(
        module,
        "scheduler",
        lambda facts: (
            SimpleNamespace(capacity=unknown),
            SimpleNamespace(_job_states=lambda jobs: states),
        ),
    )
    with pytest.raises(PermissionError, match="Unknown reservation"):
        module.submit(identity)
    assert calls == [] and not (module.OPS / "A_ATTEMPT.json").exists()


@pytest.mark.parametrize("name", ["A_ATTEMPT.json", "A_SUBMISSION.json", "A_STARTED.json"])
def test_dangling_partial_or_previous_operation_blocks(setup, name):
    module, identity, calls, _, _ = setup
    module.OPS.mkdir()
    (module.OPS / name).write_text("{}")
    before = (module.OPS / name).read_bytes()
    with pytest.raises(PermissionError, match="forbids retry"):
        module.submit(identity)
    assert calls == [] and (module.OPS / name).read_bytes() == before


@pytest.mark.parametrize(
    "error", [subprocess.TimeoutExpired(["sbatch"], 25), OSError("transport failed")]
)
def test_ambiguous_submission_retains_intent_and_unknown_without_retry(setup, monkeypatch, error):
    module, identity, calls, _, _ = setup

    def run(argv, **kwargs):
        calls.append(argv)
        raise error

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.submit(identity)["status"] == "SUBMISSION_UNKNOWN"
    assert (module.OPS / "A_ATTEMPT.json").exists() and (module.OPS / "A_SUBMISSION.json").exists()
    with pytest.raises(PermissionError, match="forbids retry"):
        module.submit(identity)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "key",
    ["SBATCH_GRES", "SBATCH_QOS", "SRF6_RESUME_TASK_0", "PYTHONSTARTUP", "BASH_ENV", "LD_PRELOAD"],
)
def test_injected_submission_environment_rejected(setup, monkeypatch, key):
    module, identity, calls, _, _ = setup
    monkeypatch.setenv(key, "injected")
    with pytest.raises(PermissionError, match="Injected"):
        module.submit(identity)
    assert calls == []


def test_changed_sealed_metadata_prevents_submission(setup):
    module, identity, calls, _, _ = setup
    (module.ROOT / "PHASES/PILOT_V3/ADMISSION.json").write_text("{}")
    with pytest.raises(PermissionError, match="Changed sealed"):
        module.submit(identity)
    assert calls == []


@pytest.mark.parametrize(
    "relative",
    [
        "JOBS/WORKERS/A",
        "JOBS/TASKS/PRIMARY-b0-U11",
        "PILOT_FROZEN/runs/b0/r0/U11/checkpoints",
        "PILOT_FROZEN/runs/b1/r0/C_SHUF/TRAINER_ATTEMPTS",
    ],
)
def test_prior_model_outputs_prevent_resampling(setup, relative):
    module, identity, calls, _, _ = setup
    (module.ROOT / "PHASE_RUNS/PILOT_V3" / relative).mkdir(parents=True)
    with pytest.raises(PermissionError, match="Prior"):
        module.submit(identity)
    assert calls == []


def test_preparation_must_leave_queue(setup, monkeypatch):
    module, identity, calls, _, _ = setup
    commands = []

    def query(argv):
        commands.append(argv)
        return "233935\n"

    monkeypatch.setattr(module, "query", query)
    with pytest.raises(PermissionError, match="still present"):
        module.submit(identity)
    assert calls == [] and commands == [["squeue", "-h", "-u", module.USER, "-o", "%i"]]


def test_real_original_capacity_counts_all_users_and_pending():
    module = load()
    requests = []

    def query(argv):
        requests.append(argv)
        if argv[0] == "sacctmgr":
            return module.QOS + "|5|gres/gpu:pro6000=8|"
        if argv[0] == "squeue":
            return "1|other|other-project|PENDING|" + module.QOS
        return "JobId=1 QOS=" + module.QOS + " ReqTRES=gres/gpu:pro6000=7 AllocTRES=(null)"

    result = capacity(query=query)
    assert result["reserved_gpus"] == 7 and not result["can_submit_two_gpu_job"]
    assert result["submitted_by_user"] == 0 and "-q" in requests[1] and "-u" not in requests[1]

    def unknown(argv):
        return (
            query(argv).replace("gres/gpu:pro6000=7", "cpu=8")
            if argv[0] == "scontrol"
            else query(argv)
        )

    with pytest.raises(PermissionError, match="Unknown GPU"):
        capacity(query=unknown)


def test_operation_identity_requires_actual_origin_and_exact_source(setup, monkeypatch):
    module, identity, _, _, _ = setup
    deployment = Path(identity["deployment"])
    files = {}
    for name in module.MEMBERS:
        path = deployment / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# SYNTHETIC_TEST_ONLY\n")
        files[name] = dict(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    binding = dict(
        schema="srf6-pilot-round-a-operations-v1",
        revision=identity["revision"],
        deployment=str(deployment),
        producer_revision=module.PRODUCER,
        files=files,
    )
    path = deployment / "OPERATION_BINDING.json"
    path.write_bytes(data(binding))
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(PermissionError, match="origin"):
        module.operation_identity(identity["revision"], deployment, pin)
    monkeypatch.setattr(module, "__file__", str(deployment / module.MEMBERS[0]))
    assert module.operation_identity(identity["revision"], deployment, pin)["binding_sha256"] == pin
    (deployment / module.MEMBERS[1]).write_text("changed")
    with pytest.raises(PermissionError, match="Operation bytes changed"):
        module.operation_identity(identity["revision"], deployment, pin)


def test_launch_only_executes_original_pair_with_bound_pins_and_project_cache(setup, monkeypatch):
    module, identity, _, _, _ = setup
    module.submit(identity)
    monkeypatch.setenv("SLURM_JOB_ID", "234001")
    monkeypatch.setenv("SLURM_JOB_NAME", module.JOB)
    monkeypatch.setenv("SLURM_EXPORT_ENV", "PATH,SRF6_OPS_REVISION")
    monkeypatch.setenv("PYTHONSTARTUP", "injected")
    monkeypatch.setenv("BASH_ENV", "injected")
    monkeypatch.setattr(module, "allocation", lambda job: "SYNTHETIC_TEST_ONLY allocation")
    executed = []
    monkeypatch.setattr(module.os, "execve", lambda *args: executed.append(args))
    module.launch(identity)
    binary, argv, env = executed[0]
    assert binary == "/bin/bash" and argv == [
        "/bin/bash",
        str(module.CODE / "scripts/sr_f6/run_pilot_v3_pair.sh"),
    ]
    assert env["SRF6_CODE"] == str(module.CODE) and env["SRF6_ROUND"] == "A"
    assert env["SLURM_EXPORT_ENV"] == "ALL"
    assert env["SRF6_PLAN_SHA256"] == module.PLAN and env["SRF6_PILOT_SPEC_SHA256"] == module.SPEC
    assert (
        env["SRF6_ATTEMPT"] == "A-234001" and "PYTHONSTARTUP" not in env and "BASH_ENV" not in env
    )
    for name in ("TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "CUDA_CACHE_PATH", "TORCH_HOME"):
        assert Path(env[name]).is_relative_to(module.OPS / "job-234001")
    with pytest.raises(PermissionError, match="No repeated launch"):
        module.launch(identity)
    assert len(executed) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("SLURM_RESTART_COUNT", "1"),
        ("SRF6_RESUME_TASK_0", ""),
        ("SLURM_JOB_NAME", "LH-SRF6-PilotV3-B"),
    ],
)
def test_launch_rejects_requeue_resume_or_other_round(setup, monkeypatch, key, value):
    module, identity, _, _, _ = setup
    module.submit(identity)
    monkeypatch.setenv("SLURM_JOB_ID", "234001")
    monkeypatch.setenv("SLURM_JOB_NAME", module.JOB)
    monkeypatch.setenv(key, value)
    with pytest.raises(PermissionError):
        module.launch(identity)
    assert not (module.OPS / "A_STARTED.json").exists()


def test_allocation_rejects_unknown_gpu_resources(setup, monkeypatch):
    module, _, _, _, _ = setup
    row = (
        f"JobId=234001 JobName={module.JOB} QOS={module.QOS} "
        f"Partition=cluster02 Account=rose NumCPUs=8 UserId={module.USER}(1)"
    )
    row += " ReqTRES=cpu=8,mem=66G,gres/gpu:pro6000=2 AllocTRES=cpu=8,mem=66G,gres/gpu:pro6000=2"
    monkeypatch.setattr(module, "query", lambda argv: row)
    assert module.allocation("234001") == row
    row = row.replace("AllocTRES=cpu=8,mem=66G,gres/gpu:pro6000=2", "AllocTRES=cpu=8,mem=66G")
    with pytest.raises(PermissionError, match="unknown two-GPU"):
        module.allocation("234001")


def test_partial_receipt_is_not_visible_to_fast_launch(setup, monkeypatch):
    module, _, _, _, _ = setup
    module.OPS.mkdir()
    target = module.OPS / "A_SUBMISSION.json"
    value = dict(status="SUBMITTED", job_id="SYNTHETIC_TEST_ONLY")
    observed = []
    original_link = module.os.link

    def partial_dump(value, stream, **kwargs):
        raw = json.dumps(value)
        stream.write(raw[:10])
        stream.flush()
        observed.append(("partial", target.exists()))
        stream.write(raw[10:])

    def publish(source, destination, **kwargs):
        observed.append(("before_publish", target.exists()))
        assert json.loads(Path(source).read_text()) == value
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(module.json, "dump", partial_dump)
    monkeypatch.setattr(module.os, "link", publish)
    module.write_once(target, value)
    assert observed == [("partial", False), ("before_publish", False)]
    assert json.loads(target.read_text()) == value
    assert {p.name for p in module.OPS.iterdir()} == {"A_SUBMISSION.json"}
    with pytest.raises(PermissionError, match="already exists"):
        module.write_once(target, dict(status="UNKNOWN"))
    assert json.loads(target.read_text()) == value
