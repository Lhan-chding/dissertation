"""Local batch orchestration checks; no real GPU, model weights or SSH calls."""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/ntu_p1.sbatch"

FAKE_PYTHON = r'''
import json
import os
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
entry = {
    "args": args,
    "cwd": os.getcwd(),
    "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "offline": os.environ.get("HF_HUB_OFFLINE"),
    "tmp": os.environ.get("TMPDIR"),
}
with Path(os.environ["SSVC_TEST_CALLS"]).open("a") as stream:
    stream.write(json.dumps(entry) + "\n")


def destination(flag):
    return Path(args[args.index(flag) + 1])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def failure(stage, code):
    if os.environ.get("SSVC_TEST_FAIL_STAGE") == stage:
        print("fixture failure at " + stage, file=sys.stderr)
        raise SystemExit(code)


def inline():
    program = sys.stdin.read()
    if "import torch" in program:
        failure("preflight", 37)
        print("fixture CUDA preflight; not actual GPU evidence")
        return 0
    # Execute the real shell wrapper's stdlib validation, not a fake PASS gate.
    result = subprocess.run(
        [os.environ["SSVC_TEST_REAL_PYTHON"], *args],
        input=program,
        text=True,
        check=False,
    )
    if (os.environ.get("SSVC_TEST_FAIL_STAGE") == "status-write"
            and "hashlib.file_digest" in program):
        summary = Path(os.environ["SSVC_RUN_DIR"]) / "result.txt"
        summary.unlink()
        summary.mkdir()
    return result.returncode


def rollout():
    out = destination("--out")
    out.mkdir(parents=True, exist_ok=True)
    if "--dry-run" in args:
        write(out / "status.json", {"phase": "P1", "status": "DRY_RUN"})
        return
    failure("rollout", 55)
    mode = os.environ.get("SSVC_TEST_AUDIT_MODE", "real")
    write(out / "status.json", {"phase": "P1", "status": "PASS"})
    if mode != "missing-audit":
        write(out / "model_audit.json", {
            "passed": mode != "failed-audit",
            "execution_kind": "CPU_FAKE_ADAPTER_FIXTURE" if mode == "fake" else "REAL_CUDA_MODEL",
        })
    if mode != "missing-lock":
        write(out / "validated_runtime_lock.json", {
            "runtime_status": "P1_PASSED",
            "execution_kind": "REAL_CUDA_MODEL",
        })


if "-" in args:
    raise SystemExit(inline())
if "-m" not in args:
    raise SystemExit("unexpected Python command: " + repr(args))
module = args[args.index("-m") + 1]
if module == "pip":
    failure("pip", 19)
    print("No broken requirements found." if "check" in args else "torch==2.13.0+cu130")
elif module == "pytest":
    failure("pytest", 23)
    print("fixture CPU regression suite passed")
elif module == "src.generate_worlds":
    write(destination("--out") / "manifest.json", {"seed": 17, "fixture": True})
elif module == "src.rollout":
    rollout()
elif module == "src.report":
    failure("report", 29)
    write(destination("--out") / "phase_status.json", {"fixture": True})
else:
    raise SystemExit("unexpected module: " + module)
'''


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def batch(tmp_path):
    work = tmp_path / "personal work"
    repo = work / "dissertation-ssvc/ssvc_flow"
    (repo / "configs").mkdir(parents=True)
    (repo / "tests").mkdir()
    shutil.copy(PROJECT / "configs/locked.json", repo / "configs/locked.json")
    tools = tmp_path / "tools"
    fake = tools / "fake_python.py"
    executable(fake, FAKE_PYTHON)
    python = work / "envs/ssvc-py312/bin/python"
    executable(
        python,
        "#!/bin/sh\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(fake))
        + ' "$@"\n',
    )
    executable(tools / "nvidia-smi", "#!/bin/sh\nprintf 'fixture NVIDIA A6000\\n'\n")
    executable(tools / "git", "#!/bin/sh\nprintf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\n'\n")
    env = {
        "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "LANG": "C.UTF-8",
        "SSVC_WORK": str(work),
        "SSVC_PYTHON": str(python),
        "SSVC_TEST_REAL_PYTHON": sys.executable,
        "SSVC_TEST_CALLS": str(tmp_path / "calls.jsonl"),
        "SLURM_JOB_ID": "987654",
        "SLURM_RESTART_COUNT": "0",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    return {
        "work": work,
        "repo": repo,
        "tools": tools,
        "run": work / "runs/slurm-987654-attempt-0",
        "env": env,
    }


def run_batch(batch, **updates):
    env = {**batch["env"], **updates}
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=batch["repo"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def calls(batch):
    path = Path(batch["env"]["SSVC_TEST_CALLS"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def module_calls(batch, module):
    return [row for row in calls(batch) if ["-m", module] == row["args"][:2]]


def assert_failed(batch, result, expected=None):
    assert result.returncode != 0, result.stdout + result.stderr
    if expected is not None:
        assert result.returncode == expected, result.stdout + result.stderr
    summary = (batch["run"] / "result.txt").read_text()
    assert "state=FAILED" in summary
    assert f"exit_code={result.returncode}" in summary
    assert "state=PASS" not in summary


def test_batch_success_isolated_outputs_and_real_p1_gate(batch):
    result = run_batch(batch)
    assert result.returncode == 0, result.stdout + result.stderr
    run = batch["run"]
    assert (run / "checks").is_dir()
    assert (run / "data/generated/manifest.json").is_file()
    assert (run / "reports/phase_status.json").is_file()
    assert "state=PASS" in (run / "result.txt").read_text()
    assert "exit_code=0" in (run / "result.txt").read_text()
    rollouts = module_calls(batch, "src.rollout")
    assert len(rollouts) == 2
    assert "--dry-run" in rollouts[0]["args"]
    for row in rollouts:
        args = row["args"]
        assert args[args.index("--phase") + 1] == "smoke"
        assert args[args.index("--model") + 1] == "qwen35_9b"
    actual = rollouts[1]["args"]
    assert actual[actual.index("--out") + 1] == str(run / "runs/P1")
    assert actual[actual.index("--data-root") + 1] == str(run / "data/generated")
    assert rollouts[0]["args"][rollouts[0]["args"].index("--out") + 1] != str(run / "runs/P1")
    suite = module_calls(batch, "pytest")
    assert len(suite) == 1
    assert suite[0]["cuda_visible"] == ""
    assert suite[0]["offline"] == "1"
    assert len(module_calls(batch, "src.generate_worlds")) == 1
    assert not (batch["repo"] / "runs").exists()
    assert not (batch["repo"] / "data/generated").exists()
    archive = run.with_name(run.name + ".tar.gz")
    checksum = archive.with_name(archive.name + ".sha256").read_text().split()[0]
    assert checksum == hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
    assert run.name + "/runs/P1/model_audit.json" in names
    assert run.name + "/result.txt" not in names
    assert not any(name.startswith(run.name + "/tmp/") for name in names)
    assert all(row["tmp"] == str(run / "tmp") for row in calls(batch))


@pytest.mark.parametrize("stage,code", [("pip", 19), ("preflight", 37), ("pytest", 23)])
def test_batch_preparation_failure_stops_before_model(batch, stage, code):
    result = run_batch(batch, SSVC_TEST_FAIL_STAGE=stage)
    assert_failed(batch, result, code)
    assert not module_calls(batch, "src.rollout")
    assert not module_calls(batch, "src.generate_worlds")


def test_batch_model_failure_preserves_exit_code_and_builds_report(batch):
    result = run_batch(batch, SSVC_TEST_FAIL_STAGE="rollout")
    assert_failed(batch, result, 55)
    assert len(module_calls(batch, "src.rollout")) == 2
    assert module_calls(batch, "src.report")
    assert (batch["run"] / "reports/phase_status.json").exists()


@pytest.mark.parametrize("stage,code", [("report", 29), ("archive", 71)])
def test_batch_finalization_failure_cannot_be_reported_as_success(batch, stage, code):
    if stage == "archive":
        executable(batch["tools"] / "tar", "#!/bin/sh\nexit 71\n")
    result = run_batch(batch, SSVC_TEST_FAIL_STAGE=stage)
    assert_failed(batch, result, code)
    assert json.loads((batch["run"] / "runs/P1/status.json").read_text())["status"] == "PASS"


def test_batch_archive_failure_keeps_original_model_error(batch):
    executable(batch["tools"] / "tar", "#!/bin/sh\nexit 71\n")
    result = run_batch(batch, SSVC_TEST_FAIL_STAGE="rollout")
    assert_failed(batch, result, 55)


def test_batch_final_status_write_failure_is_nonzero(batch):
    result = run_batch(batch, SSVC_TEST_FAIL_STAGE="status-write")
    assert (batch["run"] / "result.txt").is_dir()
    assert result.returncode != 0, result.stdout + result.stderr


def test_batch_missing_interpreter_fails_without_creating_run(batch):
    result = run_batch(batch, SSVC_PYTHON=str(batch["work"] / "missing-python"))
    assert result.returncode != 0
    assert not (batch["work"] / "runs").exists()
    assert not calls(batch)


def test_batch_scheduler_restart_uses_fresh_attempt_without_overwrite(batch):
    batch["run"].mkdir(parents=True)
    sentinel = batch["run"] / "partial.jsonl"
    sentinel.write_text("original interrupted evidence\n")
    result = run_batch(batch, SLURM_RESTART_COUNT="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.read_text() == "original interrupted evidence\n"
    restarted = batch["run"].with_name("slurm-987654-attempt-1")
    assert "state=PASS" in (restarted / "result.txt").read_text()


@pytest.mark.parametrize("mode", ["fake", "missing-audit", "failed-audit", "missing-lock"])
def test_batch_rejects_exit_zero_without_real_pass_evidence(batch, mode):
    result = run_batch(batch, SSVC_TEST_AUDIT_MODE=mode)
    assert_failed(batch, result)
    assert (batch["run"] / "runs/P1/status.json").exists()


def test_batch_refuses_existing_attempt_without_modifying_evidence(batch):
    batch["run"].mkdir(parents=True)
    sentinel = batch["run"] / "result.txt"
    sentinel.write_text("previous evidence must remain unchanged\n")
    result = run_batch(batch)
    assert result.returncode != 0
    assert sentinel.read_text() == "previous evidence must remain unchanged\n"
    assert list(batch["run"].iterdir()) == [sentinel]
    assert not calls(batch)


@pytest.mark.parametrize("job_id", [None, "not-a-job", "../escape"])
def test_batch_rejects_missing_or_invalid_slurm_before_writes(batch, job_id):
    env = dict(batch["env"])
    if job_id is None:
        env.pop("SLURM_JOB_ID")
    else:
        env["SLURM_JOB_ID"] = job_id
    result = run_batch({**batch, "env": env})
    assert result.returncode != 0
    assert not (batch["work"] / "runs").exists()
    assert not (batch["work"] / "cache").exists()
    assert not calls(batch)


def test_batch_scheduler_contract_is_explicit_and_scoped():
    lines = SCRIPT.read_text().splitlines()
    directives = "\n".join(line for line in lines if line.startswith("#SBATCH"))
    for required in (
        "--partition=cluster02",
        "--account=rose",
        "--qos=soujanya-poria-startfund-2026-03",
        "--gres=gpu:pro6000:1",
        "--time=06:00:00",
        "--no-requeue",
    ):
        assert required in directives
    assert "--mem" not in directives
    assert "--cpus" not in directives
