"""Execute the frozen batch wrapper with a local fake model, never a GPU job."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ntu_p3.sbatch"


@pytest.fixture
def batch(tmp_path):
    work = tmp_path / "work"
    repo = work / "dissertation-ssvc/ssvc_flow"
    repo.mkdir(parents=True)
    (repo / "src").mkdir()
    config = work / "config.json"
    config.write_text("{}")
    p1 = work / "p1"
    p1.mkdir()
    data = work / "data"
    data.mkdir()
    for name in ["human.json", "contact.json", "legacy.jsonl", "legacy-manifest.json"]:
        (work / name).write_text("{}")
    fake = work / "python"
    fake.write_text(
        "#!/bin/sh\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(work / "fake.py"))
        + ' "$@"\n'
    )
    fake.chmod(0o700)
    (work / "fake.py").write_text("""import json, os, subprocess, sys
from pathlib import Path
args=sys.argv[1:]
if args[0]=='-':
    program=sys.stdin.read()
    if 'import torch' in program:
        sys.exit(int(os.environ.get('FAKE_GPU_EXIT','0')))
    sys.exit(subprocess.run([sys.executable,*args],input=program,text=True).returncode)
Path(os.environ['CALLS']).write_text(json.dumps({'args':args,'cwd':os.getcwd(),'tmp':os.environ['TMPDIR']}))
sys.exit(int(os.environ.get('FAKE_EXIT','0')))
""")
    env = {
        **os.environ,
        "SLURM_JOB_ID": "345",
        "SSVC_WORK": str(work),
        "SSVC_PYTHON": str(fake),
        "SSVC_CONFIG": str(config),
        "SSVC_MODEL": "qwen25vl_3b",
        "SSVC_TRACK": "N",
        "SSVC_P1_RUN": str(p1),
        "SSVC_FROZEN_OUT": str(work / "runs/frozen-3b-N"),
        "SSVC_DATA_ROOT": str(data),
        "SSVC_HUMAN_REVIEW": str(work / "human.json"),
        "SSVC_CONTACT_MANIFEST": str(work / "contact.json"),
        "CALLS": str(work / "calls.json"),
    }
    return work, env


def run(env):
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)


def test_batch_has_only_one_named_gpu_and_no_training_or_installs():
    text = SCRIPT.read_text()
    assert "#SBATCH --gres=gpu:pro6000:1" in text
    assert "#SBATCH --qos=soujanya-poria-startfund-2026-03" in text
    assert "#SBATCH --mem" not in text and "#SBATCH --cpus-per-task" not in text
    assert "pip install" not in text and "src.grpo_update" not in text
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_reject_outside_slurm_without_writes(batch):
    work, env = batch
    env.pop("SLURM_JOB_ID")
    assert run(env).returncode == 2
    assert not (work / "runs").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("SSVC_FROZEN_OUT", "/tmp/outside"),
        ("SSVC_CONFIG", "/tmp/outside.json"),
        ("SSVC_MODEL", "unknown"),
        ("SSVC_TRACK", "Z"),
        ("SSVC_RESUME", "true"),
    ],
)
def test_reject_invalid_before_output(batch, field, value):
    work, env = batch
    env[field] = value
    assert run(env).returncode != 0
    assert not (work / "runs").exists()
    assert not (work / "calls.json").exists()


def test_gpu_failure_is_propagated_before_output(batch):
    work, env = batch
    env["FAKE_GPU_EXIT"] = "17"
    assert run(env).returncode == 17
    assert not (work / "runs").exists()


def test_n_argument_passing_and_exit_code(batch):
    work, env = batch
    env["FAKE_EXIT"] = "37"
    assert run(env).returncode == 37
    args = json.loads((work / "calls.json").read_text())["args"]
    assert args[:4] == ["-m", "src.rollout", "--phase", "frozen"]
    for flag, key in [
        ("--human-review", "SSVC_HUMAN_REVIEW"),
        ("--contact-manifest", "SSVC_CONTACT_MANIFEST"),
        ("--p1-run-dir", "SSVC_P1_RUN"),
        ("--data-root", "SSVC_DATA_ROOT"),
    ]:
        assert args[args.index(flag) + 1] == env[key]
    assert "--resume" not in args and "--legacy-data" not in args


def test_existing_output_requires_explicit_resume(batch):
    work, env = batch
    out = Path(env["SSVC_FROZEN_OUT"])
    out.mkdir(parents=True)
    (out / "evidence").write_text("keep")
    assert run(env).returncode != 0
    assert (out / "evidence").read_text() == "keep"
    env["SSVC_RESUME"] = "1"
    assert run(env).returncode == 0
    assert "--resume" in json.loads((work / "calls.json").read_text())["args"]
    assert (out / "evidence").read_text() == "keep"


def test_l_does_not_require_n_inputs(batch):
    work, env = batch
    env["SSVC_TRACK"] = "L"
    for k in ["SSVC_DATA_ROOT", "SSVC_HUMAN_REVIEW", "SSVC_CONTACT_MANIFEST"]:
        env.pop(k)
    env["SSVC_LEGACY_DATA"] = str(work / "legacy.jsonl")
    env["SSVC_LEGACY_MANIFEST"] = str(work / "legacy-manifest.json")
    assert run(env).returncode == 0
    args = json.loads((work / "calls.json").read_text())["args"]
    assert args[args.index("--legacy-data") + 1] == env["SSVC_LEGACY_DATA"]
    assert "--human-review" not in args


def test_cache_symlink_escape_is_rejected_before_write(batch, tmp_path):
    work, env = batch
    outside = tmp_path / "outside"
    outside.mkdir()
    (work / "cache").symlink_to(outside, target_is_directory=True)
    assert run(env).returncode != 0
    assert list(outside.iterdir()) == []
    assert not (work / "runs").exists()
