"""Local fake-srun/fake-Python control tests: no model, server, or GPU execution.

The production launcher has no project-root override. Tests relocate its literal
canonical prefix in a private copy, while executing the identical shell logic.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/sr_f6"
PROJECT = "/projects/_ssd/varunssd/louis-ssvc"
CACHE_KEYS = (
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_MODULES_CACHE",
    "HF_DATASETS_CACHE",
    "HF_ASSETS_CACHE",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
)
ENV_KEYS = (
    *CACHE_KEYS,
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "CUDA_VISIBLE_DEVICES",
    "HOME",
)


def executable(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o700)


@pytest.fixture
def fixture(tmp_path):
    project = tmp_path.resolve() / "canonical project"
    ops = project / "sr_f6_operations_20261027"
    code = project / "sr_f6_code_d49e9ae"
    root = project / "sr_f6_semantic_factorial_20261027"
    records, binary = project / "fake-records", project / "fake-bin"
    for directory in (ops, code / "src", root, records, binary):
        directory.mkdir(parents=True)
    (code / "src/frozen-sentinel.py").write_text("FROZEN = True\n")
    launcher = ops / "run_t03_pair.sh"
    thin = ops / "t03.sbatch"
    for target in (launcher, thin):
        source = (SCRIPTS / target.name).read_text()
        assert PROJECT in source
        target.write_text(source.replace(PROJECT, str(project)))
    executable(
        binary / "srun",
        f"#!{sys.executable}\n"
        "import json,os,pathlib,sys\n"
        "args=sys.argv[1:]; pos=args.index('/bin/bash'); lane=args[pos+3]\n"
        "record={'resources':args[:pos], 'command':args[pos:], 'pid':os.getpid()}\n"
        "(pathlib.Path(os.environ['T03_FAKE_RECORDS'])/(lane+'.json')).write_text(json.dumps(record))\n"
        "os.environ['CUDA_VISIBLE_DEVICES']='0' if lane.endswith('0') else '1'\n"
        "os.execv(args[pos],args[pos:])\n",
    )
    executable(
        project / "envs/ssvc-py312/bin/python3.12",
        f"#!{sys.executable}\n"
        "import json,os,pathlib,sys,time\n"
        "args=sys.argv[1:]; parent=args[args.index('--parent')+1]\n"
        "out=pathlib.Path(os.environ['T03_FAKE_RECORDS'])\n"
        f"keys={ENV_KEYS!r}\n"
        "record={'argv':args,'pid':os.getpid(),'cwd':os.getcwd(),'started':time.monotonic_ns(),"
        "'env':{k:os.environ.get(k) for k in keys}}\n"
        "(out/(parent+'.json')).write_text(json.dumps(record))\n"
        "time.sleep(float(os.environ.get('T03_FAKE_SLEEP','0.03')))\n"
        "failed=os.environ.get('T03_FAKE_FAIL')==parent\n"
        "(out/(parent+'.finished')).write_text(json.dumps({'failed':failed,'time':time.monotonic_ns()}))\n"
        "sys.exit(7 if failed else 0)\n",
    )
    environment = dict(
        os.environ,
        PATH=str(binary) + os.pathsep + os.environ["PATH"],
        SLURM_JOB_ID="SYNTHETIC_ONLY",
        SLURM_JOB_NAME="LH-SRF6-T03",
        SRF6_CODE=str(code),
        SRF6_ROOT=str(root),
        SRF6_PHASE_IDENTITY_SHA256="a" * 64,
        SRF6_T03_SPEC_SHA256="b" * 64,
        SRF6_T03_ATTEMPT="unique-control",
        SRF6_T03_LAUNCHER=str(launcher),
        T03_FAKE_RECORDS=str(records),
        HOME=str(project / "unchanged-home"),
    )
    for key in CACHE_KEYS:
        environment[key] = "/UNSAFE_INHERITED_CACHE_MUST_BE_REPLACED"
    return dict(
        project=project,
        ops=ops,
        root=root,
        code=code,
        launcher=launcher,
        thin=thin,
        records=records,
        env=environment,
    )


def launch(fixture, *, thin=False, edits=None):
    environment = {**fixture["env"], **(edits or {})}
    target = fixture["thin"] if thin else fixture["launcher"]
    return subprocess.run(
        ["/bin/bash", str(target)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def rows(fixture):
    return {p.stem: json.loads(p.read_text()) for p in fixture["records"].glob("b?.json")}


def test_three_parents_once_fixed_lanes_fresh_processes_resources_and_environment(fixture):
    before = (fixture["code"] / "src/frozen-sentinel.py").read_bytes()
    result = launch(fixture, thin=True)
    assert result.returncode == 0, result.stderr
    results = rows(fixture)
    assert set(results) == {"b0", "b1", "b2"}
    assert len({r["pid"] for r in results.values()}) == 3
    done0 = json.loads((fixture["records"] / "b0.finished").read_text())
    assert done0["time"] < results["b2"]["started"]
    for parent, row in results.items():
        args, env = row["argv"], row["env"]
        assert args[:4] == ["-u", "-m", "sr_f6.numerical_replay", "run"]
        assert args[args.index("--parent") + 1] == parent
        assert args[args.index("--task-id") + 1] == f"LH-SRF6-T03-unique-control-{parent}"
        assert args[args.index("--root") + 1] == str(fixture["root"])
        assert args[args.index("--phase-identity-sha256") + 1] == "a" * 64
        assert args[args.index("--spec-sha256") + 1] == "b" * 64
        assert row["cwd"] == str(fixture["code"])
        assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and env["OMP_NUM_THREADS"] == "1"
        assert env["TOKENIZERS_PARALLELISM"] == "false"
        assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert env["PYTHONPATH"] == str(fixture["code"] / "src")
        assert env["HOME"] == fixture["env"]["HOME"]
        assert env["CUDA_VISIBLE_DEVICES"] == ("1" if parent == "b1" else "0")
        parent_dir = fixture["ops"] / "T03_LAUNCHES/unique-control" / parent
        for key in CACHE_KEYS:
            assert Path(env[key]).is_relative_to(parent_dir)
        assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == str(parent_dir / "tmp")
    steps = [json.loads(p.read_text()) for p in fixture["records"].glob("LH-SRF6-T03-worker*.json")]
    assert len(steps) == 2
    assert {tuple(s["command"][4:]) for s in steps} == {("b0", "b2"), ("b1",)}
    for step in steps:
        assert step["resources"] == [
            "--exclusive",
            "--ntasks=1",
            "--gres=gpu:pro6000:1",
            "--cpus-per-task=4",
            "--mem=33G",
        ]
    assert (fixture["code"] / "src/frozen-sentinel.py").read_bytes() == before
    assert sorted(p.name for p in (fixture["code"] / "src").iterdir()) == ["frozen-sentinel.py"]


@pytest.mark.parametrize(
    "failed,expected",
    [("b0", {"b0", "b1"}), ("b1", {"b0", "b1", "b2"}), ("b2", {"b0", "b1", "b2"})],
)
def test_failure_stops_only_its_lane_without_retry_and_peer_finishes(fixture, failed, expected):
    result = launch(fixture, edits={"T03_FAKE_FAIL": failed})
    assert result.returncode == 1
    assert set(rows(fixture)) == expected
    assert all((fixture["records"] / (parent + ".finished")).exists() for parent in expected)
    original = {path.name: path.read_bytes() for path in fixture["records"].iterdir()}
    again = launch(fixture)
    assert again.returncode != 0 and "attempt exists" in again.stderr
    assert {path.name: path.read_bytes() for path in fixture["records"].iterdir()} == original


@pytest.mark.parametrize(
    "key,value",
    [
        ("SRF6_CODE", "/projects/_ssd/varunssd/louis-ssvc/sr_f6_code_new"),
        ("SRF6_ROOT", "/outside-project/run"),
        ("SRF6_T03_ATTEMPT", "../escape"),
        ("SRF6_PHASE_IDENTITY_SHA256", "partial"),
        ("SRF6_T03_SPEC_SHA256", ""),
        ("SLURM_JOB_NAME", "LH-SRF6wrong"),
    ],
)
def test_wrong_bindings_rejected_before_any_srun(fixture, key, value):
    assert launch(fixture, edits={key: value}).returncode != 0
    assert list(fixture["records"].iterdir()) == []


def test_symlinked_cache_and_launcher_escape_are_rejected(fixture, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (fixture["ops"] / "T03_LAUNCHES").symlink_to(outside, target_is_directory=True)
    assert launch(fixture).returncode != 0
    alias = fixture["ops"] / "alias.sh"
    alias.symlink_to(fixture["launcher"])
    assert launch(fixture, thin=True, edits={"SRF6_T03_LAUNCHER": str(alias)}).returncode != 0
    assert list(outside.iterdir()) == [] and list(fixture["records"].iterdir()) == []


@pytest.mark.parametrize("signum,exit_code", [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_interrupt_cleans_only_owned_steps_and_preserves_unrelated_peer(fixture, signum, exit_code):
    environment = dict(fixture["env"], T03_FAKE_SLEEP="60")
    wrapper = subprocess.Popen(
        ["/bin/bash", str(fixture["launcher"])],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    peer = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
    )
    try:
        deadline = time.monotonic() + 10
        while set(rows(fixture)) != {"b0", "b1"} and time.monotonic() < deadline:
            time.sleep(0.02)
        assert set(rows(fixture)) == {"b0", "b1"}
        child_pids = [row["pid"] for row in rows(fixture).values()]
        wrapper.send_signal(signum)
        assert wrapper.wait(timeout=10) == exit_code
        for pid in child_pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert peer.poll() is None and "b2" not in rows(fixture)
    finally:
        for process in (wrapper, peer):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)


def test_thin_wrapper_requests_only_registered_allocation_resources():
    source = (SCRIPTS / "t03.sbatch").read_text()
    for option in (
        "--job-name=LH-SRF6-T03",
        "--account=rose",
        "--partition=cluster02",
        "--qos=soujanya-poria-startfund-2026-03",
        "--gres=gpu:pro6000:2",
        "--cpus-per-task=8",
        "--mem=66G",
        "--time=23:00:00",
    ):
        assert "#SBATCH " + option in source
    assert 'exec /bin/bash "$SRF6_T03_LAUNCHER"' in source
    assert "sr_f6.numerical_replay" not in source
