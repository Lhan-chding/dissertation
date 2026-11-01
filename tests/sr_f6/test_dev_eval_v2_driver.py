import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

P = Path(__file__).resolve().parents[2] / "scripts/sr_f6_dev_eval_v2/driver.py"


@pytest.fixture
def d(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("dev_driver_tests", P)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(m, "NEW_OPS", tmp_path / "ops")
    monkeypatch.setattr(m, "EVALUATION", root / "PHASE_RUNS/PILOT_V3/EVALUATION")
    monkeypatch.setattr(m, "PLAN", root / "PLAN.json")
    monkeypatch.setattr(m, "SPEC", root / "SPEC.json")
    ids = sorted(m.PAIRS)
    tasks = [dict(task_id=str(i), kind="EVALUATION", endpoint_id=e) for i, e in enumerate(ids)]
    plan = dict(tasks=tasks, queues={"EVAL": [[str(i) for i in range(w, 12, 2)] for w in (0, 1)]})
    sp = dict(
        evaluation=dict(endpoints=[dict(endpoint_id=e) for e in ids]),
        budgets=dict(dev_eval_responses=2304),
    )
    m.PLAN.write_text(json.dumps(plan))
    m.SPEC.write_text(json.dumps(sp))
    monkeypatch.setattr(m, "PLAN_SHA", m.sha(m.PLAN))
    monkeypatch.setattr(m, "SPEC_SHA", m.sha(m.SPEC))
    return m


def test_plan_retains_exact_twelve_two_lanes(d):
    _, _, queues = d.fixed_plan()
    assert [len(x) for x in queues] == [6, 6]
    assert {item for queue in queues for item in queue} == d.PAIRS


@pytest.mark.parametrize("change", ["duplicate", "extra", "budget"])
def test_changed_registered_evaluation_rejected(d, monkeypatch, change):
    spec = d.read(d.SPEC)
    if change == "duplicate":
        spec["evaluation"]["endpoints"][1] = spec["evaluation"]["endpoints"][0]
    elif change == "extra":
        spec["evaluation"]["endpoints"].append(dict(endpoint_id="b0/U11/recovery"))
    else:
        spec["budgets"]["dev_eval_responses"] = 2305
    d.SPEC.write_text(json.dumps(spec))
    monkeypatch.setattr(d, "SPEC_SHA", d.sha(d.SPEC))
    with pytest.raises(PermissionError):
        d.fixed_plan()


@pytest.mark.parametrize(
    "relative",
    ["b0/parent/STARTED.json", "b0/parent/groups/q/RAW.jsonl", "b2/U11/step12/COMPLETE.json"],
)
def test_any_saved_evaluation_attempt_blocks_new_launch(d, relative):
    p = d.EVALUATION / relative
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    with pytest.raises(PermissionError, match="Prior evaluation"):
        d.pristine()


def test_old_eval_worker_also_blocks_launch(d):
    p = d.ROOT / "PHASE_RUNS/PILOT_V3/JOBS/WORKERS/EVAL/0/ATTEMPTS/old/STARTED.json"
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    with pytest.raises(PermissionError):
        d.pristine()


def test_pristine_does_not_touch_existing_training(d):
    p = d.ROOT / "PHASE_RUNS/PILOT_V3/PILOT_FROZEN/runs/b0/r0/U11/COMPLETE.json"
    p.parent.mkdir(parents=True)
    p.write_text("old")
    d.pristine()
    assert p.read_text() == "old"


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"AllocTRES": "cpu=8,mem=66G,gres/gpu:pro6000=1"},
        {"Requeue": "1"},
        {"QOS": "override-limits-but-killable"},
    ],
)
def test_actual_allocation(d, monkeypatch, change):
    fields = dict(
        JobId="123",
        JobName=d.JOB_NAME,
        JobState="RUNNING",
        QOS="soujanya-poria-startfund-2026-03",
        Partition="cluster02",
        Account="rose",
        Requeue="0",
        TimeLimit="23:00:00",
        UserId="varun024(123)",
        ReqTRES="cpu=8,mem=66G,gres/gpu:pro6000=2",
        AllocTRES="cpu=8,mem=66G,gres/gpu:pro6000=2",
    )
    fields.update(change)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(
        d.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=" ".join(k + "=" + v for k, v in fields.items())),
    )
    if change:
        with pytest.raises(PermissionError):
            d.allocation()
    else:
        assert d.allocation()["JobId"] == "123"


def test_lane_new_process_per_endpoint_and_no_retry(d, monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        identifier = argv[-1]
        p = d.EVALUATION / identifier / "COMPLETE.json"
        p.parent.mkdir(parents=True)
        p.write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(d.subprocess, "run", run)
    d.lane({}, d.NEW_OPS / "CONTRACT.json", "a" * 64, 0)
    assert len(calls) == 6 and len({x[-1] for x in calls}) == 6
    assert all("endpoint" in x and x[1:4] == ["-I", "-B", "-u"] for x in calls)
    with pytest.raises(FileExistsError):
        d.lane({}, d.NEW_OPS / "CONTRACT.json", "a" * 64, 0)


def test_launch_waits_for_peer_after_lane_failure(d, monkeypatch):
    waited = []

    class Child:
        def __init__(self, i):
            self.i = i

        def wait(self):
            waited.append(self.i)
            return 1 if self.i == 0 else 0

    children = []

    def spawn(*args, **kwargs):
        argv = args[0]
        assert all(
            flag in argv
            for flag in ("--export=ALL", "--exact", "--nodes=1", "--kill-on-bad-exit=0")
        )
        index = argv.index(d.sys.executable)
        assert argv[index + 1 : index + 4] == ["-I", "-B", "-u"]
        child = Child(len(children))
        children.append(child)
        return child

    monkeypatch.setattr(d.subprocess, "Popen", spawn)
    monkeypatch.setattr(d, "allocation", lambda: {})
    monkeypatch.setattr(d, "check_preflight", lambda: {})
    monkeypatch.setattr(d, "ref", lambda p: dict(path=str(p), sha256="a" * 64))
    d.NEW_OPS.mkdir()
    with pytest.raises(PermissionError, match="lanes failed"):
        d.launch(dict(execution_source={}), d.NEW_OPS / "CONTRACT.json", "a" * 64)
    assert waited == [0, 1]
    assert (d.NEW_OPS / "RESULT/FAILURE.json").exists()
    assert not (d.NEW_OPS / "RESULT/COMPLETE.json").exists()


def test_launch_waits_existing_peer_if_second_spawn_fails(d, monkeypatch):
    waited = []
    calls = []

    class Child:
        def wait(self):
            waited.append(True)
            return 0

    def spawn(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("second spawn failed")
        return Child()

    monkeypatch.setattr(d.subprocess, "Popen", spawn)
    monkeypatch.setattr(d, "allocation", lambda: {})
    monkeypatch.setattr(d, "check_preflight", lambda: {})
    monkeypatch.setattr(d, "ref", lambda p: dict(path=str(p), sha256="a" * 64))
    d.NEW_OPS.mkdir()
    with pytest.raises(OSError, match="second spawn failed"):
        d.launch(dict(execution_source={}), d.NEW_OPS / "CONTRACT.json", "a" * 64)
    assert waited == [True]
    assert (d.NEW_OPS / "RESULT/FAILURE.json").exists()


def test_child_python_blocks_bytecode_without_environment_hint(d, tmp_path):
    source = tmp_path / "operations.py"
    source.write_text("VALUE = 42\n")
    code = """import importlib.util,os,sys
assert sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1
assert 'PYTHONDONTWRITEBYTECODE' not in os.environ
assert os.environ['OMP_NUM_THREADS'] == '1'
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
spec=importlib.util.spec_from_file_location('probe',sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.VALUE == 42
"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
    env.update(OMP_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    args = d.arguments("lane", tmp_path / "contract.json", "a" * 64)
    result = subprocess.run(
        [*args[:4], "-c", code, str(source)], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.rglob("*.pyc"))
    assert not (tmp_path / "__pycache__").exists()


def test_launcher_exports_runtime_environment_for_srun():
    text = P.with_name("launch.sbatch").read_text()
    assert "export SLURM_EXPORT_ENV=ALL" in text
    assert "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    assert "export OMP_NUM_THREADS=1" in text
    assert " -I -B -u " in text
    assert "CUDA_VISIBLE_DEVICES=''" not in text
