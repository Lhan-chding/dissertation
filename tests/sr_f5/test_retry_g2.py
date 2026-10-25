import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).parents[2] / "scripts/sr_f5/retry_g2.py"
SPEC = importlib.util.spec_from_file_location("srf5_retry_test_module", PATH)
retry_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry_module)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    mod = retry_module
    root, science, launcher = (tmp_path / name for name in ("run", "science", "launcher"))
    old_launcher = tmp_path / "old_launcher"
    root.mkdir()
    for directory, relative in (
        (science, "src/sr_f5/calibration.py"),
        (launcher, "src/sr_f5/dual_gpu.py"),
        (launcher, "scripts/sr_f5/calibration.sbatch"),
    ):
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# deployed source\n")
    for name, expected in (
        ("CPU_READY.json", "CPU_READY_FOR_G2_PREFLIGHT"),
        ("CPU_TESTS.json", "PASS"),
        ("DATA/HISTORICAL_WORLD_AUDIT.json", "PASS"),
        ("RETENTION/FROZEN.json", "FROZEN_BEFORE_NEW_MODEL_OUTPUTS"),
    ):
        mod.write(root / name, dict(status=expected))
    monkeypatch.setattr(mod, "ROOT", root)
    monkeypatch.setattr(mod, "CODE", science)
    monkeypatch.setattr(mod, "PROJECT", tmp_path)
    original = dict(
        status="TWO_DUAL_GPU_JOBS_FOUR_WORKERS_SUBMITTED_AND_VERIFIED",
        source_code=str(science),
        launcher_code=str(old_launcher),
        jobs=[],
    )
    for pair in range(2):
        job_id = str(100 + pair)
        actual = dict(
            JobId=job_id,
            JobName=f"LH-SRF5-G2-P{pair}",
            Account=mod.ACCOUNT,
            QOS=mod.QOS,
            WorkDir=str(science),
            Command=str(old_launcher / "scripts/sr_f5/calibration.sbatch"),
            StdOut=str(root / "slurm" / f"G2-P{pair}-{job_id}.log"),
            ReqTRES="cpu=8,gres/gpu=2",
        )
        original["jobs"].append(
            dict(
                pair=pair,
                name=actual["JobName"],
                job_id=job_id,
                shards=[2 * pair, 2 * pair + 1],
                status="SUBMITTED",
                verified=actual,
            )
        )
        for shard in (2 * pair, 2 * pair + 1):
            log = root / "slurm" / f"G2-W{shard}-{job_id}.log"
            log.parent.mkdir(exist_ok=True)
            log.write_text("PermissionError: " + mod.EXPECTED_FAILURE + "\n")
    mod.write(root / "G2_SUBMISSION_DUAL_GPU.json", original)

    class Slurm:
        def __init__(self):
            self.active = set()
            self.terminal_state = "FAILED"
            self.queue_rows = {}

        def user_job_ids(self, _user):
            return self.active

        def queue(self):
            return dict(self.queue_rows)

        def terminal(self, job_id):
            return dict(
                job_id=job_id,
                state=self.terminal_state,
                exit_code="1:0",
                user="me",
                account=mod.ACCOUNT,
                qos=mod.QOS,
            )

        def show(self, job_id):
            return dict(
                JobName=f"LH-SRF5-G2-P{int(job_id) - 200}",
                QOS=mod.QOS,
                Account=mod.ACCOUNT,
                ReqTRES="cpu=8,gres/gpu=2",
                WorkDir=str(science),
            )

    calls = []

    def submit(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=str(199 + len(calls)) + "\n", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", submit)
    args = SimpleNamespace(
        launcher_code=launcher,
        science_code=science,
        attempt_id="env-fix-1",
        reason="Restore audited CUBLAS environment",
    )
    return SimpleNamespace(
        mod=mod, root=root, args=args, slurm=Slurm(), calls=calls, original=original
    )


def test_retry_preserves_original_and_binds_new_jobs(fixture):
    f = fixture
    original_bytes = (f.root / "G2_SUBMISSION_DUAL_GPU.json").read_bytes()
    report = f.mod.retry(f.args, slurm=f.slurm, user="me")
    assert report["status"] == "TWO_DUAL_GPU_JOBS_FOUR_WORKERS_SUBMITTED_AND_VERIFIED"
    assert [row["job_id"] for row in report["jobs"]] == ["200", "201"]
    assert len(report["failure_evidence"]) == 4
    assert report["model_records_before_retry"] == "NONE"
    assert report["science_code_changed"] is False
    assert (f.root / "G2_SUBMISSION_DUAL_GPU.json").read_bytes() == original_bytes
    assert [call[1]["env"]["SRF5_PAIR"] for call in f.calls] == ["0", "1"]
    assert all(call[1]["cwd"] == f.args.science_code for call in f.calls)


@pytest.mark.parametrize(
    "relative", ["G2/raw/x/a.jsonl", "G2/attempts/x/a.json", "G2/groups/x/a.json"]
)
def test_any_model_record_forbids_startup_retry(fixture, relative):
    f = fixture
    path = f.root / relative
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(PermissionError, match="Model records"):
        f.mod.retry(f.args, slurm=f.slurm, user="me")
    assert not f.calls


@pytest.mark.parametrize("state", ["RUNNING", "COMPLETED", "TIMEOUT", None])
def test_only_diagnosed_failed_terminal_is_eligible(fixture, state):
    f = fixture
    f.slurm.terminal_state = state
    with pytest.raises(RuntimeError, match="FAILED terminal"):
        f.mod.retry(f.args, slurm=f.slurm, user="me")
    assert not f.calls


def test_original_queued_job_blocks_even_if_accounting_failed(fixture):
    f = fixture
    f.slurm.active = {"100"}
    with pytest.raises(RuntimeError, match="queued or running"):
        f.mod.retry(f.args, slurm=f.slurm, user="me")


def test_other_project_receipt_rejected(fixture):
    f = fixture
    f.original["jobs"][0]["verified"]["WorkDir"] = "/some/other/project"
    f.mod.write(f.root / "G2_SUBMISSION_DUAL_GPU.json", f.original)
    with pytest.raises(PermissionError, match="bound to this root"):
        f.mod.retry(f.args, slurm=f.slurm, user="me")


def test_unknown_submission_cannot_be_retried_under_a_new_id(fixture, monkeypatch):
    f = fixture

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("sbatch", 60)

    monkeypatch.setattr(f.mod.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        f.mod.retry(f.args, slurm=f.slurm, user="me")
    receipt = json.loads((f.root / "G2_RETRY_env-fix-1.json").read_text())
    assert receipt["jobs"][0]["status"] == "SUBMISSION_UNKNOWN"
    f.args.attempt_id = "another-id"
    with pytest.raises(RuntimeError, match="no duplicate"):
        f.mod.retry(f.args, slurm=f.slurm, user="me")


def test_capacity_blocks_without_altering_existing_jobs(fixture):
    f = fixture
    f.slurm.queue_rows = {"other": dict(ReqTRES="cpu=8,gres/gpu=4")}
    with pytest.raises(RuntimeError, match="Capacity unavailable"):
        f.mod.retry(f.args, slurm=f.slurm, user="me")
    assert not f.calls
    receipt = json.loads((f.root / "G2_RETRY_env-fix-1.json").read_text())
    assert receipt["status"] == "CAPACITY_BLOCKED"


def test_different_failure_requires_diagnosis(fixture):
    f = fixture
    (f.root / "slurm/G2-W0-100.log").write_text("CUDA out of memory")
    with pytest.raises(PermissionError, match="diagnosed pre-generation"):
        f.mod.retry(f.args, slurm=f.slurm, user="me")
