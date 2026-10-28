"""Mock scheduler tests only; never submit a job."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6_followup import source
from sr_f6_followup import t03_dispatch as dispatch

_spec = importlib.util.spec_from_file_location(
    "followup_t03_fixture", Path(__file__).with_name("test_followup_t03.py")
)
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)
setup = fixture.setup


@pytest.fixture
def ready(setup):
    for path in (setup.root / "T03/SUBMISSIONS").iterdir():
        path.unlink()
    setup.monkeypatch.setattr(dispatch, "ROOT", setup.root)
    setup.monkeypatch.setattr(dispatch.getpass, "getuser", lambda: dispatch.USER)
    calls = []
    setup.monkeypatch.setattr(dispatch, "capacity", lambda: dict(can_submit_two_gpu_job=True))

    def submit(argv, **kwargs):
        calls.append(argv)
        assert (setup.root / "T03/SUBMISSIONS/t03.attempt.json").exists()
        return SimpleNamespace(returncode=0, stdout="321\n", stderr="")

    setup.monkeypatch.setattr(dispatch.subprocess, "run", submit)
    setup.submit_calls = calls
    return setup


def test_singleton_two_gpu_submit_then_idempotent_no_second_sbatch(ready):
    result = dispatch.submit(ready.root)
    assert result["job_id"] == "321"
    assert len(ready.submit_calls) == 1
    assert dispatch.submit(ready.root) == dict(status="ALREADY_SUBMITTED", job_id="321")
    assert len(ready.submit_calls) == 1
    args = ready.submit_calls[0]
    assert "--job-name=LH-SRF6-T03" in args and "--no-requeue" in args
    assert args[-1].endswith("scripts/sr_f6_followup/t03.sbatch")
    assert "SRF6_PRODUCER=" in next(x for x in args if x.startswith("--export="))


def test_capacity_full_is_not_submission_or_permission(ready):
    ready.monkeypatch.setattr(dispatch, "capacity", lambda: dict(can_submit_two_gpu_job=False))
    assert dispatch.submit(ready.root)["status"] == "NO_LEGAL_CAPACITY"
    assert ready.submit_calls == []
    assert not (ready.root / "T03/SUBMISSIONS/t03.attempt.json").exists()


def test_failed_and_unknown_submission_intents_cannot_retry(ready):
    def failure(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="synthetic")

    ready.monkeypatch.setattr(dispatch.subprocess, "run", failure)
    with pytest.raises(PermissionError):
        dispatch.submit(ready.root)
    assert (ready.root / "T03/SUBMISSIONS/t03.attempt.json").exists()
    with pytest.raises(PermissionError):
        dispatch.submit(ready.root)


def test_interrupted_submission_without_receipt_cannot_retry(ready):
    def interrupt(*args, **kwargs):
        raise RuntimeError("synthetic network interruption")

    ready.monkeypatch.setattr(dispatch.subprocess, "run", interrupt)
    with pytest.raises(RuntimeError):
        dispatch.submit(ready.root)
    with pytest.raises(PermissionError):
        dispatch.submit(ready.root)


def test_review_rejected_before_capacity_or_sbatch(ready):
    def deny(*args, **kwargs):
        raise PermissionError("review incomplete")

    ready.monkeypatch.setattr(source, "require_support", deny)
    ready.monkeypatch.setattr(
        dispatch, "capacity", lambda: pytest.fail("capacity before permission")
    )
    with pytest.raises(PermissionError):
        dispatch.submit(ready.root)
    assert ready.submit_calls == []


@pytest.mark.parametrize(
    "location", ["FAILURES/orphan.txt", "RUNS/orphan/.hidden", "SLOTS/b0.json"]
)
def test_any_prior_execution_member_blocks_submission(ready, location):
    fixture.write(ready.root / "T03" / location, {})
    with pytest.raises(PermissionError):
        dispatch.submit(ready.root)
    assert ready.submit_calls == []


def test_receipt_cannot_rebind_producer_or_script(ready):
    dispatch.submit(ready.root)
    path = ready.root / "T03/SUBMISSIONS/t03.attempt.json"
    value = json.loads(path.read_text())
    value["argv"][-1] = "unregistered.sbatch"
    fixture.write(path, value)
    with pytest.raises(PermissionError):
        dispatch.submit(ready.root)
    assert len(ready.submit_calls) == 1


def test_launcher_fixed_two_lanes_and_early_boundaries(tmp_path):
    path = Path(__file__).parents[2] / "scripts/sr_f6_followup/t03.sbatch"
    text = path.read_text()
    assert (
        text.count("run_parent b0")
        == text.count("run_parent b1")
        == text.count("run_parent b2")
        == 1
    )
    assert "--gres=gpu:pro6000:1" in text and "#SBATCH --gres=gpu:pro6000:2" in text
    assert 'PYTHONPATH="$SRF6_PRODUCER/src:$SRF6_CODE/src"' in text
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0
    env = dict(
        SRF6_CODE="bad",
        SRF6_PRODUCER="bad",
        SRF6_ROOT="bad",
        SLURM_JOB_ID="1",
        SLURM_JOB_NAME="wrong",
    )
    result = subprocess.run(["bash", str(path)], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "wrong job prefix" in result.stderr
