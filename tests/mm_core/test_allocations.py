import json
from types import SimpleNamespace

import pytest

from mm_core.allocations import bind_allocation, require_allocation, reserve_allocation
from mm_core.execution import BudgetLedger


@pytest.fixture
def root(tmp_path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/RESOURCE_OVERRIDE.json").write_text(
        json.dumps(
            {
                "max_concurrent_gpus": 5,
                "max_allocated_gpu_hours": 8,
            }
        )
    )
    (tmp_path / "manifests/PRE_INFERENCE_FREEZE.json").write_text("{}")
    return tmp_path


def test_concurrency_ceiling_shared_across_stage_submissions(root):
    for i in range(5):
        reserve_allocation(root, f"worker-{i}", "FORMAT_BASE_TEST", seconds=60)
    with pytest.raises(PermissionError, match="Five-GPU"):
        reserve_allocation(root, "sixth", "FORMAT_BASE_TEST", seconds=60)
    assert BudgetLedger(root).totals()["allocated_gpu_hours"] == pytest.approx(5 / 60)


def test_failed_or_unknown_submit_reservation_is_not_released(root):
    reserve_allocation(root, "attempt", "FORMAT_BASE_TEST", seconds=8 * 3600)
    with pytest.raises(RuntimeError, match="BUDGET_EXHAUSTED"):
        reserve_allocation(root, "retry", "FORMAT_BASE_TEST", seconds=60)


def test_require_real_registered_allocation_before_model_load(root, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(PermissionError, match="Slurm"):
        require_allocation(root, "FORMAT_BASE_TEST")


def test_live_scheduler_identity_and_time_checked(root, monkeypatch):
    reserve_allocation(root, "format-0", "FORMAT_BASE_TEST", seconds=900)
    bind_allocation(root, "format-0", "1234")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    good = (
        "JobId=1234 JobName=mmcore-format-0 JobState=RUNNING "
        "TresPerNode=gres/gpu:pro6000:1 TimeLimit=00:15:00"
    )
    monkeypatch.setattr(
        "mm_core.allocations.subprocess.run", lambda *a, **k: SimpleNamespace(stdout=good)
    )
    assert require_allocation(root, "FORMAT_BASE_TEST")["job_id"] == "1234"
    with pytest.raises(PermissionError, match="stage"):
        require_allocation(root, "BRIDGE")
    monkeypatch.setattr(
        "mm_core.allocations.subprocess.run",
        lambda *a, **k: SimpleNamespace(stdout=good.replace("00:15:00", "00:30:00")),
    )
    with pytest.raises(PermissionError, match="time limit"):
        require_allocation(root, "FORMAT_BASE_TEST")


def test_duplicate_allocation_binding_rejected(root):
    reserve_allocation(root, "format-0", "FORMAT_BASE_TEST", seconds=900)
    bind_allocation(root, "format-0", "1234")
    with pytest.raises(PermissionError):
        bind_allocation(root, "format-0", "1235")
