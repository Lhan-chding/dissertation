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
                "max_allocated_gpu_hours": None,
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


def test_time_estimate_preserved_without_blocking_another_allocation(root):
    reserve_allocation(root, "attempt", "FORMAT_BASE_TEST", seconds=8 * 3600)
    reserve_allocation(root, "another", "FORMAT_BASE_TEST", seconds=100 * 3600)
    assert BudgetLedger(root).totals()["allocated_gpu_hours"] == 108


def test_require_real_registered_allocation_before_model_load(root, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(PermissionError, match="Slurm"):
        require_allocation(root, "FORMAT_BASE_TEST")


def test_live_scheduler_identity_checked_but_time_is_not_a_gate(root, monkeypatch):
    reserve_allocation(root, "format-0", "FORMAT_BASE_TEST", seconds=900)
    bind_allocation(root, "format-0", "1234")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    good = (
        "JobId=1234 JobName=mmcore-format-0 JobState=RUNNING "
        "NumNodes=1 AllocTRES=cpu=4,mem=32G,node=1,gres/gpu=1,gres/gpu:pro6000=1 "
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
    assert require_allocation(root, "FORMAT_BASE_TEST")["job_id"] == "1234"


def test_duplicate_allocation_binding_rejected(root):
    reserve_allocation(root, "format-0", "FORMAT_BASE_TEST", seconds=900)
    bind_allocation(root, "format-0", "1234")
    with pytest.raises(PermissionError):
        bind_allocation(root, "format-0", "1235")


@pytest.mark.parametrize(
    "changed",
    [
        "JobId=1234 JobName=mmcore-format-0 JobState=RUNNING NumNodes=1 "
        "AllocTRES=node=1,gres/gpu=8,gres/gpu:pro6000=8 TresPerNode=gres/gpu:pro6000:8",
        "JobId=1234 JobName=mmcore-format-0 JobState=RUNNING NumNodes=2 "
        "AllocTRES=node=2,gres/gpu=2,gres/gpu:pro6000=2 TresPerNode=gres/gpu:pro6000:1",
        "JobId=1234 JobName=mmcore-format-0 JobState=RUNNING NumNodes=1 "
        "TresPerNode=gres/gpu:pro6000:1",
        "JobId=1234 JobName=mmcore-format-0 JobState=PENDING NumNodes=1 "
        "TresPerNode=gres/gpu:pro6000:1",
    ],
)
def test_live_resource_mismatch_or_pending_job_cannot_load_cuda(root, monkeypatch, changed):
    reserve_allocation(root, "format-0", "FORMAT_BASE_TEST", seconds=900)
    bind_allocation(root, "format-0", "1234")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setattr(
        "mm_core.allocations.subprocess.run", lambda *a, **k: SimpleNamespace(stdout=changed)
    )
    with pytest.raises(PermissionError):
        require_allocation(root, "FORMAT_BASE_TEST")


def test_all_five_registered_pro6000_devices_require_matching_live_counts(root, monkeypatch):
    reserve_allocation(root, "format-0", "FORMAT_BASE_TEST", seconds=900, gpus=5)
    bind_allocation(root, "format-0", "1234")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    output = (
        "JobId=1234 JobName=mmcore-format-0 JobState=RUNNING NumNodes=1 "
        "AllocTRES=cpu=4,node=1,gres/gpu=5,gres/gpu:pro6000=5 "
        "TresPerNode=gres/gpu:pro6000:5 TimeLimit=UNLIMITED"
    )
    monkeypatch.setattr(
        "mm_core.allocations.subprocess.run", lambda *a, **k: SimpleNamespace(stdout=output)
    )
    assert require_allocation(root, "FORMAT_BASE_TEST")["gpus"] == 5


def test_missing_gpu_hour_estimate_is_recorded_without_blocking_verified_allocation(
    root, monkeypatch
):
    reserve_allocation(root, "format-0", "FORMAT_BASE_TEST", seconds=900)
    bind_allocation(root, "format-0", "1234")
    (root / "accounting/COST_LEDGER.jsonl").write_text("")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    output = (
        "JobId=1234 JobName=mmcore-format-0 JobState=RUNNING NumNodes=1 "
        "AllocTRES=node=1,gres/gpu=1,gres/gpu:pro6000=1 "
        "TresPerNode=gres/gpu:pro6000:1 TimeLimit=UNLIMITED"
    )
    monkeypatch.setattr(
        "mm_core.allocations.subprocess.run", lambda *a, **k: SimpleNamespace(stdout=output)
    )
    assert require_allocation(root, "FORMAT_BASE_TEST")["job_id"] == "1234"
    check_path = next((root / "accounting/allocation_checks").glob("*.json"))
    check = json.loads(check_path.read_text())
    assert check["time_accounting_status"] == "MISSING_OR_MISMATCHED_ESTIMATE"
    assert check["gpu_hours_are_execution_gate"] is False
