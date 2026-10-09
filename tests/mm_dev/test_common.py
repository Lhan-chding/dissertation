import json
from pathlib import Path

import pytest

from mm_dev.common import CostLedger, load_plan, verify_live_allocation


def test_dispatched_plan():
    p = Path(__file__).resolve().parents[2] / "docs/mm_dev_f2/design/config/MM_DEV_F2.json"
    plan = load_plan(p)
    assert plan["expected_counts"]["scientific_completions"] == 669696
    assert plan["resource"]["max_allocated_gpu_hours"] is None


def test_plan_mutation_rejected(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text('{"plan_id":"MM-DEV-F2-QWEN35-9B-20261009"}')
    with pytest.raises(PermissionError, match="identity"):
        load_plan(p)


def test_hours_accounting_is_not_a_gate(tmp_path):
    ledger = CostLedger(tmp_path, "prep_A_0")
    ledger.reserve("allocated_gpu_hours", 100000, {"actual": True})
    assert json.loads(ledger.path.read_text())["count"] == 100000


@pytest.mark.parametrize("bad", [-1, True, float("nan"), float("inf")])
def test_invalid_accounting_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        CostLedger(tmp_path, "ENGINE_F2").reserve("completion_attempts", bad)


def _live_allocation():
    permit = {"owner": "researcher", "account": "a", "qos": "q", "partition": "p"}
    allocation = {
        "owner": "researcher",
        "account": "a",
        "qos": "q",
        "job_name": "mmdev-registered-attempt",
        "comment": "mmdev:registered:attempt",
    }
    fields = dict(
        JobId="123",
        JobState="RUNNING",
        UserId="researcher(1000)",
        Account="a",
        QOS="q",
        Partition="p",
        NumNodes="1",
        Requeue="0",
        Restarts="0",
        AllocTRES="cpu=4,gres/gpu=1,gres/gpu:pro6000=1",
        TresPerNode="gres/gpu:pro6000:1",
        JobName=allocation["job_name"],
        Comment=allocation["comment"],
    )
    return fields, allocation, permit


def test_live_allocation_checks_actual_grant():
    fields, allocation, permit = _live_allocation()
    assert verify_live_allocation(fields, allocation, permit, user="researcher", job_id="123")


@pytest.mark.parametrize(
    "changed",
    [
        {"AllocTRES": "gres/gpu=2,gres/gpu:pro6000=2"},
        {"AllocTRES": "gres/gpu=1"},
        {"AllocTRES": "gres/gpu=1,gres/gpu:6000ada=1"},
        {"AllocTRES": "gres/gpu=1,gres/gpu:pro6000=1,gres/gpu:6000ada=1"},
        {"UserId": "other(1001)"},
        {"JobState": "PENDING"},
        {"TresPerNode": "gres/gpu:6000ada:1"},
        {"TresPerNode": "gres/gpu:pro6000:10"},
        {"NumNodes": "2"},
        {"NumNodes": ""},
        {"Requeue": "1"},
        {"Restarts": "1"},
        {"Requeue": ""},
        {"Restarts": ""},
        {"JobName": "unregistered"},
        {"Comment": ""},
        {"Account": "other"},
        {"QOS": "other"},
        {"Partition": "other"},
    ],
)
def test_live_allocation_rejects_changed_or_unverified_grant(changed):
    fields, allocation, permit = _live_allocation()
    with pytest.raises(PermissionError):
        verify_live_allocation(
            {**fields, **changed}, allocation, permit, user="researcher", job_id="123"
        )


def test_live_allocation_permitted_partition_list():
    fields, allocation, permit = _live_allocation()
    assert verify_live_allocation(
        fields, allocation, {**permit, "partition": "p,other"}, user="researcher", job_id="123"
    )


@pytest.fixture
def registered_allocation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mm_core.execution import sha256_file
    from mm_dev.contract import PLAN_ID
    from mm_dev.orchestration import digest

    def put(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
        return path

    permission = {
        "plan_id": PLAN_ID,
        "run_root": str(tmp_path),
        "authorized": True,
        "max_gpus_concurrent": 5,
        "gpus_per_worker": 1,
        "gres": "gpu:pro6000:1",
        "owner": "researcher",
        "account": "a",
        "qos": "q",
        "partition": "p",
    }
    permission_path = put("manifests/ALLOCATION_PERMISSION.json", permission)
    freeze_path = put("manifests/F2_FREEZE.json", {"plan_id": PLAN_ID, "status": "FROZEN"})
    task = "prep_A_0"
    registration = {
        "plan_id": PLAN_ID,
        "permission": permission,
        "permission_receipt_sha256": sha256_file(permission_path),
        "freeze_sha256": sha256_file(freeze_path),
        "tasks": {task: {"gpus": 1}},
    }
    put("orchestration/REGISTRATION.json", registration)
    registered_hash = digest(registration)
    attempt_id = task + "_attempt0000"
    attempt_path = f"orchestration/attempts/{attempt_id}.json"
    attempt = {
        "plan_id": PLAN_ID,
        "task_id": task,
        "attempt_id": attempt_id,
        "registration_hash": registered_hash,
        "manifest_path": attempt_path,
        "job_name": f"mmdev-{registered_hash[:12]}-{attempt_id}",
        "comment": f"mmdev:{registered_hash[:12]}:{attempt_id}",
        "status": "SUBMITTING",
        "job_id": None,
    }
    put(attempt_path, attempt)
    allocation = {
        "task_id": task,
        "attempt_id": attempt_id,
        "attempt_manifest_path": attempt_path,
        "job_id": "123",
        "status": "ACTIVE",
        "gpus": 1,
        "gres": permission["gres"],
        "owner": permission["owner"],
        "account": "a",
        "qos": "q",
        "job_name": attempt["job_name"],
        "comment": attempt["comment"],
    }
    registry = {
        "plan_id": PLAN_ID,
        "registration_hash": registered_hash,
        "permission_receipt_sha256": sha256_file(permission_path),
        "allocations": [allocation],
    }
    put("manifests/ALLOCATIONS.json", registry)
    fields = {
        **_live_allocation()[0],
        "JobName": attempt["job_name"],
        "Comment": attempt["comment"],
    }
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command == ["id", "-un"]:
            return SimpleNamespace(stdout="researcher\n")
        assert command == ["scontrol", "show", "job", "123", "-o"]
        return SimpleNamespace(stdout=" ".join(f"{k}={v}" for k, v in fields.items()))

    monkeypatch.setattr("mm_dev.common.subprocess.run", run)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("MM_DEV_TASK_ID", task)
    monkeypatch.setenv("MM_DEV_ATTEMPT_ID", attempt_id)
    monkeypatch.setenv("MM_DEV_REGISTRATION_HASH", registered_hash)
    return dict(
        root=tmp_path,
        task=task,
        put=put,
        registration=registration,
        registry=registry,
        allocation=allocation,
        attempt=attempt,
        attempt_path=attempt_path,
        fields=fields,
        calls=calls,
    )


def test_registered_allocation_accepts_pre_submission_null_job_id(registered_allocation):
    from mm_dev.common import require_allocation

    f = registered_allocation
    result = require_allocation(f["root"], f["task"])
    assert f["attempt"]["job_id"] is None
    assert result["allocation"]["job_id"] == "123"
    assert len(result["attempt_manifest_sha256"]) == 64
    assert len(result["registration_sha256"]) == 64
    assert len(f["calls"]) == 2


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("allocation", "status", "COMPLETED"),
        ("allocation", "gpus", True),
        ("allocation", "task_id", "prep_AP_0"),
        ("allocation", "job_name", "other-job"),
        ("allocation", "comment", "other-comment"),
        ("allocation", "attempt_manifest_path", "../outside.json"),
        ("allocation", "attempt_id", "prep_A_0_attempt0001"),
        ("attempt", "job_id", "124"),
        ("attempt", "job_name", "different"),
        ("attempt", "comment", "different"),
        ("attempt", "registration_hash", "different"),
        ("registration", "freeze_sha256", "different"),
        ("registry", "registration_hash", "different"),
    ],
)
def test_registered_allocation_rejects_unbound_identity(registered_allocation, target, key, value):
    from mm_dev.common import require_allocation

    f = registered_allocation
    f[target][key] = value
    f["put"]("manifests/ALLOCATIONS.json", f["registry"])
    f["put"]("orchestration/REGISTRATION.json", f["registration"])
    f["put"](f["attempt_path"], f["attempt"])
    with pytest.raises(PermissionError):
        require_allocation(f["root"], f["task"])
    assert f["calls"] == []


@pytest.mark.parametrize("kind", ["same_job", "same_task", "same_attempt", "unknown_task"])
def test_registered_allocation_rejects_duplicate_claims(registered_allocation, kind):
    from mm_dev.common import require_allocation

    f = registered_allocation
    duplicate = dict(f["allocation"])
    if kind != "same_job":
        duplicate["job_id"] = "124"
    if kind in {"same_task", "unknown_task"}:
        duplicate["attempt_id"] = "prep_A_0_attempt0001"
        if kind == "unknown_task":
            duplicate["status"] = "UNKNOWN"
    elif kind == "same_attempt":
        duplicate["task_id"] = "prep_AP_0"
    f["registry"]["allocations"].append(duplicate)
    f["put"]("manifests/ALLOCATIONS.json", f["registry"])
    with pytest.raises(PermissionError):
        require_allocation(f["root"], f["task"])
    assert f["calls"] == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("MM_DEV_TASK_ID", "prep_AP_0"),
        ("MM_DEV_ATTEMPT_ID", "prep_A_0_attempt0001"),
        ("MM_DEV_REGISTRATION_HASH", "forged"),
        ("SLURM_JOB_ID", ""),
    ],
)
def test_registered_allocation_rejects_worker_environment(
    registered_allocation, monkeypatch, key, value
):
    from mm_dev.common import require_allocation

    f = registered_allocation
    monkeypatch.setenv(key, value)
    with pytest.raises(PermissionError):
        require_allocation(f["root"], f["task"])
    assert f["calls"] == []
