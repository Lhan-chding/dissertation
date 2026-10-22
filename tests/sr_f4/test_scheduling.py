import copy

import pytest

from sr_f4 import pool, scheduling
from sr_f4.prepare import registered_tasks


def original_pool(root):
    legacy = scheduling._legacy_tasks(registered_tasks())
    pool.init_pool(root, legacy)
    pool._write(root / "TASKS.json", legacy)
    with pool._locked(root) as state:
        state["tasks"]["B0-b0"].update(
            status="COMPLETE", attempts=[dict(status="COMPLETE", returncode=0)]
        )
        state["tasks"]["A-b0-LZ"].update(
            status="RUNNING",
            owner=dict(job_id="123", slot="0"),
            attempts=[dict(status="RUNNING", owner=dict(job_id="123", slot="0"))],
        )
        state["jobs"]["job-1"] = dict(status="RUNNING", job_id="123", gpus=2)
        pool._save(root, state)
    return pool._read(root / "pool/STATE.json")


def test_migration_preserves_active_work_and_is_idempotent(tmp_path):
    before = original_pool(tmp_path)
    result = scheduling.migrate_evaluation_schedule(tmp_path)
    after = pool._read(tmp_path / "pool/STATE.json")
    assert result["status"] == "APPLIED"
    assert (result["old_task_count"], result["new_task_count"]) == (36, 54)
    for key, row in before["tasks"].items():
        if row["spec"]["kind"] in ("train", "probe", "replay"):
            assert after["tasks"][key] == row
    assert after["jobs"] == before["jobs"]
    assert pool._read(tmp_path / scheduling.STATE_BACKUP) == before
    assert after["tasks"]["E-b0-PARENT-s0"]["spec"]["dependencies"] == []
    assert after["tasks"]["E-b0-LZ-s0"]["spec"]["dependencies"] == ["A-b0-LZ"]
    confirmation = after["tasks"]["EC-b0-LZ-s0"]["spec"]
    assert confirmation["evaluation_task_id"] == "E-b0-LZ-s0"
    assert len(confirmation["dependencies"]) == 7
    assert len(after["tasks"]["REPORT"]["spec"]["dependencies"]) == 36
    assert scheduling.migrate_evaluation_schedule(tmp_path)["status"] == "ALREADY_PIPELINED"
    assert pool._read(tmp_path / "pool/STATE.json") == after


def test_individual_regression_releases_before_global_confirmation(tmp_path):
    original_pool(tmp_path)
    scheduling.migrate_evaluation_schedule(tmp_path)
    summary = pool.status(tmp_path)["tasks"]
    parents = [key for key in summary if key.startswith("E-") and "PARENT" in key]
    assert len(parents) == 6
    assert all(summary[key]["status"] == "READY" for key in parents)
    assert summary["E-b0-LZ-s0"]["status"] == "WAITING"
    with pool._locked(tmp_path) as state:
        state["tasks"]["A-b0-LZ"]["status"] = "COMPLETE"
        pool._save(tmp_path, state)
    summary = pool.status(tmp_path)["tasks"]
    assert all(summary[f"E-b0-LZ-s{shard}"]["status"] == "READY" for shard in range(2))
    assert summary["E-b0-J_UNIFORM_NEW_REPEAT-s0"]["status"] == "WAITING"
    assert all(row["status"] == "WAITING" for key, row in summary.items() if key.startswith("EC-"))
    with pool._locked(tmp_path) as state:
        for row in state["tasks"].values():
            if row["spec"]["kind"] == "train":
                row["status"] = "COMPLETE"
        pool._save(tmp_path, state)
    assert pool.status(tmp_path)["tasks"]["EC-b0-LZ-s0"]["status"] == "WAITING"
    with pool._locked(tmp_path) as state:
        state["tasks"]["B3-replay"]["status"] = "COMPLETE"
        pool._save(tmp_path, state)
    summary = pool.status(tmp_path)["tasks"]
    assert all(row["status"] == "READY" for key, row in summary.items() if key.startswith("EC-"))
    assert summary["REPORT"]["status"] == "WAITING"


@pytest.mark.parametrize("key", ["E-b0-PARENT-s0", "REPORT", "PACKAGE"])
def test_migration_refuses_started_evaluation_or_delivery(tmp_path, key):
    original_pool(tmp_path)
    with pool._locked(tmp_path) as state:
        state["tasks"][key]["attempts"] = [dict(status="FAILED")]
        pool._save(tmp_path, state)
    with pytest.raises(ValueError, match="started task"):
        scheduling.migrate_evaluation_schedule(tmp_path)
    assert not (tmp_path / scheduling.AMENDMENT).exists()


@pytest.mark.parametrize("artifact", ["CONFIRM_OPENED.json", "EVAL/b0/PARENT/shard0/groups/0.json"])
def test_migration_refuses_existing_evaluation_outputs(tmp_path, artifact):
    original_pool(tmp_path)
    pool._write(tmp_path / artifact, {})
    with pytest.raises(ValueError, match=r"opened|outputs already"):
        scheduling.migrate_evaluation_schedule(tmp_path)


def test_migration_refuses_unrelated_registry_change(tmp_path):
    before = original_pool(tmp_path)
    registry = pool._read(tmp_path / "TASKS.json")
    registry[0]["block"] = 99
    pool._write(tmp_path / "TASKS.json", registry)
    with pytest.raises(ValueError, match="differs from both"):
        scheduling.migrate_evaluation_schedule(tmp_path)
    assert pool._read(tmp_path / "pool/STATE.json") == before


def test_migration_recovers_interruption_after_registry_publication(tmp_path, monkeypatch):
    before = original_pool(tmp_path)
    save = pool._save

    def interrupted(*args):
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(pool, "_save", interrupted)
    with pytest.raises(OSError, match="publication interruption"):
        scheduling.migrate_evaluation_schedule(tmp_path)
    assert pool._read(tmp_path / "pool/STATE.json") == before
    assert pool._read(tmp_path / "TASKS.json") == registered_tasks()
    monkeypatch.setattr(pool, "_save", save)
    assert scheduling.migrate_evaluation_schedule(tmp_path)["status"] == "APPLIED"
    assert pool._read(tmp_path / scheduling.STATE_BACKUP) == before


def test_new_registry_without_prepared_migration_is_rejected(tmp_path):
    original_pool(tmp_path)
    pool._write(tmp_path / "TASKS.json", registered_tasks())
    with pytest.raises(ValueError, match="without migration backups"):
        scheduling.migrate_evaluation_schedule(tmp_path)


def test_migration_rejects_unregistered_scientific_spec_change(tmp_path):
    original_pool(tmp_path)
    with pool._locked(tmp_path) as state:
        state["tasks"]["A-b0-LZ"]["spec"] = copy.deepcopy(state["tasks"]["A-b0-LZ"]["spec"])
        state["tasks"]["A-b0-LZ"]["spec"]["block"] = 2
        pool._save(tmp_path, state)
    with pytest.raises(ValueError, match="supported original"):
        scheduling.migrate_evaluation_schedule(tmp_path)
