"""Append-only activation-storage repair keeps the original QoS/freeze chain."""

import copy
import json

import pytest

from mm_core.execution import object_hash
from sr_f1.contract import PLAN_ID, file_hash
from sr_f1.freeze import (
    ENGINE_MEMORY_ALLOWED_FILES,
    ENGINE_MEMORY_REPAIR_ID,
    QOS_SCOPE_REPAIR_ID,
    verify_engine_memory_repair,
    verify_qos_scope_repair,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def memory_case(tmp_path):
    root = tmp_path
    original = {n: "original " + n for n in ENGINE_MEMORY_ALLOWED_FILES}
    original.update(
        {
            "src/sr_f1/training.py": "unchanged training",
            "scripts/sr_f1/run_worker.py": "unchanged worker",
            "docs/sr_f1/amendments/v1/AMENDMENT.md": "unchanged protocol",
        }
    )

    def snapshot(name, contents, commit):
        for n, text in contents.items():
            p = root / name / n
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        hashes = {n: file_hash(root / name / n) for n in contents}
        return dict(
            source_commit=commit,
            source_file_hashes=hashes,
            source_tree_sha256=object_hash(hashes),
            source_dirty_files=[],
        )

    first = snapshot("code_before_qos_scope_20261010", original, "a" * 40)
    write(root / "SOURCE_AND_RENDER_MANIFEST.json", first)
    write(
        root / "EXECUTION_FREEZE.json",
        dict(
            experiment="SR-F1.1-20261010",
            status="FROZEN",
            artifact_hashes={
                "SOURCE_AND_RENDER_MANIFEST.json": file_hash(
                    root / "SOURCE_AND_RENDER_MANIFEST.json"
                )
            },
        ),
    )
    write(
        root / "manifests/ALLOCATION_PERMISSION.json",
        dict(qos="soujanya-poria-startfund-2026-03", gpus_per_worker=1),
    )
    qos_prefix = "technical_incidents/qos_scope_20261010/"
    write(
        root / (qos_prefix + "STATE_BEFORE.json"),
        dict(test_sealed=True, tasks={"COMMON_START": {"attempts": []}}),
    )
    write(root / (qos_prefix + "CONTROLLER_TERMINAL.json"), dict(old_controller_terminal=True))
    qos_contents = dict(original)
    for n in ["src/sr_f1/freeze.py", "src/sr_f1/orchestration.py"]:
        qos_contents[n] += " qos fix"
    previous = snapshot("code_before_engine_memory_20261010", qos_contents, "b" * 40)
    write(root / "code_before_engine_memory_20261010/SOURCE_DEPLOYMENT.json", previous)
    qos = dict(
        repair_id=QOS_SCOPE_REPAIR_ID,
        status="USER_AUTHORIZED_QOS_SCHEDULER_ONLY",
        plan_id=PLAN_ID,
        run_root=str(root),
        original_execution_freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        original_source_tree_sha256=first["source_tree_sha256"],
        original_source_commit=first["source_commit"],
        permission_sha256=file_hash(root / "manifests/ALLOCATION_PERMISSION.json"),
        capacity_mode="TEACHER_QOS_SCHEDULER_ONLY",
        qos="soujanya-poria-startfund-2026-03",
        gpus_per_worker=1,
        scientific_protocol_unchanged=True,
        new_generation_count_before_activation=0,
        no_gpu_attempts_before_activation=True,
        preserved_source_relative_path="code_before_qos_scope_20261010",
        authorized_user_message="remove extra guard",
        authorized_at="2026-10-10",
        changed_files=["src/sr_f1/freeze.py", "src/sr_f1/orchestration.py"],
        pre_activation_artifact_hashes={
            qos_prefix + n: file_hash(root / (qos_prefix + n))
            for n in ["STATE_BEFORE.json", "CONTROLLER_TERMINAL.json"]
        },
        **previous,
    )
    write(root / "QOS_SCOPE_REPAIR.json", qos)
    prefix = "technical_incidents/engine_memory_20261010/"
    track = "engineering/engine/natural/continuous"
    evidence = {
        "orchestration/STATE.json": dict(
            test_sealed=True,
            tasks={
                "COMMON_START": {"status": "COMPLETE", "attempts": [{}]},
                "ENGINE": {"status": "BLOCKED", "attempts": [{}]},
                "SCIENCE": {"status": "WAITING", "attempts": []},
            },
        ),
        track + "/RUN_MANIFEST.json": {"frozen": True},
        track + "/checkpoints/LATEST.json": {"step": 0},
        track + "/checkpoints/commit-00.json": {"step": 0},
        track + "/rollouts/01-00-0.json": {"preserved": True},
    }
    files = {}
    for n, value in evidence.items():
        p = root / (prefix + "evidence/" + n)
        write(p, value)
        files[n] = dict(sha256=file_hash(p), bytes=p.stat().st_size)
    write(
        root / (prefix + "EVIDENCE_MANIFEST.json"),
        dict(status="BYTE_VERIFIED_PRESERVED", files=files),
    )
    write(
        root / (prefix + "TERMINAL_JOBS.json"),
        dict(old_controller_terminal=True, old_engine_terminal=True),
    )
    write(
        root / (prefix + "CPU_STATE_REVIEW.json"),
        dict(
            status="PASS_ZERO_UPDATE_FULL_STATE",
            committed_logical_step=0,
            physical_optimizer_updates=0,
            optimizer_empty=True,
            raw_rollouts=128,
            raw_record_hashes_verified=True,
            learning_rate=1e-4,
            checkpoint_state_hash="e" * 64,
        ),
    )
    after = dict(previous["source_file_hashes"])
    for n in ENGINE_MEMORY_ALLOWED_FILES:
        after[n] = "c" * 64
    actual = dict(
        source_commit="d" * 40,
        source_file_hashes=after,
        source_tree_sha256=object_hash(after),
        source_dirty_files=[],
    )
    receipt = dict(
        repair_id=ENGINE_MEMORY_REPAIR_ID,
        status="AUTHORIZED_TECHNICAL_MEMORY_REPAIR",
        plan_id=PLAN_ID,
        run_root=str(root),
        original_execution_freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        previous_qos_repair_sha256=file_hash(root / "QOS_SCOPE_REPAIR.json"),
        previous_source_commit=previous["source_commit"],
        previous_source_tree_sha256=previous["source_tree_sha256"],
        preserved_source_relative_path="code_before_engine_memory_20261010",
        scientific_protocol_unchanged=True,
        optimizer_updates_before_repair=0,
        gpu_worker_host_memory_gb=384,
        activation_storage="CPU_SAVED_NONPARAMETER_TENSORS_EXACT_DTYPE",
        authorized_user_message="fix and resume",
        authorized_at="2026-10-10",
        historical_artifact_hashes={
            prefix + n: file_hash(root / (prefix + n))
            for n in ["EVIDENCE_MANIFEST.json", "TERMINAL_JOBS.json", "CPU_STATE_REVIEW.json"]
        },
        zero_update_recovery=dict(
            staged_track_relative_path=track,
            archived_track_relative_path=prefix + "evidence/" + track,
            original_checkpoint_state_hash="e" * 64,
            original_raw_count=128,
            engine_mode="natural",
            resume_step=0,
            artifact_hashes={n: v["sha256"] for n, v in files.items() if n.startswith(track + "/")},
        ),
        changed_files=sorted(ENGINE_MEMORY_ALLOWED_FILES),
        **actual,
    )
    write(root / "ENGINE_MEMORY_REPAIR.json", receipt)
    return root, actual, receipt


def test_memory_chain_and_capacity_keep_both_original_receipts(memory_case):
    root, actual, _ = memory_case
    frozen = file_hash(root / "EXECUTION_FREEZE.json")
    qos = file_hash(root / "QOS_SCOPE_REPAIR.json")
    repair = verify_engine_memory_repair(root, actual_source=actual)
    capacity = verify_qos_scope_repair(root, actual_source=actual)
    assert repair["gpu_worker_host_memory_gb"] == 384
    assert capacity["repair_sha256"] == qos
    assert capacity["engine_memory_repair_sha256"] == repair["repair_sha256"]
    write(root / "orchestration/STATE.json", {"new_work": True})
    write(root / "engineering/engine/natural/continuous/checkpoints/LATEST.json", {"step": 4})
    assert verify_engine_memory_repair(root, actual_source=actual) == repair
    assert file_hash(root / "EXECUTION_FREEZE.json") == frozen
    assert file_hash(root / "QOS_SCOPE_REPAIR.json") == qos


@pytest.mark.parametrize(
    "field,value",
    [
        ("previous_qos_repair_sha256", "0" * 64),
        ("gpu_worker_host_memory_gb", 1024),
        ("scientific_protocol_unchanged", False),
        ("optimizer_updates_before_repair", 1),
        ("authorized_user_message", ""),
        ("activation_storage", "detached_cache"),
    ],
)
def test_memory_repair_scope_cannot_drift(memory_case, field, value):
    root, actual, receipt = memory_case
    receipt[field] = value
    write(root / "ENGINE_MEMORY_REPAIR.json", receipt)
    with pytest.raises(PermissionError):
        verify_engine_memory_repair(root, actual_source=actual)


def test_scientific_source_change_is_rejected(memory_case):
    root, actual, receipt = memory_case
    actual = copy.deepcopy(actual)
    actual["source_file_hashes"]["src/sr_f1/training.py"] = "9" * 64
    actual["source_tree_sha256"] = object_hash(actual["source_file_hashes"])
    receipt.update(actual)
    write(root / "ENGINE_MEMORY_REPAIR.json", receipt)
    with pytest.raises(PermissionError, match="scientific source"):
        verify_engine_memory_repair(root, actual_source=actual)


def test_preserved_failures_and_previous_source_are_authenticated(memory_case):
    root, actual, _ = memory_case
    p = (
        root
        / "technical_incidents/engine_memory_20261010/evidence"
        / "engineering/engine/natural/continuous/rollouts/01-00-0.json"
    )
    p.write_text("changed")
    with pytest.raises(PermissionError, match="Preserved failure"):
        verify_engine_memory_repair(root, actual_source=actual)
