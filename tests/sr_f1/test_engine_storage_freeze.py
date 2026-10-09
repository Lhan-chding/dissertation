"""Bounded storage repair extends, rather than replaces, the frozen identity chain."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from mm_core.execution import object_hash
from sr_f1.contract import PLAN_ID, file_hash
from sr_f1.freeze import (
    ENGINE_MEMORY_ALLOWED_FILES,
    ENGINE_STORAGE_ALLOWED_FILES,
    ENGINE_STORAGE_REPAIR_ID,
    verify_engine_memory_repair,
    verify_engine_storage_repair,
    verify_qos_scope_repair,
)

PREFIX = "technical_incidents/host_memory_20261010/"
TRACK = "engineering/engine/natural/continuous"


def load_memory_fixtures():
    spec = importlib.util.spec_from_file_location(
        "engine_storage_freeze_fixtures", Path(__file__).with_name("test_engine_memory_freeze.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = load_memory_fixtures()
memory_case = FIXTURES.memory_case
write = FIXTURES.write


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def storage_case(memory_case):
    root, memory_source, memory_receipt = memory_case
    saved_source = root / "code_before_engine_storage_20261010"
    previous_files = {}
    for name in memory_source["source_file_hashes"]:
        source = root / "code_before_engine_memory_20261010" / name
        content = source.read_text()
        if name in ENGINE_MEMORY_ALLOWED_FILES:
            content += " authenticated memory fix"
        target = saved_source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        previous_files[name] = file_hash(target)
    previous = dict(
        source_commit="6d" * 20,
        source_file_hashes=previous_files,
        source_tree_sha256=object_hash(previous_files),
        source_dirty_files=[],
    )
    write(saved_source / "SOURCE_DEPLOYMENT.json", previous)
    # Give the shared memory fixture real preserved source bytes for the third
    # link. Its original QoS snapshot and freeze remain byte-for-byte unchanged.
    memory_receipt.update(previous)
    write(root / "ENGINE_MEMORY_REPAIR.json", memory_receipt)
    original = memory_receipt["zero_update_recovery"]["artifact_hashes"]
    files = {}
    for name in original:
        saved = root / (PREFIX + "evidence/" + name)
        saved.parent.mkdir(parents=True, exist_ok=True)
        prior = root / "technical_incidents/engine_memory_20261010/evidence" / name
        saved.write_bytes(prior.read_bytes())
        files[name] = dict(sha256=file_hash(saved), bytes=saved.stat().st_size)
    state = root / (PREFIX + "evidence/orchestration/STATE.json")
    write(
        state,
        dict(
            test_sealed=True,
            tasks={
                "ENGINE": {
                    "status": "SUBMITTED",
                    "attempts": [{"attempt_id": "ENGINE_attempt0001", "job_id": "196139"}],
                },
                "SCIENCE": {"attempts": []},
            },
        ),
    )
    files["orchestration/STATE.json"] = dict(sha256=file_hash(state), bytes=state.stat().st_size)
    write(
        root / (PREFIX + "EVIDENCE_MANIFEST.json"),
        dict(status="BYTE_VERIFIED_PRESERVED", files=files),
    )
    write(
        root / (PREFIX + "TERMINAL_JOBS.json"),
        dict(controller_terminal=True, gpu_terminal=True, gpu_never_started=True),
    )
    write(
        root / (PREFIX + "HOST_STATE_REVIEW.json"),
        dict(
            status="PASS_UNCONSUMED_ZERO_UPDATE_STATE",
            committed_logical_step=0,
            physical_optimizer_updates=0,
            raw_rollouts=128,
            staged_artifact_hashes_unchanged=True,
            staged_artifact_hashes=original,
            recovery_activation_exists=False,
            recovery_process_exists=False,
            physical_process_files=0,
            superseded_gpu_job_id="196139",
            superseded_gpu_elapsed_seconds=0,
            science_attempts=0,
            test_sealed=True,
        ),
    )
    current_files = dict(previous_files)
    for name in ENGINE_STORAGE_ALLOWED_FILES:
        current_files[name] = "f" * 64
    actual = dict(
        source_commit="e" * 40,
        source_file_hashes=current_files,
        source_tree_sha256=object_hash(current_files),
        source_dirty_files=[],
    )
    receipt = dict(
        repair_id=ENGINE_STORAGE_REPAIR_ID,
        status="AUTHORIZED_TECHNICAL_STORAGE_REPAIR",
        plan_id=PLAN_ID,
        run_root=str(root),
        original_execution_freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        previous_engine_memory_repair_sha256=file_hash(root / "ENGINE_MEMORY_REPAIR.json"),
        previous_source_commit=previous["source_commit"],
        previous_source_tree_sha256=previous["source_tree_sha256"],
        preserved_source_relative_path=saved_source.name,
        scientific_protocol_unchanged=True,
        gpu_worker_constraint="highmem",
        minimum_gpu_host_memory_gb=80,
        cpu_activation_budget_bytes=48 * 1024**3,
        activation_storage="BOUNDED_CPU_EXACT_DTYPE_DISK_SPILL",
        superseded_unstarted_attempt_id="ENGINE_attempt0001",
        superseded_unstarted_job_id="196139",
        authorized_user_message="fix technical failures and resume the authorized training",
        authorized_at="2026-10-10",
        historical_artifact_hashes={
            PREFIX + name: file_hash(root / (PREFIX + name))
            for name in ("EVIDENCE_MANIFEST.json", "TERMINAL_JOBS.json", "HOST_STATE_REVIEW.json")
        },
        changed_files=sorted(ENGINE_STORAGE_ALLOWED_FILES),
        **actual,
    )
    write(root / "ENGINE_STORAGE_REPAIR.json", receipt)
    return root, actual, receipt


def update_bound_history(root, receipt, name, value):
    relative = PREFIX + name
    write(root / relative, value)
    receipt["historical_artifact_hashes"][relative] = file_hash(root / relative)
    write(root / "ENGINE_STORAGE_REPAIR.json", receipt)


def test_public_identity_gates_authenticate_the_complete_storage_chain(storage_case):
    root, actual, _ = storage_case
    immutable = {
        name: file_hash(root / name)
        for name in ("EXECUTION_FREEZE.json", "QOS_SCOPE_REPAIR.json", "ENGINE_MEMORY_REPAIR.json")
    }
    storage = verify_engine_storage_repair(root, actual_source=actual)
    memory = verify_engine_memory_repair(root, actual_source=actual)
    qos = verify_qos_scope_repair(root, actual_source=actual)
    assert storage["repair_sha256"] == file_hash(root / "ENGINE_STORAGE_REPAIR.json")
    assert storage["gpu_worker_constraint"] == "highmem"
    assert storage["minimum_gpu_host_memory_gb"] == 80
    assert memory["gpu_worker_host_memory_gb"] == 80
    assert memory["engine_storage_repair_sha256"] == storage["repair_sha256"]
    assert memory["repair_sha256"] == immutable["ENGINE_MEMORY_REPAIR.json"]
    assert qos["repair_sha256"] == immutable["QOS_SCOPE_REPAIR.json"]
    assert qos["engine_memory_repair_sha256"] == memory["repair_sha256"]
    assert qos["qos"] == "soujanya-poria-startfund-2026-03"
    assert qos["capacity_mode"] == "TEACHER_QOS_SCHEDULER_ONLY"
    for result in (storage, memory, qos):
        assert result["source_commit"] == actual["source_commit"]
        assert result["original_execution_freeze_sha256"] == immutable["EXECUTION_FREEZE.json"]
    assert {name: file_hash(root / name) for name in immutable} == immutable


def test_live_progress_never_replaces_the_historical_storage_activation_boundary(storage_case):
    root, actual, _ = storage_case
    gates = (verify_engine_storage_repair, verify_engine_memory_repair, verify_qos_scope_repair)
    before = [gate(root, actual_source=actual) for gate in gates]
    write(root / "orchestration/STATE.json", {"tasks": {"ENGINE": {"status": "COMPLETE"}}})
    write(root / (TRACK + "/checkpoints/LATEST.json"), {"step": 4})
    write(root / (TRACK + "/PROCESS-new.json"), {"initial_step": 0, "pid": 42})
    for name in ("ZERO_UPDATE_REUSE_ACTIVATED.json", "ZERO_UPDATE_REUSE_PROCESS.json"):
        write(root / "technical_incidents/engine_memory_20261010" / name, {"consumed": True})
    assert [gate(root, actual_source=actual) for gate in gates] == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("previous_engine_memory_repair_sha256", "0" * 64),
        ("previous_source_commit", "0" * 40),
        ("scientific_protocol_unchanged", False),
        ("gpu_worker_constraint", ""),
        ("minimum_gpu_host_memory_gb", 33),
        ("cpu_activation_budget_bytes", 96 * 1024**3),
        ("activation_storage", "lossy_disk_spill"),
        ("superseded_unstarted_attempt_id", "ENGINE_attempt0000"),
        ("superseded_unstarted_job_id", "196140"),
        ("authorized_user_message", ""),
        ("authorized_at", ""),
    ],
)
def test_storage_authorization_scope_cannot_drift(storage_case, field, value):
    root, actual, receipt = storage_case
    receipt[field] = value
    write(root / "ENGINE_STORAGE_REPAIR.json", receipt)
    with pytest.raises(PermissionError):
        verify_engine_storage_repair(root, actual_source=actual)


@pytest.mark.parametrize("field", ["controller_terminal", "gpu_terminal", "gpu_never_started"])
def test_replacement_cannot_start_until_all_original_allocations_are_terminal(storage_case, field):
    root, actual, receipt = storage_case
    terminal = read(root / (PREFIX + "TERMINAL_JOBS.json"))
    terminal[field] = False
    update_bound_history(root, receipt, "TERMINAL_JOBS.json", terminal)
    with pytest.raises(PermissionError, match="not terminal"):
        verify_engine_storage_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("committed_logical_step", 1),
        ("physical_optimizer_updates", 1),
        ("raw_rollouts", 127),
        ("staged_artifact_hashes_unchanged", False),
        ("recovery_activation_exists", True),
        ("recovery_process_exists", True),
        ("physical_process_files", 1),
        ("superseded_gpu_job_id", "196140"),
        ("superseded_gpu_elapsed_seconds", 1),
        ("science_attempts", 1),
        ("test_sealed", False),
    ],
)
def test_even_rehashed_history_must_be_an_unconsumed_zero_update_state(storage_case, field, value):
    root, actual, receipt = storage_case
    review = read(root / (PREFIX + "HOST_STATE_REVIEW.json"))
    review[field] = value
    update_bound_history(root, receipt, "HOST_STATE_REVIEW.json", review)
    with pytest.raises(PermissionError, match="Invalid unstarted state review"):
        verify_engine_storage_repair(root, actual_source=actual)


def test_review_must_bind_the_previous_recovery_inventory_exactly(storage_case):
    root, actual, receipt = storage_case
    review = read(root / (PREFIX + "HOST_STATE_REVIEW.json"))
    review["staged_artifact_hashes"].pop(TRACK + "/checkpoints/LATEST.json")
    update_bound_history(root, receipt, "HOST_STATE_REVIEW.json", review)
    with pytest.raises(PermissionError, match="preserve original step-zero"):
        verify_engine_storage_repair(root, actual_source=actual)


def test_rehashed_preserved_raw_still_must_match_the_earlier_memory_receipt(storage_case):
    root, actual, receipt = storage_case
    relative = TRACK + "/rollouts/01-00-0.json"
    raw = root / (PREFIX + "evidence/" + relative)
    write(raw, {"replacement": "forbidden"})
    manifest = read(root / (PREFIX + "EVIDENCE_MANIFEST.json"))
    manifest["files"][relative] = dict(sha256=file_hash(raw), bytes=raw.stat().st_size)
    update_bound_history(root, receipt, "EVIDENCE_MANIFEST.json", manifest)
    with pytest.raises(PermissionError, match="authenticated recovery bytes"):
        verify_engine_storage_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        (PREFIX + "HOST_STATE_REVIEW.json", "Storage history changed"),
        (PREFIX + "evidence/" + TRACK + "/rollouts/01-00-0.json", "Preserved storage artifact"),
        ("code_before_engine_storage_20261010/src/sr_f1/runtime.py", "Pre-storage source changed"),
        (
            "technical_incidents/engine_memory_20261010/evidence/"
            + TRACK
            + "/rollouts/01-00-0.json",
            "Preserved failure artifact changed",
        ),
    ],
)
def test_mutation_of_any_historical_chain_link_is_rejected(storage_case, relative, message):
    root, actual, _ = storage_case
    (root / relative).write_text("changed historical bytes")
    with pytest.raises(PermissionError, match=message):
        verify_qos_scope_repair(root, actual_source=actual)


@pytest.mark.parametrize("change", ["scientific", "missing_allowed", "dirty", "source_tree"])
def test_storage_source_identity_is_exact_and_scoped(storage_case, change):
    root, actual, receipt = storage_case
    actual = copy.deepcopy(actual)
    if change == "scientific":
        actual["source_file_hashes"]["src/sr_f1/training.py"] = "9" * 64
    elif change == "missing_allowed":
        previous = read(root / "code_before_engine_storage_20261010/SOURCE_DEPLOYMENT.json")
        actual["source_file_hashes"]["src/sr_f1/runtime.py"] = previous["source_file_hashes"][
            "src/sr_f1/runtime.py"
        ]
    elif change == "dirty":
        actual["source_dirty_files"] = ["src/sr_f1/runtime.py"]
    actual["source_tree_sha256"] = object_hash(actual["source_file_hashes"])
    if change == "source_tree":
        actual["source_tree_sha256"] = "0" * 64
    receipt.update(actual)
    write(root / "ENGINE_STORAGE_REPAIR.json", receipt)
    with pytest.raises(PermissionError):
        verify_engine_storage_repair(root, actual_source=actual)
