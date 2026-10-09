"""Interrupted I/O recovery adds an immutable fourth source-identity link."""

import copy
import importlib.util
from pathlib import Path

import pytest

from mm_core.execution import object_hash
from sr_f1.contract import PLAN_ID, file_hash
from sr_f1.freeze import (
    ENGINE_IO_ALLOWED_FILES,
    ENGINE_IO_REPAIR_ID,
    ENGINE_STORAGE_ALLOWED_FILES,
    ENGINE_STORAGE_REPAIR_ID,
    verify_engine_io_repair,
    verify_engine_memory_repair,
    verify_engine_storage_repair,
    verify_qos_scope_repair,
)


def load_storage_fixtures():
    spec = importlib.util.spec_from_file_location(
        "engine_io_freeze_fixtures", Path(__file__).with_name("test_engine_storage_freeze.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = load_storage_fixtures()
memory_case = FIXTURES.memory_case
storage_case = FIXTURES.storage_case
write = FIXTURES.write
read = FIXTURES.read
PREFIX = "technical_incidents/io_exit120_20261010/"
TRACK = "engineering/engine/natural/continuous"
MARKERS = [
    "technical_incidents/engine_memory_20261010/" + name
    for name in ("ZERO_UPDATE_REUSE_ACTIVATED.json", "ZERO_UPDATE_REUSE_PROCESS.json")
]
HISTORY_NAMES = [
    "PRESERVATION.json",
    "TERMINAL_JOBS.json",
    "CPU_STATE_REVIEW.json",
    "RECOVERY_REVIEW.json",
]


@pytest.fixture
def io_case(storage_case):
    root, storage_source, storage_receipt = storage_case
    snapshot = root / "code_before_engine_io_20261010"
    previous_files = {}
    for name in storage_source["source_file_hashes"]:
        content = (root / "code_before_engine_storage_20261010" / name).read_text()
        if name in ENGINE_STORAGE_ALLOWED_FILES:
            content += " authenticated storage fix"
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        previous_files[name] = file_hash(target)
    previous = dict(
        source_commit="56" * 20,
        source_file_hashes=previous_files,
        source_tree_sha256=object_hash(previous_files),
        source_dirty_files=[],
    )
    write(snapshot / "SOURCE_DEPLOYMENT.json", previous)
    storage_receipt.update(previous)
    write(root / "ENGINE_STORAGE_REPAIR.json", storage_receipt)
    memory = read(root / "ENGINE_MEMORY_REPAIR.json")
    original = memory["zero_update_recovery"]
    activation = dict(
        plan_id=PLAN_ID,
        repair_sha256=file_hash(root / "ENGINE_MEMORY_REPAIR.json"),
        staged_track_relative_path=TRACK,
        archived_track_relative_path="technical_incidents/engine_memory_20261010/evidence/" + TRACK,
        original_raw_count=128,
        engine_mode="natural",
        resume_step=0,
        original_checkpoint_state_hash=original["original_checkpoint_state_hash"],
        staged_inventory_sha256=object_hash(original["artifact_hashes"]),
        activating_pid=111,
    )
    write(root / MARKERS[0], activation)
    write(
        root / MARKERS[1],
        dict(
            repair_sha256=file_hash(root / "ENGINE_MEMORY_REPAIR.json"),
            activation_sha256=file_hash(root / MARKERS[0]),
            pid=222,
            resume_step=0,
        ),
    )
    files = {}
    for name in [*original["artifact_hashes"], *MARKERS]:
        if name in MARKERS:
            prior = root / name
        else:
            prior = root / "technical_incidents/host_memory_20261010/evidence" / name
        target = root / (PREFIX + "evidence/" + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(prior.read_bytes())
        files[name] = dict(sha256=file_hash(target), bytes=target.stat().st_size)
    state = root / (PREFIX + "evidence/orchestration/STATE.json")
    write(
        state,
        dict(
            test_sealed=True,
            tasks={
                "ENGINE": {
                    "status": "BLOCKED",
                    "attempts": [{"attempt_id": "ENGINE_attempt0002", "job_id": "196156"}],
                },
                "SCIENCE": {"attempts": []},
            },
        ),
    )
    files["orchestration/STATE.json"] = dict(sha256=file_hash(state), bytes=state.stat().st_size)
    write(
        root / (PREFIX + "PRESERVATION.json"),
        dict(
            artifact_hashes=files,
            files=len(files),
            bytes=sum(entry["bytes"] for entry in files.values()),
        ),
    )
    write(
        root / (PREFIX + "TERMINAL_JOBS.json"),
        dict(
            controller_terminal=True,
            gpu_terminal=True,
            controller_job_id="196155",
            gpu_job_id="196156",
            gpu_exit_code="120:0",
        ),
    )
    write(
        root / (PREFIX + "CPU_STATE_REVIEW.json"),
        dict(
            status="PASS_ZERO_UPDATE_FULL_STATE",
            committed_logical_step=0,
            physical_optimizer_updates=0,
            optimizer_empty=True,
            raw_rollouts=128,
            raw_record_hashes_verified=True,
            learning_rate=1e-4,
            checkpoint_state_hash=original["original_checkpoint_state_hash"],
        ),
    )
    write(
        root / (PREFIX + "RECOVERY_REVIEW.json"),
        dict(
            status="PASS_INTERRUPTED_ENGINE_RESTART",
            science_attempts=0,
            test_sealed=True,
            once_activation_consumed=True,
            once_process_consumed=True,
            journal_integrity=True,
            original_artifact_hashes_unchanged=True,
        ),
    )
    current_files = dict(previous_files)
    for name in ENGINE_IO_ALLOWED_FILES:
        current_files[name] = "1" * 64
    actual = dict(
        source_commit="12" * 20,
        source_file_hashes=current_files,
        source_tree_sha256=object_hash(current_files),
        source_dirty_files=[],
    )
    receipt = dict(
        repair_id=ENGINE_IO_REPAIR_ID,
        status="AUTHORIZED_TECHNICAL_IO_REPAIR",
        plan_id=PLAN_ID,
        run_root=str(root),
        original_execution_freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        previous_engine_storage_repair_sha256=file_hash(root / "ENGINE_STORAGE_REPAIR.json"),
        previous_source_commit=previous["source_commit"],
        previous_source_tree_sha256=previous["source_tree_sha256"],
        preserved_source_relative_path=snapshot.name,
        scientific_protocol_unchanged=True,
        old_onetime_consumed_preserved=True,
        failed_attempt_id="ENGINE_attempt0002",
        failed_job_id="196156",
        gpu_worker_constraint="highmem",
        minimum_gpu_host_memory_gb=80,
        cpu_activation_budget_bytes=48 * 1024**3,
        activation_storage="BOUNDED_CPU_EXACT_DTYPE_EXTERNAL_DISK_SPILL",
        quota_root="/projects/_hdd/varunhdd",
        activation_spill_directory=f"/projects/_hdd/varunhdd/louis-ssvc/{root.name}_activation_offload",
        quota_reserve_bytes=10 * 1024**3,
        authorized_user_message="fix technical failures and resume authorized experiment",
        authorized_at="2026-10-10",
        historical_artifact_hashes={
            PREFIX + name: file_hash(root / (PREFIX + name)) for name in HISTORY_NAMES
        },
        changed_files=sorted(ENGINE_IO_ALLOWED_FILES),
        **actual,
    )
    write(root / "ENGINE_IO_REPAIR.json", receipt)
    return root, actual, receipt


def bind_history(root, receipt, name, value):
    relative = PREFIX + name
    write(root / relative, value)
    receipt["historical_artifact_hashes"][relative] = file_hash(root / relative)
    write(root / "ENGINE_IO_REPAIR.json", receipt)


def bind_preserved(root, receipt, relative, value):
    path = root / (PREFIX + "evidence/" + relative)
    write(path, value)
    preservation = read(root / (PREFIX + "PRESERVATION.json"))
    preservation["artifact_hashes"][relative] = dict(
        sha256=file_hash(path), bytes=path.stat().st_size
    )
    preservation["bytes"] = sum(e["bytes"] for e in preservation["artifact_hashes"].values())
    bind_history(root, receipt, "PRESERVATION.json", preservation)


def test_entire_identity_chain_keeps_prior_receipts_immutable(io_case):
    root, actual, _ = io_case
    originals = {
        name: file_hash(root / name)
        for name in (
            "EXECUTION_FREEZE.json",
            "QOS_SCOPE_REPAIR.json",
            "ENGINE_MEMORY_REPAIR.json",
            "ENGINE_STORAGE_REPAIR.json",
            *MARKERS,
        )
    }
    io = verify_engine_io_repair(root, actual_source=actual)
    storage = verify_engine_storage_repair(root, actual_source=actual)
    memory = verify_engine_memory_repair(root, actual_source=actual)
    qos = verify_qos_scope_repair(root, actual_source=actual)
    assert io["repair_id"] == ENGINE_IO_REPAIR_ID
    assert io["repair_sha256"] == file_hash(root / "ENGINE_IO_REPAIR.json")
    assert storage["repair_id"] == ENGINE_STORAGE_REPAIR_ID
    assert storage["repair_sha256"] == originals["ENGINE_STORAGE_REPAIR.json"]
    assert storage["engine_io_repair_sha256"] == io["repair_sha256"]
    assert memory["repair_sha256"] == originals["ENGINE_MEMORY_REPAIR.json"]
    assert qos["repair_sha256"] == originals["QOS_SCOPE_REPAIR.json"]
    assert io["quota_root"] == "/projects/_hdd/varunhdd"
    for result in (io, storage, memory, qos):
        assert result["source_commit"] == actual["source_commit"]
        assert result["original_execution_freeze_sha256"] == originals["EXECUTION_FREEZE.json"]
    assert {name: file_hash(root / name) for name in originals} == originals


def test_future_actual_state_and_checkpoints_do_not_replace_history(io_case):
    root, actual, _ = io_case
    before = verify_engine_io_repair(root, actual_source=actual)
    write(root / "orchestration/STATE.json", {"tasks": {"ENGINE": {"status": "COMPLETE"}}})
    write(root / (TRACK + "/checkpoints/LATEST.json"), {"step": 4})
    write(root / (TRACK + "/PROCESS-next.json"), {"pid": 999, "initial_step": 0})
    assert verify_engine_io_repair(root, actual_source=actual) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("previous_engine_storage_repair_sha256", "0" * 64),
        ("previous_source_commit", "0" * 40),
        ("original_execution_freeze_sha256", "0" * 64),
        ("scientific_protocol_unchanged", False),
        ("old_onetime_consumed_preserved", False),
        ("failed_attempt_id", "ENGINE_attempt0001"),
        ("failed_job_id", "196139"),
        ("gpu_worker_constraint", ""),
        ("minimum_gpu_host_memory_gb", 33),
        ("cpu_activation_budget_bytes", 96 * 1024**3),
        ("quota_reserve_bytes", 0),
        ("activation_storage", "lossy"),
        ("quota_root", "/tmp"),
        ("activation_spill_directory", "/projects/_hdd/varunhdd/louis-ssvc/other_experiment"),
        ("activation_spill_directory", "/projects/_hdd/varunhdd/louis-ssvc/../activation_offload"),
        ("authorized_user_message", ""),
        ("authorized_at", ""),
    ],
)
def test_exact_authorized_io_scope_cannot_drift(io_case, field, value):
    root, actual, receipt = io_case
    receipt[field] = value
    write(root / "ENGINE_IO_REPAIR.json", receipt)
    with pytest.raises(PermissionError):
        verify_engine_io_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    ("file", "field", "value"),
    [
        ("TERMINAL_JOBS.json", "controller_terminal", False),
        ("TERMINAL_JOBS.json", "gpu_terminal", False),
        ("TERMINAL_JOBS.json", "controller_job_id", "196138"),
        ("TERMINAL_JOBS.json", "gpu_job_id", "196139"),
        ("TERMINAL_JOBS.json", "gpu_exit_code", "0:0"),
        ("CPU_STATE_REVIEW.json", "physical_optimizer_updates", 1),
        ("CPU_STATE_REVIEW.json", "committed_logical_step", 1),
        ("CPU_STATE_REVIEW.json", "optimizer_empty", False),
        ("CPU_STATE_REVIEW.json", "raw_rollouts", 127),
        ("CPU_STATE_REVIEW.json", "learning_rate", 1e-5),
        ("CPU_STATE_REVIEW.json", "checkpoint_state_hash", "0" * 64),
        ("RECOVERY_REVIEW.json", "once_activation_consumed", False),
        ("RECOVERY_REVIEW.json", "once_process_consumed", False),
        ("RECOVERY_REVIEW.json", "science_attempts", 1),
        ("RECOVERY_REVIEW.json", "test_sealed", False),
        ("RECOVERY_REVIEW.json", "journal_integrity", False),
        ("RECOVERY_REVIEW.json", "original_artifact_hashes_unchanged", False),
        ("PRESERVATION.json", "files", 0),
        ("PRESERVATION.json", "bytes", 0),
    ],
)
def test_rehashed_historical_reviews_still_must_match_boundary(io_case, file, field, value):
    root, actual, receipt = io_case
    history = read(root / (PREFIX + file))
    history[field] = value
    bind_history(root, receipt, file, history)
    with pytest.raises(PermissionError):
        verify_engine_io_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    "relative",
    [
        PREFIX + "CPU_STATE_REVIEW.json",
        PREFIX + "evidence/" + TRACK + "/rollouts/01-00-0.json",
        PREFIX + "evidence/" + MARKERS[0],
        MARKERS[0],
        MARKERS[1],
        "code_before_engine_io_20261010/src/sr_f1/runtime.py",
        "code_before_engine_storage_20261010/src/sr_f1/runtime.py",
    ],
)
def test_history_source_and_consumed_marker_tampering_is_rejected(io_case, relative):
    root, actual, _ = io_case
    (root / relative).write_text("tampered historical bytes")
    with pytest.raises(PermissionError):
        verify_qos_scope_repair(root, actual_source=actual)


def test_rehashed_raw_cannot_replace_original_authenticated_recovery(io_case):
    root, actual, receipt = io_case
    bind_preserved(root, receipt, TRACK + "/rollouts/01-00-0.json", {"substituted": True})
    with pytest.raises(PermissionError, match="original recovery bytes"):
        verify_engine_io_repair(root, actual_source=actual)


@pytest.mark.parametrize("case", ["missing", "identity", "process_binding", "pid"])
def test_even_rehashed_consumed_markers_must_bind_original_once_claim(io_case, case):
    root, actual, receipt = io_case
    if case == "missing":
        preservation = read(root / (PREFIX + "PRESERVATION.json"))
        removed = preservation["artifact_hashes"].pop(MARKERS[0])
        preservation["files"] -= 1
        preservation["bytes"] -= removed["bytes"]
        bind_history(root, receipt, "PRESERVATION.json", preservation)
    else:
        name = MARKERS[0] if case == "identity" else MARKERS[1]
        marker = read(root / name)
        marker[
            {"identity": "repair_sha256", "process_binding": "activation_sha256", "pid": "pid"}[
                case
            ]
        ] = 0 if case == "pid" else "0" * 64
        write(root / name, marker)
        bind_preserved(root, receipt, name, marker)
    with pytest.raises(PermissionError):
        verify_engine_io_repair(root, actual_source=actual)


@pytest.mark.parametrize("change", ["science", "missing_allowed", "dirty", "source_tree", "commit"])
def test_io_source_changes_remain_exactly_scoped(io_case, change):
    root, actual, receipt = io_case
    actual = copy.deepcopy(actual)
    if change == "science":
        actual["source_file_hashes"]["src/sr_f1/training.py"] = "9" * 64
    elif change == "missing_allowed":
        previous = read(root / "code_before_engine_io_20261010/SOURCE_DEPLOYMENT.json")
        actual["source_file_hashes"]["src/sr_f1/runtime.py"] = previous["source_file_hashes"][
            "src/sr_f1/runtime.py"
        ]
    elif change == "dirty":
        actual["source_dirty_files"] = ["src/sr_f1/runtime.py"]
    elif change == "commit":
        actual["source_commit"] = "invalid"
    actual["source_tree_sha256"] = object_hash(actual["source_file_hashes"])
    if change == "source_tree":
        actual["source_tree_sha256"] = "0" * 64
    receipt.update(actual)
    write(root / "ENGINE_IO_REPAIR.json", receipt)
    with pytest.raises(PermissionError):
        verify_engine_io_repair(root, actual_source=actual)


@pytest.mark.parametrize("change", ["science", "test", "attempt", "absent"])
def test_preserved_state_authenticates_failed_engine_before_science(io_case, change):
    root, actual, receipt = io_case
    state = read(root / (PREFIX + "evidence/orchestration/STATE.json"))
    if change == "science":
        state["tasks"]["SCIENCE"]["attempts"] = [{"job_id": "99"}]
    elif change == "test":
        state["test_sealed"] = False
    elif change == "attempt":
        state["tasks"]["ENGINE"]["attempts"][0]["job_id"] = "196139"
    elif change == "absent":
        state["tasks"] = {}
    bind_preserved(root, receipt, "orchestration/STATE.json", state)
    with pytest.raises(PermissionError):
        verify_engine_io_repair(root, actual_source=actual)


@pytest.mark.parametrize("name", HISTORY_NAMES)
def test_required_review_cannot_be_omitted_from_hash_binding(io_case, name):
    root, actual, receipt = io_case
    receipt["historical_artifact_hashes"].pop(PREFIX + name)
    write(root / "ENGINE_IO_REPAIR.json", receipt)
    with pytest.raises(PermissionError, match="missing or unbounded"):
        verify_engine_io_repair(root, actual_source=actual)


def test_snapshot_symlink_alias_inside_root_is_rejected_even_with_genuine_bytes(io_case):
    root, actual, receipt = io_case
    declared = root / "code_before_engine_io_20261010"
    genuine = root / "genuine_preserved_source"
    declared.rename(genuine)
    declared.symlink_to(genuine.name, target_is_directory=True)
    # The old check followed the alias before testing is_symlink, then accepted
    # this rewritten name despite every preserved source byte remaining genuine.
    receipt["preserved_source_relative_path"] = genuine.name
    write(root / "ENGINE_IO_REPAIR.json", receipt)
    with pytest.raises(PermissionError, match="snapshot must be a real directory"):
        verify_engine_io_repair(root, actual_source=actual)
