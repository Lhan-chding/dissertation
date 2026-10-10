"""MultiGPU activation storage extends all historical source and resource gates."""

import copy
import importlib.util
from pathlib import Path

import pytest

from mm_core.execution import object_hash
from sr_f1.contract import PLAN_ID, file_hash
from sr_f1.freeze import (
    ENGINE_IO_ALLOWED_FILES,
    ENGINE_IO_REPAIR_ID,
    ENGINE_MULTIGPU_REPAIR_ID,
    ENGINE_MULTIGPU_REQUIRED_FILES,
    verify_engine_io_repair,
    verify_engine_memory_repair,
    verify_engine_multigpu_repair,
    verify_engine_storage_repair,
    verify_qos_scope_repair,
)


def load_io_fixtures():
    spec = importlib.util.spec_from_file_location(
        "engine_multigpu_freeze_fixtures", Path(__file__).with_name("test_engine_io_freeze.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = load_io_fixtures()
memory_case = FIXTURES.memory_case
storage_case = FIXTURES.storage_case
io_case = FIXTURES.io_case
write = FIXTURES.write
read = FIXTURES.read
PREFIX = "technical_incidents/multigpu_20261010/"
TRACK = FIXTURES.TRACK
MARKERS = FIXTURES.MARKERS
HISTORY_NAMES = FIXTURES.HISTORY_NAMES
WORKER = "scripts/sr_f1/run_worker.py"


@pytest.fixture
def multigpu_case(io_case):
    root, io_source, io_receipt = io_case
    snapshot = root / "code_before_engine_multigpu_20261010"
    previous_files = {}
    for name in io_source["source_file_hashes"]:
        content = (root / "code_before_engine_io_20261010" / name).read_text()
        if name in ENGINE_IO_ALLOWED_FILES:
            content += " authenticated external IO fix"
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        previous_files[name] = file_hash(target)
    previous = dict(
        source_commit="b4" * 20,
        source_file_hashes=previous_files,
        source_tree_sha256=object_hash(previous_files),
        source_dirty_files=[],
    )
    write(snapshot / "SOURCE_DEPLOYMENT.json", previous)
    io_receipt.update(previous)
    write(root / "ENGINE_IO_REPAIR.json", io_receipt)
    registration = dict(
        freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        source_hashes={WORKER: previous_files[WORKER]},
        permission={"qos": "soujanya-poria-startfund-2026-03", "gpus_per_worker": 1},
        tasks={"ENGINE": {"gpus": 1, "operation": "engine"}},
    )
    write(root / "orchestration/REGISTRATION.json", registration)
    evidence = root / (PREFIX + "evidence")
    for name in [*MARKERS, "orchestration/REGISTRATION.json"]:
        target = evidence / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    write(
        evidence / "orchestration/STATE.json",
        dict(
            test_sealed=True,
            tasks={
                "ENGINE": {
                    "status": "RUNNING",
                    "attempts": [
                        {
                            "attempt_id": "ENGINE_attempt0003",
                            "job_id": "196227",
                        }
                    ],
                },
                "SCIENCE": {"attempts": []},
            },
        ),
    )
    write(evidence / TRACK / "RUN_MANIFEST.json", {"fresh_trace": True})
    checkpoint = evidence / TRACK / "checkpoints/step-00.pt"
    write(checkpoint, {"full_zero_state": True, "rng": "original_one_GPU"})
    latest = dict(path=checkpoint.name, step=0, sha256=file_hash(checkpoint), state_hash="7" * 64)
    for name in ["LATEST.json", "commit-00.json"]:
        write(checkpoint.parent / name, latest)
    raw = {}
    for slot in range(16):
        for index in range(8):
            name = f"01-{slot:02d}-{index}.json"
            path = evidence / TRACK / "rollouts" / name
            write(path, dict(slot=slot, sample_index=index, original_current_trace=True))
            raw[name] = file_hash(path)
    files = {
        str(path.relative_to(evidence)): dict(sha256=file_hash(path), bytes=path.stat().st_size)
        for path in evidence.rglob("*")
        if path.is_file()
    }
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
            controller_job_id="196225",
            gpu_job_id="196227",
            controller_terminal_state="CANCELLED",
            gpu_terminal_state="CANCELLED",
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
            checkpoint_state_hash=latest["state_hash"],
            checkpoint_file_sha256=latest["sha256"],
            raw_file_hashes=raw,
            source_commit=previous["source_commit"],
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
            full_engine_restart=True,
            no_partial_gradient_reuse=True,
            no_old_once_reuse=True,
        ),
    )
    current_files = dict(previous_files)
    for name in ENGINE_MULTIGPU_REQUIRED_FILES:
        current_files[name] = "2" * 64
    actual = dict(
        source_commit="34" * 20,
        source_file_hashes=current_files,
        source_tree_sha256=object_hash(current_files),
        source_dirty_files=[],
    )
    receipt = dict(
        repair_id=ENGINE_MULTIGPU_REPAIR_ID,
        status="AUTHORIZED_TECHNICAL_MULTIGPU_REPAIR",
        plan_id=PLAN_ID,
        run_root=str(root),
        original_execution_freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        previous_engine_io_repair_sha256=file_hash(root / "ENGINE_IO_REPAIR.json"),
        previous_source_commit=previous["source_commit"],
        previous_source_tree_sha256=previous["source_tree_sha256"],
        preserved_source_relative_path=snapshot.name,
        registration_sha256=file_hash(root / "orchestration/REGISTRATION.json"),
        previous_worker_source_sha256=previous_files[WORKER],
        worker_source_sha256=current_files[WORKER],
        scientific_protocol_unchanged=True,
        original_once_consumed_preserved=True,
        engine_restart_mode="FULL_REGISTERED_CONTINUOUS4_SPLIT2_PLUS2",
        maintained_attempt_id="ENGINE_attempt0003",
        maintained_job_id="196227",
        gpu_count=4,
        compute_device_index=0,
        storage_device_indices=[1, 2, 3],
        gpu_activation_budget_bytes=80 * 1024**3,
        cpu_activation_budget_bytes=48 * 1024**3,
        activation_storage="AUXILIARY_GPU_THEN_BOUNDED_CPU_EXACT_EXTERNAL_DISK",
        gpu_worker_constraint="highmem",
        minimum_gpu_host_memory_gb_per_gpu=80,
        minimum_gpu_host_memory_gb=320,
        cpus_per_gpu=4,
        qos="soujanya-poria-startfund-2026-03",
        resource_operations=["engine", "train"],
        **{
            name: io_receipt[name]
            for name in ["activation_spill_directory", "quota_root", "quota_reserve_bytes"]
        },
        authorized_user_message="immediately accelerate using available teacher QoS GPU resources",
        authorized_at="2026-10-10",
        historical_artifact_hashes={
            PREFIX + name: file_hash(root / (PREFIX + name)) for name in HISTORY_NAMES
        },
        changed_files=sorted(ENGINE_MULTIGPU_REQUIRED_FILES),
        **actual,
    )
    write(root / "ENGINE_MULTIGPU_REPAIR.json", receipt)
    return root, actual, receipt


def bind_history(root, receipt, name, value):
    relative = PREFIX + name
    write(root / relative, value)
    receipt["historical_artifact_hashes"][relative] = file_hash(root / relative)
    write(root / "ENGINE_MULTIGPU_REPAIR.json", receipt)


def bind_preserved(root, receipt, relative, value):
    path = root / (PREFIX + "evidence/" + relative)
    write(path, value)
    preservation = read(root / (PREFIX + "PRESERVATION.json"))
    preservation["artifact_hashes"][relative] = dict(
        sha256=file_hash(path), bytes=path.stat().st_size
    )
    preservation["bytes"] = sum(e["bytes"] for e in preservation["artifact_hashes"].values())
    bind_history(root, receipt, "PRESERVATION.json", preservation)


def test_multigpu_gate_keeps_all_prior_receipts_and_registration_byte_identical(multigpu_case):
    root, actual, _ = multigpu_case
    originals = {
        name: file_hash(root / name)
        for name in [
            "EXECUTION_FREEZE.json",
            "QOS_SCOPE_REPAIR.json",
            "ENGINE_MEMORY_REPAIR.json",
            "ENGINE_STORAGE_REPAIR.json",
            "ENGINE_IO_REPAIR.json",
            "orchestration/REGISTRATION.json",
            *MARKERS,
        ]
    }
    multi = verify_engine_multigpu_repair(root, actual_source=actual)
    io = verify_engine_io_repair(root, actual_source=actual)
    storage = verify_engine_storage_repair(root, actual_source=actual)
    memory = verify_engine_memory_repair(root, actual_source=actual)
    qos = verify_qos_scope_repair(root, actual_source=actual)
    assert multi["gpu_count"] == 4
    assert multi["compute_device_index"] == 0
    assert multi["storage_device_indices"] == [1, 2, 3]
    assert multi["minimum_gpu_host_memory_gb"] == 320
    assert multi["resource_operations"] == ["engine", "train"]
    assert io["repair_id"] == ENGINE_IO_REPAIR_ID
    assert io["repair_sha256"] == originals["ENGINE_IO_REPAIR.json"]
    assert io["minimum_gpu_host_memory_gb"] == 80
    assert io["failed_job_id"] == "196156"
    assert io["engine_multigpu_repair_sha256"] == multi["repair_sha256"]
    assert storage["repair_sha256"] == originals["ENGINE_STORAGE_REPAIR.json"]
    assert memory["repair_sha256"] == originals["ENGINE_MEMORY_REPAIR.json"]
    assert qos["repair_sha256"] == originals["QOS_SCOPE_REPAIR.json"]
    for result in (multi, io, storage, memory, qos):
        assert result["source_commit"] == actual["source_commit"]
        assert result["original_execution_freeze_sha256"] == originals["EXECUTION_FREEZE.json"]
    assert {name: file_hash(root / name) for name in originals} == originals


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_authenticated_gpu_count_defines_exact_indices_and_aggregate_memory(multigpu_case, count):
    root, actual, receipt = multigpu_case
    receipt.update(
        gpu_count=count,
        storage_device_indices=list(range(1, count)),
        minimum_gpu_host_memory_gb=80 * count,
    )
    write(root / "ENGINE_MULTIGPU_REPAIR.json", receipt)
    result = verify_engine_multigpu_repair(root, actual_source=actual)
    assert result["gpu_count"] == count
    assert result["minimum_gpu_host_memory_gb"] == 80 * count


def test_registered_full_restart_may_advance_actual_state_and_checkpoints(multigpu_case):
    root, actual, _ = multigpu_case
    before = verify_engine_multigpu_repair(root, actual_source=actual)
    write(root / "orchestration/STATE.json", {"tasks": {"ENGINE": {"status": "COMPLETE"}}})
    write(root / (TRACK + "/checkpoints/LATEST.json"), {"step": 4})
    write(root / (TRACK + "/PROCESS-next.json"), {"pid": 1234, "initial_step": 0})
    assert verify_engine_multigpu_repair(root, actual_source=actual) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_count", 1),
        ("gpu_count", 6),
        ("gpu_count", True),
        ("gpu_count", 4.0),
        ("compute_device_index", 1),
        ("storage_device_indices", [0, 1, 2]),
        ("storage_device_indices", [1, 2]),
        ("gpu_activation_budget_bytes", 90 << 30),
        ("cpu_activation_budget_bytes", 96 << 30),
        ("activation_storage", "lossy"),
        ("cpus_per_gpu", 1),
        ("minimum_gpu_host_memory_gb_per_gpu", 33),
        ("minimum_gpu_host_memory_gb", 80),
        ("gpu_worker_constraint", ""),
        ("qos", "override-limits-but-killable"),
        ("resource_operations", ["engine", "train", "evaluate"]),
        ("previous_engine_io_repair_sha256", "0" * 64),
        ("previous_source_commit", "0" * 40),
        ("registration_sha256", "0" * 64),
        ("previous_worker_source_sha256", "0" * 64),
        ("worker_source_sha256", "0" * 64),
        ("scientific_protocol_unchanged", False),
        ("original_once_consumed_preserved", False),
        ("engine_restart_mode", "STITCH_PARTIAL_GRADIENTS"),
        ("maintained_attempt_id", "ENGINE_attempt0002"),
        ("maintained_job_id", "196156"),
        ("quota_root", "/tmp"),
        ("activation_spill_directory", "/tmp/unsafe"),
        ("authorized_user_message", ""),
        ("authorized_at", ""),
    ],
)
def test_resource_source_and_restart_scope_are_bound_to_receipt(multigpu_case, field, value):
    root, actual, receipt = multigpu_case
    receipt[field] = value
    write(root / "ENGINE_MULTIGPU_REPAIR.json", receipt)
    with pytest.raises(PermissionError):
        verify_engine_multigpu_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    ("filename", "field", "value"),
    [
        ("TERMINAL_JOBS.json", "controller_terminal", False),
        ("TERMINAL_JOBS.json", "gpu_terminal", False),
        ("TERMINAL_JOBS.json", "controller_job_id", "196155"),
        ("TERMINAL_JOBS.json", "gpu_job_id", "196156"),
        ("TERMINAL_JOBS.json", "gpu_terminal_state", "RUNNING"),
        ("TERMINAL_JOBS.json", "controller_terminal_state", "RUNNING"),
        ("CPU_STATE_REVIEW.json", "physical_optimizer_updates", 1),
        ("CPU_STATE_REVIEW.json", "committed_logical_step", 1),
        ("CPU_STATE_REVIEW.json", "optimizer_empty", False),
        ("CPU_STATE_REVIEW.json", "raw_rollouts", 127),
        ("CPU_STATE_REVIEW.json", "learning_rate", 1e-5),
        ("CPU_STATE_REVIEW.json", "raw_record_hashes_verified", False),
        ("CPU_STATE_REVIEW.json", "checkpoint_state_hash", "0" * 64),
        ("CPU_STATE_REVIEW.json", "checkpoint_file_sha256", "0" * 64),
        ("CPU_STATE_REVIEW.json", "raw_file_hashes", {}),
        ("CPU_STATE_REVIEW.json", "source_commit", "0" * 40),
        ("RECOVERY_REVIEW.json", "science_attempts", 1),
        ("RECOVERY_REVIEW.json", "test_sealed", False),
        ("RECOVERY_REVIEW.json", "once_activation_consumed", False),
        ("RECOVERY_REVIEW.json", "once_process_consumed", False),
        ("RECOVERY_REVIEW.json", "journal_integrity", False),
        ("RECOVERY_REVIEW.json", "full_engine_restart", False),
        ("RECOVERY_REVIEW.json", "no_partial_gradient_reuse", False),
        ("RECOVERY_REVIEW.json", "no_old_once_reuse", False),
        ("PRESERVATION.json", "files", 0),
        ("PRESERVATION.json", "bytes", 0),
    ],
)
def test_rehashed_history_must_still_prove_maintenance_boundary(
    multigpu_case, filename, field, value
):
    root, actual, receipt = multigpu_case
    history = read(root / (PREFIX + filename))
    history[field] = value
    bind_history(root, receipt, filename, history)
    with pytest.raises(PermissionError):
        verify_engine_multigpu_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    "relative",
    [
        PREFIX + "CPU_STATE_REVIEW.json",
        PREFIX + "evidence/" + TRACK + "/rollouts/01-00-0.json",
        PREFIX + "evidence/" + TRACK + "/checkpoints/step-00.pt",
        PREFIX + "evidence/" + MARKERS[0],
        MARKERS[0],
        MARKERS[1],
        "code_before_engine_multigpu_20261010/src/sr_f1/runtime.py",
        "code_before_engine_io_20261010/src/sr_f1/runtime.py",
        "orchestration/REGISTRATION.json",
    ],
)
def test_tampered_history_registration_or_source_is_rejected(multigpu_case, relative):
    root, actual, _ = multigpu_case
    if relative.endswith("REGISTRATION.json"):
        write(root / relative, {"tampered": True})
    else:
        (root / relative).write_text("tampered")
    with pytest.raises(PermissionError):
        verify_qos_scope_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    "change", ["science", "missing_required", "dirty", "tree", "commit", "delete"]
)
def test_technical_source_allowlist_is_exact_and_non_destructive(multigpu_case, change):
    root, actual, receipt = multigpu_case
    actual = copy.deepcopy(actual)
    if change == "science":
        actual["source_file_hashes"]["src/sr_f1/training.py"] = "9" * 64
    elif change == "missing_required":
        previous = read(root / "code_before_engine_multigpu_20261010/SOURCE_DEPLOYMENT.json")
        actual["source_file_hashes"][WORKER] = previous["source_file_hashes"][WORKER]
    elif change == "dirty":
        actual["source_dirty_files"] = ["src/sr_f1/runtime.py"]
    elif change == "commit":
        actual["source_commit"] = "invalid"
    elif change == "delete":
        actual["source_file_hashes"].pop(WORKER)
    actual["source_tree_sha256"] = object_hash(actual["source_file_hashes"])
    if change == "tree":
        actual["source_tree_sha256"] = "0" * 64
    receipt.update(actual)
    write(root / "ENGINE_MULTIGPU_REPAIR.json", receipt)
    with pytest.raises(PermissionError):
        verify_engine_multigpu_repair(root, actual_source=actual)


def test_only_explicit_optional_validation_script_may_be_added(multigpu_case):
    root, actual, receipt = multigpu_case
    name = "scripts/sr_f1/validate_multigpu.py"
    actual["source_file_hashes"][name] = "8" * 64
    actual["source_tree_sha256"] = object_hash(actual["source_file_hashes"])
    receipt.update(actual)
    receipt["changed_files"] = sorted([*ENGINE_MULTIGPU_REQUIRED_FILES, name])
    write(root / "ENGINE_MULTIGPU_REPAIR.json", receipt)
    assert (
        verify_engine_multigpu_repair(root, actual_source=actual)["source_commit"]
        == actual["source_commit"]
    )


def test_symlinked_snapshot_alias_is_rejected_before_resolution(multigpu_case):
    root, actual, receipt = multigpu_case
    declared = root / "code_before_engine_multigpu_20261010"
    genuine = root / "genuine_prior_source"
    declared.rename(genuine)
    declared.symlink_to(genuine.name, target_is_directory=True)
    receipt["preserved_source_relative_path"] = genuine.name
    write(root / "ENGINE_MULTIGPU_REPAIR.json", receipt)
    with pytest.raises(PermissionError, match="snapshot must be a real directory"):
        verify_engine_multigpu_repair(root, actual_source=actual)


@pytest.mark.parametrize("change", ["science", "test", "attempt", "absent"])
def test_preserved_state_establishes_exact_maintenance_job_before_science(multigpu_case, change):
    root, actual, receipt = multigpu_case
    state = read(root / (PREFIX + "evidence/orchestration/STATE.json"))
    if change == "science":
        state["tasks"]["SCIENCE"]["attempts"] = [{"job_id": "99"}]
    elif change == "test":
        state["test_sealed"] = False
    elif change == "attempt":
        state["tasks"]["ENGINE"]["attempts"][0]["job_id"] = "196156"
    elif change == "absent":
        state["tasks"] = {}
    bind_preserved(root, receipt, "orchestration/STATE.json", state)
    with pytest.raises(PermissionError):
        verify_engine_multigpu_repair(root, actual_source=actual)


@pytest.mark.parametrize("name", HISTORY_NAMES)
def test_all_independent_cpu_and_terminal_reviews_must_be_bound(multigpu_case, name):
    root, actual, receipt = multigpu_case
    receipt["historical_artifact_hashes"].pop(PREFIX + name)
    write(root / "ENGINE_MULTIGPU_REPAIR.json", receipt)
    with pytest.raises(PermissionError, match="missing or unbounded"):
        verify_engine_multigpu_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    "relative",
    [
        "orchestration/STATE.json",
        "orchestration/REGISTRATION.json",
        TRACK + "/RUN_MANIFEST.json",
        TRACK + "/checkpoints/LATEST.json",
        TRACK + "/checkpoints/commit-00.json",
        TRACK + "/checkpoints/step-00.pt",
        TRACK + "/rollouts/01-15-7.json",
        *MARKERS,
    ],
)
def test_preservation_may_not_omit_required_state_or_consumed_marker(multigpu_case, relative):
    root, actual, receipt = multigpu_case
    preservation = read(root / (PREFIX + "PRESERVATION.json"))
    removed = preservation["artifact_hashes"].pop(relative)
    preservation["files"] -= 1
    preservation["bytes"] -= removed["bytes"]
    bind_history(root, receipt, "PRESERVATION.json", preservation)
    with pytest.raises(PermissionError):
        verify_engine_multigpu_repair(root, actual_source=actual)
