"""Baseline parallelism authenticates the successful ENGINE boundary additively."""

import copy
import importlib.util
from pathlib import Path

import pytest

from mm_core.execution import object_hash
from mm_dev.orchestration import digest, encoded
from sr_f1.contract import PLAN_ID, file_hash
from sr_f1.freeze import (
    BASELINE_PARALLEL_REQUIRED_FILES,
    _verify_execution_source,
    _verify_original_engine_multigpu_repair,
    build_baseline_parallel_repair,
    verify_baseline_parallel_repair,
    verify_engine_multigpu_repair,
    verify_qos_scope_repair,
)


def fixtures_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXTURES = fixtures_module("test_engine_multigpu_freeze")
ENGINE_FIXTURES = fixtures_module("test_freeze")
memory_case = FIXTURES.memory_case
storage_case = FIXTURES.storage_case
io_case = FIXTURES.io_case
multigpu_case = FIXTURES.multigpu_case
write, read = FIXTURES.write, FIXTURES.read
PREFIX = "technical_incidents/baseline_parallel_20261010/activation/"
WORKER = "scripts/sr_f1/run_worker.py"


def bind_history(root, receipt, name):
    relative = PREFIX + name
    receipt["historical_artifact_hashes"][relative] = file_hash(root / relative)
    write(root / "BASELINE_PARALLEL_REPAIR.json", receipt)


def write_state_journal(root, state):
    registration = read(root / "orchestration/REGISTRATION.json")
    row = dict(
        sequence=1,
        previous=None,
        event="VERIFIED_ENGINE_BOUNDARY",
        registration_hash=digest(registration),
        state=state,
    )
    row["sha256"] = digest(row)
    (root / "orchestration/journal.jsonl").write_bytes(encoded(row))
    write(root / "orchestration/STATE.json", state)


def bind_state(root, receipt, state):
    write_state_journal(root, state)
    evidence = root / PREFIX / "evidence"
    for name in ("STATE.json", "journal.jsonl"):
        (evidence / "orchestration" / name).write_bytes(
            (root / "orchestration" / name).read_bytes()
        )
        bind_history(root, receipt, "evidence/orchestration/" + name)


@pytest.fixture
def baseline_case(multigpu_case):
    root, old_actual, multi_receipt = multigpu_case
    saved = root / "code_before_baseline_parallel_20261010"
    hashes = {}
    for name in old_actual["source_file_hashes"]:
        content = (root / "code_before_engine_multigpu_20261010" / name).read_bytes()
        if name in FIXTURES.ENGINE_MULTIGPU_REQUIRED_FILES:
            content += b" actual preserved multigpu implementation"
        target = saved / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = file_hash(target)
    before = dict(
        source_commit=old_actual["source_commit"],
        source_tree_sha256=object_hash(hashes),
        source_file_hashes=hashes,
        source_dirty_files=[],
    )
    write(saved / "SOURCE_DEPLOYMENT.json", before)
    multi_receipt.update(before)
    multi_receipt["worker_source_sha256"] = hashes[WORKER]
    registration = read(root / "orchestration/REGISTRATION.json")
    registration["tasks"].update(
        COMMON_START={"gpus": 1, "operation": "common_start"},
        BASELINE={"gpus": 1, "operation": "evaluate", "stage": "baseline"},
        SCIENCE={"gpus": 1, "operation": "train"},
    )
    write(root / "orchestration/REGISTRATION.json", registration)
    multi_receipt["registration_sha256"] = file_hash(root / "orchestration/REGISTRATION.json")
    FIXTURES.bind_preserved(root, multi_receipt, "orchestration/REGISTRATION.json", registration)
    original_freeze = (root / "EXECUTION_FREEZE.json").read_bytes()
    engine = ENGINE_FIXTURES.engine_fixture(root)
    (root / "EXECUTION_FREEZE.json").write_bytes(original_freeze)
    common = read(root / "COMMON_START.json")
    common["freeze_sha256"] = file_hash(root / "EXECUTION_FREEZE.json")
    write(root / "COMMON_START.json", common)
    engine.update(
        freeze_sha256=file_hash(root / "EXECUTION_FREEZE.json"),
        common_start_sha256=file_hash(root / "COMMON_START.json"),
    )
    write(root / "ENGINE_PROBABILITY_GRADIENT_RESUME.json", engine)
    acceptance = root / "ENGINE_PROBABILITY_GRADIENT_RESUME.json"
    marker = dict(
        plan_id=PLAN_ID,
        task_id="ENGINE",
        attempt_id="ENGINE_attempt0004",
        registration_hash=digest(registration),
        status="COMPLETE",
        artifacts={
            acceptance.name: {"sha256": file_hash(acceptance), "bytes": acceptance.stat().st_size}
        },
    )
    write(root / "orchestration/completions/ENGINE.json", marker)
    state = dict(
        phase="S3",
        test_sealed=True,
        tasks={
            "COMMON_START": {"status": "COMPLETE", "attempts": []},
            "ENGINE": {
                "status": "COMPLETE",
                "completion_marker_sha256": file_hash(
                    root / "orchestration/completions/ENGINE.json"
                ),
                "attempts": [
                    {
                        "attempt_id": "ENGINE_attempt0004",
                        "job_id": "196508",
                        "status": "COMPLETED",
                        "accounting": {"terminal_state": "COMPLETED", "exit_code": "0:0"},
                    }
                ],
            },
            "BASELINE": {"status": "WAITING", "attempts": []},
            "SCIENCE": {"status": "WAITING", "attempts": []},
        },
    )
    write_state_journal(root, state)
    evidence = root / PREFIX / "evidence"
    for name in ("STATE.json", "journal.jsonl", "REGISTRATION.json", "completions/ENGINE.json"):
        target = evidence / "orchestration" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / "orchestration" / name).read_bytes())
    write(
        root / PREFIX / "TERMINAL_JOBS.json",
        dict(
            controller_job_id="196501",
            controller_terminal=True,
            controller_terminal_state="CANCELLED",
            controller_queue_empty=True,
            engine_job_id="196508",
            engine_terminal=True,
            engine_terminal_state="COMPLETED",
            engine_exit_code="0:0",
            engine_queue_empty=True,
        ),
    )
    current = dict(hashes)
    for name in BASELINE_PARALLEL_REQUIRED_FILES:
        current[name] = "9" * 64
    actual = dict(
        source_commit="bd" * 20,
        source_file_hashes=current,
        source_tree_sha256=object_hash(current),
        source_dirty_files=[],
    )
    receipt = build_baseline_parallel_repair(
        root,
        actual_source=actual,
        authorized_at="2026-10-10",
        authorized_user_message="parallel baseline up to five teacher QoS GPUs",
    )
    write(root / "BASELINE_PARALLEL_REPAIR.json", receipt)
    return root, before, actual, receipt


def test_additive_receipt_keeps_historical_engine_resource_and_worker_identity(baseline_case):
    root, before, actual, receipt = baseline_case
    old = _verify_original_engine_multigpu_repair(root, actual_source=before)
    original_paths = [
        "EXECUTION_FREEZE.json",
        "ENGINE_MULTIGPU_REPAIR.json",
        "ENGINE_IO_REPAIR.json",
        "ENGINE_STORAGE_REPAIR.json",
        "ENGINE_MEMORY_REPAIR.json",
        "QOS_SCOPE_REPAIR.json",
        "orchestration/REGISTRATION.json",
        "orchestration/completions/ENGINE.json",
        "ENGINE_PROBABILITY_GRADIENT_RESUME.json",
    ]
    originals = {name: file_hash(root / name) for name in original_paths}
    latest = verify_baseline_parallel_repair(root, actual_source=actual)
    assert latest["previous_worker_source_sha256"] == before["source_file_hashes"][WORKER]
    assert latest["worker_source_sha256"] == actual["source_file_hashes"][WORKER]
    assert latest["source_commit"] == actual["source_commit"]
    assert latest["engine_multigpu_repair"] == old
    assert verify_engine_multigpu_repair(root, actual_source=actual) == old
    assert (
        verify_qos_scope_repair(root, actual_source=actual)["source_commit"]
        == before["source_commit"]
    )
    assert _verify_execution_source(root, {}, actual_source=actual) == latest
    assert latest["max_gpu_count"] == 5
    assert latest["baseline_slot_count"] == 6784
    assert {name: file_hash(root / name) for name in original_paths} == originals
    assert receipt["repair_id"] == latest["repair_id"]


def test_source_revision_does_not_allow_early_activation(baseline_case):
    root, _, actual, receipt = baseline_case
    state = read(root / "orchestration/STATE.json")
    state["tasks"]["ENGINE"]["status"] = "RUNNING"
    bind_state(root, receipt, state)
    with pytest.raises(PermissionError, match="successful ENGINE"):
        verify_baseline_parallel_repair(root, actual_source=actual)


@pytest.mark.parametrize("task", ["BASELINE", "SCIENCE"])
def test_any_pre_activation_downstream_attempt_is_rejected(baseline_case, task):
    root, _, actual, receipt = baseline_case
    state = read(root / "orchestration/STATE.json")
    state["tasks"][task]["attempts"] = [{"status": "UNKNOWN"}]
    bind_state(root, receipt, state)
    with pytest.raises(PermissionError, match="downstream attempt"):
        verify_baseline_parallel_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    "field,value",
    [
        ("qos", "other"),
        ("max_gpu_count", 6),
        ("baseline_slot_count", 512),
        ("resource_operations", ["baseline", "train"]),
        ("partition_mode", "ADAPTIVE"),
        ("scientific_protocol_unchanged", False),
        ("previous_worker_source_sha256", "0" * 64),
        ("worker_source_sha256", "0" * 64),
        ("registration_sha256", "0" * 64),
        ("previous_engine_multigpu_repair_sha256", "0" * 64),
        ("original_execution_freeze_sha256", "0" * 64),
        ("authorized_at", ""),
        ("authorized_user_message", ""),
    ],
)
def test_parallel_scope_and_provenance_are_fixed(baseline_case, field, value):
    root, _, actual, receipt = baseline_case
    receipt[field] = value
    write(root / "BASELINE_PARALLEL_REPAIR.json", receipt)
    with pytest.raises(PermissionError):
        verify_baseline_parallel_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    "relative",
    [
        "EXECUTION_FREEZE.json",
        "ENGINE_MULTIGPU_REPAIR.json",
        "ENGINE_IO_REPAIR.json",
        "orchestration/REGISTRATION.json",
        "orchestration/completions/ENGINE.json",
        "ENGINE_PROBABILITY_GRADIENT_RESUME.json",
        "engineering/engine/natural/raw.json",
        "code_before_baseline_parallel_20261010/src/sr_f1/runtime.py",
        PREFIX + "evidence/orchestration/journal.jsonl",
        PREFIX + "evidence/orchestration/STATE.json",
        PREFIX + "TERMINAL_JOBS.json",
    ],
)
def test_original_science_source_or_handoff_tampering_fails(baseline_case, relative):
    root, _, actual, _ = baseline_case
    write(root / relative, {"tampered": True})
    with pytest.raises((PermissionError, KeyError)):
        verify_baseline_parallel_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    "change", ["runtime", "training", "evaluation", "delete", "required", "dirty", "tree", "commit"]
)
def test_source_allowlist_forbids_scientific_and_numerical_changes(baseline_case, change):
    root, before, actual, receipt = baseline_case
    actual = copy.deepcopy(actual)
    files = actual["source_file_hashes"]
    if change in {"runtime", "training", "evaluation"}:
        files[f"src/sr_f1/{change}.py"] = "0" * 64
    elif change == "delete":
        files.pop("src/sr_f1/runtime.py")
    elif change == "required":
        files[WORKER] = before["source_file_hashes"][WORKER]
    elif change == "dirty":
        actual["source_dirty_files"] = [WORKER]
    elif change == "commit":
        actual["source_commit"] = "missing"
    actual["source_tree_sha256"] = object_hash(files)
    if change == "tree":
        actual["source_tree_sha256"] = "0" * 64
    receipt.update(actual)
    receipt["worker_source_sha256"] = files[WORKER]
    receipt["changed_files"] = sorted(
        name
        for name in before["source_file_hashes"].keys() | files.keys()
        if before["source_file_hashes"].get(name) != files.get(name)
    )
    write(root / "BASELINE_PARALLEL_REPAIR.json", receipt)
    with pytest.raises(PermissionError):
        verify_baseline_parallel_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    "change", ["nonterminal", "exit", "attempt", "seal", "marker", "tasks", "journal", "prefix"]
)
def test_rehashed_handoff_cannot_substitute_successful_registered_engine(baseline_case, change):
    root, _, actual, receipt = baseline_case
    state = read(root / "orchestration/STATE.json")
    attempt = state["tasks"]["ENGINE"]["attempts"][-1]
    if change == "nonterminal":
        attempt["status"] = "RUNNING"
    elif change == "exit":
        attempt["accounting"]["exit_code"] = "1:0"
    elif change == "attempt":
        attempt["attempt_id"] = "ENGINE_attempt0002"
    elif change == "seal":
        state["test_sealed"] = False
    elif change == "marker":
        state["tasks"]["ENGINE"]["completion_marker_sha256"] = "0" * 64
    elif change == "tasks":
        state["tasks"].pop("BASELINE")
    bind_state(root, receipt, state)
    if change == "journal":
        path = root / PREFIX / "evidence/orchestration/journal.jsonl"
        row = read(path)
        row["sequence"] = 2
        row.pop("sha256")
        row["sha256"] = digest(row)
        path.write_bytes(encoded(row))
        (root / "orchestration/journal.jsonl").write_bytes(encoded(row))
        bind_history(root, receipt, "evidence/orchestration/journal.jsonl")
    elif change == "prefix":
        (root / "orchestration/journal.jsonl").write_text("changed current journal\n")
    with pytest.raises(PermissionError):
        verify_baseline_parallel_repair(root, actual_source=actual)


@pytest.mark.parametrize(
    "field,value",
    [
        ("engine_job_id", "123"),
        ("engine_terminal_state", "RUNNING"),
        ("engine_exit_code", "1:0"),
        ("engine_queue_empty", False),
        ("controller_queue_empty", False),
        ("controller_terminal", False),
        ("controller_terminal_state", "RUNNING"),
        ("controller_job_id", "196508"),
    ],
)
def test_actual_scheduler_terminal_boundary_must_be_preserved(baseline_case, field, value):
    root, _, actual, receipt = baseline_case
    path = root / PREFIX / "TERMINAL_JOBS.json"
    terminal = read(path)
    terminal[field] = value
    write(path, terminal)
    bind_history(root, receipt, "TERMINAL_JOBS.json")
    with pytest.raises(PermissionError):
        verify_baseline_parallel_repair(root, actual_source=actual)


def test_later_baseline_and_science_progress_does_not_change_historical_gate(baseline_case):
    root, _, actual, _ = baseline_case
    before = verify_baseline_parallel_repair(root, actual_source=actual)
    journal = root / "orchestration/journal.jsonl"
    previous = read(journal)
    state = copy.deepcopy(previous["state"])
    state["tasks"]["BASELINE"].update(status="COMPLETE", attempts=[{"job_id": "new"}])
    state["tasks"]["SCIENCE"].update(status="ACTIVE", attempts=[{"job_id": "science"}])
    row = dict(previous, sequence=2, previous=previous["sha256"], state=state)
    row.pop("sha256")
    row["sha256"] = digest(row)
    with journal.open("ab") as stream:
        stream.write(encoded(row))
    write(root / "orchestration/STATE.json", state)
    assert verify_baseline_parallel_repair(root, actual_source=actual) == before


def test_receipt_is_required_and_cannot_be_created_from_active_source_alone(baseline_case):
    root, _, actual, _ = baseline_case
    (root / "BASELINE_PARALLEL_REPAIR.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_baseline_parallel_repair(root, actual_source=actual)
    with pytest.raises(PermissionError):
        verify_engine_multigpu_repair(root, actual_source=actual)


def test_symlinked_source_snapshot_is_rejected(baseline_case):
    root, _, actual, _ = baseline_case
    declared = root / "code_before_baseline_parallel_20261010"
    genuine = root / "genuine_baseline_source"
    declared.rename(genuine)
    declared.symlink_to(genuine.name, target_is_directory=True)
    with pytest.raises(PermissionError, match="real directory"):
        verify_baseline_parallel_repair(root, actual_source=actual)
