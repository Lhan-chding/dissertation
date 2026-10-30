"""Recovery protocol tests use synthetic producers, never a scientific PASS."""

import ast
import inspect
import json
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from sr_f6_memory_repair import execution as legacy
from sr_f6_memory_resume import execution as e


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    from sr_f6 import pilot, pilot_jobs, runtime
    from sr_f6_memory_repair import offload

    monkeypatch.setattr(legacy, "PROJECT", tmp_path)
    root = tmp_path / "root"
    run = root / legacy.RUN
    (run / "checkpoints/step-000").mkdir(parents=True)
    old_failure = run / "TRAINER_ATTEMPTS" / legacy.OLD_EXECUTION / "FAILURE.json"
    save(old_failure, {"status": "FAILED_PRESERVED"})
    frozen_failure = old_failure.read_bytes()
    path = tmp_path / "v8/CONTRACT.json"
    tasks = [
        dict(task_id=task_id, kind="PRIMARY", parent="b1", arm=task_id.removeprefix("PRIMARY-b1-"))
        for task_id in e.QUEUE
    ]
    envelope = dict(
        full_contract_ref=dict(path="actual-v7/CONTRACT.json", sha256="7" * 64),
        full_result_ref=dict(path="actual-v7/RESULT.json", sha256="8" * 64),
        execution_source=dict(revision="9" * 40),
    )
    contract = dict(
        experiment_root=str(root),
        plan_ref=dict(path="plan", sha256="0" * 64),
        repair_source=dict(revision="1" * 40),
        pins=dict(phase="PILOT_V3"),
        max_device_bytes=44 * 2**30,
        max_host_bytes=8 * 2**30,
        max_disk_bytes=96 * 2**30,
        storage_backend="arena_v3",
    )
    checked = dict(
        status="VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH",
        contract_sha256="7" * 64,
        producer_revision=e.PRODUCER_REVISION,
        repair_revision="1" * 40,
        generated_responses=0,
        optimizer_updates=0,
        compared_gradient_rows=[0],
        offload_rows=128,
    )
    history_checks = []
    monkeypatch.setattr(e, "verify_contract", lambda *args: history_checks.append(args) or envelope)
    monkeypatch.setattr(e, "full_contract", lambda _: contract)
    monkeypatch.setattr(e, "verify_validation", lambda _: checked)
    monkeypatch.setattr(e, "_quiescent", lambda: None)
    monkeypatch.setattr(
        e, "_cuda_observation", lambda: dict(allocated_bytes=123, lifetime_peak_allocated_bytes=456)
    )
    monkeypatch.setattr(pilot_jobs, "verify_plan", lambda *_: ({"tasks": tasks}, {}))
    monkeypatch.setattr(pilot_jobs, "_worker_environment", lambda *_: None)
    locks = []
    monkeypatch.setattr(pilot_jobs, "_lock", lambda path: locks.append(path) or nullcontext())
    directories = {t["task_id"]: root / legacy.RUN.replace("/U10", "/" + t["arm"]) for t in tasks}
    monkeypatch.setattr(
        pilot_jobs, "_registered", lambda root, admitted, task: (directories[task["task_id"]], {})
    )
    verified, completed = [], set()

    def verify_task(root, plan_ref, admitted, task, receipt=None):
        if receipt is not None:
            verified.append(receipt)
            completed.add(task["task_id"])
            return receipt
        return {"complete": True} if task["task_id"] in completed else None

    monkeypatch.setattr(pilot_jobs, "_verify_task", verify_task)
    monkeypatch.setattr(runtime, "_LOADED_RUNTIMES", {})
    monkeypatch.setenv("SLURM_JOB_NAME", "LH-SRF6-Memory-Resume")
    calls, providers, budgets, summaries = [], [], [], []

    @contextmanager
    def storage(callback, **kwargs):
        kwargs["scratch_directory"].mkdir()
        providers.append(callback)
        budgets.append(kwargs)
        yield SimpleNamespace(summary=lambda: {"synthetic": True})

    monkeypatch.setattr(offload, "saved_activation_offload", storage)
    monkeypatch.setattr(
        e,
        "_storage_summary",
        lambda *args, **kwargs: summaries.append((args, kwargs)) or {"synthetic": True},
    )

    def train(root, parent, arm, execution_id, **pins):
        calls.append((parent, arm, execution_id, pins))

        class Runtime:
            pass

        live = Runtime()
        live.identity, live.model = {"task_id": execution_id}, object()
        runtime._LOADED_RUNTIMES[live] = dict(model=live.model)
        assert providers[-1]() is live.model
        save(
            directories["PRIMARY-b1-" + arm] / "TRAINER_ATTEMPTS" / execution_id / "COMPLETE.json",
            {"synthetic": True},
        )

    monkeypatch.setattr(pilot, "run", train)
    return SimpleNamespace(**locals())


def execute(f, task=None):
    return e.run_task(f.path, "6" * 64, task or e.QUEUE[0], "MEM-123")


def test_real_producer_interface_budgets_provenance_and_observations(protocol):
    f = protocol
    receipt = execute(f)
    assert f.calls == [("b1", "U10", "LH-SRF6-A-1-MEM-123-PRIMARY-b1-U10", {"phase": "PILOT_V3"})]
    assert f.verified == [receipt]
    assert f.budgets[0]["max_device_bytes"] == 44 * 2**30
    assert f.budgets[0]["max_host_bytes"] == 8 * 2**30
    assert f.budgets[0]["max_disk_bytes"] == 96 * 2**30
    assert f.budgets[0]["disk_backend"] == "arena_v3"
    assert str(f.locks[0]).endswith("WORKERS/A/1/.worker.lock")
    assert str(f.locks[1]).endswith("TASKS/PRIMARY-b1-U10/.task.lock")
    folder = f.path.parent / "EXECUTIONS" / e.QUEUE[0]
    started = json.loads((folder / "STARTED.json").read_text())
    assert started["execution_source"] == f.envelope["execution_source"]
    assert started["repair_source"] == f.contract["repair_source"]
    facts = json.loads((folder / "STORAGE_FACTS.json").read_text())
    assert facts["contract_ref"]["sha256"] == "6" * 64
    assert facts["validation_ref"] == f.envelope["full_result_ref"]
    assert facts["elapsed_seconds"] >= 0
    assert facts["cuda_before"]["allocated_bytes"] == 123
    assert facts["cuda_after"]["lifetime_peak_allocated_bytes"] == 456
    complete = json.loads((folder / "COMPLETE.json").read_text())
    assert complete["full_trainer_numerical_pass"] is False
    assert complete["gate_status"] == "NOT_SIGNED"
    assert f.old_failure.read_bytes() == f.frozen_failure
    assert not (f.root / e.JOBS / "WORKERS/A/1/COMPLETE.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "PASS"),
        ("contract_sha256", "6" * 64),
        ("offload_rows", 2),
        ("optimizer_updates", 1),
        ("generated_responses", 1),
        ("compared_gradient_rows", [0, 26]),
    ],
)
def test_no_placeholder_or_small_test_admission(protocol, field, value):
    protocol.checked[field] = value
    with pytest.raises(ValueError, match="placeholder PASS"):
        execute(protocol)
    assert not protocol.calls
    assert not (protocol.path.parent / "EXECUTIONS").exists()


def test_single_attempt_even_if_original_success_cannot_be_verified_again(protocol):
    f = protocol
    execute(f)
    f.runtime._LOADED_RUNTIMES.clear()
    f.completed.clear()
    with pytest.raises(ValueError, match="single-attempt"):
        execute(f)
    assert len(f.calls) == 1


def test_unsafe_source_step_cannot_recover(protocol):
    (protocol.run / "checkpoints/step-001").mkdir()
    with pytest.raises(ValueError, match="step zero"):
        execute(protocol)
    assert not protocol.calls


def test_original_predecessor_required_before_later_task(protocol):
    with pytest.raises(ValueError, match="predecessor incomplete"):
        execute(protocol, e.QUEUE[1])
    assert not protocol.calls


def test_later_tasks_follow_current_producer_state_without_requiring_stepzero(protocol):
    f = protocol
    for task_id in e.QUEUE:
        f.runtime._LOADED_RUNTIMES.clear()
        execute(f, task_id)
        if task_id == e.QUEUE[0]:
            (f.run / "checkpoints/step-012").mkdir()
    assert [call[1] for call in f.calls] == ["U10", "C_SHUF", "U11"]
    assert len(f.history_checks) == 3
    assert f.old_failure.read_bytes() == f.frozen_failure


@pytest.mark.parametrize("where", ["TRAINER_ATTEMPTS", "ATTEMPTS"])
def test_later_task_prior_attempt_is_never_retried(protocol, where):
    f = protocol
    f.completed.add(e.QUEUE[0])
    folder = (
        f.directories[e.QUEUE[1]]
        if where == "TRAINER_ATTEMPTS"
        else f.root / e.JOBS / "TASKS" / e.QUEUE[1]
    )
    (folder / where).mkdir(parents=True)
    with pytest.raises(ValueError, match="No retry"):
        execute(f, e.QUEUE[1])
    assert not f.calls


def test_failure_preserved_without_publishing_or_retry(protocol, monkeypatch):
    f = protocol

    def fail(*args, **kwargs):
        raise RuntimeError("actual failure placeholder")

    monkeypatch.setattr(f.pilot, "run", fail)
    with pytest.raises(RuntimeError, match="actual failure"):
        execute(f)
    folder = f.path.parent / "EXECUTIONS" / e.QUEUE[0]
    failure = json.loads((folder / "FAILURE.json").read_text())
    assert failure["sampling_retry_authorized"] is False
    assert failure["peer_cancellation_requested"] is False
    assert failure["elapsed_seconds"] >= 0
    assert not (folder / "COMPLETE.json").exists()
    assert f.verified == []
    assert f.old_failure.read_bytes() == f.frozen_failure
    with pytest.raises(ValueError, match="single-attempt"):
        execute(f)


def calls(function, name):
    tree = ast.parse(inspect.getsource(function))
    return [
        ast.dump(n, include_attributes=False)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == name
    ]


def test_original_computation_and_publication_calls_are_ast_identical():
    for name in (
        "pilot.run",
        "pilot_jobs._registered",
        "pilot_jobs._verify_task",
        "pilot_jobs._write",
        "saved_activation_offload",
        "_storage_summary",
        "_bound_model",
    ):
        assert calls(e.run_task, name) == calls(legacy.run_task, name), name
    assert e._storage_summary is legacy._storage_summary
    assert e._bound_model is legacy._bound_model
    assert e._quiescent is legacy._quiescent


def test_cuda_observer_does_not_reset_evict_or_change_computation():
    text = inspect.getsource(e._cuda_observation)
    for forbidden in ("empty_cache(", "reset_peak_memory_stats(", "synchronize(", "manual_seed("):
        assert forbidden not in text
