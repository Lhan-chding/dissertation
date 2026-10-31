"""Synthetic wrapper protocols, never GPU results or scientific acceptance."""

import json
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from sr_f6_memory_repair import execution as legacy
from sr_f6_round_b import execution as e


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def protocol(tmp_path, monkeypatch):
    from sr_f6 import pilot, pilot_jobs, runtime
    from sr_f6_memory_repair import offload

    monkeypatch.setattr(legacy, "PROJECT", tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(e, "ROOT", root)
    path = tmp_path / "ops/CONTRACT.json"
    tasks = [
        dict(
            task_id=t,
            kind="RECOVERY" if t.startswith("RECOVERY") else "PRIMARY",
            parent=t.split("-")[1],
            arm=t.split("-")[2],
        )
        for t in e.TASKS
    ]
    plan = dict(tasks=tasks, queues={"B": e.QUEUES})
    envelope = dict(execution_source={"revision": "synthetic"}, a_delivery_ref={"sha256": "audit"})
    contract = dict(
        experiment_root=str(root),
        plan_ref={"path": "plan", "sha256": "0" * 64},
        repair_source={"revision": "v7"},
        pins={"phase": "PILOT_V3"},
        max_device_bytes=44 * 2**30,
        max_host_bytes=8 * 2**30,
        max_disk_bytes=96 * 2**30,
        storage_backend="arena_v3",
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("SLURM_JOB_NAME", "LH-SRF6-PilotV3-B")
    monkeypatch.setattr(
        e, "verify_contract", lambda *_: (envelope, contract, {"sha256": "validation"})
    )
    monkeypatch.setattr(e, "_quiescent", lambda: None)
    monkeypatch.setattr(e, "_cuda_observation", lambda: {"observation_only": True})
    monkeypatch.setattr(pilot_jobs, "verify_plan", lambda *_: (plan, {}))
    monkeypatch.setattr(pilot_jobs, "_worker_environment", lambda *_: None)
    monkeypatch.setattr(pilot_jobs, "_dependency_check", lambda *_: None)
    monkeypatch.setattr(
        pilot_jobs, "_contract", lambda *_: SimpleNamespace(recovery_directory="RECOVERY")
    )
    locks = []
    monkeypatch.setattr(pilot_jobs, "_lock", lambda p, **kw: locks.append((p, kw)) or nullcontext())
    directories = {t["task_id"]: root / t["task_id"] for t in tasks}
    for directory in directories.values():
        directory.mkdir()
        save(directory / "RUN_IDENTITY.json", {})
        save(directory / "ROLLOUT_REGISTRATION.json", {})
    monkeypatch.setattr(pilot_jobs, "_registered", lambda r, a, t: (directories[t["task_id"]], {}))
    monkeypatch.setattr(runtime, "_LOADED_RUNTIMES", {})
    completed, verified = set(), []

    def verify(r, p, a, task, receipt=None):
        if receipt:
            verified.append(receipt)
            completed.add(task["task_id"])
            return receipt
        return {"complete": True} if task["task_id"] in completed else None

    monkeypatch.setattr(pilot_jobs, "_verify_task", verify)
    calls, providers, budgets, events = [], [], [], []

    @contextmanager
    def storage(callback, **kwargs):
        providers.append(callback)
        budgets.append(kwargs)
        events.append("storage-enter")
        yield SimpleNamespace(summary=lambda: {"synthetic": True})
        events.append("storage-exit")

    monkeypatch.setattr(offload, "saved_activation_offload", storage)
    monkeypatch.setattr(e, "_storage_summary", lambda *a, **k: {"synthetic": True})

    def train(r, parent, arm, execution_id, **pins):
        events.append("pilot.run")
        calls.append((parent, arm, execution_id, pins))

        class Runtime:
            pass

        live = Runtime()
        live.identity, live.model = {"task_id": execution_id}, object()
        runtime._LOADED_RUNTIMES[live] = {"model": live.model}
        assert providers[-1]() is live.model
        directory = (
            root / "RECOVERY"
            if pins["recovery_binding_ref"]
            else directories["PRIMARY-" + parent + "-" + arm]
        )
        save(directory / "TRAINER_ATTEMPTS" / execution_id / "COMPLETE.json", {"synthetic": True})

    monkeypatch.setattr(pilot, "run", train)
    return SimpleNamespace(**locals())


def execute(f, task=None):
    return e.run_task(f.path, "6" * 64, task or e.QUEUES[0][0], "B-123")


def test_primary_uses_original_pins_and_actual_storage_context(protocol):
    f = protocol
    receipt = execute(f)
    assert f.calls == [
        (
            "b2",
            "U10",
            "LH-SRF6-B-0-B-123-PRIMARY-b2-U10",
            {"phase": "PILOT_V3", "recovery_binding_ref": None},
        )
    ]
    assert f.events == ["storage-enter", "pilot.run", "storage-exit"]
    assert f.verified == [receipt]
    assert f.budgets[0]["max_device_bytes"] == 44 * 2**30
    assert f.budgets[0]["max_host_bytes"] == 8 * 2**30
    assert f.budgets[0]["max_disk_bytes"] == 96 * 2**30
    assert f.budgets[0]["disk_backend"] == "arena_v3"
    assert any(str(p).endswith(".prerequisites.lock") and kw == {"wait": True} for p, kw in f.locks)
    result = json.loads(
        (f.path.parent / "EXECUTIONS" / e.QUEUES[0][0] / "COMPLETE.json").read_text()
    )
    assert result["gate_status"] == "NOT_SIGNED" and result["full_trainer_numerical_pass"] is False


@pytest.mark.parametrize(
    "artifact", ["checkpoints", "TRAINER_ATTEMPTS", "EXECUTIONS", "groups", "audit.jsonl"]
)
def test_primary_never_adopts_partial_output(protocol, artifact):
    (protocol.directories[e.QUEUES[0][0]] / artifact).mkdir(parents=True)
    with pytest.raises(ValueError, match="Prior primary"):
        execute(protocol)
    assert not protocol.calls


def test_existing_recovery_rejected_before_prepare_or_training(protocol):
    (protocol.root / "RECOVERY").mkdir()
    with pytest.raises(ValueError, match="Prior recovery"):
        execute(protocol, e.QUEUES[1][0])
    assert not protocol.calls


def test_recovery_calls_original_finish_after_training_and_before_publication(
    protocol, monkeypatch
):
    f = protocol
    binding = {"path": "RECOVERY/RECOVERY_BINDING.json", "sha256": "4" * 64}
    monkeypatch.setattr(e, "prepare_recovery", lambda *a: (f.root / "RECOVERY", binding))

    def finish(root, admitted, task, trained_ref, binding_ref, attempt):
        f.events.append("exact-comparison")
        assert binding_ref == binding
        assert not (root / e.JOBS / "TASKS" / task["task_id"] / "COMPLETE.json").exists()
        return {"path": "original-recovery-result", "sha256": "c" * 64}

    monkeypatch.setattr(f.pilot_jobs, "_finish_recovery", finish)
    result = execute(f, e.QUEUES[1][0])
    assert f.calls[0][0:2] == ("b0", "U11")
    assert f.calls[0][3]["recovery_binding_ref"] == binding
    assert f.events[-1] == "exact-comparison"
    assert result["producer"]["path"] == "original-recovery-result"


def test_comparison_failure_preserved_without_task_publication(protocol, monkeypatch):
    f = protocol
    monkeypatch.setattr(
        e, "prepare_recovery", lambda *a: (f.root / "RECOVERY", {"path": "binding"})
    )

    def fail(*a):
        raise ValueError("exact recovery failed")

    monkeypatch.setattr(f.pilot_jobs, "_finish_recovery", fail)
    with pytest.raises(ValueError, match="exact recovery"):
        execute(f, e.QUEUES[1][0])
    folder = f.path.parent / "EXECUTIONS" / e.QUEUES[1][0]
    failure = json.loads((folder / "FAILURE.json").read_text())
    assert failure["peer_cancellation_requested"] is False
    assert not (folder / "COMPLETE.json").exists() and not f.verified
    f.runtime._LOADED_RUNTIMES.clear()
    with pytest.raises(ValueError, match="single-attempt"):
        execute(f, e.QUEUES[1][0])


def test_original_thread_count_required_before_admission(protocol, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    monkeypatch.setattr(
        e, "verify_contract", lambda *a: pytest.fail("must reject before tensor admission")
    )
    with pytest.raises(ValueError, match="single CPU thread"):
        execute(protocol)


def test_recovery_preparation_uses_step004_and_original_execution_binding(protocol, monkeypatch):
    from sr_f6 import recovery, transactions

    f = protocol
    marker = [{"step": i, "commit_sha256": str(i) * 64} for i in range(13)]

    class Tx:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def published(self):
            return marker

    monkeypatch.setattr(transactions, "RunTransactions", Tx)
    monkeypatch.setattr(f.pilot_jobs, "_verify_task", lambda *a: {"producer": {"path": "baseline"}})
    monkeypatch.setattr(
        recovery, "_bound", lambda *a: {"execution": {"path": "original-execution"}}
    )
    calls = []
    monkeypatch.setattr(
        recovery,
        "prepare_recovery",
        lambda tx, **kw: calls.append(kw) or {"binding_ref": {"path": "new-binding"}},
    )
    f.plan["tasks"].append({"task_id": "PRIMARY-b0-U11"})
    f.contract["pins"]["pilot_spec_sha256"] = "spec"
    target, binding = e.prepare_recovery(
        f.root,
        f.contract,
        f.plan,
        {"phase_context": {"original": True}},
        f.tasks[3],
        f.directories[f.tasks[3]["task_id"]],
        {},
    )
    assert target == f.root / "RECOVERY" and binding == {"path": "new-binding"}
    assert calls[0]["source_commit_sha256"] == marker[4]["commit_sha256"]
    assert calls[0]["primary_execution_ref"] == {"path": "original-execution"}
    target.mkdir()
    with pytest.raises(ValueError, match="Prior recovery"):
        e.prepare_recovery(f.root, f.contract, f.plan, {}, f.tasks[3], target, {})
    assert len(calls) == 1
