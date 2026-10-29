"""Fail-closed original provenance and single-attempt memory executor tests."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sr_f6_memory_repair import execution as e


def save(path, value, *, raw=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if raw else json.dumps(value).encode()
    path.write_bytes(payload)
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


@pytest.fixture
def original(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "PROJECT", tmp_path)
    root = tmp_path / "experiment"
    monkeypatch.setattr(e, "ROOT", root)
    run = root / e.RUN
    refs = {}
    refs["step0_commit_ref"] = save(run / "checkpoints/step-000/COMMIT.json", {"step": 0})
    save(run / "checkpoints/step-000/state.pt", b"checkpoint bytes", raw=True)
    groups = []
    for i in range(16):
        group = run / f"rollouts/step-001/groups/{i:02d}"
        raw = save(
            group / "raw.jsonl",
            b"\n".join(json.dumps({"task_id": e.OLD_EXECUTION}).encode() for _ in range(8)) + b"\n",
            raw=True,
        )
        groups.append(save(group / "COMPLETE.json", {"raw_reference": raw}))
    rollout = save(
        run / "rollouts/step-001/rollout.json",
        {"records": [{}] * 128, "complete_group_references": groups},
    )
    ready = {
        "status": "ROLLOUT_READY",
        "sampler_state": {"rollout_cursor": {"step": 1, "groups": 16, "rows": 128}},
        "references": {"rollout": rollout},
    }
    refs["original_ready_ref"] = save(run / "rollouts/step-001/READY.json", ready)
    refs["original_failure_ref"] = save(
        run / "TRAINER_ATTEMPTS" / e.OLD_EXECUTION / "FAILURE.json",
        {
            "status": "FAILED_PRESERVED",
            "error_type": "OutOfMemoryError",
            "error": "CUDA out of memory",
            "task_id": e.OLD_EXECUTION,
            "parent": "b1",
            "arm": "U10",
        },
    )
    refs["original_task_failure_ref"] = save(
        root / e.JOBS / "TASKS" / e.QUEUE[0] / "ATTEMPTS" / e.OLD_EXECUTION / "FAILURE.json",
        {"status": "FAILED_PRESERVED"},
    )
    refs["original_worker_failure_ref"] = save(
        root / e.JOBS / "WORKERS/A/1/ATTEMPTS/A-233996/FAILURE.json", {"status": "FAILED_PRESERVED"}
    )
    event = {
        "status": "UNCOMMITTED_FAILURE_NO_OPTIMIZER_STEP",
        "task_id": e.OLD_EXECUTION,
        "optimizer_started": False,
        "parameter_state_changed": False,
        "committed_source_step": 0,
    }
    payload = json.dumps(event).encode() + b"\n"
    audit = run / "NUMERICAL_TOKEN_AUDIT.jsonl"
    save(audit, payload, raw=True)
    refs["original_audit_ref"] = save(tmp_path / "repair/original-audit.jsonl", payload, raw=True)
    files = [
        p
        for d in (run / "checkpoints/step-000", run / "rollouts/step-001")
        for p in d.rglob("*")
        if p.is_file()
    ]
    contract = dict(
        experiment_root=str(root),
        **refs,
        original_audit_path=str(audit),
        original_audit_bytes=len(payload),
        original_files=[
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files
        ],
    )
    return contract


def test_actual_original_byte_tree_and_128_draws(original):
    e._original_evidence(original)


def test_appended_audit_preserves_original_event(original):
    with Path(original["original_audit_path"]).open("ab") as stream:
        stream.write(b'{"new": "actual later update"}\n')
    e._original_evidence(original)


@pytest.mark.parametrize("key", ["optimizer_started", "parameter_state_changed"])
def test_any_original_update_forbids_stepzero_restore(original, key):
    ref = original["original_audit_ref"]
    event = json.loads(Path(ref["path"]).read_bytes())
    event[key] = True
    payload = json.dumps(event).encode() + b"\n"
    original["original_audit_ref"] = save(Path(ref["path"]), payload, raw=True)
    original["original_audit_bytes"] = len(payload)
    Path(original["original_audit_path"]).write_bytes(payload)
    with pytest.raises(ValueError, match="optimizer state"):
        e._original_evidence(original)


def test_omitted_original_file_rejected(original):
    original["original_files"].pop()
    with pytest.raises(ValueError, match="tree closure"):
        e._original_evidence(original)


def test_modified_original_raw_rejected(original):
    raw = next(
        Path(ref["path"]) for ref in original["original_files"] if ref["path"].endswith("raw.jsonl")
    )
    raw.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="hash differs"):
        e._original_evidence(original)


def test_partial_ready_cannot_authorize_resampling(original):
    ref = original["original_ready_ref"]
    ready = json.loads(Path(ref["path"]).read_bytes())
    ready["sampler_state"]["rollout_cursor"]["groups"] = 15
    original["original_ready_ref"] = save(Path(ref["path"]), ready)
    with pytest.raises(ValueError, match="Complete original READY1"):
        e._original_evidence(original)


def test_raw_with_new_task_provenance_rejected(original):
    run = Path(original["experiment_root"]) / e.RUN
    ready = e.bound_ref(run, original["original_ready_ref"])
    rollout_ref = ready["references"]["rollout"]
    rollout = e.bound_ref(run, rollout_ref)
    group_ref = rollout["complete_group_references"][0]
    group = e.bound_ref(run, group_ref)
    group["raw_reference"] = save(
        Path(group["raw_reference"]["path"]),
        b"\n".join([b'{"task_id":"new-task"}'] * 8) + b"\n",
        raw=True,
    )
    rollout["complete_group_references"][0] = save(Path(group_ref["path"]), group)
    ready["references"]["rollout"] = save(Path(rollout_ref["path"]), rollout)
    original["original_ready_ref"] = save(Path(original["original_ready_ref"]["path"]), ready)
    with pytest.raises(ValueError, match="provenance"):
        e._original_evidence(original)


def test_symlink_and_external_refs_rejected(original, tmp_path):
    path = tmp_path / "alias"
    path.symlink_to(original["original_ready_ref"]["path"])
    with pytest.raises(ValueError, match="Symlink"):
        e.strict_ref(path, original["original_ready_ref"]["sha256"])
    with pytest.raises(ValueError, match="Outside"):
        e.plain_path("/private/etc/passwd")


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        e._json(b'{"schema":1,"schema":2}')


@pytest.mark.parametrize(
    "accounting,queue,good",
    [
        ("233996.1|FAILED|1:0|\n", "233996.0\n", True),
        ("233996.1|RUNNING|0:0|\n", "", False),
        ("233996.1|COMPLETED|0:0|\n", "", False),
        ("", "", False),
        ("233996.1|FAILED|1:0|\n", "233996.1\n", False),
    ],
)
def test_only_original_failed_step_terminal_is_quiescent(monkeypatch, accounting, queue, good):
    outputs = iter([accounting, queue])
    monkeypatch.setattr(e.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=next(outputs)))
    if good:
        e._quiescent()
    else:
        with pytest.raises(ValueError):
            e._quiescent()


def test_other_lane_rejected_before_contract_read():
    with pytest.raises(ValueError, match="Only original b1"):
        e.run_task("missing", "missing", {}, "PRIMARY-b0-U11", "repair")


def test_model_provider_matches_actual_runtime_shape():
    # Real Runtime has identity['task_id'], not a top-level task_id attribute.
    class Runtime:
        def __init__(self):
            self.identity = {"task_id": "new-execution"}
            self.model = object()

    item = Runtime()
    module = SimpleNamespace(_LOADED_RUNTIMES={item: {"model": item.model}})
    assert e._bound_model(module, "new-execution") is item.model
    with pytest.raises(ValueError, match="current immutable"):
        e._bound_model(module, "old-execution")
    module._LOADED_RUNTIMES[item]["model"] = object()
    with pytest.raises(ValueError, match="current immutable"):
        e._bound_model(module, "new-execution")


def test_storage_summary_failure_remains_deliverable():
    def broken():
        raise RuntimeError("Bound model storage changed")

    manager = SimpleNamespace(summary=broken)
    with pytest.raises(RuntimeError, match="storage changed"):
        e._storage_summary(manager, 123)
    assert e._failure_summary(manager)["status"] == "STORAGE_SUMMARY_FAILED_PRESERVED"


def test_fixture_storage_summary_cannot_publish_actual_complete():
    manager = SimpleNamespace(summary=lambda: {"scope": "SYNTHETIC_CPU_ONLY"})
    with pytest.raises(ValueError, match="did not complete"):
        e._storage_summary(manager, 123)


@pytest.fixture
def synthetic_protocol(original, monkeypatch):
    """Only protocol wiring is synthetic; it cannot publish production PASS."""
    from contextlib import contextmanager, nullcontext

    from sr_f6 import pilot, pilot_jobs, runtime
    from sr_f6_memory_repair import offload, validation

    root = Path(original["experiment_root"])
    contract_path = root.parent / "repair/CONTRACT.json"
    contract = dict(
        original,
        plan_ref={"path": "plan", "sha256": "0" * 64},
        repair_source={"revision": "1" * 40},
        pins={"phase": "PILOT_V3"},
        max_host_bytes=1024,
    )
    monkeypatch.setattr(e, "verify_contract", lambda *_: contract)
    validation_result = dict(
        status="VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH",
        contract_sha256="2" * 64,
        producer_revision=e.PRODUCER_REVISION,
        repair_revision="1" * 40,
        generated_responses=0,
        optimizer_updates=0,
        compared_gradient_rows=[0],
        offload_rows=128,
    )
    monkeypatch.setattr(validation, "verify_validation", lambda *_: validation_result)
    monkeypatch.setattr(e, "_quiescent", lambda: None)
    task = {"task_id": e.QUEUE[0], "kind": "PRIMARY", "parent": "b1", "arm": "U10"}
    monkeypatch.setattr(pilot_jobs, "verify_plan", lambda *_: ({"tasks": [task]}, {}))
    monkeypatch.setattr(pilot_jobs, "_worker_environment", lambda *_: None)
    monkeypatch.setattr(pilot_jobs, "_lock", lambda *_: nullcontext())
    monkeypatch.setattr(pilot_jobs, "_registered", lambda *_: (root / e.RUN, {}))
    verified = []

    def verify_task(*_, receipt=None):
        if receipt is not None:
            verified.append(receipt)
        return None

    monkeypatch.setattr(pilot_jobs, "_verify_task", verify_task)
    monkeypatch.setattr(runtime, "_LOADED_RUNTIMES", {})
    monkeypatch.setenv("SLURM_JOB_NAME", "LH-SRF6-Test")
    summary = dict(
        scope="CUDA_SAVED_ACTIVATION_STORAGE",
        model_bound=True,
        closed=True,
        pack_count=1,
        unpack_count=1,
        peak_host_bytes=64,
        numerical_pass=False,
    )
    provider = []

    @contextmanager
    def storage(callback, **_):
        provider.append(callback)
        yield SimpleNamespace(summary=lambda: summary)

    monkeypatch.setattr(offload, "saved_activation_offload", storage)
    calls = []

    def train(root, parent, arm, execution_id, **pins):
        calls.append((parent, arm, execution_id, pins))

        class Runtime:
            pass

        live = Runtime()
        live.identity, live.model = {"task_id": execution_id}, object()
        runtime._LOADED_RUNTIMES[live] = {"model": live.model}
        assert provider[0]() is live.model
        save(
            root / e.RUN / "TRAINER_ATTEMPTS" / execution_id / "COMPLETE.json", {"synthetic": True}
        )

    monkeypatch.setattr(pilot, "run", train)
    return SimpleNamespace(
        path=contract_path,
        checked=validation_result,
        calls=calls,
        summary=summary,
        verified=verified,
        root=root,
        producer=pilot,
    )


def execute_fixture(fixture):
    return e.run_task(
        fixture.path, "2" * 64, {"path": "proof", "sha256": "3" * 64}, e.QUEUE[0], "memory-test"
    )


def test_single_fresh_task_preserves_original_producer_api(synthetic_protocol):
    fixture = synthetic_protocol
    result = execute_fixture(fixture)
    assert fixture.calls == [
        ("b1", "U10", "LH-SRF6-A-1-memory-test-PRIMARY-b1-U10", {"phase": "PILOT_V3"})
    ]
    assert fixture.verified == [result]
    assert result["gate_status"] == "NOT_SIGNED"
    assert (fixture.path.parent / "EXECUTIONS" / e.QUEUE[0] / "COMPLETE.json").exists()
    # Separate process cannot activate the same extension again.
    from sr_f6 import runtime

    runtime._LOADED_RUNTIMES.clear()
    with pytest.raises(ValueError, match="single-attempt"):
        execute_fixture(fixture)
    assert len(fixture.calls) == 1


def test_placeholder_validation_never_loads_model(synthetic_protocol):
    fixture = synthetic_protocol
    fixture.checked["status"] = "PASS"
    with pytest.raises(ValueError, match="placeholder PASS"):
        execute_fixture(fixture)
    assert fixture.calls == []
    assert not (fixture.path.parent / "EXECUTIONS").exists()


def test_context_failure_never_publishes_task_complete(synthetic_protocol):
    fixture = synthetic_protocol
    fixture.summary["scope"] = "SYNTHETIC_CPU_ONLY"
    with pytest.raises(ValueError, match="storage context"):
        execute_fixture(fixture)
    assert fixture.verified == []
    assert not (fixture.root / e.JOBS / "TASKS" / e.QUEUE[0] / "COMPLETE.json").exists()
    assert (fixture.path.parent / "EXECUTIONS" / e.QUEUE[0] / "FAILURE.json").exists()


def test_new_training_failure_is_preserved_without_retry(synthetic_protocol, monkeypatch):
    fixture = synthetic_protocol

    def fail(*_, **__):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(fixture.producer, "run", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        execute_fixture(fixture)
    failure = json.loads(
        (fixture.path.parent / "EXECUTIONS" / e.QUEUE[0] / "FAILURE.json").read_text()
    )
    assert failure["sampling_retry_authorized"] is False
    assert failure["peer_cancellation_requested"] is False
    with pytest.raises(ValueError, match="single-attempt"):
        execute_fixture(fixture)
