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
        e._storage_summary(manager, 123, 456, Path("/unused"))
    assert e._failure_summary(manager)["status"] == "STORAGE_SUMMARY_FAILED_PRESERVED"


def test_fixture_storage_summary_cannot_publish_actual_complete():
    manager = SimpleNamespace(summary=lambda: {"scope": "SYNTHETIC_CPU_ONLY"})
    with pytest.raises(ValueError, match="did not complete"):
        e._storage_summary(manager, 123, 456, Path("/unused"))


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
        max_disk_bytes=2048,
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
        schema="srf6-saved-activation-hybrid-v2",
        scope="CUDA_SAVED_ACTIVATION_HYBRID_STORAGE",
        model_bound=True,
        closed=True,
        pack_count=1,
        unpack_count=1,
        peak_host_bytes=64,
        peak_disk_bytes=128,
        max_host_bytes=1024,
        max_disk_bytes=2048,
        live_host_bytes=0,
        live_disk_bytes=0,
        staging_chunk_bytes=64 * 2**20,
        peak_staging_host_bytes=32,
        storage_failures=[],
        cleanup_errors=[],
        disk_bytes_written=128,
        disk_bytes_read=128,
        numerical_pass=False,
    )
    provider = []

    @contextmanager
    def storage(callback, **kwargs):
        scratch = kwargs["scratch_directory"]
        scratch.mkdir()
        summary["scratch_directory"] = str(scratch)
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


@pytest.fixture
def allocation_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "PROJECT", tmp_path)
    old = tmp_path / "old"
    monkeypatch.setattr(e, "OLD_OPS", old)
    reference = save(
        old / "validate_FAILURE.json",
        {
            "job_id": "234358",
            "status": "FAILED_PRESERVED",
            "error_type": "PermissionError",
            "error": "Wrong actual GPU/CPU/host-memory allocation",
            "retry_authorized": False,
        },
    )
    source = {"revision": e.OLD_REPAIR_REVISION}
    save(
        old / "validate_SUBMISSION.json",
        {"source": source, "action": "validate", "job_id": "234358", "status": "SUBMITTED"},
    )
    save(old / "validate_ATTEMPT.json", {"source": source, "action": "validate"})
    prior = save(
        old / "CONTRACT.json",
        {
            "schema": "srf6-memory-repair-contract-v1",
            "repair_source": {"revision": e.OLD_REPAIR_REVISION},
            "max_host_bytes": 96 * 2**30,
            "original_files": [{"path": "frozen", "sha256": "0" * 64}],
        },
    )
    monkeypatch.setattr(e, "OLD_CONTRACT_SHA", prior["sha256"])
    return {
        "old_allocation_failure_ref": reference,
        "prior_contract_ref": prior,
        "schema": e.SCHEMA,
        "repair_source": {"revision": "1" * 40},
        "max_host_bytes": e.HOST_BYTES,
        "max_disk_bytes": e.DISK_BYTES,
        "original_files": [{"path": "frozen", "sha256": "0" * 64}],
    }


def test_v2_binds_original_premodel_allocation_failure(allocation_failure):
    e._prior_allocation_failure(allocation_failure)


@pytest.mark.parametrize("name", ["validate_STARTED.json", "VALIDATION", "resume_ATTEMPT.json"])
def test_v2_refuses_prior_actual_execution(allocation_failure, name):
    save(e.OLD_OPS / name, {})
    with pytest.raises(ValueError, match=r"model execution|resume intent"):
        e._prior_allocation_failure(allocation_failure)


def test_v2_refuses_unrelated_prior_failure(allocation_failure):
    reference = allocation_failure["old_allocation_failure_ref"]
    value = json.loads(Path(reference["path"]).read_bytes())
    value["error"] = "Actual model CUDA out of memory"
    allocation_failure["old_allocation_failure_ref"] = save(Path(reference["path"]), value)
    with pytest.raises(ValueError, match="pre-model allocation failure differs"):
        e._prior_allocation_failure(allocation_failure)


def test_v2_refuses_relabelled_old_source(allocation_failure):
    save(
        e.OLD_OPS / "validate_ATTEMPT.json",
        {"action": "validate", "source": {"revision": "f" * 40}},
    )
    with pytest.raises(ValueError, match="source identity differs"):
        e._prior_allocation_failure(allocation_failure)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cleanup_errors", ["cannot unlink owned tensor"]),
        ("storage_failures", ["disk read changed bytes"]),
        ("peak_disk_bytes", 2049),
        ("peak_host_bytes", 1025),
        ("peak_staging_host_bytes", 64 * 2**20 + 1),
        ("live_disk_bytes", 1),
        ("disk_bytes_read", 0),
    ],
)
def test_hybrid_failure_cannot_publish_success(synthetic_protocol, field, value):
    fixture = synthetic_protocol
    fixture.summary[field] = value
    with pytest.raises(ValueError, match="storage context"):
        execute_fixture(fixture)
    assert fixture.verified == []
    assert not (fixture.root / e.JOBS / "TASKS" / e.QUEUE[0] / "COMPLETE.json").exists()


def test_hybrid_storage_facts_precede_complete(synthetic_protocol):
    fixture = synthetic_protocol
    execute_fixture(fixture)
    folder = fixture.path.parent / "EXECUTIONS" / e.QUEUE[0]
    facts = json.loads((folder / "STORAGE_FACTS.json").read_bytes())
    complete = json.loads((folder / "COMPLETE.json").read_bytes())
    ref = complete["storage_facts_ref"]
    assert e.strict_ref(ref["path"], ref["sha256"]) == (folder / "STORAGE_FACTS.json").read_bytes()
    assert facts["status"] == "ACTUAL_STORAGE_CONTEXT_COMPLETE_NOT_SCIENTIFIC_PASS"
    assert facts["offload"]["max_disk_bytes"] == 2048
    assert facts["offload"]["scratch_directory"] == str(folder / "STORAGE")
    assert not list((folder / "STORAGE").iterdir())


def test_v2_cannot_change_original_evidence_or_scientific_registration(allocation_failure):
    allocation_failure["original_files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="altered original evidence"):
        e._prior_allocation_failure(allocation_failure)


def test_v2_cannot_substitute_prior_contract(allocation_failure):
    allocation_failure["prior_contract_ref"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Exact immutable v1"):
        e._prior_allocation_failure(allocation_failure)


@pytest.mark.parametrize("peak,written,read", [(128, 0, 0), (0, 128, 128), (-1, 128, 128)])
def test_hybrid_peak_and_io_counters_must_agree(synthetic_protocol, peak, written, read):
    fixture = synthetic_protocol
    fixture.summary.update(peak_disk_bytes=peak, disk_bytes_written=written, disk_bytes_read=read)
    with pytest.raises(ValueError, match="storage context"):
        execute_fixture(fixture)
    assert fixture.verified == []


def test_hybrid_zero_disk_peak_requires_zero_io(synthetic_protocol):
    fixture = synthetic_protocol
    fixture.summary.update(peak_disk_bytes=0, disk_bytes_written=0, disk_bytes_read=0)
    execute_fixture(fixture)
    assert len(fixture.verified) == 1


def test_old_storage_schema_rejected(synthetic_protocol):
    fixture = synthetic_protocol
    fixture.summary["schema"] = "srf6-saved-activation-offload-v1"
    with pytest.raises(ValueError, match="storage context"):
        execute_fixture(fixture)
    assert fixture.verified == []


@pytest.fixture
def predecessor(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "PROJECT", tmp_path)
    monkeypatch.setattr(e, "ROOT", tmp_path / "root")
    monkeypatch.setattr(e, "OPS", tmp_path / "v3")
    monkeypatch.setattr(e, "V3_OPS", tmp_path / "v3")
    monkeypatch.setattr(e, "PREDECESSOR_OPS", tmp_path / "v2")

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    prior = dict(
        schema="srf6-memory-repair-contract-v2",
        repair_source=dict(revision=e.PREDECESSOR_REVISION),
        original_files=[{"path": "untouched", "sha256": "a" * 64}],
        max_host_bytes=e.HOST_BYTES,
        max_disk_bytes=e.DISK_BYTES,
        pins={"frozen": True},
    )
    prior_ref = write(e.PREDECESSOR_OPS / "CONTRACT.json", prior)
    monkeypatch.setattr(e, "PREDECESSOR_CONTRACT_SHA", prior_ref["sha256"])
    write(
        e.PREDECESSOR_OPS / "VALIDATION/ATTEMPT.json",
        dict(schema="srf6-memory-storage-validation-v2", contract_sha256=prior_ref["sha256"]),
    )
    inner = dict(status="FAILED_PRESERVED", error_type="KeyboardInterrupt")
    outer = dict(status="FAILED_PRESERVED", job_id="234384", retry_authorized=False)
    interruption = dict(
        schema="srf6-memory-performance-interruption-v1",
        status="EXACT_CHILD_SIGINT_SENT",
        reason="performance_supersession",
        job_id="234384",
        source_revision=e.PREDECESSOR_REVISION,
        signal="SIGINT",
        child_pid=872291,
        parent_pid=800000,
        returncode=0,
        child_start_time="known-start",
        cgroup="job_234384",
        command_line="bound-child",
    )
    disposition = dict(
        schema="srf6-memory-performance-transition-v1",
        status="TECHNICAL_SUPERSESSION_PRESERVED",
        predecessor_job_id="234384",
        predecessor_source_revision=e.PREDECESSOR_REVISION,
        predecessor_contract_ref=prior_ref,
        terminal_state="FAILED",
        exit_code="1:0",
        reason="performance_supersession",
        inner_failure_ref=write(e.PREDECESSOR_OPS / "VALIDATION/FAILURE.json", inner),
        outer_failure_ref=write(e.PREDECESSOR_OPS / "validate_FAILURE.json", outer),
        interruption_ref=write(e.OPS / "INTERRUPTION_RESULT.json", interruption),
    )
    contract = {
        **prior,
        "schema": "srf6-memory-repair-contract-v3",
        "repair_source": {"revision": "a" * 40},
        "storage_backend": "arena_v3",
        "predecessor_contract_ref": prior_ref,
        "predecessor_disposition_ref": write(e.OPS / "PREDECESSOR_DISPOSITION.json", disposition),
    }
    return contract, disposition, write


def test_predecessor_preserves_exact_science_and_controlled_failure(predecessor):
    contract, disposition, _ = predecessor
    assert e._v2_predecessor_evidence(contract) == disposition


@pytest.mark.parametrize(
    "field,value",
    [
        ("pins", {"frozen": False}),
        ("original_files", []),
        ("max_disk_bytes", 2**40),
        ("extra", "unregistered"),
    ],
)
def test_v3_cannot_change_predecessor_registration(predecessor, field, value):
    contract, _, _ = predecessor
    contract[field] = value
    with pytest.raises(ValueError, match="altered v2"):
        e._v2_predecessor_evidence(contract)


@pytest.mark.parametrize(
    "path",
    [
        "resume_ATTEMPT.json",
        "resume_SUBMISSION.json",
        "resume_COMPLETE.json",
        "EXECUTIONS/partial",
        "validate_COMPLETE.json",
        "VALIDATION/RESULT.json",
    ],
)
def test_any_predecessor_resume_or_late_success_blocks_transition(predecessor, path):
    contract, _, write = predecessor
    write(e.PREDECESSOR_OPS / path, {})
    with pytest.raises(ValueError, match="forbids transition"):
        e._v2_predecessor_evidence(contract)


@pytest.mark.parametrize("error_type", ["RuntimeError", "ValueError", "OutOfMemoryError"])
def test_numerical_or_other_failure_never_recast_as_performance_retry(predecessor, error_type):
    contract, disposition, write = predecessor
    disposition["inner_failure_ref"] = write(
        e.PREDECESSOR_OPS / "VALIDATION/FAILURE.json", dict(error_type=error_type)
    )
    contract["predecessor_disposition_ref"] = write(
        e.OPS / "PREDECESSOR_DISPOSITION.json", disposition
    )
    with pytest.raises(ValueError, match="Only controlled"):
        e._v2_predecessor_evidence(contract)


def test_predecessor_wrong_source_rejected(predecessor):
    contract, disposition, write = predecessor
    disposition["predecessor_source_revision"] = "a" * 40
    contract["predecessor_disposition_ref"] = write(
        e.OPS / "PREDECESSOR_DISPOSITION.json", disposition
    )
    with pytest.raises(ValueError, match="Controlled technical"):
        e._v2_predecessor_evidence(contract)


def test_predecessor_disposition_bytes_cannot_change(predecessor):
    contract, _, _ = predecessor
    (e.OPS / "PREDECESSOR_DISPOSITION.json").write_text("{}")
    with pytest.raises(ValueError):
        e._v2_predecessor_evidence(contract)


@pytest.mark.parametrize(
    "field,value",
    [
        (None, None),
        ("arena_physical_bytes", 1),
        ("peak_arena_physical_bytes", 2049),
        ("arena_deleted_count", 0),
        ("arena_open_count", 2),
        ("arena_fsync_count", 2),
        ("disk_backend", "file_v2"),
    ],
)
def test_arena_storage_summary_bound_and_reclaimed(synthetic_protocol, tmp_path, field, value):
    scratch = tmp_path / "arena-summary"
    scratch.mkdir()
    summary = synthetic_protocol.summary
    summary.update(
        schema="srf6-saved-activation-arena-v3",
        disk_backend="arena_v3",
        scratch_directory=str(scratch),
        arena_physical_bytes=0,
        peak_arena_physical_bytes=128,
        arena_created_count=1,
        arena_deleted_count=1,
        arena_open_count=1,
        arena_fsync_count=1,
        arena_read_calls=1,
        arena_write_calls=1,
    )
    if field:
        summary[field] = value
    manager = SimpleNamespace(summary=lambda: summary)
    if field:
        with pytest.raises(ValueError, match="arena storage context"):
            e._storage_summary(manager, 1024, 2048, scratch, disk_backend="arena_v3")
    else:
        assert e._storage_summary(manager, 1024, 2048, scratch, disk_backend="arena_v3") == summary


@pytest.fixture
def v4_predecessor(predecessor, monkeypatch):
    prior, _, write = predecessor
    prior["repair_source"] = {"revision": e.V3_REVISION}
    prior_ref = write(e.V3_OPS / "CONTRACT.json", prior)
    monkeypatch.setattr(e, "V3_CONTRACT_SHA", prior_ref["sha256"])
    monkeypatch.setattr(e, "OPS", e.PROJECT / "v4")
    write(
        e.V3_OPS / "VALIDATION/ATTEMPT.json",
        dict(schema="srf6-memory-storage-validation-v3", contract_sha256=prior_ref["sha256"]),
    )
    interruption = dict(
        schema="srf6-memory-performance-interruption-v1",
        status="EXACT_CHILD_SIGINT_SENT",
        reason="performance_supersession",
        job_id="235551",
        source_revision=e.V3_REVISION,
        signal="SIGINT",
        child_pid=765722,
        parent_pid=765710,
        child_start_time="285477529",
        cgroup="job_235551",
        command_line="bound-child",
        returncode=0,
    )
    disposition = dict(
        schema="srf6-memory-performance-transition-v1",
        status="TECHNICAL_SUPERSESSION_PRESERVED",
        predecessor_job_id="235551",
        predecessor_source_revision=e.V3_REVISION,
        predecessor_contract_ref=prior_ref,
        terminal_state="FAILED",
        exit_code="1:0",
        reason="performance_supersession",
        inner_failure_ref=write(
            e.V3_OPS / "VALIDATION/FAILURE.json",
            dict(status="FAILED_PRESERVED", error_type="KeyboardInterrupt"),
        ),
        outer_failure_ref=write(
            e.V3_OPS / "validate_FAILURE.json",
            dict(status="FAILED_PRESERVED", job_id="235551", retry_authorized=False),
        ),
        interruption_ref=write(e.OPS / "INTERRUPTION_RESULT.json", interruption),
    )
    performance = dict(
        schema="srf6-memory-performance-reference-v1",
        status="OBSERVED_TIMING_PROJECTION_NOT_SERVER_ATTESTATION",
        prior_job_id="235551",
        prior_contract_ref=prior_ref,
        local_snapshot=dict(path="/local/STATUS_20261030T100149721453.json", sha256="a" * 64),
        observation_at="2026-10-30T10:01:49Z",
        progress_sequence=26,
        completed_rows=4,
        rows=[0, 1, 2, 3],
        observed_total_row_seconds=1805.6315779644065,
        max_total_row_seconds=902.8157889822033,
    )
    contract = {
        **prior,
        "schema": e.SCHEMA,
        "repair_source": {"revision": "b" * 40},
        "max_device_bytes": e.DEVICE_BYTES,
        "validation_window": dict(e.WINDOW),
        "performance_reference_ref": write(e.OPS / "PERFORMANCE_REFERENCE.json", performance),
        "v3_predecessor_contract_ref": prior_ref,
        "v3_predecessor_disposition_ref": write(
            e.OPS / "PREDECESSOR_DISPOSITION.json", disposition
        ),
    }
    return contract, disposition, write


def test_v4_preserves_complete_v2_v3_chain(v4_predecessor):
    contract, disposition, _ = v4_predecessor
    assert e.predecessor_evidence(contract) == disposition


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_device_bytes", 25 * 2**30),
        ("validation_window", {"rows": [0, 1, 2]}),
        ("storage_backend", "file_v2"),
        ("original_files", []),
    ],
)
def test_v4_budget_window_and_science_cannot_change(v4_predecessor, field, value):
    contract, _, _ = v4_predecessor
    contract[field] = value
    with pytest.raises(ValueError):
        e.predecessor_evidence(contract)


@pytest.mark.parametrize(
    "path",
    [
        "resume_ATTEMPT.json",
        "resume_SUBMISSION.json",
        "resume_COMPLETE.json",
        "VALIDATION/RESULT.json",
        "validate_COMPLETE.json",
        "EXECUTIONS/partial",
    ],
)
def test_v4_v3_success_or_resume_blocks(v4_predecessor, path):
    contract, _, write = v4_predecessor
    write(e.V3_OPS / path, {})
    with pytest.raises(ValueError, match="forbids transition"):
        e.predecessor_evidence(contract)


def test_v4_preserved_v2_bytes_are_still_required(v4_predecessor):
    contract, _, _ = v4_predecessor
    (e.PREDECESSOR_OPS / "CONTRACT.json").write_text("{}")
    with pytest.raises(ValueError):
        e.predecessor_evidence(contract)


@pytest.mark.parametrize("error_type", ["ValueError", "TechnicalWindowStop", "OutOfMemoryError"])
def test_v4_does_not_recast_scientific_or_other_technical_failures(v4_predecessor, error_type):
    contract, disposition, write = v4_predecessor
    disposition["inner_failure_ref"] = write(
        e.V3_OPS / "VALIDATION/FAILURE.json", dict(status="FAILED_PRESERVED", error_type=error_type)
    )
    contract["v3_predecessor_disposition_ref"] = write(
        e.OPS / "PREDECESSOR_DISPOSITION.json", disposition
    )
    with pytest.raises(ValueError, match="Only controlled"):
        e.predecessor_evidence(contract)


def test_all_device_residency_task_storage_completes(synthetic_protocol, tmp_path):
    from sr_f6_memory_repair.residency import DeviceResidency

    scratch = tmp_path / "all-device-summary"
    scratch.mkdir()
    summary = synthetic_protocol.summary
    summary.update(DeviceResidency(e.DEVICE_BYTES).summary())
    summary.update(
        schema="srf6-saved-activation-arena-v3",
        disk_backend="arena_v3",
        scratch_directory=str(scratch),
        unpack_count=0,
        peak_host_bytes=0,
        peak_disk_bytes=0,
        disk_bytes_read=0,
        disk_bytes_written=0,
        arena_physical_bytes=0,
        peak_arena_physical_bytes=0,
        arena_created_count=0,
        arena_deleted_count=0,
        arena_open_count=0,
        arena_fsync_count=0,
        arena_read_calls=0,
        arena_write_calls=0,
        device_resident_pack_count=2,
        device_resident_unpack_count=2,
        peak_device_bytes=128,
        peak_device_storage_count=1,
    )
    manager = SimpleNamespace(summary=lambda: summary)
    assert (
        e._storage_summary(
            manager, 1024, 2048, scratch, disk_backend="arena_v3", max_device_bytes=e.DEVICE_BYTES
        )
        == summary
    )
    # A v3 receipt cannot borrow v4 counters to invent an old offload execution.
    with pytest.raises(ValueError, match="storage context"):
        e._storage_summary(manager, 1024, 2048, scratch, disk_backend="arena_v3")
