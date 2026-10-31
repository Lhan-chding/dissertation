"""Metadata contracts fail closed without importing tensor/model runtimes."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sr_f6_round_b import contract as c


def test_no_scientific_or_queue_override_allowed():
    original = c.envelope({})
    c.shape(original)
    for key, value in [
        ("pins", {}),
        ("queues", list(reversed(c.QUEUES))),
        ("schema", "other"),
        ("resume_complete_ref", {}),
    ]:
        changed = copy.deepcopy(original)
        changed[key] = value
        with pytest.raises(ValueError):
            c.shape(changed)


def test_metadata_import_has_no_torch_or_scientific_producer_import():
    src = Path(__file__).resolve().parents[2] / "src"
    command = (
        "import sys;sys.path.insert(0,"
        + repr(str(src))
        + ");import sr_f6_round_b.contract;assert 'torch' not in sys.modules;"
        "assert not any(x.startswith('sr_f6.') for x in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", command],
        check=True,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
    )


@pytest.fixture
def predecessor(tmp_path, monkeypatch):
    monkeypatch.setattr(c.resume.legacy, "PROJECT", tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(c, "ROOT", root)
    ops = tmp_path / "ops"
    ops.mkdir()
    monkeypatch.setattr(c, "OPS", ops)
    v8 = tmp_path / "v8"
    v8.mkdir()
    monkeypatch.setattr(c, "V8_DEPLOYMENT", v8)
    monkeypatch.setattr(c.resume, "__file__", str(v8 / "src/sr_f6_memory_resume/contract.py"))
    manifest = v8 / "RESUME_SOURCE.json"
    manifest.write_text("synthetic immutable source manifest")
    monkeypatch.setattr(c, "V8_BINDING", hashlib.sha256(manifest.read_bytes()).hexdigest())
    previous = {
        "full_contract_ref": {"path": "full", "sha256": "a" * 64},
        "full_result_ref": {"path": "result", "sha256": "b" * 64},
    }
    scientific = dict(
        experiment_root=str(root),
        storage_backend="arena_v3",
        max_device_bytes=44 * 2**30,
        max_host_bytes=8 * 2**30,
        max_disk_bytes=96 * 2**30,
        plan_ref={"path": "original-plan", "sha256": "c" * 64},
    )
    monkeypatch.setattr(c.resume, "validate_metadata", lambda *a: previous)
    monkeypatch.setattr(c.resume, "full_contract", lambda *a: scientific)
    value = c.envelope({})
    complete = dict(
        status="QUEUE_COMPLETE_NOT_GATE_SIGNED",
        job_id="235953",
        contract_ref=value["resume_contract_ref"],
        queue=list(c.resume.QUEUE),
        scientific_gate_status="NOT_SIGNED",
        completed=[
            dict(
                task_id=t,
                task_complete_ref={"path": t + "-task"},
                execution_complete_ref={"path": t + "-execution"},
                storage_facts_ref={"path": t + "-facts"},
            )
            for t in c.resume.QUEUE
        ],
    )
    audit = dict(
        status="VERIFIED_A_SIX_TASK_DELIVERY_NOT_SCIENTIFIC_GATE",
        resume_complete_ref=value["resume_complete_ref"],
        verified_checkpoint_files=78,
        verified_updates=72,
        audit_generated_responses=0,
        audit_optimizer_updates=0,
        scientific_gate="NOT_SIGNED",
        tasks=[],
    )
    reads = {value["resume_complete_ref"]["path"]: complete, value["a_delivery_ref"]["path"]: audit}
    wrappers = []
    for task in sorted(c.A_TASKS):
        basename = task + ".json"
        ref = {"path": ".local/" + basename, "sha256": "d" * 64}
        audit["tasks"].append({"task_id": task, "audit_ref": ref})
        receipt = dict(
            task={"task_id": task},
            plan_ref=scientific["plan_ref"],
            producer={"path": task + "/producer.json", "sha256": "e" * 64},
        )
        folder = root / c.resume.legacy.JOBS / "TASKS" / task
        folder.mkdir(parents=True)
        (folder / "COMPLETE.json").write_text(json.dumps(receipt))
        checked = dict(
            status="VERIFIED_A_TASK_DELIVERY_NOT_SCIENTIFIC_GATE",
            task={"task_id": task},
            receipt=receipt,
            torch_threads=1,
            steps=[{}] * 12,
            checkpoint_files=[{}] * 13,
            generated_responses=0,
            optimizer_updates_performed=0,
            checkpoint_tensor_deserialization=False,
        )
        wrapper = dict(returncode=0, task=task, stdout=json.dumps(checked))
        wrappers.append((wrapper, checked))
        reads[str(ops / "A_EVIDENCE" / basename)] = wrapper
    monkeypatch.setattr(c, "read_ref", lambda ref: reads.get(ref["path"], {}))
    return value, scientific, audit, complete, wrappers


def test_bound_a_delivery_is_only_admission_precedent(predecessor):
    value, scientific, *_ = predecessor
    assert c.completed_a(value)[1] is scientific


@pytest.mark.parametrize(
    "field,value",
    [
        ("torch_threads", 12),
        ("steps", [{}] * 11),
        ("checkpoint_files", [{}] * 12),
        ("optimizer_updates_performed", 1),
        ("generated_responses", 1),
        ("checkpoint_tensor_deserialization", True),
    ],
)
def test_a_audit_environment_or_scope_mismatch_blocks_b(predecessor, field, value):
    envelope, _, _, _, wrappers = predecessor
    wrapper, checked = wrappers[0]
    checked[field] = value
    wrapper["stdout"] = json.dumps(checked)
    with pytest.raises(ValueError, match="single-thread A audit"):
        c.completed_a(envelope)


def test_missing_sixth_a_task_or_resume_failure_never_admits_b(predecessor):
    value, _, audit, *_ = predecessor
    audit["tasks"].pop()
    with pytest.raises(ValueError, match="six original"):
        c.completed_a(value)


def test_metadata_does_not_accept_changed_original_task_receipt(predecessor):
    value, *_ = predecessor
    path = c.ROOT / c.resume.legacy.JOBS / "TASKS" / next(iter(c.A_TASKS)) / "COMPLETE.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="closure changed"):
        c.completed_a(value)


def test_cold_execution_requires_actual_v7_result_before_work(monkeypatch, tmp_path):
    path = tmp_path / "CONTRACT.json"
    monkeypatch.setattr(c, "OPS", tmp_path)
    monkeypatch.setattr(c.resume.legacy, "PROJECT", tmp_path.parent)
    value = c.envelope({})
    previous = {
        "full_contract_ref": {"path": "v7", "sha256": "7" * 64},
        "full_result_ref": {"path": "v7-result", "sha256": "8" * 64},
    }
    monkeypatch.setattr(c, "validate_metadata", lambda *a: value)
    monkeypatch.setattr(c, "completed_a", lambda *a: (previous, {}))
    events = []
    monkeypatch.setattr(
        c.resume.full,
        "verify_contract",
        lambda *a: events.append("full-source-reference-verification"),
    )
    monkeypatch.setattr(
        c.resume,
        "verify_validation",
        lambda *a: (
            events.append("cold-tensors")
            or {
                "status": "SMOKE_ONLY",
                "contract_sha256": "7" * 64,
                "generated_responses": 0,
                "optimizer_updates": 0,
            }
        ),
    )
    with pytest.raises(ValueError, match="cold storage"):
        c.verify_contract(path, "f" * 64)
    assert events == ["full-source-reference-verification", "cold-tensors"]


def test_source_byte_closure_and_actual_loaded_origins(tmp_path, monkeypatch):
    import sr_f6_round_b
    from sr_f6_round_b import execution

    monkeypatch.setattr(c, "PROJECT", tmp_path)
    monkeypatch.setattr(c.resume.legacy, "PROJECT", tmp_path)
    revision = "a" * 40
    deployment = tmp_path / ("sr_f6_round_b_code_" + revision[:7])
    files = {}
    for name in c.MEMBERS:
        path = deployment / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source member " + name)
        files[name] = dict(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    manifest = dict(
        schema=c.SOURCE_SCHEMA,
        revision=revision,
        deployment=str(deployment),
        producer_revision=c.resume.legacy.PRODUCER_REVISION,
        files=files,
    )
    (deployment / "ROUND_B_SOURCE.json").write_text(json.dumps(manifest))
    source = dict(
        revision=revision,
        deployment=str(deployment),
        files=[dict(path=k, sha256=v["sha256"]) for k, v in files.items()],
    )
    monkeypatch.setattr(c, "__file__", str(deployment / c.MEMBERS[1]))
    monkeypatch.setattr(sr_f6_round_b, "__file__", str(deployment / c.MEMBERS[0]))
    # Matching bytes are insufficient if the helper is imported elsewhere.
    with pytest.raises(ValueError, match="another deployment"):
        c.sources(source)
    monkeypatch.setattr(execution, "__file__", str(deployment / c.MEMBERS[2]))
    c.sources(source)
    (deployment / c.MEMBERS[2]).write_text("changed after freeze")
    with pytest.raises(ValueError):
        c.sources(source)
