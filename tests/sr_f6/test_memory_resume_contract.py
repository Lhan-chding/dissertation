"""V8 rejects foreign/partial admission and scientific overrides without CUDA."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sr_f6_memory_resume import contract as c


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(data)
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())


@pytest.fixture
def bound(tmp_path, monkeypatch):
    monkeypatch.setattr(c.legacy, "PROJECT", tmp_path)
    monkeypatch.setattr(c, "PROJECT", tmp_path)
    monkeypatch.setattr(c, "OPS", tmp_path / "v8")
    monkeypatch.setattr(c, "FULL_OPS", tmp_path / "v7")
    monkeypatch.setattr(c, "FULL_DEPLOYMENT", tmp_path / "old")
    envelope = dict(
        schema=c.SCHEMA,
        execution_source=dict(revision="a" * 40),
        queue=list(c.QUEUE),
        **c.predecessor_refs(),
    )
    calls = []
    contract = dict(
        repair_source=dict(revision=c.FULL_REVISION, deployment=str(c.FULL_DEPLOYMENT)),
        pins={"phase": "PILOT_V3"},
    )
    monkeypatch.setattr(c, "_sources", lambda e: calls.append("source"))
    monkeypatch.setattr(
        c.full,
        "verify_validation_metadata",
        lambda *_: dict(
            status=c.full.STATUS, contract=contract, cold_verification_ref=envelope["full_cold_ref"]
        ),
    )
    complete = dict(
        status="COMPLETED_FULL_VALIDATION_NO_TRAINING",
        job_id="235864",
        source=dict(
            revision=c.FULL_REVISION,
            deployment=str(c.FULL_DEPLOYMENT),
            binding_sha256=c.FULL_BINDING,
        ),
        contract_ref=envelope["full_contract_ref"],
        result_ref=envelope["full_result_ref"],
        cold_verification_ref=envelope["full_cold_ref"],
    )
    ref = write(c.FULL_OPS / "validate_COMPLETE.json", complete)
    monkeypatch.setattr(c, "FULL_COMPLETE_SHA", ref["sha256"])
    envelope["full_complete_ref"] = ref
    return envelope, complete, contract, calls


def check(envelope, path=None):
    ref = write(path or c.OPS / "CONTRACT.json", envelope)
    return c.validate_metadata(ref["path"], ref["sha256"])


def test_only_final_execution_and_same_byte_preflight(bound, monkeypatch):
    envelope, _, original, calls = bound
    assert check(envelope) == envelope
    assert check(envelope, c.OPS / "PREFLIGHT_CONTRACT.json") == envelope
    assert c.full_contract(envelope) is original
    with pytest.raises(ValueError, match="final v8"):
        c.verify_contract(c.OPS / "PREFLIGHT_CONTRACT.json", "a" * 64)
    monkeypatch.setattr(c.full, "verify_contract", lambda *args: calls.append(args))
    ref = write(c.OPS / "CONTRACT.json", envelope)
    assert c.verify_contract(ref["path"], ref["sha256"]) == envelope
    assert calls[-1] == (
        envelope["full_contract_ref"]["path"],
        envelope["full_contract_ref"]["sha256"],
    )


@pytest.mark.parametrize("field", ["pins", "max_device_bytes", "retry", "validation_window"])
def test_no_scientific_or_storage_override_in_envelope(bound, field):
    envelope, *_ = bound
    envelope[field] = 1
    with pytest.raises(ValueError, match="scientific overrides"):
        check(envelope)


@pytest.mark.parametrize("field", list(c.predecessor_refs()))
def test_no_rebinding_of_success_refs(bound, field):
    envelope, *_ = bound
    envelope[field] = dict(envelope[field], sha256="f" * 64)
    with pytest.raises(ValueError, match="Exact completed"):
        check(envelope)


@pytest.mark.parametrize("queue", [[], list(reversed(c.QUEUE)), [*c.QUEUE, c.QUEUE[0]]])
def test_queue_order_and_no_repetition(bound, queue):
    envelope, *_ = bound
    envelope["queue"] = queue
    with pytest.raises(ValueError, match="fixed b1"):
        check(envelope)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "PASS"),
        ("job_id", "235713"),
        ("contract_ref", {}),
        ("result_ref", {}),
        ("cold_verification_ref", {}),
        ("source", {}),
    ],
)
def test_metadata_rejects_incomplete_or_foreign_outer_receipt(bound, monkeypatch, field, value):
    envelope, complete, *_ = bound
    complete[field] = value
    ref = write(c.FULL_OPS / "validate_COMPLETE.json", complete)
    monkeypatch.setattr(c, "FULL_COMPLETE_SHA", ref["sha256"])
    envelope["full_complete_ref"] = ref
    with pytest.raises(ValueError, match="outer completion"):
        check(envelope)


def test_cold_tensor_verifier_receives_original_contract(bound, monkeypatch):
    from sr_f6_memory_repair import validation_full

    envelope, *_ = bound
    calls = []
    monkeypatch.setattr(
        validation_full, "verify_validation", lambda *args: calls.append(args) or {"actual": True}
    )
    assert c.verify_validation(envelope) == {"actual": True}
    assert calls == [
        (
            envelope["full_contract_ref"]["path"],
            envelope["full_contract_ref"]["sha256"],
            envelope["full_result_ref"],
        )
    ]


def test_metadata_module_import_is_stdlib_only():
    code = "import sys; sys.path.insert(0," + repr(str(Path(c.__file__).parents[1])) + "); "
    code += (
        "from sr_f6_memory_resume import contract; assert 'torch' not in sys.modules; "
        "assert 'sr_f6' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def source_fixture(tmp_path, monkeypatch):
    import sr_f6_memory_resume
    from sr_f6_memory_resume import execution

    monkeypatch.setattr(c, "PROJECT", tmp_path)
    monkeypatch.setattr(c.legacy, "PROJECT", tmp_path)
    target = tmp_path / ("sr_f6_memory_resume_code_" + "a" * 7)
    source = dict(revision="a" * 40, deployment=str(target), files=[])
    manifest = dict(
        schema="srf6-memory-resume-source-v1",
        revision="a" * 40,
        deployment=str(target),
        producer_revision=c.legacy.PRODUCER_REVISION,
        files={},
    )
    for name in c.MEMBERS:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# source\n")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        source["files"].append(dict(path=name, sha256=sha))
        manifest["files"][name] = dict(bytes=path.stat().st_size, sha256=sha)
    write(target / "RESUME_SOURCE.json", manifest)
    monkeypatch.setattr(c, "__file__", str(target / c.MEMBERS[1]))
    monkeypatch.setattr(execution, "__file__", str(target / c.MEMBERS[2]))
    monkeypatch.setattr(sr_f6_memory_resume, "__file__", str(target / c.MEMBERS[0]))
    old = tmp_path / "frozen_v7"
    monkeypatch.setattr(c, "FULL_DEPLOYMENT", old)
    monkeypatch.setattr(c.full, "__file__", str(old / "src/sr_f6_memory_repair/full_contract.py"))
    monkeypatch.setattr(c.legacy, "__file__", str(old / "src/sr_f6_memory_repair/execution.py"))
    ref = write(old / "FULL_SOURCE.json", {"frozen": True})
    monkeypatch.setattr(c, "FULL_BINDING", ref["sha256"])
    return dict(execution_source=source)


@pytest.mark.parametrize(
    "mutation", [None, "missing", "duplicate", "absolute", "revision", "bytes", "origin", "v7"]
)
def test_actual_five_file_closure_and_frozen_origins(tmp_path, monkeypatch, mutation):
    envelope = source_fixture(tmp_path, monkeypatch)
    source = envelope["execution_source"]
    if mutation == "missing":
        source["files"].pop()
    elif mutation == "duplicate":
        source["files"].append(copy.deepcopy(source["files"][0]))
    elif mutation == "absolute":
        source["files"][0]["path"] = str(Path(source["deployment"]) / c.MEMBERS[0])
    elif mutation == "revision":
        source["revision"] = c.FULL_REVISION
    elif mutation == "bytes":
        (Path(source["deployment"]) / c.MEMBERS[0]).write_bytes(b"changed")
    elif mutation == "origin":
        monkeypatch.setattr(c, "__file__", "/alien/contract.py")
    elif mutation == "v7":
        monkeypatch.setattr(c.full, "__file__", "/copied/full_contract.py")
    if mutation is None:
        c._sources(envelope)
    else:
        with pytest.raises(ValueError):
            c._sources(envelope)
