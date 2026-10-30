"""V7 metadata fixtures; no cluster, tensors or scientific admission."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sr_f6_memory_repair import full_contract as f


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(data)
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())


@pytest.fixture
def bound(tmp_path, monkeypatch):
    monkeypatch.setattr(f.legacy, "PROJECT", tmp_path)
    for key, value in dict(
        PROJECT=tmp_path, OPS=tmp_path / "v7", V4_OPS=tmp_path / "v4", V6_OPS=tmp_path / "v6"
    ).items():
        monkeypatch.setattr(f, key, value)
    original = dict(
        schema=f.legacy.SCHEMA,
        repair_source=dict(revision="a" * 40),
        max_device_bytes=24 * 2**30,
        max_host_bytes=f.HOST_BYTES,
        max_disk_bytes=f.DISK_BYTES,
        storage_backend="arena_v3",
        validation_window=copy.deepcopy(f.WINDOW),
        original_files=[dict(path="original/kept", sha256="d" * 64)],
        pins=dict(phase="PILOT_V3"),
        plan_ref=dict(path="PLAN.json", sha256="e" * 64),
        producer=dict(revision=f.legacy.PRODUCER_REVISION),
    )
    ref = write(f.V4_OPS / "CONTRACT.json", original)
    monkeypatch.setattr(f, "V4_SHA", ref["sha256"])
    prior = dict(repair_source=dict(revision=f.V6_REVISION))
    ref = write(f.V6_OPS / "CONTRACT.json", prior)
    monkeypatch.setattr(f, "V6_SHA", ref["sha256"])
    contract = copy.deepcopy(original)
    contract.update(
        schema=f.SCHEMA,
        max_device_bytes=f.DEVICE_BYTES,
        repair_source=dict(revision="b" * 40),
        **f.predecessor_refs(),
    )
    calls = []
    monkeypatch.setattr(f.legacy, "_prior_allocation_failure", lambda c: calls.append("v1"))
    monkeypatch.setattr(f.legacy, "predecessor_evidence", lambda c: calls.append("v2_v3"))
    monkeypatch.setattr(f, "_prior_success", lambda c, p: calls.append("v6"))
    monkeypatch.setattr(f, "_sources", lambda c, p: calls.append("sources"))
    return contract, calls


def check(contract, path=None):
    ref = write(path or f.OPS / "CONTRACT.json", contract)
    return f.validate_metadata(ref["path"], ref["sha256"])


def test_metadata_inherits_original_closure_and_preflight_only(bound, monkeypatch):
    contract, calls = bound
    assert check(contract) == contract
    assert calls == ["v1", "v2_v3", "v6", "sources"]
    assert check(contract, f.OPS / "PREFLIGHT_CONTRACT.json") == contract
    with pytest.raises(ValueError, match="final contract"):
        f.verify_contract(f.OPS / "PREFLIGHT_CONTRACT.json", "a" * 64)
    with pytest.raises(ValueError, match="Unique V7"):
        check(contract, f.OPS / "OTHER.json")
    monkeypatch.setattr(f.legacy, "_source", lambda c: calls.append("source_gpu"))
    monkeypatch.setattr(f.legacy, "_original_evidence", lambda c: calls.append("closure_gpu"))
    ref = write(f.OPS / "CONTRACT.json", contract)
    assert f.verify_contract(ref["path"], ref["sha256"]) == contract
    assert calls[-2:] == ["source_gpu", "closure_gpu"]


@pytest.mark.parametrize("key", sorted(f.ADDITIONS))
def test_exact_predecessors_cannot_be_removed_or_rebound(bound, key):
    contract, _ = bound
    del contract[key]
    with pytest.raises(ValueError, match="Exact completed predecessor"):
        check(contract)


@pytest.mark.parametrize("mutation", ["files", "pins", "plan", "window", "extra", "backend"])
def test_original_design_and_failure_refs_cannot_change(bound, mutation):
    contract, _ = bound
    if mutation == "files":
        contract["original_files"][0]["sha256"] = "f" * 64
    elif mutation == "pins":
        contract["pins"]["phase"] = "PILOT_OTHER"
    elif mutation == "plan":
        contract["plan_ref"]["sha256"] = "f" * 64
    elif mutation == "window":
        contract["validation_window"]["max_total_row_seconds"] = 1200.0
    elif mutation == "extra":
        contract["retry"] = True
    else:
        contract["storage_backend"] = "file_v2"
    with pytest.raises(ValueError):
        check(contract)


@pytest.mark.parametrize("maximum", [0, 24 * 2**30, 32 * 2**30, float(44 * 2**30), True])
def test_fixed_44gib_budget_is_strict(bound, maximum):
    contract, _ = bound
    contract["max_device_bytes"] = maximum
    with pytest.raises(ValueError, match="storage/window"):
        check(contract)


def test_metadata_import_never_loads_torch_or_producer():
    code = (
        "import sys; sys.path.insert(0, " + repr(str(Path(f.__file__).parents[1])) + "); "
        "from sr_f6_memory_repair import full_contract; "
        "assert 'torch' not in sys.modules; assert 'sr_f6' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(f.legacy, "PROJECT", tmp_path)
    new, old = tmp_path / "new", tmp_path / "old"
    files, oldfiles = [], []
    for name in f.SOURCE_NAMES:
        relative = "src/sr_f6_memory_repair/" + name + ".py"
        p = new / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# " + name + "\n")
        ref = dict(path=relative, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        files.append(ref)
        if name in f.SOURCE_NAMES[:8]:
            target = old / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
            oldfiles.append(copy.deepcopy(ref))
    for name in ("operations.py", "launch.sbatch"):
        relative = "scripts/sr_f6_memory_full_ops/" + name
        p = new / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# fixture\n")
        files.append(dict(path=relative, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    monkeypatch.setattr(f, "__file__", str(new / "src/sr_f6_memory_repair/full_contract.py"))
    monkeypatch.setattr(f.legacy, "__file__", str(new / "src/sr_f6_memory_repair/execution.py"))
    return (
        dict(repair_source=dict(deployment=str(new), revision="f" * 40, files=files)),
        dict(repair_source=dict(deployment=str(old), revision=f.V6_REVISION, files=oldfiles)),
    )


@pytest.mark.parametrize("mutation", [None, "missing", "duplicate", "changed_old", "wrong_origin"])
def test_source_closure_and_frozen_dependencies(tmp_path, monkeypatch, mutation):
    contract, prior = source_fixture(tmp_path, monkeypatch)
    if mutation == "missing":
        contract["repair_source"]["files"].pop()
    elif mutation == "duplicate":
        contract["repair_source"]["files"].append(contract["repair_source"]["files"][0])
    elif mutation == "changed_old":
        ref = contract["repair_source"]["files"][1]
        path = Path(contract["repair_source"]["deployment"]) / ref["path"]
        path.write_text("# changed\n")
        ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif mutation == "wrong_origin":
        monkeypatch.setattr(f, "__file__", str(tmp_path / "alien.py"))
    if mutation is None:
        f._sources(contract, prior)
    else:
        with pytest.raises(ValueError):
            f._sources(contract, prior)


@pytest.mark.parametrize("watchdog", ["smoke_WATCHDOG_ATTEMPT.json", "smoke_WATCHDOG_RESULT.json"])
def test_actual_v6_watchdog_receipts_block_direct_contract(tmp_path, monkeypatch, watchdog):
    from sr_f6_memory_repair import smoke_reuse

    monkeypatch.setattr(f.legacy, "PROJECT", tmp_path)
    monkeypatch.setattr(f, "PROJECT", tmp_path)
    monkeypatch.setattr(f, "V6_OPS", tmp_path / "v6")
    prior = dict(repair_source=dict(revision=f.V6_REVISION, deployment=str(tmp_path / "old")))
    contract = f.predecessor_refs()
    proof = dict(
        status="VERIFIED_ACTUAL_TWO_ROW_REUSED_REFERENCE_SMOKE",
        admission="NOT_FULL_BATCH_ADMISSION",
    )
    contract["prior_smoke_cold_ref"] = write(f.V6_OPS / "SMOKE/COLD_VERIFICATION.json", proof)
    complete = dict(
        status="COMPLETED_STAGED_SMOKE_NOT_FULL_BATCH_ADMISSION",
        job_id="235771",
        contract_ref=contract["prior_smoke_contract_ref"],
        result_ref=contract["prior_smoke_result_ref"],
        source=dict(
            revision=f.V6_REVISION,
            deployment=prior["repair_source"]["deployment"],
            binding_sha256=f.V6_BINDING,
        ),
    )
    contract["prior_smoke_complete_ref"] = write(f.V6_OPS / "smoke_COMPLETE.json", complete)
    monkeypatch.setattr(
        smoke_reuse,
        "verify_smoke_metadata",
        lambda *a: dict(
            contract=prior, result=dict(cold_verification_ref=contract["prior_smoke_cold_ref"])
        ),
    )
    f._prior_success(contract, prior)
    (f.V6_OPS / watchdog).write_text("{}")
    with pytest.raises(ValueError, match="failed V6"):
        f._prior_success(contract, prior)


@pytest.mark.parametrize(
    "mutation",
    [None, "cold_absent", "cold_binding", "cold_rows", "failure", "evidence_source", "candidate"],
)
def test_full_result_metadata_never_loads_tensors_and_requires_cold_closure(
    tmp_path, monkeypatch, mutation
):
    monkeypatch.setattr(f.legacy, "PROJECT", tmp_path)
    monkeypatch.setattr(f, "PROJECT", tmp_path)
    monkeypatch.setattr(f, "OPS", tmp_path / "v7")
    output = f.OPS / "VALIDATION"
    output.mkdir(parents=True)
    key = "src/sr_f6_memory_repair/validation_full.py"
    source_ref = write(tmp_path / "source" / key, dict(fixture=True))
    contract = dict(
        repair_source=dict(
            revision="a" * 40,
            deployment=str(tmp_path / "source"),
            files=[dict(path=key, sha256=source_ref["sha256"])],
        ),
        producer=dict(revision="b" * 40),
        prior_smoke_result_ref=dict(path="historical", sha256="c" * 64),
    )
    monkeypatch.setattr(f, "validate_metadata", lambda *a: contract)
    evidence = dict(
        scope="ACTUAL_CUDA_STORAGE_VALIDATION",
        contract_sha256="d" * 64,
        repair_revision="a" * 40,
        producer_revision="b" * 40,
        prior_smoke_result_ref=contract["prior_smoke_result_ref"],
    )
    if mutation == "evidence_source":
        evidence["repair_revision"] = "e" * 40
    evidence_ref = write(output / "EVIDENCE.json", evidence)
    (output / "TENSORS.pt").write_bytes(b"fixture tensor bytes must never be opened")
    result = dict(
        schema=f.RESULT_SCHEMA,
        status=f.STATUS,
        contract_sha256="d" * 64,
        plan=f.PLAN,
        prior_smoke_result_ref=contract["prior_smoke_result_ref"],
        verifier_source_ref=source_ref,
        evidence_ref=evidence_ref,
        tensors_ref=dict(path=str(output / "TENSORS.pt"), sha256="f" * 64),
    )
    result_ref = write(output / "RESULT.json", result)
    cold = dict(
        schema=f.RESULT_SCHEMA,
        status=f.STATUS,
        contract_sha256="d" * 64,
        validation_ref=result_ref,
        verifier_source_ref=source_ref,
        repair_revision="a" * 40,
        producer_revision="b" * 40,
        offload_rows=128,
        compared_gradient_rows=[0],
        generated_responses=0,
        optimizer_updates=0,
        scope=f.PLAN,
    )
    if mutation == "cold_binding":
        cold["validation_ref"] = {**result_ref, "sha256": "0" * 64}
    if mutation == "cold_rows":
        cold["offload_rows"] = 127
    if mutation != "cold_absent":
        write(output / "COLD_VERIFICATION.json", cold)
    if mutation == "failure":
        write(output / "FAILURE.json", dict(status="FAILED_PRESERVED"))
    if mutation == "candidate":
        result_ref = write(output / "CANDIDATE_RESULT.json", result)
    strict = f.strict_ref

    def no_tensors(path, sha):
        assert Path(path).suffix != ".pt"
        return strict(path, sha)

    monkeypatch.setattr(f, "strict_ref", no_tensors)
    if mutation is None:
        checked = f.verify_result_metadata(f.OPS / "CONTRACT.json", "d" * 64, result_ref)
        assert checked["validation_ref"] == result_ref
        assert checked["cold_verification_ref"]["path"] == str(output / "COLD_VERIFICATION.json")
    else:
        with pytest.raises(ValueError):
            f.verify_result_metadata(f.OPS / "CONTRACT.json", "d" * 64, result_ref)
