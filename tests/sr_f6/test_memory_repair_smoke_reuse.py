"""Two current candidate calls; historical reference payloads are never executions."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from mm_core.training import capture_rng, state_hash
from sr_f6_memory_repair import smoke as old
from sr_f6_memory_repair import smoke_reuse as s

_spec = importlib.util.spec_from_file_location(
    "reuse_fixture", Path(__file__).with_name("test_memory_repair_validation.py")
)
f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f)


def setup(tmp_path):
    runtime, optimizer, _, rows, credits, originals = f.setup(tmp_path)
    prior_output = tmp_path / "PRIOR"
    prior_output.mkdir()
    evidence, tensors = old.measure(runtime, rows, credits, originals, prior_output, fixture=True)
    evidence["identity"] = copy.deepcopy(runtime.identity)
    runtime.calls.clear()
    output = tmp_path / "NEW"
    output.mkdir()
    return runtime, optimizer, rows, credits, originals, evidence, tensors, output


def measured(tmp_path, **kwargs):
    runtime, optimizer, rows, credits, originals, prior, old_tensors, output = setup(tmp_path)
    evidence, tensors = s.measure(
        runtime, rows, credits, originals, output, prior, old_tensors, fixture=True, **kwargs
    )
    return runtime, optimizer, rows, originals, prior, old_tensors, evidence, tensors


def test_only_two_current_candidates_and_exact_restoration(tmp_path):
    runtime, optimizer, rows, originals, prior, old_tensors, evidence, tensors = measured(tmp_path)
    assert runtime.calls == [rows[i]["row_id"] for i in (0, 26)]
    assert evidence["measurement_order"] == ["row0_candidate", "row26_candidate"]
    assert set(tensors) == set(evidence["measurement_order"])
    assert evidence["counts"] == dict(
        reference_forwards=2,
        policy_forwards=2,
        backward_calls=2,
        diagnostic_gradient_calls=0,
        generated_responses=0,
        optimizer_updates=0,
    )
    assert evidence["historical_per_row_rng_state"] == "NOT_RECORDED"
    assert evidence["global_initial_rng_match"] and evidence["candidate_rng_unchanged"]
    assert all(
        v["storage"]["max_device_bytes"] == 44 * 2**30 for v in evidence["measurements"].values()
    )
    assert s.recompute(evidence, tensors, originals, prior, old_tensors) == evidence["comparisons"]
    assert runtime.model.weight.grad is None and optimizer.state == {}
    assert evidence["restoration"]["rng_restored"]
    assert evidence["comparisons"]["admission"] == "NOT_FULL_BATCH_ADMISSION"


@pytest.mark.parametrize(
    "kind",
    [
        "reference_gradient",
        "reference_trace",
        "reference_loss",
        "reference_shape",
        "missing_reference",
        "missing_archive",
        "model_identity",
        "initial_rng",
    ],
)
def test_prior_or_initial_state_tamper_prevents_every_candidate(tmp_path, kind):
    runtime, _, rows, credits, originals, prior, old_tensors, output = setup(tmp_path)
    if kind == "reference_gradient":
        old_tensors["row26_allspill"]["weight"] += 1
    if kind == "reference_trace":
        prior["measurements"]["row0_eager"]["trace"]["train_logprobs"][0] += 1
    if kind == "reference_loss":
        prior["measurements"]["row0_eager"]["loss"]["loss"] += 1
    if kind == "reference_shape":
        old_tensors["row0_eager"]["weight"] = old_tensors["row0_eager"]["weight"].flatten()
    if kind == "missing_reference":
        del old_tensors["row0_eager"]
    if kind == "missing_archive":
        originals = originals[:2]
    if kind == "model_identity":
        prior["identity"]["reference_hash"] = "bad"
    if kind == "initial_rng":
        torch.rand(1)
    rng = state_hash(capture_rng())
    with pytest.raises((PermissionError, KeyError)):
        s.measure(runtime, rows, credits, originals, output, prior, old_tensors, fixture=True)
    assert runtime.calls == []
    assert state_hash(capture_rng()) == rng


@pytest.mark.parametrize(
    "kind",
    [
        "gradient",
        "forward",
        "count",
        "historical_execution",
        "budget",
        "plan",
        "restore",
        "rng_claim",
        "rng_after",
        "seconds",
    ],
)
def test_new_cold_recompute_rejects_tamper(tmp_path, kind):
    _, _, _, originals, prior, old_tensors, evidence, tensors = measured(tmp_path)
    if kind == "gradient":
        tensors["row26_candidate"]["weight"] += 1
    if kind == "forward":
        evidence["measurements"]["row0_candidate"]["trace"]["train_logprobs"][0] += 1
    if kind == "count":
        evidence["counts"]["reference_forwards"] = 4
    if kind == "historical_execution":
        evidence["measurements"]["row0_eager"] = prior["measurements"]["row0_eager"]
    if kind == "budget":
        evidence["measurements"]["row0_candidate"]["storage"]["max_device_bytes"] = 24 * 2**30
    if kind == "plan":
        evidence["plan"]["loss_denominator"] = 2
    if kind == "restore":
        evidence["restoration"]["rng_restored"] = False
    if kind == "rng_claim":
        evidence["historical_per_row_rng_state"] = "REPLAYED"
    if kind == "rng_after":
        evidence["measurements"]["row0_candidate"]["rng_after_sha256"] = "bad"
    if kind == "seconds":
        evidence["measurements"]["row0_candidate"]["seconds"] = 121
    with pytest.raises(PermissionError):
        s.recompute(evidence, tensors, originals, prior, old_tensors)


def test_row0_gate_failure_blocks_long_and_preserves_partial(tmp_path, monkeypatch):
    runtime, optimizer, rows, credits, originals, prior, old_tensors, output = setup(tmp_path)
    before = state_hash(runtime.model.state_dict())
    rng = state_hash(capture_rng())
    original_pair = s._pair

    def reject(*args):
        original_pair(*args)
        raise PermissionError("ordinary gate rejected")

    monkeypatch.setattr(s, "_pair", reject)
    with pytest.raises(PermissionError, match="ordinary gate") as caught:
        s.measure(runtime, rows, credits, originals, output, prior, old_tensors, fixture=True)
    assert runtime.calls == [rows[0]["row_id"]]
    assert state_hash(runtime.model.state_dict()) == before
    assert state_hash(capture_rng()) == rng
    assert caught.value.smoke_partial[0]["restoration"]["rng_restored"]
    assert optimizer.state == {}


def test_candidate_rng_advancement_stops_without_resetting_into_false_success(
    tmp_path, monkeypatch
):
    runtime, _, rows, credits, originals, prior, old_tensors, output = setup(tmp_path)
    forward = runtime.batch_sequence_forward

    def consumes(*args, **kwargs):
        torch.rand(1)
        return forward(*args, **kwargs)

    monkeypatch.setattr(runtime, "batch_sequence_forward", consumes)
    rng = state_hash(capture_rng())
    with pytest.raises(PermissionError, match="RNG advanced") as caught:
        s.measure(runtime, rows, credits, originals, output, prior, old_tensors, fixture=True)
    assert runtime.calls == [rows[0]["row_id"]]
    assert state_hash(capture_rng()) == rng
    assert caught.value.smoke_partial[0]["candidate_rng_unchanged"] is False


def test_ordinary_timeout_prevents_long_candidate(tmp_path):
    runtime, _, rows, credits, originals, prior, old_tensors, output = setup(tmp_path)
    now = [0.0]

    def clock():
        now[0] += 25
        return now[0]

    with pytest.raises(PermissionError, match="technical time"):
        s.measure(
            runtime, rows, credits, originals, output, prior, old_tensors, fixture=True, clock=clock
        )
    assert runtime.calls == [rows[0]["row_id"]]


def test_headroom_gate_unchanged(tmp_path):
    _, _, _, originals, prior, old_tensors, evidence, tensors = measured(tmp_path)
    evidence["scope"] = "ACTUAL_CUDA_ONLY"
    evidence["measurements"]["row0_candidate"]["cuda"]["finished"] = dict(
        max_allocated=9 * 2**30, total=10 * 2**30
    )
    with pytest.raises(PermissionError, match="headroom"):
        s._pair(evidence, tensors, originals, 0, prior, old_tensors)


def test_sorted_json_cold_comparison_is_stable(tmp_path):
    _, _, _, originals, prior, old_tensors, evidence, tensors = measured(tmp_path)
    loaded = json.loads(json.dumps(evidence, sort_keys=True))
    assert s.recompute(loaded, tensors, originals, prior, old_tensors) == evidence["comparisons"]


def contract_fixture(tmp_path, monkeypatch):
    from sr_f6_memory_repair import execution

    monkeypatch.setattr(execution, "PROJECT", tmp_path)
    monkeypatch.setattr(s, "PROJECT", tmp_path)
    monkeypatch.setattr(s, "PRIOR_ROOT", tmp_path / "V5")
    root = s.PRIOR_ROOT / "SMOKE"
    root.mkdir(parents=True)
    evidence = root / "EVIDENCE.json"
    proof = root / "COLD_VERIFICATION.json"
    evidence.write_text(json.dumps(dict(comparisons={"checked": True})))
    proof.write_text(json.dumps(dict(comparisons={"checked": True})))
    prior = {k: {"fixed": k} for k in s.INHERITED_KEYS}
    prior["experiment_root"] = str(tmp_path / "experiment")
    prior["repair_source"] = dict(revision="a" * 40)
    result = dict(evidence_ref=s._ref(evidence), cold_verification_ref=s._ref(proof))
    monkeypatch.setattr(
        old, "verify_smoke_metadata", lambda *a, **kw: dict(contract=prior, result=result)
    )
    contract = dict(
        prior,
        schema=s.SCHEMA,
        smoke_plan=copy.deepcopy(s.SMOKE_PLAN),
        max_host_bytes=s.HOST_BYTES,
        max_disk_bytes=s.DISK_BYTES,
        max_device_bytes=s.DEVICE_BYTES,
        storage_backend="arena_v3",
        prior_smoke_contract_ref=dict(
            path=str(s.PRIOR_ROOT / "CONTRACT.json"), sha256=s.PRIOR_CONTRACT_SHA
        ),
        prior_smoke_result_ref=dict(path=str(root / "RESULT.json"), sha256=s.PRIOR_RESULT_SHA),
    )
    return contract, prior


@pytest.mark.parametrize(
    "kind",
    [
        "missing_prior",
        "wrong_contract_sha",
        "wrong_result_path",
        "budget",
        "plan",
        "scientific_binding",
    ],
)
def test_contract_rejects_unbound_or_changed_reference(tmp_path, monkeypatch, kind):
    contract, _ = contract_fixture(tmp_path, monkeypatch)
    if kind == "missing_prior":
        del contract["prior_smoke_result_ref"]
    if kind == "wrong_contract_sha":
        contract["prior_smoke_contract_ref"]["sha256"] = "0" * 64
    if kind == "wrong_result_path":
        contract["prior_smoke_result_ref"]["path"] = str(tmp_path / "other.json")
    if kind == "budget":
        contract["max_device_bytes"] = 48 * 2**30
    if kind == "plan":
        contract["smoke_plan"]["candidate_max_seconds"] = 121
    if kind == "scientific_binding":
        contract["original_ready_ref"] = {"changed": True}
    path = tmp_path / "CONTRACT.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(PermissionError):
        s.validate_contract(path, s._ref(path)["sha256"], source=False)


def test_contract_metadata_does_not_load_tensors(tmp_path, monkeypatch):
    contract, _ = contract_fixture(tmp_path, monkeypatch)
    path = tmp_path / "CONTRACT.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("Metadata loaded tensor payload")
    )
    assert s.validate_contract(path, s._ref(path)["sha256"], source=False) == contract


def test_call_ledger_cannot_include_historical_calls():
    ledger = [
        dict(kind="extra_forward_sequences", count=1, metadata=dict(purpose=p))
        for p in ("training_reference", "training_policy")
    ] * 2
    s._verify_ledger(ledger)
    with pytest.raises(PermissionError):
        s._verify_ledger(ledger * 2)
    ledger[0] = dict(ledger[0], metadata=dict(purpose="training_policy"))
    with pytest.raises(PermissionError):
        s._verify_ledger(ledger)


def test_import_keeps_login_tensor_free():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import runpy,sys;runpy.run_path({str(Path(s.__file__))!r});"
            "assert not any(k.startswith(('torch','sr_f6','mm_core')) for k in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_prior_payload_hash_and_saved_cold_proof_both_required(tmp_path, monkeypatch):
    from sr_f6_memory_repair import execution

    _, _, _, _, originals, prior, tensors, output = setup(tmp_path)
    monkeypatch.setattr(execution, "PROJECT", tmp_path)
    monkeypatch.setattr(s, "PROJECT", tmp_path)
    payload = output / "PRIOR_TENSORS.pt"
    torch.save(tensors, payload)
    evidence_path = output / "PRIOR_EVIDENCE.json"
    evidence_path.write_text(json.dumps(prior))
    proof_path = output / "PROOF.json"
    proof_path.write_text(json.dumps({"checked": True}))
    result_path = output / "RESULT.json"
    result_path.write_text(
        json.dumps(
            dict(
                tensors_ref=s._ref(payload),
                evidence_ref=s._ref(evidence_path),
                cold_verification_ref=s._ref(proof_path),
            )
        )
    )
    contract = dict(prior_smoke_result_ref=s._ref(result_path))
    monkeypatch.setattr(old, "verify_smoke", lambda *args: {"checked": True})
    checked, values = s._load_prior(contract, originals)
    assert checked == prior
    torch.testing.assert_close(values, tensors, rtol=0, atol=0)
    monkeypatch.setattr(old, "verify_smoke", lambda *args: {"checked": False})
    with pytest.raises(PermissionError, match="cold proof"):
        s._load_prior(contract, originals)
    monkeypatch.setattr(old, "verify_smoke", lambda *args: {"checked": True})
    with payload.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash differs"):
        s._load_prior(contract, originals)


def test_cold_reader_recomputes_complete_candidate_tensors(tmp_path, monkeypatch):
    from sr_f6_memory_repair import execution, validation

    _, _, rows, originals, prior, old_tensors, evidence, tensors = measured(tmp_path)
    output = tmp_path / "COLD"
    output.mkdir()
    monkeypatch.setattr(execution, "PROJECT", tmp_path)
    monkeypatch.setattr(s, "PROJECT", tmp_path)
    monkeypatch.setattr(s, "_load_prior", lambda *args: (prior, old_tensors))
    monkeypatch.setattr(s, "_identity_match", lambda *args: None)
    monkeypatch.setattr(old, "_verify_cuda", lambda *args: None)
    monkeypatch.setattr(validation, "verify_storage", lambda *args, **kwargs: None)
    monkeypatch.setattr(validation, "_originals", lambda *args: ({}, rows, [], originals))
    evidence["scope"] = "ACTUAL_CUDA_ONLY"
    evidence["identity"] = {"hardware": {}}
    evidence["ledger"] = [
        dict(
            kind="extra_forward_sequences",
            count=1,
            metadata=dict(purpose=purpose, completion_tokens=len(rows[row]["tokens"])),
        )
        for row in (0, 26)
        for purpose in ("training_reference", "training_policy")
    ]
    for measurement in evidence["measurements"].values():
        measurement["cuda"]["finished"] = dict(max_allocated=1, total=4 * 2**30)
    evidence["comparisons"] = s.recompute(evidence, tensors, originals, prior, old_tensors)
    evidence_path, tensor_path = output / "EVIDENCE.json", output / "TENSORS.pt"
    evidence_path.write_text(json.dumps(evidence))
    torch.save(tensors, tensor_path)
    result = dict(evidence_ref=s._ref(evidence_path), tensors_ref=s._ref(tensor_path))
    contract = dict(prior_smoke_result_ref={"path": "historical", "sha256": "f" * 64})
    proof = s._cold_evidence(contract, result, output, "a" * 64)
    assert proof["status"] == "VERIFIED_ACTUAL_TWO_ROW_REUSED_REFERENCE_SMOKE"
    tensors["row26_candidate"]["weight"] += 1
    torch.save(tensors, tensor_path)
    result["tensors_ref"] = s._ref(tensor_path)
    with pytest.raises(PermissionError, match="numerical gate"):
        s._cold_evidence(contract, result, output, "a" * 64)


def test_historical_hardware_uuid_is_not_current_measurement_identity():
    base = dict(
        model="same",
        technical_consumer=old.RESULT_SCHEMA,
        bound_files_before={},
        bound_files_after={},
        hardware=dict(
            cuda_visible_device_count=1,
            cuda_name="PRO 6000",
            cuda_total_memory_bytes=100,
            compute_capability=[12, 0],
            cuda_runtime="13",
            driver_version="595",
            cuda_uuid="old",
        ),
    )
    current = copy.deepcopy(base)
    current["technical_consumer"] = s.RESULT_SCHEMA
    current["hardware"]["cuda_uuid"] = "new"
    s._identity_match(current, base)
    current["hardware"]["driver_version"] = "changed"
    with pytest.raises(PermissionError, match="compatibility"):
        s._identity_match(current, base)
