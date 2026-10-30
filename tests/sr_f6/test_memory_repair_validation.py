"""Storage validation regression fixtures; never actual model or gate evidence."""

import copy
import importlib.util
from pathlib import Path

import pytest

from mm_core.training import capture_rng, state_hash
from sr_f6_memory_repair import validation as v

_spec = importlib.util.spec_from_file_location(
    "memory_validation_fixture", Path(__file__).with_name("test_training.py")
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def setup(tmp_path):
    runtime, optimizer, scheduler, records, credits = _fixture.batch(tmp_path)
    originals = [
        dict(
            row_id=r["row_id"],
            qid=r["qid"],
            tokens=r["tokens"],
            sampler_logprobs=r["sampler_logprobs"],
            train_logprobs=[-2.0] * len(r["tokens"]),
            reference_logprobs=[-2.0] * len(r["tokens"]),
            token_path_sha256=r["token_path_sha256"],
        )
        for r in records[:27]
    ]
    return runtime, optimizer, scheduler, records, credits, originals


def test_fixed_128_rows_diagnostics_restore_and_no_update(tmp_path):
    runtime, optimizer, _scheduler, rows, credits, originals = setup(tmp_path)
    rng = state_hash(capture_rng())
    policy = state_hash(runtime.model.state_dict())
    evidence, tensors = v.measure_fixture(runtime, rows, credits, originals)
    assert evidence["scope"] == "SYNTHETIC_CPU_ONLY"
    assert runtime.calls == [rows[0]["row_id"]] + [r["row_id"] for r in rows]
    assert evidence["counts"]["diagnostic_gradient_calls"] == 192
    assert evidence["counts"]["backward_calls"] == 129
    assert evidence["counts"]["optimizer_updates"] == 0
    assert not optimizer.state
    assert state_hash(runtime.model.state_dict()) == policy
    assert state_hash(capture_rng()) == rng
    assert runtime.model.weight.grad is None
    assert v.recompute(evidence, tensors, originals) == evidence["comparisons"]
    assert not v.PLAN["full_128_paired_gradient_equivalence"]


@pytest.mark.parametrize(
    "mutation", ["row", "vector", "gradient", "count", "restore", "plan", "shape"]
)
def test_cold_recompute_rejects_tampered_actual_measurements(tmp_path, mutation):
    runtime, _, _, rows, credits, originals = setup(tmp_path)
    evidence, tensors = v.measure_fixture(runtime, rows, credits, originals)
    evidence = copy.deepcopy(evidence)
    if mutation == "row":
        evidence["traces"][2]["tokens"][0] += 1
    if mutation == "vector":
        evidence["traces"][26]["train_logprobs"][0] += 1
    if mutation == "gradient":
        tensors["offload_row0"]["weight"] += 1
    if mutation == "shape":
        tensors["accumulated"]["weight"] = tensors["accumulated"]["weight"].flatten()
    if mutation == "count":
        evidence["counts"]["backward_calls"] -= 1
    if mutation == "restore":
        evidence["restoration"]["rng_restored"] = False
    if mutation == "plan":
        evidence["plan"]["baseline_rows"] = [1]
    with pytest.raises(PermissionError):
        v.recompute(evidence, tensors, originals)


def test_original_failure_row_is_not_skipped_and_state_restores(tmp_path):
    runtime, optimizer, _, rows, credits, originals = setup(tmp_path)
    runtime.fail_row = rows[26]["row_id"]
    before = state_hash(runtime.model.state_dict())
    rng = state_hash(capture_rng())
    with pytest.raises(RuntimeError, match="injected"):
        v.measure_fixture(runtime, rows, credits, originals)
    assert len(runtime.calls) == 28
    assert state_hash(runtime.model.state_dict()) == before
    assert state_hash(capture_rng()) == rng
    assert runtime.model.weight.grad is None
    assert not optimizer.state


def test_partial_original_archive_blocks_before_model_calls(tmp_path):
    runtime, _, _, rows, credits, originals = setup(tmp_path)
    with pytest.raises(PermissionError, match="27"):
        v.measure_fixture(runtime, rows, credits, originals[:-1])
    assert not runtime.calls


def test_changed_original_token_binding_rejected_before_forward(tmp_path):
    runtime, _, _, rows, credits, originals = setup(tmp_path)
    rows[127]["tokens"][0] += 1
    with pytest.raises(PermissionError):
        v.measure_fixture(runtime, rows, credits, originals)
    assert not runtime.calls


def test_production_cold_verifier_rejects_complete_synthetic_receipt(tmp_path, monkeypatch):
    import json

    import torch

    from sr_f6_memory_repair import execution

    runtime, _, _, rows, credits, originals = setup(tmp_path)
    evidence, tensors = v.measure_fixture(runtime, rows, credits, originals)
    output = tmp_path / "VALIDATION"
    output.mkdir()
    (output / "EVIDENCE.json").write_text(json.dumps(evidence))
    torch.save(tensors, output / "TENSORS.pt")
    result = dict(
        schema=v.SCHEMA,
        status="VERIFIED_ACTUAL_STORAGE_REPAIR_BATCH",
        plan=v.PLAN,
        contract_sha256="1" * 64,
        evidence_ref=v._ref(output / "EVIDENCE.json"),
        tensors_ref=v._ref(output / "TENSORS.pt"),
        verifier_source_ref=v._ref(Path(v.__file__)),
    )
    (output / "RESULT.json").write_text(json.dumps(result))
    monkeypatch.setattr(execution, "verify_contract", lambda *_: {})
    monkeypatch.setattr(execution, "strict_ref", lambda path, sha: Path(path).read_bytes())
    monkeypatch.setattr(v, "_originals", lambda _: ({}, rows, credits, originals))
    with pytest.raises(PermissionError, match="Synthetic"):
        v.verify_validation(tmp_path / "contract.json", "1" * 64, v._ref(output / "RESULT.json"))


@pytest.mark.parametrize("disk_backend", ["file_v2", "arena_v3"])
def test_hybrid_validation_uses_owned_spill_and_restores_exact_gradients(tmp_path, disk_backend):
    runtime, optimizer, _, rows, credits, originals = setup(tmp_path)
    before = state_hash(runtime.model.state_dict())
    scratch = tmp_path / "STORAGE"
    evidence, tensors = v.measure_fixture(
        runtime,
        rows,
        credits,
        originals,
        max_host_bytes=1,
        max_disk_bytes=1024**2,
        scratch_directory=scratch,
        disk_backend=disk_backend,
    )
    assert evidence["offload"]["disk_bytes_written"] > 0
    assert evidence["offload"]["disk_bytes_read"] > 0
    assert evidence["offload"]["peak_disk_bytes"] <= 1024**2
    assert evidence["offload"]["peak_host_bytes"] <= 1
    assert not list(scratch.iterdir())
    assert state_hash(runtime.model.state_dict()) == before
    assert not optimizer.state
    assert v.recompute(evidence, tensors, originals) == evidence["comparisons"]
    for mutation in (
        "host",
        "disk",
        "staging",
        "cleanup",
        "read",
        "path",
        "live",
        "failure",
        "no_io",
        "peak_zero",
        "short_write",
    ):
        damaged = copy.deepcopy(evidence["offload"])
        if mutation == "host":
            damaged["peak_host_bytes"] = 2
        if mutation == "disk":
            damaged["peak_disk_bytes"] = 1024**2 + 1
        if mutation == "staging":
            damaged["peak_staging_host_bytes"] = 64 * 1024**2 + 1
        if mutation == "cleanup":
            damaged["cleanup_errors"] = ["failed"]
        if mutation == "read":
            damaged["disk_bytes_read"] = 0
        if mutation == "no_io":
            damaged["disk_bytes_written"] = damaged["disk_bytes_read"] = 0
        if mutation == "peak_zero":
            damaged["peak_disk_bytes"] = 0
        if mutation == "short_write":
            damaged["disk_bytes_written"] = damaged["peak_disk_bytes"] - 1
        if mutation == "path":
            damaged["scratch_directory"] = str(tmp_path)
        if mutation == "live":
            damaged["live_disk_bytes"] = 1
        if mutation == "failure":
            damaged["storage_failures"] = ["failed"]
        with pytest.raises(PermissionError):
            v.verify_storage(damaged, scratch, 1, 1024**2, fixture=True, disk_backend=disk_backend)

    if disk_backend == "arena_v3":
        for key, value in (
            ("arena_physical_bytes", 1),
            ("peak_arena_physical_bytes", 1024**2 + 1),
            ("arena_deleted_count", -1),
            ("arena_fsync_count", -1),
            ("arena_open_count", -1),
            ("arena_read_calls", 0),
            ("arena_write_calls", 0),
            ("disk_backend", "file_v2"),
            ("schema", "srf6-saved-activation-hybrid-v2"),
        ):
            damaged = dict(evidence["offload"], **{key: value})
            with pytest.raises(PermissionError):
                v.verify_storage(
                    damaged, scratch, 1, 1024**2, fixture=True, disk_backend="arena_v3"
                )
        with pytest.raises(PermissionError):
            v.verify_storage(evidence["offload"], scratch, 1, 1024**2, fixture=True)


def test_cold_hybrid_scratch_rejects_leftover_file(tmp_path):
    runtime, _, _, rows, credits, originals = setup(tmp_path)
    scratch = tmp_path / "STORAGE"
    evidence, _ = v.measure_fixture(
        runtime,
        rows,
        credits,
        originals,
        max_host_bytes=1,
        max_disk_bytes=1024**2,
        scratch_directory=scratch,
    )
    (scratch / "unclosed").write_bytes(b"preserve")
    with pytest.raises(PermissionError, match="payload remains"):
        v.verify_storage(evidence["offload"], scratch, 1, 1024**2, fixture=True)
    assert (scratch / "unclosed").read_bytes() == b"preserve"
