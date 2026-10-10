"""Probe failure boundaries and CPU parity; CUDA receipt remains separately required."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/sr_f1/validate_compute_parallel.py"
SPEC = importlib.util.spec_from_file_location("srf1_compute_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC.loader.exec_module(probe)
finally:
    sys.path.pop(0)


def test_cuda_absence_cannot_claim_success_or_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "receipt.json"
    assert probe.main(["--output", str(output), "--gpu-count", "4"]) == 1
    before = output.read_bytes()
    report = json.loads(before)
    assert report["status"] == "FAILED"
    assert report["real_9b_engine_qualified"] is False
    assert report["scientific_data_read"] is False
    assert set(report["source_file_hashes"]) == set(probe.NUMERICAL_FILES)
    assert len(report["source_file_hashes"]) == 5
    with pytest.raises(FileExistsError, match="Preserve"):
        probe.main(["--output", str(output)])
    assert output.read_bytes() == before


def execution_rows(count=4):
    return [
        dict(
            rank=i % count,
            example_index=i,
            uuid=f"device-{i % count}",
            started_ns=0,
            ended_ns=10,
            cuda_ms=1.0,
            peak_allocated_bytes=100,
        )
        for i in range(128)
    ]


def test_overlap_and_every_gpu_are_required():
    rows = execution_rows()
    assert probe.verify_worker_execution(rows, 4)["actual_compute_gpu_count"] == 4
    rows[0]["uuid"] = "device-1"
    # Later records cannot hide a changed rank/device identity.
    with pytest.raises(AssertionError):
        probe.verify_worker_execution(rows, 4)


def test_no_computation_and_no_overlap_are_rejected():
    rows = execution_rows()
    rows[0]["cuda_ms"] = 0
    with pytest.raises(AssertionError, match="CUDA elapsed"):
        probe.verify_worker_execution(rows, 4)
    rows = execution_rows()
    for index, row in enumerate(rows):
        row["started_ns"], row["ended_ns"] = index * 20, index * 20 + 10
    with pytest.raises(AssertionError, match="overlap"):
        probe.verify_worker_execution(rows, 4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_hybrid_full_updates_order_adam_rng_on_cpu(tmp_path, dtype):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = probe.run_case(tmp_path, dtype, 1, device="cpu", long_tokens=33)
    finally:
        torch.set_num_threads(previous)
    assert len(result["updates"]) == 2
    assert all(x["exact_ordered_gradient_accumulation"] for x in result["updates"])
    assert result["long_history"]["both_gradients_bitwise"]
    assert result["long_history"]["full_history_verified"]


def test_native_768_token_full_history_double_backward_on_cpu(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with probe.tiny_runtime(tmp_path, torch.bfloat16, "cpu") as (runtime, prepared):
            result = probe.long_history(runtime, prepared, 768)
    finally:
        torch.set_num_threads(previous)
    assert result == dict(
        tokens=768, logits_bitwise=True, both_gradients_bitwise=True, full_history_verified=True
    )


def test_mismatch_retains_failed_receipt_and_partial_cases(tmp_path, monkeypatch):
    def fail(_directory, _gpu_count, report):
        report["cases"] = [dict(dtype="float32", status="earlier_success")]
        raise AssertionError("Adam differs")

    monkeypatch.setattr(probe, "run_validation", fail)
    output = tmp_path / "failed.json"
    assert probe.main(["--output", str(output)]) == 1
    report = json.loads(output.read_bytes())
    assert report["status"] == "FAILED"
    assert report["error"] == "Adam differs"
    assert report["cases"] == [dict(dtype="float32", status="earlier_success")]
    assert not report["numerical_equivalence"]
