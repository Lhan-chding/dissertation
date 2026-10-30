"""V4 CPU-only fixtures: no actual CUDA or scientific certification is produced."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from mm_core.training import capture_rng, state_hash
from sr_f6_memory_repair import validation as v

_spec = importlib.util.spec_from_file_location(
    "v4_fixture", Path(__file__).with_name("test_memory_repair_validation.py")
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


class Clock:
    def __init__(self, delta=1):
        self.value, self.delta = 100.0, delta

    def __call__(self):
        self.value += self.delta
        return self.value


def run(tmp_path, clock=None, **kwargs):
    runtime, optimizer, _, rows, credits, originals = _fixture.setup(tmp_path)
    evidence, tensors = v.measure_fixture(
        runtime,
        rows,
        credits,
        originals,
        max_device_bytes=1024,
        max_host_bytes=128,
        max_disk_bytes=1024**2,
        scratch_directory=tmp_path / "STORAGE",
        disk_backend="arena_v3",
        fixture_clock=clock or Clock(),
        **kwargs,
    )
    return runtime, optimizer, rows, evidence, tensors, originals


def test_v4_prefix_once_then_same_process_continues_and_cpu_is_not_cuda(tmp_path):
    runtime, optimizer, rows, evidence, tensors, originals = run(tmp_path)
    assert runtime.calls == [rows[0]["row_id"]] + [r["row_id"] for r in rows]
    assert evidence["technical_window"]["total_row_seconds"] == 20
    assert evidence["technical_window"]["status"] == "CONTINUE_SAME_PROCESS"
    assert evidence["counts"]["diagnostic_gradient_calls"] == 192
    assert not optimizer.state
    assert evidence["plan"] == v.PLAN
    observer = evidence["cuda_observations"]
    assert observer["identity"]["scope"] == "SYNTHETIC_CPU_OBSERVATIONS"
    assert all("memory_allocated" not in row for row in observer["observations"])
    with pytest.raises(PermissionError, match="identity"):
        v.verify_cuda_observations(evidence, {}, evidence["diagnostic_indices"])
    v.verify_device_storage(evidence["offload"], 1024)
    assert v.recompute(evidence, tensors, originals) == evidence["comparisons"]


def test_slow_four_row_window_raises_partial_restores_and_never_executes_row_four(tmp_path):
    runtime, optimizer, _, rows, credits, originals = _fixture.setup(tmp_path)
    before, rng = state_hash(runtime.model.state_dict()), state_hash(capture_rng())
    with pytest.raises(v.TechnicalWindowStop) as failure:
        v.measure_fixture(
            runtime, rows, credits, originals, max_device_bytes=1024, fixture_clock=Clock(50)
        )
    evidence, _ = failure.value.validation_partial
    assert runtime.calls == [rows[0]["row_id"]] + [r["row_id"] for r in rows[:4]]
    assert evidence["counts"]["backward_calls"] == 5
    assert evidence["counts"]["diagnostic_gradient_calls"] == 12
    assert evidence["technical_window"]["total_row_seconds"] == 1000
    assert evidence["technical_window"]["status"] == "TECHNICAL_STOP_NO_SCIENTIFIC_CONCLUSION"
    assert "scope" not in evidence and "comparisons" not in evidence
    assert evidence["restoration"]["adapter_restored"]
    assert state_hash(runtime.model.state_dict()) == before and state_hash(capture_rng()) == rng
    assert not optimizer.state and runtime.model.weight.grad is None


def test_window_fixed_exact_boundary_and_duplicate_rows_rejected():
    rows = []
    threshold = v.VALIDATION_WINDOW["max_total_row_seconds"]
    for i in range(4):
        rows += [
            dict(stage="row_forward", row_index=i, sequence=2 * i, monotonic_seconds=0.0),
            dict(
                stage="row_finished",
                row_index=i,
                sequence=2 * i + 1,
                monotonic_seconds=threshold / 4,
            ),
        ]
    assert v.window_measurement(rows, list(range(64)))["status"] == "CONTINUE_SAME_PROCESS"
    rows[-1]["monotonic_seconds"] += 0.001
    assert v.window_measurement(rows, list(range(64)))["status"].startswith("TECHNICAL_STOP")
    with pytest.raises(PermissionError, match="exactly once"):
        v.window_measurement([*rows, rows[0]], list(range(64)))
    with pytest.raises(PermissionError, match="diagnostic"):
        v.window_measurement(rows, list(range(4, 68)))


def test_default_zero_retains_legacy_evidence_and_schema(tmp_path):
    runtime, _, _, rows, credits, originals = _fixture.setup(tmp_path)
    evidence, _ = v.measure_fixture(runtime, rows, credits, originals, max_device_bytes=0)
    assert "technical_window" not in evidence and "cuda_observations" not in evidence
    assert "max_device_bytes" not in evidence["offload"]
    assert v.validation_schema({}) == v.SCHEMA
    assert (
        v.validation_schema(
            dict(max_device_bytes=v.MAX_DEVICE_BYTES, validation_window=v.VALIDATION_WINDOW)
        )
        == v.V4_SCHEMA
    )
    for bad in (True, 1024):
        with pytest.raises(PermissionError):
            v.validation_schema(dict(max_device_bytes=bad, validation_window=v.VALIDATION_WINDOW))
    with pytest.raises(PermissionError):
        v.validation_schema(
            dict(
                max_device_bytes=v.MAX_DEVICE_BYTES,
                validation_window={**v.VALIDATION_WINDOW, "max_total_row_seconds": 1200},
            )
        )


def test_device_cold_verifier_allows_live_external_storage_but_no_saved_views(tmp_path):
    _, _, _, evidence, _, _ = run(tmp_path)
    summary = copy.deepcopy(evidence["offload"])
    # Retained external aliases are conservative charges, not live saved hooks.
    summary.update(
        live_device_bytes=8,
        peak_device_bytes=1024,
        device_storage_count=1,
        device_pending_storage_count=1,
        peak_device_storage_count=2,
        device_active_view_count=0,
    )
    v.verify_device_storage(summary, 1024)
    for key, value in (
        ("device_active_view_count", 1),
        ("peak_device_bytes", 1025),
        ("max_device_bytes", 2048),
        ("device_resident_unpack_count", 0),
        ("device_storage_count", 0),
        ("device_accounting_release", "python_wrapper_only"),
    ):
        bad = {**summary, key: value}
        with pytest.raises(PermissionError):
            v.verify_device_storage(bad, 1024)


def actual_shaped_observations(evidence):
    """Construct dictionary fixtures to exercise cold validation, never emit production receipts."""
    evidence = copy.deepcopy(evidence)
    identity = evidence["cuda_observations"]["identity"]
    identity.update(
        scope="ACTUAL_CUDA_PROCESS_OBSERVATIONS",
        cuda_uuid="fixture-id",
        cuda_name="fixture PRO 6000",
        cuda_total_memory_bytes=10000,
        hostname="CPU_TEST",
    )
    for row in evidence["cuda_observations"]["observations"]:
        i = row["sequence"]
        row.update(
            memory_allocated=10,
            memory_reserved=20,
            max_memory_allocated=100 + i,
            max_memory_reserved=200 + i,
            free_device_bytes=5000,
            total_device_bytes=10000,
        )
    return evidence, {
        k: identity[k] for k in ("cuda_uuid", "cuda_name", "cuda_total_memory_bytes", "hostname")
    }


@pytest.mark.parametrize(
    "mutation", ["none", "reset", "drop", "reorder", "uuid", "peak", "free", "timestamp", "window"]
)
def test_cold_cuda_closure_rejects_missing_or_inconsistent_observations(tmp_path, mutation):
    _, _, _, evidence, _, _ = run(tmp_path)
    evidence, hardware = actual_shaped_observations(evidence)
    observer = evidence["cuda_observations"]
    if mutation == "none":
        v.verify_cuda_observations(evidence, hardware, evidence["diagnostic_indices"])
        return
    if mutation == "reset":
        observer["identity"]["peak_reset_count"] = 1
    if mutation == "drop":
        observer["observations"].pop(4)
    if mutation == "reorder":
        observer["observations"][4:6] = reversed(observer["observations"][4:6])
    if mutation == "uuid":
        hardware["cuda_uuid"] = "other"
    if mutation == "peak":
        observer["observations"][9]["max_memory_allocated"] = 50
    if mutation == "free":
        observer["observations"][9]["free_device_bytes"] = 10001
    if mutation == "timestamp":
        observer["observations"][9]["monotonic_seconds"] = 0
    if mutation == "window":
        evidence["technical_window"]["total_row_seconds"] = 0
    with pytest.raises(PermissionError):
        v.verify_cuda_observations(evidence, hardware, evidence["diagnostic_indices"])


def test_observer_failure_during_restoration_still_preserves_partial_and_state(
    tmp_path, monkeypatch
):
    runtime, optimizer, _, rows, credits, originals = _fixture.setup(tmp_path)
    original = v.CudaObservations.observe

    def broken(self, stage, row=None):
        if stage == "restoration":
            raise RuntimeError("CUDA observation unavailable")
        return original(self, stage, row)

    monkeypatch.setattr(v.CudaObservations, "observe", broken)
    before = state_hash(runtime.model.state_dict())
    with pytest.raises(RuntimeError, match="observation unavailable") as failure:
        v.measure_fixture(
            runtime, rows, credits, originals, max_device_bytes=1024, fixture_clock=Clock()
        )
    evidence, tensors = failure.value.validation_partial
    assert evidence["restoration"]["adapter_restored"] and "baseline" in tensors
    assert state_hash(runtime.model.state_dict()) == before and not optimizer.state


def test_real_observer_reads_all_cuda_counters_and_never_resets_peaks(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(uuid="CPU-mock", name="CPU-mock", total_memory=1000),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (800, 1000))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 10)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 20)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda: pytest.fail("Peak reset forbidden")
    )
    observer = v.CudaObservations()
    snapshot = observer.observe("before_load")
    assert snapshot["max_memory_allocated"] == 100 and snapshot["max_memory_reserved"] == 200
    assert snapshot["free_device_bytes"] == 800 and snapshot["memory_allocated"] == 10


def test_all_device_saved_activations_need_no_artificial_disk_spill(tmp_path):
    runtime, _, _, rows, credits, originals = _fixture.setup(tmp_path)
    evidence, _ = v.measure_fixture(
        runtime,
        rows,
        credits,
        originals,
        max_device_bytes=1024**2,
        max_host_bytes=1,
        max_disk_bytes=1024**2,
        scratch_directory=tmp_path / "STORAGE",
        disk_backend="arena_v3",
        fixture_clock=Clock(),
    )
    summary = evidence["offload"]
    assert summary["unpack_count"] == 0
    assert summary["device_resident_unpack_count"] > 0
    assert summary["disk_bytes_written"] == summary["disk_bytes_read"] == 0
    v.verify_storage(
        summary,
        tmp_path / "STORAGE",
        1,
        1024**2,
        disk_backend="arena_v3",
        fixture=True,
        max_device_bytes=1024**2,
    )
