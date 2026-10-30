"""Two-row CPU fixtures never substitute for real CUDA smoke measurements."""

import copy
import importlib.util
from pathlib import Path

import pytest

from mm_core.training import capture_rng, state_hash
from sr_f6_memory_repair import smoke as s

_spec = importlib.util.spec_from_file_location(
    "smoke_fixture", Path(__file__).with_name("test_memory_repair_validation.py")
)
f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f)


def measured(tmp_path, **kwargs):
    runtime, optimizer, _, rows, credits, originals = f.setup(tmp_path)
    output = tmp_path / "SMOKE"
    output.mkdir()
    e, t = s.measure(runtime, rows, credits, originals, output, fixture=True, **kwargs)
    return runtime, optimizer, rows, originals, e, t


def test_four_fixed_measurements_only_and_restore(tmp_path):
    runtime, optimizer, rows, originals, e, t = measured(tmp_path)
    assert runtime.calls == [rows[i]["row_id"] for i in [0, 0, 26, 26]]
    assert e["scope"] == "SYNTHETIC_CPU_ONLY"
    assert e["counts"]["backward_calls"] == 4
    assert e["counts"]["diagnostic_gradient_calls"] == 0
    assert e["row0_checked_before_row26"]
    assert optimizer.state == {}
    assert runtime.model.weight.grad is None
    assert s.recompute(e, t, originals) == e["comparisons"]
    assert e["comparisons"]["admission"] == "NOT_FULL_BATCH_ADMISSION"
    assert not s.SMOKE_PLAN["row26_eager_gradient_verified"]


@pytest.mark.parametrize(
    "kind", ["gradient", "forward", "shape", "count", "restore", "seconds", "plan"]
)
def test_cold_recompute_rejects_tamper(tmp_path, kind):
    _, _, _, originals, e, t = measured(tmp_path)
    e = copy.deepcopy(e)
    if kind == "gradient":
        t["row26_candidate"]["weight"] += 1
    if kind == "forward":
        e["measurements"]["row0_candidate"]["trace"]["train_logprobs"][0] += 1
    if kind == "shape":
        t["row0_eager"]["weight"] = t["row0_eager"]["weight"].flatten()
    if kind == "count":
        e["counts"]["backward_calls"] = 3
    if kind == "restore":
        e["restoration"]["rng_restored"] = False
    if kind == "seconds":
        e["measurements"]["row26_candidate"]["seconds"] = 121
    if kind == "plan":
        e["plan"]["loss_denominator"] = 2
    with pytest.raises(PermissionError):
        s.recompute(e, t, originals)


def test_row0_gate_failure_prevents_long_row_and_restores(tmp_path, monkeypatch):
    runtime, optimizer, _, rows, credits, originals = f.setup(tmp_path)
    output = tmp_path / "SMOKE"
    output.mkdir()
    before = state_hash(runtime.model.state_dict())
    rng = state_hash(capture_rng())
    original = s._pair

    def reject(*args):
        original(*args)
        raise PermissionError("row0 failed")

    monkeypatch.setattr(s, "_pair", reject)
    with pytest.raises(PermissionError, match="row0 failed") as exc:
        s.measure(runtime, rows, credits, originals, output, fixture=True)
    assert runtime.calls == [rows[0]["row_id"]] * 2
    assert state_hash(runtime.model.state_dict()) == before
    assert state_hash(capture_rng()) == rng
    assert exc.value.smoke_partial[0]["restoration"]["rng_restored"]
    assert not optimizer.state


def test_candidate_fixed_timeout_stops_before_long_row(tmp_path):
    runtime, _, _, rows, credits, originals = f.setup(tmp_path)
    output = tmp_path / "SMOKE"
    output.mkdir()
    now = [0.0]

    def clock():
        now[0] += 121
        return now[0]

    with pytest.raises(PermissionError, match="Candidate technical"):
        s.measure(runtime, rows, credits, originals, output, fixture=True, clock=clock)
    assert len(runtime.calls) == 2


def test_incomplete_archive_blocks_all_calls(tmp_path):
    runtime, _, _, rows, credits, originals = f.setup(tmp_path)
    with pytest.raises(PermissionError, match="27"):
        s.measure(runtime, rows, credits, originals[:2], tmp_path, fixture=True)
    assert not runtime.calls


def test_snapshot_cpu_never_claims_cuda():
    assert s._cuda_snapshot(None, fixture=True) == {"scope": "SYNTHETIC_CPU_ONLY"}


def test_module_import_is_standard_library_only():
    import subprocess
    import sys

    source = Path(s.__file__)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                f"import runpy,sys;runpy.run_path({str(source)!r});"
                "assert not any(k.startswith(('torch','sr_f6','mm_core')) for k in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_json_sorted_roundtrip_keeps_cold_comparison(tmp_path):
    import json

    _, _, _, originals, evidence, tensors = measured(tmp_path)
    loaded = json.loads(json.dumps(evidence, sort_keys=True))
    assert s.recompute(loaded, tensors, originals) == evidence["comparisons"]


@pytest.mark.parametrize(
    "field,value", [("storage", None), ("storage_mode", "allspill"), ("row", 26)]
)
def test_candidate_storage_identity_cannot_be_erased(tmp_path, field, value):
    _, _, _, originals, e, t = measured(tmp_path)
    e["measurements"]["row0_candidate"][field] = value
    with pytest.raises(PermissionError):
        s.recompute(e, t, originals)


def test_final_recompute_failure_keeps_full_partial(tmp_path, monkeypatch):
    runtime, _, _, rows, credits, originals = f.setup(tmp_path)
    output = tmp_path / "SMOKE"
    output.mkdir()

    def reject(*args):
        raise PermissionError("final cold failure")

    monkeypatch.setattr(s, "recompute", reject)
    with pytest.raises(PermissionError) as exc:
        s.measure(runtime, rows, credits, originals, output, fixture=True)
    evidence, tensors = exc.value.smoke_partial
    assert len(tensors) == 4
    assert evidence["restoration"]["rng_restored"]


def test_fixed_memory_margin_gate(tmp_path):
    _, _, _, originals, e, t = measured(tmp_path)
    e["scope"] = "ACTUAL_CUDA_ONLY"
    e["measurements"]["row0_candidate"]["cuda"]["finished"] = {
        "max_allocated": 9 * 2**30,
        "total": 10 * 2**30,
    }
    with pytest.raises(PermissionError, match="headroom"):
        s._pair(e, t, originals, 0)


def test_cuda_observation_totals_bound_to_actual_hardware():
    import json

    value = dict(
        scope="ACTUAL_CUDA_MEASUREMENT",
        device_index=0,
        allocated=1,
        reserved=2,
        max_allocated=1,
        max_reserved=2,
        free=7,
        total=10,
    )
    evidence = {
        "measurements": {
            "x": {
                "cuda": {
                    k: dict(value)
                    for k in ["before", "after_forward", "after_backward", "finished"]
                }
            }
        }
    }
    hardware = dict(
        cuda_visible_device_count=1,
        cuda_name="NVIDIA PRO 6000",
        cuda_uuid="GPU-actual",
        cuda_total_memory_bytes=10,
    )
    s._verify_cuda(json.loads(json.dumps(evidence, sort_keys=True)), hardware)
    hardware["cuda_total_memory_bytes"] = 11
    with pytest.raises(PermissionError):
        s._verify_cuda(evidence, hardware)


def test_two_distinct_peak_resets_per_pair_cpu_mock(tmp_path, monkeypatch):
    import torch

    from sr_f6_memory_repair import offload, validation

    runtime, _, _, rows, credits, originals = f.setup(tmp_path)
    output = tmp_path / "SMOKE"
    output.mkdir()
    resets = []
    for name in ["empty_cache", "synchronize"]:
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda: resets.append(len(runtime.calls))
    )
    monkeypatch.setattr(
        s,
        "_cuda_snapshot",
        lambda *a, **kw: dict(scope="ACTUAL_CUDA_MEASUREMENT", max_allocated=1, total=10 * 2**30),
    )
    saved = offload.saved_activation_offload

    def cpu_hooks(*a, **kw):
        kw["fixture"] = True
        return saved(*a, **kw)

    monkeypatch.setattr(offload, "saved_activation_offload", cpu_hooks)
    verify = validation.verify_storage

    def cpu_verify(*a, **kw):
        kw["fixture"] = True
        return verify(*a, **kw)

    monkeypatch.setattr(validation, "verify_storage", cpu_verify)
    evidence, _ = s.measure(runtime, rows, credits, originals, output, fixture=False)
    assert resets == [0, 1, 2, 3]
    assert evidence["counts"]["backward_calls"] == 4


def test_nonfinite_failure_preserves_diagnostics_without_result(tmp_path, monkeypatch):
    import json

    from sr_f6.contract import PACKAGE_CONFIG_SHA256
    from sr_f6_memory_repair import validation

    runtime, _, _, rows, credits, originals = f.setup(tmp_path)
    runtime.identity["technical_consumer"] = "fixture_loader"
    contract = dict(
        experiment_root=str(tmp_path),
        producer={"revision": "a" * 40},
        repair_source={"revision": "b" * 40},
        config_ref={"sha256": PACKAGE_CONFIG_SHA256},
    )
    monkeypatch.setattr(s, "PROJECT", tmp_path)
    monkeypatch.setattr(s, "validate_contract", lambda *a, **kw: contract)
    monkeypatch.setattr(validation, "_originals", lambda c: ({}, rows, credits, originals))
    monkeypatch.setattr(validation, "_technical_runtime", lambda *a, **kw: runtime)

    def fail(*a, **kw):
        err = PermissionError("nonfinite paired numerical gate")
        err.smoke_partial = (
            {"loss": float("nan"), "reference": float("inf")},
            {"original": runtime.model.weight.detach().clone()},
        )
        raise err

    monkeypatch.setattr(s, "measure", fail)
    with pytest.raises(PermissionError, match="nonfinite"):
        s.run_smoke(tmp_path / "CONTRACT.json", "c" * 64, tmp_path / "SMOKE")
    partial = json.loads((tmp_path / "SMOKE/PARTIAL_EVIDENCE.json").read_text())
    assert partial["evidence"]["loss"] == {"nonfinite_float": "nan"}
    assert (tmp_path / "SMOKE/FAILURE.json").is_file()
    assert (tmp_path / "SMOKE/PARTIAL_TENSORS.pt").is_file()
    assert not (tmp_path / "SMOKE/RESULT.json").exists()


def test_timeout_during_long_row_restores_and_preserves_prefix(tmp_path):
    runtime, _, _, rows, credits, originals = f.setup(tmp_path)
    output = tmp_path / "SMOKE"
    output.mkdir()
    before = state_hash(capture_rng())

    def interrupt(name, evidence, tensors):
        if name == "row26_allspill_started":
            raise TimeoutError("frozen interruption")

    with pytest.raises(TimeoutError) as exc:
        s.measure(runtime, rows, credits, originals, output, fixture=True, progress=interrupt)
    assert runtime.calls == [rows[0]["row_id"]] * 2
    assert len(exc.value.smoke_partial[1]) == 2
    assert state_hash(capture_rng()) == before
    assert exc.value.smoke_partial[0]["restoration"]["rng_restored"]


def test_controlled_alarm_removed_after_exception(monkeypatch):
    import signal

    old = signal.getsignal(signal.SIGALRM)

    def fail(*args):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 1490
        signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)

    monkeypatch.setattr(s, "_run_smoke", fail)
    with pytest.raises(TimeoutError, match="1500"):
        s.run_smoke("unused", "unused", "unused")
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
    assert signal.getsignal(signal.SIGALRM) == old


def test_fault_cancels_owned_alarm_before_runtime_restoration(tmp_path, monkeypatch):
    import signal

    runtime, _, _, rows, credits, originals = f.setup(tmp_path)
    output = tmp_path / "SMOKE"
    output.mkdir()

    def measured_failure(*args):
        runtime.fail_row = rows[0]["row_id"]
        try:
            s.measure(runtime, rows, credits, originals, output, fixture=True)
        except RuntimeError:
            assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
            raise

    monkeypatch.setattr(s, "_run_smoke", measured_failure)
    with pytest.raises(RuntimeError):
        s.run_smoke("unused", "unused", "unused")
