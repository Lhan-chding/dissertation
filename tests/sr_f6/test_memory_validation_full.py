"""Independent V7 CPU fixtures; no actual CUDA acceptance or science claim."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mm_core.training import capture_rng, state_hash
from sr_f6_memory_repair import full_contract as f
from sr_f6_memory_repair import validation as old
from sr_f6_memory_repair import validation_full as v

_spec = importlib.util.spec_from_file_location(
    "full_fixture", Path(__file__).with_name("test_memory_repair_validation.py")
)
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)


class Clock:
    def __init__(self, delta=1):
        self.value, self.delta = 100.0, delta

    def __call__(self):
        self.value += self.delta
        return self.value


def contract():
    return dict(
        schema=f.SCHEMA,
        max_device_bytes=f.DEVICE_BYTES,
        max_host_bytes=f.HOST_BYTES,
        max_disk_bytes=f.DISK_BYTES,
        storage_backend="arena_v3",
        validation_window=copy.deepcopy(f.WINDOW),
        producer=dict(revision="a" * 40),
        repair_source=dict(revision="b" * 40),
        original_ready_ref=dict(path="ready", sha256="c" * 64),
        old_allocation_failure_ref=dict(path="failure", sha256="d" * 64),
        prior_smoke_result_ref=dict(path="v6result", sha256="e" * 64),
    )


def measure(tmp_path, clock=None):
    runtime, optimizer, _, rows, credits, originals = fixture.setup(tmp_path)
    runtime.fixture_rng_before = state_hash(capture_rng())
    evidence, tensors = v.measure_fixture(
        runtime,
        rows,
        credits,
        originals,
        max_device_bytes=f.DEVICE_BYTES,
        max_host_bytes=f.HOST_BYTES,
        max_disk_bytes=f.DISK_BYTES,
        disk_backend="arena_v3",
        scratch_directory=tmp_path / "STORAGE",
        fixture_clock=clock or Clock(),
    )
    return runtime, optimizer, rows, evidence, tensors, originals


def test_new_budget_is_explicit_and_old_24gib_gate_unchanged():
    c = contract()
    assert v.validation_schema(c) == f.RESULT_SCHEMA
    assert old.MAX_DEVICE_BYTES == 24 * 2**30
    with pytest.raises(PermissionError):
        old.validation_schema(c)
    for maximum in (0, 24 * 2**30, 32 * 2**30, float(f.DEVICE_BYTES), True):
        with pytest.raises(PermissionError):
            v.validation_schema({**c, "max_device_bytes": maximum})


@pytest.mark.parametrize(
    "name",
    [
        "_measure",
        "recompute",
        "_originals",
        "_technical_runtime",
        "verify_cuda_observations",
        "window_measurement",
    ],
)
def test_original_scientific_loop_and_helpers_are_mechanically_identical(name):
    def node(module):
        tree = ast.parse(Path(module.__file__).read_text())
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)

    assert ast.dump(node(v), include_attributes=False) == ast.dump(
        node(old), include_attributes=False
    )
    assert v.PLAN == old.PLAN and v.VALIDATION_WINDOW == old.VALIDATION_WINDOW


def test_128_once_original_diagnostics_and_exact_restoration(tmp_path):
    runtime, optimizer, rows, evidence, tensors, originals = measure(tmp_path)
    assert runtime.calls == [rows[0]["row_id"], *[r["row_id"] for r in rows]]
    assert evidence["counts"] == dict(
        reference_forwards=129,
        policy_forwards=129,
        backward_calls=129,
        diagnostic_gradient_calls=192,
        generated_responses=0,
        optimizer_updates=0,
    )
    assert evidence["technical_window"]["status"] == "CONTINUE_SAME_PROCESS"
    assert not optimizer.state and runtime.model.weight.grad is None
    assert state_hash(capture_rng()) == runtime.fixture_rng_before
    assert all(evidence["restoration"][key] for key in v.RESTORE_KEYS)
    assert v.recompute(evidence, tensors, originals) == evidence["comparisons"]
    assert evidence["scope"] == "SYNTHETIC_CPU_ONLY"


def test_slow_four_diagnostic_rows_never_enters_fifth_and_restores(tmp_path):
    runtime, optimizer, _, rows, credits, originals = fixture.setup(tmp_path)
    before, rng = state_hash(runtime.model.state_dict()), state_hash(capture_rng())
    with pytest.raises(v.TechnicalWindowStop) as caught:
        v.measure_fixture(
            runtime,
            rows,
            credits,
            originals,
            max_device_bytes=f.DEVICE_BYTES,
            fixture_clock=Clock(50),
        )
    evidence, _ = caught.value.validation_partial
    assert runtime.calls == [rows[0]["row_id"], *[r["row_id"] for r in rows[:4]]]
    assert evidence["counts"]["backward_calls"] == 5
    assert evidence["counts"]["diagnostic_gradient_calls"] == 12
    assert evidence["technical_window"]["total_row_seconds"] == 1000
    assert state_hash(runtime.model.state_dict()) == before and state_hash(capture_rng()) == rng
    assert not optimizer.state and runtime.model.weight.grad is None


@pytest.mark.parametrize("mutation", ["gradient", "vector", "count", "restore", "original"])
def test_full_cold_recompute_rejects_tampering(tmp_path, mutation):
    _, _, _, evidence, tensors, originals = measure(tmp_path)
    if mutation == "gradient":
        tensors["offload_row0"]["weight"] += 1
    elif mutation == "vector":
        evidence["traces"][26]["train_logprobs"][0] += 1
    elif mutation == "count":
        evidence["counts"]["diagnostic_gradient_calls"] -= 1
    elif mutation == "restore":
        evidence["restoration"]["rng_restored"] = False
    else:
        originals = copy.deepcopy(originals)
        originals[26]["tokens"][0] += 1
    with pytest.raises(PermissionError):
        v.recompute(evidence, tensors, originals)


@pytest.mark.parametrize("mutation", [None, "missing", "purpose", "tokens", "boolean_count"])
def test_exact_full_ledger_order_and_original_lengths(mutation):
    rows = [dict(tokens=list(range(i + 1))) for i in range(128)]
    ledger = [
        dict(
            kind="extra_forward_sequences",
            count=1,
            metadata=dict(purpose=p, completion_tokens=len(rows[i]["tokens"])),
        )
        for i in [0, *range(128)]
        for p in ("training_reference", "training_policy")
    ]
    if mutation == "missing":
        ledger.pop()
    elif mutation == "purpose":
        ledger[3]["metadata"]["purpose"] = "training_reference"
    elif mutation == "tokens":
        ledger[-1]["metadata"]["completion_tokens"] = 1
    elif mutation == "boolean_count":
        ledger[0]["count"] = True
    if mutation is None:
        v._verify_ledger(ledger, rows)
    else:
        with pytest.raises(PermissionError):
            v._verify_ledger(ledger, rows)


@pytest.mark.parametrize("fail_cold", [False, True])
def test_cold_failure_never_publishes_final_success(tmp_path, monkeypatch, capsys, fail_cold):
    c = contract()
    output = tmp_path / "VALIDATION"
    monkeypatch.setattr(f, "verify_contract", lambda *a: c)
    # The production path guard is retained; fixture substitutes only its root test.
    original_relative = Path.is_relative_to
    monkeypatch.setattr(
        Path,
        "is_relative_to",
        lambda self, p: (
            True if str(p) == "/projects/_ssd/varunssd/louis-ssvc" else original_relative(self, p)
        ),
    )
    monkeypatch.setattr(v, "_originals", lambda _: ({}, [], {}, []))
    monkeypatch.setattr(
        v, "_technical_runtime", lambda *a: SimpleNamespace(identity={}, reference={})
    )
    monkeypatch.setattr(
        v,
        "CudaObservations",
        lambda: SimpleNamespace(
            observe=lambda *a: None,
            evidence=lambda: {},
        ),
    )
    monkeypatch.setattr(v, "ProgressWriter", lambda *a, **kw: None)
    monkeypatch.setattr(v, "_measure", lambda *a, **kw: (dict(diagnostic_indices=[]), {}))
    monkeypatch.setattr(v, "verify_cuda_observations", lambda *a: None)
    runtime = SimpleNamespace(identity=dict(hardware={}), reference={})
    monkeypatch.setattr(v, "_technical_runtime", lambda *a, **kw: runtime)

    def cold(path, sha, ref, *, candidate=False):
        assert candidate and Path(ref["path"]).name == "CANDIDATE_RESULT.json"
        assert not (output / "RESULT.json").exists()
        assert not (output / "COLD_VERIFICATION.json").exists()
        if fail_cold:
            raise PermissionError("Actual cold failed")
        return dict(schema=v.SCHEMA, status=f.STATUS, validation_ref=ref)

    monkeypatch.setattr(v, "_verify_validation", cold)
    if fail_cold:
        with pytest.raises(PermissionError, match="Actual cold failed"):
            v.run_validation(tmp_path / "CONTRACT.json", "a" * 64, output)
        assert (output / "FAILURE.json").exists()
        assert not (output / "RESULT.json").exists()
        assert not (output / "COLD_VERIFICATION.json").exists()
    else:
        result = v.run_validation(tmp_path / "CONTRACT.json", "a" * 64, output)
        assert result["validation_ref"] == v._ref(output / "RESULT.json")
        assert (output / "RESULT.json").read_bytes() == (
            output / "CANDIDATE_RESULT.json"
        ).read_bytes()
        assert json.loads((output / "COLD_VERIFICATION.json").read_text()) == result
    stages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    names = [entry["stage"] for entry in stages]
    assert names[:2] == ["initial_contract_started", "initial_contract_complete"]
    assert "originals_started" in names and "model_load_started" in names
    assert "measurement_started" in names and "final_contract_complete" in names
    assert "cold_started" in names
    assert ("cold_complete" in names) is (not fail_cold)
    assert all(entry["elapsed_seconds"] >= 0 and entry["unix_seconds"] > 0 for entry in stages)


def test_public_cold_verifier_rejects_candidate_path(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "verify_contract", lambda *a: contract())
    with pytest.raises(PermissionError, match="outside"):
        v.verify_validation(
            tmp_path / "CONTRACT.json",
            "a" * 64,
            dict(path=str(tmp_path / "VALIDATION/CANDIDATE_RESULT.json"), sha256="b" * 64),
        )
