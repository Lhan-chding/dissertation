"""Read-only warm postflight; synthetic fixtures are never CUDA evidence."""

import copy
import csv
import importlib.util
import json
import math
import sys

import pytest

from src.core import canonical_hash, file_hash, write_json
from src.r3_warm_gate import _profile_check as _real_profile_check
from src.r3_warm_gate import _raw_inputs as _real_raw_inputs


def test_warm_gate_has_independent_postflight_entrypoint():
    assert importlib.util.find_spec("src.r3_warm_gate") is not None


@pytest.fixture(scope="module")
def warm_fixture(tmp_path_factory):
    """Run the real tiny torch fork/direct pipeline once; mock only CPU/CUDA boundaries."""
    from test_r3_warm_runtime import _fixture

    import src.r3_warm_gate as gate_api
    from src.r3_warm_runtime import run_r3_warm

    patch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("warm_postflight")
    args, adapter, _, _, checkpoint = _fixture(root, patch)
    upstream = sys.modules["src.next_stage_runtime"]
    gate = upstream.validate_prerequisites()
    r2 = upstream.validate_r2_gate()
    cold = upstream.validate_r3_cold_gate()
    r4 = upstream.validate_r4_gate()
    environment = {"status": "CPU_FIXTURE_NOT_REAL_ENVIRONMENT"}
    for binding in (r2, cold, r4):
        binding["environment"] = environment
    result = run_r3_warm(*args, _adapter_factory=adapter)
    assert result["status"] == "PASS", result["details"]

    def fixture_ledger(path):
        records = {}
        for line in path.read_text().splitlines(keepends=True):
            row = json.loads(line)
            assert line.endswith("\n")
            assert row["execution_kind"] == "CPU_FAKE_ADAPTER_FIXTURE"
            assert row["execution_checks"]["passed"] is True
            assert row["record_hash"] == canonical_hash(
                {k: v for k, v in row.items() if k != "record_hash"}
            )
            assert row["sample_key"] not in records
            records[row["sample_key"]] = row
        return records

    patch.setattr(gate_api, "KIND", "CPU_FAKE_ADAPTER_FIXTURE")
    patch.setattr(gate_api, "_check_source", lambda *a: None)
    patch.setattr(gate_api, "_check_finished_source", lambda *a: None)
    patch.setattr(gate_api, "_processor_inputs", lambda *a: {})
    patch.setattr(gate_api, "_raw_inputs", lambda *a: None)
    patch.setattr(gate_api, "_profile_check", lambda *a: None)
    patch.setattr(gate_api, "_ledger", fixture_ledger)
    patch.setattr("src.r3_gate._ledger", fixture_ledger)
    yield args[2], gate, r2, cold, r4, checkpoint
    patch.undo()


def test_completed_tiny_warm_full_state_reports_and_read_only(warm_fixture):
    from src.r3_warm_gate import validate_r3_warm_gate

    root, gate, r2, cold, r4, checkpoint = warm_fixture
    before = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    checkpoint_hash = file_hash(checkpoint["path"])
    audit = validate_r3_warm_gate(root, gate, r2, cold, r4)
    assert audit["status"] == "PASS"
    assert audit["raw_counts"]["raw_sample_count"] == 4224
    assert audit["response_artifact_audit"]["paired_response_rows"] == 280
    assert audit["response_artifact_audit"]["gradient_candidate_rows"] == 60
    assert audit["direct_validation"]["direct_csv_rows"] == 168
    assert audit["runtime_counts"]["optimizer_steps_observed_at_least"] == 80
    assert before == {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    assert file_hash(checkpoint["path"]) == checkpoint_hash


@pytest.mark.parametrize("step", [0, 1, 63, 64, 66])
def test_candidate_must_be_actual_adam65(warm_fixture, step):
    from src.optimizer_fork import load_checkpoint
    from src.r3_warm_gate import _check_mature

    checkpoint = warm_fixture[-1]
    state = load_checkpoint(checkpoint["path"], checkpoint["identity"])
    for entry in state["optimizer"]["state"].values():
        entry["step"].fill_(step)
    with pytest.raises(ValueError, match="step"):
        _check_mature(state, step=65)
    for entry in state["optimizer"]["state"].values():
        entry["step"].fill_(65)
    _check_mature(state, step=65)


@pytest.mark.parametrize(
    "mutation", ["missing_bank", "missing_comparison", "estimate", "ratio", "warning", "nan"]
)
def test_direct_statistical_coverage_and_semantic_tampering(warm_fixture, mutation):
    from src.r3_warm_gate import _check_direct_csv, _compare_direct

    root = warm_fixture[0]
    original = json.loads((root / "direct_validation_bank_00.json").read_text())
    altered = copy.deepcopy(original)
    if mutation == "missing_bank":
        with pytest.raises(ValueError, match="168"):
            _check_direct_csv(root, {0: original})
        return
    elif mutation == "missing_comparison":
        del altered["comparisons"]["auxiliary"]
    elif mutation == "warning":
        altered["warnings"] = ["FORGED"]
    else:
        cell = altered["comparisons"]["absolute_candidate"]["responses"]["overall"]["pX"]
        if mutation == "estimate":
            cell["observed_delta"] += 1e-12
        elif mutation == "ratio":
            cell["ci"]["observed_delta"]["half_width_to_abs_predicted_delta"] = 123.0
        else:
            cell["predicted_delta"] = float("nan")
    with pytest.raises(ValueError):
        _compare_direct(altered, original)


def test_direct_csv_requires_exact_stored_json_consistency(warm_fixture, tmp_path):
    from src.r3_warm_gate import _check_direct_csv

    root = warm_fixture[0]
    results = {
        i: json.loads((root / f"direct_validation_bank_{i:02d}.json").read_text()) for i in (0, 6)
    }
    with (root / "direct_validation_effects.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["observed_delta"] = str(math.nextafter(float(rows[0]["observed_delta"]), math.inf))
    with (tmp_path / "direct_validation_effects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="CSV differs"):
        _check_direct_csv(tmp_path, results)


def test_non_cuda_or_missing_direct_artifacts_cannot_pass(tmp_path):
    from src.r3_warm_gate import validate_r3_warm_gate

    write_json(tmp_path / "manifest.json", {"files": []})
    write_json(
        tmp_path / "status.json",
        {"status": "PASS", "phase": "R3-warm", "execution_kind": "REAL_CUDA_FORK"},
    )
    with pytest.raises(ValueError):
        validate_r3_warm_gate(tmp_path, {}, {}, {}, {})


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "source",
        "ref",
        "bytes",
        "manifest",
        "binding",
        "phase",
        "status",
        "deleted_status",
        "fault",
        "terminal_source",
    ],
)
def test_cached_recursive_preflight_cannot_bypass_bound_sources_or_files(
    tmp_path, monkeypatch, mutation
):
    import src.r3_warm_gate as api

    binding = {"r0_r1": {"fixture": "R0/R1"}}
    source = {"source_commit": "f" * 40, "source_files": {"src/original.py": "hash"}}
    roots = []
    for key, path_key, files_key in (
        ("r2", "r2_dir", "r2_files"),
        ("r3_cold", "r3_cold_dir", "r3_files"),
        ("r4", "r4_dir", "r4_files"),
    ):
        directory = tmp_path / key
        directory.mkdir()
        (directory / "evidence.json").write_text("{}\n")
        digest = file_hash(directory / "evidence.json")
        runtime = source if key == "r2" else {"source": source}
        write_json(directory / "runtime_lock.json", runtime)
        write_json(
            directory / "status.json",
            {
                "status": "PASS",
                **source,
                "phase": {"r2": "R2", "r3_cold": "R3-cold", "r4": "R4"}[key],
                "execution_kind": {
                    "r2": "REAL_CUDA_INFERENCE",
                    "r3_cold": "REAL_CUDA_FORK",
                    "r4": "REAL_CUDA_TRAINING",
                }[key],
            },
        )
        hashes = {
            "evidence.json": digest,
            "runtime_lock.json": file_hash(directory / "runtime_lock.json"),
        }
        write_json(
            directory / "manifest.json",
            {
                "files": [
                    {"path": name, "sha256": sha, "bytes": (directory / name).stat().st_size}
                    for name, sha in hashes.items()
                ]
            },
        )
        binding[key] = {
            "status": "PASS",
            path_key: str(directory),
            files_key: hashes,
        }
        roots.append(directory)
    warm = tmp_path / "warm"
    warm.mkdir()
    write_json(warm / "gate_binding.json", binding)
    config, plan, data, environment = (
        {"fixture": "config"},
        {"plan_hash": "plan"},
        {"fixture": "data"},
        {"fixture": "environment"},
    )
    value = {
        "status": "PASS",
        "phase": "R3-warm_PREFLIGHT",
        "execution_kind": "CPU_AUDIT",
        "model_loaded": False,
        "source": {"source_commit": "f" * 40, "source_files": {"src/original.py": "hash"}},
        "config_hash": canonical_hash(config),
        "gate_binding": binding,
        "data_binding": data,
        "plan_hash": "plan",
        "environment": environment,
    }
    monkeypatch.setattr(api, "_committed_sources", lambda rev: {"src/original.py": "hash"})
    from src.r4_gate import _check_finished_source

    monkeypatch.setattr(api, "_check_source", lambda *a: None)
    monkeypatch.setattr(api, "_check_finished_source", _check_finished_source)
    monkeypatch.setattr(api, "_load_plan", lambda *a: (plan, data))
    monkeypatch.setattr(
        sys.modules["src.next_stage_runtime"],
        "validate_runtime_environment",
        lambda *a: environment,
    )
    if mutation == "source":
        value["source"]["source_files"]["src/original.py"] = "changed"
    elif mutation == "ref":
        value["source"]["source_commit"] = "HEAD"
    elif mutation == "bytes":
        (roots[2] / "evidence.json").write_text('{"changed":true}\n')
    elif mutation == "manifest":
        write_json(roots[2] / "manifest.json", {"files": []})
    elif mutation == "binding":
        write_json(warm / "gate_binding.json", {**binding, "extra": True})
    elif mutation == "phase":
        value["phase"] = "R4_PREFLIGHT"
    elif mutation in ("status", "terminal_source"):
        terminal = json.loads((roots[0] / "status.json").read_text())
        terminal["status" if mutation == "status" else "source_commit"] = "FAIL"
        write_json(roots[0] / "status.json", terminal)
    elif mutation == "deleted_status":
        (roots[0] / "status.json").unlink()
    elif mutation == "fault":
        write_json(roots[2] / "measurement_fault.json", {"reason": "new fault"})
    path = tmp_path / "preflight.json"
    write_json(path, value)
    args = (path, warm, {"binding": binding["r0_r1"]}, config, tmp_path, *roots)
    if mutation is None:
        assert api._cached_preflight(*args) == binding
    else:
        with pytest.raises(ValueError):
            api._cached_preflight(*args)


def test_raw_input_reconstruction_and_direct_step_are_enforced(monkeypatch):
    import src.r3_warm_gate as api

    monkeypatch.setattr(api, "_check_raw_rows", lambda *a: None)
    checked = []

    class Inputs:
        def check(self, prompt, row):
            checked.append(prompt["prompt_id"])
            if row["prepared_hash"] != "rebuilt":
                raise ValueError("processor input mismatch")

    plan = {"train_prompts": [], "control_prompts": [{"prompt_id": "p"}]}
    row = {
        "prompt_id": "p",
        "checkpoint_step": 64,
        "origin_checkpoint_step": 64,
        "optimizer_step": 65,
        "prepared_hash": "rebuilt",
        "elapsed": 1.0,
        "elapsed_unit": "seconds",
        "elapsed_scope": "generation_meter_scope_including_synchronization",
        "runtime_forward_by_reason": {"generation": 2},
        "memory_measurement_status": "CUDA_MEASURED",
        "peak_memory": {
            "peak_cuda_bytes": 1,
            "peak_cuda_reserved_bytes": 2,
            "peak_cpu_rss_bytes": 3,
        },
    }
    args = (
        None,
        {"s": row},
        plan,
        {"unit": "direct_control"},
        {},
        {},
        {"model_audit": {}},
        {},
        Inputs(),
    )
    _real_raw_inputs(*args)
    assert checked == ["p"]
    row["prepared_hash"] = "self_asserted"
    with pytest.raises(ValueError, match="processor"):
        _real_raw_inputs(*args)
    row["prepared_hash"] = "rebuilt"
    row["optimizer_step"] = 64
    with pytest.raises(ValueError, match="step"):
        _real_raw_inputs(*args)
    row["optimizer_step"] = 65
    for key in ("elapsed", "peak_memory", "runtime_forward_by_reason", "memory_measurement_status"):
        value = row.pop(key)
        with pytest.raises(ValueError, match="telemetry"):
            _real_raw_inputs(*args)
        row[key] = value


@pytest.mark.parametrize("field", ["rng", "metadata", "parameters", "topology"])
def test_candidate_cannot_drop_or_change_origin_state_schema(warm_fixture, field):
    from src.optimizer_fork import load_checkpoint
    from src.r3_warm_gate import _check_mature

    checkpoint = warm_fixture[-1]
    origin = load_checkpoint(checkpoint["path"], checkpoint["identity"])
    state = copy.deepcopy(origin)
    for entry in state["optimizer"]["state"].values():
        entry["step"].fill_(65)
    _check_mature(state, step=65, origin=origin)
    if field in ("rng", "metadata"):
        state[field] = {}
    elif field == "parameters":
        key = next(iter(state["parameters"]))
        state["parameters"]["renamed"] = state["parameters"].pop(key)
    else:
        state["optimizer"]["param_groups"][0]["params"] = [123]
    with pytest.raises(ValueError):
        _check_mature(state, step=65, origin=origin)


def test_direct_negative_width_cannot_hide_within_roundoff(warm_fixture):
    from src.r3_warm_gate import _compare_direct

    root = warm_fixture[0]
    result = json.loads((root / "direct_validation_bank_00.json").read_text())
    altered = copy.deepcopy(result)
    cell = altered["comparisons"]["absolute_candidate"]["responses"]["overall"]["pX"]
    interval = cell["ci"]["predicted_delta"]
    interval["half_width"] = -1e-15
    for key in ("half_width_to_abs_point_estimate", "half_width_to_abs_predicted_delta"):
        interval[key] = -1e-15 / abs(cell["predicted_delta"]) if cell["predicted_delta"] else None
    with pytest.raises(ValueError, match="CI"):
        _compare_direct(altered, result)


@pytest.mark.parametrize("name", ["report_zh.md", "warm_report.md", "support_control_report.md"])
def test_prose_reports_cannot_be_missing_verified_facts(warm_fixture, monkeypatch, name):
    from pathlib import Path

    from src.r3_warm_gate import _reports

    root = warm_fixture[0]
    banks = json.loads((root / "candidate_manifest.json").read_text())["banks"]
    status = json.loads((root / "status.json").read_text())
    original = Path.read_text
    monkeypatch.setattr(
        Path, "read_text", lambda p, *a, **k: "" if p == root / name else original(p, *a, **k)
    )
    with pytest.raises(ValueError, match="prose report"):
        _reports(root, banks, status)


def test_profile_requires_measured_hooks_and_counters():
    profile = {
        "language_layers_instrumented": 32,
        "top_hook_installed": True,
        "vision_hooks_installed": 1,
        "internal_recompute_status": "MEASURED",
        "optimizer_step_calls_observed": 80,
        "backward_calls_observed": 3264,
    }
    _real_profile_check(profile)
    for key in list(profile):
        value = profile.pop(key)
        with pytest.raises(ValueError, match="instrumentation"):
            _real_profile_check(profile)
        profile[key] = value
