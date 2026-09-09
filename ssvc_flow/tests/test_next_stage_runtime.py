"""Downstream phases cannot turn partial, fake, or changed evidence into a gate."""

import copy

import pytest

from src.core import canonical_hash, file_hash, write_json
from src.next_stage_common import load_yaml


def seal(root):
    write_json(
        root / "manifest.json",
        {
            "files": [
                {"path": p.name, "sha256": file_hash(p), "bytes": p.stat().st_size}
                for p in sorted(root.glob("*.json"))
                if p.name not in {"manifest.json", "status.json"}
            ]
        },
    )


@pytest.fixture
def gates(tmp_path):
    from src.core import PROJECT_ROOT
    from src.next_stage_runtime import FROZEN_PATH_SOURCES, REQUIRED_R0_CHECKS
    from src.r1_reference_smoke import CERTIFICATE_CHECKS

    r0, r1, supplement = [tmp_path / s for s in ("R0", "R1", "supplement")]
    for p in (r0, r1, supplement):
        p.mkdir()
    config = load_yaml("configs/next_stage.yaml")
    checks = {name: {"status": "PASS"} for name in REQUIRED_R0_CHECKS}
    checks["dataset_manifest"].update(
        manifest_sha256="dataset", files={"calibration.jsonl": {"sha256": "calibration"}}
    )
    write_json(r0 / "closure_checks.json", checks)
    write_json(r0 / "identity.json", {"fixture": True})
    write_json(
        r0 / "status.json",
        {
            "status": "PASS",
            "phase": "R0",
            "execution_kind": "CPU_AUDIT",
            "details": {"R0_gate_passed": True},
        },
    )
    for track in ("N", "L"):
        write_json(r0 / f"baseline_lock_{track}.json", {"model": config["model"]})
    certificate = {
        "status": "PASS",
        "selected_path": "uncached_prefix_recompute",
        "model_revision": config["model"]["revision"],
        "initial_adapter_hash": "initial",
        "model_audit": {"frozen_parameter_hash": "frozen", "transformers_version": "5.14.1"},
        "fixed_sequence_boundaries": {
            "passed": True,
            "checks_completed": 27,
            "checks_expected": 27,
            "not_model_generated_not_training": True,
        },
        "checks": {
            n: {
                "passed": True,
                "sequences": 120,
                "failed_sequence_checks": 0,
                "top1_measured": n in {"reference_repeat", "behavior"},
            }
            for n in CERTIFICATE_CHECKS
        },
    }
    for name in (
        "parity_alarm_mean_abs_token_logp",
        "parity_alarm_p99_abs_token_logp",
        "parity_alarm_min_top1_agreement",
    ):
        certificate.setdefault("thresholds", {})[name] = config["R1"][name]
    write_json(r1 / "production_path_validation.json", certificate)
    write_json(
        r1 / "environment_lock.json",
        {
            "status": "PASS",
            "execution_kind": "REAL_CUDA_INFERENCE",
            "python": "3.12.14",
            "model": {"transformers_version": "5.14.1"},
            "config": config,
            # R1 writes the YAML file's byte hash, not canonical JSON.
            "config_sha256": file_hash(PROJECT_ROOT / "configs/next_stage.yaml"),
        },
    )
    write_json(
        r1 / "training_smoke.json",
        {
            "status": "PASS",
            "passed": True,
            "raw_sample_count": 128,
            "runtime_lock": {"config": config},
        },
    )
    write_json(
        r1 / "status.json",
        {
            "phase": "R1",
            "status": "PASS",
            "execution_kind": "REAL_CUDA_TRAINING_SMOKE",
            "details": {"R1_gate_passed": True},
            "source_files": {name: file_hash(PROJECT_ROOT / name) for name in FROZEN_PATH_SOURCES},
        },
    )
    payload = {
        "status": "PASS",
        "passed": True,
        "execution_kind": "REAL_CUDA_FORK",
        "new_rollouts": 0,
        "frozen_base_unchanged": True,
        "original_files_unchanged": True,
        "scratch_restoration": {
            n: True for n in ("parameters", "optimizer", "rng", "empty_optimizer")
        },
        **{
            n: {"passed": True}
            for n in ("zero_gradient_adam", "lambda_zero_control", "candidate_order")
        },
    }
    write_json(supplement / "supplement.json", payload)
    write_json(
        supplement / "input_binding.json",
        {
            "original_run": str(r1),
            "original_config": config,
            "original_config_sha256": canonical_hash(config),
            "certificate_sha256": canonical_hash(certificate),
            "initial_adapter_hash": "initial",
            "frozen_parameter_hash": "frozen",
            "model_revision": config["model"]["revision"],
            "original_file_hashes": {
                str(r1 / "production_path_validation.json"): file_hash(
                    r1 / "production_path_validation.json"
                )
            },
        },
    )
    write_json(
        supplement / "status.json",
        {"phase": "R1_SUPPLEMENT", "status": "PASS", "execution_kind": "REAL_CUDA_FORK"},
    )
    for p in (r0, r1, supplement):
        seal(p)
    return r0, r1, supplement


def test_accepts_complete_bound_evidence(gates):
    from src.next_stage_runtime import validate_prerequisites

    result = validate_prerequisites(*gates)
    assert result["binding"]["status"] == "PASS"
    assert result["certificate"]["selected_path"] == "uncached_prefix_recompute"
    assert result["config"]["model"]["id"] == "Qwen/Qwen3.5-9B"


@pytest.mark.parametrize(
    "target,field,value",
    [
        (0, "status", "INCONCLUSIVE"),
        (1, "execution_kind", "CPU_FAKE_ADAPTER_FIXTURE"),
        (2, "execution_kind", "CPU_FAKE_ADAPTER_FIXTURE"),
    ],
)
def test_rejects_incomplete_or_cpu_gate(gates, target, field, value):
    import json

    from src.next_stage_runtime import validate_prerequisites

    p = gates[target] / "status.json"
    payload = json.loads(p.read_text())
    payload[field] = value
    write_json(p, payload)
    with pytest.raises(ValueError):
        validate_prerequisites(*gates)


def test_rejects_missing_required_r0_check_even_if_summary_passes(gates):
    import json

    from src.next_stage_runtime import validate_prerequisites

    p = gates[0] / "closure_checks.json"
    value = json.loads(p.read_text())
    del value["cross_split"]
    write_json(p, value)
    seal(gates[0])
    with pytest.raises(ValueError, match="R0"):
        validate_prerequisites(*gates)


def test_rejects_changed_bound_original_file(gates):
    from src.next_stage_runtime import validate_prerequisites

    p = gates[1] / "production_path_validation.json"
    p.write_text(p.read_text() + " ")
    seal(gates[1])
    with pytest.raises(ValueError, match="binding"):
        validate_prerequisites(*gates)


def test_rejects_new_config_even_if_model_name_is_same(gates):
    from src.next_stage_runtime import validate_config_against_gate, validate_prerequisites

    gate = validate_prerequisites(*gates)
    proposed = copy.deepcopy(gate["config"])
    proposed["generation_proposed_N"]["top_p"] = 0.9
    with pytest.raises(ValueError, match="configuration"):
        validate_config_against_gate(proposed, gate)


def test_rejects_missing_certificate_thresholds(gates):
    import json

    from src.next_stage_runtime import validate_prerequisites

    path = gates[1] / "production_path_validation.json"
    value = json.loads(path.read_text())
    del value["thresholds"]
    write_json(path, value)
    seal(gates[1])
    with pytest.raises(ValueError, match="threshold"):
        validate_prerequisites(*gates)


def test_rejects_missing_calibration_binding(gates):
    import json

    from src.next_stage_runtime import validate_prerequisites

    path = gates[0] / "closure_checks.json"
    value = json.loads(path.read_text())
    del value["dataset_manifest"]["files"]["calibration.jsonl"]
    write_json(path, value)
    seal(gates[0])
    with pytest.raises(ValueError, match="calibration"):
        validate_prerequisites(*gates)


def test_environment_matches_available_historical_lock_and_records_coverage(gates, monkeypatch):
    import src.next_stage_runtime as runtime

    gate = runtime.validate_prerequisites(*gates)
    monkeypatch.setattr(runtime.platform, "python_version", lambda: "3.12.14")
    monkeypatch.setattr(
        runtime.metadata, "version", lambda n: "5.14.1" if n == "transformers" else "observed"
    )
    value = runtime.validate_runtime_environment(gate, {"transformers_version": "5.14.1"})
    assert value["status"] == "PASS"
    assert value["historical_version_coverage"] == ["python", "transformers"]
    assert "torch" in value["versions_not_recorded_in_original_R1"]
    assert value["observed_packages"]["torch"] == "observed"


@pytest.mark.parametrize("kind", ["python", "installed_transformers", "adapter_transformers"])
def test_environment_rejects_changed_version(gates, monkeypatch, kind):
    import src.next_stage_runtime as runtime

    gate = runtime.validate_prerequisites(*gates)
    monkeypatch.setattr(
        runtime.platform, "python_version", lambda: "different" if kind == "python" else "3.12.14"
    )
    monkeypatch.setattr(
        runtime.metadata,
        "version",
        lambda n: "different" if kind == "installed_transformers" else "5.14.1",
    )
    audit = {"transformers_version": "different" if kind == "adapter_transformers" else "5.14.1"}
    with pytest.raises(ValueError, match="environment"):
        runtime.validate_runtime_environment(gate, audit)
