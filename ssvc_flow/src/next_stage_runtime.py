"""Shared immutable R0/R1 evidence gates for the actual downstream runners."""

from __future__ import annotations

import copy
import json
import platform
from importlib import metadata
from pathlib import Path

from .core import PROJECT_ROOT, canonical_hash, file_hash
from .finalize_r0 import verify_manifest
from .r1_reference_smoke import CERTIFICATE_CHECKS, PATH

REQUIRED_R0_CHECKS = frozenset(
    {
        "dataset_manifest",
        "base_N",
        "base_L",
        "selection",
        "images",
        "cross_split",
        "symbolic_prompts",
        "processor_lock_N",
        "processor_lock_L",
        "token_decode_N",
        "token_decode_L",
        "inputs_N",
        "inputs_L",
        "human_review",
        "calibration_processor",
    }
)
FROZEN_PATH_SOURCES = (
    "src/model_adapters/base.py",
    "src/model_adapters/qwen35.py",
    "src/prompts.py",
    "src/grpo_update.py",
    "src/verifiers.py",
    "src/optimizer_fork.py",
)
DERIVED_CONFIG_KEYS = frozenset(
    {
        "data_root",
        "max_new_tokens",
        "parity_alarm_mean_abs_token_logp",
        "parity_alarm_p99_abs_token_logp",
        "parity_alarm_min_top1_agreement",
    }
)


def _read(path):
    return json.loads(Path(path).read_text())


def _stage(root, phase, kind, required):
    files = verify_manifest(root, required)
    status = _read(root / "status.json")
    if (
        status.get("status") != "PASS"
        or status.get("phase") != phase
        or status.get("execution_kind") != kind
    ):
        raise ValueError(f"{phase} requires PASS {kind} evidence")
    return status, files


def validate_prerequisites(r0_dir, r1_run, supplement_dir):
    """Return the original runtime config/certificate and hash-bound gate summary.

    This validates evidence, not a newly loaded model. A real runner must also
    call _certificate_check on its actual adapter before generating any output.
    Relocated local evidence is allowed; references into the original R1 root
    are mapped by relative path, never guessed by filename.
    """
    r0, r1, supplement = (Path(p).resolve() for p in (r0_dir, r1_run, supplement_dir))
    r0_status, r0_files = _stage(
        r0,
        "R0",
        "CPU_AUDIT",
        ("identity.json", "closure_checks.json", "baseline_lock_N.json", "baseline_lock_L.json"),
    )
    checks = _read(r0 / "closure_checks.json")
    if (
        r0_status.get("details", {}).get("R0_gate_passed") is not True
        or not REQUIRED_R0_CHECKS.issubset(checks)
        or any(v.get("status") != "PASS" for v in checks.values())
    ):
        raise ValueError("R0 is missing a required passing check")
    original, r1_files = _stage(
        r1,
        "R1",
        "REAL_CUDA_TRAINING_SMOKE",
        ("production_path_validation.json", "training_smoke.json", "environment_lock.json"),
    )
    if original.get("details", {}).get("R1_gate_passed") is not True:
        raise ValueError("R1 production gate is incomplete")
    for name in FROZEN_PATH_SOURCES:
        if file_hash(PROJECT_ROOT / name) != original.get("source_files", {}).get(name):
            raise ValueError(f"Certified production implementation changed: {name}")
    certificate = _read(r1 / "production_path_validation.json")
    smoke = _read(r1 / "training_smoke.json")
    config = smoke["runtime_lock"]["config"]
    environment = _read(r1 / "environment_lock.json")
    if (
        environment.get("status") != "PASS"
        or environment.get("execution_kind") != "REAL_CUDA_INFERENCE"
        or environment.get("config") != config
        # audit_r1_runtime stores file_hash(args.config), not a JSON object hash.
        or environment.get("config_sha256") != file_hash(PROJECT_ROOT / "configs/next_stage.yaml")
        or not environment.get("python")
        or not environment.get("model", {}).get("transformers_version")
        or environment["model"]["transformers_version"]
        != certificate.get("model_audit", {}).get("transformers_version")
    ):
        raise ValueError("R1 environment lock is missing or differs from its certificate/config")
    if (
        smoke.get("status") != "PASS"
        or smoke.get("passed") is not True
        or smoke.get("raw_sample_count") != 128
    ):
        raise ValueError("Original four-step smoke has not passed")
    if (
        certificate.get("status") != "PASS"
        or certificate.get("selected_path") != PATH
        or config["model"]["id"] != "Qwen/Qwen3.5-9B"
        or config["model"]["revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        or certificate.get("model_revision") != config["model"]["revision"]
    ):
        raise ValueError("Original model and production path are not qualified")
    boundary = certificate.get("fixed_sequence_boundaries", {})
    if (
        boundary.get("passed") is not True
        or boundary.get("checks_completed") != 27
        or boundary.get("checks_expected") != 27
        or boundary.get("not_model_generated_not_training") is not True
    ):
        raise ValueError("R1 fixed-sequence boundary checks are incomplete")
    for name in CERTIFICATE_CHECKS:
        check = certificate.get("checks", {}).get(name, {})
        if (
            check.get("passed") is not True
            or check.get("sequences") != 120
            or check.get("failed_sequence_checks") != 0
            or (name in ("reference_repeat", "behavior") and check.get("top1_measured") is not True)
        ):
            raise ValueError(f"R1 reference check is incomplete: {name}")
    thresholds = certificate.get("thresholds", {})
    expected_thresholds = {name for name in DERIVED_CONFIG_KEYS if name.startswith("parity_alarm_")}
    if set(thresholds) != expected_thresholds:
        raise ValueError("R1 certificate threshold fields are incomplete")
    for name, value in thresholds.items():
        if config["R1"].get(name) != value:
            raise ValueError("R1 threshold configuration binding differs")
    _, supplemental_files = _stage(
        supplement, "R1_SUPPLEMENT", "REAL_CUDA_FORK", ("input_binding.json", "supplement.json")
    )
    result = _read(supplement / "supplement.json")
    if (
        result.get("status") != "PASS"
        or result.get("passed") is not True
        or result.get("execution_kind") != "REAL_CUDA_FORK"
        or result.get("new_rollouts") != 0
        or result.get("frozen_base_unchanged") is not True
        or result.get("original_files_unchanged") is not True
        or any(
            result.get("scratch_restoration", {}).get(k) is not True
            for k in ("parameters", "optimizer", "rng", "empty_optimizer")
        )
        or any(
            result.get(k, {}).get("passed") is not True
            for k in ("zero_gradient_adam", "lambda_zero_control", "candidate_order")
        )
    ):
        raise ValueError("R1 supplemental controls or restoration are incomplete")
    binding = _read(supplement / "input_binding.json")
    if (
        binding.get("original_config") != config
        or binding.get("original_config_sha256") != canonical_hash(config)
        or binding.get("certificate_sha256") != canonical_hash(certificate)
        or binding.get("model_revision") != config["model"]["revision"]
        or binding.get("initial_adapter_hash") != certificate["initial_adapter_hash"]
        or binding.get("frozen_parameter_hash")
        != certificate["model_audit"]["frozen_parameter_hash"]
    ):
        raise ValueError("R1 supplemental input binding differs from original evidence")
    old_root = Path(binding["original_run"])
    mapped = {}
    for original_path, digest in binding.get("original_file_hashes", {}).items():
        path = Path(original_path)
        if path.is_relative_to(old_root):
            relative = path.relative_to(old_root)
            candidate = (r1 / relative).resolve()
            if (
                not candidate.is_relative_to(r1)
                or not candidate.is_file()
                or file_hash(candidate) != digest
            ):
                raise ValueError(f"R1 original file binding mismatch: {relative}")
            mapped[str(relative)] = digest
    if mapped.get("production_path_validation.json") != r1_files["production_path_validation.json"]:
        raise ValueError("R1 certificate file binding is missing")
    dataset = checks["dataset_manifest"]
    calibration_hash = dataset.get("files", {}).get("calibration.jsonl", {}).get("sha256")
    if not calibration_hash:
        raise ValueError("R0 calibration file binding is missing")
    return {
        "certificate": certificate,
        "config": copy.deepcopy(config),
        "environment_lock": environment,
        "binding": {
            "status": "PASS",
            "r0_dir": str(r0),
            "r1_run": str(r1),
            "supplement_dir": str(supplement),
            "r0_files": r0_files,
            "r1_files": r1_files,
            "supplement_files": supplemental_files,
            "data_manifest_sha256": dataset["manifest_sha256"],
            "calibration_sha256": calibration_hash,
            "dataset_files": dataset["files"],
            "original_config_hash": canonical_hash(config),
            "original_environment_hash": canonical_hash(environment),
            "source_implementation_hashes": {
                n: original["source_files"][n] for n in FROZEN_PATH_SOURCES
            },
        },
    }


def validate_config_against_gate(config, gate):
    """Allow only known runtime-derived keys to be absent from the original YAML."""
    expected = gate["config"]

    def without_derived(d):
        return {k: v for k, v in d.items() if k not in DERIVED_CONFIG_KEYS}

    if without_derived(config) != without_derived(expected):
        raise ValueError("Downstream configuration differs from the original locked protocol")
    for key in DERIVED_CONFIG_KEYS & config.keys():
        actual, wanted = config[key], expected.get(key)
        if key == "data_root" and wanted is not None:
            actual, wanted = str(Path(actual).resolve()), str(Path(wanted).resolve())
        if actual != wanted:
            raise ValueError(f"Derived configuration differs from its original lock: {key}")
    return copy.deepcopy(expected)


def validate_runtime_environment(gate, adapter_audit=None):
    """Compare recorded R1 versions and disclose its incomplete package history.

    The original R1 artifact records Python and Transformers only. Other current
    versions are saved for exact downstream resume comparisons, not presented as
    historically verified. This performs no installs or environment changes.
    """
    original = gate["environment_lock"]
    packages = (
        "transformers",
        "torch",
        "peft",
        "tokenizers",
        "accelerate",
        "huggingface-hub",
        "numpy",
        "Pillow",
        "safetensors",
    )
    observed = {}
    for name in packages:
        try:
            observed[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise ValueError(f"Runtime environment package missing: {name}") from exc
    python = platform.python_version()
    expected_transformers = original["model"]["transformers_version"]
    if (
        python != original["python"]
        or observed["transformers"] != expected_transformers
        or (
            adapter_audit is not None
            and adapter_audit.get("transformers_version") != expected_transformers
        )
    ):
        raise ValueError("Runtime environment differs from recorded R1 Python/Transformers")
    return {
        "status": "PASS",
        "python": python,
        "observed_packages": observed,
        "historical_version_coverage": ["python", "transformers"],
        "versions_not_recorded_in_original_R1": [n for n in packages if n != "transformers"],
        "original_environment_sha256": canonical_hash(original),
    }


def validate_r2_gate(r2_dir, gate):
    """Bind completed real R2 coverage, frozen state and its original R0/R1 gate."""
    from collections import Counter

    from .r2_inputs import CONDITIONS, LONG_CONDITIONS
    from .r2_runtime import validate_existing_rows

    root = Path(r2_dir).resolve()
    _, files = _stage(
        root,
        "R2",
        "REAL_CUDA_INFERENCE",
        (
            "runtime_lock.json",
            "gate_binding.json",
            "data_binding.json",
            "request_manifest.json",
            "samples.jsonl",
            "diagnostic_rollouts.jsonl",
            "inference_audit.json",
            "condition_metrics.json",
            "paired_condition_effects.csv",
            "invalid_taxonomy.csv",
        ),
    )
    runtime, audit = _read(root / "runtime_lock.json"), _read(root / "inference_audit.json")
    if (
        _read(root / "gate_binding.json") != gate["binding"]
        or runtime.get("config") != gate["config"]
    ):
        raise ValueError("R2 is bound to a different original evidence/configuration gate")
    if (
        runtime.get("selected_probability_path") != PATH
        or runtime.get("initial_adapter_hash") != gate["certificate"]["initial_adapter_hash"]
        or runtime.get("base_parameter_hash")
        != gate["certificate"]["model_audit"]["frozen_parameter_hash"]
        or audit.get("passed") is not True
        or audit.get("raw_sample_count") != 2232
        or audit.get("optimizer_steps") != 0
        or audit.get("backward_calls") != 0
        or any(
            not audit.get(f"{name}_before") or audit[f"{name}_before"] != audit.get(f"{name}_after")
            for name in ("base_hash", "adapter_hash", "all_parameter_hash")
        )
        or audit.get("adapter_hash_before") != runtime.get("initial_adapter_hash")
        or audit.get("base_hash_before") != runtime.get("base_parameter_hash")
    ):
        raise ValueError("R2 original model/adapter or frozen inference audit is incomplete")
    data = _read(root / "data_binding.json")
    if (
        data.get("calibration_sha256") != gate["binding"]["calibration_sha256"]
        or data.get("dataset_manifest_sha256") != gate["binding"]["data_manifest_sha256"]
    ):
        raise ValueError("R2 data no longer binds the passed R0 dataset")
    request_manifest = _read(root / "request_manifest.json")
    requests = request_manifest["requests"]
    if request_manifest.get("count") != 2232 or len(requests) != 2232:
        raise ValueError("R2 request coverage is incomplete")
    expected = Counter({(c, "sample"): 288 for c in CONDITIONS})
    expected.update({(c, "greedy"): 72 for c in CONDITIONS})
    expected.update({(c, "sample"): 24 for c in LONG_CONDITIONS})
    expected.update({(c, "greedy"): 12 for c in LONG_CONDITIONS})
    if Counter((r["condition"], r["decode_mode"]) for r in requests) != expected:
        raise ValueError("R2 condition/mode request allocation changed")
    records = {}
    with (root / "samples.jsonl").open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("R2 has a truncated ledger tail")
            row = json.loads(line)
            key = row["sample_key"]
            if (
                key in records
                or row.get("execution_kind") != "REAL_CUDA_INFERENCE"
                or row.get("split") != "calibration"
            ):
                raise ValueError("R2 has duplicate, fake or wrong-split records")
            records[key] = row
    if len(records) != 2232 or set(records) != {r["sample_key"] for r in requests}:
        raise ValueError("R2 generated output coverage is incomplete")
    validate_existing_rows(records, requests)
    if files["samples.jsonl"] != files["diagnostic_rollouts.jsonl"]:
        raise ValueError("R2 named raw artifact differs from its durable ledger")
    return {
        "status": "PASS",
        "r2_dir": str(root),
        "r2_files": files,
        "execution_kind": "REAL_CUDA_INFERENCE",
        "raw_sample_count": 2232,
        "environment": runtime["environment"],
        "runtime_lock_sha256": files["runtime_lock.json"],
    }


def validate_r3_cold_gate(r3_cold_dir, gate, r2_binding):
    """Require fully bound real cold-fork evidence before formal training."""
    from .r3_gate import validate_r3_cold_gate as validate

    return validate(r3_cold_dir, gate, r2_binding)


def validate_r4_gate(r4_dir, gate, r2_binding, r3_binding):
    """Require complete two-arm evidence and the preserved step64 Adam state."""
    from .r4_gate import validate_r4_gate as validate

    return validate(r4_dir, gate, r2_binding, r3_binding)
