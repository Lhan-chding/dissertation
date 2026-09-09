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
