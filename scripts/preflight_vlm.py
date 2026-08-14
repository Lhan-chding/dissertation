#!/usr/bin/env python3
"""Audit the pinned 3B-VLM boundary without downloading a model or starting training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

_MAX_YAML_BYTES = 1024 * 1024
_MAX_YAML_DEPTH = 64
_MAX_YAML_NODES = 100_000

_SFT_FIELDS = frozenset(
    {
        "experiment",
        "execution_status",
        "large_gpu_started",
        "model",
        "training",
        "gate",
        "compatibility_audit",
    }
)
_RL_FIELDS = frozenset(
    {
        "experiment",
        "execution_status",
        "large_gpu_started",
        "model",
        "training",
        "reward",
        "audit",
        "gate",
    }
)
_MODEL_FIELDS = frozenset({"name", "revision"})
_SFT_TRAINING_FIELDS = frozenset(
    {
        "stage",
        "framework",
        "transformers_revision",
        "verl_revision",
        "vllm_revision",
    }
)
_RL_TRAINING_FIELDS = frozenset(
    {
        "stage",
        "framework",
        "algorithm",
        "transformers_revision",
        "verl_revision",
        "vllm_revision",
        "learning_rate",
        "ppo_mini_batch_size",
        "rollout_samples",
    }
)
_SFT_GATE_FIELDS = frozenset(
    {
        "acknowledge_large_gpu_run",
        "require_cuda_device_audit",
        "require_verl_api_audit",
        "minimum_parse_rate",
    }
)
_RL_GATE_FIELDS = _SFT_GATE_FIELDS | {"require_image_hidden_intervention"}
_REWARD_FIELDS = frozenset({"outcome_only", "perception_reward_weight", "process_reward_weight"})
_COMPATIBILITY_FIELDS = frozenset(
    {
        "status",
        "official_release",
        "reason",
        "dockerfile_path_at_verl_revision",
        "dockerfile_url_at_verl_revision",
        "dockerfile_sha256",
        "dockerfile_packages",
        "prohibited_install",
        "permitted_verl_install_after_container_build",
        "upstream_dockerfile_reproducible_as_is",
        "hardened_descendant_vendor_pending",
    }
)
_RL_AUDIT_FIELDS = frozenset(
    {
        "verl_version_at_revision",
        "configuration_builder",
        "official_grpo_documentation",
        "official_grpo_key_source_at_revision",
        "official_release",
        "compatibility_status",
        "stale_setup_py_vllm_extra",
        "upstream_packaging_note",
        "dockerfile_path_at_verl_revision",
        "dockerfile_url_at_verl_revision",
        "dockerfile_sha256",
        "dockerfile_packages",
        "prohibited_install",
        "permitted_verl_install_after_container_build",
        "upstream_dockerfile_reproducible_as_is",
        "hardened_descendant_vendor_pending",
        "audited_output_keys",
    }
)
_DOCKERFILE_PACKAGE_FIELDS = frozenset({"torch", "transformers", "vllm"})


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _closed_mapping(value: object, name: str, allowed_fields: frozenset[str]) -> dict[str, Any]:
    mapping = _mapping(value, name)
    unknown = tuple(sorted((key for key in mapping if key not in allowed_fields), key=repr))
    if unknown:
        rendered = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"{name} contains unknown field(s): {rendered}")
    return mapping


def _read_bounded_yaml(path: Path, name: str) -> str:
    try:
        with path.open("rb") as stream:
            encoded = stream.read(_MAX_YAML_BYTES + 1)
    except OSError as error:
        raise ValueError(f"cannot read {name}: {error}") from error
    if len(encoded) > _MAX_YAML_BYTES:
        raise ValueError(f"{name} exceeds the 1 MiB YAML size limit")
    try:
        return encoded.decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"cannot read {name}: {error}") from error


def _validate_yaml_complexity(text: str, name: str) -> None:
    import yaml

    depth = 0
    node_count = 0
    collection_starts = (yaml.MappingStartEvent, yaml.SequenceStartEvent)
    collection_ends = (yaml.MappingEndEvent, yaml.SequenceEndEvent)
    scalar_nodes = (yaml.ScalarEvent,)
    for event in yaml.parse(text):
        if isinstance(event, yaml.AliasEvent):
            raise ValueError(f"{name} must not contain YAML aliases")
        if isinstance(event, collection_starts):
            depth += 1
            node_count += 1
            if depth > _MAX_YAML_DEPTH:
                raise ValueError(f"{name} YAML nesting depth exceeds {_MAX_YAML_DEPTH}")
        elif isinstance(event, collection_ends):
            depth -= 1
        elif isinstance(event, scalar_nodes):
            node_count += 1
        if node_count > _MAX_YAML_NODES:
            raise ValueError(f"{name} YAML node count exceeds {_MAX_YAML_NODES}")


def _load_yaml(path: Path, name: str) -> dict[str, Any]:
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
        keys: set[object] = set()
        for key_node, _value_node in node.value:
            key = loader.construct_object(key_node, deep=False)
            if key in keys:
                raise ValueError(f"duplicate YAML key in {name}: {key!r}")
            keys.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )

    text = _read_bounded_yaml(path, name)
    try:
        _validate_yaml_complexity(text, name)
        value = yaml.load(text, Loader=UniqueKeyLoader)
    except RecursionError as error:
        raise ValueError(f"cannot parse {name}: YAML nesting exceeds safe parser limits") from error
    except yaml.YAMLError as error:
        raise ValueError(f"cannot parse {name}: {error}") from error
    return _mapping(value, name)


def _require_equal(actual: object, expected: object, name: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(
            f"{name} must have type {type(expected).__name__}; received {type(actual).__name__}"
        )
    if actual != expected:
        raise ValueError(f"{name} must match the pinned value {expected!r}; received {actual!r}")


def _validated_parse_rate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} minimum parse rate must be a finite number from 0.98 to 1.0")
    rate = float(value)
    if not math.isfinite(rate) or not 0.98 <= rate <= 1.0:
        raise ValueError(f"{name} minimum parse rate must be a finite number from 0.98 to 1.0")
    return rate


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_hardware() -> dict[str, Any]:
    torch_version = _installed_version("torch")
    if torch_version is None:
        return {
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_devices": [],
            "inspection_error": "torch_not_installed",
        }

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        devices = (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if cuda_available
            else []
        )
        return {
            "torch_version": str(torch.__version__),
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "gpu_devices": devices,
            "inspection_error": None,
        }
    except (ImportError, OSError, RuntimeError) as error:
        return {
            "torch_version": torch_version,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_devices": [],
            "inspection_error": f"{type(error).__name__}: {error}",
        }


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"output already exists; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    except ValueError as error:
        raise ValueError("JSON payload must contain only finite numbers") from error
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_preflight_output(path: Path) -> Path:
    """Allow private evidence only in its ignored subtree or the system temp root."""

    from compbias.io.artifact_paths import validated_artifact_path

    repository_root = Path(__file__).resolve().parents[1]
    output = validated_artifact_path(
        path,
        repository_root=repository_root,
        label="output",
        suffix=".json",
    )
    private_root = repository_root / "artifacts/logs/private_vlm_evidence"
    if repository_root in output.parents and private_root not in output.parents:
        raise ValueError(
            "VLM preflight evidence inside the repository must stay under "
            "artifacts/logs/private_vlm_evidence"
        )
    return output


def _build_report(sft_path: Path, rl_path: Path) -> dict[str, Any]:
    from compbias.models.qwen_vl import (
        DEFAULT_MODEL_NAME,
        PINNED_MODEL_REVISION,
        PINNED_TRANSFORMERS_REVISION,
        PINNED_VERL_DOCKERFILE_SHA256,
        PINNED_VERL_REVISION,
        PINNED_VLLM_REVISION,
    )

    sft = _closed_mapping(
        _load_yaml(sft_path, "SFT configuration"), "SFT configuration", _SFT_FIELDS
    )
    rl = _closed_mapping(_load_yaml(rl_path, "RL configuration"), "RL configuration", _RL_FIELDS)
    sft_model = _closed_mapping(sft.get("model"), "SFT model", _MODEL_FIELDS)
    rl_model = _closed_mapping(rl.get("model"), "RL model", _MODEL_FIELDS)
    sft_training = _closed_mapping(sft.get("training"), "SFT training", _SFT_TRAINING_FIELDS)
    rl_training = _closed_mapping(rl.get("training"), "RL training", _RL_TRAINING_FIELDS)
    reward = _closed_mapping(rl.get("reward"), "RL reward", _REWARD_FIELDS)
    sft_gate = _closed_mapping(sft.get("gate"), "SFT gate", _SFT_GATE_FIELDS)
    rl_gate = _closed_mapping(rl.get("gate"), "RL gate", _RL_GATE_FIELDS)
    compatibility = _closed_mapping(
        sft.get("compatibility_audit"),
        "SFT compatibility audit",
        _COMPATIBILITY_FIELDS,
    )
    rl_audit = _closed_mapping(rl.get("audit"), "RL audit", _RL_AUDIT_FIELDS)
    _closed_mapping(
        compatibility.get("dockerfile_packages"),
        "SFT Dockerfile packages",
        _DOCKERFILE_PACKAGE_FIELDS,
    )
    _closed_mapping(
        rl_audit.get("dockerfile_packages"),
        "RL Dockerfile packages",
        _DOCKERFILE_PACKAGE_FIELDS,
    )

    expected = {
        "model_name": DEFAULT_MODEL_NAME,
        "model_revision": PINNED_MODEL_REVISION,
        "transformers_revision": PINNED_TRANSFORMERS_REVISION,
        "verl_revision": PINNED_VERL_REVISION,
        "vllm_revision": PINNED_VLLM_REVISION,
    }
    _require_equal(
        sft.get("experiment"),
        "qwen25vl3b_structured_sft",
        "SFT experiment",
    )
    _require_equal(
        rl.get("experiment"),
        "qwen25vl3b_joint_outcome_grpo",
        "RL experiment",
    )
    for stage, model, training in (
        ("SFT", sft_model, sft_training),
        ("RL", rl_model, rl_training),
    ):
        _require_equal(model.get("name"), expected["model_name"], f"{stage} model name")
        _require_equal(model.get("revision"), expected["model_revision"], f"{stage} model revision")
        for key in ("transformers_revision", "verl_revision", "vllm_revision"):
            _require_equal(training.get(key), expected[key], f"{stage} {key.replace('_', ' ')}")

    _require_equal(sft_training.get("stage"), "structured_sft", "SFT training stage")
    _require_equal(sft_training.get("framework"), "verl", "SFT training framework")
    _require_equal(rl_training.get("stage"), "joint_outcome_rl", "RL training stage")
    _require_equal(rl_training.get("framework"), "verl", "RL training framework")
    _require_equal(rl_training.get("algorithm"), "grpo", "RL algorithm")
    _require_equal(rl_training.get("learning_rate"), 1.0e-6, "RL learning rate")
    _require_equal(rl_training.get("ppo_mini_batch_size"), 16, "RL PPO mini batch size")
    _require_equal(rl_training.get("rollout_samples"), 8, "RL rollout samples")

    for name, config, gate in (("SFT", sft, sft_gate), ("RL", rl, rl_gate)):
        _require_equal(config.get("execution_status"), "not_started", f"{name} status")
        _require_equal(config.get("large_gpu_started"), False, f"{name} GPU-start flag")
        _require_equal(gate.get("acknowledge_large_gpu_run"), False, f"{name} acknowledgement gate")

    _require_equal(
        sft_gate.get("require_cuda_device_audit"),
        True,
        "SFT require CUDA device audit",
    )
    _require_equal(
        sft_gate.get("require_verl_api_audit"),
        False,
        "SFT require veRL API audit",
    )
    _require_equal(
        rl_gate.get("require_cuda_device_audit"),
        True,
        "RL require CUDA device audit",
    )
    _require_equal(
        rl_gate.get("require_verl_api_audit"),
        True,
        "RL require veRL API audit",
    )
    _require_equal(
        rl_gate.get("require_image_hidden_intervention"),
        True,
        "RL require image hidden intervention",
    )

    sft_parse_rate = _validated_parse_rate(sft_gate.get("minimum_parse_rate"), "SFT")
    _validated_parse_rate(rl_gate.get("minimum_parse_rate"), "RL")

    expected_reward_contract = {
        "outcome_only": True,
        "perception_reward_weight": 0.0,
        "process_reward_weight": 0.0,
    }
    reward_contract = {key: reward.get(key) for key in expected_reward_contract}
    for key, expected_value in expected_reward_contract.items():
        _require_equal(reward_contract[key], expected_value, f"RL reward {key}")
    _require_equal(
        compatibility.get("dockerfile_sha256"),
        PINNED_VERL_DOCKERFILE_SHA256,
        "veRL Dockerfile SHA256",
    )
    _require_equal(
        compatibility.get("prohibited_install"),
        "pip install -e .[vllm]",
        "prohibited veRL install route",
    )
    _require_equal(
        compatibility.get("upstream_dockerfile_reproducible_as_is"),
        False,
        "upstream veRL Dockerfile reproducibility boundary",
    )
    _require_equal(
        compatibility.get("hardened_descendant_vendor_pending"),
        True,
        "hardened descendant container pending gate",
    )
    expected_dockerfile_packages = {
        "torch": "2.11.0",
        "transformers": PINNED_TRANSFORMERS_REVISION,
        "vllm": PINNED_VLLM_REVISION,
    }
    _require_equal(
        compatibility.get("dockerfile_packages"),
        expected_dockerfile_packages,
        "SFT Dockerfile packages",
    )
    _require_equal(
        compatibility.get("dockerfile_path_at_verl_revision"),
        "docker/Dockerfile.stable.vllm",
        "SFT Dockerfile path",
    )
    _require_equal(
        compatibility.get("permitted_verl_install_after_container_build"),
        "pip install --no-deps -e .",
        "permitted veRL install route",
    )
    for key, expected_value in (
        ("dockerfile_sha256", PINNED_VERL_DOCKERFILE_SHA256),
        ("dockerfile_packages", expected_dockerfile_packages),
        ("prohibited_install", "pip install -e .[vllm]"),
        ("permitted_verl_install_after_container_build", "pip install --no-deps -e ."),
        ("upstream_dockerfile_reproducible_as_is", False),
        ("hardened_descendant_vendor_pending", True),
    ):
        _require_equal(rl_audit.get(key), expected_value, f"RL audit {key.replace('_', ' ')}")

    hardware = _local_hardware()
    blockers = [
        "large_gpu_acknowledgement_not_granted",
        "hardened_descendant_container_pending",
        "container_sbom_pending",
        "container_vulnerability_policy_audit_pending",
    ]
    if not hardware["cuda_available"]:
        blockers.append("cuda_gpu_not_available_on_current_host")
    if compatibility.get("status") != "verified_on_target_hardware":
        blockers.append("pinned_container_target_hardware_smoke_pending")

    packages = {
        name: _installed_version(name)
        for name in ("numpy", "pillow", "pyyaml", "torch", "transformers", "verl", "vllm")
    }
    return {
        "schema_version": 1,
        "audit_completed": True,
        "audit_scope": "metadata_and_local_hardware_only",
        "large_gpu_started": False,
        "large_gpu_acknowledged": False,
        "model_download_attempted": False,
        "training_invoked": False,
        "ready_for_large_gpu_execution": not blockers,
        "blockers": blockers,
        "pins": expected,
        "reward_contract": reward_contract,
        "parser_validity_gate": sft_parse_rate,
        "parser_validity_measured_on_model": False,
        "image_hidden_intervention_required": bool(
            rl_gate.get("require_image_hidden_intervention")
        ),
        "compatibility": {
            "status": compatibility.get("status"),
            "dockerfile_path": compatibility.get("dockerfile_path_at_verl_revision"),
            "dockerfile_sha256": compatibility.get("dockerfile_sha256"),
            "dockerfile_packages": compatibility.get("dockerfile_packages"),
            "prohibited_install": compatibility.get("prohibited_install"),
            "permitted_verl_install": compatibility.get(
                "permitted_verl_install_after_container_build"
            ),
            "upstream_dockerfile_reproducible_as_is": compatibility.get(
                "upstream_dockerfile_reproducible_as_is"
            ),
            "hardened_descendant_vendor_pending": compatibility.get(
                "hardened_descendant_vendor_pending"
            ),
        },
        "local_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "hardware": hardware,
        },
        "next_authorized_action": (
            "vendor a hardened descendant Dockerfile with immutable inputs, then produce "
            "a bound image digest, SBOM, vulnerability-policy audit, and target-hardware "
            "smoke evidence; do not start SFT or RL"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-config", type=Path, required=True)
    parser.add_argument("--rl-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        output = _validated_preflight_output(args.output)
        if os.path.lexists(output):
            raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
        report = _build_report(args.sft_config, args.rl_config)
        output.parent.mkdir(parents=True, exist_ok=True)
        output = _validated_preflight_output(output)
        _atomic_json_write(output, report)
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
