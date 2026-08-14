#!/usr/bin/env python3
"""Validate and emit a pinned Qwen-VL veRL/GRPO execution plan."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_TOP_LEVEL_FIELDS = frozenset(
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
_SECTION_FIELDS = {
    "model": frozenset({"name", "revision"}),
    "training": frozenset(
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
    ),
    "reward": frozenset({"outcome_only", "perception_reward_weight", "process_reward_weight"}),
    "audit": frozenset(
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
    ),
    "gate": frozenset(
        {
            "acknowledge_large_gpu_run",
            "require_cuda_device_audit",
            "require_verl_api_audit",
            "minimum_parse_rate",
            "require_image_hidden_intervention",
        }
    ),
}
_MAX_YAML_DEPTH = 64
_MAX_YAML_NODES = 100_000


def _validated_private_plan_path(path: Path) -> Path:
    """Allow plans only in the ignored private subtree or the system temp root."""

    from compbias.io.artifact_paths import validated_artifact_path

    repository_root = Path(__file__).resolve().parents[1]
    target = validated_artifact_path(
        path,
        repository_root=repository_root,
        label="VLM plan output",
        suffix=(".yaml", ".yml"),
    )
    private_root = repository_root / "artifacts/logs/private_vlm_plans"
    if repository_root in target.parents and private_root not in target.parents:
        raise ValueError(
            "VLM plan output inside the repository must stay under artifacts/logs/private_vlm_plans"
        )
    return target


def _write_new_plan(path: Path, payload: str) -> None:
    """Atomically publish a new private plan without following or replacing files."""

    target = _validated_private_plan_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _validated_private_plan_path(target)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def _load_unique_yaml(path: Path):
    """Load bounded safe YAML while rejecting duplicate and unknown keys."""

    from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

    loaded = load_yaml_mapping(path, label="VLM configuration")
    pending: list[tuple[object, int]] = [(loaded, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if depth > _MAX_YAML_DEPTH or nodes > _MAX_YAML_NODES:
            raise ValueError("VLM configuration exceeds the permitted depth or complexity")
        if isinstance(current, dict):
            pending.extend((value, depth + 1) for value in current.values())
        elif isinstance(current, list):
            pending.extend((value, depth + 1) for value in current)
    reject_unknown_fields(loaded, _TOP_LEVEL_FIELDS, label="configuration")
    for section, fields in _SECTION_FIELDS.items():
        if section in loaded:
            reject_unknown_fields(loaded[section], fields, label=section)
    audit = loaded.get("audit")
    if isinstance(audit, dict) and "dockerfile_packages" in audit:
        reject_unknown_fields(
            audit["dockerfile_packages"],
            {"torch", "transformers", "vllm"},
            label="audit.dockerfile_packages",
        )
    return loaded


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="VLM YAML configuration")
    parser.add_argument(
        "--acknowledge-large-gpu-run",
        action="store_true",
        help="explicitly acknowledge the reviewed large-GPU stage",
    )
    parser.add_argument(
        "--phase-d-audit",
        type=Path,
        required=True,
        help="Phase-D JSON audit with human sign-off bound to the accepted manifest",
    )
    parser.add_argument("--phase-d-audit-sha256", help="reviewed Phase-D artifact hash")
    parser.add_argument(
        "--execution-audit",
        type=Path,
        required=True,
        help="schema-v2 target execution evidence JSON",
    )
    parser.add_argument("--execution-audit-sha256", help="reviewed execution artifact hash")
    parser.add_argument(
        "--output-config",
        type=Path,
        required=True,
        help=(
            "required private YAML plan destination under "
            "artifacts/logs/private_vlm_plans/ or the system temp root"
        ),
    )
    args = parser.parse_args(argv)

    # Keep the acknowledgement gate ahead of config libraries and every heavy
    # framework.  This command never starts training or downloads a model.
    if not args.acknowledge_large_gpu_run:
        parser.error(
            "explicit acknowledgement required: pass --acknowledge-large-gpu-run "
            "after reviewing GPU cost and pinned revisions"
        )
    if args.phase_d_audit_sha256 is None or args.execution_audit_sha256 is None:
        parser.error("reviewed SHA-256 values are required for both gate evidence artifacts")

    import yaml

    from compbias.models.qwen_vl import (
        DEFAULT_MODEL_NAME,
        PINNED_MODEL_REVISION,
        PINNED_TRANSFORMERS_REVISION,
        PINNED_VERL_REVISION,
        PINNED_VLLM_REVISION,
        VLMPreflightConfig,
    )
    from compbias.rl.verl_entrypoints import (
        AUDITED_GRPO_LEAF_KEYS,
        build_grpo_execution_plan,
        load_execution_gate_evidence,
    )

    try:
        raw = _load_unique_yaml(args.config) or {}
        if not isinstance(raw, dict):
            raise ValueError("configuration must be a YAML mapping")
        model = raw.get("model")
        training = raw.get("training")
        reward = raw.get("reward")
        audit = raw.get("audit")
        if not all(isinstance(section, dict) for section in (model, training, reward, audit)):
            raise ValueError("model, training, reward, and audit sections must be mappings")
        if training.get("stage") != "joint_outcome_rl":
            raise ValueError("training.stage must be joint_outcome_rl")
        if training.get("framework") != "verl" or training.get("algorithm") != "grpo":
            raise ValueError("training must select the audited veRL/GRPO route")
        expected_reward = {
            "outcome_only": True,
            "perception_reward_weight": 0.0,
            "process_reward_weight": 0.0,
        }
        if reward != expected_reward:
            raise ValueError("reward must match the registered outcome-only contract exactly")
        if audit.get("audited_output_keys") != list(AUDITED_GRPO_LEAF_KEYS):
            raise ValueError("config audit keys differ from the strict 16-key whitelist")
        evidence = load_execution_gate_evidence(
            args.phase_d_audit,
            args.execution_audit,
            stage="joint_outcome_rl",
            phase_d_sha256=args.phase_d_audit_sha256,
            execution_audit_sha256=args.execution_audit_sha256,
        )
        config = VLMPreflightConfig(
            model_name=model.get("name", DEFAULT_MODEL_NAME),
            model_revision=model.get("revision", PINNED_MODEL_REVISION),
            transformers_revision=training.get(
                "transformers_revision", PINNED_TRANSFORMERS_REVISION
            ),
            verl_revision=training.get("verl_revision", PINNED_VERL_REVISION),
            vllm_revision=training.get("vllm_revision", PINNED_VLLM_REVISION),
            acknowledge_large_gpu_run=True,
            verl_api_audited=True,
        )
        plan = build_grpo_execution_plan(
            config,
            evidence=evidence,
            learning_rate=training.get("learning_rate", 1.0e-6),
            mini_batch_size=training.get("ppo_mini_batch_size", 16),
            rollout_samples=training.get("rollout_samples", 8),
            experiment_name=raw.get("experiment", "qwen25_vl_grpo"),
        )
    except (OSError, TypeError, ValueError, RuntimeError, yaml.YAMLError) as error:
        parser.error(str(error))

    payload = yaml.safe_dump(plan.to_mapping(), sort_keys=True)
    try:
        _write_new_plan(args.output_config, payload)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
