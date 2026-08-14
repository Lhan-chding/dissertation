#!/usr/bin/env python3
"""Validate a v2 VLM regime and emit a fail-closed large-GPU execution plan."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from compbias.audit.frozen_components import VLMRegimeSpec
from compbias.io.artifact_paths import validated_artifact_path
from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields
from compbias.models.qwen_vl import DEFAULT_MODEL_NAME, PINNED_MODEL_REVISION

_CHECKPOINTS = ("base", "10pct", "25pct", "50pct", "75pct", "final")
_INTERFACES = (
    "natural_evidence_prefix",
    "structured_scene_json",
    "post_projector_activation",
    "mid_fusion_activation",
    "pre_answer_evidence_state",
)
_METRICS = (
    "c_sel",
    "c_fork",
    "c_syn",
    "D_P",
    "D_R",
    "Gamma",
    "selection_ratio",
    "epsilon_alg",
    "iid_ood_gap",
)
_MINIMUM_VRAM = {
    "lm_only": 24,
    "projector_lm": 32,
    "vision_lora": 48,
    "max_end_to_end": 80,
}
_TOP = frozenset(
    {
        "schema_version",
        "experiment",
        "model",
        "regime",
        "training",
        "interfaces",
        "resources",
        "gate",
        "output_plan",
    }
)
_FIELDS = {
    "model": frozenset({"name", "revision"}),
    "regime": frozenset({"id", "vision_update", "projector_update", "language_update"}),
    "training": frozenset(
        {
            "framework",
            "algorithm",
            "checkpoints",
            "seeds",
            "pilot_inputs",
            "mediators_per_input",
            "forks_per_mediator",
        }
    ),
    "resources": frozenset({"minimum_gpu_vram_gb", "precision"}),
    "gate": frozenset(
        {
            "execution_permitted",
            "require_phase_d_human_review",
            "require_authenticated_extension",
            "require_target_gpu_smoke",
        }
    ),
}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    return value


def _positive_int(value: object, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _seeds(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("training.seeds must be a sequence")
    if not 1 <= len(value) <= 32:
        raise ValueError("training.seeds must contain 1-32 seeds")
    seeds = tuple(value)
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1
        for seed in seeds
    ):
        raise ValueError("training.seeds must contain non-negative 32-bit integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training.seeds must be unique")
    return seeds


def _validated_output(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("output_plan must be a non-empty path string")
    repository_root = Path(__file__).resolve().parents[1]
    target = validated_artifact_path(
        value,
        repository_root=repository_root,
        label="v2 VLM execution plan",
        suffix=".json",
    )
    private_root = repository_root / "artifacts/logs/private_vlm_plans"
    if repository_root in target.parents and private_root not in target.parents:
        raise ValueError("repository VLM plans must stay under artifacts/logs/private_vlm_plans")
    return target


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def _build_plan(raw: Mapping[str, object]) -> tuple[Path, dict[str, object]]:
    reject_unknown_fields(raw, _TOP, label="configuration")
    if raw.get("schema_version") != 2:
        raise ValueError("schema_version must equal 2")
    experiment = raw.get("experiment")
    if not isinstance(experiment, str) or not experiment.startswith("qwen25vl3b_v2_"):
        raise ValueError("experiment must be a registered qwen25vl3b_v2 name")

    sections: dict[str, Mapping[str, object]] = {}
    for name, fields in _FIELDS.items():
        section = _mapping(raw.get(name), name)
        reject_unknown_fields(section, fields, label=name)
        sections[name] = section
    model = sections["model"]
    if model != {"name": DEFAULT_MODEL_NAME, "revision": PINNED_MODEL_REVISION}:
        raise ValueError("model name and revision must match the pinned Qwen2.5-VL-3B snapshot")

    regime_raw = sections["regime"]
    regime = VLMRegimeSpec(
        regime_id=str(regime_raw.get("id")),
        vision_update=str(regime_raw.get("vision_update")),
        projector_update=str(regime_raw.get("projector_update")),
        language_update=str(regime_raw.get("language_update")),
    )
    training = sections["training"]
    if training.get("framework") != "verl" or training.get("algorithm") != "grpo":
        raise ValueError("training must use the audited veRL/GRPO route")
    if tuple(training.get("checkpoints", ())) != _CHECKPOINTS:
        raise ValueError("training.checkpoints must equal the registered checkpoint schedule")
    seeds = _seeds(training.get("seeds"))
    pilot_inputs = _positive_int(training.get("pilot_inputs"), "pilot_inputs", maximum=10_000)
    mediators = _positive_int(
        training.get("mediators_per_input"), "mediators_per_input", maximum=64
    )
    forks = _positive_int(training.get("forks_per_mediator"), "forks_per_mediator", maximum=64)
    if pilot_inputs * mediators * forks * len(seeds) > 2_000_000:
        raise ValueError("registered pilot exceeds the 2,000,000-continuation safety budget")

    interfaces = raw.get("interfaces")
    if not isinstance(interfaces, list) or tuple(interfaces) != _INTERFACES:
        raise ValueError("interfaces must equal the registered multi-interface schedule")
    resources = sections["resources"]
    if resources.get("minimum_gpu_vram_gb") != _MINIMUM_VRAM[regime.regime_id]:
        raise ValueError("minimum_gpu_vram_gb does not match the registered regime")
    if resources.get("precision") != "bf16":
        raise ValueError("resources.precision must equal bf16")

    gate = sections["gate"]
    expected_gate = {
        "execution_permitted": False,
        "require_phase_d_human_review": True,
        "require_authenticated_extension": True,
        "require_target_gpu_smoke": True,
    }
    if dict(gate) != expected_gate:
        raise ValueError("large-GPU gate must remain fully enabled and execution-permitted=false")

    output = _validated_output(raw.get("output_plan"))
    plan = {
        "schema_version": 2,
        "artifact_type": "compbias_v2_large_gpu_execution_plan",
        "experiment": experiment,
        "model": dict(model),
        "regime": regime.to_mapping(),
        "interface_regimes": {
            interface: regime.interface_regime(interface) for interface in _INTERFACES
        },
        "checkpoints": list(_CHECKPOINTS),
        "seeds": list(seeds),
        "pilot": {
            "inputs": pilot_inputs,
            "mediators_per_input": mediators,
            "forks_per_mediator": forks,
        },
        "required_metrics": list(_METRICS),
        "minimum_gpu_vram_gb": resources["minimum_gpu_vram_gb"],
        "precision": "bf16",
        "requires_large_gpu": True,
        "execution_permitted": False,
        "large_gpu_started": False,
        "blockers": [
            "authenticated_execution_extension_not_implemented",
            "phase_d_human_review_required",
            "pinned_model_snapshot_not_verified_on_target",
            "hardened_container_sbom_vulnerability_policy_pending",
            "target_gpu_cuda_smoke_pending",
        ],
        "claim_boundary": (
            "frozen acquisition permits readout/reasoning claims only"
            if regime.acquisition_frozen
            else "trainable vision permits operational perception claims, not anatomical separation"
        ),
    }
    return output, plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = load_yaml_mapping(args.config, label="v2 VLM regime configuration")
        output, plan = _build_plan(raw)
        _write_new_json(output, plan)
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 3
    print(f"BLOCKED: large-GPU execution is not authenticated; plan written to {output}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
