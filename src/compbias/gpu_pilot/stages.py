"""Fail-closed Pilot A/B execution boundary."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

from .config import load_pilot_paths
from .execution_gate import resolve_project_file

_TOP = frozenset(
    {
        "schema_version",
        "stage",
        "paths_config",
        "model_config",
        "data_config",
        "dataset_manifest",
        "natural_records",
        "output_subdir",
        "training",
        "freeze",
        "claims",
    }
)
_TRAINING = frozenset(
    {
        "seed",
        "learning_rate",
        "max_steps",
        "num_generations",
        "gradient_accumulation_steps",
        "max_prompt_length",
        "max_completion_length",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
    }
)
_FREEZE = frozenset({"vision_tower", "visual_merger", "base_language", "trainable"})
_CLAIMS = frozenset({"allowed", "forbidden"})
_EXPECTED_TRAINING = {
    "pilot_a": {
        "seed": 20260814,
        "learning_rate": 0.00001,
        "max_steps": 100,
        "num_generations": 4,
        "gradient_accumulation_steps": 8,
        "max_prompt_length": 768,
        "max_completion_length": 256,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    },
    "pilot_b_lm_only": {
        "seed": 20260814,
        "learning_rate": 0.00001,
        "max_steps": 200,
        "num_generations": 4,
        "gradient_accumulation_steps": 8,
        "max_prompt_length": 1024,
        "max_completion_length": 256,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    },
}
_EXPECTED_CLAIMS = {
    "pilot_a": {
        "allowed": ["downstream_reasoning_change", "operational_compensation"],
        "forbidden": ["rl_changed_visual_error_distribution", "visual_acquisition_improved"],
    },
    "pilot_b_lm_only": {
        "allowed": [
            "operational_visual_readout_change",
            "reasoning_change",
            "compensation_change",
        ],
        "forbidden": ["visual_acquisition_improved", "unique_perception_reasoning_boundary"],
    },
}
_EXPECTED_STATIC = {
    "pilot_a": {
        "paths_config": "configs/paths.yaml",
        "model_config": "configs/model/qwen25vl3b.yaml",
        "data_config": "configs/data/cva_chart_pilot.yaml",
        "dataset_manifest": "data/generated/cva_chart_pilot_v0_1/manifest.json",
        "natural_records": "trajectories/natural/pilot_train_records.jsonl",
        "output_subdir": "pilot_a",
    },
    "pilot_b_lm_only": {
        "paths_config": "configs/paths.yaml",
        "model_config": "configs/model/qwen25vl3b.yaml",
        "data_config": "configs/data/cva_chart_pilot.yaml",
        "dataset_manifest": "data/generated/cva_chart_pilot_v0_1/manifest.json",
        "natural_records": "trajectories/natural/natural_records.jsonl",
        "output_subdir": "pilot_b_lm_only",
    },
}


def load_stage_config(path: Path, expected_stage: str) -> dict[str, object]:
    raw = load_yaml_mapping(path, label=f"GPU {expected_stage} configuration")
    reject_unknown_fields(raw, _TOP, label="GPU pilot stage configuration")
    if raw.get("schema_version") != 1 or raw.get("stage") != expected_stage:
        raise ValueError(f"configuration must declare schema_version 1 and stage {expected_stage}")
    for section, fields in (("training", _TRAINING), ("freeze", _FREEZE), ("claims", _CLAIMS)):
        value = raw.get(section)
        if not isinstance(value, Mapping):
            raise ValueError(f"{section} must be a mapping")
        reject_unknown_fields(value, fields, label=section)
    freeze = raw["freeze"]
    assert isinstance(freeze, Mapping)
    if dict(freeze) != {
        "vision_tower": True,
        "visual_merger": True,
        "base_language": True,
        "trainable": "language_lora",
    }:
        raise ValueError("Pilot A/B must freeze vision, merger, and base language weights")
    training = raw["training"]
    claims = raw["claims"]
    assert isinstance(training, Mapping) and isinstance(claims, Mapping)
    expected_training = _EXPECTED_TRAINING[expected_stage]
    if dict(training) != expected_training or any(
        isinstance(training[key], bool) or type(training[key]) is not type(value)
        for key, value in expected_training.items()
    ):
        raise ValueError(f"{expected_stage} training values must equal the registered budget")
    if dict(claims) != _EXPECTED_CLAIMS[expected_stage]:
        raise ValueError(f"{expected_stage} claims must equal the registered claim boundary")
    output_subdir = raw.get("output_subdir")
    if (
        not isinstance(output_subdir, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", output_subdir) is None
    ):
        raise ValueError("output_subdir must be a safe single directory name")
    if any(raw.get(key) != value for key, value in _EXPECTED_STATIC[expected_stage].items()):
        raise ValueError(f"{expected_stage} paths must equal the registered pilot paths")
    return dict(raw)


def main_for_stage(stage: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Fail-closed {stage} launcher")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        project_root = Path(__file__).resolve().parents[3]
        raw_config_path = args.config.expanduser().absolute()
        try:
            raw_metadata = raw_config_path.lstat()
        except OSError as error:
            raise RuntimeError("stage config cannot be inspected") from error
        if raw_config_path.is_symlink() or not stat.S_ISREG(raw_metadata.st_mode):
            raise RuntimeError("stage config must be a regular non-symlink file")
        config_path = raw_config_path.resolve()
        try:
            config_path.relative_to(project_root)
        except ValueError as error:
            raise RuntimeError("stage config must be inside the clean project checkout") from error
        config = load_stage_config(config_path, stage)
        if not args.execute:
            print(json.dumps({"stage": stage, "ready_to_execute": False, "config_valid": True}))
            print("BLOCKED: pass --execute on the reviewed GPU server after smoke approval")
            return 2
        if os.environ.get("COMPBIAS_GPU_EXECUTION_ACK") != "I_UNDERSTAND_THIS_STARTS_GPU_TRAINING":
            raise RuntimeError("COMPBIAS_GPU_EXECUTION_ACK is missing")
        paths_config = resolve_project_file(
            project_root,
            config.get("paths_config"),
            label="paths_config",
        )
        paths = load_pilot_paths(paths_config)
        if paths.project_root != project_root:
            raise RuntimeError("paths.project_root must equal the active clean project checkout")
        config = {
            **config,
            "validated_stage_config_path": str(config_path),
            "validated_paths_config_path": str(paths_config),
        }
        from .training import run_grpo_stage

        run_grpo_stage(config)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 3
    return 0
