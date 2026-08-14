"""Fail-closed Pilot A/B execution boundary."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from compbias.io.yaml_config import load_yaml_mapping, reject_unknown_fields

_TOP = frozenset(
    {
        "schema_version",
        "stage",
        "paths_config",
        "model_config",
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
    return dict(raw)


def main_for_stage(stage: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Fail-closed {stage} launcher")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_stage_config(args.config, stage)
        if not args.execute:
            print(json.dumps({"stage": stage, "ready_to_execute": False, "config_valid": True}))
            print("BLOCKED: pass --execute on the reviewed GPU server after smoke approval")
            return 2
        if os.environ.get("COMPBIAS_GPU_EXECUTION_ACK") != "I_UNDERSTAND_THIS_STARTS_GPU_TRAINING":
            raise RuntimeError("COMPBIAS_GPU_EXECUTION_ACK is missing")
        from .training import run_grpo_stage

        run_grpo_stage(config)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 3
    return 0
