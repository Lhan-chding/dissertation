"""Validate the immutable, preregistered qualification protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs/modeling_qualification/design/protocol.json"


def load_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> dict[str, int]:
    """Reject silent changes to the locked experiment, including thresholds/splits.

    A changed scientific protocol needs a separately versioned design. The smoke
    profile is already part of this protocol and does not alter core acceptance.
    """
    if not isinstance(config, dict):
        raise ValueError("protocol must be an object")
    expected = json.loads(DESIGN.read_text(encoding="utf-8"))
    if config != expected:
        differences = sorted(
            key for key in config.keys() | expected.keys() if config.get(key) != expected.get(key)
        )
        raise ValueError(f"locked protocol mismatch: {differences}")
    core = config["profiles"]["core"]
    seeds = [s for values in core["seeds_by_role"].values() for s in values]
    if len(seeds) != len(set(seeds)):
        raise ValueError("protocol split seeds overlap")
    trajectories = len(seeds) * len(core["arms"])
    anchors = trajectories * len(core["anchors"])
    banks = sum(len(core[f"{role}_bank_indices"]) for role in ("fit", "diagnostic", "evaluation"))
    main = trajectories * core["steps"]
    forks = anchors * banks * len(config["fork_interventions"])
    derived = {
        "seed_count": len(seeds),
        "trajectory_count": trajectories,
        "main_optimizer_steps": main,
        "main_sampled_actions": main * core["B"] * core["K"],
        "anchor_count": anchors,
        "fork_optimizer_steps": forks,
        "fork_base_sampled_actions": anchors * banks * core["B"] * core["K"],
        "total_optimizer_steps": main + forks,
        "total_sampled_finite_actions": (main + anchors * banks) * core["B"] * core["K"],
    }
    if derived != config["budget_derived"]:
        raise ValueError("protocol derived budget inconsistent")
    return derived


def cpu_environment() -> dict[str, str]:
    values = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    os.environ.update(values)
    return values
