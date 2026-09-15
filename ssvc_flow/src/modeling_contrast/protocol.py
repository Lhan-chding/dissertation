"""Immutable protocol, CPU environment and complete resource forecasts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs/modeling_contrast/design/protocol.json"
CONFIG_SHA256 = "c15f7e6e955fa4de09ce2688cc24f41b032d4177eaa42379ef79d17c53df900c"


def _frozen_bytes() -> bytes:
    data = DESIGN.read_bytes()
    if hashlib.sha256(data).hexdigest() != CONFIG_SHA256:
        raise ValueError("locked protocol design byte hash mismatch")
    return data


def load_config(path: Path | str = DESIGN) -> dict[str, Any]:
    data = Path(path).read_bytes()
    if data != _frozen_bytes():
        raise ValueError("locked protocol byte mismatch; do not rewrite or tune the config")
    config = json.loads(data)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> dict[str, int]:
    expected = json.loads(_frozen_bytes())

    def encode(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    if not isinstance(config, dict) or encode(config) != encode(expected):
        raise ValueError("locked protocol semantic mismatch")
    roles = config["data_roles"]
    role_names = (
        "observation_development_seeds",
        "legacy_selection_seeds",
        "fresh_calibration_seeds",
        "fresh_locked_test_seeds",
    )
    seeds = [seed for name in role_names for seed in roles[name]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("locked protocol seeds overlap")
    fresh = config["fresh_cpu"]
    trajectories = len(roles["fresh_calibration_seeds"] + roles["fresh_locked_test_seeds"]) * len(
        fresh["main_arms"]
    )
    branches = config["branches"]
    banks = sum(len(branches[f"{role}_banks"]) for role in ("fit", "diagnostic", "evaluation"))
    main = trajectories * fresh["steps"]
    branch_banks = trajectories * len(branches["anchors"]) * banks
    forks = branch_banks * len(branches["operations"])
    derived = dict(
        expected_trajectories=trajectories,
        main_updates=main,
        fork_updates=forks,
        total_optimizer_updates=main + forks,
        training_and_branch_actions=(main + branch_banks) * fresh["B"] * fresh["K"],
    )
    if any(fresh[key] != value for key, value in derived.items()):
        raise ValueError("locked protocol derived budget mismatch")
    return derived


def cpu_environment() -> dict[str, str]:
    """Disable accelerator visibility/downloads before importing model libraries."""
    values = {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    os.environ.update(values)
    torch = sys.modules.get("torch")
    if torch is not None:
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU-only protocol refuses an initialized CUDA context")
        torch.set_num_threads(1)
        torch.set_default_device("cpu")
    return values


def resource_gate(forecast: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    """Check total new-stage cost, including already consumed resources.

    The caller must supply a measured-smoke-derived forecast. Missing, negative
    or nonfinite dimensions cannot qualify. This does not silently rescale the
    scientific sample budget to fit a resource ceiling.
    """
    config = load_config() if config is None else config
    validate_config(config)
    limits = config["resources"]
    bounds = {
        "wall_seconds": limits["max_total_walltime_hours"] * 3600,
        "peak_ram_gib": limits["max_ram_gib"],
        "added_output_bytes": limits["max_added_output_gib"] * 2**30,
        "temporary_bytes": limits["max_temporary_gib"] * 2**30,
    }
    failures = []
    for name, maximum in bounds.items():
        value = forecast.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            failures.append(f"{name}: missing or invalid")
        elif value > maximum:
            failures.append(f"{name}: {value} exceeds {maximum}")
    return {
        "passed": not failures,
        "status": "PASS" if not failures else "RESOURCE_REVIEW_REQUIRED",
        "forecast": forecast,
        "limits": bounds,
        "reasons": failures,
    }
