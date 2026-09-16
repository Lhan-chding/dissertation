"""V4 configuration and one campaign record, without per-candidate gates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

VERSION = "modeling-v4-full-response-20260916"


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def atomic_json(path, value):
    """Publish a single JSON artifact, retaining the old complete file on errors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def validate_config(value):
    """Validate scientific identity and execution bounds; never mutate the caller."""
    config = copy.deepcopy(value)
    if config.get("version") != VERSION:
        raise ValueError("Expected the V4 full-response protocol")
    if config.get("events") != ["X", "S", "W", "I"]:
        raise ValueError("Event identity must remain X/S/W/I")
    operations = config["operations"]
    maximum, workers = operations["max_concurrent_gpus"], operations["default_workers"]
    if type(maximum) is not int or not 1 <= maximum <= 3:
        raise ValueError("V4 is authorized for at most three GPUs")
    if type(workers) is not int or not 1 <= workers <= maximum:
        raise ValueError("Default workers must fit the GPU allocation")
    if operations.get("online_ssvc") is not False:
        raise ValueError("Online SSVC is outside this experiment")
    if operations.get("inherited_q3_million_fit_gate_required") is not False:
        raise ValueError("The old Q3 campaign is not a V4 prerequisite")
    if config["statistics"].get("fixed_rho_gate") is not None:
        raise ValueError("V4 rho is a continuous diagnostic, not a sample exclusion gate")
    if config["qwen"]["generation"] != {
        "max_new_tokens": 64,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "thinking": False,
        "path": "validated generation/scoring path; optimize only after one parity smoke",
    }:
        raise ValueError("V4 generation policy differs from the registered protocol")
    if config["qwen"]["id"] != "Qwen/Qwen3.5-9B":
        raise ValueError("The model family must remain Qwen3.5-9B")
    if config["qwen"]["revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a":
        raise ValueError("The registered model revision changed")
    roles = config["qwen"]["seed_roles"]
    if set(roles) != {"development", "calibration", "test"}:
        raise ValueError("Development, calibration and independent test roles are required")
    seeds = [seed for group in roles.values() for seed in group]
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Source training seed roles must be disjoint")
    if config["observations"].get("separate_reference_rng") is not True:
        raise ValueError("Reference samples must be independent of predictor samples")
    for key in ("ridge_alpha", "rbf_bandwidth_multipliers"):
        values = config["models"][key]
        if not values or any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError(f"{key} must contain positive finite values")
    if config["models"].get("query_outcomes_for_fit") is not False:
        raise ValueError("Query reference outcomes cannot enter model fitting")
    if config["data_layout"].get("per_design_raw_copy") is not False:
        raise ValueError("Raw observations must be shared between designs")
    if config["qwen"]["continuous_track"].get("online_control") is not False:
        raise ValueError("Only offline tracking is registered")
    return config


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def plan(config):
    config = validate_config(config)
    return {
        "schema": "ssvc-v4-plan-1",
        "status": "PLANNED_NOT_EXECUTED",
        "config_hash": digest(config),
        "max_concurrent_gpus": config["operations"]["max_concurrent_gpus"],
        "default_workers": config["operations"]["default_workers"],
        "stage_dependencies": {
            "A": [],
            "B": [],
            "C": ["B"],
            "D": ["C"],
            "E": ["D_POINT_RESPONSE_RESOLVED"],
        },
        "first_map": copy.deepcopy(config["qwen"]["first_development_map"]),
        "first_map_origins": [
            {"seed": seed, "arm": "X_BASE", "step": step, "role": "development"}
            for seed in config["qwen"]["seed_roles"]["development"][:2]
            for step in config["qwen"]["primary_origins"]["steps"]
        ],
        "old_q3_required": False,
        "online_control": False,
        "gpu_executed": False,
        "measured_runtime": None,
        "derived_workload_is_not_measurement": copy.deepcopy(config["derived"]),
    }


def source_record(root):
    """Record git identity once; avoid scanning model weights or raw observations."""
    root = Path(root)

    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short")}
