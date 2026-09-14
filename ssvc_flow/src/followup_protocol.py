"""Frozen followup design, budgets and non-executing plan utilities.

Importing this module never imports torch or an adapter. Runtime authorizations
are invocation flags; a design file cannot grant them to itself.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_REFERENCE = PROJECT_ROOT / "docs/mechanism_followup/design/configs/followup_design.json"
PROTOCOL_VERSION = "mechanism-followup-v2-20260914"
BANK_IDS = ("bank00", "bank03", "bank04", "bank05", "bank11", "composite_no_x_plus_three_level")
ARMS = ("X_BASE", "X_VALID", "X_VALID_NO_X_OFF")


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=reject)


@dataclass(frozen=True)
class CandidateSpec:
    id: str
    policy: str
    auxiliary_weight: float

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {"id", "policy", "auxiliary_weight"}:
            raise ValueError("Candidate requires exactly id, policy, auxiliary_weight")
        name, policy, weight = value["id"], value["policy"], value["auxiliary_weight"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Invalid candidate id")
        if policy not in ("joint", "no_x_off"):
            raise ValueError("Unknown group reward policy")
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError("Auxiliary weight must be finite and nonnegative, never bool")
        return cls(name, policy, float(weight))


def validate_design(design):
    """Require the complete supplied design, including scalar types and budgets.

    Private paths are resolved in a separate server paths file. Changing the
    experimental design requires an explicit new version and new validation.
    """
    if not isinstance(design, dict):
        raise ValueError("Design must be a mapping")
    reference = strict_json(DESIGN_REFERENCE)

    def visit(actual, expected, path):
        if type(actual) is not type(expected):
            raise ValueError(f"Design type differs at {path}")
        if isinstance(expected, dict):
            if set(actual) != set(expected):
                raise ValueError(f"Design keys differ at {path}")
            for key in expected:
                visit(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, list):
            if len(actual) != len(expected):
                raise ValueError(f"Design length differs at {path}")
            for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
                visit(left, right, f"{path}[{i}]")
        elif actual != expected or (isinstance(actual, float) and not math.isfinite(actual)):
            raise ValueError(f"Frozen design differs at {path}")

    visit(design, reference, "design")
    for unit in design["S1"]["units"]:
        for candidate in unit["candidates"]:
            CandidateSpec.from_mapping(candidate)
    return copy.deepcopy(design)


def load_design(path):
    path = Path(path)
    if not str(path) or path.stat().st_size > 1_000_000:
        raise ValueError("Missing or oversized design")
    # The shipped .yaml is JSON (a YAML 1.2 subset). For ordinary YAML, SafeLoader
    # additionally rejects duplicate keys instead of silently replacing a lock.
    if path.read_text().lstrip().startswith("{"):
        value = strict_json(path)
    else:
        import yaml

        class UniqueLoader(yaml.SafeLoader):
            pass

        def construct(loader, node, deep=False):
            pairs = loader.construct_pairs(node, deep=deep)
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"Duplicate YAML key: {key}")
                result[key] = item
            return result

        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct)
        value = yaml.load(path.read_text(), Loader=UniqueLoader)
    return validate_design(value)


def source_files():
    return {
        str(path.relative_to(PROJECT_ROOT)): file_hash(path)
        for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
    }


def calculate_budgets(design):
    units = design["S1"]["units"]
    candidates = sum(len(unit["candidates"]) for unit in units)
    replays = sum(unit["baseline_replay_updates"] for unit in units)
    direct = sum(len(unit["direct_candidate_ids"]) for unit in units)
    evaluation = design["S1"]["direct_evaluation"]
    runs = len(design["S2"]["new_runs"])
    o = design["optimization"]
    return {
        "candidate_updates": candidates,
        "baseline_replay_updates": replays,
        "total_adam_calls": candidates + replays,
        "backward_sequences": (candidates + replays) * o["B"] * o["K"],
        "direct_logical_candidates": direct,
        "direct_outputs": direct * evaluation["prompts"] * evaluation["samples_per_prompt"],
        "s2_new_runs": runs,
        "s2_training_outputs": runs * design["S2"]["steps"] * o["B"] * o["K"],
        "s2_evaluation_outputs_per_run": (
            4 * 72 * 8 + 288 * 8 + 176 * 8 + 2 * 72 * 8 + 2 * 72 * 3 * 4
        ),
        "s2_evaluation_outputs": runs
        * (4 * 72 * 8 + 288 * 8 + 176 * 8 + 2 * 72 * 8 + 2 * 72 * 3 * 4),
        "historical_endpoint_diagnostic_outputs": 2 * 72 * 3 * 4,
        "optional_extension_outputs_authorized": 0,
    }


def build_plan(design):
    design = validate_design(design)
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PLANNED_NOT_EXECUTED",
        "design": design,
        "design_hash": canonical_hash(design),
        "source_files": source_files(),
        "budgets": calculate_budgets(design),
        "parent_metadata_verified": False,
        "parent_raw_verified": False,
        "gpu_smoke_passed": False,
        "gpu_started": False,
        "training_started": False,
        "safety_status": "NOT_CERTIFIED",
    }


def write_new_json(path, value):
    """Publish a complete JSON file exactly once, without following symlinks."""
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("Output path may not contain traversal")
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("Output path may not contain symbolic links")
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_slurm(*, stage):
    if stage not in ("S1", "S2", "SMOKE"):
        raise ValueError("Expected S1, S2, or SMOKE")
    commands = {
        "S1": (
            'run-s1 --bank "${SSVC_BANK:?Set one frozen bank ID}" '
            '--smoke-evidence "${SSVC_SMOKE_EVIDENCE:?}"'
        ),
        "S2": (
            'run-s2 --seed "${SSVC_SEED:?}" --arm "${SSVC_ARM:?}" '
            '--smoke-evidence "${SSVC_SMOKE_EVIDENCE:?}" '
            '--s1-evidence "${SSVC_S1_EVIDENCE:?}" --allow-training'
        ),
        "SMOKE": "smoke",
    }
    return (
        """#!/usr/bin/env bash
# Resource requests must be supplied using currently approved cluster resources.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
set -euo pipefail
umask 077
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || { echo 'A Slurm allocation is required.' >&2; exit 2; }
: "${SSVC_PYTHON:?Set verified Python executable}"
: "${SSVC_PROJECT:?Set project ssvc_flow directory}"
: "${SSVC_PLAN:?Set validated plan path}"
: "${SSVC_OUT:?Set fresh run directory or exact resume target}"
cd "$SSVC_PROJECT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
resume_args=()
if [[ ${SSVC_RESUME:-0} == 1 ]]; then resume_args=(--resume); fi
exec "$SSVC_PYTHON" -m src.followup_cli """
        + commands[stage]
        + """ \\
  --validated-plan "$SSVC_PLAN" --out "$SSVC_OUT" --allow-gpu-execution "${resume_args[@]}"
"""
    )
