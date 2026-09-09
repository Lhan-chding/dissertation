"""Hash-bound CPU phase records and configuration for next-stage-v1."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .core import canonical_hash, phase_artifacts, write_json

STAGES = ("R0", "R1", "R2", "R3-cold", "R4", "R3-warm", "R5")
STATUSES = ("NOT_STARTED", "RUNNING", "PASS", "FAIL", "BLOCKED", "INCONCLUSIVE")
EXECUTION_KINDS = (
    "CPU_AUDIT",
    "CPU_MATH",
    "REAL_CUDA_INFERENCE",
    "REAL_CUDA_TRAINING_SMOKE",
    "REAL_CUDA_FORK",
    "REAL_CUDA_TRAINING",
)


def load_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("next-stage config must be a mapping")
    if value["model"]["id"] != "Qwen/Qwen3.5-9B":
        raise ValueError("next-stage is restricted to Qwen3.5-9B")
    return value


def bind_run(out, identity, resume=False):
    out = Path(out)
    path = out / "identity.json"
    if path.exists():
        if not resume or json.loads(path.read_text()) != identity:
            raise ValueError(
                "existing stage requires --resume with identical input/config/source hashes"
            )
    elif out.exists() and any(out.iterdir()):
        raise ValueError("refusing an occupied stage without identity")
    else:
        if resume:
            raise ValueError("cannot resume a stage without identity")
        write_json(path, identity)


def stage_status(out, stage, status, execution_kind, details, artifacts=()):
    if stage not in STAGES or status not in STATUSES or execution_kind not in EXECUTION_KINDS:
        raise ValueError("unknown stage status or execution kind")
    if execution_kind.startswith("REAL_CUDA"):
        raise ValueError("CPU phase writer cannot assert CUDA execution")
    payload = phase_artifacts(out, stage, status, details, artifacts)
    payload["execution_kind"] = execution_kind
    write_json(Path(out) / "status.json", payload)
    (Path(out) / "report_zh.md").write_text((Path(out) / "report.md").read_text(), encoding="utf-8")
    return payload


def dry_run_plan(config):
    return {
        "protocol_version": config["protocol_version"],
        "model": config["model"]["id"],
        "execution_kind": "CPU_AUDIT",
        "dry_run": True,
        "generation_counts": {
            "R0": 0,
            "R1_reference": 120,
            "R1_smoke_max": 128,
            "R1_bridge_if_engine_changes": 576,
            "R2": 2160,
            "R2_long": 72,
            "R3_cold_bank": 384,
            "R3_cold_proposal": 768,
            "R4_train": 4096,
            "R4_new_eval": 9728,
            "R4_step0_if_no_reuse": 576,
            "R3_warm_bank": 384,
            "R3_warm_proposal": 768,
            "R3_warm_direct": 3072,
        },
        "max_new_tokens_N": 64,
        "cost_status": "requires measured R1 cost",
        "automatic_sbatch_submission": False,
    }


config_hash = canonical_hash
