"""Uniform, fail-closed command surface for Study C2 stages 20--40."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .paths import FIBER_ROWS, STAGE24_EXECUTION_CONTRACT, SUPPORT_MANIFEST
from .policy_support_runtime import preflight_support, run_frozen_policy_support
from .shared_gradient_runtime import preflight_shared_gradient, run_shared_gradient_audit
from .stages import audit_legacy_trace, build_pair_artifacts, enumerate_fiber_artifacts, print_json

CONFIG = Path("configs/v5/study_c2_identifiable_reward.yaml")
GPU_STAGES = frozenset({23, 24, 25, 26, 31, 32, 33, 36, 37, 39})
STAGE_NAMES = {
    20: "AUDIT_LEGACY_STUDY_C_PARSER",
    21: "BUILD_REWARD_IDENTIFIABILITY_PAIRS",
    22: "ENUMERATE_REWARD_FIBERS",
    23: "MEASURE_FROZEN_POLICY_SUPPORT",
    24: "SHARED_BATCH_REWARD_GRADIENT_AUDIT",
    25: "TRAIN_IDENTIFIABLE_REWARD_GRPO",
    26: "EVALUATE_IDENTIFIABLE_REWARD_GRPO",
    27: "REPORT_IDENTIFIABLE_REWARD_GRPO",
    28: "REWARD_NULL_GEOMETRY_AUDIT",
    29: "CONTRAST_RANK_GRADIENT_TEST",
    30: "K_SCALING_EXCITATION_LAW",
    31: "ONE_STEP_FIBER_PURITY",
    32: "CHECKPOINT_PURITY_TRAJECTORY",
    33: "SUPPORT_IDENTIFIABILITY_PHASE_DIAGRAM",
    34: "ACTION_CENSORING_BRIDGE",
    35: "COUNTERFACTUAL_REWARD_RELABELING",
    36: "ADVANTAGE_BASELINE_CONTROL",
    37: "FIXED_BUFFER_REWARD_UPDATE",
    38: "CORRECTION_VECTOR_GEOMETRY",
    39: "REWARD_STRUCTURE_INTERACTION",
    40: "NATURAL_STATE_REWARD_DISAGREEMENT_AUDIT",
}


def _parser(stage: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Study C2 stage {stage}: {STAGE_NAMES[stage]}")
    parser.add_argument("--fixture-dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--b3-adapter", type=Path)
    parser.add_argument("--b3-sha256")
    parser.add_argument("--execution-contract", type=Path, default=STAGE24_EXECUTION_CONTRACT)
    parser.add_argument("--ack")
    parser.add_argument("--arm", choices=("answer", "state"))
    parser.add_argument("--legacy-root", type=Path, default=Path("artifacts/v5/rl/study-c-pilot"))
    parser.add_argument("--trace", type=Path, action="append", default=[])
    return parser


def _fixture(stage: int) -> dict[str, object]:
    return {
        "schema_version": 2,
        "stage": stage,
        "status": f"STUDY_C2_{STAGE_NAMES[stage]}_FIXTURE_OK",
        "gpu_invoked": False,
    }


def _require_b3(arguments: argparse.Namespace) -> tuple[Path, str]:
    if arguments.b3_adapter is None or arguments.b3_sha256 is None:
        raise ValueError("--b3-adapter and --b3-sha256 are required")
    return arguments.b3_adapter, arguments.b3_sha256


def _legacy_traces(arguments: argparse.Namespace) -> tuple[Path, ...]:
    if arguments.trace:
        return tuple(arguments.trace)
    root = arguments.legacy_root
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"legacy Study C root is unavailable: {root}")
    paths = tuple(
        sorted(
            path
            for path in root.rglob("*.jsonl")
            if "raw" in path.name and path.is_file() and not path.is_symlink()
        )
    )
    if not paths:
        raise ValueError(f"no raw Study C JSONL traces below {root}")
    return paths


def _not_reached(stage: int) -> None:
    prerequisite = "frozen-policy support" if stage >= 24 else "registered input artifacts"
    raise RuntimeError(
        f"stage {stage} is registered but {prerequisite} is unavailable; "
        "run the first Study C2 server boundary (stage 23) before continuing"
    )


def run_registered(stage: int) -> int:
    if stage not in STAGE_NAMES:
        raise ValueError(f"unregistered Study C2 stage: {stage}")
    arguments = _parser(stage).parse_args()
    if arguments.fixture_dry_run:
        print(json.dumps(_fixture(stage), sort_keys=True))
        return 0
    if not arguments.execute and not arguments.preflight_only:
        print(
            f"BLOCKED: Study C2 stage {stage} requires --execute or --preflight-only",
            flush=True,
        )
        return 2
    try:
        if stage == 20:
            if not arguments.execute:
                raise ValueError("legacy parser audit requires --execute")
            payload = audit_legacy_trace(trace_paths=_legacy_traces(arguments))
        elif stage == 21:
            if not arguments.execute:
                raise ValueError("pair construction requires --execute")
            payload = build_pair_artifacts(config_path=arguments.config)
        elif stage == 22:
            if not arguments.execute:
                raise ValueError("fiber enumeration requires --execute")
            payload = enumerate_fiber_artifacts(config_path=arguments.config)
        elif stage == 23:
            adapter, digest = _require_b3(arguments)
            if arguments.preflight_only:
                payload = preflight_support(
                    config_path=arguments.config, b3_adapter=adapter, b3_sha256=digest
                )
            else:
                payload = run_frozen_policy_support(
                    config_path=arguments.config,
                    b3_adapter=adapter,
                    b3_sha256=digest,
                    acknowledgement=arguments.ack,
                )
        elif stage == 24:
            adapter, digest = _require_b3(arguments)
            if arguments.preflight_only:
                payload = preflight_shared_gradient(
                    config_path=arguments.config,
                    execution_contract_path=arguments.execution_contract,
                    b3_adapter=adapter,
                    b3_sha256=digest,
                )
            else:
                payload = run_shared_gradient_audit(
                    config_path=arguments.config,
                    execution_contract_path=arguments.execution_contract,
                    b3_adapter=adapter,
                    b3_sha256=digest,
                    acknowledgement=arguments.ack,
                )
        else:
            if not FIBER_ROWS.is_file():
                raise RuntimeError("Study C2 fiber rows are missing")
            if stage >= 24 and not SUPPORT_MANIFEST.is_file():
                _not_reached(stage)
            _not_reached(stage)
        print_json(payload)
        return 0
    except (OSError, RuntimeError, ValueError, PermissionError) as error:
        print(f"BLOCKED: {error}", flush=True)
        return 2


__all__ = ["run_registered"]
