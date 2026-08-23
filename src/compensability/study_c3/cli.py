"""Uniform, fail-closed command surface for Study C3 stages 00--09."""

from __future__ import annotations

import argparse
import json
import os

from .analysis_execution import preflight_analysis, run_analysis
from .factorial_evaluation_execution import (
    preflight_factorial_evaluation,
    run_factorial_evaluation,
)
from .factorial_training_execution import (
    TRAINING_ACK,
    preflight_freeze_contract,
    preflight_training_arm,
    run_freeze_contract,
    run_training_arm,
)
from .gradient_execution import (
    preflight_shared_gradient_audit,
    run_shared_gradient_audit,
)
from .packaging_runtime import preflight_package, run_package
from .stage_a_b_runtime import (
    preflight_action_audit,
    preflight_existing_checkpoint_evaluation,
    preflight_trie_build,
    preflight_trie_validation,
    run_action_audit,
    run_existing_checkpoint_evaluation,
    run_trie_build,
    run_trie_validation,
)

STAGE_NAMES = {
    0: "AUDIT_EXISTING_ACTION_CHANNEL",
    1: "BUILD_VALID_WORLD_TRIE",
    2: "VALIDATE_CONSTRAINED_DECODER",
    3: "EVALUATE_EXISTING_C2_CHECKPOINTS",
    4: "FREEZE_FACTORIAL_EXECUTION_CONTRACT",
    5: "TRAIN_FACTORIAL_GRPO",
    6: "EVALUATE_FACTORIAL_CHECKPOINTS",
    7: "SHARED_GRADIENT_VALIDITY_AUDIT",
    8: "ANALYZE_RESOLUTION_VALIDITY",
    9: "PACKAGE_STUDY_C3_EVIDENCE",
}


def _parser(stage: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Study C3 stage {stage:02d}: {STAGE_NAMES[stage]}"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--fixture-dry-run", action="store_true")
    parser.add_argument("--arm", choices=("A_BIN", "X_BIN", "A_LEX", "X_LEX"))
    parser.add_argument("--date")
    return parser


def run_stage(stage: int) -> int:
    if stage not in STAGE_NAMES:
        raise ValueError(f"unregistered Study C3 stage: {stage}")
    arguments = _parser(stage).parse_args()
    if arguments.fixture_dry_run:
        print(
            json.dumps(
                {
                    "schema_version": 3,
                    "stage": stage,
                    "status": f"STUDY_C3_{STAGE_NAMES[stage]}_FIXTURE_OK",
                    "single_seed_mechanism_pilot": True,
                    "gpu_invoked": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.execute == arguments.preflight_only:
        print(
            f"BLOCKED: Study C3 stage {stage:02d} requires exactly one of "
            "--execute or --preflight-only",
            flush=True,
        )
        return 2
    try:
        if stage == 0:
            payload = preflight_action_audit() if arguments.preflight_only else run_action_audit()
        elif stage == 1:
            payload = preflight_trie_build() if arguments.preflight_only else run_trie_build()
        elif stage == 2:
            payload = (
                preflight_trie_validation() if arguments.preflight_only else run_trie_validation()
            )
        elif stage == 3:
            payload = (
                preflight_existing_checkpoint_evaluation()
                if arguments.preflight_only
                else run_existing_checkpoint_evaluation()
            )
        elif stage == 4:
            payload = (
                preflight_freeze_contract() if arguments.preflight_only else run_freeze_contract()
            )
        elif stage == 5:
            if arguments.arm is None:
                raise ValueError("Study C3 stage 05 requires --arm")
            payload = (
                preflight_training_arm(arguments.arm)
                if arguments.preflight_only
                else run_training_arm(
                    arm=arguments.arm,
                    acknowledgement=os.environ.get("COMPBIAS_STUDY_C3_ACK", ""),
                )
            )
        elif stage == 6:
            payload = (
                preflight_factorial_evaluation()
                if arguments.preflight_only
                else run_factorial_evaluation()
            )
        elif stage == 7:
            payload = (
                preflight_shared_gradient_audit()
                if arguments.preflight_only
                else run_shared_gradient_audit()
            )
        elif stage == 8:
            payload = preflight_analysis() if arguments.preflight_only else run_analysis()
        else:
            payload = (
                preflight_package()
                if arguments.preflight_only
                else run_package(date_label=arguments.date)
            )
        print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (OSError, RuntimeError, ValueError, PermissionError) as error:
        print(f"BLOCKED: {error}", flush=True)
        return 2


__all__ = ["STAGE_NAMES", "TRAINING_ACK", "run_stage"]
