#!/usr/bin/env python3
"""Fail-closed server boundary and pure-metric smoke test for v5 evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compensability_v5.evaluation.estimate_gradient_alignment import (  # noqa: E402
    estimate_gradient_alignment,
    gradient_alignment_fixture,
)
from compensability_v5.evaluation.evaluate_reward_fibers import (  # noqa: E402
    evaluate_reward_fibers,
    reward_fiber_fixture,
)
from compensability_v5.evaluation.evaluate_structural_support import (  # noqa: E402
    evaluate_structural_support,
    structural_support_fixture,
)
from compensability_v5.training.train_support_lora import (  # noqa: E402
    ServerExecutionBlocked,
    load_server_runner,
    validate_server_execution,
)

CONFIG = ROOT / "configs/v5/evaluation.yaml"
LOCK = ROOT / "configs/v5/server_package_lock.yaml"
OUTPUT_ROOT = ROOT / "artifacts/v5/evaluation"
ACK = "I_UNDERSTAND_THIS_RUNS_V5_FINAL_EVALUATION"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ack")
    parser.add_argument("--fixture-dry-run", action="store_true")
    parser.add_argument("--runtime")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--config-sha256")
    parser.add_argument("--package-lock", type=Path, default=LOCK)
    parser.add_argument("--package-lock-sha256")
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--input-sha256", action="append", default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "v5_evaluation.json")
    parser.add_argument("--k", type=int, default=8)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.fixture_dry_run:
        if arguments.execute:
            print("BLOCKED: --fixture-dry-run and --execute are mutually exclusive")
            return 2
        result = {
            "schema_version": 1,
            "status": "FIXTURE_DRY_RUN_OK",
            "reward_fibers": evaluate_reward_fibers(reward_fiber_fixture()),
            "gradient_alignment": estimate_gradient_alignment(gradient_alignment_fixture()),
            "structural_support": evaluate_structural_support(
                structural_support_fixture(), k=arguments.k
            ),
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    try:
        validation = validate_server_execution(
            phase="phase8_v5_evaluation",
            execute=arguments.execute,
            acknowledgement=arguments.ack,
            required_acknowledgement=ACK,
            config=arguments.config,
            canonical_config=CONFIG,
            config_sha256=arguments.config_sha256,
            package_lock=arguments.package_lock,
            canonical_package_lock=LOCK,
            package_lock_sha256=arguments.package_lock_sha256,
            inputs=arguments.input,
            input_sha256=arguments.input_sha256,
            output=arguments.output,
            allowed_output_root=OUTPUT_ROOT,
            required_authorization={
                "inference_allowed": True,
                "training_allowed": False,
                "rl_allowed": False,
                "downloads_allowed": False,
            },
            expected_config_phase="phase5_8_evaluation",
        )
        runner = load_server_runner(arguments.runtime)
    except (OSError, TypeError, ValueError, ServerExecutionBlocked) as error:
        print(f"BLOCKED: {error}")
        return 2
    runner(
        validation.to_mapping(),
        {
            "task": "evaluate_v5",
            "k": arguments.k,
            "pilot": "single_gpu_4090",
            "study_c_seed_count": 1,
            "registered_stop_conditions": [
                "b3_gt_b2_structural_ood",
                "reward_by_fiber_interaction",
                "answer_up_world_down",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
