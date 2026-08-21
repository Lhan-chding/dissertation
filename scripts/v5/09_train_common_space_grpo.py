#!/usr/bin/env python3
"""Fail-closed server boundary for v5 common-action-space GRPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compensability_v5.training.train_common_space_grpo import (  # noqa: E402
    COMMON_SPACE_GRPO_ACK,
    assert_common_space_reward_isolation,
    common_space_fixture,
    execute_common_space_grpo,
    load_common_space_arms,
)
from compensability_v5.training.train_support_lora import (  # noqa: E402
    ServerExecutionBlocked,
    load_server_runner,
    validate_server_execution,
)

CONFIG = ROOT / "configs/v5/common_space_grpo.yaml"
LOCK = ROOT / "configs/v5/server_package_lock.yaml"
OUTPUT_ROOT = ROOT / "artifacts/v5/rl"


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
    parser.add_argument("--common-action-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "common-space-grpo")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.fixture_dry_run:
        if arguments.execute:
            print("BLOCKED: --fixture-dry-run and --execute are mutually exclusive")
            return 2
        arms = common_space_fixture()
        assert_common_space_reward_isolation(arms)
        print(json.dumps({"status": "FIXTURE_DRY_RUN_OK", "arms": sorted(arms)}))
        return 0
    try:
        validation = validate_server_execution(
            phase="phase7_common_space_grpo",
            execute=arguments.execute,
            acknowledgement=arguments.ack,
            required_acknowledgement=COMMON_SPACE_GRPO_ACK,
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
                "training_allowed": True,
                "rl_allowed": True,
                "downloads_allowed": False,
            },
            expected_config_phase="phase7_common_space_grpo",
            expected_seed_count=1,
        )
        hash_bound_inputs = {path.resolve() for path in arguments.input}
        if (
            arguments.common_action_manifest is None
            or arguments.common_action_manifest.resolve() not in hash_bound_inputs
        ):
            raise ServerExecutionBlocked(
                "--common-action-manifest must name one of the hash-bound --input files"
            )
        arms = load_common_space_arms(arguments.common_action_manifest)
        runner = load_server_runner(arguments.runtime)
    except (OSError, TypeError, ValueError, ServerExecutionBlocked) as error:
        print(f"BLOCKED: {error}")
        return 2
    execute_common_space_grpo(validation, arms, runner=runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
