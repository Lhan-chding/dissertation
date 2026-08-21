#!/usr/bin/env python3
"""Fail-closed server boundary for budget-matched B0--B3 support LoRA."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compensability_v5.audit.budget_audit import assert_budget_matched  # noqa: E402
from compensability_v5.training.train_support_lora import (  # noqa: E402
    SUPPORT_LORA_ACK,
    ServerExecutionBlocked,
    execute_support_lora,
    load_server_runner,
    validate_server_execution,
)

CONFIG = ROOT / "configs/v5/budget_matched_lora.yaml"
LOCK = ROOT / "configs/v5/server_package_lock.yaml"
OUTPUT_ROOT = ROOT / "artifacts/v5/training"


def _fixture() -> dict[str, dict[str, object]]:
    arm: dict[str, object] = {
        "unique_source_scenes": 8,
        "rows": 48,
        "target_tokens": 512,
        "steps": 6,
        "optimizer": {"name": "adamw", "learning_rate": 0.0002, "weight_decay": 0.01},
        "lora_rank": 16,
        "lora_targets": ["q_proj", "v_proj"],
        "gradient_accumulation": 2,
        "approximate_flops": 1.0,
    }
    return {name: {**arm} for name in ("B0", "B1", "B2", "B3")}


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
    parser.add_argument("--budget-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "budget-matched-lora")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.fixture_dry_run:
        if arguments.execute:
            print("BLOCKED: --fixture-dry-run and --execute are mutually exclusive")
            return 2
        budgets = _fixture()
        assert_budget_matched(budgets, target_token_relative_tolerance=0.01)
        print(json.dumps({"status": "FIXTURE_DRY_RUN_OK", "arms": sorted(budgets)}))
        return 0
    try:
        validation = validate_server_execution(
            phase="phase4_budget_matched_lora",
            execute=arguments.execute,
            acknowledgement=arguments.ack,
            required_acknowledgement=SUPPORT_LORA_ACK,
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
                "rl_allowed": False,
                "downloads_allowed": False,
            },
            expected_config_phase="phase4_budget_matched_lora",
        )
        if arguments.budget_manifest is None or arguments.budget_manifest.resolve() not in {
            path.resolve() for path in arguments.input
        }:
            raise ServerExecutionBlocked(
                "--budget-manifest must name one of the hash-bound --input files"
            )
        payload = json.loads(arguments.budget_manifest.read_text(encoding="utf-8"))
        budgets = payload.get("budgets") if isinstance(payload, Mapping) else None
        if not isinstance(budgets, Mapping):
            raise ServerExecutionBlocked("budget manifest must contain a budgets mapping")
        assert_budget_matched(budgets, target_token_relative_tolerance=0.01)
        runner = load_server_runner(arguments.runtime)
    except (OSError, TypeError, ValueError, ServerExecutionBlocked) as error:
        print(f"BLOCKED: {error}")
        return 2
    execute_support_lora(validation, runner=runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
