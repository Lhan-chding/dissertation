"""CLI construction for Phase 1-3 server pre-work manifests."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from _guards import (
    CONFIG_PATH,
    PACKAGE_LOCK_PATH,
    ROOT,
    blocked_unless_execute,
    validate_server_inputs,
    write_execution_manifest,
)


def run_phase_preflight(
    *,
    phase: str,
    description: str,
    default_output_name: str,
    intended_artifacts: Iterable[str],
    integrity_gates: Iterable[str],
    expected_input_sha256: Iterable[str],
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--package-lock", type=Path, default=PACKAGE_LOCK_PATH)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/model/ModelScope/Qwen/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--input-sha256", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/v4/server_preflight" / default_output_name,
    )
    arguments = parser.parse_args()
    if blocked_unless_execute(arguments.execute):
        return 2
    try:
        validation = validate_server_inputs(
            config=arguments.config,
            package_lock=arguments.package_lock,
            model_path=arguments.model_path,
            inputs=arguments.input,
            input_sha256=arguments.input_sha256,
            expected_input_sha256=expected_input_sha256,
            require_raw_evidence=True,
        )
        write_execution_manifest(
            arguments.output,
            phase=phase,
            validation=validation,
            intended_artifacts=intended_artifacts,
            integrity_gates=integrity_gates,
        )
    except Exception as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"PREPARED: {phase} pre-work manifest only; phase not executed; {arguments.output}")
    return 0


__all__ = ["run_phase_preflight"]
