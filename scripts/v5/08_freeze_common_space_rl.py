#!/usr/bin/env python3
"""Freeze the local CPU-side common-action RL package before server GRPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compensability_v5.data.common_action_freeze import (  # noqa: E402
    ACTION_PARSER_ID,
    PILOT_SEED,
    freeze_common_action_space,
)


def _fixture_scene() -> list[dict[str, object]]:
    return [
        {
            "scene_id": "rl-001",
            "prompt": "Observed world: 8,2,3,4. Return four comma-separated integers only.",
            "truth": [9, 2, 3, 4],
            "answer_operation": {"operator": "sum", "indices": [0, 1]},
        }
    ]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists; overwrite forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fixture-dry-run", action="store_true")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--b3-initialization-sha256")
    parser.add_argument("--b2-initialization-sha256")
    parser.add_argument("--base-initialization-sha256")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/v5/data/common_space_rl.json",
    )
    arguments = parser.parse_args()
    if arguments.fixture_dry_run:
        if arguments.execute:
            print("BLOCKED: --fixture-dry-run and --execute are mutually exclusive")
            return 2
        package = freeze_common_action_space(
            _fixture_scene(),
            initialization_hashes={"B3": "a" * 64, "B2": "b" * 64, "Base": "c" * 64},
            action_parser_id=ACTION_PARSER_ID,
            rollout_seeds=[PILOT_SEED],
        )
        print(json.dumps({"status": "FIXTURE_DRY_RUN_OK", "arms": sorted(package["arms"])}))
        return 0
    if not arguments.execute:
        print("BLOCKED: Common-space RL freeze requires explicit --execute.")
        return 2
    initialization_values = (
        arguments.b3_initialization_sha256,
        arguments.b2_initialization_sha256,
        arguments.base_initialization_sha256,
    )
    if arguments.input_jsonl is None or any(value is None for value in initialization_values):
        print("BLOCKED: --input-jsonl and all three initialization SHA-256 values are required.")
        return 2
    try:
        package = freeze_common_action_space(
            _read_jsonl(arguments.input_jsonl),
            initialization_hashes={
                "B3": arguments.b3_initialization_sha256,
                "B2": arguments.b2_initialization_sha256,
                "Base": arguments.base_initialization_sha256,
            },
            action_parser_id=ACTION_PARSER_ID,
            rollout_seeds=[PILOT_SEED],
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"BLOCKED: {error}")
        return 2
    _write_json(arguments.output, package)
    print(
        f"V5_COMMON_ACTION_SPACE_FROZEN: output={arguments.output} scenes={len(package['scenes'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
