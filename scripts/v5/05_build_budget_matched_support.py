#!/usr/bin/env python3
"""Freeze the local CPU-side B0--B3 support package before server training."""

from __future__ import annotations

import argparse
import importlib
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compensability_v5.training.build_budget_matched_support import (  # noqa: E402
    SupportBuildError,
    build_budget_matched_support,
)


def _fixture_scene() -> list[dict[str, object]]:
    return [
        {
            "scene_id": "support-001",
            "semantic_scene_id": "semantic-001",
            "prompt": "Observed world: 8,2,3,4. Recover the true world.",
            "truth": [9, 2, 3, 4],
            "natural_observation": [8, 2, 3, 4],
            "constraint_matrix": [[1, 0, 0, 0], [0, 1, 0, 0]],
            "constraint_targets": [9, 2],
            "answer_operation": {"operator": "sum", "indices": [0, 1]},
            "transformation": {"kind": "identity"},
        }
    ]


def _training_budget() -> dict[str, object]:
    return {
        "steps": 72,
        "optimizer": {"name": "adamw", "learning_rate": 2e-5, "weight_decay": 0.0},
        "lora_rank": 16,
        "lora_targets": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "gradient_accumulation": 8,
        "approximate_flops": 1.0,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists; overwrite forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_token_counter(specification: str) -> Callable[[str], int]:
    if ":" not in specification:
        raise SupportBuildError("--token-counter must use module:callable syntax")
    module_name, attribute = specification.rsplit(":", 1)
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise SupportBuildError(f"cannot import token counter: {error}") from error
    if not callable(candidate):
        raise SupportBuildError("injected token counter must be callable")
    return candidate


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SupportBuildError(f"unsafe or missing provenance file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_provenance(parent: Path, child: Path, scenes: Path) -> dict[str, str]:
    parent_sha, child_sha, scenes_sha = _sha256(parent), _sha256(child), _sha256(scenes)
    payload = json.loads(child.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or (
        payload.get("status") != "V5_PHASE2A_NATURAL_OBSERVATIONS_FROZEN"
        or payload.get("parent_manifest_sha256") != parent_sha
        or payload.get("frozen_scenes_sha256") != scenes_sha
        or payload.get("parent_manifest_modified") is not False
        or payload.get("semantic_scene_count") != 96
    ):
        raise SupportBuildError("Phase-2a child manifest provenance drifted")
    return {
        "parent_manifest_sha256": parent_sha,
        "child_manifest_sha256": child_sha,
        "frozen_scenes_sha256": scenes_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fixture-dry-run", action="store_true")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--child-manifest", type=Path)
    parser.add_argument("--token-counter")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/v5/data/budget_matched_support.json",
    )
    arguments = parser.parse_args()
    if arguments.fixture_dry_run:
        if arguments.execute:
            print("BLOCKED: --fixture-dry-run and --execute are mutually exclusive")
            return 2
        package = build_budget_matched_support(
            _fixture_scene(),
            token_counter=lambda _text: 8,
            training_budget=_training_budget(),
            source_provenance={
                "parent_manifest_sha256": "1" * 64,
                "child_manifest_sha256": "2" * 64,
                "frozen_scenes_sha256": "3" * 64,
            },
        )
        print(json.dumps({"status": "FIXTURE_DRY_RUN_OK", "arms": sorted(package["arms"])}))
        return 0
    if not arguments.execute:
        print("BLOCKED: Support freeze requires explicit --execute.")
        return 2
    if any(
        value is None
        for value in (
            arguments.input_jsonl,
            arguments.parent_manifest,
            arguments.child_manifest,
            arguments.token_counter,
        )
    ):
        print("BLOCKED: input, parent/child manifests, and injected token counter are required.")
        return 2
    try:
        package = build_budget_matched_support(
            _read_jsonl(arguments.input_jsonl),
            token_counter=_load_token_counter(arguments.token_counter),
            training_budget=_training_budget(),
            source_provenance=_source_provenance(
                arguments.parent_manifest,
                arguments.child_manifest,
                arguments.input_jsonl,
            ),
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"BLOCKED: {error}")
        return 2
    if package["source_scene_count"] != 96 or any(
        budget["rows"] != 576 for budget in package["budgets"].values()
    ):
        print("BLOCKED: canonical Study B requires 96 sources and 576 rows per arm.")
        return 2
    package["pilot_schedule"] = {
        "hardware": "single_RTX_4090",
        "batch_size": 1,
        "gradient_accumulation": 8,
        "epochs": 1,
        "optimizer_steps": 72,
    }
    _write_json(arguments.output, package)
    print(
        "V5_BUDGET_MATCHED_SUPPORT_FROZEN: "
        f"output={arguments.output} scenes={package['source_scene_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
