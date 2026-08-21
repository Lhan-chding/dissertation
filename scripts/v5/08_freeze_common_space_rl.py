#!/usr/bin/env python3
"""Freeze the local CPU-side common-action RL package before server GRPO."""

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

from compensability_v5.audit.fiber_multiplicity import (  # noqa: E402
    enumerate_one_edit_worlds,
)
from compensability_v5.data.common_action_freeze import (  # noqa: E402
    ACTION_PARSER_ID,
    PILOT_SEED,
    freeze_common_action_space,
)
from compensability_v5.data.common_action_schema import (  # noqa: E402
    WorldAction,
    apply_answer_operation,
)

_FIBER_DEFINITION = {
    "candidate_construction": "one_edit_domain_union_truth",
    "center": "natural_observation",
    "includes_truth": True,
    "value_domain": [2, 18],
}


def _fixture_scene() -> list[dict[str, object]]:
    return [
        {
            "scene_id": "rl-001",
            "prompt": "Observed world: 8,2,3,4. Return four comma-separated integers only.",
            "truth": [9, 2, 3, 4],
            "answer_operation": {"operator": "sum", "indices": [0, 1]},
            "family": "pair_sum",
            "fiber_size": 3,
            "policy_support": 0.25,
        }
    ]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists; overwrite forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attach_study_a_support(
    scenes: list[dict[str, object]], study_a_rows: Path
) -> list[dict[str, object]]:
    audit = _read_jsonl(study_a_rows)
    support: dict[str, float] = {}
    for row in audit:
        if row.get("checkpoint") == "T" and row.get("graph_axis") == "canonical":
            scene_id = row.get("source_scene_id")
            value = row.get("exact_recovery_probability")
            if (
                not isinstance(scene_id, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
                or scene_id in support
            ):
                raise ValueError("Study A canonical T policy-support rows are malformed")
            support[scene_id] = float(value)
    result = []
    for scene in scenes:
        semantic_id = scene.get("semantic_scene_id", scene.get("scene_id"))
        if semantic_id not in support:
            raise ValueError(f"Study A policy support is missing for {semantic_id}")
        source = {
            key: value
            for key, value in scene.items()
            if key not in {"fiber_bin", "role", "support_bin"}
        }
        observed_payload = scene.get("natural_observation")
        operation = scene.get("answer_operation")
        if observed_payload is None:
            result.append({**source, "policy_support": support[str(semantic_id)]})
            continue
        if not isinstance(operation, Mapping):
            raise ValueError("Phase2a Study C scene lacks an answer operation")
        source = {
            key: value
            for key, value in source.items()
            if key not in {"candidate_worlds", "fiber_definition", "fiber_size"}
        }
        truth = WorldAction.from_mapping({"world": scene.get("truth")})
        observed = WorldAction.from_mapping({"world": observed_payload})
        if any(value < 2 or value > 18 for value in truth.world):
            raise ValueError("Phase2a Study C truth must remain inside the registered 2..18 domain")
        candidates = set(enumerate_one_edit_worlds(observed.world, range(2, 19)))
        candidates.add(truth.world)
        answer = apply_answer_operation(truth, operation)
        fiber_size = sum(
            apply_answer_operation(WorldAction(candidate), operation) == answer
            for candidate in candidates
        )
        result.append(
            {
                **source,
                "candidate_worlds": [list(candidate) for candidate in sorted(candidates)],
                "fiber_definition": dict(_FIBER_DEFINITION),
                "fiber_size": fiber_size,
                "policy_support": support[str(semantic_id)],
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fixture-dry-run", action="store_true")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--study-a-rows", type=Path)
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
    if (
        arguments.input_jsonl is None
        or arguments.study_a_rows is None
        or any(value is None for value in initialization_values)
    ):
        print("BLOCKED: --input-jsonl and all three initialization SHA-256 values are required.")
        return 2
    try:
        package = freeze_common_action_space(
            _attach_study_a_support(_read_jsonl(arguments.input_jsonl), arguments.study_a_rows),
            initialization_hashes={
                "B3": arguments.b3_initialization_sha256,
                "B2": arguments.b2_initialization_sha256,
                "Base": arguments.base_initialization_sha256,
            },
            action_parser_id=ACTION_PARSER_ID,
            rollout_seeds=[PILOT_SEED],
        )
        if package["role_counts"] != {"rl_train": 72, "rl_eval": 24}:
            raise ValueError("Study C requires the registered 72/24 disjoint split")
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
