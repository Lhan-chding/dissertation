"""Dependency-light Study C2 data, fiber, and audit stages."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from .action_protocol import (
    parse_first_world_action,
    parse_first_world_tuple,
    parse_legacy_exact_world,
)
from .fiber import full_reward_fiber_size, one_edit_fiber_size
from .io import read_jsonl, sha256_file, write_json_new, write_jsonl_new
from .matched_pair_generator import build_matched_reward_pairs
from .paths import (
    FIBER_MANIFEST,
    FIBER_ROWS,
    LEGACY_ROOT,
    PAIR_MANIFEST,
    PAIR_ROWS,
)
from .schemas import validate_study_c2_config


def load_contract(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing Study C2 config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Study C2 config root must be a mapping")
    return validate_study_c2_config(payload)


def build_pair_artifacts(*, config_path: Path) -> dict[str, object]:
    print("PROGRESS: validating the closed Study C2 configuration", flush=True)
    contract = load_contract(config_path)
    print("PROGRESS: constructing 232 deterministic matched base scenes", flush=True)
    rows = build_matched_reward_pairs(seed=int(contract["seed"]))
    write_jsonl_new(PAIR_ROWS, rows)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "status": "STUDY_C2_MATCHED_PAIRS_FROZEN",
        "seed": contract["seed"],
        "base_scene_count": len(rows) // 2,
        "prompt_count": len(rows),
        "split_prompt_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "condition_prompt_counts": dict(
            sorted(Counter(str(row["condition"]) for row in rows).items())
        ),
        "config_sha256": sha256_file(config_path),
        "rows_sha256": sha256_file(PAIR_ROWS),
        "gpu_invoked": False,
    }
    write_json_new(PAIR_MANIFEST, manifest)
    return manifest


def enumerate_fiber_artifacts(*, config_path: Path) -> dict[str, object]:
    print("PROGRESS: validating Study C2 pairs before full-domain enumeration", flush=True)
    load_contract(config_path)
    rows = read_jsonl(PAIR_ROWS)
    enriched: list[dict[str, object]] = []
    for row_index, row in enumerate(rows, start=1):
        operation = row.get("operation")
        truth = row.get("truth")
        answer = row.get("gold_answer")
        if (
            not isinstance(operation, Mapping)
            or not isinstance(truth, list)
            or len(truth) != 4
            or type(answer) is not int
        ):
            raise ValueError("Study C2 pair row is malformed")
        world = tuple(int(value) for value in truth)
        full_size = full_reward_fiber_size(operation, answer)
        one_edit_size = one_edit_fiber_size(world, operation)
        enriched.append(
            {
                **row,
                "full_reward_fiber_size": full_size,
                "log1p_full_reward_fiber_size": math.log1p(full_size),
                "one_edit_reward_fiber_size": one_edit_size,
                "observed_is_answer_equivalent": (row["observed_answer"] == row["gold_answer"]),
                "reward_identifiable": full_size == 1,
            }
        )
        if row_index == 1 or row_index % 32 == 0 or row_index == len(rows):
            print(
                f"PROGRESS: exact reward fiber {row_index}/{len(rows)} scene={row['scene_id']}",
                flush=True,
            )
    write_jsonl_new(FIBER_ROWS, enriched)
    sizes = [int(row["full_reward_fiber_size"]) for row in enriched]
    manifest: dict[str, object] = {
        "schema_version": 2,
        "status": "STUDY_C2_REWARD_FIBERS_ENUMERATED",
        "prompt_count": len(enriched),
        "minimum_full_fiber_size": min(sizes),
        "maximum_full_fiber_size": max(sizes),
        "all_truths_in_fiber": True,
        "all_collision_observations_answer_equivalent": all(
            bool(row["observed_is_answer_equivalent"])
            for row in enriched
            if row["condition"] == "collision"
        ),
        "all_separating_observations_answer_distinct": all(
            not bool(row["observed_is_answer_equivalent"])
            for row in enriched
            if row["condition"] == "separating"
        ),
        "pair_rows_sha256": sha256_file(PAIR_ROWS),
        "fiber_rows_sha256": sha256_file(FIBER_ROWS),
        "config_sha256": sha256_file(config_path),
        "gpu_invoked": False,
    }
    write_json_new(FIBER_MANIFEST, manifest)
    return manifest


def audit_legacy_trace(*, trace_paths: Sequence[Path]) -> dict[str, object]:
    completions: list[tuple[str, str, str]] = []
    for path in trace_paths:
        for row in read_jsonl(path):
            raw = row.get("completion", row.get("raw_completion"))
            if isinstance(raw, str):
                completions.append((str(path), str(row.get("scene_id", "")), raw))
    if not completions:
        raise ValueError("legacy Study C traces contain no completion text")
    rows = tuple(
        {
            "source": source,
            "scene_id": scene_id,
            "completion": completion,
            "legacy_world": parse_legacy_exact_world(completion),
            "first_line_integer_world": parse_first_world_tuple(completion),
            "first_line_world": parse_first_world_action(completion),
            "legacy_parse_success": parse_legacy_exact_world(completion) is not None,
            "first_line_integer_parse_success": parse_first_world_tuple(completion) is not None,
            "first_line_parse_success": parse_first_world_action(completion) is not None,
        }
        for source, scene_id, completion in completions
    )
    LEGACY_ROOT.mkdir(parents=True, exist_ok=False)
    rows_path = LEGACY_ROOT / "rows.jsonl"
    write_jsonl_new(rows_path, rows)
    by_source: dict[str, dict[str, int]] = {}
    for path in trace_paths:
        source_rows = [row for row in rows if row["source"] == str(path)]
        by_source[str(path)] = {
            "completion_count": len(source_rows),
            "legacy_parse_success_count": sum(
                bool(row["legacy_parse_success"]) for row in source_rows
            ),
            "first_line_integer_parse_success_count": sum(
                bool(row["first_line_integer_parse_success"]) for row in source_rows
            ),
            "first_line_valid_action_count": sum(
                bool(row["first_line_parse_success"]) for row in source_rows
            ),
        }
    summary: dict[str, object] = {
        "schema_version": 2,
        "status": "STUDY_C2_LEGACY_PARSER_AUDITED",
        "completion_count": len(rows),
        "legacy_parse_success_count": sum(bool(row["legacy_parse_success"]) for row in rows),
        "first_line_integer_parse_success_count": sum(
            bool(row["first_line_integer_parse_success"]) for row in rows
        ),
        "first_line_parse_success_count": sum(
            bool(row["first_line_parse_success"]) for row in rows
        ),
        "rows_sha256": sha256_file(rows_path),
        "source_sha256": {str(path): sha256_file(path) for path in trace_paths},
        "by_source": by_source,
        "gpu_invoked": False,
    }
    write_json_new(LEGACY_ROOT / "summary.json", summary)
    return summary


def print_json(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


__all__ = [
    "audit_legacy_trace",
    "build_pair_artifacts",
    "enumerate_fiber_artifacts",
    "load_contract",
    "print_json",
]
