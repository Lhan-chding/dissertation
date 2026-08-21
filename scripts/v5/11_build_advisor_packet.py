#!/usr/bin/env python3
"""Build the local advisor packet from completed v5 result artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compensability_v5.evaluation.build_v5_tables import (  # noqa: E402
    build_advisor_packet,
    write_advisor_artifacts,
)


def _fixture_results() -> dict[str, object]:
    return {
        "support_results": {
            "schema_version": 1,
            "status": "STUDY_B_SINGLE_SEED_COMPLETE",
            "seed": 2026082201,
            "model_snapshot_sha256": "a" * 64,
            "arm_results": {arm: {} for arm in ("B0", "B1", "B2", "B3")},
            "primary_contrasts": {},
            "stop_signal": {"triggered": False, "rule": "registered_fixture_rule"},
        },
        "confirmation_results": {
            "schema_version": 1,
            "status": "V5_STUDY_A_EXECUTED",
            "source_sha256": {"fixture": "b" * 64},
            "semantic_scene_count": 1,
            "scenario_count": 1,
            "scenario_checkpoint_count": 2,
            "by_checkpoint": {"Base": {}, "T": {}},
            "by_graph_axis": {"canonical": {}},
            "training_invoked": False,
            "rl_invoked": False,
            "prompt_search_invoked": False,
            "confirmatory_data_used": False,
        },
        "reward_results": {
            "schema_version": 1,
            "status": "STUDY_C_DIAGNOSTICS_COMPLETE",
            "seed": 2026082301,
            "group_size": 8,
            "by_arm": {"B3_answer": {}},
            "per_scene": [],
            "registered_stop_signals": {
                "reward_by_fiber_interaction": {
                    "triggered": False,
                    "rule": "registered_fixture_interaction_rule",
                },
                "answer_up_world_down_large_fibers": {
                    "triggered": False,
                    "rule": "registered_fixture_trajectory_rule",
                },
                "any_registered_signal_triggered": False,
                "subjective_threshold_used": False,
            },
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_optional_result(path: Path | None, label: str) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink JSON file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _require_result_in_root(path: Path, root: Path, label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} artifact root must be a regular directory")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} result must be contained in its artifact root") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fixture-dry-run", action="store_true")
    parser.add_argument("--support-results", type=Path, help="native Study B completed.json")
    parser.add_argument("--confirmation-results", type=Path, help="native Study A summary.json")
    parser.add_argument(
        "--reward-results", type=Path, help="native Study C diagnostics summary.json"
    )
    parser.add_argument("--study-a-root", type=Path, help="verified Study A artifact directory")
    parser.add_argument("--study-b-root", type=Path, help="verified Study B artifact directory")
    parser.add_argument("--study-c-root", type=Path, help="verified Study C artifact directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/v5/evaluation/advisor_packet",
        help="new output directory for the final facts, raw archive, and SHA manifest",
    )
    arguments = parser.parse_args()
    if arguments.fixture_dry_run:
        if arguments.execute:
            print("BLOCKED: --fixture-dry-run and --execute are mutually exclusive")
            return 2
        packet = build_advisor_packet(_fixture_results())
        print(json.dumps({"status": packet["status"]}))
        return 0
    output_root = arguments.output
    if not arguments.execute:
        print("BLOCKED: Advisor packet write requires explicit --execute.")
        return 2
    if output_root.exists() or output_root.is_symlink():
        print(f"BLOCKED: output already exists; overwrite forbidden: {output_root}")
        return 2
    try:
        loaded = {
            "support_results": _read_optional_result(arguments.support_results, "support results"),
            "confirmation_results": _read_optional_result(
                arguments.confirmation_results, "confirmation results"
            ),
            "reward_results": _read_optional_result(arguments.reward_results, "reward results"),
        }
        packet = build_advisor_packet(
            {name: payload for name, payload in loaded.items() if payload is not None}
        )
        if packet["status"] == "BLOCKED_MISSING_RESULTS":
            output_root.mkdir(parents=True)
            filename = "status.json"
            _write_json(output_root / filename, packet)
        else:
            roots = {
                name: path
                for name, path in {
                    "study_a": arguments.study_a_root,
                    "study_b": arguments.study_b_root,
                    "study_c": arguments.study_c_root,
                }.items()
                if path is not None
            }
            required = {"study_a", "study_b"}
            if packet["status"] == "ADVISOR_PACKET_READY":
                required.add("study_c")
            if set(roots) != required:
                raise ValueError(f"advisor artifacts require roots: {sorted(required)}")
            result_paths = {
                "study_a": arguments.confirmation_results,
                "study_b": arguments.support_results,
                "study_c": arguments.reward_results,
            }
            for study in required:
                result_path = result_paths[study]
                if result_path is None:
                    raise ValueError(f"{study} native result path is required")
                _require_result_in_root(result_path, roots[study], study)
            write_advisor_artifacts(packet, artifact_roots=roots, output_root=output_root)
            filename = "QWEN_V5_PILOT_RESULT_FACTS.md"
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"BLOCKED: {error}")
        return 2
    print(f"{packet['status']}: output={output_root / filename}")
    return 2 if packet["status"] == "BLOCKED_MISSING_RESULTS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
