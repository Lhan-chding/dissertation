from __future__ import annotations

import json
from pathlib import Path

import pytest

from compensability_v5.study_c2 import stages
from compensability_v5.study_c2.io import read_json, read_jsonl, sha256_file
from compensability_v5.study_c2.support import (
    summarize_policy_support,
    summarize_realized_groups,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/v5/study_c2_identifiable_reward.yaml"


def _redirect_stage_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(stages, "PAIR_ROWS", root / "pairs.jsonl")
    monkeypatch.setattr(stages, "PAIR_MANIFEST", root / "pairs_manifest.json")
    monkeypatch.setattr(stages, "FIBER_ROWS", root / "fibers.jsonl")
    monkeypatch.setattr(stages, "FIBER_MANIFEST", root / "fibers_manifest.json")
    monkeypatch.setattr(stages, "LEGACY_ROOT", root / "legacy")


def test_cpu_stages_publish_hash_bound_immutable_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _redirect_stage_paths(monkeypatch, tmp_path)
    pairs = stages.build_pair_artifacts(config_path=CONFIG)
    assert pairs["status"] == "STUDY_C2_MATCHED_PAIRS_FROZEN"
    assert pairs["prompt_count"] == 464
    assert pairs["rows_sha256"] == sha256_file(tmp_path / "pairs.jsonl")

    fibers = stages.enumerate_fiber_artifacts(config_path=CONFIG)
    assert fibers["status"] == "STUDY_C2_REWARD_FIBERS_ENUMERATED"
    assert fibers["minimum_full_fiber_size"] >= 1
    assert fibers["all_collision_observations_answer_equivalent"] is True
    assert fibers["all_separating_observations_answer_distinct"] is True
    assert fibers["fiber_rows_sha256"] == sha256_file(tmp_path / "fibers.jsonl")
    with pytest.raises(FileExistsError):
        stages.build_pair_artifacts(config_path=CONFIG)


def test_legacy_parser_stage_compares_old_and_first_line_protocols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _redirect_stage_paths(monkeypatch, tmp_path)
    trace = tmp_path / "raw_trace.jsonl"
    trace.write_text(
        json.dumps({"scene_id": "a", "completion": "2,3,4,5\nreason"})
        + "\n"
        + json.dumps({"scene_id": "b", "raw_completion": "2,3,4,5"})
        + "\n",
        encoding="utf-8",
    )
    summary = stages.audit_legacy_trace(trace_paths=(trace,))
    assert summary["completion_count"] == 2
    assert summary["legacy_parse_success_count"] == 1
    assert summary["first_line_integer_parse_success_count"] == 2
    assert summary["first_line_parse_success_count"] == 2
    assert read_json(tmp_path / "legacy/summary.json")["rows_sha256"] == sha256_file(
        tmp_path / "legacy/rows.jsonl"
    )


def test_io_rejects_symlinks_non_mappings_and_empty_jsonl(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.json"
    mapping.write_text('{"ok": true}', encoding="utf-8")
    assert read_json(mapping) == {"ok": True}
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        read_json(array)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="mappings"):
        read_jsonl(empty)
    link = tmp_path / "link.json"
    link.symlink_to(mapping)
    with pytest.raises(ValueError, match="unsafe"):
        sha256_file(link)


def test_support_gate_and_realized_group_rates() -> None:
    shortcut_only = [
        {"scene_id": "a", "kind": "S"},
        {"scene_id": "a", "kind": "F"},
    ]
    shortcut = summarize_policy_support(shortcut_only, group_candidates=(2,))
    assert shortcut["status"] == "DISAGREEMENT_PRESENT_BUT_EXACT_CORRECTION_UNEXCITED"

    mixed = [
        {"scene_id": "a", "kind": "X"},
        {"scene_id": "a", "kind": "S"},
        {"scene_id": "b", "kind": "F"},
        {"scene_id": "b", "kind": "U"},
    ]
    summary = summarize_policy_support(mixed, group_candidates=(2, 4))
    assert summary["status"] == "REWARD_CONTRAST_IDENTIFIED"
    assert summary["counts"] == {"X": 1, "S": 1, "F": 1, "U": 1}
    realized = summarize_realized_groups(mixed, group_size=2)
    assert realized["group_count"] == 2
    assert realized["ESGR"] == 0.5

    no_shortcut = summarize_policy_support([{"scene_id": "a", "kind": "X"}], group_candidates=(2,))
    assert no_shortcut["status"] == "REWARD_CONTRAST_NOT_ESTIMABLE"
    with pytest.raises(ValueError, match="scene_id"):
        summarize_policy_support([{"scene_id": 3, "kind": "X"}])
    with pytest.raises(ValueError, match="complete groups"):
        summarize_realized_groups(mixed, group_size=3)
