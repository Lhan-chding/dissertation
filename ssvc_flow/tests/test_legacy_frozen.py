"""Qualify the read-only legacy frozen path against its original action contract."""

import hashlib
import json
from pathlib import Path

import pytest

from src.legacy_frozen import (
    annotate_legacy,
    build_legacy_prompt,
    legacy_qualification,
    load_legacy_scenes,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts/v5/study_c2/data/reward_fibers.jsonl"
MANIFEST = DATA.with_name("reward_fibers_manifest.json")


def test_actual_legacy_manifest_and_prompt_are_preserved():
    scenes = load_legacy_scenes(DATA, MANIFEST)
    assert len(scenes) == 176
    assert len({s["base_scene_id"] for s in scenes}) == 88
    assert len({s["scene_id"] for s in scenes}) == 176
    assert {s["split"] for s in scenes} == {"dev", "test", "positive_control"}
    scene = scenes[0]
    prompt = build_legacy_prompt(scene)
    assert prompt["system"] is None
    assert prompt["user"] == scene["prompt"]
    assert prompt["messages"] == [{"role": "user", "content": scene["prompt"]}]
    assert prompt["prompt_hash"] == scene["prompt_sha256"]
    assert scene["legacy_exact_reproduction"] is False
    assert legacy_qualification()["passed"] is True


@pytest.mark.parametrize(
    "raw,expected,syntax",
    [
        ("3,8,5,18", "X", True),
        ("[3,8,5,18]", "X", True),
        ("v1=3,v2=8,v3=5,v4=18", "X", True),
        ("3,8,5,17", "S", True),
        ("4,8,5,17", "W", True),
        ("3,8,5,18\nexplanation", "I", False),
        ("1,8,5,18", "I", True),
        ('{"a":3,"b":8,"c":5,"d":18}', "I", False),
    ],
)
def test_original_semantic_parser_and_original_operation(raw, expected, syntax):
    scene = {
        "truth": [3, 8, 5, 18],
        "observation": [3, 8, 5, 17],
        "operation": {"operator": "sum", "indices": [0, 1]},
    }
    result = annotate_legacy(raw, scene)
    assert result["category"] == expected
    assert result["syntax_valid"] is syntax
    assert result["constraint_satisfaction"] is None
    assert result["copy_observation"] is (raw == "3,8,5,17")


def rewrite(tmp_path, transform):
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    transform(rows)
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "fiber_rows_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "prompt_count": len(rows),
            }
        )
    )
    return data, manifest


def test_reject_corrupt_data_and_symlink(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(DATA.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        load_legacy_scenes(data, MANIFEST)
    link = tmp_path / "link.jsonl"
    link.symlink_to(DATA)
    with pytest.raises(ValueError, match="symlink"):
        load_legacy_scenes(link, MANIFEST)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda rows: rows[0].update(prompt="tampered"), "prompt hash"),
        (lambda rows: rows[1].update(scene_id=rows[0]["scene_id"]), "duplicate"),
        (lambda rows: rows[0].update(truth=[True, 8, 5, 18]), "world"),
        (lambda rows: rows[0].update(operation={"operator": "sum", "indices": [0, 0]}), "indices"),
        (lambda rows: rows[0].update(gold_answer=-999), "gold"),
        (lambda rows: rows.pop(next(i for i, r in enumerate(rows) if r["split"] == "dev")), "pair"),
    ],
)
def test_reject_invalid_rehashed_legacy_input(tmp_path, change, match):
    with pytest.raises((ValueError, TypeError), match=match):
        load_legacy_scenes(*rewrite(tmp_path, change))


def test_reject_base_pair_split_leak(tmp_path):
    def change(rows):
        pair = next(row["pair_id"] for row in rows if row["split"] == "dev")
        rows[0]["pair_id"] = pair

    with pytest.raises(ValueError, match="split leakage"):
        load_legacy_scenes(*rewrite(tmp_path, change))


def test_directory_alias_is_allowed_but_file_symlink_is_not(tmp_path):
    alias = tmp_path / "storage_alias"
    alias.symlink_to(DATA.parent, target_is_directory=True)
    assert len(load_legacy_scenes(alias / DATA.name, alias / MANIFEST.name)) == 176
