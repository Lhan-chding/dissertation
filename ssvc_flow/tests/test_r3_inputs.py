"""R3 panel/plan tests use dataset metadata only, never model generation."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from src.core import canonical_hash
from src.prompts import build_prompt


@pytest.fixture
def data():
    return {
        split: [
            json.loads(line)
            for line in (Path("data/generated") / f"{split}.jsonl").read_text().splitlines()
        ]
        for split in ("train", "control")
    }


def test_train_panel_preserves_assigned_interface_and_balanced_rotation(data):
    from src.r3_inputs import STRATA, select_train_panel

    before = copy.deepcopy(data["train"])
    selected = select_train_panel(data["train"])
    assert len(selected) == len({r["base_scene_id"] for r in selected}) == 48
    assert Counter((r["family"], r["interface"]) for r in selected) == {pair: 8 for pair in STRATA}
    assert [(r["family"], r["interface"]) for r in selected] == list(STRATA) * 8
    assert selected == select_train_panel(list(reversed(data["train"])))
    for row in selected:
        assert row["interface"] == row["scene"]["interface"]
        assert row["prompt"] == build_prompt(row["scene"], row["interface"])
        assert row["scene_hash"] == canonical_hash(row["scene"])
    selected[0]["scene"]["truth_world"][0] = -100
    assert data["train"] == before


@pytest.mark.parametrize(
    "split,selector", [("train", "select_train_panel"), ("control", "select_control_panel")]
)
def test_rejects_duplicates_wrong_split_and_insufficient_groups(data, split, selector):
    from src import r3_inputs

    choose = getattr(r3_inputs, selector)
    rows = data[split]
    with pytest.raises(ValueError, match="duplicate"):
        choose([*rows, copy.deepcopy(rows[0])])
    with pytest.raises(ValueError, match=split):
        choose([{**rows[0], "split": "dev"}, *rows[1:]])
    with pytest.raises(ValueError, match="Insufficient"):
        choose(rows[:3])
    with pytest.raises(ValueError, match="interface"):
        choose([{**rows[0], "interface": "natural"}, *rows[1:]])
    with pytest.raises(ValueError, match="metadata"):
        choose([{**rows[0], "chart_type": "unknown"}, *rows[1:]])


def test_control_panel_pairs_interfaces_and_balances_metadata(data):
    from src.r3_inputs import FAMILIES, INTERFACES, select_control_panel

    selected = select_control_panel(data["control"])
    assert len(selected) == 48
    assert selected == select_control_panel(list(reversed(data["control"])))
    scenes = {r["base_scene_id"]: r["scene"] for r in selected}
    assert len(scenes) == 24
    for family in FAMILIES:
        group = [scene for scene in scenes.values() if scene["constraint_family"] == family]
        assert len(group) == 8
        assert sorted(Counter(s["chart_type"] for s in group).values()) == [4, 4]
        assert sorted(Counter(s["operation"] for s in group).values()) == [2, 3, 3]
        assert sorted(Counter((s["chart_type"], s["operation"]) for s in group).values()) == [
            1,
            1,
            1,
            1,
            2,
            2,
        ]
    for sid in scenes:
        records = [r for r in selected if r["base_scene_id"] == sid]
        assert {r["interface"] for r in records} == set(INTERFACES)
        assert records[0]["scene"] == records[1]["scene"]
    assert all(row["prompt"] == build_prompt(row["scene"], row["interface"]) for row in selected)


def test_control_balancing_handles_available_metadata_without_outcome_filtering(data):
    from src.r3_inputs import select_control_panel

    limited = [r for r in data["control"] if r["chart_type"] == "line"]
    selected = select_control_panel(limited)
    assert len(selected) == 48
    for family in {r["family"] for r in selected}:
        scenes = {r["base_scene_id"]: r["scene"] for r in selected if r["family"] == family}
        assert sorted(Counter(s["operation"] for s in scenes.values()).values()) == [2, 3, 3]
    annotated = [{**r, "model_success": i % 2 == 0, "category": "I"} for i, r in enumerate(limited)]
    after = select_control_panel(annotated)
    assert [r["prompt_id"] for r in after] == [r["prompt_id"] for r in selected]


def test_complete_plan_counts_disjointness_and_checkpoint_independence(data):
    from src.r3_inputs import build_r3_plan

    cold = build_r3_plan(data["train"], data["control"])
    warm = build_r3_plan(list(reversed(data["train"])), list(reversed(data["control"])))
    assert cold == warm
    assert json.loads(json.dumps(cold)) == cold
    assert len(cold["banks"]) == 12 and all(len(bank) == 4 for bank in cold["banks"])
    assert [pid for bank in cold["banks"] for pid in bank] == [
        r["prompt_id"] for r in cold["train_prompts"]
    ]
    assert len({pid for bank in cold["banks"] for pid in bank}) == 48
    assert cold["reuse_check_bank_indices"] == [1, 11]
    assert cold["response_banks"] == [0, 6]
    assert cold["lambda_candidates"] == [0.0, 0.01, 0.25, 1.0, 2.0]
    assert cold["counts"]["train_rollouts"] == 384
    assert cold["counts"]["control_proposal_rollouts"] == 768
    assert cold["counts"]["candidate_updates"] == 60
    assert cold["counts"]["warm_direct_rollouts"] == 3072
    assert cold["plan_hash"] == canonical_hash({k: v for k, v in cold.items() if k != "plan_hash"})
    overlap = [
        {**data["control"][0], "base_scene_id": data["train"][-1]["base_scene_id"]},
        *data["control"][1:],
    ]
    with pytest.raises(ValueError, match="overlap"):
        build_r3_plan(data["train"], overlap)


def test_input_and_prompt_hashes_bind_metadata_but_not_model_results(data):
    from src.r3_inputs import build_r3_plan, select_train_panel

    original = build_r3_plan(data["train"], data["control"])
    annotated = [{**r, "model_success": False} for r in data["train"]]
    selected = select_train_panel(annotated)
    assert [r["prompt_id"] for r in selected] == [r["prompt_id"] for r in original["train_prompts"]]
    assert original["plan_hash"] != build_r3_plan(annotated, data["control"])["plan_hash"]
    sid = original["train_prompts"][0]["base_scene_id"]
    changed = [
        {**r, "observed_world": [91, 92, 93, 94]} if r["base_scene_id"] == sid else r
        for r in data["train"]
    ]
    after = select_train_panel(changed)
    assert [r["base_scene_id"] for r in selected] == [r["base_scene_id"] for r in after]
    assert selected[0]["prompt_id"] != after[0]["prompt_id"]
