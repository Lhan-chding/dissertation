"""Decision panels are selected from metadata before any model observations."""

import copy
import itertools
import json
from collections import Counter

import pytest

from src.core import canonical_hash
from src.prompts import OPERATIONS, build_prompt
from src.r3_inputs import CHARTS, FAMILIES, INTERFACES, select_control_panel


@pytest.fixture
def controls():
    rows = []
    for family, chart, operation, index in itertools.product(
        FAMILIES, CHARTS, OPERATIONS, range(8)
    ):
        row = {
            "split": "control",
            "interface": None,
            "base_scene_id": f"{family}-{chart}-{operation}-{index}",
            "constraint_family": family,
            "chart_type": chart,
            "operation": operation,
            "observed_world": [9, 2, 3, 4],
            "truth_world": [1, 2, 3, 4],
            "image_path": "images/fixture.png",
            "image_hash": "a" * 64,
            "cue": (
                {"family": family, "known_index": 0, "known_value": 1}
                if family == "duplicate_encoding"
                else {"family": family, "edges": [[0, 1, 3], [0, 2, 4], [0, 3, 5]]}
                if family == "cross_series"
                else {"family": family}
            ),
        }
        row["prompt_hashes"] = {i: build_prompt(row, i)["prompt_hash"] for i in INTERFACES}
        rows.append(row)
    return rows


def test_balanced_disjoint_deterministic_panels(controls):
    from src.decision_modeling.panels import build_panels

    before = copy.deepcopy(controls)
    old = {p["base_scene_id"] for p in select_control_panel(controls)}
    result = build_panels(controls, config={"version": 1}, train_base_scene_ids=["training"])
    assert result == build_panels(
        list(reversed(controls)), config={"version": 1}, train_base_scene_ids=["training"]
    )
    assert result["status"] == "READY"
    assert result["excluded"]["old_R3"] == sorted(old)
    all_ids = set()
    for name in ("P", "E"):
        panel = result["panels"][name]
        ids = {p["base_scene_id"] for p in panel}
        assert len(panel) == 72 and len(ids) == 36
        assert not ids & (old | all_ids)
        all_ids |= ids
        assert set(
            Counter(
                (p["family"], p["chart_type"], p["operation"], p["interface"]) for p in panel
            ).values()
        ) == {2}
        assert all(p["max_new_tokens"] == 64 for p in panel)
    result["panels"]["P"][0]["scene"]["truth_world"][0] = 100
    assert controls == before


def test_shortage_is_reported_without_opening_other_splits(controls):
    from src.decision_modeling.panels import build_panels

    excluded = [s["base_scene_id"] for s in controls[:8]]
    result = build_panels(
        controls, config={}, train_base_scene_ids=[], debug_base_scene_ids=excluded
    )
    assert result["status"] == "INSUFFICIENT_CONTROL"
    assert result["total_deficit_scenes"] == 4
    assert result["panels"] == {}
    assert result["sealed_confirm_read"] is False


@pytest.mark.parametrize("split", ["train", "dev", "confirm", "discovery"])
def test_rejects_wrong_split(controls, split):
    from src.decision_modeling.panels import build_panels

    controls[0]["split"] = split
    with pytest.raises(ValueError, match="control"):
        build_panels(controls, config={}, train_base_scene_ids=[])


def test_rejects_duplicate_identity_and_training_overlap(controls):
    from src.decision_modeling.panels import build_panels

    with pytest.raises(ValueError, match="duplicate"):
        build_panels([*controls, controls[0]], config={}, train_base_scene_ids=[])
    with pytest.raises(ValueError, match="train/control"):
        build_panels(controls, config={}, train_base_scene_ids=[controls[0]["base_scene_id"]])


def test_binding_hash_and_config_identity(tmp_path, controls):
    from src.decision_modeling.panels import prepare_panels
    from src.modeling_v3.io import sha256_file

    control = tmp_path / "control.jsonl"
    control.write_text("".join(json.dumps(s) + "\n" for s in controls))
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"split": "train", "base_scene_id": "train-only"}) + "\n")
    bindings = {
        "control": {"path": str(control), "sha256": sha256_file(control)},
        "train": {"path": str(train), "sha256": sha256_file(train)},
    }
    config = {"protocol": "test"}
    result = prepare_panels(config, bindings, data_root=tmp_path)
    assert result["config_hash"] == canonical_hash(config)
    assert result["input_bindings"] == bindings
    control.write_text(control.read_text() + "\n")
    with pytest.raises(ValueError, match="bytes"):
        prepare_panels(config, bindings, data_root=tmp_path)


def test_config_debug_exclusion_and_prompt_identity_is_host_independent(controls):
    from src.decision_modeling.panels import build_panels

    excluded = controls[0]["base_scene_id"]
    config = {"seed": 20260919, "panels": {"debug_base_scene_ids": [excluded]}}
    first = build_panels(controls, config=config, train_base_scene_ids=[], data_root="host-a")
    second = build_panels(controls, config=config, train_base_scene_ids=[], data_root="host-b")
    for name in ("P", "E"):
        assert excluded not in {p["base_scene_id"] for p in first["panels"][name]}
        assert [p["prompt_id"] for p in first["panels"][name]] == [
            p["prompt_id"] for p in second["panels"][name]
        ]
    with pytest.raises(ValueError, match="seed"):
        build_panels(controls, config={"seed": 1}, train_base_scene_ids=[])
