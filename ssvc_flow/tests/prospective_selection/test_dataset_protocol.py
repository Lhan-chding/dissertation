"""Synthetic metadata fixtures; these tests make no real-model claim."""

import copy
from collections import Counter

import pytest

from src.core import canonical_hash
from src.prompts import build_prompt
from src.prospective_selection.dataset import build_dataset, split_training_pool
from src.prospective_selection.protocol import (
    get_lineage,
    load_protocol,
    validate_protocol,
    workload,
)
from src.prospective_selection.sampler import evaluation_seed, training_seed
from src.r4_inputs import DEV_CELLS, INTERFACES, STRATA, _n_record


def scene(index, split, family, chart, operation, interface=None):
    truth = [index // 100, index % 100, 3, 4]
    row = {
        "base_scene_id": f"{split}-{index}",
        "split": split,
        "constraint_family": family,
        "chart_type": chart,
        "operation": operation,
        "interface": interface,
        "truth_world": truth,
        "truth_structure_hash": canonical_hash(truth),
        "observed_world": [99, *truth[1:]],
        "changed_index": 0,
        "cue": {
            "family": family,
            "known_index": 0,
            "known_value": truth[0],
            "edges": [[0, 1, sum(truth[:2])]],
        },
        "image_hash": canonical_hash(["fixture-image", index]),
        "image_path": f"fixture/{index}.png",
    }
    row["prompt_hashes"] = {
        i: build_prompt(row, i)["prompt_hash"]
        for i in (INTERFACES if interface is None else [interface])
    }
    return row


@pytest.fixture(scope="module")
def inputs():
    train = [
        scene(i * 96 + j, "train", family, "line", "sum4", interface)
        for i, (family, interface) in enumerate(STRATA)
        for j in range(96)
    ]
    panels = {}
    for split, count, start in [("dev", 8, 1000), ("confirm", 9, 2000), ("control", 2, 3000)]:
        panels[split] = [
            scene(start + i * count + j, split, family, chart, operation)
            for i, (family, chart, operation) in enumerate(DEV_CELLS)
            for j in range(count)
        ]
    p = [_n_record(s, interface) for s in panels["control"] for interface in INTERFACES]
    return train, panels["dev"], panels["confirm"], {"panels": {"P": p}}


def build(inputs, **kwargs):
    train, dev, confirm, old = inputs
    defaults = {
        "existing_panels": old,
        "confirm_exposure": {
            "status": "COMPLETE",
            "exposed_base_scene_ids": [],
            "exposed_truth_hashes": [],
            "evidence_paths": ["synthetic-fixture-audit"],
        },
        "config": load_protocol(),
    }
    defaults.update(kwargs)
    return build_dataset(train, dev, confirm, **defaults)


def test_protocol_workload_is_derived_and_lineages_are_distinct():
    config = load_protocol()
    assert workload(config) == config["workload_min_test"]
    assert workload(config, 48, 64) == config["workload_max_registered_test"]
    assert get_lineage(config, 63004)["anchors"] == [96]
    invalid = copy.deepcopy(config)
    invalid["origins"]["tuning"][0]["seed"] = 61001
    with pytest.raises(ValueError, match="Lineage"):
        validate_protocol(invalid)
    with pytest.raises(ValueError, match="N/m"):
        workload(config, 13, 16)


def test_pools_schedules_balanced_disjoint_and_input_immutable(inputs):
    before = copy.deepcopy(inputs)
    data = build(inputs)
    assert inputs == before
    source, branch = data["source_prompts"], data["continuation_prompts"]
    assert Counter((p["family"], p["interface"]) for p in source) == {s: 64 for s in STRATA}
    assert Counter((p["family"], p["interface"]) for p in branch) == {s: 32 for s in STRATA}
    assert not {p["base_scene_id"] for p in source} & {p["base_scene_id"] for p in branch}
    assert not {p["scene"]["truth_structure_hash"] for p in source} & {
        p["scene"]["truth_structure_hash"] for p in branch
    }
    for schedule, count in [
        (data["source_schedule"], 384),
        *[(s, 128) for s in data["branch_schedules"].values()],
    ]:
        order = [pid for batch in schedule["train_steps"] for pid in batch]
        assert len(order) == len(set(order)) == count
        assert all(len(batch) == 4 for batch in schedule["train_steps"])
    assert (
        data["branch_schedules"]["1"]["train_steps"] != data["branch_schedules"]["2"]["train_steps"]
    )
    shuffled = tuple(list(reversed(x)) if isinstance(x, list) else x for x in inputs)
    again = build(shuffled)
    assert again["source_schedule"] == data["source_schedule"]
    assert again["branch_schedules"] == data["branch_schedules"]


def test_panels_paired_metadata_balanced_existing_p_reused(inputs):
    data = build(inputs)
    assert [p["prompt_id"] for p in data["panels"]["P"]] == [
        p["prompt_id"] for p in inputs[3]["panels"]["P"]
    ]
    for name, count in [("D", 4), ("T", 8)]:
        panel = data["panels"][name]
        assert Counter((r["family"], r["chart_type"], r["operation"]) for r in panel) == {
            s: count * 2 for s in DEV_CELLS
        }
        assert all(n == 2 for n in Counter(r["base_scene_id"] for r in panel).values())
    assert len(data["panels"]["T_H8"]) == 24
    assert {p["base_scene_id"] for p in data["panels"]["T_H8"]} <= {
        p["base_scene_id"] for p in data["panels"]["T"]
    }


def test_no_exposure_evidence_fails_closed_but_development_continues(inputs):
    with pytest.raises(ValueError, match="exposure audit"):
        build(inputs, confirm_exposure=None)
    data = build(inputs, confirm_exposure=None, development_only=True)
    assert data["status"] == "DEVELOPMENT_READY"
    assert data["test_panel_status"].startswith("BLOCKED")
    assert data["panels"]["T"] == [] and len(data["source_prompts"]) == 384
    full = build(inputs)
    assert data["source_schedule"] == full["source_schedule"]
    assert data["panels"]["D"] == full["panels"]["D"]


def test_exposed_truth_and_id_excluded_before_selection(inputs):
    exposed = inputs[2][0]
    audit = {
        "status": "COMPLETE",
        "exposed_base_scene_ids": [exposed["base_scene_id"]],
        "exposed_truth_hashes": [],
        "evidence_paths": ["fixture"],
    }
    data = build(inputs, confirm_exposure=audit)
    assert exposed["base_scene_id"] not in {p["base_scene_id"] for p in data["panels"]["T"]}
    audit["exposed_base_scene_ids"].append(inputs[2][1]["base_scene_id"])
    with pytest.raises(ValueError, match="Insufficient unexposed"):
        build(inputs, confirm_exposure=audit)


def test_truth_alias_and_cross_split_truth_overlap_rejected(inputs):
    train = copy.deepcopy(inputs[0])
    train[1]["truth_world"] = train[0]["truth_world"]
    train[1]["truth_structure_hash"] = train[0]["truth_structure_hash"]
    with pytest.raises(ValueError, match="Duplicate scene or truth"):
        split_training_pool(train)
    train, dev, confirm, old = copy.deepcopy(inputs)
    dev[0]["truth_world"] = train[0]["truth_world"]
    dev[0]["truth_structure_hash"] = train[0]["truth_structure_hash"]
    # All four chosen scenes are forced to collide, independent of metadata rank.
    for row in dev[:8]:
        row["truth_world"] = train[int(row["base_scene_id"].split("-")[1]) % 576]["truth_world"]
        row["truth_structure_hash"] = canonical_hash(row["truth_world"])
    with pytest.raises(ValueError, match="overlap"):
        build((train, dev, confirm, old))


def test_random_roles_recipe_exclusion_and_policy_independence():
    args = dict(experiment_seed=17, lineage=61001, repeat=1, step=0, prompt_id="P", draw=0)
    assert training_seed(**args) == training_seed(**args)
    assert training_seed(**args) != training_seed(**{**args, "repeat": 2})
    with pytest.raises(TypeError):
        training_seed(**args, recipe="R0")
    ev = dict(
        experiment_seed=17,
        lineage=61001,
        repeat=1,
        policy_id="p1",
        panel_id="D",
        horizon=32,
        prompt_id="P",
        draw=0,
    )
    assert evaluation_seed(**ev) != evaluation_seed(**{**ev, "policy_id": "p2"})
    assert evaluation_seed(**ev) != training_seed(**args)
