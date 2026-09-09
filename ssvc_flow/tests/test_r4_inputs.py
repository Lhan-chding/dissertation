"""R4 data-only contracts against the existing frozen, local source datasets."""

import copy
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from src.core import canonical_hash, file_hash
from src.legacy_frozen import build_legacy_prompt, load_legacy_scenes
from src.prompts import build_prompt


@pytest.fixture(scope="module")
def data():
    root = Path("data/generated")
    result = {
        split: [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "dev", "calibration", "control", "ood")
    }
    result["id"] = [
        scene for split in ("train", "dev", "calibration", "control") for scene in result[split]
    ]
    result["legacy"] = load_legacy_scenes(
        "../artifacts/v5/study_c2/data/reward_fibers.jsonl",
        "../artifacts/v5/study_c2/data/reward_fibers_manifest.json",
    )
    result["legacy_lock"] = {
        "actual_max_new_tokens": 48,
        "generation": {
            "do_sample": True,
            "max_new_tokens": 48,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "num_beams": 1,
        },
        "data_hash": "test-lock-bound-data",
    }
    return result


def _plan(data, **kwargs):
    from src.r4_inputs import build_r4_plan

    return build_r4_plan(
        data["train"],
        data["dev"],
        data["ood"],
        data["legacy"],
        id_scenes=data["id"],
        legacy_lock=data["legacy_lock"],
        **kwargs,
    )


def test_train_pool_fixed_interfaces_and_seed17_schedule(data):
    from src.r4_inputs import STRATA, build_train_schedule

    before = copy.deepcopy(data["train"])
    global_rng = random.getstate()
    schedule = build_train_schedule(data["train"])
    assert random.getstate() == global_rng
    records = schedule["train_prompts"]
    lookup = {row["prompt_id"]: row for row in records}
    assert len(lookup) == len(records) == 576
    assert schedule["sampler_seed"] == 17
    steps = schedule["train_steps"]
    assert len(steps) == 64 and all(len(set(step)) == 4 for step in steps)
    flat = [key for step in steps for key in step]
    assert len(set(flat)) == 256
    assert Counter((row["family"], row["interface"]) for row in records) == {s: 96 for s in STRATA}
    for start in range(0, 60, 6):
        keys = [key for step in steps[start : start + 6] for key in step]
        assert Counter((lookup[k]["family"], lookup[k]["interface"]) for k in keys) == {
            s: 4 for s in STRATA
        }
    tail = [key for step in steps[60:] for key in step]
    assert Counter((lookup[k]["family"], lookup[k]["interface"]) for k in tail) == {
        stratum: 3 if index < 4 else 2 for index, stratum in enumerate(STRATA)
    }
    assert schedule == build_train_schedule(list(reversed(data["train"])))
    for record in records:
        assert record["interface"] == record["scene"]["interface"]
        assert record["prompt"] == build_prompt(record["scene"], record["interface"])
        assert record["max_new_tokens"] == 64 and record["enable_thinking"] is False
    records[0]["scene"]["truth_world"][0] = -100
    assert data["train"] == before


@pytest.mark.parametrize("fault", ["duplicate", "split", "count", "interface", "cell", "hash"])
def test_train_rejects_wrong_pool_and_changed_original_prompt(data, fault):
    from src.r4_inputs import build_train_schedule

    rows = copy.deepcopy(data["train"])
    if fault == "duplicate":
        rows[1] = rows[0]
    elif fault == "split":
        rows[0]["split"] = "control"
    elif fault == "count":
        rows.pop()
    elif fault == "interface":
        rows[0]["interface"] = None
    elif fault == "cell":
        rows[0]["constraint_family"] = "unknown"
    else:
        rows[0]["prompt_hashes"][rows[0]["interface"]] = "0" * 64
    with pytest.raises(ValueError):
        build_train_schedule(rows)


def test_dev_panel_is_balanced_stable_subset_of_full_eval(data):
    from src.r4_inputs import build_dev_prompts, select_dev_panel

    panel, full = select_dev_panel(data["dev"]), build_dev_prompts(data["dev"])
    assert len(panel) == 72 and len(full) == 288
    bases = {r["base_scene_id"]: r["scene"] for r in panel}
    assert len(bases) == 36
    assert set(
        Counter(
            (s["constraint_family"], s["chart_type"], s["operation"]) for s in bases.values()
        ).values()
    ) == {2}
    assert {r["prompt_id"] for r in panel} < {r["prompt_id"] for r in full}
    lookup = {r["prompt_id"]: r for r in full}
    assert all(lookup[r["prompt_id"]] == r for r in panel)
    assert panel == select_dev_panel(list(reversed(data["dev"])))
    annotated = [
        {**r, "model_success": i % 2 == 0, "reward": -i} for i, r in enumerate(data["dev"])
    ]
    assert [r["prompt_id"] for r in select_dev_panel(annotated)] == [r["prompt_id"] for r in panel]
    with pytest.raises(ValueError):
        select_dev_panel(data["train"])
    with pytest.raises(ValueError):
        build_dev_prompts(data["dev"][:-1])


def test_existing_graph_ood_is_unique_balanced_and_uses_original_images(data):
    from src.constraint_solver import solve
    from src.r4_inputs import select_graph_ood_panel

    records = select_graph_ood_panel(data["ood"], id_scenes=data["id"])
    assert len(records) == 72
    bases = {r["base_scene_id"]: r["scene"] for r in records}
    assert len(bases) == 36
    counts = Counter((s["graph_topology"], s["chart_type"], s["operation"]) for s in bases.values())
    assert len(counts) == 12 and set(counts.values()) == {3}
    assert {s["constraint_family"] for s in bases.values()} == {"cross_series"}
    assert not {s["truth_structure_hash"] for s in bases.values()} & {
        s["truth_structure_hash"] for s in data["id"]
    }
    assert not {s["image_hash"] for s in bases.values()} & {s["image_hash"] for s in data["id"]}
    for record in records:
        scene = record["scene"]
        assert solve(scene["observed_world"], scene["cue"]) == [scene["truth_world"]]
        assert max(scene["truth_world"]) < 50
        assert scene["render_variant"] == "standard" and scene["expression_variant"] == "fixed_N_v1"
        assert file_hash(Path("data/generated") / scene["image_path"]) == scene["image_hash"]
        assert record["prompt"] == build_prompt(scene, record["interface"])
        assert record["evaluation_domain"] == "graph_OOD"
    assert records == select_graph_ood_panel(
        list(reversed(data["ood"])), id_scenes=list(reversed(data["id"]))
    )


@pytest.mark.parametrize("collision", ["base_scene_id", "truth_structure_hash", "image_hash"])
def test_ood_rejects_id_overlap_before_selection(data, collision):
    from src.r4_inputs import select_graph_ood_panel

    rows = copy.deepcopy(data["id"])
    graph = next(r for r in data["ood"] if r["ood_subtype"] == "graph_structure")
    rows[0][collision] = graph[collision]
    with pytest.raises(ValueError, match="overlap"):
        select_graph_ood_panel(data["ood"], id_scenes=rows)


@pytest.mark.parametrize(
    "fault",
    ["numeric", "render", "topology", "solution", "truth_hash", "changed_index", "duplicate_edge"],
)
def test_ood_rejects_invalid_candidate_instead_of_filtering_it(data, fault):
    from src.r4_inputs import select_graph_ood_panel

    rows = copy.deepcopy(data["ood"])
    graph = next(r for r in rows if r["ood_subtype"] == "graph_structure")
    if fault == "numeric":
        graph["truth_world"][0] = 50
    elif fault == "render":
        graph["render_variant"] = "alternate_palette"
    elif fault == "topology":
        graph["graph_topology"] = "cycle" if graph["graph_topology"] == "path" else "path"
    elif fault == "solution":
        graph["cue"]["edges"][0][2] += 1
    elif fault == "truth_hash":
        graph["truth_structure_hash"] = "0" * 64
    elif fault == "changed_index":
        graph["changed_index"] = (graph["changed_index"] + 1) % 4
    else:
        graph["cue"]["edges"].append(graph["cue"]["edges"][0])
    with pytest.raises(ValueError):
        select_graph_ood_panel(rows, id_scenes=data["id"])


def test_ood_rejects_incomplete_grid_and_duplicate_candidates(data):
    from src.r4_inputs import select_graph_ood_panel

    rows = [r for r in data["ood"] if r["graph_topology"] != "cycle"]
    with pytest.raises(ValueError, match="Insufficient"):
        select_graph_ood_panel(rows, id_scenes=data["id"])
    with pytest.raises(ValueError, match="duplicate"):
        select_graph_ood_panel([*data["ood"], data["ood"][0]], id_scenes=data["id"])


def test_ood_rejects_unknown_subtype_in_source_pool(data):
    from src.r4_inputs import select_graph_ood_panel

    rows = copy.deepcopy(data["ood"])
    rows[0]["ood_subtype"] = "unregistered"
    with pytest.raises(ValueError, match="subtype"):
        select_graph_ood_panel(rows, id_scenes=data["id"])


def test_legacy_keeps_all_original_bytes_and_48_token_lock(data):
    from src.r4_inputs import build_legacy_eval_prompts

    records = build_legacy_eval_prompts(data["legacy"], legacy_lock=data["legacy_lock"])
    assert len(records) == len({r["prompt_id"] for r in records}) == 176
    assert len({r["base_scene_id"] for r in records}) == 88
    for r in records:
        assert r["track"] == "L" and r["max_new_tokens"] == 48
        assert r["prompt"] == build_legacy_prompt(r["scene"])
        assert r["prompt"]["messages"] == [{"role": "user", "content": r["scene"]["prompt"]}]
        assert r["generation"] == data["legacy_lock"]["generation"]
    assert records == build_legacy_eval_prompts(
        list(reversed(data["legacy"])), legacy_lock=data["legacy_lock"]
    )
    for changed in ({"actual_max_new_tokens": 64}, {"generation": {"max_new_tokens": 64}}):
        with pytest.raises(ValueError, match="48"):
            build_legacy_eval_prompts(
                data["legacy"], legacy_lock={**data["legacy_lock"], **changed}
            )
    with pytest.raises(ValueError):
        build_legacy_eval_prompts(data["legacy"][:-1], legacy_lock=data["legacy_lock"])
    damaged = copy.deepcopy(data["legacy"])
    damaged[0]["prompt"] += " "
    with pytest.raises(ValueError, match="hash"):
        build_legacy_eval_prompts(damaged, legacy_lock=data["legacy_lock"])


def test_plan_is_serializable_counts_are_separate_and_identity_has_no_arm_or_step(data):
    plan = _plan(data, data_root="data/generated", ood_data_root="data/generated")
    assert json.loads(json.dumps(plan)) == plan
    assert plan["plan_hash"] == canonical_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    expected = {
        "train_prompts": 576,
        "dev_panel_prompts": 72,
        "dev_prompts": 288,
        "legacy_prompts": 176,
        "ood_prompts": 72,
    }
    assert {key: len(plan[key]) for key in expected} == expected
    assert plan["counts"]["train_rollouts_two_arms"] == 4096
    assert plan["counts"]["evaluation_rollouts_step32_and_step64"] == 9728
    assert plan["counts"]["evaluation_rollouts_step0_shared"] == 576
    assert plan["legacy_lock_hash"] == canonical_hash(data["legacy_lock"])
    for key in expected:
        for row in plan[key]:
            assert "arm" not in row and "checkpoint_step" not in row
            assert row["scene_hash"] == canonical_hash(row["scene"])
            assert row["data_root"] == (None if row["track"] == "L" else "data/generated")
    reordered = {
        **data,
        **{key: list(reversed(data[key])) for key in ("train", "dev", "ood", "legacy", "id")},
    }
    assert plan == _plan(reordered, data_root="data/generated", ood_data_root="data/generated")


def test_plan_requires_complete_train_dev_exclusions_and_disjoint_splits(data):
    from src.r4_inputs import build_r4_plan

    with pytest.raises(ValueError, match="ID"):
        build_r4_plan(
            data["train"],
            data["dev"],
            data["ood"],
            data["legacy"],
            id_scenes=data["train"],
            legacy_lock=data["legacy_lock"],
        )
    damaged = copy.deepcopy(data)
    damaged["dev"][0]["base_scene_id"] = damaged["train"][0]["base_scene_id"]
    with pytest.raises(ValueError, match="overlap"):
        _plan(damaged)
