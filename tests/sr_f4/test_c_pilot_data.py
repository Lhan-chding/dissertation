"""Fixed pilot exposures, split isolation, exact gold, and retained chart pairs."""

import copy
from collections import Counter, defaultdict

import pytest

from sr_f3_ce.prompts import CONTRACT_V2
from sr_f4 import c_pilot_data as data
from sr_f4 import c_probe_data as probe
from sr_f4 import task_contract as oracle
from sr_f4.data import base_key
from srf4_ref.sampling import seed_for


@pytest.fixture(scope="module")
def tasks():
    return data.generate_tasks()


def test_frozen_budgets_include_parents_and_all_arms():
    plan = data.plan()
    assert plan["budgets"] == {
        "train": 24576,
        "new_topk": 15360,
        "retention": 7680,
        "chartqa": 1250,
        "total": 48866,
    }
    assert data.ARMS == ("J", "EVID", "SELECT", "SELECT_SHUF")
    assert data.BLOCKS == (0, 1, 2)
    assert plan["measurement_version"] == "srf4-observable-v3-programmatic-dev-r6"
    assert plan["eval_models_per_block"] == ["PARENT", *data.ARMS]
    plan["arms"].append("unexpected")
    plan["budgets"]["total"] = -1
    assert data.plan()["arms"] == list(data.ARMS)
    assert data.plan()["budgets"]["total"] == 48866


def test_all_cells_balanced_boundaries_both_columns_and_exact_gold(tasks):
    counts = data.validate_tasks(tasks)
    assert counts == {
        data.TRAIN_SPLIT: {"worlds": 64, "queries": 128},
        data.TEST_SPLIT: {"worlds": 128, "queries": 256},
    }
    assert len({t["qid"] for t in tasks}) == 384
    for split, per_cell in ((data.TRAIN_SPLIT, 16), (data.TEST_SPLIT, 32)):
        for cell in data.CELLS:
            first = [
                t for t in tasks if t["split"] == split and t["cell"] == cell and t["branch"] == 0
            ]
            assert len(first) == per_cell
            assert Counter((t["query"]["direction"], t["query"]["aggregate"]) for t in first) == {
                (direction, aggregate): per_cell // 4
                for direction in ("largest", "smallest")
                for aggregate in ("sum", "mean")
            }
            if cell.endswith("BOUNDARY"):
                assert Counter(t["boundary_subtype"] for t in first) == {
                    "near": per_cell // 2,
                    "tied": per_cell // 2,
                }
                for subtype in ("near", "tied"):
                    assert Counter(
                        (t["query"]["direction"], t["query"]["aggregate"])
                        for t in first
                        if t["boundary_subtype"] == subtype
                    ) == {
                        (direction, aggregate): per_cell // 8
                        for direction in ("largest", "smallest")
                        for aggregate in ("sum", "mean")
                    }
    for task in tasks:
        world, query = task["world"], task["query"]
        assert world["domain"] == [10, 90]
        assert len(world["categories"]) == 2 * query["k"]
        gap = {"separated": 12, "near": 1, "tied": 0}[task["boundary_subtype"]]
        for values in world["series"].values():
            assert all(type(v) is int and 10 <= v <= 90 for v in values)
            ordered = sorted(values, reverse=query["direction"] == "largest")
            assert abs(ordered[query["k"] - 1] - ordered[query["k"]]) == gap
        assert oracle.execute(world, query) == oracle.independently_answer(world, query)
        assert task["gold_answer"] == task["gold_independent"]
        assert task["target_indices"] == oracle.selected(world, query)
        assert task["prompt"] == task["question"] + "\n" + CONTRACT_V2


def test_worlds_disjoint_deterministic_and_collision_cases_retained(tasks):
    old = {base_key(t["world"]) for t in probe.generate_tasks()}
    train = {base_key(t["world"]) for t in tasks if t["split"] == data.TRAIN_SPLIT}
    test = {base_key(t["world"]) for t in tasks if t["split"] == data.TEST_SPLIT}
    assert len(train) == 64 and len(test) == 128
    assert train.isdisjoint(test) and train.isdisjoint(old) and test.isdisjoint(old)
    assert data.generate_tasks() == tasks
    for split in (data.TRAIN_SPLIT, data.TEST_SPLIT):
        subset = [t for t in tasks if t["split"] == split]
        assert any(t["answer_collision_diagnostic"]["wrong_set_same_answer_count"] for t in subset)
        assert all(t["generation_attempt"] == 0 for t in subset)
    changed = data.generate_tasks()
    changed[0]["world"]["series"]["Alpha"][0] = 999
    changed[0]["query"]["aggregate"] = "range"
    assert changed[1] == tasks[1]
    assert changed[0]["gold_program"]["query"] == tasks[0]["query"]


def test_collision_guard_fails_without_redraw(tasks):
    changed = copy.deepcopy(tasks)
    old_world = probe.generate_tasks()[0]["world"]
    changed[0]["world"] = copy.deepcopy(old_world)
    changed[1]["world"] = copy.deepcopy(old_world)
    with pytest.raises(ValueError, match="old probe world; do not retry"):
        data.validate_tasks(changed)
    changed = copy.deepcopy(tasks)
    for task in changed:
        if (
            task["split"] == data.TEST_SPLIT
            and task["cell"] == data.CELLS[0]
            and task["world_index"] == 0
        ):
            task["world"] = copy.deepcopy(changed[0]["world"])
    with pytest.raises(ValueError, match=r"collision.*do not retry"):
        data.validate_tasks(changed)


def test_training_schedule_exact_exposure_arm_shared_and_seed_identity(tasks):
    before = copy.deepcopy(tasks)
    train_ids = {t["qid"] for t in tasks if t["split"] == data.TRAIN_SPLIT}
    all_seeds = []
    for block in data.BLOCKS:
        exposures = Counter()
        cycles = [Counter(), Counter()]
        for step in range(1, data.STEPS + 1):
            assignments = data.training_assignments(tasks, block, step)
            assert len(assignments) == 16
            assert {a["group_index"] for a in assignments} == set(range(16))
            assert Counter(a["cell"] for a in assignments) == dict.fromkeys(data.CELLS, 4)
            assert len({a["qid"] for a in assignments}) == 16
            assert sum(a["G"] for a in assignments) == 128
            for _arm in data.ARMS:
                assert (
                    data.training_assignments(
                        dict((t["qid"], t) for t in reversed(tasks)), block, step
                    )
                    == assignments
                )
            for assignment in assignments:
                assert assignment["source"] == data.TRAIN_SPLIT
                assert assignment["seeds"] == [
                    seed_for(
                        data.MASTER,
                        "train",
                        block,
                        step,
                        assignment["group_index"],
                        assignment["qid"],
                        draw,
                    )
                    for draw in range(8)
                ]
                all_seeds.extend(assignment["seeds"])
                exposures[assignment["qid"]] += 1
                cycles[(step - 1) // 8][assignment["qid"]] += 1
        assert exposures == dict.fromkeys(train_ids, 2)
        assert all(cycle == dict.fromkeys(train_ids, 1) for cycle in cycles)
    assert len(all_seeds) == len(set(all_seeds)) == 6144
    assert tasks == before


def test_eval_fixed_all_holdout_only_shared_seeds_and_input_validation(tasks):
    test_ids = {t["qid"] for t in tasks if t["split"] == data.TEST_SPLIT}
    all_seeds = []
    for block in data.BLOCKS:
        assignments = data.eval_assignments(tasks, block)
        assert len(assignments) == 256
        assert {a["qid"] for a in assignments} == test_ids
        assert Counter(a["cell"] for a in assignments) == dict.fromkeys(data.CELLS, 64)
        assert sum(a["G"] for a in assignments) == 1024
        assert data.eval_assignments(list(reversed(tasks)), block) == assignments
        for assignment in assignments:
            assert assignment["seeds"] == [
                seed_for(
                    data.MASTER, "eval", block, assignment["group_index"], assignment["qid"], draw
                )
                for draw in range(4)
            ]
            all_seeds.extend(assignment["seeds"])
    assert len(all_seeds) == len(set(all_seeds)) == 3072
    for step in (0, 17, True):
        with pytest.raises(ValueError, match="step"):
            data.training_assignments(tasks, 0, step)
    for block in (-1, 3, True):
        with pytest.raises(ValueError, match="block"):
            data.eval_assignments(tasks, block)
    with pytest.raises(ValueError, match="inventory"):
        data.training_assignments([t for t in tasks if t["split"] == data.TEST_SPLIT], 0, 1)


def retention_fixture():
    tasks = []
    for family in data.FAMILIES:
        for variant in range(8):
            for root in range(4):
                for chart in ("grouped_bar", "line"):
                    tasks.append(
                        dict(
                            qid=f"TEST_ID_{family}_{root:04d}_{chart}_v{variant}",
                            root_id=f"TEST_ID_{family}_{root:04d}",
                            pool="TEST_ID",
                            family=family,
                            variant=variant,
                            chart=chart,
                            mode=f"mode{variant}",
                            query={"family": family, "variant": variant},
                            world={"categories": ["a", "b"], "series": {"Alpha": [10 + root, 20]}},
                            score=root,
                        )
                    )
    return tasks


def test_retention_128_all_strata_preserves_chart_pairs_without_scores():
    source = retention_fixture()
    before = copy.deepcopy(source)
    selected = data.select_retention(source)
    assert len(selected) == len({t["qid"] for t in selected}) == 128
    assert Counter((t["family"], t["variant"]) for t in selected) == {
        (family, variant): 4 for family in data.FAMILIES for variant in range(8)
    }
    pairs = defaultdict(list)
    for task in selected:
        pairs[(task["family"], task["variant"], task["root_id"])].append(task)
    assert len(pairs) == 64
    assert all({t["chart"] for t in pair} == {"grouped_bar", "line"} for pair in pairs.values())
    assert data.select_retention(list(reversed(source))) == selected
    for task in source:
        task["score"] = 1e9 - task["score"]
    assert [t["qid"] for t in data.select_retention(source)] == [t["qid"] for t in selected]
    assert [{k: v for k, v in t.items() if k != "score"} for t in source] == [
        {k: v for k, v in t.items() if k != "score"} for t in before
    ]
    selected[0]["world"]["series"]["Alpha"][0] = 999
    assert all(t["world"]["series"]["Alpha"][0] != 999 for t in source)
    with pytest.raises(ValueError, match="pair"):
        data.select_retention(before[1:])
    with pytest.raises(ValueError, match="pair"):
        data.select_retention([t for t in before if t["family"] != "TOPK"])


def test_chartqa_250_deterministic_copies_and_rejects_bad_inventory():
    source = [
        dict(qid=f"chartqa_{i:04d}", pool="CHARTQA", score=i, answer=[str(i)]) for i in range(300)
    ]
    before = copy.deepcopy(source)
    selected = data.select_chartqa(source)
    assert len(selected) == len({t["qid"] for t in selected}) == 250
    assert selected == data.select_chartqa(dict((t["qid"], t) for t in reversed(source)))
    assert source == before
    selected[0]["answer"].append("changed")
    assert source == before
    for task in source:
        task["score"] = -task["score"]
    assert [t["qid"] for t in data.select_chartqa(source)] == [t["qid"] for t in selected]
    with pytest.raises(ValueError, match="250 unique"):
        data.select_chartqa(source[:249])
    with pytest.raises(ValueError, match="250 unique"):
        data.select_chartqa([*source, source[0]])
