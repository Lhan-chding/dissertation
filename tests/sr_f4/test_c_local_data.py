"""Signal-rich construction, exact lineages, fixed budgets, and shared exposure."""

import copy
from collections import Counter, defaultdict
from fractions import Fraction

import pytest

from sr_f3_ce.prompts import CONTRACT_V2
from sr_f4 import c_local_data as data
from sr_f4 import c_pilot_data as previous
from sr_f4 import c_probe_data as probe
from sr_f4 import task_contract as oracle
from sr_f4.c_local_credit import PROTOCOL_INSTRUCTION
from sr_f4.data import base_key, question
from srf4_ref.sampling import seed_for


@pytest.fixture(scope="module")
def tasks():
    return data.generate_tasks()


def test_fixed_budget_includes_parents_and_retention():
    plan = data.plan()
    assert plan["budgets"] == dict(
        train=18432, new_tasks=23040, retention=7680, chartqa=1250, evaluation=31970, total=50402
    )
    assert plan["arms"] == ["J", "EVID", "SELECT", "SHUF"]
    assert data.STEPS == 12 and data.GROUPS == 16 and data.G == 8
    assert plan["budgets"]["train"] == len(data.BLOCKS) * len(data.ARMS) * 12 * 16 * 8
    assert plan["budgets"]["new_tasks"] == len(data.BLOCKS) * (len(data.ARMS) + 1) * 384 * 4
    assert plan["budgets"]["evaluation"] == sum(
        plan["budgets"][k] for k in ("new_tasks", "retention", "chartqa")
    )
    assert plan["budgets"]["total"] == plan["budgets"]["train"] + plan["budgets"]["evaluation"]
    plan["construction"]["BINDING"] = "changed"
    plan["arms"].append("unregistered")
    assert "changed" not in data.plan()["construction"]["BINDING"]
    assert data.plan()["arms"] == list(data.ARMS)


def test_deterministic_independent_gold_lineage_balance_and_prompts(tasks):
    assert data.generate_tasks() == tasks
    assert data.validate_tasks(tasks) == {
        data.TRAIN_SPLIT: dict(roots=48, worlds=72, queries=192),
        data.TEST_SPLIT: dict(roots=96, worlds=144, queries=384),
    }
    assert len({t["qid"] for t in tasks}) == 576
    assert Counter(t["family"] for t in tasks) == {"TOPK": 480, "INTERVAL": 96}
    for task in tasks:
        world, query = task["world"], task["query"]
        data.validate_query(world, query)
        assert oracle.execute(world, query) == oracle.independently_answer(world, query)
        assert task["gold_answer"] == task["gold_independent"]
        assert task["gold_program"]["query"] == task["independent_gold_program"]["query"] == query
        assert task["target_indices"] == oracle.selected(world, query)
        assert task["target_sets"] == [world["categories"][i] for i in task["target_indices"]]
        assert (
            task["prompt"]
            == question(world, query) + " " + PROTOCOL_INSTRUCTION + "\n" + CONTRACT_V2
        )
        assert task["generation_attempt"] == 0
        assert task["generation_seed"] == (
            data.TRAIN_SEED if task["split"] == data.TRAIN_SPLIT else data.TEST_SEED
        )
    copied = copy.deepcopy(tasks)
    copied[0]["world"]["series"]["Alpha"][0] = 999
    copied[0]["query"]["k"] = 1
    assert copied[1] == tasks[1]
    assert copied[0]["gold_program"] == tasks[0]["gold_program"]


def test_fresh_worlds_and_no_public_prototype_reuse(tasks):
    old = {base_key(t["world"]) for t in previous.generate_tasks()} | {
        base_key(t["world"]) for t in probe.generate_tasks()
    }
    public = {base_key(w) for w in data._prototype_worlds()}
    assert len(public) == 9
    train = {base_key(t["world"]) for t in tasks if t["split"] == data.TRAIN_SPLIT}
    test = {base_key(t["world"]) for t in tasks if t["split"] == data.TEST_SPLIT}
    assert len(train) == 72 and len(test) == 144
    assert train.isdisjoint(test) and (train | test).isdisjoint(old | public)
    roots = defaultdict(list)
    for task in tasks:
        roots[task["root_id"]].append(task)
    assert len(roots) == 144
    for lineage in roots.values():
        assert len(lineage) == 4
        assert {t["branch"] for t in lineage} == {0, 1, 2, 3}
        assert len({t["split"] for t in lineage}) == 1
        assert len({t["world_id"] for t in lineage}) == (4 if lineage[0]["cell"] == "IMPACT" else 1)


def test_structural_competition_and_impact_controls(tasks):
    for split in (data.TRAIN_SPLIT, data.TEST_SPLIT):
        population = [t for t in tasks if t["split"] == split]
        binding = [t for t in population if t["cell"] == "BINDING" and t["branch"] == 0]
        assert {
            (t["query"]["direction"], t["design_factors"]["planned_rank_set_overlap"])
            for t in binding
        } == {(direction, overlap) for direction in ("largest", "smallest") for overlap in range(4)}
        for task in binding:
            assert (
                task["design_diagnostic"]["wrong_column_set_overlap"]
                == task["design_factors"]["planned_rank_set_overlap"]
            )
        impact = [t for t in population if t["cell"] == "IMPACT"]
        assert Counter(
            t["design_factors"]["boundary_payload_delta"] for t in impact
        ) == dict.fromkeys((0, 1, 20, 65), len(impact) // 4)
        for task in impact:
            world, query = task["world"], task["query"]
            chosen = oracle.selected(world, query)
            outgoing, incoming = task["design_factors"]["swapped_indices"]
            assert outgoing in chosen and incoming not in chosen
            swapped = [incoming if i == outgoing else i for i in chosen]
            wrong_answer = oracle.aggregate(
                [oracle.operand(world, query, i) for i in swapped], query["aggregate"]
            )
            expected = Fraction(
                -task["design_factors"]["boundary_payload_delta"],
                3 if query["aggregate"] == "mean" else 1,
            )
            assert wrong_answer - oracle.execute(world, query) == expected
            assert (
                abs(world["series"]["Alpha"][outgoing] - world["series"]["Alpha"][incoming]) == 12
            )
        signed = [t for t in population if t["cell"] == "SIGNED" and t["branch"] == 0]
        assert {
            (t["query"]["direction"], tuple(t["design_factors"]["planned_selected_signs"]))
            for t in signed
        } == {
            (direction, signs)
            for direction in ("largest", "smallest")
            for signs in ((1, 1, 1), (1, -1, 1), (-1, -1, -1), (-1, 1, -1))
        }
        assert any(
            t["design_diagnostic"]["negative_count"] and t["design_diagnostic"]["positive_count"]
            for t in signed
        )
        assert all(
            t["query"]["k"] in (2, 3) and len(t["world"]["categories"]) == 8
            for t in population
            if t["cell"] == "K_DIRECTION"
        )


def test_ties_use_printed_order_and_interval_controls_have_nonempty_sets(tasks):
    for task in tasks:
        if task["cell"] == "TIES" and task["branch"] == 0:
            positions = task["design_factors"]["printed_tie_positions"]
            labels = [task["world"]["categories"][i] for i in positions]
            assert labels != sorted(labels, key=data.MONTHS.index)
            assert len({task["world"]["series"]["Alpha"][i] for i in positions}) == 1
            assert set(task["target_indices"]) & set(positions) == set(positions[:2])
        if task["cell"] == "INTERVAL":
            assert task["target_indices"]
            if task["query"]["positive_only"]:
                assert all(
                    task["world"]["series"]["Alpha"][i] > task["world"]["series"]["Beta"][i]
                    for i in task["target_indices"]
                )


def test_fail_closed_schema_and_collision_rejection(tasks):
    changed = copy.deepcopy(tasks)
    changed[0]["query"]["positive_only"] = True
    with pytest.raises(ValueError, match="exactly the supported schema"):
        data.validate_tasks(changed)
    changed = copy.deepcopy(tasks)
    changed[0]["world"] = copy.deepcopy(data._prototype_worlds()[0])
    with pytest.raises(ValueError, match="old pilot or public prototype; do not retry"):
        data.validate_tasks(changed)
    changed = copy.deepcopy(tasks)
    changed[4]["world"] = copy.deepcopy(changed[0]["world"])
    with pytest.raises(ValueError, match=r"collision.*do not retry"):
        data.validate_tasks(changed)
    changed = copy.deepcopy(tasks)
    changed[0]["prompt"] += " hidden target list"
    with pytest.raises(ValueError, match="shared frozen protocol"):
        data.validate_tasks(changed)


def test_old_probe_worlds_are_excluded_independently_of_previous_pilot(tasks, monkeypatch):
    # The previous pilot itself need not contain an earlier probe world.
    # Excluding only previous.generate_tasks() would wrongly accept this table.
    monkeypatch.setattr(probe, "generate_tasks", lambda: [{"world": tasks[0]["world"]}])
    with pytest.raises(ValueError, match="old probe, old pilot or public prototype; do not retry"):
        data.validate_tasks(tasks)


def test_training_schedule_each_query_once_and_identical_across_arms(tasks):
    before = copy.deepcopy(tasks)
    expected = {t["qid"] for t in tasks if t["split"] == data.TRAIN_SPLIT}
    all_seeds = []
    orderings = []
    for block in data.BLOCKS:
        seen = []
        for step in range(1, 13):
            assignments = data.training_assignments(tasks, block, step)
            assert len(assignments) == 16
            assert {a["group_index"] for a in assignments} == set(range(16))
            assert sum(a["G"] for a in assignments) == 128
            assert len({a["qid"] for a in assignments}) == 16
            for _arm in data.ARMS:
                assert assignments == data.training_assignments(
                    {t["qid"]: t for t in reversed(tasks)}, block, step
                )
            for assignment in assignments:
                assert assignment["source"] == assignment["panel"] == data.TRAIN_SPLIT
                assert assignment["root_id"] in assignment["qid"]
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
                    for draw in range(data.G)
                ]
                all_seeds.extend(assignment["seeds"])
            seen.extend(a["qid"] for a in assignments)
        assert Counter(seen) == dict.fromkeys(expected, 1)
        orderings.append(tuple(seen))
    assert len(set(orderings)) == 3
    assert len(set(all_seeds)) == len(all_seeds)
    assert tasks == before


def test_eval_assignments_all_questions_and_fixed_retention_selectors(tasks):
    expected = {t["qid"] for t in tasks if t["split"] == data.TEST_SPLIT}
    for block in data.BLOCKS:
        assignments = data.eval_assignments(tasks, block)
        assert len(assignments) == 384 and {a["qid"] for a in assignments} == expected
        assert sum(a["G"] for a in assignments) == 1536
        assert {a["group_index"] for a in assignments} == set(range(384))
        for assignment in assignments:
            assert assignment["seeds"] == [
                seed_for(
                    data.MASTER, "eval", block, assignment["group_index"], assignment["qid"], draw
                )
                for draw in range(data.EVAL_G)
            ]
    assert data.plan()["retention"] == previous.plan()["retention"]
    assert data.plan()["chartqa"] == previous.plan()["chartqa"]
    assert (
        data.RETENTION_SEED == previous.RETENTION_SEED
        and data.CHARTQA_SEED == previous.CHARTQA_SEED
    )


@pytest.mark.parametrize("block, step", [(True, 1), (-1, 1), (3, 1), (0, True), (0, 0), (0, 13)])
def test_unregistered_assignments_fail(tasks, block, step):
    with pytest.raises(ValueError):
        data.training_assignments(tasks, block, step)
