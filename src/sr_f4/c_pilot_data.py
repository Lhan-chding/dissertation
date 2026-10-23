"""Frozen data and arm-independent schedules for the small C F-prime pilot.

All factories return fresh dictionaries and never mutate caller-owned tasks.
``generate_tasks`` returns both splits; assignment/selection helpers accept an
iterable of task dictionaries or a qid-to-task mapping. Training steps are 1..16.
No model calls, scores, rejection sampling, file scans, or training occur here.
"""

from __future__ import annotations

import copy
import random
from collections import Counter, defaultdict
from collections.abc import Mapping

from sr_f3_ce.prompts import CONTRACT_V2
from srf4_ref.sampling import seed_for

from . import c_probe_data as probe
from . import task_contract as oracle
from .data import FAMILIES, base_key, question

VERSION = "srf4-c-fprime-pilot-r1"
MASTER = 2026102307
ARMS = ("J", "EVID", "SELECT", "SELECT_SHUF")
BLOCKS = (0, 1, 2)
STEPS = 16
GROUPS_PER_STEP = 16
G = 8
EVAL_G = 4
MEASUREMENT_VERSION = "srf4-observable-v3-programmatic-dev-r6"
CELLS = probe.CELLS
TRAIN_SPLIT = "C_FP_TRAIN_R1"
TEST_SPLIT = "C_FP_TEST_R1"
TRAIN_SEED = 2026102307
TEST_SEED = 2026102308
RETENTION_SEED = 2026102309
CHARTQA_SEED = 2026102310
TRAIN_WORLDS_PER_CELL = 16
TEST_WORLDS_PER_CELL = 32
TRAIN_WORLDS = 64
TEST_WORLDS = 128
TRAIN_QUERIES = 128
TEST_QUERIES = 256
RETENTION_QUERIES = 128
CHARTQA_QUERIES = 250


def _rows(tasks):
    return list(tasks.values() if isinstance(tasks, Mapping) else tasks)


def _make_tasks(split, worlds_per_cell, seed):
    tasks = []
    for cell in CELLS:
        n = 6 if cell.startswith("T6-") else 8
        k = n // 2
        separated = cell.endswith("SEPARATED")
        for index in range(worlds_per_cell):
            gap = 12 if separated else 1 if index < worlds_per_cell // 2 else 0
            world = {
                "categories": probe._categories(n, index),
                "series": {
                    series: probe._column(
                        random.Random(seed_for(seed, "world", split, cell, index, series)), k, gap
                    )
                    for series in ("Alpha", "Beta")
                },
                "domain": [10, 90],
            }
            world_id = f"SRF4_{split}_{cell.replace('-', '_')}_{index:03d}"
            for branch, (ranking, value) in enumerate((("Alpha", "Beta"), ("Beta", "Alpha"))):
                query = dict(
                    family="TOPK",
                    k=k,
                    direction="largest" if (index // 2) % 2 == 0 else "smallest",
                    selection_series=ranking,
                    value=value,
                    aggregate="sum" if index % 2 == 0 else "mean",
                )
                answer = oracle.execute(world, query)
                independent = oracle.independently_answer(world, query)
                if answer != independent:
                    raise AssertionError("Independent pilot gold mismatch")
                target = oracle.selected(world, query)
                text = question(world, query)
                tasks.append(
                    dict(
                        qid=f"{world_id}_{branch}",
                        root_id=world_id,
                        world_id=world_id,
                        pair_id=world_id,
                        split=split,
                        pool=split,
                        family="TOPK",
                        cell=cell,
                        world_index=index,
                        branch=branch,
                        variant=index % 8,
                        purpose="c_fprime_pilot",
                        world=copy.deepcopy(world),
                        query=query,
                        question=text,
                        prompt=text + "\n" + CONTRACT_V2,
                        chart="grouped_bar",
                        render_style="main",
                        wording_template="standard",
                        boundary_regime="separated" if separated else "near_or_tied",
                        boundary_subtype="separated" if separated else "near" if gap else "tied",
                        boundary_by_series=probe._boundary(world, k),
                        gold_answer=oracle.encode(answer),
                        gold_independent=oracle.encode(independent),
                        gold_program=dict(
                            engine="sr_f4.task_contract.execute",
                            version=oracle.VERSION,
                            query=copy.deepcopy(query),
                        ),
                        independent_gold_program=dict(
                            engine="sr_f4.task_contract.independently_answer",
                            version=oracle.VERSION,
                            query=copy.deepcopy(query),
                        ),
                        definition_status="DEFINED",
                        target_sets=[world["categories"][i] for i in target],
                        target_indices=target,
                        generation_version=VERSION,
                        generation_seed=seed,
                        generation_attempt=0,
                        answer_collision_diagnostic=probe._answer_collisions(
                            world, query, answer, target
                        ),
                    )
                )
    return tasks


def validate_tasks(tasks):
    """Check both new splits against each other and the old probe, without hashes."""
    tasks = _rows(tasks)
    if len(tasks) != TRAIN_QUERIES + TEST_QUERIES or len({t["qid"] for t in tasks}) != len(tasks):
        raise ValueError("Pilot query count or unique qid count differs")
    old_content = {base_key(t["world"]) for t in probe.generate_tasks()}
    worlds, seen_content = defaultdict(list), {}
    for task in tasks:
        if task["split"] not in (TRAIN_SPLIT, TEST_SPLIT) or task["pool"] != task["split"]:
            raise ValueError("Unregistered pilot split")
        if task["family"] != "TOPK" or task["cell"] not in CELLS:
            raise ValueError("Unregistered pilot family or cell")
        oracle.validate_query(task["world"], task["query"])
        if task["query"]["family"] != "TOPK":
            raise ValueError("Non-TOPK query in pilot")
        key = base_key(task["world"])
        if key in old_content:
            raise ValueError("Pilot repeats an old probe world; do not retry")
        if key in seen_content and seen_content[key] != task["world_id"]:
            raise ValueError("Pilot world collision between identities or splits; do not retry")
        seen_content[key] = task["world_id"]
        worlds[task["world_id"]].append(task)
    for split, per_cell in (
        (TRAIN_SPLIT, TRAIN_WORLDS_PER_CELL),
        (TEST_SPLIT, TEST_WORLDS_PER_CELL),
    ):
        for cell in CELLS:
            population = [t for t in tasks if t["split"] == split and t["cell"] == cell]
            first = [t for t in population if t["branch"] == 0]
            if len(population) != 2 * per_cell or len(first) != per_cell:
                raise ValueError("Wrong pilot cell count")
            if {t["world_index"] for t in first} != set(range(per_cell)):
                raise ValueError("Pilot world indices differ")
            for task in population:
                index, query, world = task["world_index"], task["query"], task["world"]
                n = 6 if cell.startswith("T6-") else 8
                gap = 12 if cell.endswith("SEPARATED") else 1 if index < per_cell // 2 else 0
                if len(world["categories"]) != n or query["k"] != n // 2:
                    raise ValueError("Pilot category count or k differs")
                if any(b["gap"] != gap for b in probe._boundary(world, n // 2).values()):
                    raise ValueError("Pilot ranking boundary differs")
                if query["aggregate"] != ("sum" if index % 2 == 0 else "mean") or query[
                    "direction"
                ] != ("largest" if (index // 2) % 2 == 0 else "smallest"):
                    raise ValueError("Pilot aggregate or direction balance differs")
                if task["prompt"] != question(world, query) + "\n" + CONTRACT_V2:
                    raise ValueError("Pilot model prompt differs from native protocol")
    for pair in worlds.values():
        if len(pair) != 2 or {t["branch"] for t in pair} != {0, 1}:
            raise ValueError("Pilot world must have exactly two query branches")
        first, second = sorted(pair, key=lambda t: t["branch"])
        if first["world"] != second["world"] or first["split"] != second["split"]:
            raise ValueError("Paired pilot worlds or splits differ")
        if first["query"]["selection_series"] != "Alpha" or first["query"]["value"] != "Beta":
            raise ValueError("Pilot first branch differs")
        if second["query"]["selection_series"] != "Beta" or second["query"]["value"] != "Alpha":
            raise ValueError("Pilot second branch differs")
    return {
        TRAIN_SPLIT: {"worlds": TRAIN_WORLDS, "queries": TRAIN_QUERIES},
        TEST_SPLIT: {"worlds": TEST_WORLDS, "queries": TEST_QUERIES},
    }


def generate_tasks():
    """Return 128 train + 256 held-out queries; fail any raw-world collision."""
    tasks = [
        *_make_tasks(TRAIN_SPLIT, TRAIN_WORLDS_PER_CELL, TRAIN_SEED),
        *_make_tasks(TEST_SPLIT, TEST_WORLDS_PER_CELL, TEST_SEED),
    ]
    validate_tasks(tasks)
    return tasks


def _inventory(tasks, split, count):
    selected = [t for t in _rows(tasks) if t.get("split", t.get("pool")) == split]
    if len(selected) != count or len({t["qid"] for t in selected}) != count:
        raise ValueError("Wrong pilot assignment inventory: " + split)
    if Counter(t["cell"] for t in selected) != dict.fromkeys(CELLS, count // len(CELLS)):
        raise ValueError("Unbalanced pilot assignment inventory")
    return selected


def _block(block):
    if type(block) is not int or block not in BLOCKS:
        raise ValueError("Unregistered pilot block")


def training_assignments(tasks, block, step):
    """Sixteen G8 groups: four per cell, each query used once per 8-step cycle.

    Every arm uses this exact schedule and the same explicit per-draw seeds.
    Different arms may still produce different on-policy responses.
    """
    _block(block)
    if type(step) is not int or not 1 <= step <= STEPS:
        raise ValueError("Pilot step must be in 1..16")
    inventory = _inventory(tasks, TRAIN_SPLIT, TRAIN_QUERIES)
    cycle, within_cycle = divmod(step - 1, 8)
    chosen = []
    for cell in CELLS:
        candidates = sorted((t for t in inventory if t["cell"] == cell), key=lambda t: t["qid"])
        random.Random(seed_for(MASTER, "schedule", block, cycle, cell)).shuffle(candidates)
        chosen.extend(candidates[4 * within_cycle : 4 * (within_cycle + 1)])
    random.Random(seed_for(MASTER, "group_order", block, step)).shuffle(chosen)
    return [
        dict(
            qid=task["qid"],
            world_id=task["world_id"],
            cell=task["cell"],
            family="TOPK",
            source=TRAIN_SPLIT,
            panel=TRAIN_SPLIT,
            protocol="structured",
            group_index=i,
            G=G,
            seeds=[
                seed_for(MASTER, "train", block, step, i, task["qid"], draw) for draw in range(G)
            ],
        )
        for i, task in enumerate(chosen)
    ]


def eval_assignments(tasks, block):
    """All 256 held-out queries at G4, shared by the parent and four endpoints."""
    _block(block)
    inventory = sorted(_inventory(tasks, TEST_SPLIT, TEST_QUERIES), key=lambda t: t["qid"])
    return [
        dict(
            qid=task["qid"],
            world_id=task["world_id"],
            cell=task["cell"],
            family="TOPK",
            source=TEST_SPLIT,
            panel=TEST_SPLIT,
            protocol="structured",
            group_index=i,
            G=EVAL_G,
            seeds=[seed_for(MASTER, "eval", block, i, task["qid"], draw) for draw in range(EVAL_G)],
        )
        for i, task in enumerate(inventory)
    ]


def select_retention(tasks):
    """Select two root/chart pairs per family x variant = exactly 128 TEST_ID tasks.

    The original TEST_ID has 32 roots per stratum and one grouped-bar/line pair
    per root. Sample sorted root IDs, preserve both charts, and retain all original
    fields. Malformed/unpaired strata fail rather than silently changing policy.
    """
    candidates = [t for t in _rows(tasks) if t.get("pool") == "TEST_ID"]
    if len({t["qid"] for t in candidates}) != len(candidates):
        raise ValueError("Duplicate TEST_ID qid")
    strata = defaultdict(lambda: defaultdict(list))
    for task in candidates:
        if (
            task["family"] not in FAMILIES
            or type(task["variant"]) is not int
            or task["variant"] not in range(8)
        ):
            raise ValueError("Unregistered TEST_ID family or variant")
        strata[(task["family"], task["variant"])][task["root_id"]].append(task)
    rng = random.Random(RETENTION_SEED)
    result = []
    for family in FAMILIES:
        for variant in range(8):
            roots = strata[(family, variant)]
            if len(roots) < 2:
                raise ValueError("Need two TEST_ID root/chart pairs per family and variant")
            for pair in roots.values():
                if len(pair) != 2 or {t["chart"] for t in pair} != {"grouped_bar", "line"}:
                    raise ValueError("TEST_ID root lacks its grouped_bar/line pair")
                if pair[0]["world"] != pair[1]["world"] or pair[0]["query"] != pair[1]["query"]:
                    raise ValueError("TEST_ID chart pair differs in world or query")
            for root in sorted(rng.sample(sorted(roots), 2)):
                result.extend(copy.deepcopy(sorted(roots[root], key=lambda t: t["chart"])))
    return result


def select_chartqa(tasks):
    """Select 250 sorted-ID sampled ChartQA tasks without inspecting scores."""
    candidates = sorted(
        (t for t in _rows(tasks) if t.get("pool") == "CHARTQA"), key=lambda t: t["qid"]
    )
    if len(candidates) < CHARTQA_QUERIES or len({t["qid"] for t in candidates}) != len(candidates):
        raise ValueError("Need at least 250 unique ChartQA qids")
    chosen = random.Random(CHARTQA_SEED).sample(candidates, CHARTQA_QUERIES)
    return copy.deepcopy(sorted(chosen, key=lambda t: t["qid"]))


def plan():
    """Fresh JSON-serializable contract, including generation counts (not tokens)."""
    budgets = {
        "train": len(BLOCKS) * len(ARMS) * STEPS * GROUPS_PER_STEP * G,
        "new_topk": len(BLOCKS) * (1 + len(ARMS)) * TEST_QUERIES * EVAL_G,
        "retention": len(BLOCKS) * (1 + len(ARMS)) * RETENTION_QUERIES * EVAL_G,
        "chartqa": (1 + len(ARMS)) * CHARTQA_QUERIES,
    }
    budgets["total"] = sum(budgets.values())
    return dict(
        version=VERSION,
        master_seed=MASTER,
        arms=list(ARMS),
        blocks=list(BLOCKS),
        steps=STEPS,
        groups_per_step=GROUPS_PER_STEP,
        draws_per_group=G,
        measurement_version=MEASUREMENT_VERSION,
        train_split=TRAIN_SPLIT,
        test_split=TEST_SPLIT,
        cells=list(CELLS),
        train_worlds=TRAIN_WORLDS,
        train_queries=TRAIN_QUERIES,
        test_worlds=TEST_WORLDS,
        test_queries=TEST_QUERIES,
        evaluation_draws=EVAL_G,
        generation_seeds={TRAIN_SPLIT: TRAIN_SEED, TEST_SPLIT: TEST_SEED},
        world_seed_rule="seed_for(split_seed, 'world', split, cell, world_index, series)",
        training_seed_rule="seed_for(MASTER, 'train', block, step, group_index, qid, draw)",
        evaluation_seed_rule="seed_for(MASTER, 'eval', block, group_index, qid, draw)",
        schedule=(
            "Two 8-step independently shuffled cycles; every query twice; "
            "four groups per cell per step; all arms shared"
        ),
        schedule_seed_rule="seed_for(MASTER, 'schedule', block, cycle_0_or_1, cell)",
        group_order_seed_rule="seed_for(MASTER, 'group_order', block, step_1_to_16)",
        world_disjointness=(
            "Exact in-memory fact-table equality against old probe and between all new worlds; "
            "collision raises without retry"
        ),
        construction=(
            "Same c_probe_data bounded column construction; "
            "gaps 12/1/0 in both columns; preserve answer collisions"
        ),
        retention=dict(
            seed=RETENTION_SEED,
            queries=RETENTION_QUERIES,
            policy=(
                "Two sampled sorted root IDs per family x variant, "
                "both grouped_bar and line retained; 4 tasks per stratum"
            ),
            family_order=list(FAMILIES),
            variant_order=list(range(8)),
            draws=EVAL_G,
        ),
        chartqa=dict(
            seed=CHARTQA_SEED,
            queries=CHARTQA_QUERIES,
            policy="Sample sorted qids once; b0 parent and four b0 endpoints only",
            draws=1,
        ),
        eval_models_per_block=["PARENT", *ARMS],
        budgets=budgets,
        filtering="NO_PERFORMANCE_FILTER_NO_COLLISION_FILTER_NO_SEED_RETRY",
        primary_policy="F_PRIME_KNOWN_75",
        known_subset_min_fraction="3/4",
        beta=0.25,
        scope="SMALL_C_FPRIME_TRAINING_PILOT_NOT_CONFIRMATION",
    )
