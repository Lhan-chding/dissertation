"""Frozen signal-rich question lineages for the local-credit C pilot.

Construct before sampling, without performance filtering, rejection retries,
hash scans, model calls, or mutation of caller-owned data. Four related questions
remain four independent G8 groups. All descendants of a root share one split.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping

from sr_f3_ce.prompts import CONTRACT_V2
from srf4_ref.sampling import seed_for

from . import c_pilot_data as previous
from . import c_probe_data as probe
from . import task_contract as oracle
from .data import base_key, question

VERSION = "srf4-c-local-signal-rich-data-r1"
MASTER = 2026102401
TRAIN_SEED = 2026102402
TEST_SEED = 2026102403
ARMS = ("J", "EVID", "SELECT", "SHUF")
BLOCKS = (0, 1, 2)
STEPS = 12
GROUPS_PER_STEP = 16
GROUPS = GROUPS_PER_STEP
G = 8
EVAL_G = 4
CELLS = ("BINDING", "IMPACT", "TIES", "SIGNED", "INTERVAL", "K_DIRECTION")
TRAIN_SPLIT = "C_LOCAL_TRAIN_R1"
TEST_SPLIT = "C_LOCAL_TEST_R1"
TRAIN_ROOTS_PER_CELL = 8
TEST_ROOTS_PER_CELL = 16
TRAIN_ROOTS = 48
TEST_ROOTS = 96
TRAIN_WORLDS = 72
TEST_WORLDS = 144
TRAIN_QUERIES = 192
TEST_QUERIES = 384
RETENTION_QUERIES = previous.RETENTION_QUERIES
CHARTQA_QUERIES = previous.CHARTQA_QUERIES
RETENTION_SEED = previous.RETENTION_SEED
CHARTQA_SEED = previous.CHARTQA_SEED
MEASUREMENT_VERSION = "srf4-c-local-explicit-v1"
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August")


def _rows(tasks):
    return list(tasks.values() if isinstance(tasks, Mapping) else tasks)


def _world(alpha, beta, categories):
    return dict(
        categories=list(categories),
        series=dict(Alpha=list(alpha), Beta=list(beta)),
        domain=[10, 90],
    )


def _topk(ranking="Alpha", value="Beta", k=3, direction="largest", aggregate="mean"):
    return dict(
        family="TOPK",
        selection_series=ranking,
        value=value,
        k=k,
        direction=direction,
        aggregate=aggregate,
    )


def validate_query(world, query):
    """Fail closed: unsupported extra fields must never be silently ignored."""
    common = {"family", "value", "aggregate"}
    allowed = {
        "TOPK": common | {"selection_series", "k", "direction"},
        "INTERVAL": common | {"start", "end", "include_start", "include_end", "positive_only"},
    }
    if query.get("family") not in allowed or set(query) != allowed[query["family"]]:
        raise ValueError("Local pilot query must use exactly the supported schema")
    oracle.validate_query(world, query)


def model_prompt(world, query):
    # Lazy import permits preparation and credit implementation to be developed
    # independently. The runtime's question parser expects one question line.
    from .c_local_credit import PROTOCOL_INSTRUCTION

    if not PROTOCOL_INSTRUCTION or "\n" in PROTOCOL_INSTRUCTION or "\r" in PROTOCOL_INSTRUCTION:
        raise ValueError("Local credit protocol instruction must be a nonempty single line")
    return question(world, query) + " " + PROTOCOL_INSTRUCTION + "\n" + CONTRACT_V2


def _rank_column(rng, order, *, direction="largest", n=6):
    levels = [86, 73, 60, 47, 34, 21] if n == 6 else [87, 77, 67, 57, 47, 37, 27, 17]
    values = [v + rng.randint(-2, 2) for v in levels]
    if direction == "smallest":
        values.reverse()
    result = [None] * n
    for i, value in zip(order, values, strict=True):
        result[i] = value
    return result


def _construct(cell, index, rng):
    """Return four (world, query, label, factors) tuples without rejection."""
    n = 8 if cell in {"INTERVAL", "K_DIRECTION"} else 6
    categories = list(MONTHS[:n])
    rng.shuffle(categories)
    order = list(range(n))
    rng.shuffle(order)
    direction = "largest" if (index // 4) % 2 == 0 else "smallest"
    aggregate = "mean" if (index // 2) % 2 == 0 else "sum"
    if cell == "BINDING":
        # Deliberately set competition, rather than filter independently random
        # columns after observing whether their top-k sets happen to differ.
        overlap = (3, 2, 1, 0)[index % 4]
        beta_selected = order[:overlap] + order[3 : 6 - overlap]
        beta_rest = [i for i in order if i not in beta_selected]
        rng.shuffle(beta_selected)
        rng.shuffle(beta_rest)
        world = _world(
            _rank_column(rng, order, direction=direction),
            _rank_column(rng, beta_selected + beta_rest, direction=direction),
            categories,
        )
        return [
            (
                world,
                _topk(ranking, value, direction=direction, aggregate=aggregate),
                f"{ranking}_to_{value}",
                dict(planned_rank_set_overlap=overlap),
            )
            for ranking in ("Alpha", "Beta")
            for value in ("Alpha", "Beta")
        ]
    if cell == "IMPACT":
        # Boundary 60/48 is exactly 12. Other Alpha positions vary by root.
        ranked = [
            rng.randint(84, 90),
            rng.randint(70, 77),
            60,
            48,
            rng.randint(25, 32),
            rng.randint(10, 18),
        ]
        if direction == "smallest":
            ranked = [100 - v for v in ranked]
        alpha = [0] * n
        for i, value in zip(order, ranked, strict=True):
            alpha[i] = value
        beta = [rng.randint(15, 85) for _ in range(n)]
        low = rng.randint(12, 23)
        result = []
        for delta in (0, 1, 20, 65):
            payload = list(beta)
            payload[order[2]], payload[order[3]] = low + delta, low
            result.append(
                (
                    _world(alpha, payload, categories),
                    _topk(direction=direction, aggregate=aggregate),
                    f"impact_{delta}",
                    dict(
                        boundary_gap=12,
                        boundary_payload_delta=delta,
                        swapped_indices=[order[2], order[3]],
                        confounding_note=(
                            "Payload changes can also change Beta rank, readability, "
                            "duplicates and integer-answer status"
                        ),
                    ),
                )
            )
        return result
    if cell == "TIES":
        high, tie = rng.randint(81, 90), rng.randint(50, 64)
        ranked = [high, tie, tie, tie, rng.randint(26, 35), rng.randint(10, 19)]
        if direction == "smallest":
            ranked = [100 - v for v in ranked]
        alpha = [0] * n
        for i, value in zip(order, ranked, strict=True):
            alpha[i] = value
        # Rotate the tied labels deterministically if their printed ordering
        # equals calendar order; this is construction, not a response filter.
        tie_positions = sorted(order[1:4])
        tie_labels = sorted((categories[i] for i in tie_positions), key=MONTHS.index)
        for i, label in zip(tie_positions, tie_labels[1:] + tie_labels[:1], strict=True):
            categories[i] = label
        world = _world(alpha, rng.sample(range(10, 91), n), categories)
        opposite = "smallest" if direction == "largest" else "largest"
        return [
            (world, query, label, dict(tie_block_size=3, printed_tie_positions=tie_positions))
            for label, query in (
                ("main_k3_mean", _topk(direction=direction)),
                ("main_k3_sum", _topk(direction=direction, aggregate="sum")),
                ("main_k2_mean", _topk(k=2, direction=direction)),
                ("opposite_k3_mean", _topk(direction=opposite)),
            )
        ]
    if cell == "SIGNED":
        levels = [v + rng.randint(-2, 2) for v in (78, 67, 56, 45, 34, 23)]
        if direction == "smallest":
            levels.reverse()
        alpha = [0] * n
        for i, value in zip(order, levels, strict=True):
            alpha[i] = value
        beta = [rng.randint(10, 90) for _ in range(n)]
        signs = ((1, 1, 1), (1, -1, 1), (-1, -1, -1), (-1, 1, -1))[index % 4]
        for i, sign in zip(order[:3], signs, strict=True):
            # 3..9 remains inside the world domain for all rank-column values.
            beta[i] = alpha[i] - sign * rng.randint(3, 9)
        world = _world(alpha, beta, categories)
        return [
            (world, query, label, dict(planned_selected_signs=list(signs)))
            for label, query in (
                ("alpha_diff_mean", _topk(value="diff", direction=direction)),
                ("alpha_diff_sum", _topk(value="diff", direction=direction, aggregate="sum")),
                ("alpha_beta_mean", _topk(direction=direction)),
                ("beta_diff_mean", _topk(ranking="Beta", value="diff", direction=direction)),
            )
        ]
    if cell == "INTERVAL":
        beta = [rng.randint(30, 60) for _ in range(n)]
        differences = [
            rng.randint(-20, 25),
            rng.randint(12, 25),
            rng.randint(12, 25),
            0,
            rng.randint(12, 25),
            -rng.randint(12, 20),
            rng.randint(12, 25),
            -rng.randint(12, 20),
        ]
        world = _world([a + d for a, d in zip(beta, differences, strict=True)], beta, categories)
        return [
            (
                world,
                dict(
                    family="INTERVAL",
                    start=1,
                    end=6,
                    include_start=inc_start,
                    include_end=inc_end,
                    positive_only=positive,
                    value="diff",
                    aggregate=aggregate,
                ),
                label,
                dict(interval_positions=[1, 6], equality_position=3, negative_position=5),
            )
            for label, inc_start, inc_end, positive in (
                ("main", False, True, True),
                ("include_start", True, True, True),
                ("exclude_end", False, False, True),
                ("no_filter", False, True, False),
            )
        ]
    if cell == "K_DIRECTION":
        ranking = "Alpha" if index % 2 == 0 else "Beta"
        other = "Beta" if ranking == "Alpha" else "Alpha"
        columns = {ranking: _rank_column(rng, order, n=8), other: rng.sample(range(10, 91), n)}
        world = _world(columns["Alpha"], columns["Beta"], categories)
        return [
            (
                world,
                _topk(ranking, other, k, selected_direction, aggregate),
                f"{selected_direction}_{k}",
                dict(category_count=8, selected_count=k),
            )
            for k in (2, 3)
            for selected_direction in ("largest", "smallest")
        ]
    raise ValueError("Unregistered local task cell")


def _diagnostic(world, query):
    chosen = oracle.selected(world, query)
    values = [oracle.operand(world, query, i) for i in chosen]
    result = dict(
        selected_count=len(chosen),
        selected_values=[oracle.encode(v) for v in values],
        negative_count=sum(v < 0 for v in values),
        zero_count=sum(v == 0 for v in values),
        positive_count=sum(v > 0 for v in values),
    )
    if query["family"] == "TOPK":
        other = copy.deepcopy(query)
        other["selection_series"] = "Beta" if query["selection_series"] == "Alpha" else "Alpha"
        other_set = oracle.selected(world, other)
        result.update(
            wrong_column_set_overlap=len(set(chosen) & set(other_set)),
            wrong_column_same_answer=oracle.execute(world, other) == oracle.execute(world, query),
        )
    return result


def _make_tasks(split, count, seed):
    result = []
    for cell in CELLS:
        for index in range(count):
            root = f"SRF4_{split}_{cell}_{index:03d}"
            rng = random.Random(seed_for(seed, "world", split, cell, index))
            for branch, (world, query, label, factors) in enumerate(_construct(cell, index, rng)):
                validate_query(world, query)
                answer = oracle.execute(world, query)
                independent = oracle.independently_answer(world, query)
                if answer is None or answer != independent:
                    raise ValueError(
                        "Local question must be defined with agreeing independent gold"
                    )
                chosen = oracle.selected(world, query)
                world_id = f"{root}_W{branch if cell == 'IMPACT' else 0}"
                result.append(
                    dict(
                        qid=f"{root}_Q{branch}",
                        root_id=root,
                        pair_id=root,
                        world_id=world_id,
                        split=split,
                        pool=split,
                        family=query["family"],
                        cell=cell,
                        world_index=index,
                        branch=branch,
                        variant=branch,
                        label=label,
                        purpose="c_signal_rich_local_credit_pilot",
                        world=copy.deepcopy(world),
                        query=copy.deepcopy(query),
                        question=question(world, query),
                        prompt=model_prompt(world, query),
                        chart="grouped_bar",
                        render_style="main",
                        wording_template="standard_with_local_declarations_r1",
                        design_factors=copy.deepcopy(factors),
                        design_diagnostic=_diagnostic(world, query),
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
                        target_indices=chosen,
                        target_sets=[world["categories"][i] for i in chosen],
                        generation_version=VERSION,
                        generation_seed=seed,
                        generation_attempt=0,
                    )
                )
    return result


def _prototype_worlds():
    """Exact public design tables, not additional input scans or test examples."""
    m6 = MONTHS[:6]
    result = [_world([84, 68, 55, 43, 29, 16], [19, 47, 68, 83, 72, 54], m6)]
    for beta in (
        [29, 46, 55, 55, 69, 54],
        [29, 46, 42, 41, 69, 54],
        [29, 46, 62, 42, 69, 54],
        [29, 46, 82, 17, 69, 54],
    ):
        result.append(_world([88, 73, 60, 48, 27, 12], beta, m6))
    result.extend(
        [
            _world(
                [80, 58, 58, 58, 22, 11],
                [31, 47, 68, 83, 72, 54],
                ["May", "June", "April", "January", "February", "March"],
            ),
            _world([87, 74, 61, 49, 31, 17], [89, 47, 82, 16, 68, 54], m6),
            _world([70, 67, 58, 41, 66, 28, 81, 20], [20, 26, 22, 41, 37, 50, 30, 65], MONTHS),
            _world([42, 86, 57, 21, 74, 35, 63, 12], [66, 24, 78, 43, 51, 87, 19, 60], MONTHS),
        ]
    )
    return result


def validate_tasks(tasks):
    rows = _rows(tasks)
    if len(rows) != TRAIN_QUERIES + TEST_QUERIES or len({t["qid"] for t in rows}) != len(rows):
        raise ValueError("Local pilot count or unique qid count differs")
    old = (
        {base_key(t["world"]) for t in previous.generate_tasks()}
        | {base_key(t["world"]) for t in probe.generate_tasks()}
        | {base_key(w) for w in _prototype_worlds()}
    )
    contents, worlds, roots = {}, defaultdict(list), defaultdict(list)
    for task in rows:
        if task["split"] not in (TRAIN_SPLIT, TEST_SPLIT) or task["pool"] != task["split"]:
            raise ValueError("Unregistered local split")
        if task["cell"] not in CELLS or task["family"] != task["query"]["family"]:
            raise ValueError("Unregistered local cell or native family mismatch")
        validate_query(task["world"], task["query"])
        key = base_key(task["world"])
        if key in old:
            raise ValueError(
                "Local world repeats old probe, old pilot or public prototype; do not retry"
            )
        if key in contents and contents[key] != task["world_id"]:
            raise ValueError("Local world collision between identities or splits; do not retry")
        contents[key] = task["world_id"]
        worlds[task["world_id"]].append(task)
        roots[task["root_id"]].append(task)
        value = oracle.execute(task["world"], task["query"])
        if (
            value is None
            or value != oracle.independently_answer(task["world"], task["query"])
            or task["gold_answer"] != oracle.encode(value)
            or task["gold_independent"] != oracle.encode(value)
        ):
            raise ValueError("Local gold identity differs")
        if task["target_indices"] != oracle.selected(task["world"], task["query"]):
            raise ValueError("Local selected indices differ")
        if task["prompt"] != model_prompt(task["world"], task["query"]):
            raise ValueError("Local prompt differs from shared frozen protocol")
    for group in worlds.values():
        if len({base_key(t["world"]) for t in group}) != 1:
            raise ValueError("Same world identity has different facts")
    for group in roots.values():
        if len(group) != 4 or {t["branch"] for t in group} != set(range(4)):
            raise ValueError("Local root must have exactly four distinct branches")
        if len({(t["split"], t["cell"], t["world_index"]) for t in group}) != 1:
            raise ValueError("Local root crosses split or cell")
        expected = 4 if group[0]["cell"] == "IMPACT" else 1
        if len({t["world_id"] for t in group}) != expected:
            raise ValueError("Unexpected physical world count within local root")
        if len({json.dumps(t["query"], sort_keys=True) for t in group}) != (
            1 if expected == 4 else 4
        ):
            raise ValueError("Unexpected query variety within local root")
    result = {}
    for split, per_cell, expected_worlds in (
        (TRAIN_SPLIT, TRAIN_ROOTS_PER_CELL, TRAIN_WORLDS),
        (TEST_SPLIT, TEST_ROOTS_PER_CELL, TEST_WORLDS),
    ):
        subset = [t for t in rows if t["split"] == split]
        if Counter(t["cell"] for t in subset) != dict.fromkeys(CELLS, per_cell * 4):
            raise ValueError("Local split family balance differs")
        for cell in CELLS:
            if {t["world_index"] for t in subset if t["cell"] == cell} != set(range(per_cell)):
                raise ValueError("Local root indices differ")
        if len({t["world_id"] for t in subset}) != expected_worlds:
            raise ValueError("Local physical world count differs")
        result[split] = dict(
            roots=per_cell * len(CELLS), worlds=expected_worlds, queries=len(subset)
        )
    return result


def generate_tasks():
    tasks = [
        *_make_tasks(TRAIN_SPLIT, TRAIN_ROOTS_PER_CELL, TRAIN_SEED),
        *_make_tasks(TEST_SPLIT, TEST_ROOTS_PER_CELL, TEST_SEED),
    ]
    validate_tasks(tasks)
    return tasks


def _inventory(tasks, split, count):
    rows = [t for t in _rows(tasks) if t.get("split", t.get("pool")) == split]
    if (
        len(rows) != count
        or len({t["qid"] for t in rows}) != count
        or Counter(t["cell"] for t in rows) != dict.fromkeys(CELLS, count // len(CELLS))
    ):
        raise ValueError("Wrong local assignment inventory")
    return sorted(rows, key=lambda t: t["qid"])


def _block(block):
    if type(block) is not int or block not in BLOCKS:
        raise ValueError("Unregistered local block")


def _assignment(task, block, i, *, step=None):
    training = step is not None
    prefix = (MASTER, "train", block, step, i) if training else (MASTER, "eval", block, i)
    draws = G if training else EVAL_G
    return dict(
        qid=task["qid"],
        root_id=task["root_id"],
        world_id=task["world_id"],
        cell=task["cell"],
        family=task["family"],
        source=task["split"],
        panel=task["split"],
        protocol="structured",
        group_index=i,
        G=draws,
        seeds=[seed_for(*prefix, task["qid"], draw) for draw in range(draws)],
    )


def training_assignments(tasks, block, step):
    _block(block)
    if type(step) is not int or not 1 <= step <= STEPS:
        raise ValueError("Local pilot step must be in 1..12")
    inventory = _inventory(tasks, TRAIN_SPLIT, TRAIN_QUERIES)
    random.Random(seed_for(MASTER, "schedule", block)).shuffle(inventory)
    chosen = inventory[(step - 1) * GROUPS_PER_STEP : step * GROUPS_PER_STEP]
    return [_assignment(task, block, i, step=step) for i, task in enumerate(chosen)]


def eval_assignments(tasks, block):
    _block(block)
    return [
        _assignment(task, block, i)
        for i, task in enumerate(_inventory(tasks, TEST_SPLIT, TEST_QUERIES))
    ]


def select_retention(tasks):
    return previous.select_retention(tasks)


def select_chartqa(tasks):
    return previous.select_chartqa(tasks)


def plan():
    budgets = dict(
        train=18432, new_tasks=23040, retention=7680, chartqa=1250, evaluation=31970, total=50402
    )
    return dict(
        version=VERSION,
        master_seed=MASTER,
        arms=list(ARMS),
        blocks=list(BLOCKS),
        steps=STEPS,
        groups_per_step=GROUPS_PER_STEP,
        draws_per_group=G,
        evaluation_draws=EVAL_G,
        measurement_version=MEASUREMENT_VERSION,
        train_split=TRAIN_SPLIT,
        test_split=TEST_SPLIT,
        cells=list(CELLS),
        train_roots=TRAIN_ROOTS,
        train_worlds=TRAIN_WORLDS,
        train_queries=TRAIN_QUERIES,
        test_roots=TEST_ROOTS,
        test_worlds=TEST_WORLDS,
        test_queries=TEST_QUERIES,
        generation_seeds={TRAIN_SPLIT: TRAIN_SEED, TEST_SPLIT: TEST_SEED},
        world_seed_rule="seed_for(split_seed, 'world', split, cell, root_index)",
        training_seed_rule="seed_for(MASTER, 'train', block, step, group_index, qid, draw)",
        evaluation_seed_rule="seed_for(MASTER, 'eval', block, group_index, qid, draw)",
        schedule=(
            "One shuffled full 192-query inventory per block, twelve consecutive batches "
            "of sixteen; each query once; all arms shared; report batch family imbalance"
        ),
        schedule_seed_rule="seed_for(MASTER, 'schedule', block)",
        world_disjointness=(
            "Exact in-memory fact-table comparison with old probe, old F-prime pilot, nine public "
            "prototype worlds, and all new physical worlds; collision fails without retry"
        ),
        eval_models_per_block=["PARENT", *ARMS],
        budgets=budgets,
        retention=copy.deepcopy(previous.plan()["retention"]),
        chartqa=copy.deepcopy(previous.plan()["chartqa"]),
        construction=dict(
            BINDING=(
                "Six categories; planned Alpha/Beta selected-set overlap cycles 3/2/1/0 "
                "crossed with direction; full ranking-column x payload-column 2x2"
            ),
            IMPACT=(
                "Six categories; ranking boundary gap 12; four payload deltas 0/1/20/65; "
                "four worlds share root; preserve zero-impact collision"
            ),
            TIES=(
                "Six categories; one extreme, three-way tie, two opposite extremes; "
                "printed tie labels noncalendar; k3 mean/sum, k2 mean, opposite k3 mean"
            ),
            SIGNED=(
                "Six categories; selected difference signs cycle all positive/mixed/"
                "all negative/mixed crossed with direction; Alpha-selected difference "
                "mean/sum, Beta-value mean, Beta-selected difference mean"
            ),
            INTERVAL=(
                "Eight categories; positions1..6, positive endpoint and internal values, "
                "equality at3, negative at5; endpoint inclusion and positive filter variants"
            ),
            K_DIRECTION=(
                "Eight categories; k2/3 x largest/smallest, cross-column value; "
                "ranking column balanced"
            ),
        ),
        filtering="NO_PERFORMANCE_FILTER_NO_COLLISION_FILTER_NO_SEED_RETRY",
        scope="SMALL_SIGNAL_RICH_LOCAL_CREDIT_C_PILOT_NOT_CONFIRMATION",
    )
