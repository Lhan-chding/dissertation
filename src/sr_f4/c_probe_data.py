"""Fixed TOPK opportunity probe, prepared without model calls or data scans.

Construction fixes both ranking-column boundaries before any model answers.
No draw/retry loop, answer-collision rejection, or parent-performance filter is
used. Gold, boundary checks, and alternative-set collisions are audit sidecars;
the model input remains the native question plus the unchanged V2 contract.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from sr_f3_ce.prompts import CONTRACT_V2

from . import task_contract as oracle
from .data import base_key, question, write_jsonl

VERSION = "srf4-topk-opportunity-r1"
SPLIT = "C_TOPK_DEV_R1"
GENERATION_SEED = 2026102305
MEASUREMENT_VERSION = "srf4-observable-v3-programmatic-dev-r5"
CELLS = ("T6-SEPARATED", "T6-BOUNDARY", "T8-SEPARATED", "T8-BOUNDARY")
BLOCKS = ("b0", "b1", "b2")
WORLDS_PER_CELL = 8
QUERIES_PER_WORLD = 2
DRAWS_PER_GROUP = 8
WORLD_COUNT = len(CELLS) * WORLDS_PER_CELL
QUERY_COUNT = WORLD_COUNT * QUERIES_PER_WORLD
GROUP_COUNT = QUERY_COUNT * len(BLOCKS)
RESPONSE_COUNT = GROUP_COUNT * DRAWS_PER_GROUP
CATEGORIES = ("January", "February", "March", "April", "May", "June", "July", "August")
MODEL_INPUT_FIELDS = ("qid", "pool", "chart", "prompt", "render_style")


def _categories(n, index):
    """Noncanonical printed order; ties always use this actual order."""
    categories = list(CATEGORIES[:n])
    offset = 1 + index % (n - 1)
    categories = categories[offset:] + categories[:offset]
    return categories[::-1] if index // (n - 1) % 2 else categories


def _column(rng, k, gap):
    center = rng.randint(44, 56)
    lower = center - 6 if gap == 12 else center
    upper = lower + gap
    values = [
        *rng.sample(range(10, lower - 3), k - 1),
        lower,
        upper,
        *rng.sample(range(upper + 4, 91), k - 1),
    ]
    rng.shuffle(values)
    return values


def _boundary(world, k):
    result = {}
    for name, values in world["series"].items():
        ordered = sorted(values)
        result[name] = {
            "lower": ordered[k - 1],
            "upper": ordered[k],
            "gap": ordered[k] - ordered[k - 1],
        }
    return result


def _answer_collisions(world, query, answer, target):
    """Enumerate wrong k-sets with the same answer; never reject such worlds."""
    matches = []
    values = world["series"][query["value"]]
    target = frozenset(target)
    candidates = 0
    for indices in combinations(range(len(world["categories"])), query["k"]):
        if frozenset(indices) == target:
            continue
        candidates += 1
        candidate = sum((Fraction(values[i]) for i in indices), Fraction())
        if query["aggregate"] == "mean":
            candidate /= query["k"]
        if candidate == answer:
            matches.append([world["categories"][i] for i in indices])
    return {
        "policy": "RECORD_ONLY_NO_FILTER",
        "alternative_k_sets": candidates,
        "wrong_set_same_answer_count": len(matches),
        "wrong_set_same_answer_sets": matches,
    }


def generate_tasks():
    """Return exactly 32 new worlds and their 64 fixed paired questions."""
    tasks = []
    for cell_index, cell in enumerate(CELLS):
        n = 6 if cell.startswith("T6-") else 8
        k = n // 2
        separated = cell.endswith("SEPARATED")
        for index in range(WORLDS_PER_CELL):
            gap = 12 if separated else 1 if index < 4 else 0
            subtype = "separated" if separated else "near" if gap else "tied"
            # A distinct integer stream per world/column, with no search or retry.
            world = {
                "categories": _categories(n, index),
                "series": {
                    name: _column(
                        random.Random(GENERATION_SEED + 1000 * cell_index + 10 * index + si),
                        k,
                        gap,
                    )
                    for si, name in enumerate(("Alpha", "Beta"))
                },
                "domain": [10, 90],
            }
            world_id = f"SRF4_{SPLIT}_{cell.replace('-', '_')}_{index:03d}"
            for branch, (ranking, value) in enumerate((("Alpha", "Beta"), ("Beta", "Alpha"))):
                query = {
                    "family": "TOPK",
                    "k": k,
                    "direction": "largest" if (index // 2) % 2 == 0 else "smallest",
                    "selection_series": ranking,
                    "value": value,
                    "aggregate": "sum" if index % 2 == 0 else "mean",
                }
                answer = oracle.execute(world, query)
                independent = oracle.independently_answer(world, query)
                if answer != independent:
                    raise AssertionError("Independent gold mismatch")
                target = oracle.selected(world, query)
                text = question(world, query)
                tasks.append(
                    {
                        "qid": f"{world_id}_{branch}",
                        "root_id": world_id,
                        "world_id": world_id,
                        "split": SPLIT,
                        "pool": SPLIT,
                        "family": "TOPK",
                        "cell": cell,
                        "world_index": index,
                        "branch": branch,
                        "variant": index,
                        "purpose": "topk_opportunity_probe",
                        "boundary_regime": "separated" if separated else "near_or_tied",
                        "boundary_subtype": subtype,
                        "boundary_by_series": _boundary(world, k),
                        "world": copy.deepcopy(world),
                        "query": query,
                        "question": text,
                        "prompt": text + "\n" + CONTRACT_V2,
                        "chart": "grouped_bar",
                        "render_style": "main",
                        "wording_template": "standard",
                        "gold_answer": oracle.encode(answer),
                        "gold_independent": oracle.encode(independent),
                        "gold_program": {
                            "engine": "sr_f4.task_contract.execute",
                            "version": oracle.VERSION,
                            "query": copy.deepcopy(query),
                        },
                        "independent_gold_program": {
                            "engine": "sr_f4.task_contract.independently_answer",
                            "version": oracle.VERSION,
                            "query": copy.deepcopy(query),
                        },
                        "definition_status": "DEFINED",
                        "target_sets": [world["categories"][i] for i in target],
                        "target_indices": target,
                        "pair_id": world_id,
                        "changed_fields": ["query.selection_series", "query.value"],
                        "invariant_fields": [
                            "world",
                            "chart",
                            "render_style",
                            "query.aggregate",
                            "query.direction",
                            "query.k",
                        ],
                        "generation_attempt": 0,
                        "generation_version": VERSION,
                        "answer_collision_diagnostic": _answer_collisions(
                            world, query, answer, target
                        ),
                    }
                )
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks):
    """Fail on a construction defect without drawing replacement worlds."""
    if len(tasks) != QUERY_COUNT or len({t["qid"] for t in tasks}) != QUERY_COUNT:
        raise ValueError("Wrong task count or duplicate qid")
    worlds = {}
    for task in tasks:
        if task["cell"] not in CELLS or task["split"] != SPLIT or task["pool"] != SPLIT:
            raise ValueError("Unexpected probe identity")
        index, branch = task["world_index"], task["branch"]
        if type(index) is not int or not 0 <= index < WORLDS_PER_CELL or branch not in (0, 1):
            raise ValueError("Unexpected world index or query branch")
        world_id = f"SRF4_{SPLIT}_{task['cell'].replace('-', '_')}_{index:03d}"
        if task["world_id"] != world_id or task["qid"] != f"{world_id}_{branch}":
            raise ValueError("Probe IDs differ from their frozen cell, world and query")
        worlds.setdefault(task["world_id"], []).append(task)
        world, query = task["world"], task["query"]
        oracle.validate_query(world, query)
        n = 6 if task["cell"].startswith("T6-") else 8
        if len(world["categories"]) != n or query["k"] != n // 2:
            raise ValueError("Wrong category count or top-k")
        if query["family"] != "TOPK" or task["family"] != "TOPK":
            raise ValueError("Only TOPK belongs in this probe")
        if query["aggregate"] != ("sum" if index % 2 == 0 else "mean") or query["direction"] != (
            "largest" if (index // 2) % 2 == 0 else "smallest"
        ):
            raise ValueError("Query differs from its frozen world index")
        if world["categories"] != _categories(n, index):
            raise ValueError("Printed category order differs from its frozen world index")
        if task["chart"] != "grouped_bar" or task["render_style"] != "main":
            raise ValueError("Unregistered chart or rendering style")
        expected_gap = (
            12 if task["cell"].endswith("SEPARATED") else (1 if task["world_index"] < 4 else 0)
        )
        if any(b["gap"] != expected_gap for b in _boundary(world, query["k"]).values()):
            raise ValueError("Ranking boundary does not satisfy the fixed cell")
        if task["boundary_by_series"] != _boundary(world, query["k"]):
            raise ValueError("Boundary sidecar mismatch")
        gold = oracle.encode(oracle.execute(world, query))
        independent = oracle.encode(oracle.independently_answer(world, query))
        if not gold == independent == task["gold_answer"] == task["gold_independent"]:
            raise ValueError("Exact gold mismatch")
        indices = oracle.selected(world, query)
        if task["target_indices"] != indices or task["target_sets"] != [
            world["categories"][i] for i in indices
        ]:
            raise ValueError("Target selection mismatch")
        if task["prompt"] != question(world, query) + "\n" + CONTRACT_V2:
            raise ValueError("Model prompt differs from the native question and contract")
    if len(worlds) != WORLD_COUNT:
        raise ValueError("Wrong world count")
    content = set()
    for pair in worlds.values():
        if len(pair) != 2 or {t["branch"] for t in pair} != {0, 1}:
            raise ValueError("World lacks its fixed query pair")
        first, second = sorted(pair, key=lambda t: t["branch"])
        if first["world"] != second["world"]:
            raise ValueError("Paired questions have different worlds")
        key = base_key(first["world"])
        if key in content:
            raise ValueError("Duplicate probe fact table; do not redraw")
        content.add(key)
        a, b = first["query"], second["query"]
        if (a["selection_series"], a["value"], b["selection_series"], b["value"]) != (
            "Alpha",
            "Beta",
            "Beta",
            "Alpha",
        ) or {
            key: value for key, value in a.items() if key not in ("selection_series", "value")
        } != {key: value for key, value in b.items() if key not in ("selection_series", "value")}:
            raise ValueError("Unexpected paired query intervention")
    for cell in CELLS:
        population = [pair[0] for pair in worlds.values() if pair[0]["cell"] == cell]
        if len(population) != WORLDS_PER_CELL or Counter(
            (t["query"]["aggregate"], t["query"]["direction"]) for t in population
        ) != {
            ("sum", "largest"): 2,
            ("sum", "smallest"): 2,
            ("mean", "largest"): 2,
            ("mean", "smallest"): 2,
        }:
            raise ValueError("Unbalanced probe cell")
    return {
        "worlds": WORLD_COUNT,
        "queries": QUERY_COUNT,
        "groups": GROUP_COUNT,
        "responses": RESPONSE_COUNT,
    }


def generate_groups(tasks=None):
    tasks = generate_tasks() if tasks is None else tasks
    validate_tasks(tasks)
    tasks = sorted(tasks, key=lambda t: (CELLS.index(t["cell"]), t["world_index"], t["branch"]))
    return [
        {
            "source": SPLIT,
            "block": block,
            "qid": task["qid"],
            "world_id": task["world_id"],
            "cell": task["cell"],
            "family": "TOPK",
            "group_index": index,
            "group_id": f"{SPLIT}-{block}/s000/g{index:05d}",
            "G": DRAWS_PER_GROUP,
        }
        for block in BLOCKS
        for index, task in enumerate(tasks)
    ]


def plan():
    return {
        "version": VERSION,
        "split": SPLIT,
        "generation_seed": GENERATION_SEED,
        "query_contract": oracle.VERSION,
        "measurement_version": MEASUREMENT_VERSION,
        "blocks": list(BLOCKS),
        "cells": list(CELLS),
        "worlds_per_cell": WORLDS_PER_CELL,
        "queries_per_world": QUERIES_PER_WORLD,
        "draws_per_group": DRAWS_PER_GROUP,
        "worlds": WORLD_COUNT,
        "queries": QUERY_COUNT,
        "groups": GROUP_COUNT,
        "responses": RESPONSE_COUNT,
        "training_updates": 0,
        "primary_policy": "F_PRIME_KNOWN_75",
        "known_subset_min_fraction": "3/4",
        "primary_signal": "active eligible group with P=PASS and both S=PASS and S=FAIL",
        "original_F": "DIAGNOSTIC_RETAINED",
        "boundary_gaps": {"separated": 12, "near": 1, "tied": 0},
        "boundary_columns": ["Alpha", "Beta"],
        "difficult_world_subtypes": {"indices_0_to_3": "near", "indices_4_to_7": "tied"},
        "generation_policy": "SINGLE_FIXED_CONSTRUCTION_NO_REJECTION_OR_SEED_RETRY",
        "answer_collisions": "RECORD_ONLY_NO_FILTER",
        "output_policy": "REPORT_ALL_CELLS_AND_EACH_BLOCK_RETAIN_TECHNICAL_FAILURES",
        "scope": "DEVELOPMENT_OPPORTUNITY_DIAGNOSTIC_NOT_TRAINING_OR_CONFIRMATION",
        "rendering": "grouped_bar/main; eight-category title must describe eight categories",
    }


def prepare_data(output):
    """Write audit/model-input manifests into a new or empty directory; no images."""
    root = Path(output)
    tasks = generate_tasks()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve existing probe artifacts; use a new empty output directory")
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "TASKS.jsonl", tasks)
    write_jsonl(
        root / "MODEL_INPUTS.jsonl",
        [{key: task[key] for key in MODEL_INPUT_FIELDS} for task in tasks],
    )
    write_jsonl(root / "PROBE_GROUPS.jsonl", generate_groups(tasks))
    (root / "PLAN.json").write_text(json.dumps(plan(), ensure_ascii=False, indent=2) + "\n")
    result = {
        "status": "MANIFESTS_READY_IMAGES_NOT_RENDERED",
        "version": VERSION,
        "counts": validate_tasks(tasks),
        "model_calls": 0,
        "training_updates": 0,
        "rejected_worlds": 0,
        "worlds_with_answer_collisions": len(
            {
                t["world_id"]
                for t in tasks
                if t["answer_collision_diagnostic"]["wrong_set_same_answer_count"]
            }
        ),
        "queries_with_answer_collisions": sum(
            bool(t["answer_collision_diagnostic"]["wrong_set_same_answer_count"]) for t in tasks
        ),
    }
    (root / "DATA_READY.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.output), indent=2))


if __name__ == "__main__":
    main()
