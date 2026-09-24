"""Validated prospective experiment registry and derived (not measured) work."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..core import canonical_hash

SELECTABLE = {
    "R0": [2, 0, 0, 0],
    "R1": [0, 2, 0, 0],
    "R2": [2, 0, 1, 0],
    "R3": [2, 0, 0, 1],
    "R4": [2, 0, 0.5, 0.5],
    "R5": [0, 2, 1, 0],
    "R6": [0, 2, 0, 1],
    "R7": [0, 2, 0.5, 0.5],
}
EXTERNAL_BASELINES = ("GDPO_R4", "SAW_R4", "DIRECT_REPAIR_R4")
DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "docs/prospective_selection/design/protocol.json"
)


def load_protocol(path=None):
    config = json.loads(Path(path or DEFAULT_PATH).read_text())
    validate_protocol(config)
    return config


def lineage_registry(config):
    """Independent lineage identities; anchors remain correlated within lineage."""
    return [
        {
            **copy.deepcopy(row),
            "lineage_id": str(row["seed"]),
            "origin_ids": [f"{row['seed']}_t{step}" for step in row["anchors"]],
        }
        for role in ("development", "tuning", "test_pool")
        for row in config["origins"][role]
    ]


def get_lineage(config, lineage):
    matches = [r for r in lineage_registry(config) if r["lineage_id"] == str(lineage)]
    if len(matches) != 1:
        raise ValueError(f"Unregistered source lineage: {lineage}")
    return matches[0]


def validate_protocol(config):
    if config.get("protocol") != "ssvc-prospective-selection-v2-20260924":
        raise ValueError("Unknown prospective protocol")
    if config["actions"]["selectable"] != SELECTABLE:
        raise ValueError("All selectors require the same eight frozen recipes")
    if tuple(config["actions"]["external_baselines"]) != EXTERNAL_BASELINES:
        raise ValueError("External baseline registry differs")
    execution, train = config["execution"], config["training"]
    if not 1 <= execution["max_concurrent_gpu_jobs"] <= 5 or execution["gpus_per_job"] != 1:
        raise ValueError("At most five independent single-GPU jobs are allowed")
    if not execution["test_requires_frozen_rule_and_predecision"]:
        raise ValueError("Test requires a frozen rule and predecision")
    expected = {
        "B": 4,
        "K": 8,
        "branch_steps": 32,
        "source_steps_max": 96,
        "source_pool_scenes": 384,
        "branch_pool_scenes": 192,
        "test_continuation_repeats": 2,
        "development_continuation_repeats": 1,
        "source_schedule_seed": 73001,
        "branch_schedule_seeds": [73011, 73012],
        "training_rng_excludes_recipe": True,
        "evaluation_rng_restored": True,
    }
    for key, value in expected.items():
        if train.get(key) != value:
            raise ValueError(f"Frozen training contract differs: {key}")
    seeds = set()
    for role, count, start in [
        ("development", 12, 61001),
        ("tuning", 4, 62001),
        ("test_pool", 48, 63001),
    ]:
        rows = config["origins"][role]
        if len(rows) != count:
            raise ValueError(f"Unexpected {role} lineage count")
        for i, row in enumerate(rows):
            expected_role = "locked_test" if role == "test_pool" else role
            expected_recipe = (
                ("R0", "R0", "R1", "R1")[i % 4] if role == "test_pool" else ("R0", "R1")[i % 2]
            )
            anchors = [[32], [96], [32], [96]][i % 4] if role == "test_pool" else [32, 96]
            if (
                row["seed"] != start + i
                or row["seed"] in seeds
                or row["role"] != expected_role
                or row["source_recipe"] != expected_recipe
                or row["anchors"] != anchors
                or row["source_steps"] != max(anchors)
            ):
                raise ValueError("Lineage roles, seeds, recipes and anchors must remain registered")
            seeds.add(row["seed"])
    if config["origins"]["test_n_choices"] != [12, 16, 20, 24, 32, 48]:
        raise ValueError("Unexpected final N choices")
    if config["panels"]["T"]["draws_choices"] != [16, 32, 64]:
        raise ValueError("Unexpected final sampling choices")
    return config


def workload(config, n=None, m=None):
    """Conservative unique-action upper bounds, never a completion claim."""
    n = config["origins"]["test_n_initial"] if n is None else n
    m = config["panels"]["T"]["draws_initial"] if m is None else m
    if (
        n not in config["origins"]["test_n_choices"]
        or m not in config["panels"]["T"]["draws_choices"]
    ):
        raise ValueError("N/m must be preregistered choices")
    origins, t, panels = config["origins"], config["training"], config["panels"]
    development = origins["development"] + origins["tuning"]
    contexts = sum(len(row["anchors"]) for row in development)
    source = sum(row["source_steps"] for row in development + origins["test_pool"][:n])
    dev_branches = (
        contexts
        * (len(SELECTABLE) + len(EXTERNAL_BASELINES))
        * t["development_continuation_repeats"]
    )
    # Four selectors plus BestStatic/R0 and three external controls; overlap deduplicates.
    test_branches = n * t["test_continuation_repeats"] * (4 + 2 + len(EXTERNAL_BASELINES))
    dev_updates, test_updates = dev_branches * t["branch_steps"], test_branches * t["branch_steps"]
    total = source + dev_updates + test_updates
    return {
        "test_lineages": n,
        "test_draws": m,
        "source_updates": source,
        "development_updates": dev_updates,
        "test_updates_upper": test_updates,
        "all_updates_upper": total,
        "all_training_outputs_upper": total * t["B"] * t["K"],
        "development_branches": dev_branches,
        "test_unique_branches_upper": test_branches,
        "prestate_outputs": (contexts + n) * 2 * panels["P"]["prompts"] * panels["P"]["draws"],
        "development_H32_outputs": dev_branches * panels["D"]["prompts"] * panels["D"]["draws_H32"],
        "test_H32_outputs_upper": test_branches * panels["T"]["prompts"] * m,
        "test_H8_diagnostic_outputs_upper": test_branches
        * panels["T_H8"]["prompts"]
        * panels["T_H8"]["draws"],
        "historical_E_outputs": 16
        * panels["historical_E"]["prompts"]
        * panels["historical_E"]["draws"],
    }


def protocol_id(config):
    return canonical_hash(config)
