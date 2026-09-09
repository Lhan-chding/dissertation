"""Checkpoint-independent R3 panels and predeclared bank assignment.

This module accepts already loaded train/control scenes. It performs no file,
model, random-global-state or split-access operations. The runner verifies source
file manifests, solver uniqueness, image bytes and the remaining split gates.
"""

from __future__ import annotations

import copy
import itertools
import random
from collections import Counter, defaultdict

from .core import canonical_hash
from .prompts import OPERATIONS, build_prompt

FAMILIES = ("duplicate_encoding", "cross_series", "trend")
INTERFACES = ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH")
CHARTS = ("grouped_bar", "line")
STRATA = tuple(itertools.product(FAMILIES, INTERFACES))
SELECTION_VERSION = "R3-fixed-panels-v1-2026-09-10"
TRAIN_HASH_NAMESPACE = "R3-train-prompt-selection-v1"
CONTROL_HASH_NAMESPACE = "R3-control-scene-selection-v1"
REUSE_CHECK_SEED = 20260909
LAMBDA_CANDIDATES = (0.0, 0.01, 0.25, 1.0, 2.0)


def _validate_scenes(scenes, split):
    records, seen = list(scenes), set()
    for scene in records:
        if not isinstance(scene, dict) or scene.get("split") != split:
            raise ValueError(f"R3 {split} selection requires only {split} scenes")
        sid = scene.get("base_scene_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"R3 {split} requires a nonempty base_scene_id")
        if sid in seen:
            raise ValueError(f"duplicate R3 {split} base_scene_id")
        seen.add(sid)
        allowed = (
            (scene.get("constraint_family"), FAMILIES),
            (scene.get("chart_type"), CHARTS),
            (scene.get("operation"), OPERATIONS),
        )
        if any(not isinstance(value, str) or value not in choices for value, choices in allowed):
            raise ValueError(f"unknown R3 {split} family/chart/operation metadata")
        interface = scene.get("interface")
        if interface not in INTERFACES and not (split == "control" and interface is None):
            raise ValueError(f"R3 {split} has invalid or missing fixed interface")
    return records


def _prompt_record(scene, interface):
    scene = copy.deepcopy(scene)
    prompt = build_prompt(scene, interface)
    identity = {
        "phase": "R3",
        "split": scene["split"],
        "base_scene_id": scene["base_scene_id"],
        "interface": interface,
        "prompt_hash": prompt["prompt_hash"],
    }
    return {
        "prompt_id": canonical_hash(identity),
        "base_scene_id": scene["base_scene_id"],
        "family": scene["constraint_family"],
        "constraint_family": scene["constraint_family"],
        "interface": interface,
        "split": scene["split"],
        "chart_type": scene["chart_type"],
        "operation": scene["operation"],
        "scene_hash": canonical_hash(scene),
        "prompt_hash": prompt["prompt_hash"],
        "scene": scene,
        "prompt": prompt,
    }


def select_train_panel(train_scenes):
    """Hash-rank eight scenes per assigned family/interface, then cycle all six."""
    scenes = _validate_scenes(train_scenes, "train")
    groups = defaultdict(list)
    for scene in scenes:
        groups[(scene["constraint_family"], scene["interface"])].append(scene)
    chosen = {}
    for stratum in STRATA:
        ranked = sorted(
            groups[stratum],
            key=lambda scene: canonical_hash(
                [TRAIN_HASH_NAMESPACE, scene["base_scene_id"], scene["interface"]]
            ),
        )
        if len(ranked) < 8:
            raise ValueError(f"Insufficient R3 train prompts in stratum {stratum}")
        chosen[stratum] = ranked[:8]
    return [
        _prompt_record(chosen[stratum][index], stratum[1])
        for index in range(8)
        for stratum in STRATA
    ]


def _balanced_control_scenes(group, family):
    """Prefer least-used cells, then chart and operation, breaking ties by hash.

    With at least two candidates in each of the six metadata cells this selects
    cell counts 1/1/1/1/2/2, chart counts 4/4 and operation counts 2/3/3. Scarce
    metadata cells retain the same rule applied to available scenes only.
    """
    if len(group) < 8:
        raise ValueError(f"Insufficient R3 control scenes in family {family}")
    cells = defaultdict(list)
    for scene in group:
        cells[(scene["chart_type"], scene["operation"])].append(scene)
    cells = {
        cell: sorted(
            scenes,
            key=lambda scene: canonical_hash([CONTROL_HASH_NAMESPACE, scene["base_scene_id"]]),
        )
        for cell, scenes in cells.items()
    }
    used_cells, used_charts, used_operations = Counter(), Counter(), Counter()
    chosen = []
    for _ in range(8):
        available = [cell for cell, rows in cells.items() if used_cells[cell] < len(rows)]
        cell = min(
            available,
            key=lambda cell: (
                used_cells[cell],
                used_charts[cell[0]],
                used_operations[cell[1]],
                canonical_hash([CONTROL_HASH_NAMESPACE, "cell-tie", family, *cell]),
            ),
        )
        chosen.append(cells[cell][used_cells[cell]])
        used_cells[cell] += 1
        used_charts[cell[0]] += 1
        used_operations[cell[1]] += 1
    return chosen


def select_control_panel(control_scenes):
    """Select eight base scenes per family, then present both original interfaces."""
    scenes = _validate_scenes(control_scenes, "control")
    groups = defaultdict(list)
    for scene in scenes:
        groups[scene["constraint_family"]].append(scene)
    return [
        _prompt_record(scene, interface)
        for family in FAMILIES
        for scene in _balanced_control_scenes(groups[family], family)
        for interface in INTERFACES
    ]


def build_r3_plan(train_scenes, control_scenes):
    """Return the same sealed prompt IDs/banks for cold and warm checkpoints.

    Rollout keys and seeds belong to the runner and must also bind the actual
    checkpoint. Identical prompt IDs never authorize reuse of cold rollouts at
    a warm checkpoint.
    """
    train = _validate_scenes(train_scenes, "train")
    control = _validate_scenes(control_scenes, "control")
    if {s["base_scene_id"] for s in train} & {s["base_scene_id"] for s in control}:
        raise ValueError("R3 train/control base_scene_id overlap in provided source splits")
    train_prompts = select_train_panel(train)
    control_prompts = select_control_panel(control)
    train_ids = [row["prompt_id"] for row in train_prompts]
    control_ids = [row["prompt_id"] for row in control_prompts]
    banks = [train_ids[index : index + 4] for index in range(0, 48, 4)]
    plan = {
        "phase": "R3",
        "selection_version": SELECTION_VERSION,
        "train_prompts": train_prompts,
        "control_prompts": control_prompts,
        "banks": banks,
        "reuse_check_bank_indices": sorted(random.Random(REUSE_CHECK_SEED).sample(range(12), 2)),
        "response_banks": [0, 6],
        "lambda_candidates": list(LAMBDA_CANDIDATES),
        "reuse_check_lambdas": [0.0, 1.0],
        "warm_direct_lambdas": [0.0, 1.0],
        "counts": {
            "train_base_scenes": 48,
            "train_prompts": 48,
            "control_base_scenes": 24,
            "control_prompts": 48,
            "banks": 12,
            "prompts_per_bank": 4,
            "rollouts_per_train_prompt": 8,
            "train_rollouts": 384,
            "rollouts_per_control_prompt": 16,
            "control_proposal_rollouts": 768,
            "candidates_per_bank": 5,
            "candidate_updates": 60,
            "response_candidates": 10,
            "reuse_check_comparisons": 4,
            "warm_direct_candidates": 4,
            "warm_direct_rollouts": 3072,
            "cold_direct_rollouts": 0,
        },
        "selection_rules": {
            "train_hash_namespace": TRAIN_HASH_NAMESPACE,
            "train_order": "eight rounds over the fixed six strata; contiguous groups of four",
            "train_strata_order": [{"family": f, "interface": i} for f, i in STRATA],
            "control_hash_namespace": CONTROL_HASH_NAMESPACE,
            "control_cell_order": (
                "least selected cell, then chart, then operation, then stable hash"
            ),
            "reuse_check_rng": "Python random.Random(seed).sample(range(12), 2), sorted",
            "reuse_check_seed": REUSE_CHECK_SEED,
            "checkpoint_invariant_prompt_ids": True,
            "checkpoint_bound_rollout_rng_required": True,
            "outcome_filtering": False,
        },
        "input_hashes": {
            "train_scenes": canonical_hash(sorted(train, key=lambda s: s["base_scene_id"])),
            "control_scenes": canonical_hash(sorted(control, key=lambda s: s["base_scene_id"])),
        },
        "train_prompt_ids_hash": canonical_hash(train_ids),
        "control_prompt_ids_hash": canonical_hash(control_ids),
        "banks_hash": canonical_hash(banks),
    }
    return {**plan, "plan_hash": canonical_hash(plan)}
