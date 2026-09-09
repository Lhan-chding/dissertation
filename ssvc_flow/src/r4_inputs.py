"""Predeclared R4 inputs, without model calls, filesystem writes or global RNG.

The caller verifies source-file manifests and image bytes against passed R0
evidence. Graph OOD is selected from the existing graph_structure pool, whose
truths were generated disjointly from ID. This module also checks every graph
candidate against the supplied ID exclusion catalogue. The catalogue's scope is
reported explicitly; it never authorizes opening the sealed confirm split.
"""

from __future__ import annotations

import copy
import itertools
import re
from collections import Counter, defaultdict

from .constraint_solver import solve
from .core import canonical_hash
from .legacy_frozen import EVALUATION_SPLITS, build_legacy_prompt
from .prompts import OPERATIONS, build_prompt

FAMILIES = ("duplicate_encoding", "cross_series", "trend")
INTERFACES = ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH")
CHARTS = ("grouped_bar", "line")
TOPOLOGIES = ("path", "cycle")
STRATA = tuple(itertools.product(FAMILIES, INTERFACES))
DEV_CELLS = tuple(itertools.product(FAMILIES, CHARTS, OPERATIONS))
OOD_CELLS = tuple(itertools.product(TOPOLOGIES, CHARTS, OPERATIONS))
SELECTION_VERSION = "R4-fixed-inputs-v1-2026-09-10"
TRAIN_SEED = 17
TRAIN_NAMESPACE = "R4-train-stratified-hash-order-v1"
DEV_NAMESPACE = "R4-dev-panel-hash-selection-v1"
OOD_NAMESPACE = "R4-graph-ood-hash-selection-v1"
ID_SPLITS = frozenset({"train", "dev", "control", "calibration", "confirm", "natural_pool"})


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_n(scenes, split, expected_count=None):
    scenes, seen = list(scenes), set()
    for scene in scenes:
        if not isinstance(scene, dict) or scene.get("split") != split:
            raise ValueError(f"R4 {split} requires only {split} scenes")
        sid = scene.get("base_scene_id")
        if not isinstance(sid, str) or not sid:
            raise ValueError("R4 requires a nonempty base_scene_id")
        if sid in seen:
            raise ValueError(f"duplicate R4 {split} base_scene_id")
        seen.add(sid)
        for value, choices in (
            (scene.get("constraint_family"), FAMILIES),
            (scene.get("chart_type"), CHARTS),
            (scene.get("operation"), OPERATIONS),
        ):
            if not isinstance(value, str) or value not in choices:
                raise ValueError("unknown R4 family/chart/operation metadata")
        interface = scene.get("interface")
        if split == "train" and interface not in INTERFACES:
            raise ValueError("R4 train requires each scene's original fixed interface")
        if split != "train" and interface is not None:
            raise ValueError("R4 evaluation requires original paired-interface scenes")
        if split != "ood" and scene.get("ood_subtype") is not None:
            raise ValueError("R4 ID pool contains OOD metadata")
        if split == "ood" and scene.get("ood_subtype") not in {
            "graph_structure",
            "numeric_shift",
            "render_expression",
        }:
            raise ValueError("unknown R4 OOD subtype")
    if expected_count is not None and len(scenes) != expected_count:
        raise ValueError(f"R4 {split} requires exactly {expected_count} base scenes")
    return scenes


def _root(value):
    return None if value is None else str(value)


def _n_record(scene, interface, data_root=None):
    scene = copy.deepcopy(scene)
    prompt = build_prompt(scene, interface)
    if scene.get("prompt_hashes", {}).get(interface) != prompt["prompt_hash"]:
        raise ValueError("R4 original N prompt hash mismatch")
    identity = {
        "protocol": "R4-original-N-prompts-v1",
        "base_scene_id": scene["base_scene_id"],
        "split": scene["split"],
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
        "track": "N",
        "evaluation_domain": "graph_OOD" if scene["split"] == "ood" else "ID",
        "chart_type": scene["chart_type"],
        "operation": scene["operation"],
        "scene_hash": canonical_hash(scene),
        "prompt_hash": prompt["prompt_hash"],
        "scene": scene,
        "prompt": prompt,
        "max_new_tokens": 64,
        "enable_thinking": False,
        "data_root": _root(data_root),
    }


def build_train_schedule(train_scenes, *, data_root=None):
    """Lock all 576 original prompts and the shared seed-17, 64 x B4 order.

    Each stratum is ranked by a seed-bound stable hash. Interleaving the six
    rankings yields four occurrences per stratum in every 24 slots. Only the
    first 256 slots are consumed, without replacement. No arm/checkpoint or
    outcome enters this ordering; generation seeds still bind each arm's policy.
    """
    scenes = _validate_n(train_scenes, "train", 576)
    groups = defaultdict(list)
    for scene in scenes:
        groups[(scene["constraint_family"], scene["interface"])].append(scene)
    if any(len(groups[stratum]) != 96 for stratum in STRATA):
        raise ValueError("R4 train requires 96 fixed prompts in each of six strata")
    for stratum in STRATA:
        groups[stratum].sort(
            key=lambda s: canonical_hash(
                [TRAIN_NAMESPACE, TRAIN_SEED, s["base_scene_id"], s["interface"]]
            )
        )
    prompts = [
        _n_record(groups[stratum][index], stratum[1], data_root)
        for index in range(96)
        for stratum in STRATA
    ]
    prompt_ids = [record["prompt_id"] for record in prompts]
    steps = [prompt_ids[index : index + 4] for index in range(0, 256, 4)]
    return {"sampler_seed": TRAIN_SEED, "train_prompts": prompts, "train_steps": steps}


def _dev_scenes(dev_scenes):
    scenes = _validate_n(dev_scenes, "dev", 144)
    counts = Counter((s["constraint_family"], s["chart_type"], s["operation"]) for s in scenes)
    if counts != {cell: 8 for cell in DEV_CELLS}:
        raise ValueError("R4 complete dev requires eight scenes per family/chart/operation cell")
    return scenes


def select_dev_panel(dev_scenes, *, data_root=None):
    """Select two scenes in each of 18 metadata cells, paired as 72 prompts."""
    groups = defaultdict(list)
    for scene in _dev_scenes(dev_scenes):
        groups[(scene["constraint_family"], scene["chart_type"], scene["operation"])].append(scene)
    selected = [
        scene
        for cell in DEV_CELLS
        for scene in sorted(
            groups[cell], key=lambda s: canonical_hash([DEV_NAMESPACE, s["base_scene_id"]])
        )[:2]
    ]
    return [
        _n_record(scene, interface, data_root) for scene in selected for interface in INTERFACES
    ]


def build_dev_prompts(dev_scenes, *, data_root=None):
    """All 144 dev bases at both original interfaces; panel identities persist."""
    return [
        _n_record(scene, interface, data_root)
        for scene in sorted(_dev_scenes(dev_scenes), key=lambda s: s["base_scene_id"])
        for interface in INTERFACES
    ]


def _id_catalog(id_scenes):
    rows, seen = list(id_scenes), set()
    for scene in rows:
        if not isinstance(scene, dict) or scene.get("split") not in ID_SPLITS:
            raise ValueError("R4 ID exclusions require ID source scenes")
        sid = scene.get("base_scene_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("missing or duplicate R4 ID exclusion identity")
        seen.add(sid)
        if any(not _valid_hash(scene.get(key)) for key in ("truth_structure_hash", "image_hash")):
            raise ValueError("R4 ID exclusions require original truth/image hashes")
    if not rows:
        raise ValueError("R4 graph OOD requires a nonempty ID exclusion catalogue")
    return rows


def _validate_graph(scene):
    truth = scene.get("truth_world")
    if (
        not isinstance(truth, list)
        or len(truth) != 4
        or any(type(value) is not int or not 0 <= value < 50 for value in truth)
        or scene.get("truth_structure_hash") != canonical_hash(truth)
    ):
        raise ValueError("R4 graph OOD truth must preserve ID numeric range and truth hash")
    if (
        scene["constraint_family"] != "cross_series"
        or scene.get("graph_topology") not in TOPOLOGIES
        or scene.get("render_variant") != "standard"
        or scene.get("expression_variant") != "fixed_N_v1"
        or (scene.get("image_width"), scene.get("image_height")) != (768, 512)
        or not _valid_hash(scene.get("image_hash"))
    ):
        raise ValueError("R4 graph OOD must change graph topology only")
    cue = scene.get("cue")
    if not isinstance(cue, dict) or cue.get("family") != "cross_series":
        raise ValueError("R4 graph OOD requires cross_series edges")
    # The independent solver validates edge indices/totals and the 0..99 observed domain.
    solutions = solve(scene.get("observed_world"), cue)
    if solutions != [truth] or scene.get("solution_count") != 1:
        raise ValueError("R4 graph OOD failed independent unique-repair solver")
    changed = [i for i in range(4) if truth[i] != scene["observed_world"][i]]
    if type(scene.get("changed_index")) is not int or changed != [scene["changed_index"]]:
        raise ValueError("R4 graph OOD changed_index mismatch")
    pairs = [(edge[0], edge[1]) for edge in cue["edges"]]
    degree = Counter(index for edge in pairs for index in edge)
    expected = [1, 1, 2, 2] if scene["graph_topology"] == "path" else [2, 2, 2, 2]
    if len(set(pairs)) != len(pairs) or sorted(degree.values()) != expected:
        raise ValueError("R4 graph OOD edge structure disagrees with path/cycle topology")
    if scene.get("relation_count") != len(pairs):
        raise ValueError("R4 graph OOD relation_count mismatch")


def select_graph_ood_panel(ood_scenes, *, id_scenes, data_root=None):
    """Validate the graph candidate pool, then hash-select three per 12 cells.

    Numeric/render-expression OOD rows are outside this panel. A malformed graph
    candidate fails the pool before selection; it is never silently filtered.
    Image-byte verification belongs to the caller; this function checks hashes.
    """
    source = _validate_n(ood_scenes, "ood")
    candidates = [s for s in source if s.get("ood_subtype") == "graph_structure"]
    exclusions = _id_catalog(id_scenes)
    for key in ("base_scene_id", "truth_structure_hash", "image_hash"):
        values = [s.get(key) for s in candidates]
        if set(values) & {s[key] for s in exclusions}:
            raise ValueError(f"R4 graph OOD / ID {key} overlap")
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate graph OOD {key}")
    groups = defaultdict(list)
    for scene in candidates:
        _validate_graph(scene)
        groups[(scene["graph_topology"], scene["chart_type"], scene["operation"])].append(scene)
    if any(len(groups[cell]) < 3 for cell in OOD_CELLS):
        raise ValueError(
            "Insufficient R4 graph OOD scenes: three required per topology/chart/operation cell"
        )
    selected = [
        scene
        for cell in OOD_CELLS
        for scene in sorted(
            groups[cell], key=lambda s: canonical_hash([OOD_NAMESPACE, s["base_scene_id"]])
        )[:3]
    ]
    return [
        _n_record(scene, interface, data_root) for scene in selected for interface in INTERFACES
    ]


def build_legacy_eval_prompts(legacy_scenes, *, legacy_lock):
    """Keep all 176 loader-verified L prompts under their original 48-token lock."""
    lock = copy.deepcopy(legacy_lock)
    generation = lock.get("generation", {})
    if lock.get("actual_max_new_tokens") != 48 or generation.get("max_new_tokens") != 48:
        raise ValueError("R4 legacy requires its original verified 48-token lock")
    scenes, seen, pairs = list(legacy_scenes), set(), defaultdict(list)
    if len(scenes) != 176:
        raise ValueError("R4 legacy requires the complete 176 original evaluation prompts")
    for scene in scenes:
        sid = scene.get("scene_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("missing or duplicate R4 legacy scene_id")
        seen.add(sid)
        if (
            scene.get("split") not in EVALUATION_SPLITS
            or scene.get("track") != "L"
            or scene.get("interface") not in {"collision", "separating"}
            or scene["interface"] != scene.get("condition")
            or scene.get("base_scene_id") != scene.get("pair_id")
        ):
            raise ValueError("R4 legacy requires loader-verified original evaluation pairs")
        pairs[scene["base_scene_id"]].append(scene)
    if len(pairs) != 88 or any(
        len(pair) != 2
        or {r["interface"] for r in pair} != {"collision", "separating"}
        or any(
            pair[0].get(key) != pair[1].get(key)
            for key in ("truth", "observation", "split", "family", "facts")
        )
        for pair in pairs.values()
    ):
        raise ValueError("R4 legacy requires 88 matching original scene pairs")
    records = []
    for original in sorted(scenes, key=lambda s: s["scene_id"]):
        scene = copy.deepcopy(original)
        prompt = build_legacy_prompt(scene)
        records.append(
            {
                "prompt_id": canonical_hash(
                    {
                        "protocol": "R4-original-L-prompts-v1",
                        "scene_id": scene["scene_id"],
                        "prompt_hash": prompt["prompt_hash"],
                    }
                ),
                "base_scene_id": scene["base_scene_id"],
                "family": scene["family"],
                "constraint_family": scene["family"],
                "interface": scene["interface"],
                "split": scene["split"],
                "track": "L",
                "evaluation_domain": "legacy",
                "operation": scene["operation"],
                "scene_hash": canonical_hash(scene),
                "prompt_hash": prompt["prompt_hash"],
                "scene": scene,
                "prompt": prompt,
                "max_new_tokens": 48,
                "enable_thinking": False,
                "generation": copy.deepcopy(generation),
                "data_root": None,
            }
        )
    return records


def build_r4_plan(
    train_scenes,
    dev_scenes,
    ood_scenes,
    legacy_scenes,
    *,
    id_scenes,
    legacy_lock,
    data_root=None,
    ood_data_root=None,
):
    """Return JSON-serializable shared input identities and endpoint budgets.

    The hash binds provided sources, roots, the legacy lock and fixed selections.
    It does not replace runtime source-file/image verification or authorize reuse
    across different checkpoint, arm, prompt, generation or sample identities.
    """
    train, dev, ood, legacy = map(list, (train_scenes, dev_scenes, ood_scenes, legacy_scenes))
    if {s["base_scene_id"] for s in train} & {s["base_scene_id"] for s in dev}:
        raise ValueError("R4 train/dev base_scene_id overlap")
    exclusions = _id_catalog(id_scenes)
    excluded = {s["base_scene_id"]: s for s in exclusions}
    if any(excluded.get(s["base_scene_id"]) != s for s in [*train, *dev]):
        raise ValueError("R4 ID exclusions must include the full unchanged train and dev sources")
    schedule = build_train_schedule(train, data_root=data_root)
    plan = {
        "phase": "R4",
        "selection_version": SELECTION_VERSION,
        **schedule,
        "dev_panel_prompts": select_dev_panel(dev, data_root=data_root),
        "dev_prompts": build_dev_prompts(dev, data_root=data_root),
        "legacy_prompts": build_legacy_eval_prompts(legacy, legacy_lock=legacy_lock),
        "ood_prompts": select_graph_ood_panel(
            ood,
            id_scenes=exclusions,
            data_root=data_root if ood_data_root is None else ood_data_root,
        ),
        "counts": {
            "train_base_scenes": 576,
            "train_prompts": 576,
            "optimizer_steps_per_arm": 64,
            "prompts_per_step": 4,
            "rollouts_per_prompt": 8,
            "train_rollouts_per_arm": 2048,
            "train_rollouts_two_arms": 4096,
            "dev_panel_base_scenes": 36,
            "dev_panel_prompts": 72,
            "dev_base_scenes": 144,
            "dev_prompts": 288,
            "legacy_base_scenes": 88,
            "legacy_prompts": 176,
            "ood_base_scenes": 36,
            "ood_prompts": 72,
            "evaluation_rollouts_step0_shared": 576,
            "evaluation_rollouts_step32_two_arms": 1152,
            "evaluation_rollouts_step64_N_two_arms": 4608,
            "evaluation_rollouts_step64_L_two_arms": 2816,
            "evaluation_rollouts_step64_OOD_two_arms": 1152,
            "evaluation_rollouts_step32_and_step64": 9728,
            "evaluation_rollouts_including_step0": 10304,
        },
        "selection_rules": {
            "train_namespace": TRAIN_NAMESPACE,
            "train_seed": TRAIN_SEED,
            "train_order": (
                "seed-bound hash within stratum, interleave six strata, "
                "first 256 slots without replacement"
            ),
            "strata": [{"family": family, "interface": interface} for family, interface in STRATA],
            "dev_namespace": DEV_NAMESPACE,
            "dev_scenes_per_cell": 2,
            "ood_namespace": OOD_NAMESPACE,
            "ood_scenes_per_cell": 3,
            "ood_source": "existing audited graph_structure pool; original image paths",
            "outcome_filtering": False,
            "shared_arm_prompt_order": True,
            "checkpoint_bound_on_policy_samples_required": True,
            "evaluation_sample_key_deduplication_required": True,
        },
        "id_exclusion_scope": {
            "provided_scenes": len(exclusions),
            "split_counts": dict(sorted(Counter(s["split"] for s in exclusions).items())),
            "global_disjointness_requires_passed_R0_manifest_binding": True,
        },
        "input_hashes": {
            **{
                key: canonical_hash(sorted(rows, key=lambda s: s["base_scene_id"]))
                for key, rows in (
                    ("train_scenes", train),
                    ("dev_scenes", dev),
                    ("ood_scenes", ood),
                    ("id_exclusions", exclusions),
                )
            },
            "legacy_scenes": canonical_hash(sorted(legacy, key=lambda s: s["scene_id"])),
        },
        "legacy_lock_hash": canonical_hash(legacy_lock),
        "legacy_generation": copy.deepcopy(legacy_lock["generation"]),
    }
    plan["train_steps_hash"] = canonical_hash(plan["train_steps"])
    plan["prompt_ids_hashes"] = {
        key: canonical_hash([r["prompt_id"] for r in plan[key]])
        for key in (
            "train_prompts",
            "dev_panel_prompts",
            "dev_prompts",
            "legacy_prompts",
            "ood_prompts",
        )
    }
    return {**plan, "plan_hash": canonical_hash(plan)}
