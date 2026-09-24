"""Metadata-only prospective panel selection with explicit exposure accounting.

Truths are used only to validate disjointness, never to rank candidate scenes.
No model inference or job submission occurs here. The returned raw panel store
is for runners; selectors receive separately permissioned feature packets.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..core import canonical_hash
from ..modeling_v3.io import sha256_file
from ..r4_inputs import DEV_CELLS, INTERFACES, STRATA, _n_record, _validate_n
from .protocol import lineage_registry, validate_protocol
from .sampler import build_schedule


def _truth(scene):
    truth = scene.get("truth_world")
    if not isinstance(truth, list) or len(truth) != 4 or any(type(v) is not int for v in truth):
        raise ValueError("Original four-integer truth is required for overlap audit")
    value = canonical_hash(truth)
    if scene.get("truth_structure_hash") != value:
        raise ValueError("Original truth hash mismatch")
    return value


def _identities(scenes):
    return {s["base_scene_id"] for s in scenes}, {_truth(s) for s in scenes}


def _disjoint(named):
    prior_ids, prior_truths = set(), set()
    for name, scenes in named.items():
        ids, truths = _identities(scenes)
        if len(ids) != len(scenes) or len(truths) != len(scenes):
            raise ValueError(f"Duplicate scene or truth inside {name}")
        if ids & prior_ids or truths & prior_truths:
            raise ValueError(f"Scene/truth overlap across pools at {name}")
        prior_ids.update(ids)
        prior_truths.update(truths)


def _rank(scene, namespace, seed):
    return canonical_hash([namespace, seed, scene["base_scene_id"]]), scene["base_scene_id"]


def split_training_pool(train_scenes, *, data_root=None):
    scenes = _validate_n(train_scenes, "train", 576)
    _disjoint({"train": scenes})
    groups = defaultdict(list)
    for scene in scenes:
        groups[scene["constraint_family"], scene["interface"]].append(scene)
    if Counter({s: len(groups[s]) for s in STRATA}) != {s: 96 for s in STRATA}:
        raise ValueError("Training pool requires 96 scenes in each of six strata")
    for stratum in STRATA:
        groups[stratum].sort(key=lambda s: _rank(s, "prospective-train-partition-v1", 73001))
    pools = {
        "source": [groups[s][i] for i in range(64) for s in STRATA],
        "continuation": [groups[s][i] for i in range(64, 96) for s in STRATA],
    }
    _disjoint(pools)
    return {
        name: [_n_record(s, s["interface"], data_root) for s in pool]
        for name, pool in pools.items()
    }


def _exposure(exposure):
    if not isinstance(exposure, dict) or exposure.get("status") != "COMPLETE":
        raise ValueError("A completed historical confirm exposure audit is required")
    for key in ("exposed_base_scene_ids", "exposed_truth_hashes", "evidence_paths"):
        values = exposure.get(key)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"Exposure audit must explicitly provide {key}")
    if not exposure["evidence_paths"]:
        raise ValueError(
            "Exposure audit requires evidence paths; absence is not proof of no exposure"
        )
    return set(exposure["exposed_base_scene_ids"]), set(exposure["exposed_truth_hashes"])


def _select_panel(
    scenes,
    *,
    split,
    count_per_cell,
    seed,
    panel,
    excluded_ids=(),
    excluded_truths=(),
    data_root=None,
):
    candidates = _validate_n(scenes, split)
    _disjoint({split: candidates})
    groups = defaultdict(list)
    for scene in candidates:
        if scene["base_scene_id"] not in excluded_ids and _truth(scene) not in excluded_truths:
            groups[scene["constraint_family"], scene["chart_type"], scene["operation"]].append(
                scene
            )
    shortages = {
        str(cell): count_per_cell - len(groups[cell])
        for cell in DEV_CELLS
        if len(groups[cell]) < count_per_cell
    }
    if shortages:
        raise ValueError(f"Insufficient unexposed {panel} metadata cells: {shortages}")
    selected = [
        scene
        for cell in DEV_CELLS
        for scene in sorted(groups[cell], key=lambda s: _rank(s, f"prospective-{panel}-v1", seed))[
            :count_per_cell
        ]
    ]
    return [
        {**_n_record(scene, interface, data_root), "panel": panel}
        for scene in selected
        for interface in INTERFACES
    ]


def _base_scenes(prompts):
    scenes = {}
    for record in prompts:
        scene = record["scene"]
        sid = record["base_scene_id"]
        if sid != scene["base_scene_id"]:
            raise ValueError("Prompt/base-scene identity mismatch")
        if sid in scenes and scenes[sid] != scene:
            raise ValueError("Paired interfaces disagree on underlying scene")
        scenes[sid] = scene
    return list(scenes.values())


def _reuse_p(existing_panels, data_root):
    raw = existing_panels.get("panels", existing_panels)
    if "P" not in raw or len(raw["P"]) != 72:
        raise ValueError("Original D2 P manifest with 72 prompts is required")
    rows = copy.deepcopy(raw["P"])
    scenes = _base_scenes(rows)
    if len(scenes) != 36 or any(s.get("split") != "control" for s in scenes):
        raise ValueError("Original P requires 36 control scenes")
    counts = Counter((r["base_scene_id"], r["interface"]) for r in rows)
    if counts != {(s["base_scene_id"], interface): 1 for s in scenes for interface in INTERFACES}:
        raise ValueError("Original P must preserve both interfaces exactly once")
    for row in rows:
        verified = _n_record(row["scene"], row["interface"], data_root)
        for key in ("prompt_id", "prompt_hash", "scene_hash"):
            if row.get(key) != verified[key]:
                raise ValueError(f"Original D2 P binding differs: {key}")
        row.update(verified)
        row["panel"] = "P"
    return rows


def build_dataset(
    train_scenes,
    dev_scenes,
    confirm_scenes,
    *,
    existing_panels,
    confirm_exposure,
    config,
    data_root=None,
    development_only=False,
):
    validate_protocol(config)
    train_scenes, dev_scenes, confirm_scenes = (
        list(train_scenes),
        list(dev_scenes),
        list(confirm_scenes),
    )
    pools = split_training_pool(train_scenes, data_root=data_root)
    p = _reuse_p(existing_panels, data_root)
    d = _select_panel(
        dev_scenes, split="dev", count_per_cell=4, seed=74001, panel="D", data_root=data_root
    )
    t = []
    if not development_only:
        exposure_ids, exposure_truths = _exposure(confirm_exposure)
        # Historical ID exposure also excludes aliases carrying the same truth.
        exposure_truths |= {_truth(s) for s in confirm_scenes if s["base_scene_id"] in exposure_ids}
        t = _select_panel(
            confirm_scenes,
            split="confirm",
            count_per_cell=8,
            seed=74002,
            panel="T",
            excluded_ids=exposure_ids,
            excluded_truths=exposure_truths,
            data_root=data_root,
        )
    _disjoint(
        {
            "train": list(train_scenes),
            "P": _base_scenes(p),
            "D": _base_scenes(d),
            "T": _base_scenes(t),
        }
    )
    # Four scenes per family: balanced charts and deterministic operation rotation.
    t_by_cell = defaultdict(list)
    for scene in _base_scenes(t):
        t_by_cell[scene["constraint_family"], scene["chart_type"], scene["operation"]].append(scene)
    from ..prompts import OPERATIONS
    from ..r4_inputs import CHARTS, FAMILIES

    t8_ids = set()
    for fi, family in enumerate(FAMILIES if t else ()):
        for chart in CHARTS:
            for offset in range(2):
                cell = family, chart, tuple(OPERATIONS)[(fi + offset) % len(OPERATIONS)]
                chosen = min(t_by_cell[cell], key=lambda s: _rank(s, "prospective-T8-v1", 74002))
                t8_ids.add(chosen["base_scene_id"])
    t8 = [{**copy.deepcopy(row), "panel": "T_H8"} for row in t if row["base_scene_id"] in t8_ids]
    source_schedule = build_schedule(pools["source"], seed=73001, steps=96, role="source")
    branch_schedules = {
        str(i): build_schedule(
            pools["continuation"], seed=seed, steps=32, role="continuation", repeat=i
        )
        for i, seed in enumerate(config["training"]["branch_schedule_seeds"], 1)
    }
    result = {
        "schema": "prospective-data-v1",
        "status": "DEVELOPMENT_READY" if development_only else "READY",
        "test_panel_status": "BLOCKED_EXPOSURE_AUDIT_NOT_APPLIED" if development_only else "READY",
        "selection_used_outcomes": False,
        "source_prompts": pools["source"],
        "continuation_prompts": pools["continuation"],
        "source_schedule": source_schedule,
        "branch_schedules": branch_schedules,
        "panels": {"P": p, "D": d, "T": t, "T_H8": t8},
        "confirm_exposure": copy.deepcopy(confirm_exposure),
        "registry": {
            "lineages": lineage_registry(config),
            "independent_unit": "source_lineage",
            "old_D2_role": "HISTORICAL_EXPLORATION_ONLY",
            "T_access": "metadata preparation; outcomes only after frozen decisions",
            "disjointness": "verified base_scene_id and original truth_world hash",
        },
    }
    result["data_id"] = canonical_hash(result)
    return result


def prepare_data(
    config,
    bindings,
    existing_panels,
    confirm_exposure=None,
    *,
    data_root=None,
    development_only=False,
):
    """Import exactly three explicitly bound files once; never infer exposure."""
    expected = {"train", "dev"} if development_only else {"train", "dev", "confirm"}
    if set(bindings) != expected:
        raise ValueError(f"Explicit bindings required for {sorted(expected)}")
    rows = {}
    for split, binding in bindings.items():
        path = Path(binding["path"])
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"Input bytes differ from frozen {split} binding")
        rows[split] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = build_dataset(
        rows["train"],
        rows["dev"],
        rows.get("confirm", []),
        existing_panels=existing_panels,
        confirm_exposure=confirm_exposure,
        config=config,
        data_root=data_root,
        development_only=development_only,
    )
    result["input_bindings"] = copy.deepcopy(bindings)
    return result
