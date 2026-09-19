"""Frozen metadata-only P/E panels; never read sealed confirmation data.

The original R3 selector is replayed on the complete control metadata before
exclusions. Selection uses only a seed, scene identity and metadata cell. Prompt
construction and its stored hash check happen after that selection. Discovery
completion streams on P are independent RNG roles, not an additional panel.
"""

from __future__ import annotations

import copy
import itertools
import json
from collections import defaultdict
from pathlib import Path

from ..core import canonical_hash
from ..modeling_v3.io import sha256_file
from ..prompts import OPERATIONS
from ..r3_inputs import CHARTS, FAMILIES, INTERFACES, select_control_panel
from ..r4_inputs import _n_record, _validate_n

SEED = 20260919
NAMESPACE = "decision-modeling-v1-P-E"
CELLS = tuple(itertools.product(FAMILIES, CHARTS, OPERATIONS))


def _ids(values):
    values = list(values)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("Exclusions require nonempty base_scene_id strings")
    return set(values)


def build_panels(
    control_scenes,
    *,
    config,
    train_base_scene_ids,
    debug_base_scene_ids=(),
    discovery_base_scene_ids=(),
    data_root=None,
):
    """Build two balanced 36-scene panels or return explicit per-cell shortages.

    ``discovery_base_scene_ids`` excludes any previously used scene-based
    development pool; it does not exclude independently sampled discovery
    completions on the newly frozen P prompts. Callers must supply the complete
    original control pool so the old R3 24-scene selection is reproducible.
    No fallback to dev, train, or sealed confirm is permitted.
    """
    if config.get("seed", SEED) != SEED:
        raise ValueError("Panel selection seed must remain 20260919")
    panel_config = config.get("panels", {})
    for key, expected in (
        ("scenes_per_panel", 36),
        ("prompts_per_panel", 72),
        ("scenes_per_cell", 2),
        ("exclude_old_r3", True),
    ):
        if panel_config.get(key, expected) != expected:
            raise ValueError(f"Frozen panel setting differs: {key}")
    if set(panel_config.get("interfaces", INTERFACES)) != set(INTERFACES):
        raise ValueError("Both original paired interfaces are required")
    debug_ids = _ids(debug_base_scene_ids) | _ids(panel_config.get("debug_base_scene_ids", ()))
    controls = _validate_n(control_scenes, "control")
    train = _ids(train_base_scene_ids)
    if train & {s["base_scene_id"] for s in controls}:
        raise ValueError("train/control base_scene_id overlap")
    old = {p["base_scene_id"] for p in select_control_panel(controls)}
    if len(old) != 24:
        raise ValueError("The original R3 exclusion must contain 24 base scenes")
    excluded = {
        "old_R3": sorted(old),
        "train": sorted(train),
        "debug": sorted(debug_ids),
        "prior_discovery": sorted(_ids(discovery_base_scene_ids)),
    }
    blocked = set().union(*(set(ids) for ids in excluded.values()))
    grouped = defaultdict(list)
    for scene in controls:
        if scene["base_scene_id"] not in blocked:
            grouped[scene["constraint_family"], scene["chart_type"], scene["operation"]].append(
                scene
            )
    for cell in CELLS:
        grouped[cell].sort(
            key=lambda s: (
                canonical_hash([NAMESPACE, SEED, list(cell), s["base_scene_id"]]),
                s["base_scene_id"],
            )
        )
    availability = [
        {
            "family": f,
            "chart_type": c,
            "operation": o,
            "available_scenes": len(grouped[f, c, o]),
            "required_scenes": 4,
            "deficit_scenes": max(0, 4 - len(grouped[f, c, o])),
        }
        for f, c, o in CELLS
    ]
    deficit = sum(row["deficit_scenes"] for row in availability)
    result = {
        "schema": "decision-modeling-panels-v1",
        "selection_seed": SEED,
        "selection_namespace": NAMESPACE,
        "selection_used_outcomes": False,
        "sealed_confirm_read": False,
        "status": "INSUFFICIENT_CONTROL" if deficit else "READY",
        "config_hash": canonical_hash(config),
        "excluded": excluded,
        "control_scenes": len(controls),
        "eligible_scenes": sum(len(v) for v in grouped.values()),
        "availability": availability,
        "total_deficit_scenes": deficit,
        "control_metadata_hash": canonical_hash(sorted(controls, key=lambda s: s["base_scene_id"])),
        "panels": {},
        "generated_additions": [],
        "discovery_policy": "Independent completion RNG on P; never pooled into endpoint MC",
        "E_policy": "Evaluate only after development recipe and readout rules are frozen",
    }
    if not deficit:
        for name, offset in (("P", 0), ("E", 2)):
            result["panels"][name] = [
                {**_n_record(grouped[cell][offset + i], interface, data_root), "panel": name}
                for i in range(2)
                for cell in CELLS
                for interface in INTERFACES
            ]
    else:
        result["required_action"] = (
            "Append independent scenes with the existing generate_worlds generator, "
            "binding generator version/seed and an exclusion catalogue; then refreeze. "
            "Do not read sealed confirm or substitute another split."
        )
    result["inputs_hash"] = canonical_hash(result)
    return result


def prepare_panels(
    config, bindings, *, data_root=None, debug_base_scene_ids=(), discovery_base_scene_ids=()
):
    """Read explicitly hash-bound train/control JSONL files and no other splits.

    Bindings have ``control`` and ``train`` entries, each with ``path`` and
    ``sha256``. Extra split bindings are rejected instead of opened. Keep actual
    private paths in runtime configuration; portable frozen manifests can omit
    those paths and retain their digests.
    """
    if set(bindings) != {"control", "train"}:
        raise ValueError("Only explicit train/control bindings are accepted")
    rows = {}
    for split in ("control", "train"):
        binding = bindings[split]
        path = Path(binding["path"])
        if path.name != f"{split}.jsonl":
            raise ValueError("Binding filename must match its train/control role")
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"Input bytes differ from frozen {split} identity")
        rows[split] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if any(row.get("split") != split for row in rows[split]):
            raise ValueError(f"Input binding contains a different split from {split}")
    train_ids = [s["base_scene_id"] for s in rows["train"]]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("duplicate train base_scene_id")
    result = build_panels(
        rows["control"],
        config=config,
        train_base_scene_ids=train_ids,
        debug_base_scene_ids=debug_base_scene_ids,
        discovery_base_scene_ids=discovery_base_scene_ids,
        data_root=data_root,
    )
    result["input_bindings"] = copy.deepcopy(bindings)
    result["inputs_hash"] = canonical_hash({k: v for k, v in result.items() if k != "inputs_hash"})
    return result
