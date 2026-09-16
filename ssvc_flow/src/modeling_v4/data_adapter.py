"""Metadata-only V4 input and nested panel planning; no model imports."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

from ..modeling_v3.io import canonical_hash, sha256_file

NAMESPACE = "SSVC_MODELING_V4_20260916"
CANDIDATES = (
    {"id": "joint_0", "policy": "joint", "auxiliary_weight": 0.0},
    {"id": "joint_1", "policy": "joint", "auxiliary_weight": 1.0},
    {"id": "no_x_off_1", "policy": "no_x_off", "auxiliary_weight": 1.0},
)
CONTRASTS = (
    ("joint_1_minus_joint_0", "joint_1", "joint_0"),
    ("no_x_off_1_minus_joint_0", "no_x_off_1", "joint_0"),
    ("joint_1_minus_no_x_off_1", "joint_1", "no_x_off_1"),
)


def bound_json(binding):
    path = Path(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("Input bytes differ from the registered identity: " + str(path))
    return json.loads(path.read_text())


def checkpoint_binding(value):
    result = copy.deepcopy(value)
    if "file_sha256" in result:
        result["sha256"] = result.pop("file_sha256")
    if not {"path", "sha256", "identity"} <= result.keys():
        raise ValueError("Historical checkpoint needs path, sha256 and original identity")
    return result


def build_panels(control_scenes, *, data_root=None):
    """Observation, signature and held-out prompt scenes in all eighteen cells.

    Round-robin cell ordering makes the first 24 observation prompts a registered
    prefix, without selecting on responses or rereading a confirmation dataset.
    """
    from ..r4_inputs import _n_record

    grouped = defaultdict(dict)
    for scene in control_scenes:
        if scene.get("split") != "control":
            raise ValueError("V4 panels may only read original control scenes")
        cell = (scene["constraint_family"], scene["chart_type"], scene["operation"])
        sid = scene["base_scene_id"]
        if sid in grouped[cell] and grouped[cell][sid] != scene:
            # Original manifests can carry paired interface records; the scene
            # and prompt fields differ only by that interface bookkeeping.
            old = {k: v for k, v in grouped[cell][sid].items() if k != "interface"}
            new = {k: v for k, v in scene.items() if k != "interface"}
            if old != new:
                raise ValueError("Conflicting control scene identity")
        grouped[cell][sid] = scene
    if len(grouped) != 18 or any(len(v) < 4 for v in grouped.values()):
        raise ValueError("Need four disjoint scenes in all 18 panel cells")
    selected = {
        cell: sorted(
            values.values(), key=lambda s: canonical_hash([NAMESPACE, "panel", s["base_scene_id"]])
        )
        for cell, values in grouped.items()
    }
    interfaces = ("SYMBOLIC_FRESH", "IMAGE_CUE_FRESH")
    # Interleave families within each chart/operation cell. The first four
    # chart/operation cells give four prompts in each family/interface stratum.
    cells = sorted(selected, key=lambda c: (c[1], c[2], c[0]))
    observation = [
        _n_record(selected[c][i], interface, data_root)
        for i in range(2)
        for c in cells
        for interface in interfaces
    ]
    signature = [
        _n_record(selected[c][2], interface, data_root) for c in cells for interface in interfaces
    ]
    cross_prompt = [
        _n_record(selected[c][3], interface, data_root) for c in cells for interface in interfaces
    ]
    assert not {p["base_scene_id"] for p in observation} & {p["base_scene_id"] for p in signature}
    return {
        "observation": observation,
        "signature": signature,
        "cross_prompt": cross_prompt,
        "first_map": observation[:24],
        "selection_used_outcomes": False,
    }


def build_bank_plan(train_prompts, *, origin_id, calibration_banks, query_banks):
    """Nested banks from fixed, disjoint 384/192 training pools."""
    if not 0 <= calibration_banks <= 96 or not 0 <= query_banks <= 48:
        raise ValueError("Bank counts exceed the disjoint training pools")
    grouped = defaultdict(list)
    for prompt in train_prompts:
        if prompt.get("split") != "train":
            raise ValueError("A training bank cannot contain control/test prompts")
        grouped[prompt["family"], prompt["interface"]].append(prompt)
    if len(train_prompts) != 576 or len({p["prompt_id"] for p in train_prompts}) != 576:
        raise ValueError("Exactly 576 unique original training prompts required")
    if len(grouped) != 6 or set(map(len, grouped.values())) != {96}:
        raise ValueError("Training prompts must form six balanced strata")
    pools = {"calibration": [], "query": []}
    per_group = {}
    for group, prompts in grouped.items():
        ordered = sorted(
            prompts, key=lambda p: canonical_hash([NAMESPACE, "partition", p["prompt_id"]])
        )
        per_group[group] = {"calibration": ordered[:64], "query": ordered[64:]}
    banks = []
    for role, count, size in (("calibration", calibration_banks, 64), ("query", query_banks, 32)):
        groups = {
            g: sorted(
                v[role],
                key=lambda p: canonical_hash([NAMESPACE, origin_id, "bank_order", p["prompt_id"]]),
            )
            for g, v in per_group.items()
        }
        pools[role] = [groups[g][i]["prompt_id"] for i in range(size) for g in sorted(groups)]
        for i in range(count):
            banks.append(
                {
                    "bank_id": f"{role}_{i:03d}",
                    "role": role,
                    "prompt_ids": pools[role][4 * i : 4 * i + 4],
                    "candidates": list(CANDIDATES),
                }
            )
    result = {
        "origin_id": origin_id,
        "pools": pools,
        "banks": banks,
        "training_outcomes_used_for_order": False,
    }
    result["plan_hash"] = canonical_hash(result)
    return result


def build_training_schedule(train_prompts, seed):
    grouped = defaultdict(list)
    for p in train_prompts:
        grouped[p["family"], p["interface"]].append(p)
    # Validate the original full panel through the same metadata-only contract.
    build_bank_plan(train_prompts, origin_id="schedule", calibration_banks=0, query_banks=0)
    for rows in grouped.values():
        rows.sort(key=lambda p: canonical_hash([NAMESPACE, "source", seed, p["prompt_id"]]))
    order = [grouped[g][i]["prompt_id"] for i in range(96) for g in sorted(grouped)][:512]
    return {
        "seed": seed,
        "train_steps": [order[i : i + 4] for i in range(0, 512, 4)],
        "schedule_hash": canonical_hash(order),
        "positions": 512,
    }


def prepare_inputs(config, bindings):
    """Read only training/control data and explicitly bound historical checkpoints.

    A missing X_VALID checkpoint must be represented by an explicit
    ``fallback_source: {seed: 41001, arm: X_BASE, step: 32, reason: ...}``.
    It is never silently replaced by the same X_BASE origin.
    """
    parent = bound_json(bindings["parent_validated_plan"])
    from ..modeling_v3.vlm_campaign import build_q4_bridge_plan

    v3config = json.loads(
        (Path(__file__).parents[2] / "configs/modeling_v3/protocol.json").read_text()
    )
    legacy = build_q4_bridge_plan(v3config, bindings)
    origins = bindings.get("bridge_origins")
    if origins is None:
        raise ValueError("Bind historical X_BASE64 and X_VALID64 (or explicit step32 fallback)")
    if len(origins) != 2:
        raise ValueError("Bridge requires two distinct historical/fallback origins")
    checked = []
    for i, value in enumerate(origins):
        if "fallback_source" in value:
            fallback = value["fallback_source"]
            if (
                i != 1
                or fallback.get("seed") not in (41001, 41002, 41003)
                or fallback.get("arm") != "X_BASE"
                or fallback.get("step") != 32
                or not fallback.get("reason")
            ):
                raise ValueError(
                    "Only documented second-origin development step32 fallback is registered"
                )
            checked.append(copy.deepcopy(value))
        else:
            spec = checkpoint_binding(value)
            if (
                spec["identity"].get("arm") != ("X_BASE" if i == 0 else "X_VALID")
                or spec["identity"].get("step") != 64
            ):
                raise ValueError("Bridge historical origins must be X_BASE64/X_VALID64")
            if sha256_file(spec["path"]) != spec["sha256"]:
                raise ValueError("Historical origin bytes changed")
            checked.append(spec)
    controls = [
        p for p in parent["training_data_binding"]["files"] if Path(p).name == "control.jsonl"
    ]
    if len(controls) != 1:
        raise ValueError("One original control.jsonl binding required")
    control_key = controls[0]
    data_root = Path(parent["paths"]["raw_dataset_root"]).resolve()
    control_path = (data_root / control_key).resolve()
    if not control_path.is_relative_to(data_root):
        raise ValueError("Control binding escapes the registered dataset root")
    if sha256_file(control_path) != parent["training_data_binding"]["files"][control_key]:
        raise ValueError("Control input changed")
    control_scenes = [
        json.loads(line) for line in Path(control_path).read_text().splitlines() if line
    ]
    panels = build_panels(control_scenes, data_root=parent["paths"]["raw_dataset_root"])
    result = {
        "schema": "ssvc-v4-inputs-1",
        "parent_binding": bindings["parent_validated_plan"],
        "bridge_origins": checked,
        "bridge_probes": legacy["probes"],
        "null_origin": legacy["null_origin"],
        "null_actions": legacy["null_actions"],
        "train_prompts": parent["training_plan"]["train_prompts"],
        "panels": panels,
        "data_root": parent["paths"]["raw_dataset_root"],
        "probability_tolerances": bindings["v3_probability_tolerances"],
        "config_hash": canonical_hash(config),
        "historical_data_role": "development_only",
    }
    build_training_schedule(result["train_prompts"], 41001)
    result["inputs_hash"] = canonical_hash(result)
    return result


def query_record(
    *,
    run_id,
    origin_id,
    seed,
    arm,
    step,
    bank_id,
    candidate_id,
    baseline_id,
    role,
    prompt,
    delta_reference,
    inference_fingerprint,
):
    if role not in {"calibration", "query", "bridge"}:
        raise ValueError("Unknown candidate role")
    return {
        "run_id": run_id,
        "origin_id": origin_id,
        "seed": seed,
        "arm": arm,
        "step": step,
        "bank_id": bank_id,
        "candidate_id": candidate_id,
        "baseline_id": baseline_id,
        "role": role,
        "probe_id": prompt["base_scene_id"],
        "prompt_id": prompt["prompt_id"],
        "actual_parameter_delta_reference": delta_reference,
        "inference_fingerprint": inference_fingerprint,
    }
