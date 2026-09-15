"""Prepare Q1 fixed historical policies without collecting a new trajectory.

Only identities choose origins/banks. The old finite evaluator is run anew at
retained parameter vectors; this is metered new scoring, never packet replay.
Missing originals are not reconstructed from summaries or regenerated seeds.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from .cpu_campaign import _binding, _digest, _finish, _hash, _json, _npz, _verify_complete


def _member(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    result = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not result.is_relative_to(root):
        raise ValueError(f"historical member escapes parent: {relative}")
    return result


def _array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()
    ).hexdigest()


def select_historical_origins(manifest, max_origins=12):
    """Identity-only round-robin arm/anchor strata; one hash-selected bank each."""
    if type(max_origins) is not int or max_origins < 1:
        raise ValueError("max_origins must be a positive integer")
    entries = manifest["trajectories"]
    if len({row["id"] for row in entries}) != len(entries):
        raise ValueError("duplicate historical trajectory identity")
    banks = sorted({bank for indices in manifest["bank_roles"].values() for bank in indices})
    if not banks or any(type(bank) is not int or bank < 0 for bank in banks):
        raise ValueError("original bank identities required")
    strata = {}
    for entry in entries:
        for ai, step in enumerate(manifest["anchors"]):
            ident = [entry["id"], entry["seed"], entry["arm"], int(step)]
            bank = min(banks, key=lambda b, i=ident: _digest(["Q1-historical-bank-v1", *i, b]))
            record = {
                "trajectory_id": entry["id"],
                "seed": entry["seed"],
                "arm": entry["arm"],
                "original_role": entry["split"],
                "anchor_index": ai,
                "step": int(step),
                "bank": bank,
                "selection_hash": _digest(["Q1-historical-origin-v1", *ident]),
            }
            strata.setdefault((entry["arm"], int(step)), []).append(record)
    for rows in strata.values():
        rows.sort(key=lambda row: row["selection_hash"])
    selected = []
    depth = 0
    while len(selected) < max_origins:
        before = len(selected)
        for key in sorted(strata):
            if depth < len(strata[key]) and len(selected) < max_origins:
                selected.append(strata[key][depth])
        if len(selected) == before:
            break
        depth += 1
    if not selected:
        raise ValueError("historical manifest contains no origins")
    return selected


def prepare_historical_observation_collection(parent_raw_root, out, config):
    """Produce an immutable ``observe_toy`` collection from original N4/M2 states.

    Optional ``config['historical_observation']['max_origins']`` changes the
    prespecified identity subset (default 12, balanced by arm/anchor). This does
    not depend on scores, errors, rewards, or response magnitudes. The original
    full NPZ/config/RNG files remain in place and are bound by path and hash.
    """
    started = time.perf_counter()
    parent, out = Path(parent_raw_root).resolve(), Path(out).resolve()
    if out.is_relative_to(parent):
        raise ValueError("historical preparation output must be outside the original parent")
    if out.exists() and not (out / "COMPLETE.json").is_file():
        raise FileExistsError("incomplete historical preparation retained; use a new output path")
    settings = config.get("historical_observation", {})
    missing, bindings = [], {}

    def cover(path, expected=None, role="original"):
        path = Path(path).resolve()
        if not path.is_file():
            missing.append({"path": str(path), "role": role, "status": "MISSING_INPUTS"})
            return None
        actual = _hash(path)
        if expected is not None and actual != expected:
            raise ValueError(f"historical original hash mismatch: {path}")
        bindings[str(path)] = {"sha256": actual, "bytes": path.stat().st_size, "role": role}
        return actual

    manifest_path = parent / "manifest.json"
    manifest_digest = cover(manifest_path, role="original_collection_manifest")
    selected, entries, manifest = [], {}, {}
    if manifest_digest is not None:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("events") != ["X", "S", "W", "I"]
            or manifest.get("operation_names") != ["joint_0", "joint_1", "no_x_off_1"]
            or manifest.get("parameter_count") != 737
            or manifest.get("probe_count") != 72
        ):
            raise ValueError("original 737-parameter, 72-probe, X/S/W/I policy contract differs")
        selected = select_historical_origins(manifest, settings.get("max_origins", 12))
        entries = {entry["id"]: entry for entry in manifest["trajectories"]}
        for filename, field in (
            ("dataset.npz", "dataset_sha256"),
            ("probe_metadata.json", "probe_identity_sha256"),
            ("train_metadata.json", "train_identity_sha256"),
            ("parameter_layout.json", "parameter_layout_sha256"),
        ):
            cover(parent / filename, manifest[field], field)
        if cover(parent / "resolved_config.json", role="complete_original_source_config"):
            original_config = json.loads((parent / "resolved_config.json").read_text())
            if _digest(original_config) != manifest["config_sha256"]:
                raise ValueError("original complete source config hash mismatch")
            if (
                original_config["toy_model"]["parameter_count"] != 737
                or original_config["toy_model"]["dtype"] != "float64"
                or original_config["dataset"]["candidate_actions"] != 16
            ):
                raise ValueError("original finite model configuration differs")
        toy_path = Path(__file__).resolve().parents[1] / "modeling_qualification/toy.py"
        if "toy.py" not in manifest.get("source_hashes", {}):
            missing.append(
                {
                    "path": str(manifest_path),
                    "role": "original_toy_source_hash",
                    "status": "MISSING_INPUTS",
                }
            )
        else:
            cover(toy_path, manifest["source_hashes"]["toy.py"], "unchanged_frozen_toy_evaluator")
        for trajectory in dict.fromkeys(row["trajectory_id"] for row in selected):
            entry = entries[trajectory]
            for field in ("observations_file", "rng_file"):
                cover(_member(parent, entry[field]), entry["sha256"][field], field)
    binding = _binding(
        config,
        stage="Q1_HISTORICAL_POLICY_PREPARATION",
        parent_root=str(parent),
        original_files=bindings,
        selection=selected,
    )
    if missing:
        if out.exists():
            raise ValueError("previously bound historical originals are now missing")
        out.mkdir(parents=True)
        result = {
            "status": "MISSING_INPUTS",
            "missing_inputs": missing,
            "parent_root": str(parent),
            "new_optimizer_updates": 0,
            "new_forward_calls": 0,
            "consumable_by_observe_toy": False,
        }
        _json(out / "MISSING_INPUTS_V3.json", result)
        return result
    if out.exists():
        return _verify_complete(out, binding)["summary"]
    # All existence/hash checks finish before a new forward is performed.
    from src.modeling_qualification.toy import _numpy_forward

    with np.load(parent / "dataset.npz", allow_pickle=False) as data:
        features = data["probe_features"].copy()
        categories = data["probe_categories"].copy()
    if features.shape != (72, 16, 44) or categories.shape != (72, 16):
        raise ValueError("original probe feature/label dimensions differ")
    if not np.isfinite(features).all() or categories.dtype.kind not in "iu":
        raise ValueError("invalid original features/categories")
    if np.any((categories < 0) | (categories > 3)):
        raise ValueError("original category outside X/S/W/I")
    metadata = json.loads((parent / "probe_metadata.json").read_text())
    if len(metadata) != 72 or len({row["prompt_id"] for row in metadata}) != 72:
        raise ValueError("72 distinct original probe identities required")
    originals = {}
    adam_keys = (
        "adam_m",
        "adam_v",
        "adam_step",
        "branch_adam_m",
        "branch_adam_v",
        "branch_adam_step",
        "checkpoint_grad",
        "checkpoint_grad_is_none",
    )
    for trajectory in dict.fromkeys(row["trajectory_id"] for row in selected):
        path = _member(parent, entries[trajectory]["observations_file"])
        with np.load(path, allow_pickle=False) as data:
            absent = [
                name
                for name in ("theta", "branch_theta", "anchors", *adam_keys)
                if name not in data
            ]
            if absent:
                missing.append(
                    {"path": str(path), "missing_arrays": absent, "status": "MISSING_INPUTS"}
                )
                continue
            # Only selected vectors are materialized; Adam stays in the bound full original.
            for record in (row for row in selected if row["trajectory_id"] == trajectory):
                ai, step, bank = record["anchor_index"], record["step"], record["bank"]
                if int(data["anchors"][ai]) != step:
                    raise ValueError("original observation anchor index differs from manifest")
                original_theta = data["theta"][step].copy()
                branch_theta = data["branch_theta"][ai, bank].copy()
                if original_theta.shape != (737,) or branch_theta.shape != (3, 737):
                    raise ValueError("original selected policy parameter dimensions differ")
                if not np.isfinite(original_theta).all() or not np.isfinite(branch_theta).all():
                    raise ValueError("nonfinite original selected policy parameters")
                originals[record["selection_hash"]] = original_theta, branch_theta
    if missing:
        out.mkdir(parents=True)
        result = {
            "status": "MISSING_INPUTS",
            "missing_inputs": missing,
            "parent_root": str(parent),
            "new_optimizer_updates": 0,
            "new_forward_calls": 0,
            "consumable_by_observe_toy": False,
        }
        _json(out / "MISSING_INPUTS_V3.json", result)
        return result
    out.mkdir(parents=True)
    _json(out / "BINDING.json", binding)
    _json(
        out / "HISTORICAL_SELECTION.json",
        {
            "rule": "identity_hash_round_robin_arm_anchor_then_identity_hash_bank",
            "namespace": "Q1-historical-v1",
            "max_origins": settings.get("max_origins", 12),
            "selected": selected,
            "labels_errors_rewards_read_for_selection": False,
            "historical_data_role": "PUBLIC_DEVELOPMENT_ONLY",
        },
    )
    _json(out / "ORIGINAL_SOURCE_BINDINGS.json", bindings)
    data_root = out / "dataset"
    data_root.mkdir()
    _npz(data_root / "dataset.npz", probe_features=features, probe_categories=categories)
    (data_root / "probe_metadata.json").write_bytes((parent / "probe_metadata.json").read_bytes())
    _finish(data_root, binding, {"status": "REUSED_ORIGINAL_PROBE_PANEL"})
    cache, forward_rows, trajectories = {}, [], []
    io_seconds = 0.0

    def score(theta, identity):
        fingerprint = _array_hash(theta)
        if fingerprint not in cache:
            call_started = time.perf_counter()
            probabilities = _numpy_forward(theta, features, categories)[1]
            elapsed = time.perf_counter() - call_started
            if (
                not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(-1), 1, atol=1e-14, rtol=0)
            ):
                raise ValueError("rescored historical policy is not a normalized finite law")
            cache[fingerprint] = probabilities
            forward_rows.append(
                {
                    "parameter_sha256": fingerprint,
                    "first_policy_identity": identity,
                    "new_batched_forward_calls": 1,
                    "prompt_forward_count": 72,
                    "scored_action_values": 72 * 16,
                    "forward_seconds": elapsed,
                    "action_probabilities_sha256": _array_hash(probabilities),
                }
            )
        return cache[fingerprint]

    for record in selected:
        tid, step, bank = record["trajectory_id"], record["step"], record["bank"]
        entry = entries[tid]
        theta, branches = originals[record["selection_hash"]]
        origin_p = score(theta, f"{tid}:step{step}:origin")
        branch_p = np.stack(
            [
                score(value, f"{tid}:step{step}:bank{bank}:op{op}")
                for op, value in enumerate(branches)
            ]
        )
        name = f"historical_{tid}_a{step}_b{bank}"
        relative = f"trajectories/{name}"
        unit = _member(out, relative)
        unit.mkdir(parents=True)
        io_started = time.perf_counter()
        identity = {
            **record,
            "role": "development",
            "anchors": [step],
            "bank_counts": {"calibration": 1, "heldout": 0},
            "bank_original_ids": [[bank]],
            "original_trajectory_id": tid,
            "original_anchor_indices": [record["anchor_index"]],
            "initialization_seed": original_config["toy_model"]["model_initialization_seed"],
            "historical_reused_fixed_policies": True,
            "policy_probability_source": "NEW_EXACT_FINITE_SCORING_AT_STORED_PARAMETERS",
            "original_observations_file": str(_member(parent, entry["observations_file"])),
            "original_observations_sha256": entry["sha256"]["observations_file"],
            "original_rng_file": str(_member(parent, entry["rng_file"])),
            "original_rng_sha256": entry["sha256"]["rng_file"],
            "original_adam_references": {
                key: {
                    "file": str(_member(parent, entry["observations_file"])),
                    "array": key,
                    "indices": [record["anchor_index"], bank]
                    if key.startswith("branch_")
                    else [step],
                }
                for key in adam_keys
            },
            "original_config_sha256": manifest["config_sha256"],
            "optimizer_updates": 0,
            "original_packet_replay": False,
        }
        _json(unit / "identity.json", identity)
        _npz(unit / "historical_parameters.npz", origin_theta=theta, branch_theta=branches)
        _npz(unit / "source_probabilities.npz", origin_action_p=origin_p[None])
        _npz(unit / "calibration_action_p.npz", action_p=branch_p[None, None])
        row = {
            "id": name,
            "path": relative,
            "seed": record["seed"],
            "arm": record["arm"],
            "role": "development",
            "original_trajectory_id": tid,
            "original_anchor": step,
            "original_bank": bank,
            "source_updates": 0,
            "fork_updates": 0,
            "historical_reused_fixed_policies": True,
        }
        _finish(unit, binding, row)
        trajectories.append(row)
        io_seconds += time.perf_counter() - io_started
    # A concurrent modification to any original invalidates the collection.
    for path, item in bindings.items():
        if _hash(path) != item["sha256"]:
            raise ValueError(f"historical original changed during preparation: {path}")
    costs = {
        "scope": "NEW_SCORING_OF_REUSED_HISTORICAL_FIXED_POLICIES",
        "new_batched_forward_calls": len(forward_rows),
        "new_prompt_forward_count": sum(row["prompt_forward_count"] for row in forward_rows),
        "new_scored_action_values": sum(row["scored_action_values"] for row in forward_rows),
        "forward_seconds": sum(row["forward_seconds"] for row in forward_rows),
        "selected_policy_requests": 4 * len(selected),
        "exact_parameter_cache_hits": 4 * len(selected) - len(forward_rows),
        "optimizer_updates": 0,
        "new_sampled_actions": 0,
        "output_io_seconds": io_seconds,
        "forwards": forward_rows,
    }
    _json(out / "SCORING_COST.json", costs)
    summary = {
        "stage": "Q1_HISTORICAL_POLICY_PREPARATION",
        "status": "PREPARED",
        "role": "development",
        "pilot": False,
        "scientific_status": "DEVELOPMENT_ONLY",
        "provenance": "REUSED_HISTORICAL_FIXED_POLICIES_NEW_SCORING",
        "parent_root": str(parent),
        "original_manifest_sha256": manifest_digest,
        "trajectory_count": len(trajectories),
        "trajectory_count_scope": "Q1_ORIGIN_VIEW_UNITS_NOT_NEW_TRAJECTORIES",
        "selected_origin_count": len(selected),
        "selected_bank_count": len(selected),
        "unique_original_trajectory_count": len({row["trajectory_id"] for row in selected}),
        "trajectories": trajectories,
        "source_updates": 0,
        "fork_updates": 0,
        "new_gpu_calls": 0,
        "online_ssvc": False,
        "original_packet_replay": False,
        "host": platform.node(),
        "wall_seconds": time.perf_counter() - started,
        "scoring_cost": costs,
        "consumable_by_observe_toy": True,
    }
    _json(out / "HISTORICAL_PREPARATION_SUMMARY.json", summary)
    return _finish(out, binding, summary)
